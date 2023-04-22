function [gainOverNoisedB,R,pilotIndex,D,D_small,APpositions,UEpositions,distances] = generateSetup_correlatedFading(L,K,N,tau_p,nbrOfSetups,seed,ASD_varphi,ASD_theta)
%This function generates realizations of the simulation setup described in
%Section V.
%
%INPUT:
%L               = Number of APs per setup
%K               = Number of UEs in the network
%N               = Number of antennas per AP
%tau_p           = Number of orthogonal pilots
%nbrOfSetups     = Number of setups with random UE and AP locations
%
%OUTPUT:
%gainOverNoisedB = Matrix with dimension L x K x nbrOfSetups where
%                  element (l,k,n) is the channel gain (normalized by the
%                  noise variance) between AP l and UE k in setup n
%R               = Matrix with dimension N x N x L x K x nbrOfSetups
%                  where (:,:,l,k,n) is the spatial correlation matrix
%                  between AP l and UE k in setup n, normalized by noise
%pilotIndex      = Matrix with dimension K x nbrOfSetups containing the
%                  pilot assigned to the users
%D               = Matrix with dimension L x K x nbrOfSetups where (l,k,n)
%                  is zero AP l serves UE k in setup n and zero otherwise
%APpositions     = Vector of length L with the AP locations, where the real
%                  part is the horizontal position and the imaginary part
%                  is the vertical position
%UEpositions     = Vector of length K with UE positions, measured in the
%                  same way as APpositions
%distances       = Matrix with same dimension as gainOverNoisedB containing
%                  the distances in meter between APs and UEs

%% Define simulation setup

if (nargin>5)&&(seed>0)
    rng(seed)
end

%Size of the coverage area (as a square with wrap-around)
squareLength = 500; %meter

%Communication bandwidth
B = 20e6;

%Noise figure (in dB)
noiseFigure = 7;

%Compute noise power
noiseVariancedBm = -174 + 10*log10(B) + noiseFigure;

%Pathloss parameters
alpha = 36.7;
constantTerm = -30.5;

%Standard deviation of the shadow fading
sigma_sf = 4;

%Decorrelation distance of the shadow fading
decorr = 9;

%Height difference between an AP and a UE
distanceVertical = 10;

%Define the antenna spacing (in number of wavelengths)
antennaSpacing = 1/2; %Half wavelength distance

%Set threshold for when a non-master AP decides to serve a UE
threshold = -200; %dB

%Prepare to save results
gainOverNoisedB = zeros(L,K,nbrOfSetups);
R = zeros(N,N,L,K,nbrOfSetups);
distances = zeros(L,K,nbrOfSetups);
pilotIndex = zeros(K,nbrOfSetups);
D = zeros(L,K,nbrOfSetups);
D_small = zeros(L,K,nbrOfSetups);

masterAPs = zeros(K,1);


%% Go through all setups
for n = 1:nbrOfSetups
    
    %Random AP locations with uniform distribution
    APpositions = (rand(L,1) + 1i*rand(L,1)) * squareLength;
    
    %Prepare to compute UE locations
    UEpositions = zeros(K,1);
    
    
    %Compute alternative AP locations by using wrap around
    wrapHorizontal = repmat([-squareLength 0 squareLength],[3 1]);
    wrapVertical = wrapHorizontal';
    wrapLocations = wrapHorizontal(:)' + 1i*wrapVertical(:)';
    APpositionsWrapped = repmat(APpositions,[1 length(wrapLocations)]) + repmat(wrapLocations,[L 1]);
    
    %Prepare to store shadowing correlation matrix
    shadowCorrMatrix = sigma_sf^2*ones(K,K);
    shadowAPrealizations = zeros(K,L);

    
    
    %Add UEs 
    for k = 1:K
        
        %Generate a random UE location in the area
        UEposition = (rand(1,1) + 1i*rand(1,1)) * squareLength;
        
        %Compute distances assuming that the APs are 10 m above the UEs
        [distanceAPstoUE,whichpos] = min(abs(APpositionsWrapped - repmat(UEposition,size(APpositionsWrapped))),[],2);
        distances(:,k,n) = sqrt(distanceVertical^2+distanceAPstoUE.^2);
        
        %If this is not the first UE
        if k-1>0
            
            %Compute distances from the new prospective UE to all other UEs
            shortestDistances = zeros(k-1,1);
            
            for i = 1:k-1
                shortestDistances(i) = min(abs(UEposition - UEpositions(i) + wrapLocations));
            end
            
            %Compute conditional mean and standard deviation necessary to
            %obtain the new shadow fading realizations, when the previous
            %UEs' shadow fading realization have already been generated.
            %This computation is based on Theorem 10.2 in "Fundamentals of
            %Statistical Signal Processing: Estimation Theory" by S. Kay
            newcolumn = sigma_sf^2*2.^(-shortestDistances/decorr);
            term1 = newcolumn'/shadowCorrMatrix(1:k-1,1:k-1);
            meanvalues = term1*shadowAPrealizations(1:k-1,:);
            stdvalue = sqrt(sigma_sf^2 - term1*newcolumn);
            
        else %If this is the first UE
            
            %Add the UE and begin to store shadow fading correlation values
            meanvalues = 0;
            stdvalue = sigma_sf;
            newcolumn = [];
            
        end
        
        %Generate the shadow fading realizations
        shadowing = meanvalues + stdvalue*randn(1,L);
        
        %Compute the channel gain divided by noise power
        gainOverNoisedB(:,k,n) = constantTerm - alpha*log10(distances(:,k,n)) + shadowing' - noiseVariancedBm;
        
        
            
        %Update shadowing correlation matrix and store realizations
        shadowCorrMatrix(1:k-1,k) = newcolumn;
        shadowCorrMatrix(k,1:k-1) = newcolumn';
        shadowAPrealizations(k,:) = shadowing;
        
        %Store the UE position
        UEpositions(k) = UEposition;
            
         
        %Determine the master AP for UE k by looking for AP with best
        %channel condition
        [~,master] = max(gainOverNoisedB(:,k,n));
        D(master,k,n) = 1;
        masterAPs(k) = master;
        
        %Assign orthogonal pilots to the first tau_p UEs
        if k <= tau_p
            
            pilotIndex(k,n) = k;
            
        else %Assign pilot for remaining UEs
            
            %Compute received power from to the master AP from each pilot
            pilotinterference = zeros(tau_p,1);
            
            for t = 1:tau_p
                
                pilotinterference(t) = sum(db2pow(gainOverNoisedB(master,pilotIndex(1:k-1,n)==t,n)));
                
            end
            
            %Find the pilot with the least receiver power
            [~,bestpilot] = min(pilotinterference);
            pilotIndex(k,n) = bestpilot;
            
        end
        
        

        %Go through all APs
        for l = 1:L
            
            %Compute nominal angle between UE k and AP l
            angletoUE_varphi = angle(UEpositions(k)-APpositionsWrapped(l,whichpos(l)));
            angletoUE_theta = acos(distanceVertical/distances(l,k,n));
            %Generate normalized spatial correlation matrix using the local
            %scattering model
            if nargin>6
                R(:,:,l,k,n) = db2pow(gainOverNoisedB(l,k,n))*functionRlocalscattering_theta(N,angletoUE_varphi,angletoUE_theta,ASD_varphi,ASD_theta,antennaSpacing);
            else
                R(:,:,l,k,n) = db2pow(gainOverNoisedB(l,k,n))*eye(N);
            end
        end
        
    end
    
    
    %Each AP serves the UE with the strongest channel condition on each of
    %the pilots where the AP isn't the master AP, but only if its channel
    %is not too weak compared to the master AP
    for l = 1:L
        
        for t = 1:tau_p
            
            pilotUEs = find(t==pilotIndex(:,n));
            [gainValue,UEindex] = max(gainOverNoisedB(l,pilotUEs,n));
            
            if gainValue - gainOverNoisedB(masterAPs(pilotUEs(UEindex)),pilotUEs(UEindex),n) >= threshold
                    
                    D(l,pilotUEs(UEindex),n) = 1;
                    
            end

        end
        
    end
    
    for k=1:K
        
        tempmat = -inf*ones(L,1);
        tempmat(D(:,k,n)==1,1) = gainOverNoisedB(D(:,k,n)==1,k,n);
        [~,servingAP] = max(tempmat);
        D_small(servingAP,k,n) = 1;
    
    end
    
    
end

