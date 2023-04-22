% ==================================================================
% This Matlab script computes the expectatations that appear in the 
% uplink and downlink SE expressions in the paper:
% 
% Shuaifei Chen, Jiayi Zhang, Emil Björnson, Özlem Tuğfe Demir, and Bo Ai, 
% ``Energy-efficient cell-free massive MIMO through sparse large-scale fading 
% processing", Transactions on Wireless Communications, to appear. 2023.

% Download article: https://arxiv.org/abs/2208.13552
% ==================================================================
% This is version 1.02 (Last edited: 2023-4-22)
%
% License: This code is licensed under the GPLv2 license. If you in any way
% use this code for research that results in publications, please cite our
% paper as described above.
% ==================================================================
%     Inputs:
%     Hhat              = Matrix with dimension L*N  x nbrOfRealizations x K
%                        where (:,n,k) is the estimated collective channel to
%                        UE k in channel realization n.
%     H                 = Matrix with dimension L*N  x nbrOfRealizations x K
%                        with the true channel realizations. The matrix is
%                        organized in the same way as Hhat.
%     D                 = DCC matrix with dimension L x K
%                        where (l,k) is one if AP l serves UE k and zero otherwise
%     C                 = Matrix with dimension N x N x L x K where (:,:,l,k) is
%                        the spatial correlation matrix of the channel
%                        estimation error between AP l and UE k,
%                        normalized by noise variance
%     nbrOfRealizations = Number of channel realizations
%     N                 = Number of antennas per AP
%     K                 = Number of UEs per cell
%     L                 = Number of APs
%     p                 = Vector of UE transmit powers
%    ---------------------------------------------------------------
%     Outputs:
%     signal_LP_MMSE    = Matrix with dimension K x K x L
%                        where (i,k,l) is the Monte-Carlo estimation of
%                        expected value of h_{il}^HD_{kl}w_{kl} where w_{kl} is
%                        LP-MMSE combiner/precoder
%     signal2_LP_MMSE   = Matrix with dimension K x K x L
%                        where (i,k,l) is the Monte-Carlo estimation of
%                        expected value of |h_{il}^HD_{kl}w_{kl}|^2 where w_{kl} is
%                        LP-MMSE combiner/precoder
%     scaling_LP_MMSE   = Matrix with dimension L x K
%                        where (l,k) is the Monte-Carlo estimation of
%                        expected value of the norm square of D_{kl}w_{kl}
%                        for LP-MMSE combiner/precoder
% ==================================================================
% This Matlab script is written with reference to:
% 
% Özlem Tuğfe Demir, Emil Björnson and Luca Sanguinetti (2021),
% ``Foundations of User-Centric Cell-Free Massive MIMO",
% Foundations and Trends in Signal Processing: Vol. 14: No. 3-4,
% pp 162-472. DOI: 10.1561/2000000109
% ==================================================================

function [Signal,Signal2,Scaling] = functionComputeExpectations(Hhat,H,D,C,opts,schemeCategory,combining)

%Simulation setup terms
nbrOfRealizations = opts.nbrOfRealizations;
N = opts.N;
K = opts.K;
L = opts.L;
Pk = opts.pk;

%Prepare to save simulation results
Signal = struct();
Signal2 = struct();
Scaling = struct();

%For the cases that each AP serves all UEs
if sum(strcmp(schemeCategory, 'All'))
    
    %Compute the power control coefficients in Eq. (7.35)
    pk = Pk.all;
    %Scale C by power coefficients
    Cp = zeros(size(C));
    for k=1:K
        Cp(:,:,:,k) = pk(k)*C(:,:,:,k);
    end
    if sum(strcmp(combining, 'MMSE'))
    [signal,signal2,scaling] = computeExpectations(Hhat,H,ones(L,K),Cp,K,L,N,nbrOfRealizations,pk,'MMSE');
    Signal.all.mmse = signal;
    Signal2.all.mmse = signal2;
    Scaling.all.mmse = scaling;
    end
    
    if sum(strcmp(combining, 'MR'))
    [signal,signal2,scaling] = computeExpectations(Hhat,H,ones(L,K),Cp,K,L,N,nbrOfRealizations,pk,'MR');
    Signal.all.mr = signal;
    Signal2.all.mr = signal2;
    Scaling.all.mr = scaling;
    end
end

%For the case that DCC has been performed before computing the expectations
if sum(strcmp(schemeCategory, 'DCC'))
    
    %Compute the power control coefficients in Eq. (7.35)
    pk = Pk.dcc;
    
    %Scale C by power coefficients
    Cp = zeros(size(C));
    for k=1:K
        Cp(:,:,:,k) = pk(k)*C(:,:,:,k);
    end
    
    if sum(strcmp(combining, 'MMSE'))
    [signal,signal2,scaling] = computeExpectations(Hhat,H,D,Cp,K,L,N,nbrOfRealizations,pk,'MMSE');
    Signal.dcc.mmse = signal;
    Signal2.dcc.mmse = signal2;
    Scaling.dcc.mmse = scaling;
    end
    
    if sum(strcmp(combining, 'MR'))
    [signal,signal2,scaling] = computeExpectations(Hhat,H,D,Cp,K,L,N,nbrOfRealizations,pk,'MR');
    Signal.dcc.mr = signal;
    Signal2.dcc.mr = signal2;
    Scaling.dcc.mr = scaling;
    end
end

end

%% Compute scaling factors for combining/precoding
% For uplink combining, gki = conj(vec(signal(i,k,:)))
% For downlink precoding, fki = diag(1./sqrt(scaling(:,i)))*vec(conj(signal(k,i,:)))

function [signal,signal2,scaling] = computeExpectations(Hhat,H,D,Cp,K,L,N,nbrOfRealizations,p,combining)

%Store the N x N identity matrix
eyeN = eye(N);

PowMat = diag(p);

%Prepare to store simulation results
signal = zeros(K,K,L);
signal2 = zeros(K,K,L);
scaling = zeros(L,K);

%Go through all channel realizations
for n=1:nbrOfRealizations
    
    %Go through all APs
    for l = 1:L
        %Extract channel realizations from all UEs to AP l
        Hallj = reshape(H(1+(l-1)*N:l*N,n,:),[N K]);
        
        %Extract channel estimate realizations from all UEs to AP l
        Hhatallj = reshape(Hhat(1+(l-1)*N:l*N,n,:),[N K]);
        
        %Extract which UEs are served by AP l
        servedUEs = find(D(l,:)==1);
        %Obtain the statistical matrices used for
        %computing partial combining/precoding schemes
        Cpserved = reshape(sum(Cp(:,:,l,servedUEs),4),[N N]);
        Pserved = PowMat(servedUEs,servedUEs);
        
        %Compute MR combining
        V_MR = Hhatallj(:,servedUEs);
        %MR combining scaled by square root of transmit powers
        Vp_MR = V_MR*sqrt(Pserved);
        %Compute LP-MMSE combining
        V_LP_MMSE = (((Vp_MR*Vp_MR')+Cpserved+eyeN)\Vp_MR)*sqrt(Pserved);
        
        %Go through all UEs served by the AP
        for ind = 1:length(servedUEs)
            
            %Extract UE index
            k = servedUEs(ind);
            
            if sum(strcmp(combining, 'MMSE'))
            %Compute unnormalized LP-MMSE precoding vector that AP l
            %selects for UE k using uplink-downlink duality
            v = V_LP_MMSE(:,ind);
            elseif sum(strcmp(combining, 'MR'))
            %Compute unnormalized MR precoding vector that AP l
            %selects for UE k using uplink-downlink duality
            v = V_MR(:,ind);
            end
            
            %Compute realizations of the terms inside the expectations
            %of the signal and interference terms in the SE expressions and
            %update Monte-Carlo estimates
            
            signal(:,k,l) = signal(:,k,l) + Hallj'*v/nbrOfRealizations;
            
            signal2(:,k,l) = signal2(:,k,l) + abs(Hallj'*v).^2/nbrOfRealizations;
            
            scaling(l,k) = scaling(l,k) + sum(abs(v).^2,1)/nbrOfRealizations;
            
        end
    end
    
end

end