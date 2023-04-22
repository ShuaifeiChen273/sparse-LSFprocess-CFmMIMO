% ==================================================================
% This Matlab script generates Figure 5 in the paper:
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
%     Simulation setup terms:
%     nbrOfSetups         = Number of setups with random UE locations
%     nbrOfRealizations   = Number of realizations for small-scale fading
%     L                   = Number of APs per setup
%     N                   = Number of antennas per AP
%     K                   = Number of UEs in the network
%     tau_c               = Number of channel uses per coherence block
%     tau_p               = Number of pilots per coherence block
%     tau_u               = Number of channel uses for uplink per coherence block
%     tau_d               = Number of channel uses for uplink per coherence block
%     p                   = Total uplink transmit power per UE (mW)
%     rho_tot             = Total downlink transmit power per AP (mW)
%     ASD_varphi          = Azimuth angular standard deviation (in radians)
%     ASD_theta           = Elevation angular standard deviation (in radians)
%     schemes             = Set of schemes considered in simulations
%     nbrOfSchemes        = Number of considered schemes
%     schemeCategory      = Categories of schemes considered in simulations
%     displaySetup        = Text describing the setup terms
% ==================================================================

close all;
clear;

seed = 97006855;
ss = RandStream('mt19937ar','Seed',seed);
RandStream.setGlobalStream(ss);


%% Define simulation setup
setups = struct();
nbrOfSetups = 100;                  setups.nbrOfSetups = nbrOfSetups;
nbrOfRealizations = 1000;           setups.nbrOfRealizations = nbrOfRealizations;
L = 40;                             setups.L = L;
N = 4;                              setups.N = N;
K = 20;                             setups.K = K;
tau_p = 10;                         setups.tau_p = tau_p;
schemeCategory = {'All','DCC'};
schemes = {'V-LSFP','P-LSFP','HEUR','FPA'};

tau_c = 200;                    setups.tau_c = tau_c;
tau_u = 0;                      setups.tau_u = tau_u;
tau_d = tau_c - tau_p;          setups.tau_d = tau_d;
p = 100;                        setups.p = p;
rho_tot = 1000;                 setups.rho_tot = rho_tot;
ASD_varphi = deg2rad(10);       setups.ASD_varphi = ASD_varphi;
ASD_theta = deg2rad(10);        setups.ASD_theta = ASD_theta;
combining = {'MMSE','MR'};      setups.combining = combining;
nbrOfSchemes = length(schemes);
Rset = 7;

Seed = round(rand(nbrOfSetups,1)*200) + 1;

%Prepare to save simulation results
SE_MMSE_tot = zeros(K,nbrOfSchemes,nbrOfSetups);    %Spectral efficiency
SE_MR_tot = zeros(K,nbrOfSchemes,nbrOfSetups);    %Spectral efficiency

%
%% Go through all setups
for n = 1:nbrOfSetups
    
    %Display simulation progress
    tt = tic;
    fprintf('==================================================\n');
    disp(['   Setup ' num2str(n) ' out of ' num2str(nbrOfSetups)]);
    
    seedn = Seed(n);
    %Generate one setup with UEs and APs at random locations
    [gainOverNoisedB,R,pilotIndex,D] = generateSetup_correlatedFading(L,K,N,tau_p,1,seedn,ASD_varphi,ASD_theta);
    gainOverNoise = db2pow(gainOverNoisedB);
    
    %Generate channel realizations with estimates and estimation
    %error correlation matrices
    [Hhat,H,~,C] = functionChannelEstimates(R,nbrOfRealizations,L,K,N,tau_p,pilotIndex,p);
    
    pk = struct();
    rholk = struct();
    %Compute the power control coefficients in Eq. (7.35)
    if sum(strcmp(schemeCategory, 'All'))
        pk.all = distPowerControl(gainOverNoise,ones(L,K),p,0.5);
        rholk.all = distPowerAllocation(gainOverNoise,ones(L,K),rho_tot,0.5);
    end
    if sum(strcmp(schemeCategory, 'DCC'))
        pk.dcc = distPowerControl(gainOverNoise,D,p,0.5);
        rholk.dcc = distPowerAllocation(gainOverNoise,D,rho_tot,0.5);
    end
    
    setups.Hhat = Hhat;
    setups.H = H;
    setups.C = C;
    setups.D = D;
    setups.gainOverNoise = gainOverNoise;
    setups.pk = pk;
    setups.rholk = rholk;
    
    %Obtain the expectations for the computation of SE
    [signal,signal2,scaling] = functionComputeExpectations(Hhat,H,D,C,setups,schemeCategory,combining);
    
    SE = functionComputeSE(signal,signal2,scaling,setups,schemes);
    
    %Save the SE values
    if sum(strcmp(combining, 'MMSE'))
        SE_MMSE_tot(:,:,n) = SE.MMSE;
    end
    
    if sum(strcmp(combining, 'MR'))
        SE_MR_tot(:,:,n) = SE.MR;
    end
    
    
    %Remove large matrices at the end of analyzing this setup
    clear Hhat H B C R;
    
    %Display simulation progress
    timer = toc(tt);
    fprintf('-- Time cost: %.2f\t', timer);
    disp(datetime);
end
%}

%% Plot simulation results

figure;

subplot(2,1,1)
hold on; box on;

plot(sort(reshape(SE_MR_tot(:,1,:),[K*nbrOfSetups,1])),linspace(0,1,K*nbrOfSetups),'r-','LineWidth',4);
plot(sort(reshape(SE_MR_tot(:,2,:),[K*nbrOfSetups,1])),linspace(0,1,K*nbrOfSetups),'b--','LineWidth',2);
plot(sort(reshape(SE_MR_tot(:,3,:),[K*nbrOfSetups,1])),linspace(0,1,K*nbrOfSetups),'k-','LineWidth',1);
plot(sort(reshape(SE_MR_tot(:,4,:),[K*nbrOfSetups,1])),linspace(0,1,K*nbrOfSetups),'k--','LineWidth',1);

p1 = plot(sort(reshape(SE_MMSE_tot(:,1,:),[K*nbrOfSetups,1])),linspace(0,1,K*nbrOfSetups),'r-','LineWidth',4);
p2 = plot(sort(reshape(SE_MMSE_tot(:,2,:),[K*nbrOfSetups,1])),linspace(0,1,K*nbrOfSetups),'b--','LineWidth',2);
p3 = plot(sort(reshape(SE_MMSE_tot(:,3,:),[K*nbrOfSetups,1])),linspace(0,1,K*nbrOfSetups),'k-','LineWidth',1);
p4 = plot(sort(reshape(SE_MMSE_tot(:,4,:),[K*nbrOfSetups,1])),linspace(0,1,K*nbrOfSetups),'k--','LineWidth',1);

legend([p1 p2 p3 p4],{'V-LSFD','P-LSFD','HEUR','FPA'},'Interpreter','Latex','Location','SouthEast','NumColumns',1);
xlim([0 8]);
text(4,0.2,'L-MMSE','Interpreter','Latex');
text(0.5,0.8,'MR','Interpreter','Latex');
title({'(a) $L=40,\ N=4$'},'Interpreter','Latex');

subplot(2,1,2)
hold on; box on;

plot(sort(reshape(SE_MR_tot(:,1,:),[K*nbrOfSetups,1])),linspace(0,1,K*nbrOfSetups),'r-','LineWidth',4);
plot(sort(reshape(SE_MR_tot(:,2,:),[K*nbrOfSetups,1])),linspace(0,1,K*nbrOfSetups),'b--','LineWidth',2);
plot(sort(reshape(SE_MR_tot(:,3,:),[K*nbrOfSetups,1])),linspace(0,1,K*nbrOfSetups),'k-','LineWidth',1);
plot(sort(reshape(SE_MR_tot(:,4,:),[K*nbrOfSetups,1])),linspace(0,1,K*nbrOfSetups),'k--','LineWidth',1);

p1 = plot(sort(reshape(SE_MMSE_tot(:,1,:),[K*nbrOfSetups,1])),linspace(0,1,K*nbrOfSetups),'r-','LineWidth',4);
p2 = plot(sort(reshape(SE_MMSE_tot(:,2,:),[K*nbrOfSetups,1])),linspace(0,1,K*nbrOfSetups),'b--','LineWidth',2);
p3 = plot(sort(reshape(SE_MMSE_tot(:,3,:),[K*nbrOfSetups,1])),linspace(0,1,K*nbrOfSetups),'k-','LineWidth',1);
p4 = plot(sort(reshape(SE_MMSE_tot(:,4,:),[K*nbrOfSetups,1])),linspace(0,1,K*nbrOfSetups),'k--','LineWidth',1);

legend([p1 p2 p3 p4],{'V-LSFD','P-LSFD','HEUR','FPA'},'Interpreter','Latex','Location','SouthEast','NumColumns',1);
xlim([0 8]);
title({'(b) $L=160,\ N=1$'},'Interpreter','Latex');
text(4,0.2,'L-MMSE','Interpreter','Latex');
text(0.5,0.8,'MR','Interpreter','Latex');

set(gcf,'unit','centimeters','position',[0 0 15 16]);

%% Auxiliary Function

%Scalable fractional uplink power control
function p_dist = distPowerControl(gainOverNoise,D,p,nu)
%Number of APs and number of UEs
[L,K] = size(D);

%Prepare to save power allocation coefficients
p_dist = zeros(K,1);

%Go through all UEs
for k = 1:K
    
    %Extract which APs that serve UE k
    servingAPs = find(D(:,k)==1);
    
    %Determine which UEs that are served by partially the same set
    %of APs as UE k, i.e., the set in (5.15)
    servedUEs = find(sum(D(servingAPs,:),1)>=1);
    
    %Prepare to compute denominator in (7.35) for exponent nu
    normalization = 0;
    
    %Go through all UEs that are partially served by the same APs as UE k
    for i = servedUEs
        %Extract APs that serve UE i
        servingAPsi = find(D(:,i)==1);
        %Compute the denominator of (7.35) with exponents nu
        normalization = max(normalization,  (sum(gainOverNoise(servingAPsi,i))).^nu);
    end
    %Compute p_k according to (7.35) for exponent nu
    p_dist(k) = p*(sum(gainOverNoise(servingAPs,k)))^nu/normalization;
    
end
end

%Scalable fractional downlink power allocation
function rho_dist = distPowerAllocation(gainOverNoise,D,rho_tot,nu)
%Number of APs and number of UEs
[L,K] = size(D);

%Prepare to save power allocation coefficients
rho_dist = zeros(L,K);

%Go through all APs
for l = 1:L
    
    %Extract which UEs are served by AP l
    servedUEs = find(D(l,:)==1);
    
    %Compute denominator in Eq. (7.47)
    normalizationAPl = sum((gainOverNoise(l,servedUEs)).^nu);
    
    for ind = 1:length(servedUEs)
        
        rho_dist(l,servedUEs(ind)) = rho_tot*(gainOverNoise(l,servedUEs(ind)))^nu/normalizationAPl;
        
    end
    
end

end

