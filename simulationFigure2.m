% ==================================================================
% This Matlab script generates Figure 2 in the paper:
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
%     schemesSparse       = Set of sparse schemes considered in simulations
%     schemesNonSparse    = Set of nonsparse schemes considered in simulations
%     lambda1             = Regularization parameters for element sparsity
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
lambda1 = [1e-4 1e-2];              setups.lambda1 = lambda1;
L = 40;                             setups.L = L;
N = 4;                              setups.N = N;
K = 20;                             setups.K = K;
tau_p = 10;                         setups.tau_p = tau_p;
schemeCategory = {'All','DCC'};
schemesSparse = {'S-LSFD'};
schemesNonSparse = {'P-LSFD'};
schemes = ['O-LSFD',repmat(schemesSparse,[1 length(lambda1)]),schemesNonSparse];

tau_c = 200;                    setups.tau_c = tau_c;
tau_u = tau_c - tau_p;          setups.tau_u = tau_u;
tau_d = 0;                      setups.tau_d = tau_d;
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
EE_MMSE_tot = zeros(nbrOfSchemes,nbrOfSetups);      %Energy matrices
D_MMSE_tot = zeros(L,K,nbrOfSchemes,nbrOfSetups);   %Association matrices
SE_MR_tot = zeros(K,nbrOfSchemes,nbrOfSetups);    %Spectral efficiency
EE_MR_tot = zeros(nbrOfSchemes,nbrOfSetups);      %Energy matrices
D_MR_tot = zeros(L,K,nbrOfSchemes,nbrOfSetups);   %Association matrices

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
    %Compute the power control coefficients in Eq. (7.35)
    if sum(strcmp(schemeCategory, 'All'))
        pk.all = distPowerControl(gainOverNoise,ones(L,K),p,0.5);
    end
    if sum(strcmp(schemeCategory, 'DCC'))
        pk.dcc = distPowerControl(gainOverNoise,D,p,0.5);
    end
    
    setups.Hhat = Hhat;
    setups.H = H;
    setups.C = C;
    setups.D = D;
    setups.gainOverNoise = gainOverNoise;
    setups.pk = pk;
    
    %Obtain the expectations for the computation of SE
    [signal,signal2,scaling] = functionComputeExpectations(Hhat,H,D,C,setups,schemeCategory,combining);
    
    [SE,Dmat,P,Rho] = functionComputeSE(signal,signal2,scaling,setups,schemes);
    
    SEdl = struct();
    if sum(strcmp(combining, 'MMSE'));SEdl.MMSE = zeros(size(SE.MMSE)); end
    if sum(strcmp(combining, 'MR'));SEdl.MR = zeros(size(SE.MR)); end
    
    [EE] = functionComputeEE(SE,SEdl,Dmat,K,N,p,P,Rho,tau_p,tau_u,tau_d,tau_c,nbrOfSchemes,combining);
    
    %Save the SE values
    if sum(strcmp(combining, 'MMSE'))
        SE_MMSE_tot(:,:,n) = SE.MMSE;
        EE_MMSE_tot(:,n) = EE.MMSE;
        D_MMSE_tot(:,:,:,n) = Dmat.MMSE;
    end
    
    if sum(strcmp(combining, 'MR'))
        SE_MR_tot(:,:,n) = SE.MR;
        EE_MR_tot(:,n) = EE.MR;
        D_MR_tot(:,:,:,n) = Dmat.MR;
    end
    
    %Remove large matrices at the end of analyzing this setup
    clear Hhat H B C R;
    
    %Display simulation progress
    timer = toc(tt);
    fprintf('-- Elapsed Time: %.2f\t', timer);
    disp(datetime);
    
end


%% Plot simulation results
figure;

subplot(3,1,1);
hold on; box on;

SEmean = mean(SE_MMSE_tot,[1 3]);

x = 0.5:2.5;
p1 = plot(x,SEmean(1)*ones(size(x)),'k-','LineWidth',1);hold on;
p2 = plot(x,SEmean(4)*ones(size(x)),'k--','LineWidth',1);hold on;

meanValue= SEmean([2 3])';

b = bar(meanValue,0.5);hold on;

xticks([1 2])
xticklabels({'10^{-4}','10^{-2}'})
legend([b p1 p2],{'S-LSFD','O-LSFD','P-LSFD'},'Interpreter','Latex','Location','SouthWest','NumColumns',2);
xlabel('$\lambda$','Interpreter','Latex');
ylabel('Average SE [bit/s/Hz]','Interpreter','Latex');
ylim([4.85 4.95]);
xlim([0.5 2.5]);
title({'(a) Average SE'},'Interpreter','Latex');

subplot(3,1,2);
hold on; box on;

x = 0.5:2.5;

EEmean = mean(EE_MMSE_tot,2);
p1 = plot(x,EEmean(1)*ones(size(x)),'k-','LineWidth',1);hold on;
p2 = plot(x,EEmean(4)*ones(size(x)),'k--','LineWidth',1);hold on;

meanValue = EEmean([2 3])';

b = bar(meanValue,0.5);hold on;

xticks([1 2])
xticklabels({'10^{-4}','10^{-2}'})
legend([b p1 p2],{'S-LSFD','O-LSFD','P-LSFD'},'Interpreter','Latex','Location','SouthEast','NumColumns',2);
xlabel('$\lambda$','Interpreter','Latex');
ylabel('Average EE [Mbit/Joule]','Interpreter','Latex');
ylim([0 2]);
xlim([0.5 2.5]);
title({'(b) Average EE'},'Interpreter','Latex');

subplot(3,1,3);
hold on; box on;

x = 0.5:2.5;
APsum = reshape(sum(D_MMSE_tot),size(SE_MMSE_tot));
APmean = mean(APsum,[1 3]);

p1 = plot(x,APmean(1)*ones(size(x)),'k-','LineWidth',1);hold on;
p2 = plot(x,APmean(4)*ones(size(x)),'k--','LineWidth',1);hold on;

meanValue = APmean([2 3])';

b = bar(meanValue,0.5);hold on;


xticks([1 2])
xticklabels({'10^{-4}','10^{-2}'})
legend([b p1 p2],{'S-LSFD','O-LSFD','P-LSFD'},'Interpreter','Latex','Location','NorthEast','NumColumns',1);
xlabel('$\lambda$','Interpreter','Latex');
ylabel('Average no. AP/UE','Interpreter','Latex');
ylim([5 42.5]);
xlim([0.5 2.5]);
title({'(c) Average AP/UE'},'Interpreter','Latex');

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

