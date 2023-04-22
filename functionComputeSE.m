% ==================================================================
% This Matlab script computes the distributed SE with UatF bound in the paper:
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
%     Signal              = Monte-Carlo estimation of expected value of
%                           first moment of the receive-combined channels
%     Signal2             = Monte-Carlo estimation of expected value of
%                           second moment of the receive-combined channels
%     Scaling             = Monte-Carlo estimation of expected value of
%                           the norm square of the combiners/precoders
%     opts                = Simulation setup terms
%     - L                 = Number of APs per setup
%     - K                 = Number of UEs in the network
%     - lambda1           = Regularization parameters for element sparsity
%     schemes             = Set of schemes considered in simulations
%     nbrOfSchemes        = Number of considered schemes
%    ---------------------------------------------------------------
%     Outputs:
%     SE                  = SE achieved with all considered
%                           combining/precoding and precessing schemes
%     Dmat                = Association matrx achieved with all considered
%                           combining/precoding and precessing schemes
% ==================================================================

function [SE,Dmat,P,Rho] = functionComputeSE(Signal,Signal2,Scaling,opts,schemes)
%Add all subfolders in the current matlab path
addpath(genpath('functions'));

%Simulation setup terms
K = opts.K;
L = opts.L;
combining = opts.combining;

%Number of considered schemes
nbrOfSchemes = length(schemes);

%Prepare to save simulation results
SE = struct();
Dmat = struct();
P = struct();
Rho = struct();

SE_MMSE = zeros(K,nbrOfSchemes);
P_MMSE = zeros(K,nbrOfSchemes);
Rho_MMSE = zeros(L,K,nbrOfSchemes);
D_MMSE = zeros(L,K,nbrOfSchemes);

SE_MR = zeros(K,nbrOfSchemes);
P_MR = zeros(K,nbrOfSchemes);
Rho_MR = zeros(L,K,nbrOfSchemes);
D_MR = zeros(L,K,nbrOfSchemes);

if sum(strcmp(schemes, 'S-LSFD'))
    Lambda1 = opts.lambda1;
    ind = find(strcmp(schemes, 'S-LSFD'));
    pk = opts.pk.all;
    
    signal = Signal.all.mmse;
    signal2 = Signal2.all.mmse;
    scaling = Scaling.all.mmse;
    [se,se_,D] = computeSE_SLSFD(signal,signal2,scaling,Lambda1,pk,opts);
    SE_MMSE(:,1) = se;
    D_MMSE(:,:,1) = ones(L,K);
    SE_MMSE(:,ind(1):ind(length(Lambda1))) = se_;
    D_MMSE(:,:,ind(1):ind(length(Lambda1))) = D;
    P_MMSE(:,1:ind(length(Lambda1))) = repmat(pk,1,length(Lambda1)+1);
    
    signal = Signal.all.mr;
    signal2 = Signal2.all.mr;
    scaling = Scaling.all.mr;
    [se,se_,D] = computeSE_SLSFD(signal,signal2,scaling,Lambda1,pk,opts);
    SE_MR(:,1) = se;
    D_MR(:,:,1) = ones(L,K);
    SE_MR(:,ind(1):ind(length(Lambda1))) = se_;
    D_MR(:,:,ind(1):ind(length(Lambda1))) = D;
    P_MR(:,1:ind(length(Lambda1))) = repmat(pk,1,length(Lambda1)+1);
end


%Compute downlink SE with partial LSFP followed by centralized FPA
if sum(strcmp(schemes, 'P-LSFD'))
    ind = find(strcmp(schemes, 'P-LSFD'));
    pk = opts.pk.dcc;
    
    signal = Signal.dcc.mmse;
    signal2 = Signal2.dcc.mmse;
    scaling = Scaling.dcc.mmse;
    [se] = computeSE_PLSFD(signal,signal2,scaling,pk,opts);
    SE_MMSE(:,ind) = se;
    P_MMSE(:,ind) = pk;
    D_MMSE(:,:,ind) = opts.D;
    
    signal = Signal.dcc.mr;
    signal2 = Signal2.dcc.mr;
    scaling = Scaling.dcc.mr;
    [se] = computeSE_PLSFD(signal,signal2,scaling,pk,opts);
    SE_MR(:,ind) = se;
    P_MR(:,ind) = pk;
    D_MR(:,:,ind) = opts.D;
end

%Compute downlink SE with virtual LSFP followed by centralized FPA
if sum(strcmp(schemes, 'V-LSFP'))
    ind = find(strcmp(schemes, 'V-LSFP'));
    if sum(strcmp(combining, 'MMSE'))
        signal = Signal.all.mmse;
        signal2 = Signal2.all.mmse;
        scaling = Scaling.all.mmse;
        [se,rho] = computeSE_VLSFP(signal,signal2,scaling,opts);
        SE_MMSE(:,ind) = se;
        P_MMSE(:,ind) = opts.pk.all;
        Rho_MMSE(:,:,ind) = rho;
        D_MMSE(:,:,ind) = ones(L,K);
    end
    
    if sum(strcmp(combining, 'MR'))
        signal = Signal.all.mr;
        signal2 = Signal2.all.mr;
        scaling = Scaling.all.mr;
        [se,rho] = computeSE_VLSFP(signal,signal2,scaling,opts);
        SE_MR(:,ind) = se;
        P_MR(:,ind) = opts.pk.all;
        Rho_MR(:,:,ind) = rho;
        D_MR(:,:,ind) = ones(L,K);
    end
end

%Compute downlink SE with virtual LSFP followed by centralized FPA
if sum(strcmp(schemes, 'P-LSFP'))
    ind = find(strcmp(schemes, 'P-LSFP'));
    if sum(strcmp(combining, 'MMSE'))
        signal = Signal.dcc.mmse;
        signal2 = Signal2.dcc.mmse;
        scaling = Scaling.dcc.mmse;
        [se,rho] = computeSE_PLSFP(signal,signal2,scaling,opts);
        SE_MMSE(:,ind) = se;
        P_MMSE(:,ind) = opts.pk.dcc;
        Rho_MMSE(:,:,ind) = rho;
        D_MMSE(:,:,ind) = opts.D;
    end
    
    if sum(strcmp(combining, 'MR'))
        signal = Signal.dcc.mr;
        signal2 = Signal2.dcc.mr;
        scaling = Scaling.dcc.mr;
        [se,rho] = computeSE_PLSFP(signal,signal2,scaling,opts);
        SE_MR(:,ind) = se;
        P_MR(:,ind) = opts.pk.dcc;
        Rho_MR(:,:,ind) = rho;
        D_MR(:,:,ind) = opts.D;
    end
end


%Compute downlink SE with distributed FPA
if sum(strcmp(schemes, 'FPA'))
    ind = find(strcmp(schemes, 'FPA'));
    if sum(strcmp(combining, 'MMSE'))
        signal = Signal.dcc.mmse;
        signal2 = Signal2.dcc.mmse;
        scaling = Scaling.dcc.mmse;
        [se] = computeSE_FPA(signal,signal2,scaling,opts);
        SE_MMSE(:,ind) = se;
        P_MMSE(:,ind) = opts.pk.dcc;
        Rho_MMSE(:,:,ind) = opts.rholk.dcc;
        D_MMSE(:,:,ind) = opts.D;
    end
    
    if sum(strcmp(combining, 'MR'))
        signal = Signal.dcc.mr;
        signal2 = Signal2.dcc.mr;
        scaling = Scaling.dcc.mr;
        [se] = computeSE_FPA(signal,signal2,scaling,opts);
        SE_MR(:,ind) = se;
        P_MR(:,ind) = opts.pk.dcc;
        Rho_MR(:,:,ind) = opts.rholk.dcc;
        D_MR(:,:,ind) = opts.D;
    end
end

%Compute downlink SE with distributed FPA followed by centralized FPA
if sum(strcmp(schemes, 'HEUR'))
    ind = find(strcmp(schemes, 'HEUR'));
    if sum(strcmp(combining, 'MMSE'))
        signal = Signal.dcc.mmse;
        signal2 = Signal2.dcc.mmse;
        scaling = Scaling.dcc.mmse;
        [se,rho] = computeSE_HEUR(signal,signal2,scaling,opts);
        SE_MMSE(:,ind) = se;
        P_MMSE(:,ind) = opts.pk.dcc;
        Rho_MMSE(:,:,ind) = rho;
        D_MMSE(:,:,ind) = opts.D;
    end
    
    if sum(strcmp(combining, 'MR'))
        signal = Signal.dcc.mr;
        signal2 = Signal2.dcc.mr;
        scaling = Scaling.dcc.mr;
        [se,rho] = computeSE_HEUR(signal,signal2,scaling,opts);
        SE_MR(:,ind) = se;
        P_MR(:,ind) = opts.pk.dcc;
        Rho_MR(:,:,ind) = rho;
        D_MR(:,:,ind) = opts.D;
    end
end

if sum(strcmp(combining, 'MMSE'))
SE.MMSE = SE_MMSE;
Dmat.MMSE = D_MMSE;
Rho.MMSE = Rho_MMSE;
P.MMSE = P_MMSE;
end

if sum(strcmp(combining, 'MR'))
SE.MR = SE_MR;
Dmat.MR = D_MR;
Rho.MR = Rho_MR;
P.MR = P_MR;
end
end
