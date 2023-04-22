% ==================================================================
%     Compute uplink SE considering two cases:
%     1. O-LSFD:  Each AP serves all UEs
%     2. S-LSFD:  Perform sparse optimization based on the original
%                 channel statics in case O-LSFD.
%                 obtain S-LSFD vector and association matrix D.
%                 Replace O-LSFD vector with S-LSFD vector and compute
%                 SE based on the orginial channel statics and transmit power
%     Shuaifei Chen
% ==================================================================
%     Inputs:
%     signal              = Matrix with dimension K x K x L where (i,k,l)
%                           is the Monte-Carlo estimation of expected value of
%                           h_{il}^HD_{kl}w_{kl} where w_{kl} is LP-MMSE combiner/precoder
%     signal2             = Matrix with dimension K x K x L where (i,k,l)
%                           is the Monte-Carlo estimation of expected value of
%                           |h_{il}^HD_{kl}w_{kl}|^2 where w_{kl} is LP-MMSE combiner/precoder
%     scaling             = Matrix with dimension L x K where (l,k) is
%                           the Monte-Carlo estimation of expected value of
%                           the norm square of D_{kl}w_{kl} for LP-MMSE combiner/precoder
%     lambda1             = Regularization parameters for element sparsity
%     opts                = Simulation setup terms
%     - L                 = Number of APs per setup
%     - K                 = Number of UEs in the network
%     - tau_c             = Number of channel uses per coherence block
%     - tau_d             = Number of channel uees for uplink per coherence block
%     - pk                = Uplink transmit power per UE (mW)
%     - lambda2           = Regularization parameters for element sparsity
%    ---------------------------------------------------------------
%     Outputs:
%     se_scheme           = SE achieved with all considered combiners/precoders
%     D_scheme            = Association matrx achieved with all considered
%                           combiners/precoders
%     a_scheme            = LSFD vectors achieved with all considered combiners/precoders
% ==================================================================
%     Notes:
%     Unless otherwise specified, the equitions and algorithms
%     involved in this script refer to the following reference:
%     Ozlem Tugfe Demir, Emil Bjornson and Luca Sanguinetti (2021),
%     "Foundations of User-Centric Cell-Free Massive MIMO",
%     Foundations and Trends in Signal Processing: Vol. 14: No. 3-4,
%     pp 162-472. DOI: 10.1561/2000000109
% ==================================================================
%     Change logs:
%     - Ver. 0.1: Shuaifei Chen (May 8, 2022)
%       This test version includes a primary functional structure,
%       but the computation for SE with Monte Carlo simulation seems
%       incorrect.
% ==================================================================

function [se_olsfd,se_slsfd,D_slsfd] = computeSE_SLSFD(signal,signal2,scaling,Lambda1,pk,opts)

%Extract the simulation setup terms
tau_c = opts.tau_c;
tau_u = opts.tau_u;
K = opts.K;
L = opts.L;

%Prepare to save simulation results
a_olsfd = zeros(L,K);
se_slsfd = zeros(K,length(Lambda1));
D_slsfd = zeros(L,K,length(Lambda1));

%Compute the prelog factor
preLogFactor = tau_u/tau_c;

%Prepare to store arrays for the compuataion of the n-opt LSFD vectors
gki = zeros(L,K,K);
Gki2 = zeros(L,L,K,K);
Gki2p = zeros(L,L,K,K);

%Prepare to store the matrices and vectors for sparse optimization
Akbar = zeros(L,L,K);
xk = zeros(L,K);

%Go through all UEs
for k = 1:K
    
    %Go through all UEs
    for i = 1:K
        
        %Compute the expecatation terms in (5.41) for the computation of n-opt LSFD
        gki(:,k,i) = conj(vec(signal(i,k,:)));
        
        Gki2(:,:,k,i) = gki(:,k,i)*gki(:,k,i)';
        for ell = 1:L
            Gki2(ell,ell,k,i) = signal2(i,k,ell);
        end
        
        Gki2p(:,:,k,i) = pk(i)*Gki2(:,:,k,i);
        
    end
    
    %Compute n-opt LSFD vectors for UE k
    Ak = sum(Gki2p(:,:,k,:),4)+diag(scaling(:,k));
    
    Akbar(:,:,k) = sqrtm(Ak);
    xk(:,k) = pk(k)*gki(:,k,k);
    
    a_olsfd(:,k) = Ak\xk(:,k);
end

%% Case 1: O-LSFD
%Prepare to store arrays for the computaion of the terms in (7.2)-(7.5)
%for the distributed opearation
bk = zeros(K,1);

ck = zeros(K,K);

%Compute (7.5) for distributed operation
sigma2 = sum(abs(a_olsfd).^2.*scaling,1).';

%Go through all UEs
for k = 1:K
    %Compute (7.2) for distributed operation
    bk(k) =  abs(a_olsfd(:,k)'*gki(:,k,k))^2;
    
    %Compute (7.3)-(7.4) for distributed operation
    for i = 1:K
        ck(i,k) = real(a_olsfd(:,k)'*Gki2(:,:,k,i)*a_olsfd(:,k));
        
    end
    
    ck(k,k) = ck(k,k) - bk(k);
    
end

%Compute uplink SEs for full and fractional power control schemes using
%Theorem 5.2
se_olsfd = preLogFactor*log2(1+bk.*pk./(ck'*pk+sigma2));

%% Case 2: S-LSFD
for ii = 1:length(Lambda1)
    
    lambda1 = Lambda1(ii);
    
    a_slsfd = zeros(L,K);
    for k = 1:K
        var_x0 = a_olsfd(:,k);
        var_A = Akbar(:,:,k);
        var_b = (Akbar(:,:,k)')\xk(:,k);
        var_mu = lambda1;
        [a_slsfd(:,k)] = warmRestart(var_x0, var_A, var_b, var_mu);
    end
    %Set the small entries to zero and obtain the association matrix
    D = abs(a_slsfd) >= 1e-3;
    
    %Prepare to store arrays for the computaion of the terms in (7.2)-(7.5)
    %for the distributed opearation
    bk = zeros(K,1);
    
    ck = zeros(K,K);
    
    %Compute (7.5) for distributed operation
    sigma2 = sum(abs(a_slsfd).^2.*scaling,1).';
    
    %Go through all UEs
    for k = 1:K
        %Compute (7.2) for distributed operation
        bk(k) =  abs(a_slsfd(:,k)'*gki(:,k,k))^2;
        
        %Compute (7.3)-(7.4) for distributed operation
        for i = 1:K
            ck(i,k) = real(a_slsfd(:,k)'*Gki2(:,:,k,i)*a_slsfd(:,k));
            
        end
        
        ck(k,k) = ck(k,k) - bk(k);
        
    end
    
    %Compute uplink SEs for full and fractional power control schemes using
    %Theorem 5.2
    se_slsfd(:,ii) = preLogFactor*log2(1+bk.*pk./(ck'*pk+sigma2));
    D_slsfd(:,:,ii) = D;
    
end
end