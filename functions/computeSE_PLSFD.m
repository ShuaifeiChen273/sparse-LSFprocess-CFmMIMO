% ==================================================================
%     Compute uplink SE with P-LSFD
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

function [se_plsfd] = computeSE_PLSFD(signal,signal2,scaling,pk,opts)

%Extract the simulation setup terms
tau_c = opts.tau_c;
tau_u = opts.tau_u;
K = opts.K;
L = opts.L;
D = opts.D;

%Prepare to save simulation results
a_plsfd = zeros(L,K);

%Compute the prelog factor
preLogFactor = tau_u/tau_c;

%Prepare to store arrays for the compuataion of the n-opt LSFD vectors
gki = zeros(L,K,K);
Gki2 = zeros(L,L,K,K);
Gki2p = zeros(L,L,K,K);

%Go through all UEs
for k = 1:K
    
    %Extract which APs that serve UE k
    servingAPs = find(D(:,k)==1);
    
    %Determine which UEs that are served by partially the same set
    %of APs as UE k, i.e., the set in (5.15)
    servedUEs = find(sum(D(servingAPs,:),1)>=1);
        
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
    Ak = sum(Gki2p(servingAPs,servingAPs,k,servedUEs),4)+diag(scaling(servingAPs,k));
    a_plsfd(servingAPs,k) = pk(k)*Ak\gki(servingAPs,k,k);
end

%Prepare to store arrays for the computaion of the terms in (7.2)-(7.5)
%for the distributed opearation
bk = zeros(K,1);

ck = zeros(K,K);

%Compute (7.5) for distributed operation
sigma2 = sum(abs(a_plsfd).^2.*scaling,1).';

%Go through all UEs
for k = 1:K
    %Compute (7.2) for distributed operation
    bk(k) =  abs(a_plsfd(:,k)'*gki(:,k,k))^2;
    
    %Compute (7.3)-(7.4) for distributed operation
    for i = 1:K
        ck(i,k) = real(a_plsfd(:,k)'*Gki2(:,:,k,i)*a_plsfd(:,k));
        
    end
    
    ck(k,k) = ck(k,k) - bk(k);
    
end

%Compute uplink SEs for full and fractional power control schemes using
%Theorem 5.2
se_plsfd = preLogFactor*log2(1+bk.*pk./(ck'*pk+sigma2));

end

%% Auxiliary Function

%Scalable fractional uplink power control in (7.35)
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

%Scalable fractional downlink power allocation in Eq. (7.47)
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

%Solving group sparse optimization problem with CVX
function [x, val] = functionGWcvx(Al, b, lambda1, lambda2)

[N,K,L] = size(Al);

% In the following problem, Axl and nromsl are the auxiliary variables to
% express the problem as a convex problem using CVX functions
% cvx_begin quiet
cvx_begin quiet
variable x(K,L) complex
variable Axl(N,L) complex
variable nromsl(L,1)
minimize 1*((b-sum(Axl,2))'*(b-sum(Axl,2))) + lambda1*norm(x(:),1) + lambda2*sum(nromsl)

subject to
for l = 1:L
    Axl(:,l) == Al(:,:,l)*x(:,l);
    norm(x(:,l),2) <= nromsl(l);
end
cvx_end
x = x.';
val = cvx_optval;
end