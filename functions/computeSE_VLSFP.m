% ==================================================================
%     Compute downlink SE with virtual LSFP followed by centralized FPA
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
%     opts                = Simulation setup terms
%     - D                 = Matrix with dimension L x K where (l,k)=1 indicates 
%                           that AP l serves UE k
%     - L                 = Number of APs per setup
%     - K                 = Number of UEs in the network
%     - tau_c             = Number of channel uses per coherence block
%     - tau_d             = Number of channel uees for uplink per coherence block
%     - rho_tot           = Total downlink transmit power per AP (mW)
%    ---------------------------------------------------------------
%     Outputs:
%     se_scheme           = SE achieved with all considered combiners/precoders
%     b_scheme            = Normalized LSFP vectors achieved with all
%                           considered combiners/precoders
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

function [se_vlsfp,rho_vlsfp] = computeSE_VLSFP(signal,signal2,scaling,opts)

%Extract the simulation setup terms
tau_c = opts.tau_c;
tau_d = opts.tau_d;
K = opts.K;
L = opts.L;
rho_tot = opts.rho_tot;
gainOverNoise = opts.gainOverNoise;
D = ones(size(gainOverNoise));

%Prepare to save simulation results
b_vlsfp = zeros(L,K);
rho_vlsfp = zeros(L,K);
se_vlsfp = zeros(K,1);

%Compute the prelog factor
preLogFactor = tau_d/tau_c;

%Prepare to store the interference matrix in (7.26)
Ck = zeros(L,L,K,K);

%Go through all UEs
for k = 1:K
    %Find the APs that serve UE k
    servingAPs = find(D(:,k)==1);
    %The number of APs that serve UE k
    La = length(servingAPs);
    %Compute the vector in (7.25) for UE k (only the non-zero indices correspondig to
    %serving APs are considered)
    fkk = conj(vec(signal(k,k,servingAPs)))./sqrt(scaling(servingAPs,k));
    
    CCk = eye(La);
    %Go through all UEs
    for i = 1:K
        %Compute the matrices
        if i==k
            Cik = fkk*fkk';
        else
            Cik = diag(1./sqrt(scaling(servingAPs,k)))...
                *(conj(vec(signal(i,k,servingAPs)))...
                *conj(vec(signal(i,k,servingAPs)))')...
                *diag(1./sqrt(scaling(servingAPs,k)));
        end
        
        for j = 1:La
            Cik(j,j) = signal2(i,k,servingAPs(j))/scaling(servingAPs(j),k);
        end
        
        Ck(1:La,1:La,i,k) = Cik;
        
        CCk = CCk + Cik;
    end
    
    %Compute normalized virtual LSFD vectors for UE k
    b_vlsfp(servingAPs,k) =  CCk\fkk;
    b_vlsfp(:,k) = b_vlsfp(:,k)/norm(b_vlsfp(:,k));
end



%% Centralized fractional power allocation
pow = zeros(K,1);
maxPow = zeros(K,1);

upsilon1 = -0.4;
upsilon2 = 0.5;
upsilon3 = 0.2;
for k = 1:K
    %Extract which UEs are served by AP l
    servingAPs = find(D(:,k)==1);
    pow(k) =  sum(gainOverNoise(servingAPs,k).^upsilon3).^upsilon1;
    pow(k) = pow(k)/max(abs(b_vlsfp(servingAPs,k)).^2)^upsilon2;
    maxPow(k) = max(abs(b_vlsfp(servingAPs,k)).^2);
end
normalizationFactor = 0;

for ell = 1:L
    servedUEs = find(D(ell,:)==1);
    temporScalar = maxPow(servedUEs)'*pow(servedUEs)/rho_tot;
    normalizationFactor = max(normalizationFactor,temporScalar);
end
rho_cent = pow/normalizationFactor;

for k = 1:K
    for l = 1:L
        rho_vlsfp(l,k) = rho_cent(k)*abs(b_vlsfp(l,k))^2;
    end
end

%% Compute downlink SE
% Go through each UE
for k = 1:K
    %Find APs that serve UE k
    servingAPs = find(D(:,k)==1);
    fkk = conj(vec(signal(k,k,servingAPs)))./sqrt(scaling(servingAPs,k));
    numm = rho_cent(k)*abs(b_vlsfp(servingAPs,k)'*fkk)^2;
    denomm = 1-numm;
    for i = 1:K
        servingAPs = find(D(:,i)==1);
        La = length(servingAPs);
        Cki = Ck(1:La,1:La,k,i);
        denomm = denomm + rho_cent(i)*real(b_vlsfp(servingAPs,i)'*Cki*b_vlsfp(servingAPs,i));
    end
    se_vlsfp(k) = preLogFactor*log2(1+numm/denomm);
end

end