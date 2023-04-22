function [se_heur,rho_heur] = computeSE_HEUR(signal,signal2,scaling,opts)

tau_c = opts.tau_c;
tau_d = opts.tau_d;
K = opts.K;
L = opts.L;
rho_tot = opts.rho_tot;
rho_dist = opts.rholk.dcc;
D = opts.D;
gainOverNoise = opts.gainOverNoise;

% Prepare normalized LSFP vectors
b_heur = rho_dist;
se_heur = zeros(K,1);
rho_heur = zeros(L,K);

preLogFactor = tau_d/tau_c;

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
    b_heur(:,k) = b_heur(:,k)/norm(b_heur(:,k));
end
%Take the real part (in the SINR expression,the imaginary terms cancel
%each other)
Ck = real(Ck);

pow = zeros(K,1);
maxPow = zeros(K,1);

upsilon1 = -0.4;
upsilon2 = 0.5;
upsilon3 = 0.2;
for k = 1:K
    %Extract which UEs are served by AP l
    servingAPs = find(D(:,k)==1);
    pow(k) =  sum(gainOverNoise(servingAPs,k).^upsilon3).^upsilon1;
    pow(k) = pow(k)/max(abs(b_heur(servingAPs,k)).^2)^upsilon2;
    maxPow(k) = max(abs(b_heur(servingAPs,k)).^2);
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
        rho_heur(l,k) = rho_cent(k)*abs(b_heur(l,k))^2;
    end
end

%% Compute downlink SE
% Go through each UE
for k = 1:K
    %Find APs that serve UE k
    servingAPs = find(D(:,k)==1);
    fkk = conj(vec(signal(k,k,servingAPs)))./sqrt(scaling(servingAPs,k));
    numm = rho_cent(k)*abs(b_heur(servingAPs,k)'*fkk)^2;
    denomm = 1-numm;
    for i = 1:K
        servingAPs = find(D(:,i)==1);
        La = length(servingAPs);
        Cki = Ck(1:La,1:La,k,i);
        denomm = denomm + rho_cent(i)*real(b_heur(servingAPs,i)'*Cki*b_heur(servingAPs,i));
    end
    se_heur(k) = preLogFactor*log2(1+numm/denomm);
end
end