% ==================================================================
%     Compute total energy efficiency 
%     Shuaifei Chen
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
%     Change logs:  
%     - Ver. 0.1: Shuaifei Chen (May 8, 2022)
%       This test version includes a primary functional structure, 
%       but the computation for SE with Monte Carlo simulation seems 
%       incorrect.
% ==================================================================

function [EE] = functionComputeEE(SEul,SEdl,Dmat,K,N,p,P,Rho,tau_p,tau_u,tau_d,tau_c,nbrOfSchemes,combining)

%Communication bandwidth (MHz)
B = 20;

eta_ue = 0.4;

eta_ap = 0.4;

%(mW)
pc_ue = 100;
pc_ap = 200;

%(mW)
pfix_fh = 825;
pfix_cpu = 5000;

%(mW)
p_pro = 800;

%(mW/(Mbit/s))
p_dec = 0.8;
p_cod = 0.1;

% %(mW)
% p_lsfd = 200;
% p_lsfp = 200;

%mW
ptf_fh = 10;

EE = struct();


%% MMSE
if sum(strcmp(combining, 'MMSE'))
%Prepare to store simulation results
EE_MMSE = zeros(nbrOfSchemes,1);

SE_MMSEul = SEul.MMSE;
SE_MMSEdl = SEdl.MMSE;

D_MMSE = Dmat.MMSE;

for i = 1:nbrOfSchemes
    pk = P.MMSE(:,i);
    rho = Rho.MMSE(:,:,i);
    
    D = D_MMSE(:,:,i);
    activeAPs = find(sum(D,2)~=0);
    
    sumSEul = sum(SE_MMSEul(:,i));
    sumSEdl = sum(SE_MMSEdl(:,i));
    
    %(mW)
    p_ue = K*(tau_p*p/tau_c/eta_ue + pc_ue) + tau_u*sum(pk)/tau_c/eta_ue;
    
    p_ap = 0;
    p_fh = 0;
    
    for j = 1:length(activeAPs)
        l = activeAPs(j);
        servingUEs = find(D(l,:) == 1);
        Kl = length(servingUEs);
        %(mW)
        p_ap = p_ap + N*pc_ap + N*Kl*p_pro ...
            + tau_d/tau_c/eta_ap*sum(rho(l,servingUEs));
        
        %(mW)
        p_fh = p_fh + pfix_fh + (tau_u + tau_d)/tau_c*Kl*ptf_fh;
    end
    
%     p_cpu = pfix_cpu + B*sumSEul*p_dec + B*sumSEdl*p_cod  ...
%         + sum(D,'all')*(tau_u/tau_c*p_lsfd + tau_d/tau_c*p_lsfp);
        p_cpu = pfix_cpu + B*sumSEul*p_dec + B*sumSEdl*p_cod;
    
    %(Mbit/Joule)
    EE_MMSE(i) = B*(sumSEul + sumSEdl)/(p_ue+p_ap+p_fh+p_cpu)*1e3;
    
end

EE.MMSE = EE_MMSE;
end

%% MR
if sum(strcmp(combining, 'MR'))
%Prepare to store simulation results
EE_MR = zeros(nbrOfSchemes,1);

SE_MRul = SEul.MR;
SE_MRdl = SEdl.MR;

D_MR = Dmat.MR;

for i = 1:nbrOfSchemes
    
    pk = P.MR(:,i);
    rho = Rho.MR(:,:,i);
    
    D = D_MR(:,:,i);
    
    sumSEul = sum(SE_MRul(:,i));
    sumSEdl = sum(SE_MRdl(:,i));
    
    %(mW)
    p_ue = K*(tau_p*p/tau_c/eta_ue + pc_ue) + tau_u*sum(pk)/tau_c/eta_ue;
    
    p_ap = 0;
    p_fh = 0;
    
    for j = 1:length(activeAPs)
        l = activeAPs(j);
        servingUEs = find(D(l,:) == 1);
        Kl = length(servingUEs);
        %(mW)
        p_ap = p_ap + N*pc_ap + N*Kl*p_pro ...
            + tau_d/tau_c/eta_ap*sum(rho(l,servingUEs));
        
        %(mW)
        p_fh = p_fh + pfix_fh + (tau_u + tau_d)/tau_c*Kl*ptf_fh;
    end
    
%     p_cpu = pfix_cpu + B*sumSEul*p_dec + B*sumSEdl*p_cod  ...
%         + sum(D,'all')*(tau_u/tau_c*p_lsfd + tau_d/tau_c*p_lsfp);
    p_cpu = pfix_cpu + B*sumSEul*p_dec + B*sumSEdl*p_cod;
    
    %(Mbit/Joule)
    EE_MR(i) = B*(sumSEul + sumSEdl)/(p_ue+p_ap+p_fh+p_cpu)*1e3;
    
end
EE.MR = EE_MR;
end
end
