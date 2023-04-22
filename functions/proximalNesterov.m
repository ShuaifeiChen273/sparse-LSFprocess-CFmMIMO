% ==================================================================
% This Matlab script performs Algorithm 2 in the paper:
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
%     x0                  = Initial optimized vector with dimension K*1
%     A                   = Matrix with dimension L*K
%     b                   = Vector with dimension L*1
%     mu                  = Regularization parameter for element sparsity
%     mu0                 = Initial regularization parameter
%    ---------------------------------------------------------------
%     Outputs:
%     x                   = Optimized sparse vector with dimension K*1
%     out                 = Simulation setup terms
% ==================================================================
% This Matlab script is written with reference to:
% 
% http://faculty.bicmr.pku.edu.cn/~wenzw/optbook/pages/lasso_proxg/LASSO_Nesterov_inn.html
% ==================================================================

function [x, out] = proximalNesterov(x0, A, b, mu, mu0, opts)

if ~isfield(opts, 'maxit'); opts.maxit = 10000; end
if ~isfield(opts, 'ftol'); opts.ftol = 1e-8; end
if ~isfield(opts, 'gtol'); opts.gtol = 1e-6; end
if ~isfield(opts, 'verbose'); opts.verbose = 1; end
if ~isfield(opts, 'alpha0'); opts.alpha0 = 1e-3; end
if ~isfield(opts, 'ls'); opts.ls = 1; end
if ~isfield(opts, 'bb'); opts.bb = 0; end

k = 0;
t = opts.alpha0;
tt = tic;
x = x0;
y = x;
xp = x0;

fp = inf;
r = A*x0 - b;
g = A'*r;
tmp = .5*norm(r,2)^2;
f =  tmp + mu0*norm(x,1);
tmpf = tmp + mu*norm(x,1);
nrmG = norm(x - prox(x - g,mu),2);
out = struct();
out.fvec = tmp + mu0*norm(x,1);

% Paramters for line search
Cval = tmpf; Q = 1; gamma = 0.85; rhols = 1e-6;

%% Main loop
while k < opts.maxit && nrmG > opts.gtol && abs(f - fp) > opts.ftol

    gp = g;
    yp = y;
    fp = tmpf;
    
    % Nesterov step
    theta = (k - 1) / (k + 2);
    y = x + theta * (x - xp);
    xp = x;
    r = A * y - b;
    g = A' * r;
    
    % BB step length and line search
    if opts.bb && opts.ls
        dy = y - yp;
        dg = g - gp;
        dyg = abs(dy'*dg);
        
        if dyg > 0
            if mod(k,2) == 0
                t = norm(dy,2)^2/dyg;
            else
                t = dyg/norm(dg,2)^2;
            end
        end
        
        t = min(max(t,opts.alpha0),1e12);

    else
        t = opts.alpha0;
    end

    x = prox(y - t * g, t * mu);

    if opts.ls
        nls = 1;
        while 1
            tmp = 0.5 * norm(A*x - b, 2)^2;
            tmpf = tmp + mu*norm(x,1);
            if tmpf <= Cval - 0.5*rhols*t*norm(x-y,2)^2 || nls == 5
                break;
            end
            
            t = 0.2*t; nls = nls + 1;
            x = prox(y - t * g, t * mu);
        end

        % Update the function value
        f = tmp + mu0*norm(x,1);

        % Update the parameters for line search
        Qp = Q; Q = gamma*Qp + 1; Cval = (gamma*Qp*Cval + tmpf)/Q;

    else
        f = 0.5 * norm(A*x - b, 2)^2 + mu0*norm(x,1);
    end

    nrmG = norm(x - y,2)/t;

    k = k + 1;
    out.fvec = [out.fvec, f];
    
    if k > 20 && min(out.fvec(k-19:k)) > out.fvec(k-20)
        break;
    end
end

if k == opts.maxit
    out.flag = 1;
else
    out.flag = 0;
end

out.fvec = out.fvec(1:k);
out.fval = f;
out.itr = k;
out.tt = toc(tt);
out.nrmG = nrmG;
end
%% Auxiliary Function
% Proximal operator
function y = prox(x, mu)
y = max(abs(x) - mu, 0);
y = sign(x) .* y;
end