% ==================================================================
% This Matlab script performs Algorithm 1 in the paper:
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
%     mu0                 = Regularization parameter for element sparsity
%    ---------------------------------------------------------------
%     Outputs:
%     x                   = Optimized sparse vector with dimension K*1
%     out                 = Simulation setup terms
%     - fvec              = Function values of each iterations
%     - itr               = Number of outer iterations
%     - itr_inn           = Number of inner iterations
%     - fval              = Function value when the iteration stops
%     - tt                = Elapsed time
% ==================================================================
% This Matlab script is written with reference to:
% 
% http://faculty.bicmr.pku.edu.cn/~wenzw/optbook/pages/LASSO_con/LASSO_con.html
% ==================================================================

function [x, out] = warmRestart(x0, A, b, mu0)

% Default parameters
opts = struct();
%     out                 = Simulation setup terms
%     - maxit             = Maximal mumber allowed for outer iterationss
%     - maxit_inn         = Maximal mumber allowed for inner iterationss
%     - ftol              = Threshold for stop regarding function value
%     - gtol              = Threshold for stop regarding gradient value
%     - factor            = Shrinking factor for regularization parameter
%     - mu1               = Regularization parameter with a large value
%     - alpha0            = Initial step length
if ~isfield(opts, 'maxit'); opts.maxit = 30; end
if ~isfield(opts, 'maxit_inn'); opts.maxit_inn = 200; end
if ~isfield(opts, 'ftol'); opts.ftol = 1e-8; end
if ~isfield(opts, 'gtol'); opts.gtol = 1e-6; end
if ~isfield(opts, 'factor'); opts.factor = 0.1; end
if ~isfield(opts, 'mu1'); opts.mu1 = 100; end
if ~isfield(opts, 'gtol_init_ratio'); opts.gtol_init_ratio = 1/opts.gtol; end
if ~isfield(opts, 'ftol_init_ratio'); opts.ftol_init_ratio = 1e5; end
if ~isfield(opts, 'opts1'); opts.opts1 = struct(); end
if ~isfield(opts, 'etaf'); opts.etaf = 1e-1; end
if ~isfield(opts, 'etag'); opts.etag = 1e-1; end
L = eigs(A'*A,1);
if ~isfield(opts, 'alpha0'); opts.alpha0 = 1/L; end


% Start with a large mu1
out = struct();
out.fvec = [];
k = 0;
x = x0;
mu_t = opts.mu1;
tt = tic;

f = Func(A, b, mu_t, x);

opts1 = opts.opts1;
% Start with large ftol and gtol
opts1.ftol = opts.ftol*opts.ftol_init_ratio;
opts1.gtol = opts.gtol*opts.gtol_init_ratio;
out.itr_inn = 0;

%% Warm-restart
while k < opts.maxit
    
    opts1.maxit = opts.maxit_inn;
    opts1.gtol = max(opts1.gtol * opts.etag, opts.gtol);
    opts1.ftol = max(opts1.ftol * opts.etaf, opts.ftol);
    opts1.alpha0 = opts.alpha0;
    
    fp = f;
    [x, out1] = proximalNesterov(x, A, b, mu_t, mu0, opts1);
    f = out1.fval;
    out.fvec = [out.fvec, out1.fvec];
    k = k + 1;
    
    nrmG = norm(x - prox(x - A'*(A*x - b),mu0),2);
    
    if ~out1.flag
        mu_t = max(mu_t * opts.factor, mu0);
    end
    
    if mu_t == mu0 && (nrmG < opts.gtol || abs(f-fp) < opts.ftol)
        break;
    end
    
    out.itr_inn = out.itr_inn + out1.itr;
end

out.fval = f;
out.tt = toc(tt);
out.itr = k;

%% Auxiliary Function
% Objective function
    function f = Func(A, b, mu0, x)
        w = A * x - b;
        f = 0.5 * (w' * w) + mu0 * norm(x, 1);
    end

% Proximal operator
    function y = prox(x, mu)
        y = max(abs(x) - mu, 0);
        y = sign(x) .* y;
    end
end