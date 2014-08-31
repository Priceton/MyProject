function [model, llh] = classRvmEbFp(X, t, alpha)
% Relevance Vector Machine classification training by empirical bayesian (ARD)
% using fix point update (Mackay update)
if nargin < 3
    alpha = 0.02;
end
n = size(X,2);
X = [X;ones(1,n)];
d = size(X,1);
alpha = alpha*ones(d,1);

tol = 1e-4;
maxiter = 100;
llh = -inf(1,maxiter);
infinity = 1e+10;
for iter = 2:maxiter
    used = alpha<infinity;
    alphaUsed = alpha(used);
    [w,V,lllh] = optLogitNewton(X(used,:),t,alphaUsed);  % lllh = logitloglikelihood
    w2 = w.^2;
    
    logdetS = -2*sum(log(diag(V)));
    llh(iter) = lllh+0.5*(sum(log(alphaUsed))-logdetS-dot(alphaUsed,w2)-n*log(2*pi)); % 7.114
    if abs(llh(iter)-llh(iter-1)) < tol; break; end

    dgSigma = dot(V,V,2);
    gamma = 1-alphaUsed.*dgSigma;   % 7.89
    alpha(used) = gamma./w2;           % 7.87
end
llh = llh(2:iter);

model.used = used;
model.w = w;
model.alpha = alpha;

