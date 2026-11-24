function [X_d,phi,lambda] = FPCA(t,X,FVE)
% Functional PCA on an equally spaced grid (95 percent FVE by default).
% X has subjects in rows. phi contains the retained L2-normalized basis
% functions; lambda contains all numerically positive covariance eigenvalues.
if nargin < 3
    FVE = 95;
end
[~,delta_t] = fml_validate_grid(t,X);
validateattributes(FVE,{'numeric'},{'real','finite','scalar','>',0,'<=',100});
n = size(X,1);
if n < 2
    error('FML:InsufficientSamples','FPCA requires at least two observations.');
end
X_cen = double(X) - mean(double(X),1);
% Economy SVD is the same PCA with divisor n, without forming a covariance.
[~,S,V] = svd(X_cen,'econ');
s = diag(S);
tol = max(size(X_cen))*eps(max(s));
keep = s > tol;
if ~any(keep)
    error('FML:ZeroVariance','FPCA is undefined for zero-variance data.');
end
s = s(keep);
V = V(:,keep);
lambda = (s.^2/n)*delta_t;
d = find(cumsum(lambda)/sum(lambda) >= FVE/100,1,'first');
if isempty(d)
    d = numel(lambda); % FVE=100 can have final-bit summation error.
end
phi = V(:,1:d)'/sqrt(delta_t);
X_d = (X_cen*phi')*delta_t;
end

