function [X_d,phi,lambda] = FPCA(t,X,FVE)
% Functional principal component analysis.
% Input:
% t: 1*p time vector;
% X: n*p data matrix, each row contains function values of an
% individual;
% FVE: fraction in percentage of variance explained to select the number of PCs, 95 as default.
% Output:
% X_d: n*d PC scores;
% phi: d*p PC basis functions;
% lambda: the eigenvalues.

% Author: Ruoxu Tan; date: 2024/Apr/30; Matlab version: R2023b.

if nargin < 3
    FVE = 95;
end

if iscolumn(t)
    t = t';
end
    
if length(t) ~= size(X,2)
    error('Dimensions of the input functional data do not match.')
end

delta_t = mean(diff(t)); 
n = size(X,1);
mu = mean(X,1);
X_cen = X - mu;
Cov = X_cen' * X_cen / n;
[phi,lambda,expd] = pcacov(Cov);
norm_phi = sqrt(sum(phi.^2.*delta_t,1));       
phi = phi ./ norm_phi;
phi = phi';
lambda = norm_phi'.^2 .* lambda; 

d = 1;
s = expd(1);
while s<FVE
    d = d+1;
    s = s+expd(d);
end

X_d = zeros(n,d);
for i = 1:n
    xi = sum(X_cen(i,:).*phi(1:d,:).*delta_t,2);
    X_d(i,:) = xi';
end

phi = phi(1:d,:);

end

