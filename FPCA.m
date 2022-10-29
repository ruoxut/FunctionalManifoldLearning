function [ X_d,X_p,d ] = FPCA( t,X )
% Functional principal component analysis.
% Input:
% t: p*1 time vector;
% X: p*n data matrix, each column contains function values of an individual.
% Output:
% X_d: d*n PC scores;
% X_p: p*n dimension-reduced function values;
% d: the number of PCs explaining 95% of the variance.

% Author: Ruoxu Tan; date: 2022/Oct/29; Matlab version: R2020a.

if isrow(t)
    t = t';
end
    
if length(t) ~= size(X,1)
    error('Dimensions of the input functional data do not match.')
end

n = size(X,2);
mu = mean(X,2);
X_cen = X - mu;
Cov = X_cen * X_cen' / n;
[phi,~,expd] = pcacov(Cov);
norm_phi = sqrt(trapz(t,phi.^2,1));       
phi = phi ./ norm_phi;

d = 1;
s = expd(1);
while s<95
    d = d+1;
    s = s+expd(d);
end
X_d = zeros(d,n);
X_p = zeros(length(t),n);
for i = 1:n
    xi = trapz(t,X_cen(:,i).*phi(:,1:d),1);
    X_d(:,i) = xi';
    X_p(:,i) = mu + sum(xi.*phi(:,1:d),2);
end

end

