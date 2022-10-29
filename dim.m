function [ d ] = dim( t,X,delta )
% Intrinsic dimension estimation for functional data by adapting the method
% proposed by Facco et al. (2017).
% Reference:
% Facco, E., d’Errico, M., Rodriguez, A. and Laio, A. (2017). Estimating the intrinsic 
% dimension of datasets by a minimal neighborhood information. Scientific Reports, 7, 1–8.
% Input:
% t: p*1 time interval;
% X: p*n data matrix, each column includes values of a curve;
% delta: a number in (0,1), the fraction of data to be used.
% Output:
% d: estimated intrinsic dimension.

% Author: Ruoxu Tan; date: 2022/Oct/29; Matlab version: R2020a.

if delta >= 1 || delta < 0
    error('delta must be in (0,1).')
end

if nargin < 3
    delta = 0.9;
end

n = size(X,2);
mu_X = mean(X,2);
X_cen = X - mu_X;
Cov = X_cen * X_cen' / n;
[phi,~,expd] = pcacov(Cov);
norm_phi = sqrt(trapz(t,phi.^2));       
phi = phi ./ norm_phi;

n_c = 1;
s = expd(1);
while s<95
    n_c = n_c+1;
    s = s+expd(n_c);
end
Xi_n_c = zeros(n_c,n); 
for i = 1:n
    xi = trapz(t,X_cen(:,i).*phi(:,1:n_c));
    Xi_n_c(:,i) = xi'; 
end

G = zeros(n);
for i = 1:n
    for j = i+1:n
        G(i,j) = norm(Xi_n_c(:,i)-Xi_n_c(:,j));
        G(j,i) = G(i,j);
    end
end

mu = zeros(1,n);
for i = 1:n
    d_i = sort(G(i,:));
    mu(i) = d_i(3)/d_i(2);
end
n_sub = round(delta*n);
if n_sub == n
    n_sub = n-1;
end

mu = sort(mu);
mu = mu(1:n_sub);

X_data = log(mu);
Y_data = -log(1-(1:n_sub)./n); 
d = round(sum(X_data.*Y_data) / sum(X_data.^2));

end