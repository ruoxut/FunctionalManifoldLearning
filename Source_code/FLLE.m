function [mu] = FLLE(t,X,Z,x,K_pca,h)
% Functional local linear estimator on tangent space
% Input:
% t: 1*p time interval;
% X: n*p covariate matrix, each row contains function values of an individual; 
% Z: n*d outcome matrix, each row contains the outcome of an individual;
% x: 1*p function values where we examine the estimator;
% K_pca: number of nearest neighbours used in local PCA;
% h: bandwidth;
% Output:
% mu: 1*d estimated conditional mean at x.

% Author: Ruoxu Tan; date: 2025/May; Matlab version: R2024b.
if length(t) ~= size(X,2) || size(X,1)~=size(Z,1)
    error('Dimensions of the input data do not match.')
end

if iscolumn(t)
    t = t';
end

d = size(Z,2);
n = size(X,1);

if K_pca < d
    error('K_pca is smaller than the intrinsic dimension.')
end  

% Tangent space at x
delta_t = mean(diff(t));
G_x = sqrt(sum((X-x).^2.*delta_t,2));
[~,ind] = sort(G_x);
mu_x = mean(X(ind(1:K_pca),:),1);
X_x_cen = X(ind(1:K_pca),:)-mu_x; 
[~,~,phi_x] = svds(X_x_cen,d);   
norm_phi_x = sqrt(sum(phi_x.^2.*delta_t,1));       
phi_x = phi_x ./ norm_phi_x; 
TM_x = phi_x'; % d*p matrix, each row contains a basis function

X_mat = zeros(n,d+1);
X_mat(:,1) = 1;
for i = 1:n
    X_mat(i,2:end) = sum((X(i,:)-x).*TM_x.*delta_t,2)';
end

W = diag(normpdf(G_x./h)./(h^d));
e_1 = zeros(1,d+1);
e_1(1) = 1;

if cond(X_mat' * W * X_mat) > 1e3
    mu = e_1 / (X_mat' * W * X_mat + eye(d+1).*n^(-3));
else
    mu = e_1 / (X_mat' * W * X_mat);
end

mu = mu * X_mat' * W * Z; 

end