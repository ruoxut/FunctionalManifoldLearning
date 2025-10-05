function [Y_est] = FQDA(t,X,Y,X_new)
% Functional quadratic discriminant analysis by Delaigle and Hall (2013)
% Input:
% t: 1*p time interval;
% X: n*p data matrix, each row contains function values of an individual;
% Y: n*1 classes;
% X_new: 1*p new functional observation to be classified;
% Output:
% Y_est: estimated class of X_new.

n = size(X,1);
Cls = unique(Y);
[~,phi,~] = FPCA(t,X,95); 
d = size(phi,1);
delta_t = mean(diff(t));
T = zeros(size(Cls));

for k = 1:length(Cls)
    idx_k = Y==Cls(k);
    n_k = length(idx_k);
    X_k = X(idx_k,:);
    [~,phi_k,lambda_k] = FPCA(t,X_k,99);
    d_k = d;
    if size(phi_k,1) < d
        d_k = size(phi_k,1);
    end
    mu_k = mean(X_k,1);

    xi_k = sum((X_new-mu_k).*phi_k(1:d_k,:).*delta_t,2);
    T(k) = sum(xi_k.^2./lambda_k(1:d_k) + log(lambda_k(1:d_k))) - 2*log(n_k/n); 
end

[~,idx] = min(T);

Y_est = Cls(idx);
 
end