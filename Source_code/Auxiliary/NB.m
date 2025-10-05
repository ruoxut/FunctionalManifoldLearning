function [Y_est] = NB(t,X,Y,X_new)
% The nonparametric Bayesian classifier by Dai et al. (2017).
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
Q = zeros(size(Cls));

parfor k = 1:length(Cls)
    idx_k = Y==Cls(k);
    n_k = length(idx_k);
    Q(k) = log(n_k/n);
    for j = 1:d
        xi_kj = sum(X(idx_k,:).*phi(j,:).*delta_t,2);
        X_new_j = sum(X_new.*phi(j,:).*delta_t,2);
        f_kj = ksdensity(xi_kj,X_new_j);
        Q(k) = Q(k) + log(f_kj);
    end
end

[~,idx] = max(Q);
Y_est = Cls(idx);

end