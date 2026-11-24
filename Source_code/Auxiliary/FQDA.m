function Y_est = FQDA(t,X,Y,X_new)
% Functional QDA; X_new has one or many new curves in rows.
% Training PCA, class PCA and means are computed once per batch.
[~,delta_t,Y,Cls] = fml_validate_classifier_inputs(t,X,Y,X_new);
n = size(X,1);
[~,phi] = FPCA(t,X,95);
d = size(phi,1);
T = zeros(size(X_new,1),numel(Cls));
for k = 1:numel(Cls)
    idx_k = Y == Cls(k);
    n_k = nnz(idx_k);
    if n_k < 2
        error('FML:InsufficientClassSamples','FQDA requires at least two observations per class.');
    end
    X_k = X(idx_k,:);
    [~,phi_k,lambda_k] = FPCA(t,X_k,99);
    d_k = min(d,size(phi_k,1));
    lambda_k = lambda_k(1:d_k)';
    if any(~isfinite(lambda_k) | lambda_k <= 0)
        error('FML:InvalidEigenvalues','FQDA requires positive finite retained eigenvalues.');
    end
    mu_k = mean(X_k,1);
    xi_k = (X_new-mu_k)*phi_k(1:d_k,:)'*delta_t;
    T(:,k) = sum(xi_k.^2./lambda_k + log(lambda_k),2) - 2*log(n_k/n);
end
if any(~isfinite(T(:)))
    error('FML:NonfiniteScore','FQDA scores overflowed; inspect data and retained eigenvalues.');
end
[~,idx] = min(T,[],2);
Y_est = Cls(idx);
end
