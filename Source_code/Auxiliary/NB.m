function Y_est = NB(t,X,Y,X_new)
% Nonparametric Bayesian classifier of Dai et al. (2017).
% X_new has one or many curves in rows. Gaussian KDEs use the same default
% bandwidth chosen by ksdensity, evaluated by log-sum-exp to avoid log(0).
[~,delta_t,Y,Cls] = fml_validate_classifier_inputs(t,X,Y,X_new);
n = size(X,1);
[~,phi] = FPCA(t,X,95);
train_scores = X*phi'*delta_t;
new_scores = X_new*phi'*delta_t;
Q = zeros(size(X_new,1),numel(Cls));
for k = 1:numel(Cls)
    idx_k = Y == Cls(k);
    n_k = nnz(idx_k);
    Q(:,k) = log(n_k/n);
    for j = 1:size(phi,1)
        samples = train_scores(idx_k,j);
        % Same normal kernel/unbounded support/default fitted bandwidth.
        [~,~,bandwidth] = ksdensity(samples,samples(1),'Kernel','normal');
        if ~isfinite(bandwidth) || bandwidth <= 0
            error('FML:InvalidBandwidth','NB KDE returned a nonpositive or nonfinite bandwidth.');
        end
        u = (new_scores(:,j)-samples')/bandwidth;
        exponent = -0.5*u.^2;
        largest = max(exponent,[],2);
        log_density = largest + log(sum(exp(exponent-largest),2)) ...
            - log(n_k) - log(bandwidth) - 0.5*log(2*pi);
        Q(:,k) = Q(:,k) + log_density;
    end
end
if any(isnan(Q(:))) || any(all(~isfinite(Q),2))
    error('FML:NonfiniteScore','NB scores are nonfinite; inspect score scales and KDE bandwidths.');
end
[~,idx] = max(Q,[],2);
Y_est = Cls(idx);
end
