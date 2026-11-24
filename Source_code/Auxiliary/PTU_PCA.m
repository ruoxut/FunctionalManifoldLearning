function D = PTU_PCA(t,X,K,K_pca,d,opt)
% PTU on global FPCA scores, retaining intrinsic tangent dimension d.
% The global score dimension is max(d95,d), as specified in the paper.
[~,delta_t] = fml_validate_grid(t,X);
n = size(X,1);
validateattributes(K,{'numeric'},{'real','finite','scalar','integer','positive','<',n},mfilename,'K');
validateattributes(d,{'numeric'},{'real','finite','scalar','integer','positive','<=',min(size(X,2),n-1)},mfilename,'d');

X_cen = X-mean(X,1);
Cov = X_cen'*X_cen/n;
[phi,latent,expd] = pcacov(Cov);
if ~isfinite(sum(latent)) || sum(latent) <= 0
    error('FML:ZeroVariance','PTU_PCA requires positive variation among curves.');
end
d95 = find(cumsum(expd) >= 95,1);
if isempty(d95)
    error('FML:InvalidPCA','Could not determine the FPCA variance threshold.');
end
n_pc = max(d95,d);
rank_tol = max(size(X_cen))*eps(max(latent));
if latent(n_pc) <= rank_tol
    error('FML:GlobalRank','The global covariance has insufficient rank for d=%d.',d);
end
phi = phi(:,1:n_pc);
phi = phi./sqrt(sum(phi.^2,1)*delta_t);
X_d = X_cen*phi*delta_t;

G = zeros(n);
for i = 1:n
    for j = i+1:n
        G(i,j) = norm(X_d(i,:)-X_d(j,:));
        G(j,i) = G(i,j);
    end
end
for i = 1:n
    [~,ind] = sort(G(i,:));
    ind = ind(ind ~= i);
    G(i,ind(K+1:end)) = Inf;
end
G = max(G,G');
D = fml_ptu_engine(X_d,G,K_pca,d,opt,1);
end

