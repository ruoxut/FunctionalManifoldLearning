function [D,Path] = FPTU_adj_knn(t,X,K,K_pca,d,opt)
% Functional parallel transport unfolding on a mutual K-nearest-neighbor graph.
% X contains one curve per row; opt=1 rescales projected edges.
[t,delta_t] = fml_validate_grid(t,X);
n = size(X,1);
validateattributes(K,{'numeric'},{'real','finite','scalar','integer','positive','<',n},mfilename,'K');

G = zeros(n);
for i = 1:n
    for j = i+1:n
        G(i,j) = sqrt(trapz(t,(X(i,:)-X(j,:)).^2));
        G(j,i) = G(i,j);
    end
end
for i = 1:n
    [~,ind] = sort(G(i,:));
    ind = ind(ind ~= i); % Self is not necessarily the first zero for duplicates.
    G(i,ind(K+1:end)) = Inf;
end
G = max(G,G'); % The paper uses mutual, rather than union, K-NN.

if nargout > 1
    [D,Path] = fml_ptu_engine(X,G,K_pca,d,opt,delta_t);
else
    D = fml_ptu_engine(X,G,K_pca,d,opt,delta_t);
end
end

