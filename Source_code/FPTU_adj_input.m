function [D,Path] = FPTU_adj_input(t,X,G_adj,K_pca,d,opt)
% Functional parallel transport unfolding using a supplied adjacency matrix.
% X contains one curve per row; opt=1 rescales projected edges.
[~,delta_t] = fml_validate_grid(t,X);
n = size(X,1);
if ~(isnumeric(G_adj) || islogical(G_adj)) || ~isreal(G_adj) || ...
        ~isequal(size(G_adj),[n,n]) || any(~isfinite(G_adj(:))) || ...
        any(G_adj(:) ~= 0 & G_adj(:) ~= 1) || ~isequal(G_adj,G_adj')
    error('FML:InvalidAdjacency','G_adj must be a symmetric n-by-n binary adjacency matrix.');
end
G = Inf(n);
G(1:n+1:end) = 0;
for i = 1:n
    for j = i+1:n
        if G_adj(i,j) == 1
            G(i,j) = sqrt(sum((X(i,:)-X(j,:)).^2)*delta_t);
            G(j,i) = G(i,j);
        end
    end
end
if nargout > 1
    [D,Path] = fml_ptu_engine(X,G,K_pca,d,opt,delta_t);
else
    D = fml_ptu_engine(X,G,K_pca,d,opt,delta_t);
end
end

