function [ X_d,G,n_com ] = graph_clustering( t,X,D,g,d,n_start )
% Graph-based clustering using the proximity graph D.
% Input:
% t: 1*p time interval;
% X: n*p data matrix, each column contains function values of an individual;
% D: n*n proximity graph;
% g: number of clusters;
% d: intrinsic dimension;
% n_start: Number of replicates used in k-means clustering, =20 as default; use larger value if the results are not stable.
% Output:
% X_d: 1*n_com cell, each cell includes d*n_com(i) MDS low-d coordinates;
% n_com: number of components;
% G: 1*n group labels.

% Author: Ruoxu Tan; date: 2023/Feb/8; Matlab version: R2020a.

if nargin < 6
    n_start = 20;
end
[t,~] = fml_validate_grid(t,X);
n = size(X,1);
validateattributes(g,{'numeric'},{'real','finite','scalar','integer','positive','<=',n},mfilename,'g');
validateattributes(d,{'numeric'},{'real','finite','scalar','integer','positive','<=',size(X,2)},mfilename,'d');
validateattributes(n_start,{'numeric'},{'real','finite','scalar','integer','positive'},mfilename,'n_start');
if ~isnumeric(D) || ~isreal(D) || ~isequal(size(D),[n,n]) || ...
        any(isnan(D(:))) || any(D(:) < 0)
    error('FML:InvalidDistances','D must be a nonnegative real n-by-n matrix without NaNs.');
end
finite = isfinite(D);
D_transpose = D';
tol = 1e-10*max([1;D(finite)]);
if ~isequal(finite,finite') || any(abs(D(finite)-D_transpose(finite)) > tol) || ...
        any(~isfinite(diag(D))) || any(abs(diag(D)) > tol)
    error('FML:InvalidDistances','D must be symmetric with a zero diagonal; Inf denotes disconnected pairs.');
end
D = (D+D')/2;
D(1:n+1:end) = 0;

% Finding connected components
finite(1:n+1:end) = false;
G_com = conncomp(graph(finite)); % Avoid recursion limits on large components.

G = G_com;
n_com = max(G_com);
X_d = cell(1,n_com);
L = zeros(1,n_com); 

% Perform MDS on each connected component 
for i = 1:n_com
    D_sq = D(G_com==i,G_com==i).^2;
    n_i = size(D_sq,1);
    if any(~isfinite(D_sq(:)))
        error('FML:InvalidDistances','Distances within component %d must be finite and safely squareable.',i);
    end
    X_d{1,i} = zeros(d,n_i);
    L(1,i) = n_i;
    if n_i == 1
        continue
    end
    % H*D_sq*H by centering, avoiding two dense matrix products.
    G_cen = -0.5*(D_sq-mean(D_sq,1)-mean(D_sq,2)+mean(D_sq(:)));
    G_cen = (G_cen+G_cen')/2;
    n_coord = min(d,n_i-1);
    if n_coord >= n_i-1
        [phi,lambda] = eig(G_cen,'vector');
    else
        [phi,lambda,flag] = eigs(G_cen,n_coord,'largestreal');
        lambda = diag(lambda);
        if flag ~= 0
            [phi,lambda] = eig(G_cen,'vector');
        end
    end
    [lambda,order] = sort(lambda,'descend');
    lambda = lambda(1:n_coord);
    phi = phi(:,order(1:n_coord));
    eig_tol = 100*eps(max(1,max(abs(lambda))))*n_i;
    if any(lambda < -eig_tol)
        warning('FML:NonEuclideanMDS', ...
            'Component %d has fewer than %d nonnegative MDS dimensions; negative dimensions are omitted.',i,n_coord);
    end
    lambda = max(lambda,0);
    X_d{1,i}(1:n_coord,:) = sqrt(lambda).*phi';
end

%% Clustering
% Combine the closest two clusters successively
if n_com > g
    n_G = unique(G);

    while length(n_G) > g
        ind_i = repelem(n_G',1,length(n_G));
        ind_j = ind_i';
        D_ave = Inf .* ones(length(n_G));
        for i = 1:length(n_G)
            for j = i+1:length(n_G)
                D_ave(i,j) = 0;
                X_i = X(G==ind_i(i,j),:);
                X_j = X(G==ind_j(i,j),:);
                for k = 1:size(X_i,1)
                    D_ave(i,j) = D_ave(i,j) + sum(sqrt(trapz(t,(X_i(k,:) - X_j).^2,2)));
                end
                D_ave(i,j) = D_ave(i,j) / (size(X_i,1)*size(X_j,1));
            end
        end        
       [i_min,j_min] = find(D_ave == min(D_ave(:)),1); % Deterministic linkage ties.
       G(G==ind_j(i_min,j_min)) = ind_i(i_min,j_min);
       n_G = unique(G);
    end
    
    for i = 1:length(n_G)
        G(G==n_G(i)) = i;
    end
    
end

% Perform k-n_com+1 means clustering on the largest component
if n_com < g
    [~,ind] = max(L);
    n_clusters = g-n_com+1;
    if L(ind) < n_clusters || size(unique(X_d{1,ind}','rows'),1) < n_clusters
        error('FML:InsufficientEmbedding','The largest component has too few distinct embedded points for the requested clusters.');
    end
    if ind ~= n_com % Change the group label to be performed k-n_com+1 means to the largest one
        G_tem = G;
        G(1,G_tem==ind) = n_com;
        G(1,G_tem==n_com) = ind;        
    end
   
    G(1,G==n_com) = kmeans(X_d{1,ind}',g-n_com+1,'Replicates',n_start) + n_com - 1; 
end
    
end

