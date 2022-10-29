function [ X_d,G,n_com ] = graph_clustering( t,X,D,g,d,n_start )
% Graph-based clustering using the proximity graph D.
% Input:
% t: p*1 time interval;
% X: p*n data matrix, each column contains function values of an individual;
% D: n*n proximity graph;
% g: number of clusters;
% d: intrinsic dimension;
% n_start: Number of replicates used in k-means clustering, =20 as default; use larger value if the results are not stable.
% Output:
% X_d: 1*n_com cell, each cell includes d*n_com(i) MDS low-d coordinates;
% n_com: number of components;
% G: 1*n group labels.

% Author: Ruoxu Tan; date: 2022/Oct/29; Matlab version: R2020a.

if nargin < 6
    n_start = 20;
end

% Finding connected components
n = size(D,2);
G_com = zeros(1,n);

if sum(sum(isinf(D))) == 0
    G_com = G_com + 1;
else
    n_c = 1;
    for i = 1:n        
        if G_com(1,i) == 0
            G_com = DFS(D,G_com,n_c,i);
            n_c = n_c+1;
        end
    end
end

G = G_com;
n_com = max(G_com);
X_d = cell(1,n_com);
L = zeros(1,n_com); 

% Perform MDS on each connected component 
for i = 1:n_com
    D_sq = D(G_com==i,G_com==i).^2;
    n_i = size(D_sq,1);
    G_cen = -0.5 * (eye(n_i)-(1/n_i) * ones(n_i,1) * ones(1,n_i)) * D_sq * (eye(n_i)-(1/n_i) * ones(n_i,1) * ones(1,n_i));
    [phi,lambda,~] = pcacov(G_cen); 
    if n_i > d  
        X_d{1,i} = sqrt(lambda(1:d,1)) .* phi(:,1:d)';
    else
        X_d{1,i} = sqrt(lambda(:,1)) .* phi';
    end
    L(1,i) = size(X_d{1,i},2); % Sample size in ith component
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
                X_i = X(:,G==ind_i(i,j));
                X_j = X(:,G==ind_j(i,j));
                for k = 1:size(X_i,2)
                    D_ave(i,j) = D_ave(i,j) + sum(sqrt(trapz(t,(X_i(:,k) - X_j).^2)));
                end
                D_ave(i,j) = D_ave(i,j) / (size(X_i,2)*size(X_j,2));
            end
        end        
       [i_min,j_min] = find(D_ave == min(min(D_ave)));
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
    if ind ~= n_com % Change the group label to be performed k-n_com+1 means to the largest one
        G_tem = G;
        G(1,G_tem==ind) = n_com;
        G(1,G_tem==n_com) = ind;        
    end
   
    G(1,G==n_com) = kmeans(X_d{1,ind}',g-n_com+1,'Replicates',n_start) + n_com - 1; 
end
    
end

