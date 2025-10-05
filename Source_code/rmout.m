function [X,G,ind_rm] = rmout(t,X,G,K,n_min)
% Remove components whose sizte is less than or equal to n_min.
% Input:
% t: 1*p time interval;
% X: n*p data matrix, each column contains function values of an individual;
% G: 1*n group labels;
% K: number of nearest neibourghs;
% n_min: the minimum size of components to be left.
% Output:
% X: n_left*p data matrix, each column contains function values of an individual;
% G: 1*n_left group labels;
% ind_rm: removed indices.

% Author: Ruoxu Tan; date: 2022/Oct/29; Matlab version: R2020a.

n = size(X,1);
D = zeros(n);

for i = 1:n
    for j = i+1:n
        D(i,j) = sqrt(trapz(t,(X(i,:)-X(j,:)).^2));
        D(j,i) = D(i,j);
    end
    
    [~,ind] = sort(D(i,:));
    D(i,ind(K+2:end)) = Inf;
end

for i = 1:n
    for j = i+1:n
        D(i,j) = max([D(i,j),D(j,i)]);
        D(j,i) = D(i,j);
    end
end

% Finding connected components  
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

n_G = unique(G_com);
L_n = zeros(1,length(n_G));

for i = 1:length(n_G)
    L_n(i) = sum(G_com==n_G(i));
end

n_G_rm = n_G(L_n <= n_min); 
ind_rm = [];
for k = 1:length(n_G_rm)
    ind_rm = [ind_rm find(G_com==n_G_rm(k))];
end

X(ind_rm,:) = [];
G(ind_rm) = [];

end

