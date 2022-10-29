function [ G_s,Path ] = FIsomap( t,X,K )
% Functional isometric map.
% Input:
% t: p*1 time interval;
% X: p*n data matrix, each column contains function values of an individual;
% K: number of nearest neighbours;
% Output:
% G_s: n*n proximity graph.
% Path: n*n cell, (i,j)-cell contains the shortest path indices from i to j.

% Author: Ruoxu Tan; date: 2022/Oct/29; Matlab version: R2020a.

if isrow(t)
    t = t';
end
    
if length(t) ~= size(X,1)
    error('Dimensions of the input functional data do not match.')
end

n = size(X,2);

%% Proximity graph
G = zeros(n);
for i = 1:n
    for j = i+1:n
        G(i,j) = sqrt(trapz(t,(X(:,i)-X(:,j)).^2));
        G(j,i) = G(i,j);
    end
    
    [~,ind] = sort(G(i,:));
    G(i,ind(K+2:end)) = Inf;
end

for i = 1:n
    for j = i+1:n
        G(i,j) = max([G(i,j),G(j,i)]);
        G(j,i) = G(i,j);
    end
end

% Shortest path graph and corresponding paths
G_s = zeros(n);
Path = cell(n);
for i = 1:n
    Dis = zeros(1,n);
    Pre = zeros(1,n);
    Dis(:) = Inf;
    Dis(i) = 0;
    Q = 1:n;
    while ~isempty(Q)
        [~,j] = min(Dis(Q));
        u = Q(j);
        Q(j) = [];
        for k = 1:n
            if k ~= u 
                alt = Dis(u)+G(u,k);
                if alt < Dis(k)
                Dis(k) = alt;
                Pre(k) = u;
                end
            end
        end
    end
    G_s(i,:) = Dis;
    
    for j = 1:n
        if j == i 
            Path{i,j} = i;
        else 
            u = j;
            while u ~= 0            
            Path{i,j} = [u, Path{i,j}];
            u = Pre(u);
            end
        end
    end
end

end

