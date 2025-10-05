function [ D,Path ] = FPTU_adj_input( t,X,G_adj,K_pca,d,opt )
% Functional parallel transport unfolding.
% Input:
% t: 1*p time interval;
% X: n*p data matrix, each row contains function values of an individual; 
% G_adj: n*n adjacency matrix, (i,j)=1 (0) means i and j are adjecent (not adjecent).
% K_pca: number of nearest neighbours used in local PCA;
% d: intrinsic dimension;
% opt: =1 means rescale; otherwise not rescale;
% Output:
% D: n*n proximity graph.
% Path: n*n cell, (i,j)-cell contains the shortest path indices from i to j.

% Author: Ruoxu Tan; date: 2023/Jul/1; Matlab version: R2020a.

if K_pca < d
    error('K_pca is smaller than the intrinsic dimension.')
end

if iscolumn(t)
    t = t';
end
    
if length(t) ~= size(X,2)
    error('Dimensions of the input functional data do not match.')
end

n = size(X,1);
delta_t = mean(diff(t));

%% Proximity graph
G = ones(n).*Inf;
for i = 1:n
    for j = i+1:n
        if G_adj(i,j) == 1
            G(i,j) = sqrt(sum((X(i,:)-X(j,:)).^2.*delta_t));
            G(j,i) = G(i,j);
        end
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
 
%% Tangent spaces and parallel transport    
TM = cell(1,n);% Tangent spaces
for i = 1:n
    [~,ind] = sort(G_s(i,:));
    mu_i = mean(X(ind(2:K_pca+1),:),1);
    X_i_cen = X(ind(2:K_pca+1),:)-mu_i;  
    [~,~,phi_i] = svds(X_i_cen,d);     
    norm_phi_i = sqrt(sum(phi_i.^2.*delta_t,1));      
    phi_i = phi_i ./ norm_phi_i; 
    TM{1,i} = phi_i';
end

% Discrete parallel transport
R = cell(n);
for i = 1:n
    for j = 1:n
        if j ~= i
            Phi_ij = TM{1,i} * TM{1,j}' * mean(diff(t));
            [U,~,V] = svd(Phi_ij); 
            R{j,i} = V * U';
        end
    end
end

% Proximity graph based on parallel transport
D = Inf * ones(n);
for i = 1:n
    D(i,i) = 0;
    for j = 1:n
        if length(Path{i,j}) > 1
            v_i = zeros(d,length(Path{i,j})-1);
            for k = 1:size(v_i,2)    
                xi_i = sum((X(Path{i,j}(k),:) - X(Path{i,j}(k+1),:)).*TM{1,Path{i,j}(k+1)}.*delta_t,2);
                if opt == 1
                    xi_i = xi_i ./ norm(xi_i) .* sqrt(sum((X(Path{i,j}(k),:) - X(Path{i,j}(k+1),:)).^2.*delta_t,2));% Rescaling
                end
                
                s = k+1;
                while Path{i,j}(s) ~= j
                    xi_i = R{Path{i,j}(s+1),Path{i,j}(s)} * xi_i; %T_s M to T_s+1 M                  
                    s = s+1;
                end
                v_i(:,k) = xi_i;
            end
            v = sum(v_i,2);
            D(i,j) = norm(v);
        end
    end
end
D = (D+D')./2;

end

