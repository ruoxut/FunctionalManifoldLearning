function [ D ] = PTU_PCA( t,X,K,K_pca,d,opt )
% Parallel transport unfolding applied on functional principal scores.
% Input:
% t: p*1 time interval;
% X: p*n data matrix, each column contains function values of an individual;
% K: number of nearest neighbours;
% K_pca: number of nearest neighbours used in local PCA;
% d: intrinsic dimension;
% opt: 1 rescale; otherwise not rescale.
% Output:
% D: n*n proximity graph.

% Author: Ruoxu Tan; date: 2022/Oct/29; Matlab version: R2020a.

if K_pca < d
    error('K_pca is smaller than the intrinsic dimension.')
end

if isrow(t)
    t = t';
end
    
if length(t) ~= size(X,1)
    error('Dimensions of the input functional data do not match.')
end

n = size(X,2);
 
%% FPCA
mu_t = mean(X,2);
X_cen = X - mu_t;
Cov_t = X_cen * X_cen' / n;
[phi,~,expd] = pcacov(Cov_t);
norm_phi = sqrt(trapz(t,phi.^2,1));       
phi = phi ./ norm_phi;

d_pca = 1;
s = expd(1);
while s<95
    d_pca = d_pca+1;
    s = s+expd(d_pca);
end
d_pca = max([d_pca d]);

X_d = zeros(d_pca,n); 
for i = 1:n
    xi = trapz(t,X_cen(:,i).*phi(:,1:d_pca),1);
    X_d(:,i) = xi'; 
end

%% Proximity graph
G = zeros(n);
for i = 1:n
    for j = i+1:n
        G(i,j) = norm(X_d(:,i)-X_d(:,j));
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
 
%% Tangent spaces and parallel transport  
Cov = cell(1,n);
mu = cell(1,n);
TM = cell(1,n); % Tangent spaces
for i = 1:n
    [~,ind] = sort(G_s(i,:));
    mu{1,i} = mean(X_d(:,ind(2:K_pca+1)),2);
    X_i_cen = X_d(:,ind(2:K_pca+1))-mu{1,i}; 
    Cov{1,i} = X_i_cen * X_i_cen' ./ K_pca; 
    [phi_i,~,~] = pcacov(Cov{1,i});
    TM{1,i} = phi_i(:,1:d);
end

% Discrete parallel transport
R = cell(n);
for i = 1:n
    for j = 1:n
        if j ~= i
            Phi_ij = TM{1,i}' * TM{1,j};
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
                xi_i = TM{1,Path{i,j}(k+1)}' * (X_d(:,Path{i,j}(k)) - X_d(:,Path{i,j}(k+1)));
                if opt == 1
                xi_i = xi_i ./ norm(xi_i) .* norm(X_d(:,Path{i,j}(k)) - X_d(:,Path{i,j}(k+1)));% Rescaling
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

