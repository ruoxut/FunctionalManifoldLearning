function [G_adj] = Adj_Graph(t,X,delta)
% Adjacency matrix based on Dimeglio et al. (2014)'s algorithm.
% Input:
% t: 1*p time interval;
% X: n*p data matrix, each row contains function values of an individual;
% delta: a tuning parameter in (0,1] to control the number of edges, larger
% delta leads to more edges.
% Output:
% G_adj: n*n adjacency matrix, (i,j)=1 (0) means i and j are adjecent (not adjecent).

% Author: Ruoxu Tan; date: 2025/May; Matlab version: R2024b.
[~,delta_t] = fml_validate_grid(t,X);
validateattributes(delta,{'numeric'},{'real','finite','scalar','>',0,'<=',1},mfilename,'delta');
n = size(X,1);
if n == 1
    G_adj = 0;
    return
end

G_ini = zeros(n,n);
G_rank = zeros(n,n);
for i = 1:n
    for j = i+1:n
        G_ini(i,j) = sqrt(sum((X(i,:)-X(j,:)).^2.*delta_t)); 
        G_ini(j,i) = G_ini(i,j);
    end
end
for i = 1:n
    [~,ind] = sort(G_ini(i,:));
    G_rank(i,:) = [i,ind(ind ~= i)];
end

if any(G_ini(triu(true(n),1)) == 0)
    % Adjacency-matrix construction discards zero-weight duplicate edges.
    [edge_i,edge_j] = find(triu(true(n),1));
    edge_weights = G_ini(sub2ind([n,n],edge_i,edge_j));
    G_ini_g = graph(edge_i,edge_j,edge_weights,n);
else
    G_ini_g = graph(G_ini); % Preserve the original MST ordering in regular cases.
end
G_MST_g = minspantree(G_ini_g);
G_MST = adjacency(G_MST_g);
G_MST = full(G_MST);

G_adj = G_MST;
r = zeros(n,1);

for i = 1:n
    xi_i = zeros(n,1);
    for j = 1:n
        if G_MST(i,j) ~= 0
            xi_i(j,1) = G_ini(i,j);
        end
    end
    r(i,1) = delta * max(xi_i);      
end

num_test = 100;
lambda = linspace(0,1,num_test)';

for i = 1:n
    parfor j = i+1:n
        if G_MST(i,j) == 0
            Z_lambda = lambda.*X(i,:)+(1-lambda).*X(j,:);
            k = 1;
            while k <= num_test
                if lambda(k)*G_ini(i,j) <= r(j,1) || (1-lambda(k))*G_ini(i,j) <= r(i,1)
                    k = k + 1;
                else     
                    s = 2;
                    while s <= n
                        if sqrt(sum((Z_lambda(k,:)-X(G_rank(i,s),:)).^2.*delta_t)) <= r(G_rank(i,s),1)
                            break
                        else
                            s = s + 1;
                        end
                    end
                    if s <= n
                        k = k + 1; 
                    else
                        break
                    end
                end
            end

            if k == num_test+1
                G_adj(i,j) = 1;
            end
        end
    end
    G_adj((i+1):n,i) = G_adj(i,(i+1):n);
end

end

