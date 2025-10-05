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
if iscolumn(t)
    t = t';
end

if length(t) ~= size(X,2)
    error('Dimensions of the input functional data do not match.')
end

if delta <= 0 || delta > 1 
    error('The tuning parameter must be in (0,1].')
end

delta_t = mean(diff(t));
n = size(X,1);

G_ini = zeros(n,n);
G_rank = zeros(n,n);
for i = 1:n
    for j = i+1:n
        G_ini(i,j) = sqrt(sum((X(i,:)-X(j,:)).^2.*delta_t)); 
        G_ini(j,i) = G_ini(i,j);
    end
end
for i = 1:n
    [~,G_rank(i,:)] = sort(G_ini(i,:));
end

G_ini_g = graph(G_ini);
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

for i = 1:n
    parfor j = i+1:n
        if G_MST(i,j) == 0
            Z_lambda = linspace(0,1,num_test)'.*X(i,:)+(1-linspace(0,1,num_test)').*X(j,:);
            k = 1;
            while k <= num_test
                if k*G_ini(i,j)/num_test <= r(j,1) || (num_test-k)*G_ini(i,j)/num_test <= r(i,1)
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

