function [Z_d,G_FSML] = FSML(t,X,Y,xi,varargin)
% Functional supervised manifold learning
% Input:
% t: 1*p time interval;
% X: n*p data matrix, each row contains function values of an individual;
% Y: n*1 classes;
% xi: a tuning parameter controling the minimum distance between classes;
% delta: a tuning parameter in (0,1] to control the number of edges of the adjacency matrix, 
% larger delta leads to more edges;
% K_pca: number of nearest neighbours used in local PCA;
% d: intrinsic dimension;
% opt_FPTU: the option for FPTU, =1 means rescale; otherwise not rescale;
% Output:
% Z_d: n*d low-dimensional outcomes;
% G_FSML: n*n proximity matrix.

% Author: Ruoxu Tan; date: 2025/May; Matlab version: R2024b.
p = inputParser;
addParameter(p,'delta',0.6);
addParameter(p,'K_pca',15);
addParameter(p,'opt_FPTU',0);
addParameter(p,'d',2);
parse(p,varargin{:})

delta = p.Results.delta;
K_pca = p.Results.K_pca;
d = p.Results.d;
opt_FPTU = p.Results.opt_FPTU;

if iscolumn(t)
    t = t';
end

n = size(X,1);

G_adj = Adj_Graph(t,X,delta);

[ D,~ ] = FPTU_adj_input( t,X,G_adj,K_pca,d,opt_FPTU ); 

G_FSML = D;

for i = 1:n
    for j = i+1:n
        if Y(i) ~= Y(j)
            G_FSML(i,j) = G_FSML(i,j) + xi./(G_FSML(i,j)+sqrt(xi));
            G_FSML(j,i) = G_FSML(i,j);
        end
    end
end
warning('off','stats:mdscale:IterOrEvalLimit');
Z_d = mdscale(G_FSML,d,"Criterion","metricstress");
    
end

