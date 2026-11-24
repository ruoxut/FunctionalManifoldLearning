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
% Geodesic: optional precomputed n-by-n geodesic matrix in the input row order;
% Output:
% Z_d: n*d low-dimensional outcomes;
% G_FSML: n*n proximity matrix.

% Author: Ruoxu Tan; date: 2025/May; Matlab version: R2024b.
p = inputParser;
addParameter(p,'delta',0.6);
addParameter(p,'K_pca',15);
addParameter(p,'opt_FPTU',0);
addParameter(p,'d',2);
addParameter(p,'Geodesic',[]);
parse(p,varargin{:})

delta = p.Results.delta;
K_pca = p.Results.K_pca;
d = p.Results.d;
opt_FPTU = p.Results.opt_FPTU;
[t,Y] = fml_validate_supervised(t,X,Y,K_pca,d);
validateattributes(xi,{'numeric'},{'real','finite','scalar','nonnegative'});
n = size(X,1);
D = p.Results.Geodesic;
if isempty(D)
    G_adj = Adj_Graph(t,X,delta);
    D = FPTU_adj_input(t,X,G_adj,K_pca,d,opt_FPTU);
end
D = fml_validate_distances(D,n);
G_FSML = D;
if xi > 0
    different_class = Y ~= Y';
    G_FSML(different_class) = D(different_class) ...
        + xi./(D(different_class)+sqrt(xi));
end
Z_d = fml_mds(G_FSML,d);
    
end

