function G = DFS(D,G,g,j)
% The depth-first search algorithm to find the connected component
% including the jth individual.
% Input:
% D: n*n proximity graph;
% G: 1*n initial vector of group labels;
% g: the group label;
% j: the jth individual with the group label g;
% Output: 
% G: 1*n updated vector of group labels including the members of group g.

% Author: Ruoxu Tan; date: 2022/Oct/29; Matlab version: R2020a.

G(1,j) = g;
for k = 1:size(D,2)
    if D(j,k)<Inf && G(1,k) == 0
       G = DFS(D,G,g,k);
    end
end    
end

