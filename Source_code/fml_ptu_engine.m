function [D,Path] = fml_ptu_engine(X,G,K_pca,d,opt,metric_weight)
% Shared exact PTU calculation. G retains the caller's distance convention.
% The original Dijkstra traversal and strict-improvement tie rule are retained.
n = size(X,1);
validateattributes(d,{'numeric'},{'real','finite','scalar','integer','positive','<=',size(X,2)},mfilename,'d');
validateattributes(K_pca,{'numeric'},{'real','finite','scalar','integer','>',d,'<',n},mfilename,'K_pca');
validateattributes(opt,{'numeric','logical'},{'real','finite','scalar'},mfilename,'opt');

G_s = Inf(n);
Pre = zeros(n);
visit_order = zeros(n);
for source = 1:n
    Dis = Inf(1,n);
    Dis(source) = 0;
    Q = 1:n;
    count = 0;
    while ~isempty(Q)
        [best,j] = min(Dis(Q));
        if isinf(best)
            break
        end
        u = Q(j);
        Q(j) = [];
        count = count+1;
        visit_order(source,count) = u;
        for k = 1:n
            if k ~= u
                alt = Dis(u)+G(u,k);
                if alt < Dis(k)
                    Dis(k) = alt;
                    Pre(source,k) = u;
                end
            end
        end
    end
    G_s(source,:) = Dis;
end

TM = cell(1,n);
for i = 1:n
    [~,ind] = sort(G_s(i,:));
    ind = ind(ind ~= i & isfinite(G_s(i,ind)));
    if isempty(ind)
        continue % An isolated point has no edges to unfold.
    end
    if numel(ind) < K_pca
        error('FML:InsufficientComponent', ...
            'Point %d has only %d other points in its component, fewer than K_pca=%d. Remove small components or choose a supported K_pca.', ...
            i,numel(ind),K_pca);
    end
    local = X(ind(1:K_pca),:);
    local = local-mean(local,1);
    [~,S,phi,flag] = svds(local,d);
    if flag ~= 0
        [~,S,phi] = svd(local,'econ');
        S = S(1:d,1:d);
        phi = phi(:,1:d);
    end
    singular_values = diag(S);
    rank_tol = max(size(local))*eps(max(singular_values));
    if any(~isfinite(singular_values)) || singular_values(end) <= rank_tol
        error('FML:LocalRank', ...
            'The local covariance at point %d has rank below d=%d. Check duplicates, dimension, and neighborhood size.',i,d);
    end
    phi = phi./sqrt(sum(phi.^2,1)*metric_weight);
    TM{i} = phi';
end

% Only graph-edge transports and projections are ever used by a shortest path.
R = cell(n);
edge_vectors = cell(n);
for u = 1:n
    neighbors = find(isfinite(G(u,:)) & (1:n) ~= u);
    for v = neighbors
        [U,~,V] = svd(TM{u}*TM{v}'*metric_weight);
        R{v,u} = V*U';
        edge = X(u,:)-X(v,:);
        xi = TM{v}*edge'*metric_weight;
        if opt == 1
            edge_norm = sqrt(sum(edge.^2)*metric_weight);
            projected_norm = norm(xi);
            if edge_norm == 0
                xi(:) = 0;
            elseif projected_norm <= 64*eps(edge_norm)
                error('FML:ZeroProjection', ...
                    'The nonzero edge %d to %d has a numerically zero tangent projection; rescaling is undefined.',u,v);
            else
                xi = xi/projected_norm*edge_norm;
            end
        end
        edge_vectors{v,u} = xi;
    end
end

D = Inf(n);
for source = 1:n
    accumulated = zeros(d,n);
    D(source,source) = 0;
    order = visit_order(source,visit_order(source,:) ~= 0);
    for v = order
        u = Pre(source,v);
        if u ~= 0
            % Same ordered product and sum as unfolding every edge separately.
            accumulated(:,v) = R{v,u}*accumulated(:,u)+edge_vectors{v,u};
            D(source,v) = norm(accumulated(:,v));
        end
    end
end
D = (D+D')/2;

if nargout > 1
    Path = cell(n);
    for source = 1:n
        for target = 1:n
            if isfinite(G_s(source,target))
                path = target;
                u = target;
                while u ~= source
                    u = Pre(source,u);
                    path = [u,path]; %#ok<AGROW>
                end
                Path{source,target} = path;
            else
                Path{source,target} = []; % No path exists between components.
            end
        end
    end
end
end
