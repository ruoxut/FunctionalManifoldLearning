function geometry = FLLE_prepare(t,X,x,K_pca,d)
% Cache distances, tangent basis and design, independently of Z and h.
[~,delta_t] = fml_validate_grid(t,X);
n = size(X,1);
validateattributes(d,{'numeric'},{'real','finite','scalar','integer','positive'});
validateattributes(K_pca,{'numeric'}, ...
    {'real','finite','scalar','integer','>=',d+1,'<=',n});
if d > size(X,2)
    error('FLLE:InvalidDimension','d cannot exceed the number of grid points.')
end
if ~isnumeric(x) || ~isreal(x) || ~isvector(x) || ...
        numel(x) ~= size(X,2) || any(~isfinite(x(:)))
    error('FLLE:InvalidQuery','x must be a finite curve on the input grid.')
end
x = reshape(x,1,[]);
centered = X-x;
distance = sqrt(sum(centered.^2,2).*delta_t);
if any(~isfinite(distance))
    error('FLLE:NumericalFailure','Curve distances overflowed.')
end
[~,ind] = sort(distance);
local = X(ind(1:K_pca),:);
local = local-mean(local,1);
[~,S,phi] = svd(local,'econ');
s = diag(S);
rank_tol = max(size(local))*eps(max(s));
if numel(s) < d || s(d) <= rank_tol
    error('FLLE:InsufficientRank', ...
        'The local neighborhood has fewer than d independent tangent directions.')
end
% Euclidean unit singular vectors become L2 unit vectors on this grid.
design = [ones(n,1), (centered*phi(:,1:d)).*sqrt(delta_t)];
geometry = struct('distance',distance,'design',design,'n',n,'d',d);
end
