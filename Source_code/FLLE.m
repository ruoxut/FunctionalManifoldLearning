function mu = FLLE(t,X,Z,x,K_pca,h)
% Functional local linear estimator on the estimated tangent space.
% The six-argument interface is unchanged. Repeated predictions may reuse
% FLLE_prepare(t,X,x,K_pca,d), then call FLLE_predict(geometry,Z,h).

if ~isnumeric(Z) || ~isreal(Z) || ~ismatrix(Z) || ...
        size(Z,1) ~= size(X,1) || isempty(Z) || any(~isfinite(Z(:)))
    error('FLLE:InvalidResponse','Z must be a finite n-by-d numeric matrix.')
end
geometry = FLLE_prepare(t,X,x,K_pca,size(Z,2));
mu = FLLE_predict(geometry,Z,h);
end
