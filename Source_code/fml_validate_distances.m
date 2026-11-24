function D = fml_validate_distances(D,n)
% Validate a reusable pairwise geodesic/proximity matrix.
validateattributes(D,{'numeric'}, ...
    {'real','finite','nonnegative','size',[n,n],'nonempty'});
tol = 64*eps(max(1,max(D(:))));
if max(abs(D-D'),[],'all') > tol || any(abs(diag(D)) > tol)
    error('FML:InvalidDistances','Distances must be symmetric with a zero diagonal.')
end
end
