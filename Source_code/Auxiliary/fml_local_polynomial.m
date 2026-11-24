function [beta,intercept_weights,logdet] = fml_local_polynomial(X,Y,x,h,degree)
% Stable Gaussian weighted polynomial fit in centered/scaled coordinates.
% beta(k+1) multiplies (X-x)^k; logdet is the determinant of the raw
% sum_j K_h(X_j-x) v_j*v_j' Gram matrix, computed without cancellation.
dx = X-x;
scale = max(abs(dx));
if scale == 0
    error('FML:SingularSmoother','Local polynomial covariates have zero spread.');
end
u = dx/h;
logw = -0.5*u.^2 - log(h) - 0.5*log(2*pi);
largest = max(logw);
if ~isfinite(largest)
    error('FML:NoKernelSupport','Gaussian weights have no representable support.');
end
rootw = exp(0.5*(logw-largest));
z = dx/scale;
B = z.^(0:degree);
A = B.*rootw;
[Q,R] = qr(A,0);
diagonal = abs(diag(R));
if numel(diagonal) < degree+1 || any(diagonal <= max(size(A))*eps(norm(R,inf)))
    error('FML:SingularSmoother','Local polynomial design is numerically rank deficient.');
end
coeff = R\(Q'*(rootw.*Y));
beta = coeff./(scale.^(0:degree))';
if nargout > 1
    e1 = zeros(1,degree+1);
    e1(1) = 1;
    intercept_weights = ((e1/R)*Q').*rootw';
end
if nargout > 2
    logdet = 2*sum(log(diagonal)) + (degree+1)*largest ...
        + degree*(degree+1)*log(scale);
end
end
