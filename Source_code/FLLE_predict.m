function mu = FLLE_predict(geometry,Z,h)
% Evaluate the same local linear/ridged objective using cached geometry.
validateattributes(h,{'numeric'},{'real','finite','scalar','positive'});
if ~isnumeric(Z) || ~isreal(Z) || ...
        ~isequal(size(Z),[geometry.n,geometry.d]) || any(~isfinite(Z(:)))
    error('FLLE:InvalidResponse','Z must be a finite n-by-d numeric matrix.')
end
log_w = -0.5*(geometry.distance./h).^2 ...
    - geometry.d*log(h)-0.5*log(2*pi);
log_scale = max(log_w);
scale = exp(log_scale);
if ~isfinite(log_scale) || scale == 0 || ~isfinite(scale)
    error('FLLE:DegenerateWeights', ...
        'Kernel weights underflowed or overflowed; use a valid, larger bandwidth.')
end
% A common factor cancels from ordinary least squares. Scale the ridge by
% the same factor to retain the existing absolute n^(-2) ridge objective.
sqrt_w = exp(0.5*(log_w-log_scale));
A = geometry.design.*sqrt_w;
B = Z.*sqrt_w;
if cond(A'*A) > 1e3
    ridge = exp(-2*log(geometry.n)-log_scale);
    if ~isfinite(ridge) || ridge == 0
        error('FLLE:DegenerateWeights','The scaled ridge is outside numeric range.')
    end
    A = [A; sqrt(ridge).*eye(geometry.d+1)];
    B = [B; zeros(geometry.d+1,geometry.d)];
end
% QR avoids solving the squared-condition-number normal equations.
beta = A\B;
mu = beta(1,:);
if any(~isfinite(mu))
    error('FLLE:NumericalFailure','Local linear prediction is not finite.')
end
end
