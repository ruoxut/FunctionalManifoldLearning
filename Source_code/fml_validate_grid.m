function [t,delta_t] = fml_validate_grid(t,X)
% Validate the common, equally spaced evaluation grid used by these methods.
if ~isnumeric(t) || ~isreal(t) || ~isvector(t) || numel(t) < 2 || any(~isfinite(t(:)))
    error('FML:InvalidGrid','t must contain at least two finite real time points.');
end
t = t(:)';
steps = diff(t);
if any(steps <= 0)
    error('FML:InvalidGrid','Time points must be strictly increasing.');
end
delta_t = mean(steps);
tol = max(1e-8*delta_t,32*eps(max(1,max(abs(t)))));
if any(abs(steps-delta_t) > tol)
    error('FML:NonuniformGrid', ...
        'These routines require an equally spaced evaluation grid; evaluate the smoothed curves on such a grid first.');
end
if ~isnumeric(X) || ~isreal(X) || ~ismatrix(X) || isempty(X) || ...
        size(X,2) ~= numel(t) || any(~isfinite(X(:)))
    error('FML:InvalidData','X must be a finite real matrix with one curve per row and one column per time point.');
end
end
