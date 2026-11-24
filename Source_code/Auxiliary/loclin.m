function [t,f1,f2] = loclin(X,Y,a,b,moment_convention)
% Ridged local-linear curve and local-quadratic first derivative.
% The original CV/plug-in bandwidth stages are retained; polynomial fits use
% centered/scaled QR rather than expanded determinant formulas.
% Optional moment_convention:
%   'classification' (default): Sk=mean(K(u).*u.^k), JCGS Eq. (3.2).
%   'clustering': Sk=mean(K(u).*u.^k)/h, Sinica Eq. (2.2).
% Both use lambda=J^(-2), with J=numel(X). The derivative's determinant ridge
% retains the original raw sum K_h(dx)*dx^k convention (not specified by those
% displayed curve-estimation formulas). At exact singularity we diagnose the
% unsupported fit rather than silently returning an arbitrary estimate.
if nargin < 3
    a = quantile(X,0.025);
    b = quantile(X,0.975);
end
if nargin < 5
    moment_convention = 'classification';
end
moment_convention = validatestring(moment_convention,{'classification','clustering'});
if ~isnumeric(X) || ~isnumeric(Y) || ~isreal(X) || ~isreal(Y) || ...
        ~isvector(X) || ~isvector(Y) || numel(X) ~= numel(Y) || ...
        any(~isfinite(X(:))) || any(~isfinite(Y(:)))
    error('FML:InvalidData','Smoothing requires equally long finite real X and Y vectors.');
end
X = double(X(:));
Y = double(Y(:));
validateattributes(a,{'numeric'},{'real','finite','scalar'});
validateattributes(b,{'numeric'},{'real','finite','scalar','>',a});
n = numel(X);
n_unique = numel(unique(X));
if n_unique < 5
    error('FML:InsufficientSamples','The quartic bandwidth pilot requires five distinct covariates.');
end
t = linspace(a,b,200)';
f1 = zeros(size(t));
f2 = zeros(size(t));
if all(Y == Y(1))
    f1(:) = Y(1);
    return
end
Wei = X >= a & X <= b;
if ~any(Wei)
    error('FML:NoSmoothingData','No observations fall in the smoothing interval.');
end
hCV = select_pilot_bandwidth(X,Y,a,b);
fpilot = zeros(n,1);
WV = zeros(n,n);
for i = find(Wei)'
    [beta,WV(i,:)] = fml_local_polynomial(X,Y,X(i),hCV,1);
    fpilot(i) = beta(1);
end
nu = n-2*trace(WV)+sum(WV(:).^2);
if ~isfinite(nu) || nu <= 0
    error('FML:InvalidPilot','The smoothing variance pilot has no positive residual degrees of freedom.');
end
Var = sum((Y-fpilot).^2.*Wei)/nu;

% Unweighted global quartic pilot, equivalent to the old normal equations
% before their determinant was artificially clamped to 1e-4.
center = mean(X);
scale = max(abs(X-center));
z = (X-center)/scale;
B = z.^(0:4);
[Q,R] = qr(B,0);
if any(abs(diag(R)) <= max(size(B))*eps(norm(R,inf)))
    error('FML:SingularSmoother','The quartic bandwidth pilot is rank deficient.');
end
c = R\(Q'*Y);
f_qua_2 = (2*c(3)+6*c(4)*z+12*c(5)*z.^2)/scale^2;
f_qua_4 = repmat(24*c(5)/scale^4,n,1);
Theta_24 = sum(f_qua_2.*f_qua_4.*Wei)/n;
if Theta_24 > 0
    C_2K = (15/(16*sqrt(pi)))^(1/7);
else
    C_2K = (3/(8*sqrt(pi)))^(1/7);
end
if Theta_24 == 0
    hCV3 = 2*(b-a);
else
    hCV3 = C_2K*(Var/(n*abs(Theta_24)))^(1/7);
end
check_bandwidth(hCV3);
fCV3 = zeros(n,1);
ffCV3 = zeros(n,1);
for i = find(Wei)'
    beta = fml_local_polynomial(X,Y,X(i),hCV3,3);
    fCV3(i) = 2*beta(3);
    ffCV3(i) = 6*beta(4);
end
Theta1 = sum(fCV3.^2.*Wei)/n;
Theta2 = sum(ffCV3.^2.*Wei)/n;
if Theta1 == 0
    h1 = 2*(b-a);
else
    h1 = (Var/(2*sqrt(pi)*Theta1*n))^(1/5);
end
if Theta2 == 0
    h2 = 2*(b-a);
else
    h2 = 0.8843*(Var/(Theta2*n))^(1/7);
end
check_bandwidth(h1);
check_bandwidth(h2);
log_lambda = -2*log(n);
% The derivative historically used the number of distinct times for lambda.
log_lambda_derivative = -2*log(n_unique);
for i = 1:numel(t)
    [beta,~,logdet] = fml_local_polynomial(X,Y,t(i),h1,1);
    % Raw linear det already cancels h^(-2) against dx^2. Thus dividing
    % raw moments by J and using u=dx/h gives this normalized determinant.
    logdet = logdet-2*log(n);
    if strcmp(moment_convention,'clustering')
        logdet = logdet-2*log(h1);
    end
    f1(i) = beta(1)*ridge_factor(logdet,log_lambda);
    [beta,~,logdet] = fml_local_polynomial(X,Y,t(i),h2,2);
    f2(i) = beta(2)*ridge_factor(logdet,log_lambda_derivative);
end
if any(~isfinite(f1)) || any(~isfinite(f2))
    error('FML:NonfiniteSmoother','The local polynomial estimates are nonfinite.');
end
end

function check_bandwidth(h)
if ~isreal(h) || ~isfinite(h) || h <= 0
    error('FML:InvalidBandwidth','The plug-in smoother produced a nonpositive or nonfinite bandwidth.');
end
end

function h = select_pilot_bandwidth(X,Y,a,b)
% Search the original leave-one-out loss across scales before refining one
% local bracket. Unsupported candidates are excluded, not interpolated.
width = b-a;
spacing = diff(unique(X));
spacing = median(spacing(spacing > 0));
lower = max(spacing/2,sqrt(eps)*width);
upper = width;
if ~isfinite(lower) || ~isfinite(upper) || lower <= 0 || lower >= upper
    error('FML:InvalidPilotRange', ...
        'The smoothing interval must be wide enough relative to the observation spacing.');
end
h_grid = unique(logspace(log10(lower),log10(upper),20));
loss = Inf(size(h_grid));
for k = 1:numel(h_grid)
    candidate = CVloss(X,Y,h_grid(k),a,b);
    if isreal(candidate) && isfinite(candidate)
        loss(k) = candidate;
    end
end
[best_loss,best] = min(loss);
if ~isfinite(best_loss)
    error('FML:NoValidSmoothingBandwidth', ...
        'No pilot bandwidth supports finite leave-one-out local-linear fits.');
end
h = h_grid(best);
left = max(1,best-1);
right = min(numel(h_grid),best+1);
if left < right
    opt = optimset('Display','off','MaxIter',20);
    try
        [log_h,refined_loss,exitflag] = fminbnd( ...
            @(log_h) CVloss(X,Y,exp(log_h),a,b), ...
            log(h_grid(left)),log(h_grid(right)),opt);
        if exitflag > 0 && isreal(refined_loss) && isfinite(refined_loss) && ...
                isfinite(log_h) && refined_loss <= best_loss
            h = exp(log_h);
        elseif exitflag <= 0 || ~isreal(refined_loss) || ~isfinite(refined_loss)
            warning('FML:SmoothingPilotRefinement', ...
                'Pilot refinement did not converge to a finite fit; retaining the grid minimum.');
        end
    catch err
        warning('FML:SmoothingPilotRefinement', ...
            'Pilot refinement failed; retaining the grid minimum (%s).',err.identifier);
    end
end
if h <= lower*(1+1e-3) || h >= upper*(1-1e-3)
    warning('FML:SmoothingPilotBoundary', ...
        'The pilot bandwidth is at the search boundary; inspect the observation spacing and interval.');
end
end

function factor = ridge_factor(logdet,log_lambda)
% det/(det+lambda) avoids underflow and treats a zero determinant's sign as +.
if logdet < log_lambda
    ratio = exp(logdet-log_lambda);
    factor = ratio/(1+ratio);
else
    factor = 1;
end
end

