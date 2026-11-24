function [ d ] = dim( t,X,delta )
% Intrinsic dimension estimation for functional data by adapting the method
% proposed by Facco et al. (2017).
% Reference:
% Facco, E., d’Errico, M., Rodriguez, A. and Laio, A. (2017). Estimating the intrinsic 
% dimension of datasets by a minimal neighborhood information. Scientific Reports, 7, 1–8.
% Input:
% t: 1*p time interval;
% X: n*p data matrix, each row includes values of a curve;
% delta: a number in (0,1), the fraction of data to be used.
% Output:
% d: estimated intrinsic dimension.

% Author: Ruoxu Tan; date: 2022/Oct/29; Matlab version: R2020a.

if nargin < 3
    delta = 0.9;
end

[~,delta_t] = fml_validate_grid(t,X);
validateattributes(delta,{'numeric'},{'real','finite','scalar','>',0,'<',1},mfilename,'delta');
n = size(X,1);
if n < 3
    error('FML:InsufficientSample','Dimension estimation requires at least three curves.');
end
mu_X = mean(X,1);
X_cen = X - mu_X;
Cov = X_cen' * X_cen / n;
[phi,~,expd] = pcacov(Cov);
if any(~isfinite(expd)) || sum(expd) <= 0
    error('FML:ZeroVariance','Dimension estimation requires positive variation among curves.');
end
norm_phi = sqrt(sum(phi.^2.*delta_t,1));       
phi = phi ./ norm_phi;
phi = phi';

n_c = find(cumsum(expd) >= 95,1);
if isempty(n_c)
    error('FML:InvalidPCA','Could not determine the FPCA variance threshold.');
end
Xi_n_c = zeros(n,n_c); 
for i = 1:n
    xi = sum(X_cen(i,:).*phi(1:n_c,:).*delta_t,2);
    Xi_n_c(i,:) = xi'; 
end

G = zeros(n);
for i = 1:n
    for j = i+1:n
        G(i,j) = norm(Xi_n_c(i,:)-Xi_n_c(j,:));
        G(j,i) = G(i,j);
    end
end

mu = zeros(1,n);
for i = 1:n
    d_i = sort(G(i,[1:i-1,i+1:n]));
    if d_i(1) <= 0
        error('FML:DuplicateScores','Dimension estimation is undefined for duplicate FPCA scores.');
    end
    mu(i) = d_i(2)/d_i(1);
end
n_sub = round(delta*n);
if n_sub == n
    n_sub = n-1;
end
if n_sub < 1
    error('FML:InsufficientSample','delta retains no observations for dimension estimation.');
end

mu = sort(mu);
mu = mu(1:n_sub);

X_data = log(mu);
Y_data = -log(1-(1:n_sub)./n); 
denominator = sum(X_data.^2);
if ~isfinite(denominator) || denominator <= 0
    error('FML:DegenerateDimension','Nearest-neighbor distance ratios do not identify a finite dimension.');
end
d = round(sum(X_data.*Y_data)/denominator);
if ~isfinite(d) || d < 1 || d > min(n-1,size(X,2))
    error('FML:InvalidDimension','The estimated dimension is not supported by the sample and evaluation grid.');
end

end
