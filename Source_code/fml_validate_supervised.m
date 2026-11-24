function [t,Y] = fml_validate_supervised(t,X,Y,K_pca,d)
% Shared input contract for supervised fitting and cross-validation.
[t,~] = fml_validate_grid(t,X);
n = size(X,1);
if ~(isnumeric(Y) || islogical(Y)) || ~isreal(Y) || ...
        ~isvector(Y) || numel(Y) ~= n || any(~isfinite(Y(:)))
    error('FML:InvalidLabels','Y must contain one finite numeric label per curve.')
end
Y = Y(:);
validateattributes(d,{'numeric'}, ...
    {'real','finite','scalar','integer','positive','<',n,'<=',size(X,2)});
validateattributes(K_pca,{'numeric'}, ...
    {'real','finite','scalar','integer','>=',d+1,'<=',n});
end
