function [t,delta_t,Y,Cls] = fml_validate_classifier_inputs(t,X,Y,X_new)
% Shared validation for the functional baseline classifiers.
[t,delta_t] = fml_validate_grid(t,X);
if ~(isnumeric(Y) || islogical(Y)) || ~isreal(Y) || ~isvector(Y) || ...
        numel(Y) ~= size(X,1) || any(~isfinite(Y(:)))
    error('FML:InvalidLabels','Y must contain one finite numeric label per training row.');
end
Y = Y(:);
Cls = unique(Y);
if numel(Cls) < 2
    error('FML:InsufficientClasses','Classification requires at least two training classes.');
end
if ~isnumeric(X_new) || ~isreal(X_new) || ~ismatrix(X_new) || ...
        size(X_new,2) ~= size(X,2) || any(~isfinite(X_new(:)))
    error('FML:InvalidData','X_new must be a finite n_new-by-p matrix.');
end
end
