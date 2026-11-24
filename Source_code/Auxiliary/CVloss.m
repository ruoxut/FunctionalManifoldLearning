function S = CVloss(X,Y,h,a,b)
% Leave-one-out Gaussian local-linear loss used by the plug-in pilot.
X = X(:);
Y = Y(:);
if ~isfinite(h) || h <= 0
    S = Inf;
    return
end
S = 0;
for i = 1:numel(X)
    if X(i) >= a && X(i) <= b
        keep = true(numel(X),1);
        keep(i) = false;
        try
            beta = fml_local_polynomial(X(keep),Y(keep),X(i),h,1);
        catch err
            if startsWith(err.identifier,'FML:')
                S = Inf; % Reject an unsupported bandwidth, never return NaN.
                return
            end
            rethrow(err)
        end
        S = S + (Y(i)-beta(1))^2;
    end
end
end

