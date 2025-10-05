function [Y_est] = CC(t,X,Y,X_new)
% The centroid classifier proposed by Delaigle and Hall (2012).
% Input:
% t: 1*p time interval;
% X: n*p data matrix, each row contains function values of an individual;
% Y: n*1 classes;
% X_new: 1*p new functional observation to be classified;
% Output:
% Y_est: estimated class of X_new.

Cls = unique(Y);

Cls_pair = [Cls(1) Cls(2)];
idx = Y==Cls_pair(1) | Y==Cls_pair(2);
[phi,~] = CV_CC(t,X(idx,:),Y(idx));
X_0_mean = mean(X(Y==Cls_pair(1),:),1);
X_1_mean = mean(X(Y==Cls_pair(2),:),1);

delta_t = mean(diff(t));
T = (sum(X_new.*phi.*delta_t)-sum(X_1_mean.*phi.*delta_t))^2 -...
    (sum(X_new.*phi.*delta_t)-sum(X_0_mean.*phi.*delta_t))^2 ;
if T > 0
    Y_est = Cls_pair(1);
else
    Y_est = Cls_pair(2);
end

k = 3;
% Pair-wise classification for classes more than 2, more advanced
% techniques like ECOC can be applied if needed.
while Cls_pair(2) ~= Cls(end)
    Cls_pair = [Cls(Cls==Y_est) Cls(k)];
    idx = Y==Cls_pair(1) | Y==Cls_pair(2);
    [phi,~] = CV_CC(t,X(idx,:),Y(idx));
    X_0_mean = mean(X(Y==Cls_pair(1),:),1);
    X_1_mean = mean(X(Y==Cls_pair(2),:),1);

    T = (sum(X_new.*phi.*delta_t)-sum(X_1_mean.*phi.*delta_t))^2 -...
        (sum(X_new.*phi.*delta_t)-sum(X_0_mean.*phi.*delta_t))^2 ;
    if T > 0
        Y_est = Cls_pair(1);
    else
        Y_est = Cls_pair(2);
    end
    k = k+1;
end

end