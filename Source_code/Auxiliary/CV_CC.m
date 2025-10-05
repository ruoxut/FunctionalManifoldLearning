function [phi,k_CV] = CV_CC(t,X,Y)
% Cross validation to select phi used in the centroid classifier.
% Input:
% t: 1*p time interval;
% X: n*p data matrix, each row contains function values of an individual;
% Y: n*1 classes;
% Output:
% phi: 1*p the selected projection function.
% k_CV: number of basis functions.

n = size(X,1);
idx = randperm(n);
X = X(idx,:);
Y = Y(idx,:);
Cls = unique(Y);
if length(Cls)>2
    error('The number of classes are greater than two.')
end

delta_t = mean(diff(t));
[~,psi,lambda] = FPCA(t,X);
%lambda = lambda';

n_CV = 10; 
group_idx = round(linspace(0,n,n_CV+1));
k_range = 1:size(psi,1);

Loss = zeros(length(k_range),1);

parfor k = 1:length(k_range)
    for i = 1:n_CV
        X_out = X((group_idx(i)+1):group_idx(i+1),:);
        Y_out = Y((group_idx(i)+1):group_idx(i+1));
        X_in = X;        
        X_in((group_idx(i)+1):group_idx(i+1),:) = [];
        Y_in = Y;
        Y_in((group_idx(i)+1):group_idx(i+1)) = [];
        X_0_mean = mean(X_in(Y_in==Cls(1),:),1);
        X_1_mean = mean(X_in(Y_in==Cls(2),:),1);

        Y_out_pre = zeros(size(Y_out));
        mu = sum((X_1_mean-X_0_mean).*psi.*delta_t,2);
        phi = sum(lambda(1:k).^(-1) .* mu(1:k) .* psi(1:k,:),1);

        for j = 1:size(X_out,1)
            T = (sum(X_out(j,:).*phi.*delta_t)-sum(X_1_mean.*phi.*delta_t))^2 -...
                (sum(X_out(j,:).*phi.*delta_t)-sum(X_0_mean.*phi.*delta_t))^2 ;
            if T > 0
                Y_out_pre(j,:) = Cls(1);
            else
                Y_out_pre(j,:) = Cls(2);
            end
        end
        Loss(k) = Loss(k) + sum(Y_out_pre ~= Y_out);
    end
end

[~,ind] = min(Loss);
k_CV = k_range(ind);

X_0_mean = mean(X(Y==Cls(1),:),1);
X_1_mean = mean(X(Y==Cls(2),:),1);
mu = sum((X_1_mean-X_0_mean).*psi.*delta_t,2);
phi = sum(lambda(1:k_CV).^(-1) .* mu(1:k_CV) .* psi(1:k_CV,:),1);

end