function [phi,k_CV] = CV_CC(t,X,Y)
% Ten-fold selection of the centroid projection; global FPCA is retained.
[t,delta_t,Y,Cls] = fml_validate_classifier_inputs(t,X,Y,zeros(0,size(X,2)));
if numel(Cls) ~= 2
    error('FML:InvalidClasses','CV_CC requires exactly two classes.');
end
n = size(X,1);
idx = randperm(n);
X = X(idx,:);
Y = Y(idx);
[~,psi,lambda] = FPCA(t,X);
d = size(psi,1);
lambda = lambda(1:d)';
scores = X*psi'*delta_t;
group_idx = round(linspace(0,n,11));
Loss = zeros(1,d);
for i = 1:10
    out = (group_idx(i)+1):group_idx(i+1);
    if isempty(out)
        continue
    end
    in = true(n,1);
    in(out) = false;
    if ~any(in & Y == Cls(1)) || ~any(in & Y == Cls(2))
        error('FML:InvalidFold','A centroid CV training fold is missing a class.');
    end
    m0 = mean(scores(in & Y == Cls(1),:),1);
    m1 = mean(scores(in & Y == Cls(2),:),1);
    weights = (m1-m0)./lambda;
    % Column k is the same projection truncated after its first k PCs.
    projected = cumsum(scores(out,:).*weights,2);
    a0 = cumsum(m0.*weights,2);
    a1 = cumsum(m1.*weights,2);
    choose_first = (projected-a1).^2 - (projected-a0).^2 > 0;
    predicted = repmat(Cls(2),numel(out),d);
    predicted(choose_first) = Cls(1);
    Loss = Loss + sum(predicted ~= Y(out),1);
end
[~,k_CV] = min(Loss);
m0 = mean(scores(Y == Cls(1),:),1);
m1 = mean(scores(Y == Cls(2),:),1);
phi = ((m1(1:k_CV)-m0(1:k_CV))./lambda(1:k_CV))*psi(1:k_CV,:);
end
