function Y_est = CC(t,X,Y,X_new)
% Centroid classifier of Delaigle and Hall (2012).
% X_new may contain one or many rows. Each required pairwise classifier is
% fitted/tuned once per call; the original ordered tournament is retained.
[t,delta_t,Y,Cls] = fml_validate_classifier_inputs(t,X,Y,X_new);
n_new = size(X_new,1);
winner = ones(n_new,1);
for k = 2:numel(Cls)
    previous = unique(winner)';
    for j = previous
        use = winner == j;
        pair = Y == Cls(j) | Y == Cls(k);
        phi = CV_CC(t,X(pair,:),Y(pair));
        m0 = mean(X(Y == Cls(j),:),1)*phi'*delta_t;
        m1 = mean(X(Y == Cls(k),:),1)*phi'*delta_t;
        score = X_new(use,:)*phi'*delta_t;
        choose_next = (score-m1).^2 - (score-m0).^2 <= 0;
        rows = find(use);
        winner(rows(choose_next)) = k;
    end
end
Y_est = Cls(winner);
end
