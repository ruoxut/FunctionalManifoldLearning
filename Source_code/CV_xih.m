function [xi_CV,h_reg_CV,Loss,D0] = CV_xih(t,X,Y,varargin)
% Cross-validation selection for xi used in supervised manifold learning
% and bandwidth used in functional regression.
% Input:
% t: 1*p time interval;
% X: n*p data matrix, each row contains function values of an individual;
% Y: n*1 classes;
% delta: a tuning parameter in (0,1] to control the number of edges of the adjacency matrix, 
% larger delta leads to more edges;
% K_pca: : number of nearest neighbours used in local PCA;
% d: intrinsic dimension;
% opt_FPTU: the option for FPTU, =1 means rescale; otherwise not rescale;
% method: the classification method for low dimensional data.
% para_method: possible tuning parameter of the method, e.g., k of the k-NN.
% Geodesic: optional precomputed geodesics in the original input row order.
% Output:
% xi_CV: CV selected xi.
% h_reg_CV: CV selected h_reg. 
% Loss: Total misclassification CV loss.
% D0: reusable geodesic matrix in the original input row order.
% Candidate grids: xi=(s_D*rho)^2, with s_D the median positive pairwise
% geodesic distance and rho=0 plus 19 log-spaced values from 0.05 to 4.5.
% Each outer fold uses 20 log-spaced bandwidths from 0.2*h0 to 5*h0,
% where h0 is based on functional input variation, independently of xi.

% Author: Ruoxu Tan; date: 2025/May; Matlab version: R2024b. 

p = inputParser;
addParameter(p,'delta',0.6);
addParameter(p,'K_pca',15);
addParameter(p,'opt_FPTU',0);
addParameter(p,'d',2);
addParameter(p,'method','LDA');
addParameter(p,'para_method',10);
addParameter(p,'Geodesic',[]);
parse(p,varargin{:})

delta = p.Results.delta;
K_pca = p.Results.K_pca;
opt_FPTU = p.Results.opt_FPTU;
d = p.Results.d;
method = p.Results.method;
para_method = p.Results.para_method;
[t,Y] = fml_validate_supervised(t,X,Y,K_pca,d);
method = validatestring(method,{'k_NN','SVM','LDA'});

n = size(X,1);
n_fold_CV = 10;
if n < n_fold_CV || numel(unique(Y)) < 2
    error('FML:InvalidCVSample','Ten-fold classification CV requires at least ten curves and two classes.')
end
min_outer_train = n-ceil(n/n_fold_CV);
min_inner_train = min_outer_train-ceil(min_outer_train/n_fold_CV);
if K_pca > min_inner_train
    error('FML:InvalidCVNeighborhood', ...
        'K_pca exceeds the smallest inner training fold size (%d).',min_inner_train)
end
if strcmp(method,'k_NN')
    validateattributes(para_method,{'numeric'}, ...
        {'real','finite','scalar','integer','positive','<=',min_outer_train});
end
D0 = p.Results.Geodesic;
if ~isempty(D0)
    D0 = fml_validate_distances(D0,n);
end

idx = randperm(n);
X = X(idx,:);
Y = Y(idx,:);

group_idx = round(linspace(0,n,n_fold_CV+1));

% The paper computes geodesics once before the nested sample splits.
if isempty(D0)
    G_adj = Adj_Graph(t,X,delta);
    G_FSML = FPTU_adj_input(t,X,G_adj,K_pca,d,opt_FPTU);
    G_FSML = fml_validate_distances(G_FSML,n);
    D0 = zeros(n);
    D0(idx,idx) = G_FSML;
else
    G_FSML = D0(idx,idx);
end
pair_distances = G_FSML(triu(true(n),1) & G_FSML > 0);
d_scale = median(pair_distances);
if isempty(d_scale) || ~isfinite(d_scale) || d_scale <= 0
    error('FML:DegenerateDistances','Positive finite geodesics are required to form the xi grid.')
end
rho_range = [0,logspace(log10(0.05),log10(4.5),19)];
xi_range = (d_scale*rho_range).^2;
if any(~isfinite(xi_range)) || any(xi_range(2:end) <= 0)
    error('FML:DegenerateDistances','The geodesic scale is outside the numeric range for xi.')
end

Loss = zeros(size(xi_range));
h_reg_CV_mat = nan(length(xi_range), n_fold_CV);

X_out = cell(1,n_fold_CV);
Y_out = cell(1,n_fold_CV);
X_in = cell(1,n_fold_CV);
Y_in = cell(1,n_fold_CV);
G_FSML_in = cell(1,n_fold_CV);
outer_geometry = cell(1,n_fold_CV);
inner_cache = cell(1,n_fold_CV);
h_reg_ranges = cell(1,n_fold_CV);
delta_t = mean(diff(t));

for i = 1:n_fold_CV
    X_out{i} = X((group_idx(i)+1):group_idx(i+1),:);
    Y_out{i} = Y((group_idx(i)+1):group_idx(i+1));
    X_in{i} = X;        
    X_in{i}((group_idx(i)+1):group_idx(i+1),:) = [];
    Y_in{i} = Y;
    Y_in{i}((group_idx(i)+1):group_idx(i+1)) = [];
    if numel(unique(Y_in{i})) < 2
        error('FML:MissingCVClass', ...
            'An outer training fold contains fewer than two classes; choose another split or more data.')
    end
    G_FSML_in{i} = G_FSML;
    G_FSML_in{i}((group_idx(i)+1):group_idx(i+1),:) = [];
    G_FSML_in{i}(:,(group_idx(i)+1):group_idx(i+1)) = [];
    % h has the units of L2 distances in X, not of the MDS responses Z.
    % Fix one range per outer training fold and reuse it for every xi.
    n_in = size(X_in{i},1);
    X_centered = X_in{i}-mean(X_in{i},1);
    s_X = norm(X_centered,'fro')*sqrt(delta_t/(d*(n_in-1)));
    h_pilot = s_X*n_in^(-1/(d+4));
    h_reg_ranges{i} = h_pilot*logspace(log10(0.2),log10(5),20);
    if any(~isfinite(h_reg_ranges{i})) || any(h_reg_ranges{i} <= 0)
        error('FML:DegeneratePredictors', ...
            'Positive finite input variation is required to form the bandwidth grid in outer fold %d.',i)
    end
    inner_idx = randperm(size(X_in{i},1));
    outer_geometry{i} = prepare_queries(t,X_in{i},X_out{i},K_pca,d);
    inner_cache{i} = prepare_inner_cv(t,X_in{i},K_pca,d,n_fold_CV,inner_idx);
end

parfor k = 1:length(xi_range)
    h_reg_CV_row = zeros(1, n_fold_CV);
    fold_loss = 0;
    for i = 1:n_fold_CV
        G_FSML_in_i = G_FSML_in{i};
        if xi_range(k) > 0
            different_class = Y_in{i} ~= Y_in{i}';
            G_FSML_in_i(different_class) = G_FSML_in_i(different_class) ...
                + xi_range(k)./(G_FSML_in_i(different_class)+sqrt(xi_range(k)));
        end
        Z_d_in = [];
        try
            Z_d_in = fml_mds(G_FSML_in_i,d);
        catch err
            if strcmp(err.identifier,'stats:mdscale:NoSolution') || ...
                    strcmp(err.identifier,'stats:mdscale:ColocatedPoints')
                fold_loss = Inf;
                break
            end
            rethrow(err)
        end

        Z_out_pre = zeros(length(Y_out{i}),d);        
        [h_reg_CV_row(i),inner_loss] = CV_FLLE_cached(Z_d_in,inner_cache{i},h_reg_ranges{i});
        if all(~isfinite(inner_loss))
            fold_loss = Inf;
            break
        end

        for j = 1:length(Y_out{i})
            try
                Z_out_pre(j,:) = FLLE_predict(outer_geometry{i}{j},Z_d_in,h_reg_CV_row(i));
            catch err
                if is_prediction_failure(err)
                    fold_loss = Inf;
                    break
                end
                rethrow(err)
            end
        end
        if ~isfinite(fold_loss)
            break
        end

        Y_out_pre = zeros(size(Y_out{i}));
        if strcmp(method,'k_NN')
            Mdl = fitcknn(Z_d_in,Y_in{i},'NumNeighbors',para_method);
            Y_out_pre = predict(Mdl,Z_out_pre); 
        elseif strcmp(method,'SVM')
            Mdl = fitcecoc(Z_d_in,Y_in{i});
            Y_out_pre = predict(Mdl,Z_out_pre);
        elseif strcmp(method,'LDA')
            Mdl = fitcecoc(Z_d_in,Y_in{i},'Learners','discriminant');
            Y_out_pre = predict(Mdl,Z_out_pre);
        else
            error('The classification method is not defined.')
        end

        fold_loss = fold_loss + sum(Y_out_pre ~= Y_out{i});
    end
    Loss(k) = fold_loss;
    h_reg_CV_mat(k,:) = h_reg_CV_row;
end

[~,ind_opt] = min(Loss);
if ~isfinite(Loss(ind_opt))
    error('FML:NoValidCVFit', ...
        'Every xi/bandwidth candidate failed; inspect the bandwidth range and local ranks.')
end
h_reg_CV = mean(h_reg_CV_mat(ind_opt,:)); 
xi_CV = xi_range(ind_opt);

end
 

function geometry = prepare_queries(t,X,X_query,K_pca,d)
geometry = cell(size(X_query,1),1);
for j = 1:size(X_query,1)
    geometry{j} = FLLE_prepare(t,X,X_query(j,:),K_pca,d);
end
end

function cache = prepare_inner_cv(t,X,K_pca,d,n_CV,idx)
n = size(X,1);
group_idx = round(linspace(0,n,n_CV+1));
cache = cell(1,n_CV);
for i = 1:n_CV
    positions = (group_idx(i)+1):group_idx(i+1);
    test = idx(positions);
    train = idx;
    train(positions) = [];
    geometry = prepare_queries(t,X(train,:),X(test,:),K_pca,d);
    cache{i} = struct('train',train,'test',test,'geometry',{geometry});
end
end

function [h_reg_CV,Loss] = CV_FLLE_cached(Z,cache,h_reg_range)
Loss = zeros(size(h_reg_range));
for k = 1:length(h_reg_range)
    failed = false;
    for i = 1:numel(cache)
        fold = cache{i};
        Z_in = Z(fold.train,:);
        for j = 1:numel(fold.test)
            try
                mu = FLLE_predict(fold.geometry{j},Z_in,h_reg_range(k));
                Loss(k) = Loss(k)+sum((Z(fold.test(j),:)-mu).^2);
            catch err
                if is_prediction_failure(err)
                    Loss(k) = Inf;
                    failed = true;
                    break
                end
                rethrow(err)
            end
        end
        if failed
            break
        end
    end
end
[~,ind_opt] = min(Loss);
h_reg_CV = h_reg_range(ind_opt);
end

function failed = is_prediction_failure(err)
failed = strcmp(err.identifier,'FLLE:DegenerateWeights') ...
    || strcmp(err.identifier,'FLLE:NumericalFailure');
end

