function [xi_CV,h_reg_CV,Loss] = CV_xih(t,X,Y,varargin)
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
% Output:
% xi_CV: CV selected xi.
% h_reg_CV: CV selected h_reg. 
% Loss: Total misclassification CV loss.

% Author: Ruoxu Tan; date: 2025/May; Matlab version: R2024b. 

p = inputParser;
addParameter(p,'delta',0.6);
addParameter(p,'K_pca',15);
addParameter(p,'opt_FPTU',0);
addParameter(p,'d',2);
addParameter(p,'method','LDA');
addParameter(p,'para_method',10);
parse(p,varargin{:})

delta = p.Results.delta;
K_pca = p.Results.K_pca;
opt_FPTU = p.Results.opt_FPTU;
d = p.Results.d;
method = p.Results.method;
para_method = p.Results.para_method;

n = size(X,1);
n_fold_CV = 10;

idx = randperm(n);
X = X(idx,:);
Y = Y(idx,:);

group_idx = round(linspace(0,n,n_fold_CV+1));

G_adj = Adj_Graph(t,X,delta);
[G_FSML,~] = FPTU_adj_input( t,X,G_adj,K_pca,d,opt_FPTU ); 
d_mean = mean(G_FSML(G_FSML~=0));
xi_range = linspace(d_mean^2/2,d_mean^2*20,20);

Loss = zeros(size(xi_range));
h_reg_CV_mat = zeros(length(xi_range), n_fold_CV);

X_out = cell(1,n_fold_CV);
Y_out = cell(1,n_fold_CV);
X_in = cell(1,n_fold_CV);
Y_in = cell(1,n_fold_CV);
G_FSML_in = cell(1,n_fold_CV);

for i = 1:n_fold_CV
    X_out{i} = X((group_idx(i)+1):group_idx(i+1),:);
    Y_out{i} = Y((group_idx(i)+1):group_idx(i+1));
    X_in{i} = X;        
    X_in{i}((group_idx(i)+1):group_idx(i+1),:) = [];
    Y_in{i} = Y;
    Y_in{i}((group_idx(i)+1):group_idx(i+1)) = [];
    G_FSML_in{i} = G_FSML;
    G_FSML_in{i}((group_idx(i)+1):group_idx(i+1),:) = [];
    G_FSML_in{i}(:,(group_idx(i)+1):group_idx(i+1)) = [];
end

parfor k = 1:length(xi_range)
    h_reg_CV_row = zeros(1, n_fold_CV);
    for i = 1:n_fold_CV
        G_FSML_in_i = G_FSML_in{i};
        for j = 1:size(G_FSML_in_i,1)
            for m = j+1:size(G_FSML_in_i,1)
                if Y_in{i}(j) ~= Y_in{i}(m)
                    G_FSML_in_i(j,m) = G_FSML_in_i(j,m) + xi_range(k)./(G_FSML_in_i(j,m)+sqrt(xi_range(k)));
                    G_FSML_in_i(m,j) = G_FSML_in_i(j,m);
                end
            end
        end 
        warning('off','stats:mdscale:IterOrEvalLimit');
        Z_d_in = mdscale(G_FSML_in_i,d,"Criterion","metricstress");

        Z_out_pre = zeros(length(Y_out{i}),d);        
        n_in = length(Y_in{i});

        sd_ave = mean(std(Z_d_in,0,1));
        h_pilot = sd_ave / (n_in^(1/(d+4))); 
        h_reg_range = linspace(h_pilot/5,h_pilot*5,20); 
        [h_reg_CV_row(i),~] = CV_FLLE(t,X_in{i},Z_d_in,K_pca,h_reg_range,n_fold_CV);

        for j = 1:length(Y_out{i})
            Z_out_pre(j,:) = FLLE(t,X_in{i},Z_d_in,X_out{i}(j,:),K_pca,h_reg_CV_row(i));
        end

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

        Loss(k) = Loss(k) + sum(Y_out_pre ~= Y_out{i});
    end
    h_reg_CV_mat(k,:) = h_reg_CV_row;
end

[~,ind_opt] = min(Loss);
h_reg_CV = mean(h_reg_CV_mat(ind_opt,:)); 
xi_CV = xi_range(ind_opt);

end
 

function [h_reg_CV,Loss] = CV_FLLE(t,X,Z,K_pca,h_reg_range,n_CV)
% Cross-validation selection for the bandwidth h_reg in local linear regression on tangent space.
% Input:
% t: 1*p time interval;
% X: n*p data matrix, each row contains function values of an individual; 
% Z: n*d outcome matrix, each row contains the outcome of an individual;
% larger delta leads to more edges;
% K_pca: number of nearest neighbours used in local PCA;
% h_reg_range: contains candidates for h_reg;
% n_CV: n_CV-fold cross validation;
% Output: 
% h_reg_CV: CV selected h_reg.

if length(t) ~= size(X,2) || size(X,1)~=size(Z,1)
    error('Dimensions of the input data do not match.')
end

if iscolumn(t)
    t = t';
end

n = size(X,1);

idx = randperm(n);
X = X(idx,:);
Z = Z(idx,:);

group_idx = round(linspace(0,n,n_CV+1));

Loss = zeros(size(h_reg_range));

X_out = cell(1,n_CV);
Z_out = cell(1,n_CV);
X_in = cell(1,n_CV);
Z_in = cell(1,n_CV);
for i = 1:n_CV
    X_out{i} = X((group_idx(i)+1):group_idx(i+1),:);
    Z_out{i} = Z((group_idx(i)+1):group_idx(i+1),:);
    X_in{i} = X;        
    X_in{i}((group_idx(i)+1):group_idx(i+1),:) = [];
    Z_in{i} = Z;
    Z_in{i}((group_idx(i)+1):group_idx(i+1),:) = [];
end

parfor k = 1:length(h_reg_range)
    for i = 1:n_CV 
        for j = 1:size(X_out,1)
            Z_out_pre_j = FLLE(t,X_in{i},Z_in{i},X_out{i}(j,:),K_pca,h_reg_range(k));
            Loss(k) = Loss(k) + norm(Z_out{i}(j,:)-Z_out_pre_j,"fro")^2;
        end
    end
end

[~,ind_opt] = min(Loss);
h_reg_CV = h_reg_range(ind_opt);

end

