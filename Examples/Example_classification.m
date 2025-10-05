%% The file contains a simple example implementing the clustering approach by Tan et al. (2024).

% References:
% Tan R. and Zang Y. (2025+). Supervised manifold learning for functional
% data. Journal of Computational Graphical Statistics.

% Author: Ruoxu Tan; date: 2025/Oct/5; Matlab version: R2024b.

%% Pre-set values
n = 200; % Training sample sizes
n_test = 500; % Test sample size  
t_num = 200; % Number of dense time points
t_obs_num = 50; % Number of observations per individual
t = linspace(0,1,t_num); % Time interval where we evaluate/estimate functional data
R = 20; % Signal to noise ratio

t_obs = linspace(0,1,t_obs_num); % Time interval where we observe functional data

%% Generate data. 

% Model (i) in the paper
Y_train = random('Binomial',1,0.5,[n,1]);
Y_test = random('Binomial',1,0.5,[n_test,1]);
X_train_obs = zeros([n,t_obs_num]);
X_test_obs = zeros([n_test,t_obs_num]);
d = 2;

Z_1 = zeros(n,1);
Z_1(Y_train == 0) = random('Uniform',-1,0.2,[sum(Y_train == 0),1]);
Z_1(Y_train == 1) = random('Uniform',-0.2,1,[sum(Y_train == 1),1]);
Z_2 = random('Gamma',4,0.5,[n,1]);
Z_1_test = zeros(n_test,1);
Z_1_test(Y_test == 0) = random('Uniform',-1,0.2,[sum(Y_test == 0),1]);
Z_1_test(Y_test == 1) = random('Uniform',-0.2,1,[sum(Y_test == 1),1]);
Z_2_test = random('Gamma',4,0.5,[n_test,1]);

mu = normpdf(t,0.2,0.08) + normpdf(t,0.5,0.1) + normpdf(t,0.8,0.13); 

for i = 1:n
    if Z_1(i) == 0
        gamma = t_obs;
    else
        gamma = (exp(Z_1(i).*t_obs)-1)./(exp(Z_1(i))-1);
    end
    X_train_obs(i,:) = Z_2(i).*interp1(t,mu,gamma);
end

for i = 1:n_test
    if Z_1_test(i) == 0
        gamma = t_obs;
    else
        gamma = (exp(Z_1_test(i).*t_obs)-1)./(exp(Z_1_test(i))-1);
    end
        X_test_obs(i,:) = Z_2_test(i).*interp1(t,mu,gamma);
end
       
% Adding noise and smoothing.
mu_X = mean(X_test_obs,1);
Cov_X = (X_test_obs-mu_X)'*(X_test_obs-mu_X)/n;
Var_X = mean(diag(Cov_X));
sigma = sqrt(Var_X/R);
epsilon = random('Normal',0,sigma,[n+n_test,t_obs_num]);

X_train = zeros([n,length(t)]);
X_test = zeros([n_test,length(t)]);

for i = 1:n
    X_train_obs(i,:) = X_train_obs(i,:) + epsilon(i,:);
    [~,X_train(i,:),~] = loclin(t_obs,X_train_obs(i,:),min(t),max(t));
end

for i = 1:n_test
    X_test_obs(i,:) = X_test_obs(i,:) + epsilon(i+n,:);
    [~,X_test(i,:),~] = loclin(t_obs,X_test_obs(i,:),min(t),max(t));
end        

%% Classification
% Prespecified parameters.
delta = 0.6;
K_pca = 15;
opt_FPTU = 0;

% FSML+kNN
method = 'k_NN'; para_method = 20;

% Selection of tuning parameters
[xi_CV,h_reg_CV,~] = CV_xih(t,X_train,Y_train,'delta',delta,...
    'K_pca',K_pca,'d',d,'opt_FPTU',opt_FPTU,'method',method,'para_method',para_method);

% Functional supervised manifold learning to produce low-d representations
[Z_train_est,~] = FSML(t,X_train,Y_train,xi_CV,'delta',delta,'K_pca',K_pca,'d',d,'opt_FPTU',opt_FPTU);

% Functional local linear estimator to estimate low-d representations for
% the testing sample
Z_test = zeros(n_test,d);
for j = 1:n_test
    Z_test(j,:) = FLLE(t,X_train,Z_train_est,X_test(j,:),K_pca,h_reg_CV);
end

% Training and evaluating classifiers
Clsfy = fitcknn(Z_train_est,Y_train,'NumNeighbors',para_method);
Y_test_est_k_NN = predict(Clsfy,Z_test);
Err_k_NN = sum(Y_test_est_k_NN~=Y_test)/n_test;

% FSML+SVM
method = 'SVM'; 

% Selection of tuning parameters
[xi_CV,h_reg_CV,~] = CV_xih(t,X_train,Y_train,'delta',delta,...
    'K_pca',K_pca,'d',d,'opt_FPTU',opt_FPTU,'method',method);

% Functional supervised manifold learning to produce low-d representations
[Z_train_est,~] = FSML(t,X_train,Y_train,xi_CV,'delta',delta,'K_pca',K_pca,'d',d,'opt_FPTU',opt_FPTU);



% Functional local linear estimator to estimate low-d representations for
% the testing sample
Z_test = zeros(n_test,d);
for j = 1:n_test
    Z_test(j,:) = FLLE(t,X_train,Z_train_est,X_test(j,:),K_pca,h_reg_CV);
end

% Training and evaluating classifiers
Clsfy = fitcecoc(Z_train_est,Y_train);
Y_test_est_SVM = predict(Clsfy,Z_test);
Err_SVM = sum(Y_test_est_SVM~=Y_test)/n_test;
 
fprintf('The error percentage of FSML+kNN is %0.1f.\n',100*Err_k_NN)
fprintf('The error percentage of FSML+SVM is %0.1f.\n',100*Err_SVM) 