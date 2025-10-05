%% The file contains a simple example implementing the clustering approach by Tan et al. (2024).

% References: 
% Tan, R., Zang, Y. and Yin, G. (2024). Nonlinear dimension reduction for functional data with application to clustering.
% Statistica Sinica, 34, 1391-1412.

% Author: Ruoxu Tan; date: 2025/Oct/5; Matlab version: R2024b.

%% Pre-set values
n = 200; % Sample size
K = 10; % K-nearest neighbours
p = 200; % Number of dense time points
t = linspace(0,1,p); % Time interval where we evaluate/estimate functional data
p_obs = 60; % Number of observations per individual
n_start = 20; % Number of replicates used in k-means clustering

t_obs = linspace(0,1,p_obs); % Time interval where we observe functional data
n_1 = n/2;% Half of inviduals forms one group.
R = 10; % Signal to noise ratio

%% Generate data.
X_obs = zeros(n,p_obs);
X = zeros(n,p);
           
% Model (i) in the paper
Z_1 = [random('Uniform',0,7,[n_1,1]) ; random('Uniform',7,10,[n-n_1,1])];
Z_2 = random('Uniform',0,4,[n,1]);
A = [Z_1(:,1).*cos(Z_1(:,1)) Z_1(:,1).*sin(Z_1(:,1))+Z_2(:,1) Z_2(:,1)];
 
for i = 1:n
    X_obs(i,:) = A(i,1).*sin(2.*pi.*t_obs) + A(i,2).*cos(2.*pi.*t_obs) +  A(i,3).*sin(4.*pi.*t_obs); 
end
      
% Add noise.
mu_X = mean(X_obs,1);
Cov_X = (X_obs-mu_X)'*(X_obs-mu_X)/n;
Var_X = mean(diag(Cov_X));
sigma = sqrt(Var_X/R);
epsilon = random('Normal',0,sigma,[n,p_obs]);

for i = 1:n
    X_obs(i,:) = X_obs(i,:) + epsilon(i,:);
end
       
% Presmoothing using the ridged local linear estimator.
parfor i = 1:n
    [~,X(i,:),~]  = loclin( t_obs,X_obs(i,:),min(t_obs),max(t_obs) );
end
       
       
%% Clustering
G = zeros(1,n); % True group labels  
G(1,1:n_1) = 1;
G(1,n_1+1:n) = 2;

[X,G] = rmout(t,X,G,K,2);% Remove outliers.
g = length(unique(G));

d_est = dim( t,X,0.9 ); % Intrinsic dimension estimated
          
% FPTU + graph clustering
[ D_FPTU_res,~ ] = FPTU_adj_knn( t,X,K,K,d_est,1 ); % Perform the functional parallel transport unfolding.
[ X_d_FPTU_res,G_FPTU_res,~ ] = graph_clustering( t,X,D_FPTU_res,g,d_est,n_start ); % Perform the graph-based clustering using the proximity graph D.            
           
% Standard k-means clustering
G_st = kmeans(X,g,'Replicates',n_start);

% Compute the adjusted Rand index.
ADI_FPTU_res = rand_index(G,G_FPTU_res,'adjusted'); 
ADI_st = rand_index(G,G_st,'adjusted');

% Print results.
fprintf('The adjusted Rand index of FPTU + graph clustering is %0.2f.\n',ADI_FPTU_res);
fprintf('The adjusted Rand index of the standard k-means clustering is %0.2f.\n',ADI_st);
