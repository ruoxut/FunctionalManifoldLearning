%% The file reproduces the simulation of geodesic distance estimation in Tan et al. (2022+). 

% References: 
% Tan, R., Zang, Y. and Yin, G. (2022+). Nonlinear dimension reduction for functional data with
% application to clustering. Statistica Sinica. 

% Author: Ruoxu Tan; date: 2022/Oct/29; Matlab version: R2020a.

n_all = [100 200 500];% Sample sizes
K_all = [10 13 18];% K-nearest neighbours
n_rep = 100;% Number of simulation repetition
p = 200;% Number of dense time points
t = linspace(0,1,p)';% Time interval where we evaluate/estimate functional data
p_all = [30 60 100];% Numbers of observations per individual

Noise = 0;% 1: with noise; others: without noise

for n_ind = 1:length(n_all)
    n = n_all(n_ind);
    K = K_all(n_ind);
    p_obs = p_all(n_ind);
    t_obs = linspace(0,1,p_obs)';
    K_pca = K;
    Er_all = zeros(n_rep,5); 

    for n_rep_ind = 1:n_rep
        %% Generate data.
        rng(3*n_rep_ind)

        theta = random('Uniform',0,pi,[n,1]); 
        phi = random('Uniform',0,2*pi,[n,1]);
 
        A = [sin(theta).*cos(phi) sin(theta).*sin(phi) cos(theta)]; 

        D_true = zeros(n); %True geodesic distance
        for i = 1:n
            for j = 1:n
                if j ~= i
                    C = norm(A(i,:) - A(j,:));
                    D_true(i,j) = 2*asin(C/2);
                end
            end
        end
        
        X_obs = zeros(p_obs,n);
        X = zeros(p,n);

        if Noise == 1
            R = 3; % 10 % Signal to noise ratio

            for i = 1:n
                X_obs(:,i) = (A(i,1).*sin(2.*pi.*t_obs) + A(i,2).*cos(2.*pi.*t_obs) +  A(i,3).*sin(4.*pi.*t_obs)).*(sqrt(2)); 
            end
            
            mu_X = mean(X_obs,2);
            Cov_X = (X_obs-mu_X)*(X_obs-mu_X)'/n;
            Var_X = mean(diag(Cov_X));
            sigma = sqrt(Var_X/R);
            epsilon = random('Normal',0,sigma,[p_obs,n]);

            for i = 1:n
                X_obs(:,i) = X_obs(:,i) + epsilon(:,i);
            end
            
            % Presmoothing
            parfor i = 1:n
                [~,X(:,i),~]  = loclin( t_obs,X_obs(:,i),min(t_obs),max(t_obs) );
            end
            
        else
            R = Inf;

            for i = 1:n
                X(:,i) = (A(i,1).*sin(2.*pi.*t) + A(i,2).*cos(2.*pi.*t) +  A(i,3).*sin(4.*pi.*t)).*(sqrt(2)); 
            end
            
        end
        
        d = 2; %Intrinsic dimension known
        %d = dim( t,X,0.9 ); %Intrinsic dimension estimated

        %% Geodesic distance estimation
        [D_FPTU_res,~] = FPTU( t,X,K,K_pca,d,1 );
        [D_FPTU_nonres,~] = FPTU( t,X,K,K_pca,d,0 );
        D_PTU_PCA_res = PTU_PCA( t,X,K,K_pca,d,1 );
        D_PTU_PCA_nonres = PTU_PCA( t,X,K,K_pca,d,0 );
        [D_Iso,~] =  FIsomap( t,X,K ); 
        
        % Compute the mean relative errors (MREs).
        Er_FPTU_res = abs(D_FPTU_res - D_true)./ D_true;
        Er_FPTU_nonres = abs(D_FPTU_nonres - D_true)./ D_true;
        Er_PTU_PCA_res = abs(D_PTU_PCA_res - D_true)./ D_true;
        Er_PTU_PCA_nonres = abs(D_PTU_PCA_nonres - D_true)./ D_true;
        Er_Iso = abs(D_Iso - D_true)./ D_true;
 
        for i = 1:n
            for j = 1:n
                if j >= i
                    Er_FPTU_res(i,j) = Inf;
                    Er_FPTU_nonres(i,j) = Inf;
                    Er_PTU_PCA_res(i,j) = Inf;
                    Er_PTU_PCA_nonres(i,j) = Inf;
                    Er_Iso(i,j) = Inf;
                end
            end
        end
        Erv_FPTU_res = Er_FPTU_res(~isinf(Er_FPTU_res));
        Erv_FPTU_nonres = Er_FPTU_nonres(~isinf(Er_FPTU_nonres));
        Erv_PTU_PCA_res = Er_PTU_PCA_res(~isinf(Er_PTU_PCA_res));
        Erv_PTU_PCA_nonres = Er_PTU_PCA_nonres(~isinf(Er_PTU_PCA_nonres));
        Erv_Iso = Er_Iso(~isinf(Er_Iso));
        
        Er_all(n_rep_ind,:) = [mean(Erv_FPTU_res) mean(Erv_FPTU_nonres) ...
                              mean(Erv_PTU_PCA_res) mean(Erv_PTU_PCA_nonres) mean(Erv_Iso)];

    end
    
    % Store and print the MREs.
    fname = sprintf('220308_naive_Geo_dknown_n%d_K%d_R%d',n,K,R);
    save(fname,'Er_all');
    fprintf(fname)
    res_mean = mean(Er_all,1).*100;
    res_std = std(Er_all,0,1).*100;
    fprintf(': %0.2f (%0.2f) & %0.2f (%0.2f) & %0.2f (%0.2f) & %0.2f (%0.2f) & %0.2f (%0.2f)',...
        res_mean(1),res_std(1),res_mean(2),res_std(2),res_mean(3),res_std(3),res_mean(4),res_std(4),res_mean(5),res_std(5));
    fprintf('\n')

end