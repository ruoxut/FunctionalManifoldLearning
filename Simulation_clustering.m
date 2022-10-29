%% The file reproduces the simulation of clustering in Tan et al. (2022+).

% Note:
% Proj_Kmeans.m and Scaled_Kmeans.m are downloaded from https://researchers.ms.unimelb.edu.au/~aurored/
% to perform the DHP method by Delaigle et al. (2019).
% rand_index.m is downloaded from https://github.com/cmccomb/rand_index to
% compute the adjusted Rand index.
% loclin.m performs the ridged local linear estimator and it is also
% available at https://github.com/ruoxut/loclin.

% References: 
% Tan, R., Zang, Y. and Yin, G. (2022+). Nonlinear dimension reduction for functional data with
% application to clustering. Statistica Sinica.
% Delaigle, A., Hall, P. and Pham, T. (2019). Clustering functional data into groups using
% projections. JRSS,B, 81, 271-304.

% Author: Ruoxu Tan; date: 2022/Oct/29; Matlab version: R2020a.

n_all = [300 500]; % Sample sizes
K_all = [13 15 18]; % K-nearest neighbours
p = 200; % Number of dense time points
t = linspace(0,1,p)'; % Time interval where we evaluate/estimate functional data
p_obs_all = [60 100]; % Numbers of observations per individual
n_rep = 100; % Number of simulation repetition
n_start = 20; % Number of replicates used in k-means clustering

for ind_1 = 1:length(n_all)

    n = n_all(ind_1);
    p_obs = p_obs_all(ind_1);
    t_obs = linspace(0,1,p_obs)'; % Time interval where we observe functional data

    n_1 = n/2;% Half of inviduals forms one group.

    D_FPTU_res = cell(n_rep,length(K_all)); % To store the resulting proximity graphs.
    D_FPTU_nonres = cell(n_rep,length(K_all));
    D_PTU_PCA_res = cell(n_rep,length(K_all));
    D_PTU_PCA_nonres = cell(n_rep,length(K_all));
    D_Isomap = cell(n_rep,length(K_all));

    G_FPTU_res = cell(n_rep,length(K_all));% To store the resulting group labels.
    G_FPTU_nonres = cell(n_rep,length(K_all));
    G_PTU_PCA_res = cell(n_rep,length(K_all));
    G_PTU_PCA_nonres = cell(n_rep,length(K_all));
    G_Isomap = cell(n_rep,length(K_all));
    G_DHP = cell(1,n_rep);
    G_PCA = cell(1,n_rep);
    G_st = cell(1,n_rep);

    ADI_FPTU_res = zeros(n_rep,length(K_all));% To store the resulting ADIs.
    ADI_FPTU_nonres = zeros(n_rep,length(K_all));
    ADI_PTU_PCA_res = zeros(n_rep,length(K_all));
    ADI_PTU_PCA_nonres = zeros(n_rep,length(K_all));
    ADI_Isomap = zeros(n_rep,length(K_all));
    ADI_DHP = zeros(n_rep,1);
    ADI_PCA = zeros(n_rep,1);
    ADI_st = zeros(n_rep,1);

    A_ADI = cell(1,8);

    R = 10; % Signal to noise ratio

    for n_rep_ind = 1:n_rep
        %% Generate data.
        rng(3*n_rep_ind)

        X_obs = zeros(p_obs,n);
        X = zeros(p,n);

        % Uncomment the corresponding part to select a particular model.
        % Model (1) -------------------------------------------------------
        %Z_1 = [random('Uniform',0,7,[n_1,1]) ; random('Uniform',7,10,[n-n_1,1])];
        %Z_2 = random('Uniform',0,4,[n,1]);
        %A = [Z_1(:,1).*cos(Z_1(:,1)) Z_1(:,1).*sin(Z_1(:,1))+Z_2(:,1) Z_2(:,1)];
 
        %for i = 1:n
        %    X_obs(:,i) = A(i,1).*sin(2.*pi.*t_obs) + A(i,2).*cos(2.*pi.*t_obs) +  A(i,3).*sin(4.*pi.*t_obs); 
        %end
        %------------------------------------------------------------------
        
        % Model (2) -------------------------------------------------------
        %Z_1 = [random('Uniform',-3*pi/2,0,[n_1,1]);random('Uniform',0,3*pi/2,[n-n_1,1])];
        %Z_2 = random('Uniform',1,4,[n,1]);
        %A = [sin(Z_1) sign(Z_1).*(cos(Z_1)-1) Z_2];

        %for i = 1:n
        %    X_obs(:,i) = A(i,1).*sin(2.*pi.*t_obs) + A(i,2).*cos(2.*pi.*t_obs) +  A(i,3).*sin(4.*pi.*t_obs); 
        %end
        %------------------------------------------------------------------

        % Model (3) -------------------------------------------------------
        A = [random('Uniform',-1,0,[n_1,1]) ; random('Uniform',0,1,[n-n_1,1])];
        B = random('Normal',1,0.1,[n,1]);
        mu = normpdf(t,0.2,0.08) + normpdf(t,0.5,0.1) + normpdf(t,0.8,0.13); 

        for i = 1:n
            if A(i) == 0
                gamma = t_obs;
            else
                gamma = (exp(A(i).*t_obs)-1)./(exp(A(i))-1);
            end
            X_obs(:,i) = abs(B(i)).*interp1(t,mu,gamma);
        end
        %------------------------------------------------------------------

        % Adding noise.
        mu_X = mean(X_obs,2);
        Cov_X = (X_obs-mu_X)*(X_obs-mu_X)'/n;
        Var_X = mean(diag(Cov_X));
        sigma = sqrt(Var_X/R);
        epsilon = random('Normal',0,sigma,[p_obs,n]);

        for i = 1:n
            X_obs(:,i) = X_obs(:,i) + epsilon(:,i);
        end
       
        % Presmoothing using the ridged local linear estimator.
        parfor i = 1:n
            [~,X(:,i),~]  = loclin( t_obs,X_obs(:,i),min(t_obs),max(t_obs) );
        end
       
       
       %% Clustering
        G = zeros(1,n); % Group labels  
        G(1,1:n_1) = 1;
        G(1,n_1+1:n) = 2;

        [X,G] = rmout(t,X,G,min(K_all),2);% Remove outliers.

        g = length(unique(G));

        %d_est = dim( t,X,0.9 ); % Intrinsic dimension estimated
        d_est = 2; % Intrinisc dimension known
       
        parfor ind_2 = 1:length(K_all)
            K = K_all(ind_2);

            % FPTU/FPU_PCA dimension reduction and graph clustering
            [ D_FPTU_res{n_rep_ind,ind_2},~ ] = FPTU( t,X,K,K,d_est,1 );
            [ X_d_FPTU_res,G_FPTU_res{n_rep_ind,ind_2},~ ] = graph_clustering( t,X,D_FPTU_res{n_rep_ind,ind_2},g,d_est,n_start );

            [ D_FPTU_nonres{n_rep_ind,ind_2},~ ] = FPTU( t,X,K,K,d_est,0 );
            [ X_d_FPTU_nres,G_FPTU_nonres{n_rep_ind,ind_2},~ ] = graph_clustering( t,X,D_FPTU_nonres{n_rep_ind,ind_2},g,d_est,n_start );

            [ D_PTU_PCA_res{n_rep_ind,ind_2} ] = PTU_PCA( t,X,K,K,d_est,1 );
            [ X_d_PTU_PCA_res,G_PTU_PCA_res{n_rep_ind,ind_2},~ ] = graph_clustering( t,X,D_PTU_PCA_res{n_rep_ind,ind_2},g,d_est,n_start );

            [ D_PTU_PCA_nonres{n_rep_ind,ind_2} ] = PTU_PCA( t,X,K,K,d_est,0 );
            [ X_d_PTU_PCA_nres,G_PTU_PCA_nonres{n_rep_ind,ind_2},~ ] = graph_clustering( t,X,D_PTU_PCA_nonres{n_rep_ind,ind_2},g,d_est,n_start );


            ADI_FPTU_res(n_rep_ind,ind_2) = rand_index(G,G_FPTU_res{n_rep_ind,ind_2},'adjusted'); 
            ADI_FPTU_nonres(n_rep_ind,ind_2) = rand_index(G,G_FPTU_nonres{n_rep_ind,ind_2},'adjusted'); 
            ADI_PTU_PCA_res(n_rep_ind,ind_2) = rand_index(G,G_PTU_PCA_res{n_rep_ind,ind_2},'adjusted'); 
            ADI_PTU_PCA_nonres(n_rep_ind,ind_2) = rand_index(G,G_PTU_PCA_nonres{n_rep_ind,ind_2},'adjusted'); 

            % Functional Isomap and graph clustering
            [ D_Isomap{n_rep_ind,ind_2},~ ] = FIsomap( t,X,K );
            [ X_d_FIsomap,G_Isomap{n_rep_ind,ind_2},~ ] = graph_clustering( t,X,D_Isomap{n_rep_ind,ind_2},g,d_est,n_start );
 
            ADI_Isomap(n_rep_ind,ind_2) = rand_index(G,G_Isomap{n_rep_ind,ind_2},'adjusted'); 

       end
       
       % DHP clustering
       [G_DHP{1,n_rep_ind},~,~,ADI_DHP(n_rep_ind,1)] = Proj_Kmeans(X',n_start,G,4,0.2,1);

       % FPCA and k-means clustering
       [X_d_PCA,~,~] = FPCA(t,X);

       G_PCA{1,n_rep_ind} = kmeans(X_d_PCA',g,'Replicates',n_start);

       ADI_PCA(n_rep_ind,1) = rand_index(G,G_PCA{1,n_rep_ind},'adjusted');

       % Standard k-means clustering
       G_st{1,n_rep_ind} = kmeans(X',g,'Replicates',n_start);

       ADI_st(n_rep_ind,1) = rand_index(G,G_st{1,n_rep_ind},'adjusted');

    end
    
    A_ADI{1,1} = mean(ADI_FPTU_res,1)*100;
    A_ADI{1,2} = mean(ADI_FPTU_nonres,1)*100;
    A_ADI{1,3} = mean(ADI_PTU_PCA_res,1)*100;
    A_ADI{1,4} = mean(ADI_PTU_PCA_nonres,1)*100;
    A_ADI{1,5} = mean(ADI_Isomap,1)*100;
    A_ADI{1,6} = mean(ADI_DHP,1)*100;
    A_ADI{1,7} = mean(ADI_PCA,1)*100;
    A_ADI{1,8} = mean(ADI_st,1)*100;
    
    % Store and print results.
    fname = sprintf('(3)n%dR%d_dtrue_220310',n,R);
    save(fname,'ADI_FPTU_res','ADI_FPTU_nonres','ADI_PTU_PCA_res','ADI_PTU_PCA_nonres','ADI_Isomap','ADI_DHP','ADI_PCA','ADI_st','A_ADI',...
        'G_FPTU_res','G_FPTU_nonres','G_PTU_PCA_res','G_PTU_PCA_nonres','G_Isomap','G_DHP','G_PCA','G_st','K_all');
    fprintf(fname);fprintf(': \n');
    fprintf('FPTU_res_K_%d, FPTU_res_K_%d, FPTU_res_K_%d, %0.2f & %0.2f & %0.2f \n', K_all(1),K_all(2),K_all(3),A_ADI{1,1});
    fprintf('FPTU_nonres_K_%d, FPTU_nonres_K_%d, FPTU_nonres_K_%d, %0.2f & %0.2f & %0.2f \n', K_all(1),K_all(2),K_all(3),A_ADI{1,2});
    fprintf('PTU_PCA_res_K_%d, PTU_PCA_res_K_%d, PTU_PCA_res_K_%d, %0.2f & %0.2f & %0.2f \n', K_all(1),K_all(2),K_all(3),A_ADI{1,3});
    fprintf('PTU_PCA_nonres_K_%d, PTU_PCA_nonres_K_%d, PTU_PCA_nonres_K_%d, %0.2f & %0.2f & %0.2f \n', K_all(1),K_all(2),K_all(3),A_ADI{1,4});
    fprintf('FIso_K_%d, FIso_K_%d, FIso_K_%d, %0.2f & %0.2f & %0.2f \n', K_all(1),K_all(2),K_all(3),A_ADI{1,5});
    fprintf('DHP, PCA, st, %0.2f & %0.2f & %0.2f \n',A_ADI{1,6},A_ADI{1,7},A_ADI{1,8});

end
