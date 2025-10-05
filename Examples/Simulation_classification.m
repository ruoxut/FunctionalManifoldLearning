%% The file reproduces the simulation results in Tan and Zang (2025+).

% References:
% Tan R. and Zang Y. (2025+). Supervised manifold learning for functional
% data. Journal of Computational Graphical Statistics.

% Matlab version: R2024b; last updated: 2025/May.

n_sample = [200 500]; % Training sample sizes
n_test = 500; % Test sample size
n_rep = 200; % Number of simulation repetitions
model_ind = 1; % Model choice, 1:5 corresponds to model (i) to (v) in the paper.
t_num = 200;  
t_obs_num_all = [50,100];
t = linspace(0,1,t_num); 
R = 20;

for n_sample_ind = 1:length(n_sample)
    n = n_sample(n_sample_ind);
    t_obs_num = t_obs_num_all(n_sample_ind);
    t_obs = linspace(0,1,t_obs_num);

    Err_k_NN = zeros(n_rep,1);
    Err_SVM = zeros(n_rep,1);
    Err_LDA = zeros(n_rep,1);
    Err_CC = zeros(n_rep,1);
    Err_FQDA = zeros(n_rep,1);
    Err_NB = zeros(n_rep,1);

    X_train = cell(n_rep,1);
    X_test = cell(n_rep,1);
    Y_train = cell(n_rep,1);
    Y_test = cell(n_rep,1);
    
    parfor n_rep_ind = 1:n_rep
        rng(3*n_rep_ind)

        if model_ind == 1 % Model (i)--------------------------------------
            Y_train{n_rep_ind} = random('Binomial',1,0.5,[n,1]);
            Y_test{n_rep_ind}  = random('Binomial',1,0.5,[n_test,1]);
            X_train_obs = zeros([n,t_obs_num]);
            X_test_obs = zeros([n_test,t_obs_num]);
            d = 2;

            Z_1 = zeros(n,1);
            Z_1(Y_train{n_rep_ind} == 0) = random('Uniform',-1,0.2,[sum(Y_train{n_rep_ind} == 0),1]);
            Z_1(Y_train{n_rep_ind} == 1) = random('Uniform',-0.2,1,[sum(Y_train{n_rep_ind} == 1),1]);
            Z_2 = random('Gamma',4,0.5,[n,1]);
            Z_1_test = zeros(n_test,1);
            Z_1_test(Y_test{n_rep_ind} == 0) = random('Uniform',-1,0.2,[sum(Y_test{n_rep_ind} == 0),1]);
            Z_1_test(Y_test{n_rep_ind} == 1) = random('Uniform',-0.2,1,[sum(Y_test{n_rep_ind} == 1),1]);
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
        %------------------------------------------------------------------

        elseif model_ind == 2 % Model (ii)---------------------------------
            Y_train{n_rep_ind} = random('Binomial',1,0.5,[n,1]);
            Y_test{n_rep_ind}  = random('Binomial',1,0.5,[n_test,1]);
            X_train_obs = zeros([n,t_obs_num]);
            X_test_obs = zeros([n_test,t_obs_num]);
            d = 2;
 
            Z_1 = random('Uniform',0,2.*pi,[n,1]);
            Z_2 = random('Uniform',0,8,[n,1]);

            Z_1_test = random('Uniform',0,2.*pi,[n_test,1]);
            Z_2_test = random('Uniform',0,8,[n_test,1]);
            
            for i = 1:n
                if Y_train{n_rep_ind}(i) == 0
                    X_train_obs(i,:) = Z_1(i).*cos(Z_1(i)).*sin(2.*pi.*t_obs) + Z_1(i).*sin(Z_1(i)).*cos(2.*pi.*t_obs) ...
                        + Z_2(i).*sin(4.*pi.*t_obs);
                else
                    X_train_obs(i,:) = Z_1(i).*cos(Z_1(i)+pi).*sin(2.*pi.*t_obs) + Z_1(i).*sin(Z_1(i)+pi).*cos(2.*pi.*t_obs) ...
                        + Z_2(i).*sin(4.*pi.*t_obs);
                end
            end

            for i = 1:n_test
                if Y_test{n_rep_ind}(i) == 0
                    X_test_obs(i,:) = Z_1_test(i).*cos(Z_1_test(i)).*sin(2.*pi.*t_obs) + Z_1_test(i).*sin(Z_1_test(i)).*cos(2.*pi.*t_obs) ...
                        + Z_2_test(i).*sin(4.*pi.*t_obs);
                else
                    X_test_obs(i,:) = Z_1_test(i).*cos(Z_1_test(i)+pi).*sin(2.*pi.*t_obs) + Z_1_test(i).*sin(Z_1_test(i)+pi).*cos(2.*pi.*t_obs) ...
                        + Z_2_test(i).*sin(4.*pi.*t_obs);
                end
            end
        %------------------------------------------------------------------

        elseif model_ind == 3 % Model (iii)--------------------------------
            theta = random('Uniform',0,2*pi,[n,1]);
            phi = random('Uniform',0,2*pi,[n,1]);  
            Y_train{n_rep_ind} = theta>=phi;
            d = 2;
            
            theta_test = random('Uniform',0,2*pi,[n_test,1]);
            phi_test = random('Uniform',0,2*pi,[n_test,1]);  
            Y_test{n_rep_ind} = theta_test>=phi_test;
            
            
            X_train_obs = (2+cos(theta)).*cos(phi).*sin(2.*pi.*t_obs) + (2+cos(theta)).*sin(phi).*cos(2.*pi.*t_obs) ...
                    + sin(theta)*sin(4.*pi.*t_obs);
            
            X_test_obs = (2+cos(theta_test)).*cos(phi_test).*sin(2.*pi.*t_obs) + (2+cos(theta_test)).*sin(phi_test).*cos(2.*pi.*t_obs) ...
                    + sin(theta_test)*sin(4.*pi.*t_obs);
        %------------------------------------------------------------------  

        elseif model_ind == 4 % Model (iv)---------------------------------
            Y_train{n_rep_ind} = random('Binomial',1,0.5,[n,1]);
            Y_test{n_rep_ind}  = random('Binomial',1,0.5,[n_test,1]);
            d = 3;

            xi_1 = zeros(n,1);xi_2 = zeros(n,1);xi_3 = zeros(n,1);
            xi_1(Y_train{n_rep_ind}==0) = random('Normal',-1,3/5,[sum(Y_train{n_rep_ind}==0),1]);
            xi_1(Y_train{n_rep_ind}==1) = random('Normal',-1/2,9/10,[sum(Y_train{n_rep_ind}==1),1]);
            xi_2(Y_train{n_rep_ind}==0) = random('Normal',2,2/5,[sum(Y_train{n_rep_ind}==0),1]);
            xi_2(Y_train{n_rep_ind}==1) = random('Normal',5/2,1/2,[sum(Y_train{n_rep_ind}==1),1]);
            xi_3(Y_train{n_rep_ind}==0) = random('Normal',-3,1/5,[sum(Y_train{n_rep_ind}==0),1]);
            xi_3(Y_train{n_rep_ind}==1) = random('Normal',-5/2,3/10,[sum(Y_train{n_rep_ind}==1),1]);
            X_train_obs = xi_1.*log(t_obs+2) + xi_2.*t_obs + xi_3.*t_obs.^3;
            
            xi_1_test = zeros(n_test,1);xi_2_test = zeros(n_test,1);xi_3_test = zeros(n_test,1);
            xi_1_test(Y_test{n_rep_ind}==0) = random('Normal',-1,3/5,[sum(Y_test{n_rep_ind}==0),1]);
            xi_1_test(Y_test{n_rep_ind}==1) = random('Normal',-1/2,9/10,[sum(Y_test{n_rep_ind}==1),1]);
            xi_2_test(Y_test{n_rep_ind}==0) = random('Normal',2,2/5,[sum(Y_test{n_rep_ind}==0),1]);
            xi_2_test(Y_test{n_rep_ind}==1) = random('Normal',5/2,1/2,[sum(Y_test{n_rep_ind}==1),1]);
            xi_3_test(Y_test{n_rep_ind}==0) = random('Normal',-3,1/5,[sum(Y_test{n_rep_ind}==0),1]);
            xi_3_test(Y_test{n_rep_ind}==1) = random('Normal',-5/2,3/10,[sum(Y_test{n_rep_ind}==1),1]);
            X_test_obs = xi_1_test.*log(t_obs+2) + xi_2_test.*t_obs + xi_3_test.*t_obs.^3;
        %------------------------------------------------------------------
        
        elseif model_ind == 5 % Model (v)---------------------------------
            Y_train{n_rep_ind} = random('Binomial',1,0.5,[n,1]);
            Y_test{n_rep_ind}  = random('Binomial',1,0.5,[n_test,1]);
            d = 50;
            phi = zeros(d,length(t_obs));
            phi(1,:) = ones(1,length(t_obs));
            for j = 2:d
                if mod(j,2) == 1
                   phi(j,:) = sqrt(2).*sin((j-1).*pi.*t_obs);
                else
                   phi(j,:) = sqrt(2).*cos(j.*pi.*t_obs);
                end
            end

            for i = 1:n
                if Y_train{n_rep_ind}(i) == 0
                    sigma = exp(-(1:d)./6);
                    X_train_obs(i,:) = random('Normal',0,1,[1,d]).*sigma*phi;
                else
                    sigma = exp(-(1:d)./4);
                    X_train_obs(i,:) = t_obs + random('Normal',0,1,[1,d]).*sigma*phi;
                end
            end

            for i = 1:n_test
                if Y_test{n_rep_ind}(i) == 0
                    sigma = exp(-(1:d)./6);
                    X_test_obs(i,:) = random('Normal',0,1,[1,d]).*sigma*phi;
                else
                    sigma = exp(-(1:d)./4);
                    X_test_obs(i,:) = t_obs + random('Normal',0,1,[1,d]).*sigma*phi;
                end
            end
        %------------------------------------------------------------------
        else
            error('The model does not exist.')
        end

        % Adding noise and smoothing.
        mu_X = mean(X_test_obs,1);
        Cov_X = (X_test_obs-mu_X)'*(X_test_obs-mu_X)/n;
        Var_X = mean(diag(Cov_X));
        sigma = sqrt(Var_X/R);
        epsilon = random('Normal',0,sigma,[n+n_test,t_obs_num]);

        X_train{n_rep_ind} = zeros([n,length(t)]);
        X_test{n_rep_ind} = zeros([n_test,length(t)]);

        for i = 1:n
            X_train_obs(i,:) = X_train_obs(i,:) + epsilon(i,:);
            [~,X_train{n_rep_ind}(i,:),~] = loclin(t_obs,X_train_obs(i,:),min(t),max(t));
        end

        for i = 1:n_test
            X_test_obs(i,:) = X_test_obs(i,:) + epsilon(i+n,:);
            [~,X_test{n_rep_ind}(i,:),~] = loclin(t_obs,X_test_obs(i,:),min(t),max(t));
        end        

        % Prespecified parameters.
        delta = 0.6;
        K_pca = 15;
        opt_FPTU = 0;
        if model_ind == 5
            d = dim(t,X_train{n_rep_ind},0.8);
            if d > K_pca
                K_pca = d+1;
            end
        end

        % FSML+kNN
        method = 'k_NN'; para_method = 20;
        [xi_CV,h_reg_CV,~] = CV_xih(t,X_train{n_rep_ind},Y_train{n_rep_ind},'delta',delta,...
            'K_pca',K_pca,'d',d,'opt_FPTU',opt_FPTU,'method',method,'para_method',para_method);
        [Z_train_est,~] = FSML(t,X_train{n_rep_ind},Y_train{n_rep_ind},xi_CV,'delta',delta,'K_pca',K_pca,'d',d,'opt_FPTU',opt_FPTU);
        Z_test = zeros(n_test,d);
        for j = 1:n_test
            Z_test(j,:) = FLLE(t,X_train{n_rep_ind},Z_train_est,X_test{n_rep_ind}(j,:),K_pca,h_reg_CV);
        end
        Clsfy = fitcknn(Z_train_est,Y_train{n_rep_ind},'NumNeighbors',para_method);
        Y_test_est_k_NN = predict(Clsfy,Z_test);
        Err_k_NN(n_rep_ind) = sum(Y_test_est_k_NN~=Y_test{n_rep_ind})/n_test;

        % FSML+SVM
        method = 'SVM'; 
        [xi_CV,h_reg_CV,~] = CV_xih(t,X_train{n_rep_ind},Y_train{n_rep_ind},'delta',delta,...
            'K_pca',K_pca,'d',d,'opt_FPTU',opt_FPTU,'method',method);
        [Z_train_est,~] = FSML(t,X_train{n_rep_ind},Y_train{n_rep_ind},xi_CV,'delta',delta,'K_pca',K_pca,'d',d,'opt_FPTU',opt_FPTU);
        Z_test = zeros(n_test,d);
        for j = 1:n_test
            Z_test(j,:) = FLLE(t,X_train{n_rep_ind},Z_train_est,X_test{n_rep_ind}(j,:),K_pca,h_reg_CV);
        end
        Clsfy = fitcecoc(Z_train_est,Y_train{n_rep_ind});
        Y_test_est_SVM = predict(Clsfy,Z_test);
        Err_SVM(n_rep_ind) = sum(Y_test_est_SVM~=Y_test{n_rep_ind})/n_test;
 
        % FSML+LDA
        method = 'LDA';
        [xi_CV,h_reg_CV,~] = CV_xih(t,X_train{n_rep_ind},Y_train{n_rep_ind},'delta',delta,...
            'K_pca',K_pca,'d',d,'opt_FPTU',opt_FPTU,'method',method);
        [Z_train_est,~] = FSML(t,X_train{n_rep_ind},Y_train{n_rep_ind},xi_CV,'delta',delta,'K_pca',K_pca,'d',d,'opt_FPTU',opt_FPTU);
        Z_test = zeros(n_test,d);
        for j = 1:n_test
            Z_test(j,:) = FLLE(t,X_train{n_rep_ind},Z_train_est,X_test{n_rep_ind}(j,:),K_pca,h_reg_CV);
        end
        Clsfy = fitcecoc(Z_train_est,Y_train{n_rep_ind},'Learners','discriminant');
        Y_test_est_LDA = predict(Clsfy,Z_test);
        Err_LDA(n_rep_ind) = sum(Y_test_est_LDA~=Y_test{n_rep_ind})/n_test;
        
        % Centroid Classifier by Delaigle and Hall (2012)
        Y_test_est_CC = zeros(n_test,1);
        for j = 1:n_test
            Y_test_est_CC(j) = CC(t,X_train{n_rep_ind},Y_train{n_rep_ind},X_test{n_rep_ind}(j,:));
        end
        Err_CC(n_rep_ind) = sum(Y_test_est_CC~=Y_test{n_rep_ind})/n_test;
        
        % Functional quadratic discriminant analysis by Delaigle and Hall (2013)
        Y_test_est_FQDA = zeros(n_test,1);
        for j = 1:n_test
            Y_test_est_FQDA(j) = FQDA(t,X_train{n_rep_ind},Y_train{n_rep_ind},X_test{n_rep_ind}(j,:));
        end
        Err_FQDA(n_rep_ind) = sum(Y_test_est_FQDA~=Y_test{n_rep_ind})/n_test;

        % Nonparametric Bayesian Classifier by Dai et al. (2017)
        Y_test_est_NB = zeros(n_test,1);
        for j = 1:n_test
            Y_test_est_NB(j) = NB(t,X_train{n_rep_ind},Y_train{n_rep_ind},X_test{n_rep_ind}(j,:));
        end
        Err_NB(n_rep_ind) = sum(Y_test_est_NB~=Y_test{n_rep_ind})/n_test;       
    end
    delete(gcp("nocreate"));

    % Store and print results.
    fname = sprintf('(%d)n%d_FSML_20250507',model_ind,n);
    save(fname, 'Err_k_NN', 'Err_SVM', 'Err_LDA', 'Err_CC', 'Err_FQDA', 'Err_NB',...
        'X_train', 'X_test', 'Y_train', 'Y_test')  
    fprintf(fname);fprintf(': \n');
    fprintf('Err_k_NN: %0.1f (%0.1f) \n',100*mean(Err_k_NN),100*std(Err_k_NN))
    fprintf('Err_SVM: %0.1f (%0.1f) \n',100*mean(Err_SVM),100*std(Err_SVM))
    fprintf('Err_LDA: %0.1f (%0.1f) \n',100*mean(Err_LDA),100*std(Err_LDA))
    fprintf('Err_CC: %0.1f (%0.1f) \n',100*mean(Err_CC),100*std(Err_CC))
    fprintf('Err_FQDA: %0.1f (%0.1f) \n',100*mean(Err_FQDA),100*std(Err_FQDA))
    fprintf('Err_NB: %0.1f (%0.1f) \n',100*mean(Err_NB),100*std(Err_NB))  

end
