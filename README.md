# Functional Manifold Learning

The rep contains the code to implement the functional manifold learning approaches proposed in the following papers:
    
1. Tan, R., Zang, Y. and Yin, G. (2024). Nonlinear dimension reduction for functional data with application to clustering. Statistica Sinica, 34, 1391-1412. 
2. Tan R. and Zang Y. (2025+). Supervised manifold learning for functional data. Journal of Computational and Graphical Statistics.

Please cite the papers above if you use the code.

## Simulation reproduction and simple examples 
The `Examples` folder contains a few examples of using the code. Specifically,
1. `Simulation_clustering.m` and `Geodesic_estimation.m` reproduce the simulations of clustering and geodesic distance estimation, respectively, in Tan et al. (2024);
2. `Simulation_classification.m` reproduces the simulation of classification in Tan and Zang (2025);
3. `Example_clustering.m` and `Example_classification.m` are short examples. 


## How to use our approaches on your data
### Clustering
Suppose you have the data objects `t,X` and `Y`, where `t` is an 1\*p vector representing the time interval, `X` is an n\*p data matrix where each row contains function values of an individual.

1. `[D_FPTU,~] = FPTU_adj_knn(t,X,K,K,d,opt)` performs the functional parallel transport unfolding, where `K` is the number of nearest neighbours, `d` is the intrinsic dimension, and `opt=1` or `0` means the rescaling of FPTU or not. The output `D_FPTU` contains estimated geodesic distances.  
2. `[X_d,G,~] = graph_clustering(t,X,D,g,d,n_start)` performs the graph-based clustering using the proximity graph `D`, where `g` is the number of clusters, and `n_start` is the number of replicates used in the $k$-means clustering with 20 as default. The output `X_d` contains the MDS low-$d$ representations, and `G` is the estimated cluster labels.

See `Example_clustering.m` for a concrete example.

### Classification
Suppose you have the data objects `t,X` and `Y`, where `t` is an 1\*p vector representing the time interval, `X` is an n\*p data matrix where each row contains function values of an individual, and `Y` is an n\*1 vector of classes. Here,  `X` and `Y` are the training data, and you want to predict the label of `x`, an 1\*p observation.

1. `[xi_CV,h_reg_CV,~] = CV_xih(t,X,Y)` selects the $\xi$ and $h_{\rm{reg}}$ by the nested CV. 
2. `[Z_d,~] = FSML(t,X,Y,xi_CV)` produces the manifold learning outcomes $Z_i$.
3. `[Z_x] = FLLE(t,X,Z_d,x,K_pca,h_reg_CV)` estimates $E(Z|X)$ at `x` based on `X` and `Z_d`, where `K_pca` is number of nearest neighbours used in local PCA, say 10 or 15.
4. A multivariate classifier can be trained using `Z_d` and `Y`, which induces a functional classifier that predicting the label of `x` by plugging `Z_x`.

User-specified parameters such as `delta`, `d`, `method`, etc, can be specified as inputs in `CV_xih` and `FSML`; see the original files for more details. See `Example_classification.m` for a concrete example.
