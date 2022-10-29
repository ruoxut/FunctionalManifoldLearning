function [clusid,nopf,puty,adri] = Proj_Kmeans(dataset,nstart,Trueid,q,rho,tob)

% Author: Tung Pham, The University of Melbourne 
% Code to apply the clustering technique DHP with basis tob, as developed by Delaigle, Hall and Pham (2017).  Clustering functional data using projections.
% This is the main code. It uses functions defined in the file Scaled_Kmeans.m


% Arguments:
% dataset:  n times T matrix, where n is the number of curves in the sample and 
% T is the number of equally spaced points in the time grid where the curve is observed.

% nstart: number of random starts in K-means clustering. Suggested: nstart=30. Take nstart larger if results are not stable.
% Trueid: true clutser membership (1 or 2) of the data with the individuals in the same order as in the dataset matrix. 
%    Is required in simulations for computing purities and adjusted rand index. 
%    If you do no have Trueids, for example in a real data application, just input a vector of size n and ignore the adri and puty outputs.

% q: the number of basis functions to consider in the construction (called r in the paper). 
% rho: value of rho.
% tob: type of basis
%      1: Haar basis, with q<=32 (for q higher, you need to change the code Scaled_Kmeans, but we do not advise doing this as it will take long to run)
%      2: DB2 basis, the data set must be a matrix of n times 128, and q = 16.
%      3: PC basis

% Return:
% clusid : assigned cluster membership (1 or 2) of the data with the individuals in the same order as in the dataset matrix
% nopf: number of projection functions
% adri: adjusted rand index (in simulations, when comparing with Trueid).
% puty: purity  (in simulations, when comparing with Trueid).
% If you do no have Trueids, for example in a real data application, ignore the adri and puty outputs.


% K : number of clusters. We always apply binary clustering in our case (we use hierarchical clustering when K>2)
K=2;

%Apply the algorithm for p=1 and p=2
[optid1,purclus1,adrclus1,tnclus1] = Scaled_Kmeans(dataset,K,nstart,Trueid,1,q,tob);
[optid2,purclus2,adrclus2,tnclus2] = Scaled_Kmeans(dataset,K,nstart,Trueid,2,q,tob);

%classes (=id) obtained for p=1 and for p=2
optidset = [optid1,optid2];

%purities obtained for p=1 and for p=2
purset   = [purclus1,purclus2];

%arand index obtained for p=1 and for p=2
adrset   = [adrclus1,adrclus2];

%tighness obtained for p=1 and for p=2
tnset    = [tnclus1,tnclus2];


%tightness for p=1
T1 = tnset(1);

nopf = 1;

finished = 0;

%consider increasing values of p recursively
while(finished == 0)
    
    %tighntess of previous value of p    
    a = tnset(nopf);
    
    %tighntess of current value of p
    b = tnset(nopf+1);
    
    %Stop increasing p if criterion satisfied
    if( (a-b) <= (T1 * rho)) 
            finished = 1;
    end
    
    %otherwise, keep increasing p until criterion satisfied
    if (finished == 0) 

        %Apply the algorithm recusrisvely for p>2
    
        nopf = nopf + 1;
        
        [optidp,purclusp,adrclusp,tnclusp] = Scaled_Kmeans(dataset,K,nstart,Trueid,nopf+1,q,tob);
        
        optidset =  [optidset,optidp];
        purset   =  [purset,purclusp];
        adrset   =  [adrset,adrclusp];
        tnset    =  [tnset,tnclusp];
    
    end
    
    %Stop at some point if the criterion is never satisfied
    
    if (nopf >= 10 ) 
    
        finished = 1;
       
    end
     
    
end

clusid = optidset(:,nopf);
puty   = purset(:,nopf);
adri   = adrset(:,nopf);

end


