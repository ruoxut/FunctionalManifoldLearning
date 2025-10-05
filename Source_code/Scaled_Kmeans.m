function   [optid,purclus,adrclus,tnclus] =  Scaled_Kmeans(dataset,K,nstart,Trueid,p,q,tob)

% Author: Tung Pham, University of Melbourne
% Code to apply the clustering technique DHP with basis tob, as developed by Delaigle, Hall and Pham (2017).  Clustering functional data using projections.
% Clusters data set using projection functions
% The projection functions are linear combination of a basis.


% Arguments:
% dataset: an n times T matrix, n is the number of curves,
% T is the number of discrete points on each curve.

% K : The number of clusters
% nstart: number of starts for K-means clustering
% Trueid : The correct id of each clusters  

% p : The number of projection functions 
% q : The number of orthogonal functions used to approximate projection
% functions (called r in the paper)

% tob : type of basis;
% 1: Haar
% 2: DB2 for this basis, the data must be a matrix of n times 128.
% 3: PC


% Return:
% optid : the id obtained by our method.
% purclus: the purity value
% adrclus: the adjusted rand index.
% tnclus: the tightness of our clustered data.



%% Start program.

sizedat = size(dataset) ;

T = sizedat(2);

% Create basis functions (3 possible choices)


if (tob == 1)
    
    %create the first 2^powh Haar basis functions, such that 2^powh >=q. 
    hh= 1;
    powh=0;
	while(hh < q)
    	    hh=2*hh;
    	    powh=powh+1;
    	end
    powh=powh-1;
    
    haarbase = haar_basis(T,powh) ;
    basis = haarbase(:,1:q);
    
end

if (tob == 2)
    
    ldb2 = log2(T);
    
    if (rem(ldb2,1) ~= 0)
        
        disp('Error, the number of grids must be a power of 2.')
        return;
    end
    
    [wfun,xgrid] = wpfun('db2',q-1,ldb2);

    dgrid = [0:(T-1)]*3 + 2 ;

    basis = transpose(wfun(:,dgrid)) ;
    
end

if (tob == 3)
    
    basis = pca_basis(dataset,q);
    
end



%Rotation and Reflection matrices

alpha = pi/90 ;

rotmat = [cos(alpha), -sin(alpha); sin(alpha), cos(alpha)];  %Rotation matrix of angle alpha

refmat = [1,0;0,-1] ;  % Reflection matrix through axis x.


%Create all 2-element subsets of {1,2,...,q} 

combq2  = nchoosek(1:q,2) ;

combq22 = [combq2;combq2];

scombq2 = size(combq22) ;


% Initial setting of coefficients

hcoefs = zeros(q,p);          % create a zero matrix of coefficents

for i = 1: p 
    hcoefs(i,i) = 1  ;
end

opthcoefs = hcoefs;

optsumd = Inf;

optid = zeros(sizedat(1),0);

nloops =  scombq2(1);

datahp = dataset * basis ;  % Project data on  bases

%Start a loop to find the optimal coefficients of functions

for i = 1: nloops    
    
    [sbi] =  combq22(i,:);
    hcoefs = opthcoefs ;
    
    for v = 1: 180
           
        hcoefs(sbi,:) = rotmat * hcoefs(sbi,:) ;
    
        projdat = datahp * hcoefs;
    
        sprojdat = scalemat(projdat);
    
        [idat,clus, sumdvec] = kmeans(sprojdat,K, 'emptyaction', 'singleton','Replicates',nstart);

        sumd = sum(sumdvec);

        if(optsumd > sumd) 
        
            optsumd = sumd;
            opthcoefs = hcoefs;
            optid = idat;
            
        end
        
        refhcoefs = hcoefs;
        refhcoefs(sbi,:) = refmat * refhcoefs(sbi,:);
    
        projdat = datahp * refhcoefs ;
    
        sprojdat = scalemat(projdat);
          
        [idat,clus,sumdvec] = kmeans(sprojdat,K, 'emptyaction', 'singleton', 'Replicates',nstart);
        
        sumd = sum(sumdvec);

        if(optsumd > sumd) 
        
            optsumd = sumd;
            opthcoefs = refhcoefs;
            optid = idat;
            
        end 
        
    end
    
end

purclus = purity(Trueid,optid)/sizedat(1); % Compute the purity of our clustering method.

tnclus  = tightness(optid,dataset); % Compute the tightness of our clustering method .

adrclus = Adjusted_Rand(Trueid,optid); %Compute the adjusted rand index.

end 



function   [hnkx] =  haar_func_nk(x,n,k)

%find the value of Haar function 

hnkx = 2^(n/2) * haar_mother(2^n * x - k);

end


function   [hx] =  haar_mother(x)
%find the value of Haar mother function 

if (x >= 0 & x < 0.5) 
    hx = 1 ;
end

if (x >= 0.5 & x < 1)
    hx = -1;
end

if (x <0 | x >= 1)
    hx = 0;
end

end


function [hbs]  = haar_basis(T,hlevel)

% Produce Haar basis functions on grid of size T [0,(T-1)/T]
haarfunc = zeros(T,1);

hbs = zeros(T,0);

for n = 0:hlevel 
    
    for k = 0:(2^n-1)
    
        for t = 0:(T-1)
        
            haarfunc(t+1,1) = haar_func_nk(t/T,n,k);
            
        end
        
        hbs  = [hbs,haarfunc] ;
    end
end

hbs = [ones(T,1),hbs] ;

end

function[pcb] = pca_basis(dataset,q)

    meandat = mean(dataset);

    subdat = bsxfun(@minus,dataset,meandat);

    covdat  = transpose(subdat) * subdat ;
    
    covdat = 0.5 * (covdat + transpose(covdat));

    [eigvec,eigval]  = eig(covdat) ;
    
    sizedat = size(dataset);
    
    T = sizedat(2);
    
    lowq = T + 1 - q;

    pcb = eigvec(:,lowq : T);

end


function   [smat] = scalemat(mat)

% To scale a matrix by its columns

meanmat = mean(mat) ;

submat = bsxfun(@minus,mat,meanmat);

normcol = sqrt(sum(submat.^2,1));

smat = bsxfun(@rdivide,mat,normcol);

end

function   [tnd] = tightness(clusid,dat)

% to find the tightness of a data set given a set of clustering index. 

idset  = unique(clusid) ;

K = length(idset);

tnd = 0;

for k = 1:K
    
    idk =  find(clusid == idset(k)) ;

    datk = dat(idk,:) ;
    
    meank = mean(datk); 
    
    subdatk = bsxfun(@minus,datk,meank);
    
    subdatk = times(subdatk,subdatk);
    
    tnd = tnd + sum(sum(subdatk)); 
    
end

end



function   [purAB] = purity(A,B)

% To find the purity between two divisions A and B of the same set.
% A is the correct division
% B is the clustered division

setclus  = unique(B) ; % find the cluster name of each cluster.

K = length(setclus) ;

intmat = zeros(K,K) ;  % matrix of intersection between two divisions 

for k = 1:K
    
    clusterk =  find(B == setclus(k)) ;
    
    for j = 1 : K
        
        classj = find(A==setclus(j)) ;

        intmat(k,j) = length(intersect(clusterk,classj)) ;        
    end
end

purAB = 0;

for k = 1:K 
    purAB = max(intmat(k,:)) + purAB ;
end
end


function   [adrAB] = Adjusted_Rand(A,B)

% To find the purity between two divisions A and B of the same set.
% A is the correct division
% B is the clustered division

setclus  = unique(B) ; % find the cluster name of each cluster.

K = length(setclus) ;

intmat = zeros(K,K) ;  % matrix of intersection between two divisions 

for k = 1:K
    
    clusterk =  find(B == setclus(k)) ;
    
    for j = 1:K
        
        classj = find(A==setclus(j)) ;

        intmat(k,j) = length(intersect(clusterk,classj)) ;     
        
    end
end

adrAB = adi(intmat);

end

function [adr]= adi(indmat)

sumcol = sum(indmat,1);

sumrow = sum(indmat,2);

Nchoose2 = 0;

for k = 1 : length(sumrow)
    
    for j = 1 :length(sumcol)
        
        Nchoose2 = Nchoose2 + 0.5*(indmat(k,j)^2 - indmat(k,j)) ;
    
    end
end

Achoose2 = 0;

for k = 1: length(sumrow)

    Achoose2 = Achoose2 + 0.5*(sumrow(k)^2 - sumrow(k));
    
end

Bchoose2  = 0;
for j = 1: length(sumcol)
    
    Bchoose2 = Bchoose2 + 0.5*(sumcol(j)^2 -sumcol(j));
    
end
N = sum(sum(indmat));
N2 = 0.5*(N^2-N);

adrnomi = (Nchoose2 - (Achoose2 * Bchoose2)/N2);
adrdeno = 0.5*(Achoose2 + Bchoose2) - (Achoose2 * Bchoose2)/N2;

adr = adrnomi/adrdeno;

end

