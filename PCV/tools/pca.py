from PIL import Image
from numpy import *


def pca(X):
    """    Principal Component Analysis
        input: X, matrix with training data stored as flattened arrays in rows
        return: projection matrix (with important dimensions first), variance and mean.
    """
    
    # get dimensions
    num_data,dim = X.shape
    
    # center data
    mean_X = X.mean(axis=0)
    X = X - mean_X
    
    if dim>num_data:
        # PCA - compact trick used
        M = dot(X,X.T) # covariance matrix
        e,EV = linalg.eigh(M) # eigenvalues and eigenvectors
        tmp = dot(X.T,EV).T # this is the compact trick
        V = tmp[::-1] # reverse since last eigenvectors are the ones we want
        S = sqrt(e)[::-1] # reverse since eigenvalues are in increasing order
        for i in range(V.shape[1]):
            V[:,i] /= S
    else:
        # PCA - SVD used
        U,S,V = linalg.svd(X)
        V = V[:num_data] # only makes sense to return the first num_data
    
    # return the projection matrix, the variance and the mean
    return V,S,mean_X


def center(X):
    """    Center the square matrix X (subtract col and row means). """
    
    n,m = X.shape
    if n != m:
        raise Exception('Matrix is not square.')
    
    colsum = X.sum(axis=0) / n
    rowsum = X.sum(axis=1) / n
    totalsum = X.sum() / (n**2)
    
    #center
    Y = array([[ X[i,j]-rowsum[i]-colsum[j]+totalsum for i in range(n) ] for j in range(n)])
    
    return Y