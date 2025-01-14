#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Functions to construct manifold learning.
Diffusion maps

@author: Zheming Gao
"""

import numpy as np
import scipy as sp
from sklearn.metrics import pairwise_distances

def gaussian_kernel_matrix_knrst(D, k_nearest, base = np.e):
    """
    Computes the Gaussian kernel matrix for a given data matrix D.
    
    Parameters:
    - D (numpy.ndarray): Data matrix where each column is a data point.
    - k_nearest: 
        Calculate sigma_i as the median of Euclidean distances 
                between x_i to its k nearest neighbors.
                
                sigma_{ij} = sigma_i * sigma_j.
    
    
    Returns:
    - K (numpy.ndarray): The Gaussian kernel matrix.
    """
   
    # Compute pairwise distances (D is #features by #samples)
    pairwise_dist = pairwise_distances(D.T)
    
    # Compute the sigma 
    # distances to p * n-th nearest neighbor are used. 
    # Default value is p = .01
    # Determine the epsilon value
    
    n = pairwise_dist.shape[0]
    # n_neighbors = k_nearest
    
    sig = np.zeros(n)
    for i in range(n):
        cand_arr = pairwise_dist[:,i]
        cand_arr[i] = np.inf # make the ith element big enough.
        # Get the indices of the sorted array
        sorted_indices = np.argsort(cand_arr)

        # Get the indices of the k smallest elements
        smallest_indices = sorted_indices[:n]

        # Get the first k smallest elements using these indices
        smallest_elements = cand_arr[smallest_indices]
        
        # The first must be zero. Disgard the first item.
        sig[i] = np.median(smallest_elements)
    
    # sigma = np.median(np.sort(pairwise_dist, axis=1)[:, n_neighbors])
    sig_mat = np.outer(sig, sig)
    
    # Compute the Gaussian kernel matrix
    K = base ** (- pairwise_dist ** 2 / sig_mat)
    
    return K

def diff_map(dataX, k_nn, t = 2, l = 3):
    """
    input:  dataX --- data matrix (p by n), each column a point.
            t --- parameter for transition matrix.
            l --- number of dimensions. (1 <= l <= n-1)

            
    output:
        diff_coor --- matrix after diffusion map \Psi_t(X): n by l
        diff_dis --- diffusion distance matrix.
        Phi, Sig, Psi --- SVD results.
        Pmat --- the P matrix 
        
    """
    Kmat = gaussian_kernel_matrix_knrst(dataX.T, k_nn)
    # Dmat = np.diag(np.sum(Kmat, axis = 1))
    # Pmat = np.linalg.inv(Dmat) @ Kmat
    invD = np.diag(1 / np.sum(Kmat, axis = 1))
    Pmat = invD @ Kmat
    Pmat_t = np.linalg.matrix_power(Pmat, t)  # matrix power P^t
        
    # SVD of P.
    # Sig is the array of singular values.
    Phi, Sig, PsiT = sp.linalg.svd(Pmat)
    Psi = PsiT.T
    
    # diff_map matrix (n by l) with power t.
    diff_coor = Psi[:,1:l+1] @ np.diag(Sig[1:l+1] ** t)
    phi_1 = Phi[:,0]  # obtain denominator
    n = Pmat.shape[0]
    diff_dis = np.zeros(Pmat.shape)
    for i in range(n):
        for j in range(i, n):
            
            # temp_sum = 0
            # for k in range(n):
            #     diff_pt = Pmat_t[i,k] - Pmat_t[j,k] 
            #     temp_sum = temp_sum + diff_pt ** 2 / np.abs(phi_1[k])
            # diff_dis[i,j] = np.sqrt(temp_sum)
            diff_dis[i,j] = np.linalg.norm(diff_coor[i,:] - diff_coor[j,:])
            diff_dis[j,i] = diff_dis[i,j]
    return diff_coor, diff_dis

def Gauss_weight_diff_knn (dataX, k_nrst, base = 1.5, t = 8, l = 3):
    """
    Calculating Gaussian weights between x^i and x^j  
        w_ij = exp (|| x^i - x^j ||^2 / sigma_ij)
    where sigma_ij = sigma_i * sigma_j,  sigma_i is the 
    median diffusion distance between x^i and its k nearest neighbors.


    input:  dataX --- data matrix (p by n), each column a point.
            k_nrst --- k nearest element for choosing sigma
            t --- parameter for transition matrix.
            l --- number of dimensions. (1 <= l <= n-1)            
    output:
        weights --- weights matrix.
    """
    
    # Calculate diffusion distance based on dataX, t and l.
    _, diff_dis = diff_map(dataX, k_nrst, t, l)
    
    # find sigma_i for x^i.
    n = dataX.shape[1]
    sig = np.zeros(n)
    for i in range(n):
        cand_arr = diff_dis[:,i]
        cand_arr[i] = np.inf # make the ith element big enough.
        # Get the indices of the sorted array
        sorted_indices = np.argsort(cand_arr)

        # Get the indices of the k smallest elements
        smallest_indices = sorted_indices[:k_nrst]

        # Get the first k smallest elements using these indices
        smallest_elements = cand_arr[smallest_indices]
        
        # The first must be zero. Disgard the first item.
        sig[i] = np.median(smallest_elements)
    
    # # Standardize sig
    # sig = sig / np.max(sig)
    weights = np.zeros((n,n))
    # for i in range(n):
    #     for j in range(i,n):
    #         weights[i,j] = np.exp(- np.linalg.norm(dataX[:,i] - dataX[:,j]) ** 2 / (sig[i] * sig[j]) )
    #         weights[j,i] = weights[i,j]
    
    for i in range(n):
        for j in range(n):
            # weights[i,j] = np.exp(- np.linalg.norm(dataX[:,i] - dataX[:,j]) ** 2 / (sig[i] * sig[j]) )
            weights[i,j] = base ** (- np.linalg.norm(dataX[:,i] - dataX[:,j]) ** 2 / (sig[i] * sig[j]) )
            
    return weights



#%% Code for implementing diffusion map with pyDiffMap package.

# import pydiffmap
# #### To initialize a diffusion map object:
# # mydmap = diffusion_map.DiffusionMap.from_sklearn(n_evecs = 1, epsilon = 1.0, alpha = 0.5, k=64)
# """
# where n_evecs is the number of eigenvectors that are computed, epsilon is a 
# scale parameter used to rescale distances between data points, alpha is a 
# normalization parameter (typically between 0.0 and 1.0) that influences the 
# effect of the sampling density, and k is the number of nearest neighbors 
# considered when the kernel is computed. A larger k means increased accuracy 
# but larger computation time. The from_sklearn command is used because we are 
# constructing using the scikit-learn nearest neighbor framework. For additional 
# optional arguments, see documentation.
# """
# # or just simply 

# mydmap = pydiffmap.diffusion_map.DiffusionMap.from_sklearn(n_evecs = 1, alpha = 0.5, epsilon = 'bgh', k = k_nrst)
# # To fit to a dataset X (array-like, shape (n_query, n_features)):
# mydmap.fit(matrixData.T)
# # The diffusion map coordinates can also be accessed directly via:
# dmap = mydmap.fit_transform(matrixData.T)
# # This returns an array dmap with shape (n_query, n_evecs). E.g. dmap[:,0] is the first diffusion coordinate evaluated on the data X.

# #In order to compute diffusion coordinates at the out of sample location(s) Y:
# dmap_Y = mydmap.transform(matrixData.T)
