#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
BIC for selecting gamma.


"""

import numpy as np
import admm_cc
import CC_split_algo
import CC_weights_graphs
import FBS_algo
import funcs
import scipy as sp


def compute_bic(X, labels, cluster_centers):
    """
    Compute BIC (Bayesian Information Criterion) for a clustering model.

    Parameters:
        X (np.ndarray): Data matrix (p, n), where p = dimension, n = number of data points.
        labels (np.ndarray): Cluster labels assigned by the model (1D array of length n).
        cluster_centers (np.ndarray): Cluster center assignments (p, n), 
                                      where each column corresponds to its assigned cluster center.

    Returns:
        float: BIC score
    """
    p, n = X.shape  # Get dimensions
    k = len(np.unique(labels))  # Number of clusters

    # Compute log-likelihood assuming Gaussian clusters
    log_likelihood = 0
    # for i in range(k):
    for ii in labels:
        # Find points in cluster i
        cluster_points = X[:, labels == ii]  # (p, num_points_in_cluster)
        
        if cluster_points.shape[1] == 0:  # If cluster is empty, skip
            continue
        
        # Compute variance of cluster (assuming diagonal covariance)
        variance = np.mean(np.sum((cluster_points - cluster_centers[:, labels == ii])**2, axis=0)) / p
        
        if variance == 0:  # Avoid log(0) issue
            variance = 1e-10
        
        cluster_log_likelihood = -0.5 * cluster_points.shape[1] * (
            p * np.log(2 * np.pi * variance) + p
        )
        log_likelihood += cluster_log_likelihood

    # Compute number of free parameters
    num_params = k * p + k + k - 1  # Means + variances + cluster weights

    # Compute BIC
    bic = -2 * log_likelihood + num_params * np.log(n)

    return bic

# # Example Usage
# p, n = 2, 100  # Dimension = 2, 100 data points
# X = np.random.rand(p, n)  # Example dataset

# # Run K-Means
# kmeans = KMeans(n_clusters=3, random_state=42).fit(X.T)

# # Generate cluster centers in the same shape as X
# cluster_centers = np.zeros_like(X)
# for i in range(n):  
#     cluster_centers[:, i] = kmeans.cluster_centers_[kmeans.labels_[i]]  # Assign cluster centers

# # Compute BIC
# bic_value = compute_bic(X, kmeans.labels_, cluster_centers)
# print(f"BIC Score: {bic_value}")


def compute_ebic(X, labels, U):
    """
    Compute eBIC (extended Bayesian Information Criterion) for a clustering model.

    Chi et al. Provable Convex Co-clustering of Tensors, 2020, JMLR
    
    eBIC(gamma) = n*log(||X - U||_F^2 / n) + 2 * DoF * log(n).
    
    where X and U are p by n matrices. Dof is the degree of freedom, which is 
    estimated by number of clusters in U.
    
    Inputs:
        X (np.ndarray): Data matrix (p, n), where p = dimension, n = number of data points.
        labels (np.ndarray): Cluster labels assigned by the model (1D array of length n).
        U (np.ndarray): Cluster center assignments (p, n), 
                                     where each column corresponds to its assigned cluster center.

    Returns:
        float: eBIC score
    """
    p, n = X.shape  # Get dimensions
    k = len(np.unique(labels))  # Number of clusters

    RSS = np.linalg.norm(X - U, 'fro') ** 2
    ebic = n * np.log(RSS / n) + 2 * k * np.log(n)
        
    return ebic 



def CC_bic(model_name, X, D, weights_vec, gamma_cand, max_iter_admm = 200):
    """
    Performs validation to select the regularization parameter.
    convex clustering models.
    
    
    Parameters:
        model_name (string): 'CC ' or 'GME'.
        X (np.ndarray): Data matrix
        k_nrst: # of nearest neighbors for building graph.
        gamma (list or np.ndarray): Regularization parameter sequence
        fraction (float): Fraction of entries to hold out for validation
        repeat: repeat this procedure, for each gamma. (Statistically meaningful)
    Returns:
        dict: Contains validation results, selected parameters, and clustering results.
        gamma_opt: Best gamma with smallest validation error.
    """
    
    
    p, n = X.shape  # Get matrix dimensions
    nGamma = len(gamma_cand)
    validation_bic = np.zeros(nGamma)
    K_bic = np.zeros(nGamma)
    # Precalculate Dw, B, theta_B for GME model.
    # C = eye(D.shape[0]).toarray() @ np.diag(weights_vec) # Let matrix C be idenity.
    Dw = np.diag(weights_vec) @ D
    sig1_Dw = np.linalg.norm(Dw, 2)  # Calculate largest singular value of Dw.
    gamma_up = 1 / sig1_Dw ** 2
    tol_sim_path = 1e-06
    U_old = X # Initiate U
    
    ## Form full-size incidence matrix and weights vector for CC.
    # weights_full = CC_split_algo.recover_full_weights(D, weights_vec)
    # D_full, _, edge_weights_full = CC_split_algo.compactify_edges(weights_full, n, method='admm')
    
    
    # Loop over gamma values
    for ig in range(nGamma):
        gamma = gamma_cand[ig]
        # Solve using the cobra_pod method (needs implementation)
        if model_name == 'CC':
            
            
            Lambda = np.zeros((p, D.shape[0]))
            sol = CC_split_algo.cvxclust_admm(X, Lambda, D, weights_vec, gamma = gamma, nu=1,
                                max_iter=1000, tol_abs=1e-3, tol_rel=1e-3,
                                norm_type=2, accelerate=True)
            U_opt = sol['U']
            V_opt = sol['V']
            # Compute labels
            ##########################----####----####
            # DU = D @ U_opt.T  # Calculate matrix V = DU
            # Compute the Frobenius norm for each column in DU
            # differences = np.linalg.norm(DU.T, axis=0)
            differences = np.linalg.norm(V_opt, axis=0)
            # Find indices where differences are zero, indicating connected nodes
            connected_ix = np.where(differences <= tol_sim_path)[0]
    
            # Initialize a sparse adjacency matrix
            AA = sp.sparse.lil_matrix((n, n), dtype=int)
    
            for kk in list(connected_ix):
                # find index pair non-zeros in kk th row of D matrix
                ii, jj = np.where(D[kk,:] != 0)[0]    
                # Build A matrix
                AA[ii, jj] = 1
                AA[jj, ii] = 1
            # Find clusters
            rrr = funcs.find_clusters(AA)
            uni_clusters = rrr['cluster']  # indices of clusters for columns in U
            
            uni_clusters_id = np.unique(uni_clusters)
            K_bic[ig] = len(uni_clusters_id)  # record the number of clusters
            
            
        if model_name == 'GME':
            
            if gamma <= gamma_up:
                theta = 1
                B = Dw
            if gamma > gamma_up:
                theta = 1 / (np.sqrt(gamma) * sig1_Dw)
                B = theta * Dw   # Assign B = theta * Dw to satisfy the convexity condition.
            # GME3-CC model  #### Use warm start U0 = U_old ###
            U_opt, _, _, _, _ = FBS_algo.fbs_gme3_cc_mat(B, Dw, theta, sig1_Dw, gamma, U_old, X, max_iter_admm = max_iter_admm)
            # Compute labels
            ##########################----####----####
            DU = Dw @ U_opt.T  # Calculate matrix V = DwU
            # Compute the Frobenius norm for each column in DU
            differences = np.linalg.norm(DU.T, axis=0)
            # Find indices where differences are zero, indicating connected nodes
            connected_ix = np.where(differences <= tol_sim_path)[0]
    
            # Initialize a sparse adjacency matrix
            AA = sp.sparse.lil_matrix((n, n), dtype=int)
    
            for kk in list(connected_ix):
                # find index pair non-zeros in kk th row of D matrix
                ii, jj = np.where(Dw[kk,:] != 0)[0]    
                # Build A matrix
                AA[ii, jj] = 1
                AA[jj, ii] = 1
            # Find clusters
            rrr = funcs.find_clusters(AA)
            uni_clusters = rrr['cluster']  # indices of clusters for columns in U
            
            uni_clusters_id = np.unique(uni_clusters)
            K_bic[ig] = len(uni_clusters_id)  # record the number of clusters
            
        # Take the mean of similar columns of U. Save as U_sim
        U_BIC = U_opt.copy()
        for kk in list(uni_clusters_id):
            cluster_idx = np.where(uni_clusters == kk)[0]
            col_mean = np.mean(U_opt[:,cluster_idx], axis = 1)
            U_BIC[:, cluster_idx] = np.tile(col_mean, (len(cluster_idx), 1)).T
        
        
        y_pred_BIC = funcs.pred_labels(U_BIC)
        U_old = U_BIC  # update with warm start.
        
        # accumulate error for this gamma.
        # validation_bic[ig] = compute_bic(X, y_pred_BIC, U_BIC)
        validation_bic[ig] = compute_ebic(X, y_pred_BIC, U_BIC)
    
    # Only consider gammas that 1 < clusters < n
    opt_id_cand = np.where((K_bic > 1) & (K_bic < n))[0]  # [0] to extract indices
    
    # Select ga_opt from only opt_id_cand.
    opt_id = opt_id_cand[np.argmin(validation_bic[opt_id_cand])]
    gamma_opt = gamma_cand[opt_id]
    
    return gamma_opt, validation_bic, K_bic



from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture

def Kmeans_bic(X, K_cand, rand_seed = 1):
    """
    Output best number of clusters for kmeans clustering w.r.t. eBIC.
    
    Parameters
    ----------
    X : Data matrix p by n.
        
    K_cand : list of numbers of clusters.

    Returns
    -------
    K_opt: optimal number of clusters
    validation_bic: list of eBIC values

    """
    KM_bic_vec = np.ones(len(K_cand))
    p, n = X.shape

    for i in range(len(K_cand)):
        kk = K_cand[i]
        # Fit the K-means model to the data
        kmeans = KMeans(n_clusters = kk, random_state=rand_seed)
        kmeans.fit(X.T)
        labels = kmeans.labels_
        centroids = kmeans.cluster_centers_.T  # Convert centroids to p x K
        # K = len(centroids[0])  # Number of clusters
        
        # BIC(C | X) = L(X | C) - (p / 2) * log n
            # where L(X | C) is the log-likelihood of the dataset X according to
            # model C, p is the number of parameters in the model C, 
            # and n is the number of points in the dataset.
        # Compute variance sig^2
        total_variance = np.sum([np.linalg.norm(X[:, i] - centroids[:, labels[i]])**2 for i in range(n)])
        sigma_sq = total_variance / (p * n)
    
        # Compute log-likelihood
        log_likelihood = - (n * p / 2) * np.log(2 * np.pi * sigma_sq) - (1 / (2 * sigma_sq)) * total_variance
        KM_bic_vec[i] = log_likelihood - .5 * kk * np.log(n)
        
        
        # rss = kmeans.inertia_
        # KM_bic_vec[i] = n * np.log(rss / n) + 2 * kk * np.log(n)
        # KM_bic_vec[i] = rss + .5 * kk * p * np.log(n)

    # find num of clusters with minimal bic.
    # Select ga_opt from only opt_id_cand.
    # opt_id = np.argmin(KM_bic_vec)
    opt_id = np.argmax(KM_bic_vec)
    KM_Kopt = K_cand[opt_id]
    
    return KM_Kopt, KM_bic_vec

def GMM_bic(X, K_cand, rand_seed = 1):
    """
    Output best number of clusters for GMM clustering w.r.t. eBIC.
    
    Parameters
    ----------
    X : Data matrix p by n.
        
    K_cand : list of numbers of clusters.

    Returns
    -------
    K_opt: optimal number of clusters
    validation_bic: list of eBIC values

    """
    GMM_bic_vec = np.ones(len(K_cand))
    p, n = X.shape

    for i in range(len(K_cand)):
        kk = K_cand[i]
        # Fit the K-means model to the data
        gmm = GaussianMixture(n_components = kk, random_state=rand_seed)
        gmm.fit(X.T)
        GMM_bic_vec[i] = gmm.bic(X.T)

    # find num of clusters with minimal bic.
    # Select ga_opt from only opt_id_cand.
    opt_id = np.argmin(GMM_bic_vec)
    GMM_Kopt = K_cand[opt_id]
    
    return GMM_Kopt, GMM_bic_vec

from sklearn.metrics import silhouette_score

def Kmeans_sci(X, K_cand, rand_seed = 1):
    """
    Output best number of clusters for kmeans clustering w.r.t. silhouette.
    
    Parameters
    ----------
    X : Data matrix p by n.
        
    K_cand : list of numbers of clusters.

    Returns
    -------
    K_opt: optimal number of clusters

    """
    KM_sci_vec = np.ones(len(K_cand))
    p, n = X.shape

    for i in range(len(K_cand)):
        kk = K_cand[i]
        # Fit the K-means model to the data
        kmeans = KMeans(n_clusters = kk, random_state=rand_seed)
        kmeans.fit(X.T)
        labels = kmeans.labels_        
        KM_sci_vec[i] = silhouette_score(X.T, labels) 

    # find num of clusters with max SCI.
    # Select ga_opt from only opt_id_cand.
    opt_id = np.argmax(KM_sci_vec)
    KM_Kopt = K_cand[opt_id]
    
    return KM_Kopt, KM_sci_vec