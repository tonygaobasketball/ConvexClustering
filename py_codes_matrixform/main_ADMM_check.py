#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Testing ADMM algorithm

"""


#%%%% Generate Gaussian clusters graph.

import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
from scipy.spatial import distance_matrix
from sklearn.decomposition import PCA
import funcs
import scipy as sp
import diff_map_funcs
import random
import time

# Generate Gaussian clusters.

# Parameters
K = 4                 # number of clusters
p = 100                 # dimension
n_in_cluster = 5 * 1  # number of data points in each cluster
n = K * n_in_cluster  # total number of data points
rand_seed = 1

k_nrst = n_in_cluster - 1
# k_nrst = n - 1
# k_nrst = 2

# Generate the data
###--------------------
scale_para = 0.01
GM_data, labels, means, covariances = funcs.generate_gaussian_clusters(K, p, n_in_cluster, random_seed=rand_seed)
###--------------------


matrixData = GM_data.T
# Project the high-dimensional data onto 2 dimensions using first 2 dimension
if p > 2:
    # GM_data: n by p matrix
    Trans_mat = np.concatenate([np.eye(2), np.zeros([p-2,2])])  # p by 2 matrix
    data_2d = GM_data @ Trans_mat


# Plot the data in 2d (first 2d)
if p == 2:
    # Plot the generated data
    plt.figure(figsize=(8, 6))
    plt.scatter(matrixData[0, :], matrixData[1, :], c=labels, cmap='viridis', alpha=0.6, edgecolor='k')
    plt.title("Generated Gaussian Clusters")
    plt.xlabel("x 1")
    plt.ylabel("x 2")
    plt.grid(True)
    plt.show()

# Project the high-dimensional data onto 2 dimensions using (first 2d)
if p > 2:
    plt.figure(figsize=(8, 6))
    plt.scatter(data_2d[:, 0], data_2d[:, 1], c=labels, cmap='viridis', alpha=0.6, edgecolor='k')
    plt.title("Generated Gaussian Clusters (first 2d)")
    plt.xlabel("x 1")
    plt.ylabel("x 2")
    plt.grid(True)
    plt.show()

# Use CC_weights_graphs.py
import CC_weights_graphs
weight_mat0 = CC_weights_graphs.assign_weights(matrixData, k_nrst)
D, weights_vec = CC_weights_graphs.create_graph_MST_KNN(matrixData, weight_mat0, k_nrst, print_G = 'y')

#%% GME-CC clustering path

import FBS_algo
from scipy.sparse import eye, kron
from scipy.linalg import toeplitz
import admm_cc
###**********++++++++++++++++++++++++++++++++
# Calculate C and B, and their singular values.

# Define identity matrices
I_p = eye(p).toarray()
I_n = eye(n).toarray()

C = eye(D.shape[0]).toarray() @ np.diag(weights_vec) # Let matrix C be idenity.
# C = eye(D.shape[0]).toarray() @ np.diag(weights_vec / np.linalg.norm(weights_vec))

B = C @ D
sig1_B = np.linalg.norm(B, 2)
###**********++++++++++++++++++++++++++++++++

#% Run FBS and generate interchange graphs.
import imageio
from matplotlib.colors import to_rgb

rho = 1
mu = 1
#------------------------------
# Define the number of frames (linear frame)
numFrames = 10
gamma_up = 1 / sig1_B**2 + 5
gamma_cand = np.linspace(1e-06, gamma_up, numFrames)
#------------------------------

# Singular value of Ctilde
sig1_C = np.linalg.norm(C, 2)

p, n = matrixData.shape
tol_sim_path = 1e-03
res_ = []


U_opt = matrixData  # initiate for U_opt = mat_x

# Set gamma = 1/2 * gamma_up to test FBS
gamma = 1/2 * gamma_up      

#### Use warm start ###
U0 = U_opt       # initialize U
 
if p == 2:
    # Generating plots with different parameter gamma.
    # ===========
    # GME3-CC model
    
    U_opt, V_opt, Z_opt, cput, re_tol = FBS_algo.fbs_gme3_cc_mat(D, C, sig1_C, gamma, U0, matrixData)
    res_.append(re_tol)
    
    ##########################----####----####
    ## cluster path fuse function (Eric's code)
    # DU = D @ U_opt.T  # Calculate matrix V = DU
    DU = C @ D @ U_opt.T  # Calculate matrix V = DwU
    
    # Compute the Frobenius norm for each column in DU
    differences = np.linalg.norm(DU.T, axis=0)
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
    size_clusters = rrr['size']   # sizes of each unique cluster
    uni_clusters_id = np.unique(uni_clusters)
    
    # Take the mean of similar columns of U. Save as U_sim
    U_sim = U_opt.copy()
    for kk in list(uni_clusters_id):
        cluster_idx = np.where(uni_clusters == kk)[0]
        col_mean = np.mean(U_opt[:,cluster_idx], axis = 1)
        U_sim[:, cluster_idx] = np.tile(col_mean, (len(cluster_idx), 1)).T

    
if p > 2:
    Trans_mat = np.concatenate([np.eye(2), np.zeros([p-2,2])])  # p by 2 matrix
    # Generating plots with different parameter gamma.
    # ===========
    # GME3-CC model
    # U0 = mat_x       # initialize U
    U_opt, V_opt, Z_opt, cput, re_tol = FBS_algo.fbs_gme3_cc_mat(D, C, sig1_C, gamma, U0, matrixData)
    res_.append(re_tol)
    # U_opt = decompose_vec(u_opt, n)
    
    ##########################----####----####
    ## cluster path fuse function (Eric's code)
    
    
    # DU = D @ U_opt.T  # Calculate matrix V = DU
    DU = C @ D @ U_opt.T  # Calculate matrix V = DwU
    
    
    
    # Compute the Frobenius norm for each column in DU
    differences = np.linalg.norm(DU.T, axis=0)
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
    size_clusters = rrr['size']   # sizes of each unique cluster
    uni_clusters_id = np.unique(uni_clusters)
    
    # Take the mean of similar columns of U. Save as U_sim
    U_sim = U_opt.copy()
    for kk in list(uni_clusters_id):
        cluster_idx = np.where(uni_clusters == kk)[0]
        col_mean = np.mean(U_opt[:,cluster_idx], axis = 1)
        U_sim[:, cluster_idx] = np.tile(col_mean, (len(cluster_idx), 1)).T
    

