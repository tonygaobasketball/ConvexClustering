#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Test new codes for convex clutering.
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import funcs
import scipy as sp
import time
import CC_split_algo


#%% Import synthetic data. 
"""
1. dataX
2. y_true

"""
matrixData = dataX
y_true = y_true
# y_true = np.array(range(len(labels)))
# y_true = labels

# # Perform on given data. (run Load_data.py)
# dataX_original = dataX
# y_true_original = y_true
# sample_fraction = 0.2  # Choose rate of each class
# matrixData, y_true = stratified_sampling_preserve_stats(dataX_original.T, y_true_original, sample_fraction)

p, n = matrixData.shape
print(f'{p}, {n}, {len(np.unique(labels))}')

#%%%% Building graph.
if p > 2:
    # Perform PCA
    ## record if using PCA.
    PCA_id = 1
    
    pca = PCA(n_components=2)  # Change the number of components as needed
    x_pca = pca.fit_transform(matrixData.T)
    Trans_mat = pca.components_.T
    
    # # Center the data (subtract the mean of each feature)
    matrixData_centered = matrixData - np.mean(matrixData, axis=1, keepdims=True)
    # Compute PCA manually
    x_pca_manual = matrixData_centered.T @ Trans_mat
    
    data_plt = matrixData.T @ Trans_mat
    
    plt.figure(figsize=(8, 6))
    plt.scatter(data_plt[:, 0], data_plt[:, 1], c=y_true, cmap='viridis', alpha=0.6, edgecolor='k')
    plt.title("Generated Gaussian Clusters (PCA)")
    plt.xlabel("x 1")
    plt.ylabel("x 2")
    plt.grid(True)
    plt.show()
    

if p == 2:
    Trans_mat = np.eye(p)
    data_plt = matrixData.T @ Trans_mat
    # Plot the generated data
    plt.figure(figsize=(8, 6))
    plt.scatter(data_plt[:, 0], data_plt[:, 1], c=y_true, cmap='viridis', alpha=0.6, edgecolor='k')
    plt.title("Generated Gaussian Clusters")
    plt.xlabel("x 1")
    plt.ylabel("x 2")
    plt.grid(True)
    plt.show()
    
    
# # Use CC_weights_graphs.py
# import CC_weights_graphs
# weight_mat0 = CC_weights_graphs.assign_weights(matrixData, k_nrst)
# D, weights_vec = CC_weights_graphs.create_graph_MST_KNN(matrixData, weight_mat0, k_nrst, print_G = 'y')

# Use DMST:
import CC_weights_graphs
k_nrst = 3
weight_mat0 = CC_weights_graphs.assign_weights(matrixData, k_nrst)
D, weights_vec = CC_weights_graphs.build_DMST(matrixData, weight_mat0, t = 3)

#%% 
#-------------------
#% Convex Clustering (Chi & Lange 2015)
#-------------------


###**********++++++++++++++++++++++++++++++++
# Calculate C and B, and their singular values.


# Define the number of frames
#------------------------------
# regular grids (all gamma > 0)
gamma_cand = 2 ** (np.array(range(-190, 60), dtype = float) / 8)
# gamma_cand = 2 ** (np.array(range(-15, 7), dtype = float))

# numFrames = 300
# s_gamma_cand = 2 ** np.array(range(-16, -1), dtype = float)
# l_gamma_cand = np.linspace(2 ** (-2), 2 ** 7, numFrames)
# gamma_cand = np.append(s_gamma_cand,l_gamma_cand)
# numFrames = len(gamma_cand)
#------------------------------

# Loop through each frame
mat_x = matrixData
p, n = mat_x.shape
# recover full weights with length n(n-1)/2
# weights_full = CC_split_algo.recover_full_weights(D, weights_vec)


# Dist = np.zeros([numFrames, n])


# print("ADMM Example:")
# p, n = 5, 10
# np.random.seed(123)
# X = np.random.randn(p, n)
# m_full = n * (n - 1) // 2
# w = np.ones(m_full)
# gamma_cand = np.linspace(1e-6, 50.0, num=3000)
# gamma_seq = gamma_cand
centroids_tensor = np.zeros([len(gamma_cand), n, 2])

Count_CPUt = time.time()   # record CPU time for all loops.

sol_admm = CC_split_algo.cvxclust_path_admm(mat_x, D, weights_vec, gamma_cand, nu=1, tol_abs =1e-3, tol_rel= 1e-3,
                    max_iter=1000, norm_type=2, accelerate=True)
# sol_admm contains: 
    # {"U": list_U, "V": list_V, "Lambda": list_Lambda,
    #       "nGamma": len(gamma_seq), "iters": iter_vec}
Count_CPUt = time.time() - Count_CPUt
avg_CPUt_CC = Count_CPUt / len(gamma_cand)   # output it.

# record clustering centroids for each gamma.
for i in range(len(gamma_cand)):
    U_opt = sol_admm['U'][i]
    centroids_tensor[i,:,:] = U_opt.T @ Trans_mat
    


print("ADMM iteration counts per gamma:")
print(sol_admm["iters"])


plt.figure()
for j in range(n):
    # For each data point, plot its centroid route
    route_pts = centroids_tensor[:,j,:]  # numFrames by p.
    # Plot route for jth centroid.
    plt.plot(route_pts[:,0], route_pts[:,1], color = 'green', linestyle = '-', linewidth = 0.5)
# plot all data points.
plt.scatter(data_plt[:,0], data_plt[:,1], c='b', marker='*', label='Data Points')

# plot ground truth mean.
# gt_mean_2d = np.array(true_cls_centers) @ Trans_mat
# plt.scatter(gt_mean_2d[:,0], gt_mean_2d[:,1], c='r', marker='o', label='GT-mean')

    
plt.xlabel('Component 1')
plt.ylabel('Component 2')
plt.legend()
plt.title(f'Convex clustering path (p = {p}, n = {n})')
# plt.title('toy4-5c cluster path (k_near = {})'.format(k_nearest))
plt.show()
# plt.close


print(f'CPU time for one CC model is (average): {avg_CPUt_CC:.4f}')

#%% GME-CC clustering path

import FBS_algo
###**********++++++++++++++++++++++++++++++++
# Calculate C and B, and their singular values.

Dw = np.diag(weights_vec) @ D
sig1_Dw = np.linalg.norm(Dw, 2)  # Calculate largest singular value of Dw.
B_base = Dw 
# Use different B for different scale of gamma.
gamma_up = 1 / sig1_Dw ** 2


# Define the number of frames
#------------------------------
# (linear frame with gamma_up)
# numFrames = 200
# gamma_up = 1 / sig1_B ** 2 + 10
# gamma_cand = np.linspace(1e-06, gamma_up, numFrames)
#------------------------------
# regular grids (all gamma > 0)
gamma_cand = 2 ** (np.array(range(-120, 50), dtype = float) / 8)
# gamma_cand = 2 ** (np.array(range(-15, 3), dtype = float))

# numFrames = 200
# gamma_cand = np.linspace(2 ** -11, 1, numFrames)
numFrames = len(gamma_cand)
#------------------------------
gamma_cand = gamma_cand * p * n / D.shape[0]   # scaled for size of the dataset.


gamma_set = list(gamma_cand)
mat_x = matrixData
# true_cls_centers = means
# B_ga = Dw * np.sqrt(gamma) / sig1_Dw    # Assign B = B_ga to satisfy the convexity condition.

p, n = mat_x.shape
centroids_tensor = np.zeros([numFrames, n, 2])
tol_sim_path = 1e-03
res_ = []


U_opt = mat_x  # initiate for U_opt = mat_x
Count_CPUt = time.time()   # record CPU time for all loops.
for i in range(numFrames):
    # Clear the figure
    print("------")
    print("gamma candidate {}".format(i))
    print("------")
    gamma = gamma_set[i]        
    
    if gamma <= gamma_up:
        theta = 1
        B = B_base
    if gamma > gamma_up:
        theta = 1 / (np.sqrt(gamma) * sig1_Dw)
        B = theta * B_base   # Assign B = theta * Dw to satisfy the convexity condition.

    
    #### Use warm start ###
    U0 = U_opt       # initialize U
 
    
    # Generating plots with different parameter gamma.
    # ===========
    # GME3-CC model
    
    # U_opt, V_opt, Z_opt, cput = FBS_algo.fbs_gme3_cc_mat(D, C, sig1_C, mu, rho, gamma, U0, mat_x)
    U_opt, V_opt, Z_opt, cput, re_tol = FBS_algo.fbs_gme3_cc_mat(B, Dw, theta, sig1_Dw, gamma, U0, mat_x)
    res_.append(re_tol)
    
    ##########################----####----####
    ## cluster path fuse function (Eric's code)
    # DU = D @ U_opt.T  # Calculate matrix V = DU
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
    size_clusters = rrr['size']   # sizes of each unique cluster
    uni_clusters_id = np.unique(uni_clusters)
    
    # Take the mean of similar columns of U. Save as U_sim
    U_sim = U_opt.copy()
    for kk in list(uni_clusters_id):
        cluster_idx = np.where(uni_clusters == kk)[0]
        col_mean = np.mean(U_opt[:,cluster_idx], axis = 1)
        U_sim[:, cluster_idx] = np.tile(col_mean, (len(cluster_idx), 1)).T
    
    centroids_tensor[i,:,:] = U_sim.T @ Trans_mat
    # Plot paths using U_sim.


Count_CPUt = time.time() - Count_CPUt
avg_CPUt_GME = Count_CPUt / numFrames   # output it.


plt.figure()
for j in range(n):
    # For each data point, plot its centroid route
    route_pts = centroids_tensor[:,j,:]  # numFrames by p.
    # Plot route for jth centroid.
    plt.plot(route_pts[:,0], route_pts[:,1], color = 'green', linestyle = '-', linewidth = 0.5)
# plot all data points.
plt.scatter(data_plt[:,0], data_plt[:,1], c='b', marker='*', label='Data Points')

# plot ground truth mean.
# gt_mean_2d = np.array(true_cls_centers) @ Trans_mat
# plt.scatter(gt_mean_2d[:,0], gt_mean_2d[:,1], c='r', marker='o', label='GT-mean')

    
plt.xlabel('Component 1')
plt.ylabel('Component 2')
plt.legend()
plt.title(f'CNC-GME path (p = {p}, n = {n})')
# plt.title('toy4-5c cluster path (k_near = {})'.format(k_nearest))
plt.show()
# plt.close

print(f'CPU time for one GME-CC model is (average): {avg_CPUt_GME:.4f}')