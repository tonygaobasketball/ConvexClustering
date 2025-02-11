#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
main BM with validation


"""


from sklearn.metrics import rand_score, adjusted_rand_score
from sklearn.metrics import silhouette_score, calinski_harabasz_score
import numpy as np
import scipy as sp
import CC_split_algo
from sklearn.decomposition import PCA

#%% I. Load datasets from Load_data.py
"""
dataX: p by n np array 
y_true: n-dim np array
"""

matrixData = dataX  # Keep it p by n matrix.
labels = y_true
# labels = y_true_habitat
# labels = y_true_diet

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
# k_nrst = 2
# weight_mat0 = CC_weights_graphs.assign_weights(matrixData, k_nrst)
# D, weights_vec = CC_weights_graphs.create_graph_MST_KNN(matrixData, weight_mat0, k_nrst, print_G = 'y')

# Use DMST:
import CC_weights_graphs
k_nrst = 3
weight_mat0 = CC_weights_graphs.assign_weights(matrixData, k_nrst)
D, weights_vec = CC_weights_graphs.build_DMST(matrixData, weight_mat0, t = 3)

# # Set parameters of validation
# repeat_val = 1    # number of times repeated for validation process.
# fraction_val = .1   # fraction of hold-out set.

#%% CC model.
#-------------------
#% Convex Clustering (Chi & Lange 2015)
#-------------------
import funcs
import CC_BIC

###**********++++++++++++++++++++++++++++++++
# Calculate C and B, and their singular values.

model_name = 'CC'
# Loop through each frame
mat_x = matrixData
p, n = mat_x.shape

tol_sim_path = 1e-06
# Define the number of frames
#------------------------------
# regular grids (all gamma > 0)
# gamma_cand = 2 ** (np.array(range(-8, 9), dtype = float)) 
gamma_cand = 2 ** (np.array(range(-15, 14), dtype = float)/2)
# gamma_cand = gamma_cand * p * n / D.shape[0]   # scaled for size of the dataset.
# Apply hold-out validation for the model.
# ga_opt, val_err = CC_validate.CC_validate(model_name, mat_x, k_nrst, gamma_cand, fraction = fraction_val, repeat = repeat_val)
# gamma_cand = np.linspace(1e-08, 100.0, num=50)
ga_opt, bic_val, K_val = CC_BIC.CC_bic(model_name, mat_x, D, weights_vec, gamma_cand)


# Plot the curve (BIC)
plt.figure(figsize=(8, 5))
plt.plot(np.log2(gamma_cand), bic_val, marker='o', linestyle='-', color='b', label=r'eBIC')
# Add labels and title
plt.xlabel(r'$\log_{2}(\gamma)$')
plt.ylabel(r'eBIC')
plt.title(model_name)
plt.legend()
# plt.grid(True)
# Show the plot
plt.show()

# Run model with optimal gamma again.
Lambda = np.zeros((p, D.shape[0]))
sol = CC_split_algo.cvxclust_admm(mat_x, Lambda, D, weights_vec, ga_opt, nu=1,
                    max_iter=1000, tol_abs=1e-3, tol_rel=1e-3,
                    norm_type=2, accelerate=True)
U_opt = sol['U']
V_opt = sol['V']

# DU = D @ U_opt.T  # Calculate matrix V = DwU

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
size_clusters = rrr['size']   # sizes of each unique cluster
uni_clusters_id = np.unique(uni_clusters)
CC_K_opt = len(uni_clusters_id)

# Take the mean of similar columns of U. Save as U_sim
U_sim = U_opt.copy()
for kk in list(uni_clusters_id):
    cluster_idx = np.where(uni_clusters == kk)[0]
    col_mean = np.mean(U_opt[:,cluster_idx], axis = 1)
    U_sim[:, cluster_idx] = np.tile(col_mean, (len(cluster_idx), 1)).T
    
print(f'optimal gamma = {ga_opt:.4f}')
print(f'optimal number of clusters = {CC_K_opt}')

# Predicted labels.
y_pred = funcs.pred_labels(U_sim)

## Calculate measurements.
# 1. Rand Index (RI)
ri = rand_score(y_true, y_pred)

# 2. Adjusted Rand Index (ARI)
ari = adjusted_rand_score(y_true, y_pred)

# 3. Variation of Information (VI)
vi = funcs.variation_of_information(y_true, y_pred)

# 4. Calinski–Harabasz Index (CHI)
if len(np.unique(y_pred)) == len(y_pred) or len(np.unique(y_pred)) == 1:
    chi = -99
    silhouette = -99
else:
    chi = calinski_harabasz_score(dataX.T, y_pred)
    silhouette = silhouette_score(dataX.T, y_pred)

## Print the best result of measurements.
# CC_ACC = max(CC_ACC_vec)
CC_RI = ri
CC_ARI = ari
CC_VI = vi

CC_CHI = chi
CC_SCI = silhouette

print("Scores for CC:\n")
# print(f"CC_ACC: {CC_ACC:.4f}")
print(f"CC_RI: {CC_RI:.4f}")
print(f"CC_ARI: {CC_ARI:.4f}")
print(f"CC_VI: {CC_VI:.4f}")
print(f"CC_CHI: {CC_CHI:.4f}")
print(f"CC_SCI: {CC_SCI:.4f}\n")


#%% IV. CNC-GME model.

import FBS_algo
import CC_BIC

model_name = 'GME'
###**********++++++++++++++++++++++++++++++++
# Calculate C and B, and their singular values.
# C = eye(D.shape[0]).toarray() @ np.diag(weights_vec) # Let matrix C be idenity.
# Dw = C @ D
Dw = np.diag(weights_vec) @ D


#% Run FBS and generate interchange graphs.
import imageio
from matplotlib.colors import to_rgb

mat_x = matrixData
p, n = mat_x.shape
tol_sim_path = 1e-04
max_iter_admm = 1000
# Define the number of frames
#------------------------------
# (linear frame with gamma_up)
# numFrames = 200
# gamma_up = 1 / sig1_B ** 2 + 10
# gamma_cand = np.linspace(1e-06, gamma_up, numFrames)
#------------------------------
# regular grids (all gamma > 0)
# gamma_cand = 2 ** (np.array(range(-26, 20), dtype = float)/4)
gamma_cand = 2 ** (np.array(range(-15, 4), dtype = float)/2)
# gamma_cand = 2 ** np.array(range(-15, 0), dtype = float)

# numFrames = 10
# gamma_cand = np.linspace(2 ** -7, 2 ** -5, numFrames)
numFrames = len(gamma_cand)
#------------------------------
gamma_cand = gamma_cand * p * n / D.shape[0]   # scaled for size of the dataset.


# Apply hold-out validation for the model.
# ga_opt, val_err = CC_validate.CC_validate(model_name, mat_x, k_nrst, gamma_cand, fraction = fraction_val, repeat = repeat_val)
ga_opt, bic_val, K_val = CC_BIC.CC_bic(model_name, mat_x, D, weights_vec, gamma_cand, max_iter_admm=max_iter_admm)



# Plot the curve (BIC)
plt.figure(figsize=(8, 5))
plt.plot(np.log2(gamma_cand), bic_val, marker='o', linestyle='-', color='b', label=r'eBIC')
# Add labels and title
plt.xlabel(r'$\log_{2}(\gamma)$')
plt.ylabel(r'eBIC')
plt.title(model_name)
plt.legend()
# plt.grid(True)
# Show the plot
plt.show()




# Run model with optimal gamma again.
sig1_Dw = np.linalg.norm(Dw, 2)  # Calculate largest singular value of Dw.
B_base = Dw 
# Use different B for different scale of gamma.
gamma_up = 1 / sig1_Dw ** 2

if ga_opt <= gamma_up:
    theta = 1
    B = B_base
if ga_opt > gamma_up:
    theta = 1 / (np.sqrt(ga_opt) * sig1_Dw)
    B = theta * B_base   # Assign B = theta * Dw to satisfy the convexity condition.


U0 = mat_x.copy()       # initialize U
# GME3-CC model
U_opt, _,  _, _, _ = FBS_algo.fbs_gme3_cc_mat(B, Dw, theta, sig1_Dw, ga_opt, U0, mat_x, max_iter_admm = max_iter_admm)

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
size_clusters = rrr['size']   # sizes of each unique cluster
uni_clusters_id = np.unique(uni_clusters)
GME_K_opt = len(uni_clusters_id)

# Take the mean of similar columns of U. Save as U_sim
U_sim = U_opt.copy()
for kk in list(uni_clusters_id):
    cluster_idx = np.where(uni_clusters == kk)[0]
    col_mean = np.mean(U_opt[:,cluster_idx], axis = 1)
    U_sim[:, cluster_idx] = np.tile(col_mean, (len(cluster_idx), 1)).T

    
# Predicted labels.
y_pred = funcs.pred_labels(U_sim)

## Calculate measurements.
# 1. Rand Index (RI)
ri = rand_score(y_true, y_pred)

# 2. Adjusted Rand Index (ARI)
ari = adjusted_rand_score(y_true, y_pred)

# 3. Variation of Information (VI)
vi = funcs.variation_of_information(y_true, y_pred)

# 4. Calinski–Harabasz Index (CHI)
if len(np.unique(y_pred)) == len(y_pred) or len(np.unique(y_pred)) == 1:
    chi = -99
    silhouette = -99
else:
    chi = calinski_harabasz_score(dataX.T, y_pred)
    silhouette = silhouette_score(dataX.T, y_pred)

## Print the best result of measurements.
# CC_ACC = max(CC_ACC_vec)
GME_RI = ri
GME_ARI = ari
GME_VI = vi

GME_CHI = chi
GME_SCI = silhouette

print("Scores for CNC-GME:\n")
print(f'optimal gamma = {ga_opt:.4f}')
print(f'optimal number of clusters = {GME_K_opt}')    
# print(f"ACC: {GME_ACC:.4f}")
print(f"RI: {GME_RI:.4f}")
print(f"ARI: {GME_ARI:.4f}")
print(f"VI: {GME_VI:.4f}")
print(f"CHI: {GME_CHI:.4f}")
print(f"SCI: {GME_SCI:.4f}\n") 



    
#%% Apply K-means method
### Apply GMM 


# Import necessary libraries
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score, silhouette_score
import funcs
import CC_BIC

# K-means 
model_name = "Kmeans"
max_num_K = 20
K_cand = list(range(2, max_num_K+1))

# # Use eBIC for Kmeans.
# KM_Kopt, KM_bics = CC_BIC.Kmeans_bic(matrixData, K_cand)

# Use silhouette index.
KM_Kopt, KM_scis = CC_BIC.Kmeans_sci(matrixData, K_cand)

kmeans_opt = KMeans(n_clusters = KM_Kopt, random_state = KM_Kopt)
kmeans_opt.fit(matrixData.T)
y_pred = kmeans_opt.labels_

# Calculate accuracy and other metrics
# KM_ACC = funcs.cluster_acc(y_true, y_pred)
KM_RI = rand_score(y_true, y_pred)
KM_ARI = adjusted_rand_score(y_true, y_pred)
KM_VI = funcs.variation_of_information(y_true, y_pred)
if KM_Kopt == len(y_pred) or KM_Kopt == 1:
    KM_CHI = -99
    KM_SCI = -99
else:
    KM_CHI = calinski_harabasz_score(dataX.T, y_pred)
    KM_SCI = silhouette_score(dataX.T, y_pred) 

# Plot the curve (BIC)
plt.figure(figsize=(8, 5))
# plt.plot(K_cand, KM_bics, marker='o', linestyle='-', color='b', label=r'BIC')
plt.plot(K_cand, KM_scis, marker='o', linestyle='-', color='b', label=r'SCI')
# Add labels and title
plt.xlabel(r'$k$')
# plt.ylabel(r'BIC')
plt.ylabel(r'SCI')
plt.title(model_name)
plt.legend()
# plt.grid(True)
# Show the plot
plt.show()

print("Clustering measurements for K-means:\n")
print(f"optimal # of clusters = {KM_Kopt}")
# print(f"KM_ACC: {KM_ACC:.4f}")
print(f"KM_RI: {KM_RI:.4f}")
print(f"KM_ARI: {KM_ARI:.4f}")
print(f"KM_VI: {KM_VI:.4f}")
print(f"KM_CHI: {KM_CHI:.4f}")
print(f"KM_SCI: {KM_SCI:.4f} \n")
    


# GMM
model_name = "GMM"
max_num_K = 20
K_cand = list(range(2, max_num_K+1))

# Use eBIC for GMM.
GMM_Kopt, GMM_bics = CC_BIC.GMM_bic(matrixData, K_cand)

gmm_opt = GaussianMixture(n_components = GMM_Kopt, random_state = GMM_Kopt)
gmm_opt.fit(matrixData.T)
y_pred = gmm_opt.predict(matrixData.T)

# Calculate accuracy and other metrics
# GMM_ACC = funcs.cluster_acc(y_true, y_pred)
GMM_RI = rand_score(y_true, y_pred)
GMM_ARI = adjusted_rand_score(y_true, y_pred)
GMM_VI = funcs.variation_of_information(y_true, y_pred)
if GMM_Kopt == len(y_pred) or GMM_Kopt == 1:
    GMM_CHI = -99
    GMM_SCI = -99
else:
    GMM_CHI = calinski_harabasz_score(dataX.T, y_pred)
    GMM_SCI = silhouette_score(dataX.T, y_pred) 
    
# Plot the curve (BIC)
plt.figure(figsize=(8, 5))
plt.plot(K_cand, GMM_bics, marker='o', linestyle='-', color='b', label=r'BIC')
# Add labels and title
plt.xlabel(r'$k$')
plt.ylabel(r'BIC')
plt.title(model_name)
plt.legend()
# plt.grid(True)
# Show the plot
plt.show()

print("Clustering measurements for GMM:\n")
print(f"optimal # of clusters = {GMM_Kopt}")
# print(f"GMM_ACC: {GMM_ACC:.4f}")
print(f"GMM_RI: {GMM_RI:.4f}")
print(f"GMM_ARI: {GMM_ARI:.4f}")
print(f"GMM_VI: {GMM_VI:.4f}")
print(f"GMM_CHI: {GMM_CHI:.4f}")
print(f"GMM_SCI: {GMM_SCI:.4f}")

#%%%%% 
# ### GME model plot GME_VI's against k_nrst.

# knn_cand = np.arange(4, 22, 2)
# clustr_list = []
# vi_list = []


# for k_nrst in knn_cand:

#     weight_mat0 = CC_weights_graphs.assign_weights(matrixData, k_nrst)
#     # D, weights_vec = CC_weights_graphs.create_graph_MST_KNN(matrixData, weight_mat0, k_nrst, print_G = 'y')
#     D, weights_vec = CC_weights_graphs.build_DMST(matrixData, weight_mat0, t = 3)
    

#     model_name = 'GME'
#     ###**********++++++++++++++++++++++++++++++++
#     Dw = np.diag(weights_vec) @ D
#     sig1_Dw = np.linalg.norm(Dw, 2)  # Calculate largest singular value of Dw.
    
#     #% Run FBS and generate interchange graphs.
#     import imageio
#     from matplotlib.colors import to_rgb

#     mat_x = matrixData
#     p, n = mat_x.shape
#     tol_sim_path = 1e-06
#     # Define the number of frames
#     #------------------------------
#     # (linear frame with gamma_up)
#     # numFrames = 200
#     # gamma_up = 1 / sig1_B ** 2 + 10
#     # gamma_cand = np.linspace(1e-06, gamma_up, numFrames)
#     #------------------------------
#     # regular grids (all gamma > 0)
#     gamma_cand = 2 ** (np.array(range(-8, 9), dtype = float))
#     # gamma_cand = 2 ** np.array(range(-10, 4), dtype = float)

#     # numFrames = 50
#     # gamma_cand = np.linspace(1e-4, 1/2, numFrames)
#     numFrames = len(gamma_cand)
#     #------------------------------
#     gamma_cand = gamma_cand * p * n / D.shape[0]   # scaled for size of the dataset.


#     # Apply hold-out validation for the model.
#     # ga_opt, val_err = CC_validate.CC_validate(model_name, mat_x, k_nrst, gamma_cand, fraction = fraction_val, repeat = repeat_val)
#     ga_opt, bic_val, K_val = CC_BIC.CC_bic(model_name, mat_x, D, weights_vec, gamma_cand)
#     # Run model with optimal gamma again.
#     B_base = Dw 
#     # Use different B for different scale of gamma.
#     gamma_up = 1 / sig1_Dw ** 2

#     if ga_opt <= gamma_up:
#         theta = 1
#         B = B_base
#     if ga_opt > gamma_up:
#         theta = 1 / (np.sqrt(ga_opt) * sig1_Dw)
#         B = theta * B_base   # Assign B = theta * Dw to satisfy the convexity condition.


#     U0 = mat_x.copy()       # initialize U
#     # GME3-CC model
#     U_opt, _,  _, _, _ = FBS_algo.fbs_gme3_cc_mat(B, Dw, theta, sig1_Dw, ga_opt, U0, mat_x)

#     ##########################----####----####
#     DU = Dw @ U_opt.T  # Calculate matrix V = DwU

#     # Compute the Frobenius norm for each column in DU
#     differences = np.linalg.norm(DU.T, axis=0)
#     # Find indices where differences are zero, indicating connected nodes
#     connected_ix = np.where(differences <= tol_sim_path)[0]

#     # Initialize a sparse adjacency matrix
#     AA = sp.sparse.lil_matrix((n, n), dtype=int)

#     for kk in list(connected_ix):
#         # find index pair non-zeros in kk th row of D matrix
#         ii, jj = np.where(Dw[kk,:] != 0)[0]    
#         # Build A matrix
#         AA[ii, jj] = 1
#         AA[jj, ii] = 1
#     # Find clusters
#     rrr = funcs.find_clusters(AA)
#     uni_clusters = rrr['cluster']  # indices of clusters for columns in U
#     size_clusters = rrr['size']   # sizes of each unique cluster
#     uni_clusters_id = np.unique(uni_clusters)
#     K_opt = len(uni_clusters_id)
#     clustr_list.append(K_opt)

#     # Take the mean of similar columns of U. Save as U_sim
#     U_sim = U_opt.copy()
#     for kk in list(uni_clusters_id):
#         cluster_idx = np.where(uni_clusters == kk)[0]
#         col_mean = np.mean(U_opt[:,cluster_idx], axis = 1)
#         U_sim[:, cluster_idx] = np.tile(col_mean, (len(cluster_idx), 1)).T
        
#     print(f'optimal gamma = {ga_opt:.4f}')
#     print(f'optimal number of clusters = {K_opt}')    
        
#     # Predicted labels.
#     y_pred = funcs.pred_labels(U_sim)

    
#     # 3. Variation of Information (VI)
#     vi = funcs.variation_of_information(y_true, y_pred)
#     vi_list.append(vi)

# # Plot vi_list against knn_cand

# # Create figure and axis
# fig, ax1 = plt.subplots(figsize=(8, 5))

# # Plot on the first y-axis
# ax1.plot(knn_cand, vi_list, marker='o', linestyle='-', color='orange', label="VI")
# ax1.set_xlabel('k_nrst')
# ax1.set_ylabel('VI')
# ax1.tick_params(axis='y')

# # Create a second y-axis
# ax2 = ax1.twinx()
# ax2.plot(knn_cand, clustr_list, marker='s', linestyle='--', color='blue', label="Clusters")
# ax2.set_ylabel('Number of Clusters')
# ax2.tick_params(axis='y')

# # Add legend
# fig.legend(loc="upper left", bbox_to_anchor=(0.15, 0.85))

# # Show the plot
# plt.show()