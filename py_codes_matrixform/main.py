
"""
Gaussian mixture model

@author: Zheming Gao
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
p = 20                 # dimension
n_in_cluster = 5 * 1  # number of data points in each cluster
n = K * n_in_cluster  # total number of data points
rand_seed = 10

k_nrst = n_in_cluster - 1
# k_nrst = n - 1
# k_nrst = 2

# Generate the data
###--------------------
# GM_data, labels, means, covariances = funcs.generate_gaussian_clusters_2dmean(K, p, n_in_cluster, random_seed=rand_seed)
###--------------------
scale_para = 0.01
GM_data, labels, means, covariances = funcs.generate_gaussian_clusters(K, p, n_in_cluster, random_seed=rand_seed)
###--------------------
# scale_para = 0.1
# GM_data, labels, means, covariances = funcs.generate_gaussian_clusters_corner(K, p, n_in_cluster, overlap_factor = scale_para, random_seed=rand_seed)
###--------------------
# p_low = 10
# p_high = 1000
# scale_para = 0.05
# GM_data, labels, means, covariances = funcs.generate_gaussian_clusters_low4high_dim(K, p_low, p_high, n_in_cluster, overlap_factor = scale_para, random_seed=rand_seed)
# p = p_high
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

#%% recale weights. (for specific use only.)
# Create a histogram
plt.figure(figsize=(8, 6))
plt.hist(weights_union, bins=10, color='blue', alpha=0.7, edgecolor='black')

# Add labels and title
plt.title("Distribution of Weights", fontsize=16)
plt.xlabel("Weight Value", fontsize=12)
plt.ylabel("Frequency", fontsize=12)

# Show the plot
plt.grid(alpha=0.3)
plt.show()

# Replace values in weights_union. 
# if value less than or equalt to , then replaced by multiplying .

for i in range(len(weights_union)):
    if weights_union[i] <= 0.70:
        weights_union[i] = 0.125 * weights_union[i]

weights_vec = weights_union



# #%% If only use KNN for generating the data:

# # Union graph G_u
# D = kn_D
# weights_vec = kn_weights    # weights vector of G_u
# sig1_D = np.linalg.norm(D, 2)  # largest singular value of D matrix.


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
numFrames = 200
gamma_up = 1 / sig1_B ** 2 + 50
gamma_cand = np.linspace(1e-06, gamma_up, numFrames)
#------------------------------

gamma_set = list(gamma_cand)
mat_x = matrixData
avg_CPUt, _ = FBS_algo.GME_CC_path(gamma_set, D, C, matrixData, means)
print(f'CPU time for one GME-CC model is (average): {avg_CPUt:.4f}')

#%%
#-------------------
#% Convex Clustering (Chi & Lange 2015)
#-------------------

from scipy.sparse import eye, kron
from scipy.linalg import toeplitz
import admm_cc

###**********++++++++++++++++++++++++++++++++
# Calculate C and B, and their singular values.

# Define identity matrices
I_p = eye(p).toarray()
I_n = eye(n).toarray()


#% Run FBS and generate interchange graphs.
import imageio
from matplotlib.colors import to_rgb

rho = 1
mu = 1
#------------------------------
# Define the number of frames
grids = 2 ** np.array(range(-16, 20), dtype = float)
gamma_cand = grids
numFrames = len(gamma_cand)
#------------------------------

# Initialize the GIF file
filename = 'cc_temp.gif'

# Loop through each frame
mat_x = matrixData
X = mat_x.T
centroids_tensor = np.zeros([numFrames, n, 2])


for i in range(numFrames):
    # Clear the figure
    print("------")
    print("gamma candidate {}".format(i))
    print("------")
    plt.figure()
        
    # Generating plots with different parameter gamma.
    # ===========
    # GME3-CC model
    gamma = gamma_cand[i]
    # nu = 0.1  # Augmented Lagrangian parameter
    nu = np.sqrt(gamma) * 0.05
    print('nu = {:.4f}, ga = {:.4f}'.format(nu, gamma))
    # gamma = 100   # parameter of l1-norm dist of centroids
    
    sigma = gamma * weights_vec / nu  # sigma values in prox_l1
    # sigma = gamma * (weights_vec / np.linalg.norm(weights_vec)) / nu # 
    # U_opt, V_opt, Z_opt, cput = fbs_gme3_cc_mat(D, C, sig1_C, mu, rho, gamma, U0, mat_x)
    U_opt, _, _ = admm_cc.convex_clustering_admm(mat_x, D, nu, sigma)
    
    
    if p == 2:
        centroids_tensor[i,:,:] = U_opt.T
    if p > 2:
        centroids_tensor[i,:,:] = U_opt.T @ Trans_mat

print(f'GIF saved as {filename}')


plt.figure()

if p == 2:
    for j in range(n):
        # For each data point, plot its centroid route
        route_pts = centroids_tensor[:,j,:]  # numFrames by p.
        # Plot route for jth centroid.
        plt.plot(route_pts[:,0], route_pts[:,1], color = 'green', linestyle = '-', linewidth = 0.5)
    
    # plot all data points.
    plt.scatter(mat_x.T[:,0], mat_x.T[:,1], c='b', marker='*', label='Data Points')
    
    # Add generated ground-true means to the plot
    gt_mean_2d = np.array(means)[:,:2]
    plt.scatter(gt_mean_2d[:,0], gt_mean_2d[:,1], c='r', marker='o', label='GT-mean')
    
    plt.xlabel('X1')
    plt.ylabel('X2')
    plt.legend()
    plt.title('Convex clustering path')
    # plt.title('toy4-5c cluster path (k_near = {})'.format(k_nearest))
    plt.show()
    # plt.close
if p > 2:
    for j in range(n):
        # For each data point, plot its centroid route
        route_pts = centroids_tensor[:,j,:]  # numFrames by p.
        # Plot route for jth centroid.
        plt.plot(route_pts[:,0], route_pts[:,1], color = 'green', linestyle = '-', linewidth = 0.5)
    
    # plot all data points.
    plt.scatter(data_2d[:,0], data_2d[:,1], c='b', marker='*', label='Data Points')
    # Add generated ground-true means to the plot
    gt_mean_2d = np.array(means)[:,:2]
    plt.scatter(gt_mean_2d[:,0], gt_mean_2d[:,1], c='r', marker='o', label='GT-mean')
    
    plt.xlabel('Component 1')
    plt.ylabel('Component 2')
    plt.legend()
    plt.title('Convex clustering path')
    # plt.title('toy4-5c cluster path (k_near = {})'.format(k_nearest))
    plt.show()
    # plt.close
    

#%% Apply GMM methods for clustering

# Import necessary libraries
from sklearn.mixture import GaussianMixture
from sklearn.metrics import accuracy_score, adjusted_rand_score, normalized_mutual_info_score
from sklearn.metrics import silhouette_score
import numpy as np
import matplotlib.pyplot as plt

data = matrixData.T
# Fit the Gaussian Mixture Model (GMM) to the data
gmm = GaussianMixture(n_components = K, random_state = rand_seed)
gmm.fit(data)
label_gmm = gmm.predict(data)

# Save the predicted labels
np.save("label_gmm.npy", label_gmm)

# Function to calculate clustering accuracy
def clustering_accuracy(true_labels, predicted_labels):
    """
    Compute the accuracy by finding the best mapping between true and predicted labels.
    """
    from scipy.optimize import linear_sum_assignment

    # Create a contingency table
    contingency_matrix = np.zeros((K, K), dtype=int)
    for true, pred in zip(true_labels, predicted_labels):
        contingency_matrix[true, pred] += 1

    # Find the optimal mapping
    row_ind, col_ind = linear_sum_assignment(-contingency_matrix)
    accuracy = contingency_matrix[row_ind, col_ind].sum() / len(true_labels)
    return accuracy

# Calculate accuracy

# Print the results
"""
Metrics Used

	1.	Accuracy: Measures the proportion of correctly assigned labels after finding the best label mapping.
	2.	Adjusted Rand Index (ARI): Ranges from -1 to 1, where 1 indicates perfect clustering and 0 indicates random clustering.
	3.	Normalized Mutual Information (NMI): Ranges from 0 to 1, with higher values indicating better clustering.
"""
print("GMM results: ")
ACC_gmm = clustering_accuracy(labels, label_gmm)
ARI_gmm = adjusted_rand_score(labels, label_gmm)
NMI_gmm = normalized_mutual_info_score(labels, label_gmm)
SC_gmm = silhouette_score(data, label_gmm)


print("ACC:", ACC_gmm)
print("ARI:", ARI_gmm)
print("NMI:", NMI_gmm)
print("SC:", SC_gmm)


# Ensure that n = 2 for visualization
if p == 2:
    # Define a common colormap and normalization
    cmap = plt.cm.viridis
    norm = plt.Normalize(vmin=0, vmax=K-1)  # Normalize to the number of clusters

    # Create a figure with two subplots: one for the true labels and one for the GMM predicted labels
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Plot the true labels using circles
    scatter1 = axes[0].scatter(data[:, 0], data[:, 1], c=labels, cmap='viridis', s=20, alpha=0.6, marker='o')
    axes[0].set_title("True Labels")
    axes[0].set_xlabel("Dimension 1")
    axes[0].set_ylabel("Dimension 2")
    axes[0].grid(True)

    # Plot the GMM predicted labels using crosses (x marker)
    scatter2 = axes[1].scatter(data[:, 0], data[:, 1], c=labels, cmap='turbo', s=20, alpha=0.6, marker='x')
    axes[1].set_title("GMM Predicted Labels (Crosses)")
    axes[1].set_xlabel("Dimension 1")
    axes[1].set_ylabel("Dimension 2")
    axes[1].grid(False)

    # Show the plots
    plt.tight_layout()
    plt.show()
    
#%% Apply K-means method

# Import necessary libraries
from sklearn.cluster import KMeans
from sklearn.metrics import accuracy_score, adjusted_rand_score, normalized_mutual_info_score, silhouette_score
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import linear_sum_assignment

# Fit the K-means model to the data
kmeans = KMeans(n_clusters=K, random_state=rand_seed)
kmeans.fit(data)
label_kmeans = kmeans.labels_

# Save the predicted labels
np.save("label_kmeans.npy", label_kmeans)

# Function to calculate clustering accuracy
def clustering_accuracy(true_labels, predicted_labels):
    """
    Compute the accuracy by finding the best mapping between true and predicted labels.
    """
    # Create a contingency table
    contingency_matrix = np.zeros((K, K), dtype=int)
    for true, pred in zip(true_labels, predicted_labels):
        contingency_matrix[true, pred] += 1

    # Find the optimal mapping using the Hungarian algorithm
    row_ind, col_ind = linear_sum_assignment(-contingency_matrix)
    accuracy = contingency_matrix[row_ind, col_ind].sum() / len(true_labels)
    return accuracy

# Calculate accuracy and other metrics
ACC_kmeans = clustering_accuracy(labels, label_kmeans)
ARI_kmeans = adjusted_rand_score(labels, label_kmeans)
NMI_kmeans = normalized_mutual_info_score(labels, label_kmeans)
SC_kmeans = silhouette_score(data, label_kmeans)

# Print the results
print("K-means results: ")
print("ACC:", ACC_kmeans)
print("ARI:", ARI_kmeans)
print("NMI:", NMI_kmeans)
print("SC:", SC_kmeans)

# Ensure that n = 2 for visualization
if n == 2:
    # Define a common colormap and normalization
    cmap = plt.cm.viridis
    norm = plt.Normalize(vmin=0, vmax=K-1)  # Normalize to the number of clusters

    # Create a figure with two subplots: one for the true labels and one for the K-means predicted labels
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Plot the true labels using circles, with consistent colors
    scatter1 = axes[0].scatter(data[:, 0], data[:, 1], c=labels, cmap=cmap, norm=norm, s=20, alpha=0.6, marker='o')
    axes[0].set_title("True Labels")
    axes[0].set_xlabel("Dimension 1")
    axes[0].set_ylabel("Dimension 2")
    axes[0].grid(True)

    # Plot the K-means predicted labels using crosses (x marker), with consistent colors
    scatter2 = axes[1].scatter(data[:, 0], data[:, 1], c=label_kmeans, cmap=cmap, norm=norm, s=20, alpha=0.6, marker='x')
    axes[1].set_title("K-means Predicted Labels (Crosses)")
    axes[1].set_xlabel("Dimension 1")
    axes[1].set_ylabel("Dimension 2")
    axes[1].grid(True)

    # Show the plots
    plt.tight_layout()
    plt.show()

 