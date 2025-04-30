"""
Plot clustering paths with different level of weights.

@author: Zheming Gao
"""

import numpy as np
import matplotlib.pyplot as plt
from itertools import combinations
import networkx as nx
from matplotlib.ticker import MaxNLocator
#%% # Generating data:

def generate_clusters_unit_ball(p, n_k, K, clusterCenters, rand_seed):
    """
    Generate K clusters of p-dimensional data points. Each cluster has n_k data points
    sampled uniformly from a unit ball centered at the specified clusterCenters.

    Parameters:
    - p (int): Dimension of the data space.
    - n_k (int): Number of points per cluster.
    - K (int): Number of clusters.
    - clusterCenters (np.ndarray): Shape (p, K), each column is a cluster center.
    - rand_seed (int): Random seed for reproducibility.

    Returns:
    - clusters: List of dictionaries, each containing 'points' for the cluster.
    - labels:  list of labels in integers.
    """
    np.random.seed(rand_seed)
    clusters = []
    labels = []

    def sample_unit_ball(d, n):
        """
        Sample n points uniformly from a d-dimensional unit ball.
        """
        vec = np.random.randn(d, n)
        vec /= np.linalg.norm(vec, axis=0, keepdims=True)  # normalize to unit sphere
        radius = np.random.rand(n) ** (1.0 / d)  # radius scaling for uniform distribution
        return vec * radius

    for i in range(K):
        # Generate uniformly sampled points in unit ball
        unit_ball_samples = sample_unit_ball(p, n_k)
        cluster_center = clusterCenters[:, i].reshape(p, 1)
        cluster_data = unit_ball_samples + cluster_center  # shift to center
        label = i * np.ones(n_k)
        clusters.append({'points': cluster_data})
        labels.append(label)

    return clusters, labels






# Function to create the graph with weighted edges

def create_full_level_weighted_graph(matrixData, K1, K2, n_k, level_weights):
    G = nx.Graph()
    
    # Add nodes
    for i in range(matrixData.shape[1]):
        G.add_node(i, pos=(matrixData[0, i], matrixData[1, i]))

    # Add edges within each cluster (assign weight = 1, color = 'black')
    for i in range(K1 + K2):
        start_idx = i * n_k
        end_idx = (i + 1) * n_k
        for j in range(start_idx, end_idx):
            for k in range(j + 1, end_idx):
                G.add_edge(j, k, weight=level_weights[3], color='black')

    # Add edges between the left-top clusters 
    left_top_indices = list(range(3 * n_k))  # First 3 clusters
    for i in left_top_indices:
        for j in left_top_indices:
            if i < j and not G.has_edge(i, j):  # Avoid overwriting existing edges
                G.add_edge(i, j, weight=level_weights[1], color='lightblue')

    # Add edges between the right-bottom clusters (assign weight = 0.5, color = 'lightblue')
    right_bottom_indices = list(range(3 * n_k, 5 * n_k))  # Last 2 clusters
    for i in right_bottom_indices:
        for j in right_bottom_indices:
            if i < j and not G.has_edge(i, j):  # Avoid overwriting existing edges
                G.add_edge(i, j, weight=level_weights[2], color='lightblue')

    # Add edges between left-top clusters and right-bottom clusters (assign weight = 0.1, color = 'gray')
    for i in left_top_indices:
        for j in right_bottom_indices:
            G.add_edge(i, j, weight=level_weights[0], color='gray', style='dashed')

    # Function to plot the graph with the correct edge colors
    pos = nx.get_node_attributes(G, 'pos')
    edges = G.edges(data=True)
    edge_colors = [edge[2]['color'] for edge in edges]

    plt.figure(figsize=(10, 10))
    nx.draw(G, pos, with_labels=False, node_size=300, node_color='lightgreen', font_size=10, edge_color=edge_colors, width=2)
    plt.title('Graph with Weighted Edges Based on Clusters')
    plt.show()
    
    
    p, n = matrixData.shape

    # Step 1: Compute Pairwise Euclidean Distances (For Graph Structure)
    # distances = squareform(pdist(matrixData.T, metric='euclidean'))

    # Step 2: Generate All Possible Edges (Fully Connected Graph)
    edges = [(i, j) for i, j in combinations(range(n), 2)]  # Unique node pairs

    # Step 3: Construct Incidence Matrix D_dense
    m = len(edges)  # Number of edges in fully connected graph
    D_dense = np.zeros((m, n))

    for idx, (i, j) in enumerate(edges):
        D_dense[idx, i] = 1
        D_dense[idx, j] = -1

    # Step 4: Extract Weights from `weight_mat` (Consistent with `D_dense`)
    # w_dense = np.array([weight_mat[i, j] for i, j in edges])
    w_dense = np.array([G[i][j]['weight'] for i, j in edges])
    

    return D_dense, w_dense

#--------------------------------
# Data 2
#--------------------------------
# Generate toy3-5c example

# Parameters
K1 = 3
n_k = 6
p = 2
cls_centers = np.array([[-5, 0], [-5, 5], [0, 5]]).T

# Generate the clusters
rnd_seed = 7
data1, label1 = generate_clusters_unit_ball(p, n_k, K1, cls_centers, rnd_seed)
n1 = n_k * K1

# Parameters
K2 = 2
n_k = 6
p = 2
cls_centers = np.array([[5, -1], [1, -5]]).T

# Generate the clusters
rnd_seed = 6
data2, label2 = generate_clusters_unit_ball(p, n_k, K2, cls_centers, rnd_seed)
n2 = n_k * K2



# Stacking data1 and data2
mat1 = np.hstack([data1[i]['points'] for i in range(K1)])
mat2 = np.hstack([data2[i]['points'] for i in range(K2)])
matrixData = np.hstack([mat1, mat2])
# record labels
y1 = np.concatenate([label1[i] for i in range(K1)])
y2 = np.concatenate([label2[i] for i in range(K2)])
y_true = np.concatenate([y1, np.max(y1) + 1 + y2])


# Record the dimension of the data.
p, n =  matrixData
data_plt = matrixData.T 

# Plot the data
plt.figure(dpi = 200)
plt.scatter(matrixData[0, :], matrixData[1, :], color = 'blue', marker = 'o', alpha = 0.7)
plt.xlabel('x1')
plt.ylabel('x2')
# plt.title('Generated level-weight Data')
# plt.grid(True)
plt.show()
#--------------------------------



#%% Graph constructions.
import CC_split_algo
import time

# Assign level weights, on full graph.


# Assuming data has been generated with the previous code
# Create the graph
level_weights = [0.1, 1, 1, 10]
# level_weights = [0.1, 1, 1, 1]
# level_weights = [1, 1, 1, 1]
# level_weights = [1, 1, 1, 10]
D, weights_vec = create_full_level_weighted_graph(matrixData, K1, K2, n_k, level_weights)






###**********++++++++++++++++++++++++++++++++
#-------------------
# Convex Clustering (Chi & Lange 2015)
#-------------------

# Define the number of frames
#------------------------------
# regular grids (all gamma > 0)
gamma_cand = 2 ** (np.array(range(-44, 8), dtype = float) / 4)
#------------------------------

# Loop through each frame
mat_x = matrixData
p, n = mat_x.shape



centroids_tensor = np.zeros([len(gamma_cand), n, 2])
Count_CPUt = time.time()   # record CPU time for all loops.

sol_admm = CC_split_algo.cvxclust_path_admm(mat_x, D, weights_vec, gamma_cand, nu=.1, tol_abs =1e-6, tol_rel= 1e-6,
                    max_iter=1000, norm_type=2, accelerate=True)
# sol_admm contains: 
    # {"U": list_U, "V": list_V, "Lambda": list_Lambda,
    #       "nGamma": len(gamma_seq), "iters": iter_vec}
Count_CPUt = time.time() - Count_CPUt
avg_CPUt_CC = Count_CPUt / len(gamma_cand)   # output it.
tol_sim_path = 1e-3
Dist = np.zeros([len(gamma_cand), n])
CC_num_K = np.zeros(len(gamma_cand))   # record number of clusters.

# record clustering centroids for each gamma.
for i in range(len(gamma_cand)):
    U_opt = sol_admm['U'][i]
    
    V_opt = sol_admm['V'][i]

    K_i, uni_label_i = CC_split_algo.CC_num_clusters(U_opt, V_opt, D, tol_sim_path)
    uni_clusters_id = np.unique(uni_label_i)
    # Take the mean of similar columns of U. Save as U_sim

    U_sim = U_opt.copy()
    for kk in list(uni_clusters_id):
        cluster_idx = np.where(uni_label_i == kk)[0]
        col_mean = np.mean(U_opt[:,cluster_idx], axis = 1)
        U_sim[:, cluster_idx] = np.tile(col_mean, (len(cluster_idx), 1)).T
    # record clustering centroids for each gamma.
    centroids_tensor[i,:,:] = U_sim.T
    


plt.figure(dpi = 200)
for j in range(n):
    # For each data point, plot its centroid route
    route_pts = centroids_tensor[:,j,:]  # numFrames by p.
    # Plot route for jth centroid.
    plt.plot(route_pts[:,0], route_pts[:,1], color = 'black', linestyle = '-', linewidth = 0.5)


# plot all data points. (different colors and shapes)

plt.scatter(data_plt[y_true == 0,0], data_plt[y_true == 0,1], c = 'blue', marker='o', label='Data Points', alpha = 0.7)
plt.scatter(data_plt[y_true == 1,0], data_plt[y_true == 1,1], c = 'blue', marker='x', label='Data Points', alpha = 0.7)
plt.scatter(data_plt[y_true == 2,0], data_plt[y_true == 2,1], c = 'blue', marker='^', label='Data Points', alpha = 0.7)

plt.scatter(data_plt[y_true == 3,0], data_plt[y_true == 3,1], c = 'red', marker='^', label='Data Points', alpha = 0.7)
plt.scatter(data_plt[y_true == 4,0], data_plt[y_true == 4,1], c = 'red', marker='o', label='Data Points', alpha = 0.7)



# # plot ground truth mean.
# plt.scatter(gt_mean_2d[0,:], gt_mean_2d[1,:], c='green', marker='x', label='GT-mean')

    
# plt.xlabel(r'$x_1$')
# plt.ylabel(r'$x_2$')
# plt.legend()
# plt.title(f'Convex clustering path (p = {p}, n = {n})')

ax = plt.gca()
ax.xaxis.set_major_locator(MaxNLocator(nbins=5))  # Limit to ~4 x-ticks
ax.yaxis.set_major_locator(MaxNLocator(nbins=5))  # Limit to ~4 y-ticks

# plt.xticks(np.arange(-2, 4, 2))
# plt.yticks(np.arange(-2, 2.3, 2))
plt.tick_params(axis='both', labelsize=16)  # Increase tick label font size


plt.grid(False)
plt.show()
# plt.close

