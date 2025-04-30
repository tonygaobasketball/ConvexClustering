"""
Half-moon dataset
Clustering paths

@author: Zheming Gao
"""

import numpy as np
import matplotlib.pyplot as plt
import time
import CC_split_algo



# Function: half_moon data.
def generate_2half_moons(num_points=8, cluster_points=7, cluster_radius=0.1, vertical_scale=0.5, random_seed = 1):
    """
    Generate a 2 half-moons dataset with moderately loose clusters.

    Parameters:
    - num_points (int): Number of core points in each half-moon.
    - cluster_points (int): Number of additional points per core point.
    - cluster_radius (float): Radius for generating clusters around each core point.
    - vertical_scale (float): Scaling factor for the vertical height of the moons.
    - random_seed (int): Random seed for reproducibility.

    Returns:
    - dataX (np.ndarray): Data matrix (2, n), where each column is a data point.
    - labels (np.ndarray): Label array (n,), with labels 0 and 1 for each half-moon.
    - GTmeans (np.ndarray): Ground-truth mean matrix (2, n), where each column is the ground-truth mean for each data point.
    - means (np.ndarray): Ground-truth means of each cluster (2, 2), each column is a mean for one cluster.
    """
    np.random.seed(random_seed)

    # Generate first half-moon (label = 0)
    theta1 = np.linspace(0, np.pi, num_points)
    x1 = np.cos(theta1)
    y1 = vertical_scale * np.sin(theta1)

    # Generate second half-moon (label = 1)
    theta2 = np.linspace(0, np.pi, num_points)
    x2 = 1 - np.cos(theta2)
    y2 = vertical_scale * (-np.sin(theta2) - .5)

    # Combine the two half-moons
    X1 = np.vstack((x1, y1)).T  # Shape (num_points, 2)
    X2 = np.vstack((x2, y2)).T  # Shape (num_points, 2)
    X = np.vstack((X1, X2))  # Shape (2 * num_points, 2)

    # Compute cluster means
    mean1 = np.mean(X1, axis=0)
    mean2 = np.mean(X2, axis=0)
    means = np.vstack((mean1, mean2))  # Shape (2, 2)

    # Generate clusters around each point
    cluster_data = []
    GTmeans = []
    labels = []
    for i in range(X.shape[0]):
        cluster = X[i, :] + cluster_radius * np.random.randn(cluster_points, 2)
        cluster_data.append(cluster)
        
        # Assign GT mean to each generated data point
        GT_mean = mean1 if i < num_points else mean2  # First half-moon gets mean1, second gets mean2
        GTmeans.append(np.tile(GT_mean, (cluster_points, 1)))

        labels.extend([0] * cluster_points if i < num_points else [1] * cluster_points)  # First half-moon gets 0, second gets 1

    # Convert list of clusters into a single matrix
    cluster_data = np.vstack(cluster_data)  # Shape (num_points * 2 * cluster_points, 2)
    GTmeans = np.vstack(GTmeans)  # Shape (num_points * 2 * cluster_points, 2)

    # Reshape into (p, n) format
    dataX = cluster_data.T  # Shape (2, n)
    GTmeans = GTmeans.T  # Shape (2, n)
    labels = np.array(labels)  # Shape (n,)

    return dataX, labels, GTmeans, means



######################################################
# Generate Half-moons data
num_points = 5       # Number of core points in each half-moon.
cluster_points = 4    # 
cluster_radius = 0.17   
vertical_scale = 1.2
rand_seed = 6
dataX, labels, ground_true_means, means = generate_2half_moons(num_points, cluster_points, cluster_radius, vertical_scale, random_seed=rand_seed)

true_cls_centers = means
GTmeans = ground_true_means

# Record the dimension of the data.
p, n =  dataX.shape
######################################################



matrixData = dataX
# y_true = np.array(range(len(labels)))
y_true = labels
p, n = matrixData.shape
print(f'{p}, {n}, {len(np.unique(labels))}')

# p == 2 for half-moon data.
data_plt = matrixData.T

# Plot the generated data
plt.figure(dpi = 200)
# plot ground truth mean.
gt_mean_2d = np.array(true_cls_centers)
plt.scatter(gt_mean_2d[:,0], gt_mean_2d[:,1], c='green', marker='x', label='GT-mean')


plt.scatter(data_plt[:, 0], data_plt[:, 1], c=y_true, cmap='RdYlBu', alpha=0.6, edgecolor='k')
plt.title("Generated Gaussian Clusters")
plt.xlabel("x 1")
plt.ylabel("x 2")
plt.grid(True)
plt.show()




#%% Graph constructions.
import CC_graphs

# Assign all one weights.
"""
w_type:
1: inverse Euclidean distance based weights;
2: Gaussian distance based with Euclidean calcuated sigma_ij [Chi et al. 2019]
    default k_nrst = 3
3: naive all-one weights.
"""    
for w_type in [1,2,3]:

# w_type = 1  # Use uniform weights
    weight_mat0 = CC_graphs.assign_weights(matrixData, weights_type=w_type) 
    
    #+++# Method: Fully connected.
    # D, weights_vec = CC_graphs.create_graph_dense(matrixData, weight_mat0, print_G = 'y')
    
    
    #+++# Method: MST.
    D, weights_vec = CC_graphs.create_graph_MST(matrixData, weight_mat0, print_G = 'y')
    
    #+++# Method: kNNG.
    # k_nrst = 3
    # D, weights_vec = CC_graphs.create_graph_KNN(matrixData, weight_mat0, k_nrst, print_G = 'y')
    
    
    #+++# Method: MST+kNNG
    # k_nrst = 3
    # D, weights_vec = CC_graphs.create_graph_MST_KNN(matrixData, weight_mat0, 
    #                                                 k_nrst, print_G = 'y')
    
    #+++# Method: DMST:
    # t = 3
    # D, weights_vec = CC_graphs.build_DMST(matrixData, weight_mat0, t, 
    #                                                 print_G = 'y')  # print the graph.
    
    # #+++# Method: epsilon-ball (question)
    # ### eps = 1
    # D, weights_vec = CC_graphs.build_eps_ball_graph(matrixData, weight_mat0, print_G = 'y')  # print the graph.
    
    
    
    
    
    
    ###**********++++++++++++++++++++++++++++++++
    #-------------------
    # Convex Clustering (Chi & Lange 2015)
    #-------------------
    
    
    # Define the number of frames
    #------------------------------
    # regular grids (all gamma > 0)
    gamma_cand = 2 ** (np.array(range(-50, 80), dtype = float) / 4)
    #------------------------------
    
    # Loop through each frame
    mat_x = matrixData
    p, n = mat_x.shape

    centroids_tensor = np.zeros([len(gamma_cand), n, 2])
    Count_CPUt = time.time()   # record CPU time for all loops.
    
    sol_admm = CC_split_algo.cvxclust_path_admm(mat_x, D, weights_vec, gamma_cand, nu=.2, tol_abs =1e-3, tol_rel= 1e-3,
                        max_iter=1000, norm_type=2, accelerate=True)
    # sol_admm contains: 
        # {"U": list_U, "V": list_V, "Lambda": list_Lambda,
        #       "nGamma": len(gamma_seq), "iters": iter_vec}
    Count_CPUt = time.time() - Count_CPUt
    avg_CPUt_CC = Count_CPUt / len(gamma_cand)   # output it.
    tol_sim_path = 1e-6
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
    # plot all data points.
    plt.scatter(data_plt[:,0], data_plt[:,1], c=y_true, cmap = 'RdYlBu', marker='o', label='Data Points')
    
    # plot ground truth mean.
    gt_mean_2d = np.array(true_cls_centers)
    plt.scatter(gt_mean_2d[:,0], gt_mean_2d[:,1], c='green', marker='x', label='GT-mean')
    
        
    # plt.xlabel(r'$x_1$')
    # plt.ylabel(r'$x_2$')
    # plt.legend()
    # plt.title(f'Convex clustering path (p = {p}, n = {n})')
    
    ax = plt.gca()  # Get current axis
    
    # Remove the frame (spines)
    for spine in ax.spines.values():
        spine.set_visible(False)
    
    # Remove ticks and tick labels
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    
    plt.grid(False)
    plt.show()
    # plt.close
