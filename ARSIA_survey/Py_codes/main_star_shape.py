"""
Solution path on star-shape data.

@author: Zheming Gao
"""
import numpy as np
import matplotlib.pyplot as plt
import Load_star_shape_dt

# Load star-shape data.
density = 1 # Set density to be less (1), medium (2) or high (3).

# density 1: (15, 12, 16)  n = 43
# density 2: (31, 20, 28)  n = 79
# density 3: (98, 60, 60)  n = 218

matrixData, y_true = Load_star_shape_dt.star_shape_dt(density)
p, n = matrixData.shape
id_1 = np.where(y_true == 1)[0]
id_2 = np.where(y_true == 2)[0]
id_3 = np.where(y_true == 3)[0]
true_cls_centers = np.array([np.mean(matrixData[:, id_1], axis = 1),
                             np.mean(matrixData[:, id_2], axis = 1),
                             np.mean(matrixData[:, id_3], axis = 1)])
# Plot the data.
data_plt = matrixData.T 
# Plot the generated data
# plt.figure(figsize=(8, 6))
plt.figure(dpi = 200)
# plot ground truth mean.
gt_mean_2d = true_cls_centers.T
plt.scatter(gt_mean_2d[0,:], gt_mean_2d[1,:], c='green', marker='x', label='GT-mean')


plt.scatter(data_plt[:, 0], data_plt[:, 1], c=y_true, cmap='RdYlBu', alpha=0.6, edgecolor='k')
plt.title("Generated star-shape")
plt.xlabel("x 1")
plt.ylabel("x 2")
plt.grid(True)
plt.show()

    
#%% Graph constructions.
import CC_graphs
import CC_split_algo
import time

# Assign all one weights.
"""
w_type:
1: inverse Euclidean distance based weights;
2: Gaussian distance based with Euclidean calcuated sigma_ij [Chi et al. 2019]
    default k_nrst = 5
3: naive all-one weights.
4: # ref: 2022, Dunlap and Mourrat, LOCAL VERSIONS OF SUM-OF-NORMS CLUSTERING
"""    
w_type = 1  # Set weight type as described above.
weight_mat0 = CC_graphs.assign_weights(matrixData, weights_type=w_type) 



#+++# Method: Fully connected.
# D, weights_vec = CC_graphs.create_graph_dense(matrixData, weight_mat0, print_G = 'y')


#+++# Method: MST.
# D, weights_vec = CC_graphs.create_graph_MST(matrixData, weight_mat0, print_G = 'y')

#+++# Method: kNNG.
# k_nrst = 2
# D, weights_vec = CC_graphs.create_graph_KNN(matrixData, weight_mat0, k_nrst, print_G = 'y')


#+++# Method: MST+kNNG
# k_nrst = 2
# D, weights_vec = CC_graphs.create_graph_MST_KNN(matrixData, weight_mat0, 
#                                                 k_nrst, print_G = 'y')

#+++# Method: DMST:
t = 3
D, weights_vec = CC_graphs.build_DMST(matrixData, weight_mat0, t = 3, 
                                                print_G = 'y')  # print the graph.

#+++# Method: epsilon-ball (question)
### eps = 1
# D, weights_vec = CC_graphs.build_eps_ball_graph(matrixData, weight_mat0, print_G = 'y')  # print the graph.



###**********++++++++++++++++++++++++++++++++
#-------------------
# Convex Clustering (Chi & Lange 2015)
#-------------------


# Define the number of frames
#------------------------------
# regular grids (all gamma > 0)
gamma_cand = 2 ** (np.array(range(-40, 110), dtype = float) / 4)
#------------------------------

# Loop through each frame
mat_x = matrixData
p, n = mat_x.shape

centroids_tensor = np.zeros([len(gamma_cand), n, 2])
Count_CPUt = time.time()   # record CPU time for all loops.

sol_admm = CC_split_algo.cvxclust_path_admm(mat_x, D, weights_vec, gamma_cand, nu=.2, tol_abs =1e-3, tol_rel= 1e-3,
                    max_iter=1000, norm_type=2, accelerate=True)

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


# Define a dictionary of custom colors
custom_colors = {1: 'red', 2: 'orange', 3: 'blue'}
# Convert class labels to colors
colors = [custom_colors[label] for label in y_true]

plt.figure(dpi = 200)
for j in range(n):
    # For each data point, plot its centroid route
    route_pts = centroids_tensor[:,j,:]  # numFrames by p.
    # Plot route for jth centroid.
    plt.plot(route_pts[:,0], route_pts[:,1], color = 'black', linestyle = '-', linewidth = 0.5)
# plot all data points.
# plt.scatter(data_plt[:,0], data_plt[:,1], c=y_true, cmap = 'RdYlBu', marker='o', label='Data Points')
plt.scatter(data_plt[:,0], data_plt[:,1], c=colors, marker='o', label='Data Points', alpha = 0.7)

# # plot ground truth mean.
# plt.scatter(gt_mean_2d[0,:], gt_mean_2d[1,:], c='green', marker='x', label='GT-mean')

    


ax = plt.gca()  # Get current axis

# Remove the frame (spines)
for spine in ax.spines.values():
    spine.set_visible(False)

# Remove ticks and tick labels
ax.set_xticks([])
ax.set_yticks([])
ax.set_xticklabels([])
ax.set_yticklabels([])

# plt.xlabel(r'$x_1$')
# plt.ylabel(r'$x_2$')
# plt.legend()
# plt.title(f'Convex clustering path (p = {p}, n = {n})')
# plt.grid(True)
plt.grid(False)
plt.show()
# plt.close