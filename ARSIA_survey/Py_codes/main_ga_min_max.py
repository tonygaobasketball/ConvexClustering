import numpy as np
import matplotlib.pyplot as plt
import time
import CC_split_algo
import CC_graphs
from matplotlib.ticker import MaxNLocator


def disjoint_2balls(N_in, r, seed=1):
    """
    Generates two 2D clusters, each within a unit ball and adds 8 perimeter points per cluster.
    
    Parameters:
    - N_in: int, number of random points in each cluster
    - r: float, distance of cluster centers from origin along x-axis
    - seed: int, for reproducibility
    
    Returns:
    - data: np.ndarray of shape (2, 2 * N_in + 16), columns are data points
    - labels: np.ndarray of shape (2 * N_in + 16,), 0 for left cluster, 1 for right cluster
    """
    np.random.seed(seed)
    
    def sample_unit_ball(n):
        theta = np.random.uniform(0, 2 * np.pi, n)
        radius = np.sqrt(np.random.uniform(0, 1, n))  # sqrt for uniform density
        x = radius * np.cos(theta)
        y = radius * np.sin(theta)
        return np.vstack((x, y))

    # Random points inside unit balls
    left_cluster = sample_unit_ball(N_in) + np.array([[-r], [-1]])
    right_cluster = sample_unit_ball(N_in) + np.array([[r], [1]])
    
    # # Perimeter points
    # left_perim = perimeter_points(center=[-r, -1])
    # right_perim = perimeter_points(center=[r, 1])
    
    # Combine all data
    # data = np.hstack((left_cluster, right_cluster, left_perim, right_perim))
    data = np.hstack((left_cluster, right_cluster))
    
    # Labels: 0 for left, 1 for right
    # labels = np.array([0] * N_in + [1] * N_in + [0] * 8 + [1] * 8)
    labels = np.array([0] * N_in + [1] * N_in)
    
    return data, labels


######################################################
# Generate two disjoint balls
# Generate Half-moons data
n_in_pts = 10       # Number of core points in each half-moon.

# rr = 0.4   # distance between ball center and origin.
rr = 1.45  
# rr = 2

rnd_seed = 3

# Set alpha for convex combination of the weights.
## weight_mat = al * weight_Gau + (1 - al) * weight_Uni
# al_cand = [0]
al_cand = [0, 0.7, 0.95, 1]  # Only uniform weights
# al_cand = [1]


for alal in al_cand:

    matrixData, y_true = disjoint_2balls(n_in_pts, rr, rnd_seed)
    
    
    p, n = matrixData.shape
    id_0 = np.where(y_true == 0)[0]
    id_1 = np.where(y_true == 1)[0]
    true_cls_centers = np.array([np.mean(matrixData[:, id_0], axis = 1),
                                 np.mean(matrixData[:, id_1], axis = 1)])
    
    
    
    # Plot the data.
    data_plt = matrixData.T 
    gt_mean_2d = true_cls_centers.T
    
    
    # # Plot the generated data
    
    # plt.figure(dpi = 200)
    # # plot ground truth mean.
    # plt.scatter(gt_mean_2d[0,:], gt_mean_2d[1,:], c='green', marker='x', label='GT-mean')
    
    
    # plt.scatter(data_plt[:, 0], data_plt[:, 1], c=y_true, cmap='RdYlBu', alpha=0.6, edgecolor='k')
    # plt.title("Generated disjoint balls")
    # plt.xlabel("x 1")
    # plt.xlim([-3,3])
    # plt.ylabel("x 2")
    # plt.ylim([-3,3])
    # plt.grid(True)
    # plt.show()
    
    
    ######################################################
    
    
    

    
    # Assign all one weights.
    """
    w_type:
    1: inverse Euclidean distance based weights;
    2: Gaussian distance based with Euclidean calcuated sigma_ij [Chi et al. 2019]
        default k_nrst = 10% * n
    3: naive all-one weights.
    4: # ref: 2022, Dunlap and Mourrat, LOCAL VERSIONS OF SUM-OF-NORMS CLUSTERING
    """    
    # w_type = 2  # Set weight type as described above.
    # weight_mat0 = CC_graphs.assign_weights(matrixData, weights_type=w_type) 
    weight_mat0 = CC_graphs.conv_comb_weights(matrixData, al = alal)
    
    
    #+++# Method: Fully connected.
    D, weights_vec = CC_graphs.create_graph_dense(matrixData, weight_mat0, print_G = 'n')
    
    
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
    # t = 3
    # D, weights_vec = CC_graphs.build_DMST(matrixData, weight_mat0, t = 3, 
    #                                                 print_G = 'n')  # print the graph.
    
    #+++# Method: epsilon-ball (question)
    ### eps = 1
    # D, weights_vec = CC_graphs.build_eps_ball_graph(matrixData, weight_mat0, print_G = 'y')  # print the graph.
    
    
    
    ####%%
    #-------------------
    # Convex Clustering (Chi & Lange 2015)
    #-------------------
    
    
    ###**********++++++++++++++++++++++++++++++++
    # Calculate C and B, and their singular values.
    id_set = [id_0, id_1]
    ga_lb, ga_ub, ga_check = CC_graphs.compute_gamma_min_max(matrixData, weight_mat0, true_cls_centers.T, id_set)
    print(f"gamma_min: {ga_lb:.4f} \n gamma_max: {ga_ub:.4f}" )
    
    # Define the number of frames
    #------------------------------
    # regular grids (all gamma > 0)
    gamma_cand = 2 ** (np.array(range(-40, 240), dtype = float) / 4)
    #------------------------------
    
    # Loop through each frame
    mat_x = matrixData
    p, n = mat_x.shape
    # recover full weights with length n(n-1)/2
    # weights_full = CC_split_algo.recover_full_weights(D, weights_vec)
    
    centroids_tensor = np.zeros([len(gamma_cand), n, 2])
    Count_CPUt = time.time()   # record CPU time for all loops.
    
    sol_admm = CC_split_algo.cvxclust_path_admm(mat_x, D, weights_vec, gamma_cand, nu=.2, tol_abs =1e-3, tol_rel= 1e-3,
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
        
    
    # Define a dictionary of custom colors
    custom_colors = {0: 'red',  1: 'blue'}
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
    
    plt.title(rf'$\gamma_L^w$ = {ga_lb:.3f}, $\gamma_U^w$ = {ga_ub:.3f}', fontsize=16)
    # plt.title(rf'$\alpha$ = {alal:.2f}', fontsize=16)
    plt.grid(False)
    

    # # Remove the frame (spines)
    # for spine in ax.spines.values():
    #     spine.set_visible(False)

    # Remove ticks and tick labels
    # ax.set_xticks([])
    # ax.set_yticks([])
    # ax.set_xticklabels([])
    # ax.set_yticklabels([])
    
    ax = plt.gca()
    ax.xaxis.set_major_locator(MaxNLocator(nbins=5))  # Limit to ~4 x-ticks
    ax.yaxis.set_major_locator(MaxNLocator(nbins=5))  # Limit to ~4 y-ticks

    # plt.xticks(np.arange(-2, 4, 2))
    # plt.yticks(np.arange(-2, 2.3, 2))
    plt.tick_params(axis='both', labelsize=16)  # Increase tick label font size

    plt.show()
    # plt.close
