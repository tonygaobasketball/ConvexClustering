"""
Project: Convex clustering with general Moreau envelope.

Forward-backward splitting algorithm (matrix variables)


@author: Zheming Gao
"""

import funcs
import numpy as np
import time
import imageio
import scipy as sp
import matplotlib.pyplot as plt


def prox_mat_l21_admm(D, W0, gamma, rho, tol, max_iter):
    """
    Compute the 3rd iteration of the algorithm (matrix form)
    
    Parameters:
    D (numpy.ndarray): Matrix D.
    W0 (numpy.ndarray): Input matrix.
    gamma (float): Regularization parameter.
    rho (float): Penalty parameter for ADMM.
    max_iter (int): Maximum number of iterations.
    tol (float): Tolerance for convergence.
    
    Returns:
    numpy.ndarray: The optimized matrix V_opt.
    """
    
    # Initialize variables
    V = W0.copy()  # Initialize solution W
    p, n = V.shape  
    I_n = np.eye(n)
    U = V @ D.T  # Initialize variable U
    La = np.zeros_like(U)  # Initialize multiplier matrix
    A = (1 / gamma) * I_n + rho * (D.T @ D)
    
    # Soft-threshold function
    soft = lambda t, T: np.maximum(t - T, 0) + np.minimum(t + T, 0)
    
    # ADMM iterations
    for k in range(max_iter):
        # Update V^(k+1)
        BB =(1 / gamma) * W0 + rho * U @ D - La @ D
        # V_new = np.linalg.solve(A, B)
        #########
        # Try Chester's algorithm here.
        V_new = BB @ (np.linalg.inv(A))  # vanilla method.
        #########
        
        # introduce temp matrix 
        M_temp = La + rho * V_new @ D.T
        
        # Update U^(k+1)
        U_new = np.zeros(U.shape)
        for j in range(U.shape[1]):
            temp = 1 - 1/np.linalg.norm(M_temp[:,j],2)
            U[:,j] = max(0, temp) * M_temp[:,j] / rho
        
        
        # Update Lambda (Lagrangian multipliers)
        temp = V_new @ D.T - U_new
        La_new = La + rho * (temp)
        
        # Calculate the residuals.
        primal_res = np.linalg.norm(temp, 'fro')
        # dual_res =np.linalg.norm(rho * (U_new - U) @ D, 'fro')
        dual_res = np.linalg.norm(rho * (V_new - V), 'fro')
        
        
        # print(f"ADMM primal res: {primal_res} \n ADMM dual res: {dual_res}")
        
        # Check for convergence
        if primal_res < tol and dual_res < tol:
            print(f"ADMM Converged in {k+1} iterations.")
            break
        
        # # Check for convergence
        # if np.linalg.norm(V_new - V, 'fro') ** 2 < tol and np.linalg.norm(U_new - U, 'fro') ** 2 < tol:
        #     print(f"ADMM Converged in {k+1} iterations.")
        #     break
        
        # Update variables
        V = V_new
        U = U_new
        La = La_new
    
    # Calculate solution u = x + \gamma * z
    V_opt = V_new
    if k == max_iter - 1:
        print(f"ADMM reaches max step {max_iter}.")
    
    # iter_admm = k
    return V_opt



def MPD(V_0, C, A, alpha_k=0.5, tol=1e-8, max_iterations=500):
    """
    Matrix Proximal Descent 
    Compute the 1st iteration of the algorithm.
    
    
    Parameters:
    V0 (numpy.ndarray): Input matrix.
    C (numpy.ndarray): Matrix C.
    sig1_C (float):  the largest singular value of C.
    A (numpy.ndarray): Matrix U * D' * C'.
    alpha_k (float): Step size.
    max_iterations (int): Maximum number of iterations.
    tol (float): Tolerance for convergence.
    
    Returns:
    numpy.ndarray: The optimized matrix V.
    """
    
    V = V_0.copy()
    for k in range(max_iterations):
        Y = np.array(V @ (np.eye(C.shape[1]) - alpha_k * C.T @ C) - alpha_k * A @ C)
        
        V_new = np.zeros_like(V)
        for j in range(V.shape[1]):
            Y_j = Y[:, j]
            norm_Y_j = np.linalg.norm(Y_j, 2)
            # factor = max(0, 1 - alpha_k / (V.shape[1] * norm_Y_j))
            factor = max(0, 1 - alpha_k / norm_Y_j)
            V_new[:, j] = factor * Y_j
        
        # Check the stopping criterion
        if np.linalg.norm(V_new - V, 'fro') < tol:
            print(f"MPD Converged in {k+1} iterations.")
            break
        
        # Update V
        V = V_new

    return V

def fbs_gme3_cc_mat(D, C, sig1_C, gamma, U0, mat_x):
    """
    Forward-backward splitting (FBS) algorithm for convex clustering with generalized Moreau envelope (3rd model).
    Matrix form.
    
    Original FBS to solve 
    min f(x) + g(x)
    
    y^k = x^k - \mu_k \nabla g(x^k)
    z^k = Prox_{\mu_k f} (y^k)
    x^k+1 = x^k + \la_k (z^k - x^k)
    
    !!! Updated: 12/03/2024. 
    Update Info: 
       1. D_w = C @ D, replace D in the 2,1 norm.
       2. Use (28.37) on page 524 of Combettes' book.
    !!!
    
    
    Parameters:
    B (numpy.ndarray): Given matrix B.
    Dw (numpy.ndarray): Matrix Dw such that Dw = C D.
    sig1_C (float):  the largest singular value of C.
    gamma (float): Regularization parameter, gamma > 0.
    U0 (numpy.ndarray): Initial value of solution U.
    mat_X (numpy.ndarray): Data matrix.
    
    Returns:
    tuple: (u_opt, v_opt, z_opt, cput)
        - u_opt (numpy.ndarray): Solution u.
        - v_opt (numpy.ndarray): Solution v.
        - z_opt (numpy.ndarray): Solution z.
        - cput (float): CPU time running by the algorithm.
    """
    STOP_TOL = 1e-6
    Max_Iters = 10

    p,n = mat_x.shape  # record dimension of data.
    B = C @ D  # Define parameter B
    Dw = C @ D

    # Initialization
    U = U0.copy()
    UDw = U @ Dw.T
    U_old = U.copy()
    UDw_old = UDw.copy()

    # Matrix Proximal Descent parameters.
    V_old = np.ones((p,Dw.shape[0]))
    mat_A_MPD = UDw_old @ C.T
    step_size_MPD = 1.5 / (sig1_C ** 2)
    tol_MPD = 1e-4
    max_iter_MPD = 20
    
    # ADMM parameters.
    rho_admm = 10   # \rho-Lipschitz
    tol_admm = 1e-4
    max_iter_admm = 100

    iter = 0
    relative_delta = np.inf  # Stopping criteria value

    start_time = time.time()   # count CPU time.
    # mu_fbs (float): Parameter of FBS algorithm.
    # la_fbs (float): Parameter of FBS algorithm.
    mu_fbs = 1
    la_fbs = 1

    while (relative_delta > STOP_TOL) and (iter < Max_Iters):
        iter += 1
            
        # MPD iteration for V^k
        C_MPD = np.eye(C.shape[0])
        mat_A_MPD = UDw_old @ C_MPD.T
        V = MPD(V_old, C_MPD, mat_A_MPD, step_size_MPD, tol_MPD, max_iter_MPD)

        # iterate Z^k
        Z = (UDw_old - V) @ C.T @ B 
        
        W = (1 - mu_fbs) * U_old + mu_fbs * (mat_x + gamma * Z) 
        
        #********
        # ADMM iteration for M^k
        M = prox_mat_l21_admm(Dw, W, gamma, rho_admm, tol_admm, max_iter_admm)
        #********
        
        U = U_old + la_fbs * (M - U_old)
        
        # #********
        # # tCG for solving U^k+1 (By Chester Holtz)
        # La = prox_mat_l21_tcg(D, W, gamma)
        # U = W + La @ D
        # #********
        
        # Calculate residue
        # relative_delta = np.linalg.norm((U - U_old) @ Dw.T, 'fro') / np.linalg.norm(UDw_old, 'fro')
        relative_delta = np.linalg.norm((U - U_old), 'fro') 
        
        
        V_old = V.copy()
        # Z_old = Z.copy()
        U_old = U.copy()
        UDw_old =  U_old @ Dw.T
    
    if iter == Max_Iters:
        print(f"FBS reaches max step {Max_Iters}.")
    else:
        print(f"FBS Converged in {iter} iterations.")
    end_time = time.time()
    cput = end_time - start_time

    U_opt = U
    V_opt = V
    Z_opt = Z
    
    print(f'FBS res: {relative_delta}')
    
    return U_opt, V_opt, Z_opt, cput, relative_delta


def GME_CC_path(gamma_set, D, C, mat_x, true_cls_centers):
    
    """
    Input:
        gamma_set --- candidates of gamma (list)
        D --- incidence matrix (np arrray)
        C --- weight diagonal matrix (np array)
        mat_x --- data matrix p by n (np array)
    Output:
        solution_set --- U to each gamma (list)
        avg_CPUt --- average CPU time for solving the GME-CC once
        Plot the clustering paths
        
    """
    # Singular value of Ctilde
    sig1_C = np.linalg.norm(C, 2)
    # number of gamma candidate
    numFrames = len(gamma_set)

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
        
        #### Use warm start ###
        U0 = U_opt       # initialize U
     
        if p == 2:
            # Generating plots with different parameter gamma.
            # ===========
            # GME3-CC model
            
            # U_opt, V_opt, Z_opt, cput = FBS_algo.fbs_gme3_cc_mat(D, C, sig1_C, mu, rho, gamma, U0, mat_x)
            U_opt, V_opt, Z_opt, cput, re_tol = fbs_gme3_cc_mat(D, C, sig1_C, gamma, U0, mat_x)
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
            
            centroids_tensor[i,:,:] = U_sim.T
            # Plot paths using U_sim.

            
        if p > 2:
            Trans_mat = np.concatenate([np.eye(2), np.zeros([p-2,2])])  # p by 2 matrix
            # Generating plots with different parameter gamma.
            # ===========
            # GME3-CC model
            # U0 = mat_x       # initialize U
            U_opt, V_opt, Z_opt, cput, re_tol = fbs_gme3_cc_mat(D, C, sig1_C, gamma, U0, mat_x)
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
            
            centroids_tensor[i,:,:] = U_sim.T @ Trans_mat
            # Plot paths using U_sim.


    Count_CPUt = time.time() - Count_CPUt
    avg_CPUt = Count_CPUt / numFrames   # output it.
    
    ## Plot clustering path.
    if p == 2: 
        plt.figure()
        
        for j in range(n):
            # For each data point, plot its centroid route
            route_pts = centroids_tensor[:,j,:]  # numFrames by p.
            # Plot route for jth centroid.
            plt.plot(route_pts[:,0], route_pts[:,1], color = 'green', linestyle = '-', linewidth = 0.5)
        
        # plot all data points.
        plt.scatter(mat_x.T[:,0], mat_x.T[:,1], c='b', marker='*', label='Data Points')
        # Add generated ground-true means to the plot
        gt_mean_2d = np.array(true_cls_centers)[:,:2]
        plt.scatter(gt_mean_2d[:,0], gt_mean_2d[:,1], c='r', marker='o', label='GT-mean')
        
        plt.xlabel('X1')
        plt.ylabel('X2')
        plt.legend()
        # plt.title('toy4-5c cluster path (k_near = {})'.format(k_nearest))
        plt.title('GME-CC path')
        plt.show()
        # plt.close
        
    if p > 2:  ## Plot clustering path (first 2d)
        data_2d = mat_x.T @ Trans_mat
        
        plt.figure()
        
        for j in range(n):
            # For each data point, plot its centroid route
            route_pts = centroids_tensor[:,j,:]  # already 2d.
            # Plot route for jth centroid.
            plt.plot(route_pts[:,0], route_pts[:,1], color = 'green', linestyle = '-', linewidth = 0.5)
        
        # plot all data points.
        plt.scatter(data_2d[:,0], data_2d[:,1], c='b', marker='*', label='Data Points')
        # Add generated ground-true means to the plot
        gt_mean_2d = np.array(true_cls_centers)[:,:2]
        plt.scatter(gt_mean_2d[:,0], gt_mean_2d[:,1], c='r', marker='o', label='GT-mean')
        
        plt.xlabel('Component 1')
        plt.ylabel('Component 2')
        plt.legend()
        # plt.title('toy4-5c cluster path (k_near = {})'.format(k_nearest))
        plt.title('GME-CC path (first 2d)')
        plt.show()
        # plt.close
    
    return avg_CPUt, centroids_tensor



def fbs_gme3_cc_mat_lpn(D, C, sig1_C, gamma, U0, mat_x):
    """
    Forward-backward splitting (FBS) algorithm for convex clustering with generalized Moreau envelope (3rd model).
    Matrix form.
    
    Original FBS to solve 
    min f(x) + g(x)
    
    y^k = x^k - \mu_k \nabla g(x^k)
    z^k = Prox_{\mu_k f} (y^k)
    x^k+1 = x^k + \la_k (z^k - x^k)
    
    !!! Updated: 12/03/2024. 
    Update Info: 
       1. D_w = C @ D, replace D in the 2,1 norm.
       2. Use (28.37) on page 524 of Combettes' book.
    !!!
    
    
    Parameters:
    B (numpy.ndarray): Given matrix B.
    Dw (numpy.ndarray): Matrix Dw such that Dw = C D.
    sig1_C (float):  the largest singular value of C.
    gamma (float): Regularization parameter, gamma > 0.
    U0 (numpy.ndarray): Initial value of solution U.
    mat_X (numpy.ndarray): Data matrix.
    
    Returns:
    tuple: (u_opt, v_opt, z_opt, cput)
        - u_opt (numpy.ndarray): Solution u.
        - v_opt (numpy.ndarray): Solution v.
        - z_opt (numpy.ndarray): Solution z.
        - cput (float): CPU time running by the algorithm.
    """
    STOP_TOL = 1e-6
    Max_Iters = 10

    p,n = mat_x.shape  # record dimension of data.
    B = C @ D  # Define parameter B
    Dw = C @ D

    # Initialization
    U = U0.copy()
    UDw = U @ Dw.T
    U_old = U.copy()
    UDw_old = UDw.copy()

    # Matrix Proximal Descent parameters.
    V_old = np.ones((p,Dw.shape[0]))
    mat_A_MPD = UDw_old @ C.T
    step_size_MPD = 1.5 / (sig1_C ** 2)
    tol_MPD = 1e-4
    max_iter_MPD = 20
    
    # ADMM parameters.
    rho_admm = 10   # \rho-Lipschitz
    tol_admm = 1e-4
    max_iter_admm = 100

    iter = 0
    relative_delta = np.inf  # Stopping criteria value

    start_time = time.time()   # count CPU time.
    # mu_fbs (float): Parameter of FBS algorithm.
    # la_fbs (float): Parameter of FBS algorithm.
    mu_fbs = 1
    la_fbs = 1

    while (relative_delta > STOP_TOL) and (iter < Max_Iters):
        iter += 1
            
        # MPD iteration for V^k
        C_MPD = np.eye(C.shape[0])
        mat_A_MPD = UDw_old @ C_MPD.T
        V = MPD(V_old, C_MPD, mat_A_MPD, step_size_MPD, tol_MPD, max_iter_MPD)

        # iterate Z^k
        Z = (UDw_old - V) @ C.T @ B 
        
        W = (1 - mu_fbs) * U_old + mu_fbs * (mat_x + gamma * Z) 
        
        #********
        # ADMM iteration for M^k
        M = prox_mat_l21_admm(Dw, W, gamma, rho_admm, tol_admm, max_iter_admm)
        #********
        
        U = U_old + la_fbs * (M - U_old)
        
        # #********
        # # tCG for solving U^k+1 (By Chester Holtz)
        # La = prox_mat_l21_tcg(D, W, gamma)
        # U = W + La @ D
        # #********
        
        # Calculate residue
        # relative_delta = np.linalg.norm((U - U_old) @ Dw.T, 'fro') / np.linalg.norm(UDw_old, 'fro')
        relative_delta = np.linalg.norm((U - U_old), 'fro') 
        
        
        V_old = V.copy()
        # Z_old = Z.copy()
        U_old = U.copy()
        UDw_old =  U_old @ Dw.T
    
    if iter == Max_Iters:
        print(f"FBS reaches max step {Max_Iters}.")
    else:
        print(f"FBS Converged in {iter} iterations.")
    end_time = time.time()
    cput = end_time - start_time

    U_opt = U
    V_opt = V
    Z_opt = Z
    
    print(f'FBS res: {relative_delta}')
    
    return U_opt, V_opt, Z_opt, cput, relative_delta