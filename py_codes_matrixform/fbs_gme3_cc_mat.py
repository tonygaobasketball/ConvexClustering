"""
Project: Convex clustering with general Moreau envelope.

Forward-backward splitting algorithm (matrix variables)


@author: Zheming Gao
"""

from Funcs_FBS_mat import *

import numpy as np
import time

def fbs_gme3_cc_mat(D, C, sig1_C, mu, rho, gamma, U0, mat_x):
    """
    Forward-backward splitting (FBS) algorithm for convex clustering with generalized Moreau envelope (3rd model).
    Matrix form.
    
    Parameters:
    D (numpy.ndarray): Given matrix D.
    C (numpy.ndarray): Matrix C such that B = C D.
    sig1_C (float):  the largest singular value of C.
    mu (float): Parameter of FBS algorithm.
    rho (float): Lipschitz constant.
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
    Max_Iters = 2000

    p,n = mat_x.shape  # record dimension of data.
    B = C @ D  # Define parameter B

    # Initialization
    U = U0.copy()
    UD = U @ D.T
    U_old = U.copy()
    UD_old = UD.copy()

    # Matrix Proximal Descent parameters.
    V_old = np.ones((p,D.shape[0]))
    mat_A_MPD = UD_old @ C.T
    step_size_MPD = 1.5 / (sig1_C ** 2)
    tol_MPD = 1e-6
    max_iter_MPD = 1000
    
    # ADMM parameters.
    rho_admm = 2.5
    tol_admm = 1e-6
    max_iter_admm = 1000
    
    # # ISTA parameters
    # v = np.ones((n-1) * p)
    # sig1_C = np.linalg.norm(C.toarray(), 2)
    # # sig1_C = np.linalg.norm(C, 2)  # Largest singular value of C
    # alpha_ista = .5 * sig1_C ** 2 * 1.1  # alpha >= 1/2 * \lambda_max (C^T C)
    # Nit_ista = 10
    # soft = lambda t, T: np.maximum(t - T, 0) + np.minimum(t + T, 0)  # Soft-threshold function

    # rho_admm = 1.5
    # max_iter_admm = 2000

    iter = 0
    relative_delta = np.inf  # Stopping criteria value

    start_time = time.time()

    while (relative_delta > STOP_TOL) and (iter < Max_Iters):
        iter += 1
            
        # MPD iteration for V^k
        mat_A_MPD = UD_old @ C.T
        V = MPD(V_old, C, mat_A_MPD, step_size_MPD, tol_MPD, max_iter_MPD)

        # iterate Z^k
        Z = (UD_old - V) @ C.T @ B 
        
        W = mat_x + gamma * Z
        # #********
        # # ADMM iteration for U^k+1
        # U = prox_mat_l21_admm(D, W, gamma, rho_admm, tol_admm, max_iter_admm)
        # #********
        
        #********
        # tCG for solving U^k+1 (By Chester Holtz)
        La = prox_mat_l21_tcg(D, W, gamma)
        U = W - La @ D
        #********
        
        # Calculate residue
        relative_delta = np.linalg.norm((U - U_old) @ D.T, 'fro') / np.linalg.norm(UD_old, 'fro')

        V_old = V.copy()
        # Z_old = Z.copy()
        U_old = U.copy()
        UD_old =  U_old @ D.T
        
    print(f"FBS Converged in {iter} iterations.")
    end_time = time.time()
    cput = end_time - start_time

    U_opt = U
    V_opt = V
    Z_opt = Z

    return U_opt, V_opt, Z_opt, cput
