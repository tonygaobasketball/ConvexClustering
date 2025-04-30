"""
Project: Convex clustering with general Moreau envelope.

Forward-backward splitting algorithm


@author: Zheming Gao
"""

from prox_l1_admm import prox_l1_admm
import numpy as np
import time

def fbs_gme3_cc(Dtilde, C, mu, rho, gamma, u0, vec_x, p):
    """
    Forward-backward splitting (FBS) algorithm for convex clustering with generalized Moreau envelope (3rd model).
    
    Parameters:
    Dtilde (numpy.ndarray): Given matrix \tilde{D}.
    C (numpy.ndarray): Matrix C such that B = C Dtilde.
    mu (float): Parameter of FBS algorithm.
    rho (float): Lipschitz constant.
    gamma (float): Regularization parameter, gamma > 0.
    u0 (numpy.ndarray): Initial value of solution u.
    vec_x (numpy.ndarray): Data vector.
    p (int): Number of features.
    
    Returns:
    tuple: (u_opt, v_opt, z_opt, cput)
        - u_opt (numpy.ndarray): Solution u.
        - v_opt (numpy.ndarray): Solution v.
        - z_opt (numpy.ndarray): Solution z.
        - cput (float): CPU time running by the algorithm.
    """
    STOP_TOL = 1e-6
    Max_Iters = 200

    n = len(vec_x) // p  # Number of data points
    B = C @ Dtilde  # Define parameter B

    # Initialization
    u = u0.copy()
    Dtil_u = Dtilde @ u
    u_old = u.copy()
    Dtil_u_old = Dtil_u.copy()

    # ISTA parameters
    v = np.ones((n-1) * p)
    sig1_C = np.linalg.norm(C.toarray(), 2)
    # sig1_C = np.linalg.norm(C, 2)  # Largest singular value of C
    alpha_ista = .5 * sig1_C ** 2 * 1.1  # alpha >= 1/2 * \lambda_max (C^T C)
    Nit_ista = 10
    soft = lambda t, T: np.maximum(t - T, 0) + np.minimum(t + T, 0)  # Soft-threshold function

    rho_admm = 1.5
    max_iter_admm = 2000

    iter = 0
    relative_delta = np.inf  # Stopping criteria value

    start_time = time.time()

    while (relative_delta > STOP_TOL) and (iter < Max_Iters):
        iter += 1
        
        # ISTA for minimizing v
        for k in range(Nit_ista):
            v_old = v.copy()
            v = soft(v + 1 / (2 * alpha_ista) * (C.T @ C @ (Dtil_u - v)), 1 / (2 * alpha_ista))
            if np.linalg.norm(v - v_old) < STOP_TOL:
                break

        z = B.T @ C @ (Dtil_u - v)
        w = vec_x + gamma * z
        u = prox_l1_admm(Dtilde, w, gamma, rho_admm, max_iter_admm, STOP_TOL)
        Dtil_u = Dtilde @ u

        # Calculate residue
        relative_delta = np.linalg.norm(Dtilde @ (u - u_old), 2) / np.linalg.norm(Dtil_u_old, 2)
        u_old = u.copy()
        Dtil_u_old = Dtil_u.copy()

    end_time = time.time()
    cput = end_time - start_time

    u_opt = u
    v_opt = v
    z_opt = z

    return v_opt, z_opt, u_opt, cput
