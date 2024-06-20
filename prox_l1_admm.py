"""
Project: Convex clustering with general Moreau envelope.

ADMM for proximal operator of l_1-norm.


@author: Zheming Gao
"""

import numpy as np
import scipy

def prox_l1_admm(Dtilde, w0, gamma, rho, max_iter, tol):
    """
    Compute the proximal operator of f(u) = ||Du||_1 using ADMM.
    
    Parameters:
    Dtilde (numpy.ndarray): Matrix D.
    w0 (numpy.ndarray): Input vector.
    gamma (float): Regularization parameter.
    rho (float): Penalty parameter for ADMM.
    max_iter (int): Maximum number of iterations.
    tol (float): Tolerance for convergence.
    
    Returns:
    numpy.ndarray: The optimized vector w_opt.
    """
    
    # Initialize variables
    DtildeT = Dtilde.T
    w = w0.copy()  # Initialize solution w
    np_len = len(w0)
    I_np = np.eye(np_len)
    v = Dtilde @ w  # Initialize variable z
    la = np.zeros_like(v)  # Initialize variable y
    A = (1 / gamma) * I_np + rho * (Dtilde.T @ Dtilde)
    
    # Soft-threshold function
    soft = lambda t, T: np.maximum(t - T, 0) + np.minimum(t + T, 0)

    cho_facts = scipy.linalg.cho_factor(A, lower=False, overwrite_a=True, check_finite=False)
    
    # ADMM iterations
    for k in range(max_iter):
        # Update w
        b = (1 / gamma) * w + rho * DtildeT @ v - DtildeT @ la
        w_new = scipy.linalg.cho_solve(cho_facts, b, overwrite_b=True, check_finite=False)
        
        # Update v
        v_new = soft(Dtilde @ w_new + (1 / rho) * la, 1 / rho)
        
        # Update lambda (Lagrangian multipliers)
        la = la + rho * (Dtilde @ w_new - v_new)
        
        # Check for convergence
        if np.linalg.norm(w_new - w) ** 2 < tol and np.linalg.norm(v_new - v) ** 2 < tol:
            break
        
        # Update variables
        w = w_new
        v = v_new
    
    # Calculate solution u = x + \gamma * z
    w_opt = w_new
    # iter_admm = k
    return w_opt

# # Example usage
# Dtilde = np.array([[1, 2], [3, 4], [5, 6]])
# w0 = np.array([0.5, 1.5])
# gamma = 0.1
# rho = 1.0
# max_iter = 100
# tol = 1e-4

# w_opt = prox_l1_admm(Dtilde, w0, gamma, rho, max_iter, tol)
# print("Optimized w:", w_opt)
