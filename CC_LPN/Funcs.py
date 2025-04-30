#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Functions needed for LPN network.

"""
import numpy as np
import torch





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
        dual_res =np.linalg.norm(rho * (U_new - U) @ D, 'fro')
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
    # iter_admm = k
    return V_opt


def linspace_gen(p, lb, ub, points_per_dim):
    """
    Generate linspace for p dim box, with lower and upper bounds pre-defined 
    for 1-dim.
    
    Input: 
        p --- dimension
        lb --- lower bound
        up --- upper bound
        points_per_dim --- # of grids on each dimension
    
    return: grid_points  array.
    """
    # Generate 1D grid for each dimension
    grid_1d = np.linspace(lb, ub, points_per_dim)
    # Generate a meshgrid for p dimensions
    mesh = np.meshgrid(*[grid_1d] * p, indexing='ij')
    # Stack the grid points into a (N, p) array, where N is the total number of grid points
    grid_points = np.vstack(map(np.ravel, mesh)).T
    
    return grid_points

def L21_data_gen(Dw, p, n, K, ele_lb, ele_ub):
    """
    Generate training data in p by n dimension, total amount_data
    

    Input:
        ----------
        Dw : m by n array.
            denotes the matrix parameter in the L_21 norm.
        p : scalar
        n : scalar
            data dimension p by n.
        K: number of data to generate for each element of matrix.
        
    Returns
        -------
        data tensor
    """
    

    # # Parameters
    # p = 2       # Number of rows in each matrix
    # n = 2       # Number of columns in each matrix
    # ele_lb = -1 # Lower bound of values
    # ele_ub = 1  # Upper bound of values
    # K = 3       # Number of discrete values in the range
    
    # Generate the discrete values in the range [ele_lb, ele_ub]
    values = torch.linspace(ele_lb, ele_ub, K)
    
    # Compute the total number of matrices
    num_elements = p * n  # Total elements in one matrix
    total_matrices = K ** num_elements
    print(f"Total number of matrices: {total_matrices}")
    
    # Generate all possible combinations of matrix elements
    all_combinations = torch.cartesian_prod(*[values] * num_elements)
    
    # Reshape into matrices of shape (p, n) and save as a tensor
    all_matrices = all_combinations.view(-1, p, n)
    
    # Print results
    print("Shape of all_matrices tensor:", all_matrices.shape)  # (Total Matrices, p, n)
    
    data_gen_tensor = torch.zeros(total_matrices, p, n)
    true_f_tensor = torch.zeros(total_matrices, p, n)
    for j in range(total_matrices):
        W0 = all_matrices[j,:,:].numpy()
        prox_W = prox_mat_l21_admm(Dw, W0, gamma=1, rho=1, tol=1e-03, max_iter=500)
        
        data_gen_tensor[j] = torch.tensor(W0, dtype = torch.float64)
        true_f_tensor[j] = torch.tensor(prox_W, dtype = torch.float64)
    
    return data_gen_tensor, true_f_tensor

