#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ADMM algos for convex clustering

E. C. Chi & K. Lange, 2015, Splitting methods for convex clustering

@author: Zheming Gao
"""

import numpy as np

# Define the Prox operator for L2 norm with sigma_l
def prox_l2_norm(v, sigma_l):
    norm_v = np.linalg.norm(v, axis=0)
    return np.maximum(0, 1 - sigma_l / norm_v) * v

# Define the Prox operator for L1 norm with sigma_l (soft-thresholding)
def prox_l1_norm(v, sigma):
    return np.sign(v) * np.maximum(np.abs(v) - sigma, 0)


def convex_clustering_admm(X, D, nu, sigma, max_iter = 200, tol=1e-4):
    """
    Convex clustering using ADMM with L1 norm prox.
    
    Parameters:
    X (np.ndarray): The input data matrix of size (p, n), where p is the dimension of each data point and n is the number of nodes.
    D (np.ndarray): The incidence matrix of size (K, n), where K is the number of edges.
    nu (float): Regularization parameter.
    sigma (np.ndarray): Vector of sigma values for each edge (size K).
    max_iter (int): Maximum number of iterations.
    tol (float): Tolerance for stopping criterion.
    
    Returns:
    U (np.ndarray): Clustered points (p by n).
    """
    
    p, n = X.shape  # p is the dimension, n is the number of nodes
    K = D.shape[0]  # K is the number of edges
    
    
    # Initialize variables
    U = np.copy(X)  # Initialize U to be the same as X (p by n)
    V = np.zeros((p, K))  # V is p by K
    Lambda = np.zeros((p, K))  # Lambda is p by K
    
    # Calculate the average of the columns of X
    X_bar = np.mean(X, axis=1).reshape(-1, 1)  # X_bar is p by 1
    X_bar_matrix = np.tile(X_bar, (1, n))  # Replicate X_bar across all columns (p by n)
    
    for m in range(max_iter):
        # Step 1: Update Y_i
        Y = np.zeros_like(U)
        for i in range(n):
            sum_lambda_v = np.zeros(p)
            sum_lambda_v_incident = np.zeros(p)
            for l in range(K):
                if D[l, i] == 1:  # Node i is incident to edge l1
                    sum_lambda_v += Lambda[:, l] + nu * V[:, l]
                elif D[l, i] == -1:  # Node i is incident to edge l2
                    sum_lambda_v_incident += Lambda[:, l] + nu * V[:, l]
            Y[:, i] = X[:, i] + sum_lambda_v - sum_lambda_v_incident
        
        # Step 2: Update U
        U = (1 / (1 + n * nu)) * (Y + nu * (n * X_bar_matrix))
        
        # Step 3: Update V and Lambda for each edge
        for l in range(K):
            u_l1 = U[:, np.where(D[l] == 1)[0][0]]  # Node l1
            u_l2 = U[:, np.where(D[l] == -1)[0][0]]  # Node l2
            v_new = prox_l2_norm(u_l1 - u_l2 - (1 / nu) * Lambda[:, l], sigma[l])
            V[:, l] = v_new
            Lambda[:, l] = Lambda[:, l] + nu * (V[:, l] - (u_l1 - u_l2))

            
        # Convergence check: 
            # rr = V - (U[:, l1] - U[:, l2])
            # ss = -nu * (sum_)
            
        diff_sum = 0
        for l in range(K):
            u_l1 = U[:, np.where(D[l] == 1)[0][0]]  # Node l1
            u_l2 = U[:, np.where(D[l] == -1)[0][0]]  # Node l2
            diff_sum += np.linalg.norm(V[:, l] - (u_l1 - u_l2))
        
        if diff_sum < tol:
            print('ADMM converges at step {}.'.format(m))
            break
    
    if diff_sum >= tol:
        print('Max_iter reached with delta = {}.'.format(diff_sum))
    
    return U, V, Lambda



# # Function to generate a proper incidence matrix D for a graph with n nodes and K edges
# def generate_incidence_matrix(n, edges):
#     """
#     Generate an incidence matrix for a graph with n nodes and K edges.

#     Parameters:
#     n (int): Number of nodes.
#     edges (list of tuples): Each tuple is a pair of nodes (i, j) representing an edge between nodes i and j.

#     Returns:
#     D (np.ndarray): The incidence matrix of size (K, n).
#     """
#     K = len(edges)  # Number of edges
#     D = np.zeros((K, n))  # Incidence matrix with K rows (edges) and n columns (nodes)
    
#     for idx, (i, j) in enumerate(edges):
#         D[idx, i] = -1  # Node i (start of edge) gets -1
#         D[idx, j] = 1   # Node j (end of edge) gets 1
    
#     return D

# # Example of usage

# p, n = 5, 10  # 5 dimensions and 10 nodes
# # Define a simple graph with 10 nodes and 15 edges
# edges = [(0, 1), (1, 2), (2, 3), (3, 4), (4, 5), (5, 6), (6, 7), (7, 8), (8, 9), (0, 9), 
#          (0, 5), (1, 6), (2, 7), (3, 8), (4, 9)]
# K = len(edges)

# # Generate the incidence matrix for the graph
# D = generate_incidence_matrix(n, edges)

# X = np.random.randn(p, n)  # Random input data matrix (p by n)
# nu = 10  # Regularization parameter
# gamma = 100   # parameter of l1-norm dist of centroids
# # weights_vec = np.random.randn(K)    # weights vector on edges
# weights_vec = np.ones(K)
# sigma = gamma * weights_vec / nu  # sigma values in prox_l1


# # Solve using ADMM
# U, _, _ = convex_clustering_admm(X, D, nu, sigma, max_iter=100, tol=1e-4)

# # Display the result
# # print("Clustered points (U):\n", U)