"""
Project: Convex clustering with general Moreau envelope.

functions for GME3-CC matrix form.


@author: Zheming Gao
"""

import numpy as np
from scipy import sparse
import time


def tconjgrad(A, b, gamma, tol = 1e-08, max_iter = 1e+04):
    """
    truncated conjugate gradient method. 
    
    By Chester Holtz
    """
    
    # Initialize x and M.
    n = len(b)
    x = np.zeros_like(b)
    M0 = sparse.identity(n)
    M = lambda x : M0@x
    
    # print("shapes: A, b ,x.")
    # print(A.shape)
    # print(b.shape)
    # print(x.shape)
    
    r = b - A @ x
    z = M(r)
    p = z.copy()
    rsold = np.sum(r * z, axis=0)

    err = 1
    i = 0
    info = {'res': [], 'time': []}
    inittime = time.time()     # start counting CPU time.
    while (err > tol) and (i < max_iter):
        i += 1
        Ap = A @ p
        alpha = rsold / np.sum(p * Ap,axis=0)
        # update x
        x += alpha * p

        # add additional condition (check if norm(x^k) <= gamma)
        if np.linalg.norm(x) > gamma:
            p_sqr = np.sum(p * p, axis = 0)
            x_sqr = np.sum(x * x, axis = 0)
            xp = np.sum(x * p, axis = 0)
            nume = - xp + np.sqrt(xp ** 2 - x_sqr * p_sqr)
            t = nume / p_sqr
            x += t * p
            info['res'].append(np.linalg.norm(A @ x - b))
            info['time'].append(time.time() - inittime)
            return x, info
        
        r -= alpha * Ap
        z = M(r)
        rsnew = np.sum(r * z, axis=0)
        err = np.sqrt(np.sum(rsnew))
        p = z + (rsnew / rsold) * p
        rsold = rsnew
        info['res'].append(np.linalg.norm(A @ x - b))
        info['time'].append(time.time() - inittime)
        
    return x, info


def tconjgrad_nb(H, b, gamma, tol = 1e-08, max_iter = 1e+04):
    """
    truncated conjugate gradient method. 
    Nicolas Boumal's notes (Spring 2023)
    
    Solving problem:
        min 1/2 v.T H v - b.T v
        s.t.      ||v||_2 <= gamma
                    v \in R^n.
    
    """
    
    # Initialize v.
    v = np.zeros_like(b)
    rold = b.copy()
    p = rold

    err = 1
    i = 0
    info = {'res': [], 'time': [], 'steps': []}
    inittime = time.time()     # start counting CPU time.
    while (err > tol) and (i < max_iter):
        i += 1
        Hp = H @ p
        pHp = np.sum(p * Hp,axis=0)
        alpha = np.dot(rold, rold) / pHp
        v += alpha * p
        
        if pHp <=0 or np.linalg.norm(v) >= gamma:
            ps = np.dot(p, p)
            vs = np.dot(v, v)
            vp = np.dot(v, p)
            nume = -vp + np.sqrt(vp ** 2 - ps * (vs - gamma ** 2))
            t = nume / ps
            v += t * p
            info['res'].append(np.linalg.norm(H @ v - b))
            info['time'].append(time.time() - inittime)
            info['steps'].append(i)
        
            return v, info
        
        rnew = rold - alpha * Hp

        # if np.linalg.norm(rnew) <= np.min(tol, np.linalg.norm(b)):
        #     return v, info
        err = np.linalg.norm(rnew)
        
        beta = np.dot(rnew, rnew) / np.dot(rold, rold)
        p = rnew + beta * p
        rold = rnew

        info['res'].append(np.linalg.norm(H @ v - b))
        info['time'].append(time.time() - inittime)
        info['steps'].append(i)

    return v, info

def truncated_conjugate_gradient(H, g, delta, tol=1e-8, max_iter=None):
    """
    Truncated Conjugate Gradient Method for solving trust-region subproblem
    
    Parameters:
    H : numpy.ndarray
        Symmetric positive-definite Hessian matrix.
    g : numpy.ndarray
        Gradient vector.
    delta : float
        Trust region radius.
    tol : float, optional
        Tolerance for convergence (default is 1e-8).
    max_iter : int, optional
        Maximum number of iterations (default is size of g).
        
    Returns:
    s : numpy.ndarray
        Solution vector.
    n_iter : int
        Number of iterations performed.
    """
    n = len(g)
    s = np.zeros(n)
    r = -g
    d = r
    rsold = np.dot(r, r)
    err = 1
    i = 0
    
    while (err > tol) and (i < max_iter):
        Hd = H @ d
        alpha = rsold / np.dot(d, Hd)
        
        if np.linalg.norm(s + alpha * d) > delta:
            d_sqr = np.dot(d, d)
            s_sqr = np.dot(s, s)
            sd = np.dot(s, d)
            nume = - sd + np.sqrt(sd ** 2 - s_sqr * d_sqr)
            t = nume / d_sqr
            s = s + t * d
            return s, i + 1
        
        s = s + alpha * d
        r = r - alpha * Hd
        rsnew = np.dot(r, r)
        
        err = np.sqrt(rsnew)
        
        d = r + (rsnew / rsold) * d
        rsold = rsnew
    
    return s, i + 1


def prox_mat_l21_tcg(D, W0, gamma, tol = 1e-08, max_iter = 1e+04):
    """
    Compute the 3rd iteration of the algorithm (matrix form)
    with truncated conjugate gradient method. 
    
    By Chester Holtz
    
    Parameters:
    D (numpy.ndarray): Matrix D.
    W0 (numpy.ndarray): Input matrix.
    gamma (float): Regularization parameter.
    max_iter (int): Maximum number of iterations.
    tol (float): Tolerance for convergence.
    
    Returns:
    numpy.ndarray: The optimized matrix V_opt.
    """
    
    # Initialize variables
    p, n = W0.shape  
    
    # A = DD^T
    # B = DW^T, each b is a column of B.
    A = D @ D.T
    B = D @ W0.T
    m = A.shape[0]
    La_new = np.zeros((p, m))
    for j in range(p): 
        b = B[:,j]
        u,info = tconjgrad_nb(A, b, gamma, tol, max_iter)
        print("dim {}: tCG Converged in {} iterations.".format(j, info['steps'][0]))
        # u,_ = truncated_conjugate_gradient(A, -b,  gamma, tol, max_iter)
        La_new[j,:] = u
    return La_new
    
        
    
    

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

# # Example usage
# Dtilde = np.array([[1, 2], [3, 4], [5, 6]])
# w0 = np.array([0.5, 1.5])
# gamma = 0.1
# rho = 1.0
# max_iter = 100
# tol = 1e-4

# w_opt = prox_l1_admm(Dtilde, w0, gamma, rho, max_iter, tol)
# print("Optimized w:", w_opt)


def MPD(V_0, C, A, alpha_k=0.1, tol=1e-8, max_iterations=1000):
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
        Y = np.array(V @ (np.eye(C.shape[1]) - alpha_k * C.T @ C) + alpha_k * A @ C)
        
        V_new = np.zeros_like(V)
        for j in range(V.shape[1]):
            Y_j = Y[:, j]
            norm_Y_j = np.linalg.norm(Y_j, 2)
            factor = max(0, 1 - alpha_k / (V.shape[1] * norm_Y_j))
            V_new[:, j] = factor * Y_j
        
        # Check the stopping criterion
        if np.linalg.norm(V_new - V, 'fro') < tol:
            print(f"MPD Converged in {k+1} iterations.")
            break
        
        # Update V
        V = V_new

    return V



from sklearn.neighbors import NearestNeighbors

def find_k_nearest_neighbors(U, k):
    """
    Finds the k-nearest neighbors for each column of the matrix U.

    Parameters:
    U (numpy.ndarray): A p by n matrix where we want to find the k-nearest neighbors for each column.
    k (int): The number of nearest neighbors to find for each column.

    Returns:
    list: A list of tuples (P_j, indices_j), where each P_j contains the k-nearest neighbors of column u_j of U,
          and indices_j contains the original column indices of these nearest neighbors in U.
    """
    p, n = U.shape  # Get the shape of the matrix U
    # Initialize the NearestNeighbors model with k+1 neighbors (including the point itself)
    nearest_neighbors = NearestNeighbors(n_neighbors=k+1, algorithm='auto').fit(U.T)
    P = []  # List to store the nearest neighbors for each column
    indices_list = []  # List to store the indices of the nearest neighbors

    for j in range(n):
        # Find the k+1 nearest neighbors for the j-th column vector (u_j)
        distances, indices = nearest_neighbors.kneighbors(U[:, j].reshape(1, -1))
        # Create P_j by selecting the k-nearest neighbors (excluding the point itself)
        P_j = U[:, indices[0][1:]].T  # Transpose to get k rows each of length p
        P.append(P_j)  # Append the matrix P_j to the list
        indices_list.append(indices[0][1:])  # Append the indices (excluding the point itself) to the list
    
    return list(zip(P, indices_list))

def build_incidence_matrix(U, k):
    """
    Builds the directed graph incidence matrix for each column of U based on k-nearest neighbors.

    Parameters:
    U (numpy.ndarray): A p by n matrix where we want to find the k-nearest neighbors for each column.
    k (int): The number of nearest neighbors to find for each column.

    Returns:
    list: A list of incidence matrices A_j for each column u_j.
    """
    results = find_k_nearest_neighbors(U, k)
    p, n = U.shape  # Get the shape of the matrix U
    incidence_matrices = []

    for j, (P_j, indices_j) in enumerate(results):
        A_j = np.zeros((n, k))  # Initialize the incidence matrix with zeros
        for idx, neighbor_idx in enumerate(indices_j):
            A_j[j, idx] = 1  # Mark the edge from u_j to each of its k-nearest neighbors
            A_j[neighbor_idx, idx] = -1  # Mark the edge from u_j to each of its k-nearest neighbors
        incidence_matrices.append(A_j.T)
    
    return incidence_matrices



# # Example usage
# U = np.random.rand(5, 10)  # Example matrix U with shape (5, 10)
# k = 3  # Number of nearest neighbors to find

# incidence_matrices = build_incidence_matrix(U, k)  # Build the incidence matrices

# # Print the incidence matrices for each column
# for j, A_j in enumerate(incidence_matrices):
#     print(f"Incidence matrix for column {j+1}:\n{A_j}\n")
    

# # Stack the matrices vertically
# stacked_matrix = np.vstack(incidence_matrices)
# print(f"Stacked incidence matrix: \n{stacked_matrix}\n")


def delete_redundant_rows(matrix):
    """
    Deletes redundant rows where a row r_i equals -r_i, keeping only one of them.

    Parameters:
    matrix (numpy.ndarray): The input matrix.

    Returns:
    numpy.ndarray: The matrix with redundant rows removed.
    """
    rows = [list(row) for row in matrix]  # Convert matrix rows to a list of lists for easier manipulation
    unique_rows = []

    while rows:
        row = np.array(rows.pop(0))  # Take the first row and convert it to a numpy array
        neg_row = (-row).tolist()  # Find the negative of the row and convert to list
        if neg_row in rows:
            rows.remove(neg_row)  # Remove the negative row if it exists
        unique_rows.append(row.tolist())  # Add the current row to the unique list

    return np.array(unique_rows)  # Convert ba ck to a numpy array

# # #%% Generate high-pass filter matrix.
# # Define parameters
# K = 10
# gamma = 2.5
# hh = np.ones(K) / K
# h = np.concatenate([np.zeros(K - 1), [1], np.zeros(K - 1)]) - np.convolve(hh, hh, mode='full')
# g = np.convolve(h, [1, -1], mode='full')
# G = sparse_convmtx(g, n - 1 - (len(g) - 1))
# c = (1 / np.sqrt(gamma)) * g
# C = (1 / np.sqrt(gamma)) * G
