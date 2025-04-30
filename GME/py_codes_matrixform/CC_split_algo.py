"""
Convex Clustering via ADMM and AMA

This module implements convex clustering using two different optimization
strategies: ADMM and AMA. The main functions are:

  • cvxclust and cvxclust_path_admm – perform convex clustering via ADMM.
  • cvxclust_path_ama – perform convex clustering via AMA.

Both methods require:
  - A data matrix X (of shape p×n, with p features and n samples).
  - A weight vector w (of length n*(n-1)/2 in dictionary order for all pairs (i,j) with i < j).
Only edges with positive weight are used.
Two penalty norms are supported:
  - norm_type = 1: elementwise L1 (soft–thresholding).
  - norm_type = 2: group L2 (block soft–thresholding).

Dependencies: numpy, scipy.linalg

"""

import numpy as np
from scipy.linalg import cho_factor, cho_solve

def recover_full_weights(D, weights_vec):
    """
    Recover the full weight vector for a complete graph on n nodes given a reduced
    incidence matrix and its corresponding weight vector.

    Parameters
    ----------
    D : ndarray, shape (m, n)
        The reduced incidence matrix. Each row represents an edge and should contain
        exactly one +1 and one -1 (with the +1 occurring at the smaller node index).
    weights_vec : ndarray, shape (m,)
        The weight vector corresponding to the m edges (in the same order as the rows of D).

    Returns
    -------
    weights_full : ndarray, shape (n*(n-1)//2,)
        The full weight vector for the complete graph in dictionary order. If an edge is
        missing in D, its weight is set to 0.

    Example
    -------
    >>> # Suppose we have a graph on 4 nodes and only two edges:
    >>> # Edge between node 0 and 2 with weight 1.5, and edge between node 1 and 3 with weight 2.0.
    >>> D = np.array([[1, 0, -1, 0],
    ...               [0, 1, 0, -1]])
    >>> weights_vec = np.array([1.5, 2.0])
    >>> # The full weight vector for 4 nodes has length 4*3//2 = 6.
    >>> # The full ordering (dictionary order) is: (0,1), (0,2), (0,3), (1,2), (1,3), (2,3)
    >>> # In this case, edge (0,2) is present with weight 1.5 and edge (1,3) is present with weight 2.0;
    >>> # all other edges receive weight 0.
    >>> weights_full = recover_full_weights(D, weights_vec, 4)
    >>> print(weights_full)
    [0.  1.5 0.  0.  2.0 0. ]
    """
    # Total number of edges in the complete graph
    m, n = D.shape
    N_full = n * (n - 1) // 2
    weights_full = np.zeros(N_full)
    
    def full_index(i, j, n):
        """
        Compute the index in dictionary order for the edge (i, j) with 0 <= i < j < n.
        The ordering is:
            (0,1), (0,2), ..., (0, n-1),
            (1,2), (1,3), ..., (1, n-1),
            ...
            (n-2, n-1)
        """
        # Number of edges with first node < i
        return int(i * n - (i * (i + 1)) // 2 + (j - i - 1))
    
    for row in range(m):
        # Find the two nonzero entries in the current row.
        nz = np.nonzero(D[row])[0]
        if len(nz) != 2:
            raise ValueError(f"Row {row} does not have exactly two nonzero entries.")
        vals = D[row, nz]
        # Identify the index of the +1 and the -1. We assume +1 comes from the smaller node.
        if np.isclose(vals[0], 1) and np.isclose(vals[1], -1):
            i, j = nz[0], nz[1]
        elif np.isclose(vals[0], -1) and np.isclose(vals[1], 1):
            i, j = nz[1], nz[0]
        else:
            raise ValueError(f"Row {row} does not have the expected 1 and -1 entries.")
        if i >= j:
            raise ValueError(f"In row {row}, the indices are not in order: {i} is not less than {j}.")
        idx = full_index(i, j, n)
        weights_full[idx] = weights_vec[row]
    
    return weights_full

# # --- Example Usage ---
# if __name__ == '__main__':
#     # Example with 4 nodes. Full edge set has 4*3/2 = 6 edges.
#     # Suppose only the edges (0,2) and (1,3) are present.
#     D = np.array([[1, 0, -1, 0],   # Edge from node 0 to node 2
#                   [0, 1,  0, -1]])  # Edge from node 1 to node 3
#     weights_vec = np.array([1.5, 2.0])
#     weights_full = recover_full_weights(D, weights_vec, 4)
#     print("Recovered full weights vector:")
#     print(weights_full)
#     # Expected order (dictionary order for 4 nodes): 
#     # (0,1), (0,2), (0,3), (1,2), (1,3), (2,3)
#     # So output should be: [0.0, 1.5, 0.0, 0.0, 2.0, 0.0]

# --------------------------------------------------------------------
# Helper: Build the Incidence Matrix (for both ADMM and AMA)
# --------------------------------------------------------------------
def compactify_edges(w, n, method='admm'):
    """
    Given a weight vector w (of length n*(n-1)/2) in dictionary order,
    return the incidence matrix, list of edge index pairs, and the positive weights.

    Parameters
    ----------
    w : array_like
        1D array of weights (length n*(n-1)/2).
    n : int
        Number of samples.
    method : str, optional
        Indicator (either 'admm' or 'ama'); treated identically here.

    Returns
    -------
    D : ndarray, shape (n, m)
        Incidence matrix with one column per edge (with +1 and –1 in the corresponding rows).
    edge_list : list of tuple
        List of (i, j) index pairs (0-indexed) for which w > 0.
    edge_weights : ndarray, shape (m,)
        Array of weights for the selected edges.
    """
    w = np.asarray(w)
    edge_list = []
    edge_weights = []
    idx = 0
    for i in range(n):
        for j in range(i+1, n):
            if w[idx] > 0:
                edge_list.append((i, j))
                edge_weights.append(w[idx])
            idx += 1
    edge_weights = np.array(edge_weights)
    m = len(edge_list)
    D = np.zeros((n, m))
    for k, (i, j) in enumerate(edge_list):
        D[i, k] = 1
        D[j, k] = -1
    return D, edge_list, edge_weights

# --------------------------------------------------------------------
# ADMM-based Convex Clustering
# --------------------------------------------------------------------

def cvxclust_path_admm(X, D, edge_weights, gamma_seq, nu=1, tol_abs=1e-5, tol_rel=1e-4,
                       max_iter=10000, norm_type=2, accelerate=True):
    """
    Compute the convex clustering path via ADMM over a sequence of gamma values.
    """
    p, n = X.shape
    # D, _, edge_weights = compactify_edges(w, n, method='admm')
    m = D.shape[0]
    Lambda = np.zeros((p, m))
    V = np.zeros((p, m))
    list_U, list_V, list_Lambda, iter_vec = [], [], [], []

    for gam in gamma_seq:
        sol = cvxclust_admm(X, Lambda, D, edge_weights, gamma=gam, nu=nu,
                            max_iter=max_iter, tol_abs=tol_abs, tol_rel=tol_rel,
                            norm_type=norm_type, accelerate=accelerate)
        # Warm start for next gamma:
        Lambda = sol['Lambda']
        V = sol['V']
        list_U.append(sol['U'])
        list_V.append(V.copy())
        list_Lambda.append(Lambda.copy())
        iter_vec.append(sol['iter'])

    return {"U": list_U, "V": list_V, "Lambda": list_Lambda,
            "nGamma": len(gamma_seq), "iters": iter_vec}

def cvxclust_admm(X, Lambda, D, w_edges, gamma, nu, max_iter, tol_abs, tol_rel,
                  norm_type, accelerate):
    """
    Perform ADMM iterations for convex clustering for one given gamma.
    """
    p, n = X.shape
    m = D.shape[0]
    # Precompute the Laplacian L = D DT and system matrix A = I + nu * L.
    L = D.T @ D
    A_mat = np.eye(n) + nu * L
    c_factor, lower = cho_factor(A_mat)
    U = np.zeros((p, n))
    V = np.zeros((p, m))
    V_old = np.zeros_like(V)
    alpha = 1.5 if accelerate else 1.0

    primal_residuals, dual_residuals = [], []
    for it in range(1, max_iter + 1):
        B = (V - Lambda) @ D
        rhs = X + nu * B
        U = cho_solve((c_factor, lower), rhs.T).T
        diff = U @ D.T  # differences between centroids
        diff_hat = alpha * diff + (1 - alpha) * V_old
        V_old = V.copy()
        V_new = np.zeros_like(V)
        for j in range(m):
            d = diff_hat[:, j] + Lambda[:, j]
            if norm_type == 2:
                norm_d = np.linalg.norm(d, 2)
                threshold = gamma * w_edges[j] / nu
                V_new[:, j] = (1 - threshold / norm_d) * d if norm_d > threshold else 0.0
            elif norm_type == 1:
                threshold = gamma * w_edges[j] / nu
                V_new[:, j] = np.sign(d) * np.maximum(np.abs(d) - threshold, 0)
            else:
                raise ValueError("norm_type must be 1 or 2.")
        V = V_new
        Lambda = Lambda + (diff - V)
        r = diff - V
        primal_norm = np.linalg.norm(r, 'fro')
        s = nu * (V - V_old)
        dual_norm = np.linalg.norm(s, 'fro')
        primal_residuals.append(primal_norm)
        dual_residuals.append(dual_norm)
        eps_primal = np.sqrt(p * m) * tol_abs + tol_rel * max(np.linalg.norm(diff, 'fro'), np.linalg.norm(V, 'fro'))
        eps_dual = np.sqrt(p * n) * tol_abs + tol_rel * np.linalg.norm(Lambda, 'fro')
        if primal_norm < eps_primal and dual_norm < eps_dual:
            break

    return {"U": U, "V": V, "Lambda": Lambda, "nu": nu,
            "primal": np.array(primal_residuals), "dual": np.array(dual_residuals),
            "iter": it}

# --------------------------------------------------------------------
# AMA-based Convex Clustering
# --------------------------------------------------------------------
def proj_dual(L, gamma, w_edges, norm_type):
    """
    Project each column of L onto the dual ball.
    For norm_type == 2: scale the column if its l2 norm exceeds gamma*w_edge.
    For norm_type == 1: clip each element.
    """
    L_proj = np.empty_like(L)
    m = L.shape[1]
    for j in range(m):
        if norm_type == 2:
            norm_val = np.linalg.norm(L[:, j], 2)
            thresh = gamma * w_edges[j]
            L_proj[:, j] = L[:, j] * (thresh / norm_val) if norm_val > thresh else L[:, j]
        elif norm_type == 1:
            L_proj[:, j] = np.clip(L[:, j], -gamma * w_edges[j], gamma * w_edges[j])
        else:
            raise ValueError("norm_type must be 1 or 2.")
    return L_proj

def AMA_step_size(w, n):
    """Return a step size for AMA; here simply 1/n."""
    return 1.0 / n

def cvxclust_ama(X, Lambda, D, w_edges, gamma, nu, max_iter=100, tol=1e-4,
                 norm_type=2, accelerate=True):
    """
    Perform AMA iterations for convex clustering for one given gamma.

    Updates:
      U = X – Y @ Dᵀ,
      then Λ_new = proj( Y + nu * (U @ D) ).
    Optionally applies FISTA acceleration.
    """
    p, n = X.shape
    m = D.shape[1]
    if nu >= 2.0 / n:
        print("Warning: nu is too large. Adjusting step size.")
        nu = max(AMA_step_size(w_edges, n), 1.0 / n)
    if accelerate:
        Y = Lambda.copy()
        t = 1.0
        Lambda_old = Lambda.copy()
    else:
        Y = Lambda.copy()
    history = []
    for it in range(1, max_iter + 1):
        U = X - Y @ D.T
        grad = U @ D
        tentative = Y + nu * grad
        Lambda_new = proj_dual(tentative, gamma, w_edges, norm_type)
        res = np.linalg.norm(Lambda_new - Y, 'fro')
        history.append(res)
        if res < tol:
            Y = Lambda_new.copy()
            break
        if accelerate:
            t_new = (1.0 + np.sqrt(1.0 + 4.0 * t * t)) / 2.0
            Y_next = Lambda_new + ((t - 1.0) / t_new) * (Lambda_new - Lambda_old)
            Lambda_old = Lambda_new.copy()
            t = t_new
            Y = Y_next.copy()
        else:
            Y = Lambda_new.copy()
    U = X - Lambda_new @ D.T
    return {"U": U, "Lambda": Lambda_new, "nu": nu, "iter": it, "history": history}

def cvxclust_path_ama(X, w, gamma_seq, nu=1, tol=1e-3, max_iter=10000,
                      norm_type=2, accelerate=True):
    """
    Compute the convex clustering path via AMA over a sequence of gamma values.
    """
    p, n = X.shape
    D, _, edge_weights = compactify_edges(w, n, method='ama')
    m = D.shape[1]
    Lambda = np.zeros((p, m))
    list_U, list_Lambda, iter_vec = [], [], []
    for gam in gamma_seq:
        sol = cvxclust_ama(X, Lambda, D, edge_weights, gamma=gam, nu=nu,
                           max_iter=max_iter, tol=tol, norm_type=norm_type, accelerate=accelerate)
        iter_vec.append(sol["iter"])
        nu = sol["nu"]   # update step size if needed
        Lambda = sol["Lambda"]   # warm start next gamma
        list_U.append(sol["U"])
        list_Lambda.append(Lambda.copy())
    return {"U": list_U, "Lambda": list_Lambda, "nGamma": len(gamma_seq), "iters": iter_vec}

# # --------------------------------------------------------------------
# # Example Usage
# # --------------------------------------------------------------------
# if __name__ == '__main__':
#     # -- ADMM Example --
#     print("ADMM Example:")
#     p, n = 5, 10
#     np.random.seed(123)
#     X = np.random.randn(p, n)
#     m_full = n * (n - 1) // 2
#     w = np.ones(m_full)
#     gamma_seq = np.linspace(0.0, 5.0, num=10)
#     sol_admm = cvxclust_path_admm(X, w, gamma_seq, nu=1, tol_abs =1e-3, tol_rel= 1e-3,
#                         max_iter=1000, norm_type=2, accelerate=True)
#     print("ADMM iteration counts per gamma:")
#     print(sol_admm["iters"])

#     # -- AMA Example --
#     print("\nAMA Example:")
#     np.random.seed(123)
#     X = np.random.randn(p, n)
#     w = np.ones(m_full)
#     sol_ama = cvxclust_path_ama(X, w, gamma_seq, nu=1, tol=1e-3,
#                                 max_iter=1000, norm_type=2, accelerate=True)
#     print("AMA iteration counts per gamma:")
#     print(sol_ama["iters"])