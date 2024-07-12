"""
Project: Convex clustering with general Moreau envelope.

Function tools.


@author: Zheming Gao
"""

import numpy as np
from scipy.sparse import spdiags
import scipy as sp
import matplotlib.pyplot as plt


def sparse_convmtx(h, n):
    """
    Create a sparse convolution matrix from a row vector.
    
    Parameters:
    h (array-like): Input row vector of length M.
    n (int): Desired size of the resulting matrix.
    
    Returns:
    scipy.sparse.dia_matrix: Sparse convolution matrix of size (n, n + M - 1).
    """
    
    # Ensure h is a row vector
    h = np.asarray(h).flatten()  # Convert h to a flat numpy array if it is not already
    
    M = len(h)  # Length of the row vector h
    hh = np.tile(h, (n, 1))  # Replicate the row vector h n times to create an n x M matrix
    offsets = np.arange(M)  # Create an array of offsets from 0 to M-1
    
    # Create the sparse convolution matrix using spdiags
    A = spdiags(hh.T, offsets, n, n + M - 1)
    
    return A


def incidence_matrix(n):
    """
    Generate an edge incidence matrix for an n-node chain graph.
    
    Parameters:
    n (int): Number of nodes in the chain graph.
    
    Returns:
    numpy.ndarray: Incidence matrix of size (n-1, n).
    """
    
    # Create an identity matrix of size n
    I_n = np.eye(n)
    
    # Generate the incidence matrix for the chain graph
    E = - I_n[1:n, :] + I_n[0:n-1, :]
    
    return E

# # Example usage
# n = 5
# E = incidence_matrix(n)
# print(E)


def decompose_vec(u, n):
    """
    Decompose a long vector u into smaller vectors and save them as columns of a matrix U.
    
    Parameters:
    u (array-like): Input long vector of optimal solutions.
    n (int): Number of data points.
    
    Returns:
    numpy.ndarray: Matrix with decomposed vectors as columns.
    """
    
    u = np.asarray(u)  # Ensure u is a numpy array
    p = len(u) // n  # Number of features
    U = np.zeros((p, n))  # Initialize an empty matrix of size (p, n)
    
    for i in range(n):
        temp = u[p*i : p*(i+1)]  # Extract the sub-vector
        U[:, i] = temp  # Assign the sub-vector to the ith column of U
    
    return U

# # Example usage
# u = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12])
# n = 3
# U = decompose_vec(u, n)
# print(U)


def identical_cols(A, tolerance):
    """
    Identify identical columns in a matrix A within a given tolerance.
    
    Parameters:
    A (numpy.ndarray): Input matrix.
    tolerance (float): If the Euclidean norm between columns is less than this tolerance, 
                       they are considered identical.
    
    Returns:
    U (numpy.ndarray): Matrix with unique columns of A.
    uniqueCols (list): Indices of unique columns.
    AL (list): List of dictionaries where each dictionary contains the original columns
               that correspond to each unique column in U.
    """
    
    numCols = A.shape[1]  # Number of columns in A
    uniqueCols = np.ones(numCols, dtype=bool)  # Boolean array to track unique columns
    groups = [[] for _ in range(numCols)]  # List to keep track of groups of identical columns

    # Compare each column with every other column
    for i in range(numCols):
        if uniqueCols[i]:
            groups[i].append(A[:, i])
            for j in range(i+1, numCols):
                if np.linalg.norm(A[:, i] - A[:, j]) < tolerance:
                    uniqueCols[j] = False
                    groups[i].append(A[:, j])

    # Extract the unique columns
    U = A[:, uniqueCols]

    # Initialize AL as a list of dictionaries
    AL = []
    uniqueIndex = 0

    # Fill AL with groups of identical columns
    for i in range(numCols):
        if uniqueCols[i]:
            AL.append({'columns': np.column_stack(groups[i])})
            uniqueIndex += 1

    return U, np.where(uniqueCols)[0], AL

# # Example usage
# A = np.array([[1, 2, 1.1], [2, 3, 2.1], [3, 4, 3.1]])
# tolerance = 0.2
# U, uniqueCols, AL = identical_cols(A, tolerance)
# print("Unique Columns:\n", U)
# print("Indices of Unique Columns:\n", uniqueCols)
# print("AL Structure:\n", AL)





def plot_clusters(AL):
    """
    Plot clusters stored in the AL structure.
    
    Parameters:
    AL (list): List of dictionaries where each dictionary contains the columns
               corresponding to each unique column in U.
    """
    # Define colors for plotting
    colors = plt.cm.get_cmap('tab10', len(AL)).colors
    
    plt.figure()
    plt.title('Scatter Plot of Clusters')
    plt.xlabel('X')
    plt.ylabel('Y')
    
    # Plot each group of columns in AL
    for k, group in enumerate(AL):
        # Each group is a matrix where each column is a data point
        data = group['columns']
        
        # Scatter plot each column in the group
        plt.scatter(data[0, :], data[1, :], s=50, color=colors[k], label=f'Cluster {k+1}', alpha=0.7)
    
    plt.legend()
    plt.grid(True)
    plt.show()

# # Example usage
# # Assuming AL is the output from the identical_cols function
# AL = [{'columns': np.array([[1, 1.1], [2, 2.1], [3, 3.1]])},
#       {'columns': np.array([[4, 4.1], [5, 5.1], [6, 6.1]])},
#       {'columns': np.array([[7, 7.1], [8, 8.1], [9, 9.1]])}]

# plot_clusters(AL)



##### -------------------------
## Functions of diffusion maps
##### -------------------------
from sklearn.metrics import pairwise_distances

from sklearn.metrics import pairwise_distances


def gaussian_kernel_matrix(D, p = 0.01):
    """
    Computes the Gaussian kernel matrix for a given data matrix D.
    
    Parameters:
    - D (numpy.ndarray): Data matrix where each column is a data point.
    - sigma (float): The standard deviation parameter for the Gaussian kernel.
    
    Returns:
    - K (numpy.ndarray): The Gaussian kernel matrix.
    """
   
    # Compute pairwise distances (D is #features by #samples)
    pairwise_dist = pairwise_distances(D.T)
    
    # Compute the sigma 
    # distances to p * n-th nearest neighbor are used. 
    # Default value is p = .01
    # Determine the epsilon value
    
    n_samples = pairwise_dist.shape[0]
    n_neighbors = int(p * n_samples)
    sigma = np.median(np.sort(pairwise_dist, axis=1)[:, n_neighbors])
    
    
    # Compute the Gaussian kernel matrix
    K = np.exp(- pairwise_dist ** 2 / sigma)
    
    return K


def diff_map(dataX, t, l):
    """
    input:  dataX --- data matrix (p by n), each column a point.
            t --- parameter for transition matrix.
            l --- number of dimensions. (1 <= l <= n-1)

            
    output:
        diff_coor --- matrix after diffusion map \Psi_t(X): n by l
        diff_dis --- diffusion distance matrix.
        Phi, Sig, Psi --- SVD results.
        Pmat --- the P matrix 
        
    """
    Kmat = gaussian_kernel_matrix(dataX)
    # Dmat = np.diag(np.sum(Kmat, axis = 1))
    # Pmat = np.linalg.inv(Dmat) @ Kmat
    invD = np.diag(1 / np.sum(Kmat, axis = 1))
    Pmat = invD @ Kmat
    # Pmat_t = np.linalg.matrix_power(Pmat, t)  # matrix power P^t
        
    # SVD of P.
    # Sig is the array of singular values.
    Phi, Sig, PsiT = sp.linalg.svd(Pmat)
    Psi = PsiT.T
    
    # diff_map matrix (n by l) with power t.
    diff_coor = Psi[:,1:l+1] @ np.diag(Sig[1:l+1] ** t)
    
    # phi_1 = Phi[:,0]  # obtain denominator
    n = Pmat.shape[0]
    diff_dis = np.zeros(Pmat.shape)
    for i in range(n):
        for j in range(i, n):
            
            # temp_sum = 0
            # for k in range(n):
            #     diff_pt = Pmat_t[i,k] - Pmat_t[j,k] 
            #     temp_sum = temp_sum + diff_pt ** 2 / phi_1[k]
            # diff_dis[i,j] = np.sqrt(temp_sum)
            diff_dis[i,j] = np.linalg.norm(diff_coor[i,:] - diff_coor[j,:])
            diff_dis[j,i] = diff_dis[i,j]
    return diff_coor, diff_dis


def Gauss_weight_k_nrst (dataX, k_nrst):
    """
    Calculating Gaussian weights between x^i and x^j  
        w_ij = exp (|| x^i - x^j ||^2 / sigma_ij)
    where sigma_ij = sigma_i * sigma_j,  sigma_i is the 
    median Euclidean distance between x^i and its k nearest neighbors.


    input:  dataX --- data matrix (p by n), each column a point.
            k_nrst --- k nearest element for choosing sigma

            
    output:
        weights --- weights matrix.
    """
    # find sigma_i for x^i.
    n = dataX.shape[1]
    
    # Calculate Euclidean distance between x^i and x^j 
    E_dis = np.zeros((n,n))
    for i in range(n):
        for j in range(i, n):
            E_dis[i,j] = np.linalg.norm(dataX[:,i] - dataX[:,j])
            E_dis[j,i] = E_dis[i,j]
       
    sig = np.zeros(n)
    for i in range(n):
        cand_arr = E_dis[:,i]
        cand_arr[i] = np.inf # make the ith element big enough.
        # Get the indices of the sorted array
        sorted_indices = np.argsort(cand_arr)

        # Get the indices of the k smallest elements
        smallest_indices = sorted_indices[:k_nrst]

        # Get the first k smallest elements using these indices
        smallest_elements = cand_arr[smallest_indices]
        
        # The first must be zero. Disgard the first item.
        sig[i] = np.median(smallest_elements)
    
    # Standardize sig
    sig = sig / np.max(sig)
    
    weights = np.zeros((n,n))
    # for i in range(n):
    #     for j in range(i,n):
    #         weights[i,j] = np.exp(- np.linalg.norm(dataX[:,i] - dataX[:,j]) ** 2 / (sig[i] * sig[j]) )
    #         weights[j,i] = weights[i,j]
    
    for i in range(n):
        for j in range(n):
            weights[i,j] = np.exp(- np.linalg.norm(dataX[:,i] - dataX[:,j]) ** 2 / (sig[i] * sig[j]) )
            
    return weights
    

def Gauss_weight_diff_k_nrst (dataX, t, l, k_nrst):
    """
    Calculating Gaussian weights between x^i and x^j  
        w_ij = exp (|| x^i - x^j ||^2 / sigma_ij)
    where sigma_ij = sigma_i * sigma_j,  sigma_i is the 
    median diffusion distance between x^i and its k nearest neighbors.


    input:  dataX --- data matrix (p by n), each column a point.
            t --- parameter for transition matrix.
            l --- number of dimensions. (1 <= l <= n-1)
            k_nrst --- k nearest element for choosing sigma

            
    output:
        weights --- weights matrix.
        
        
    """
    
    # Calculate diffusion distance based on dataX, t and l.
    _, diff_dis = diff_map(dataX, t, l)
    
    # find sigma_i for x^i.
    n = dataX.shape[1]
    sig = np.zeros(n)
    for i in range(n):
        cand_arr = diff_dis[:,i]
        cand_arr[i] = np.inf # make the ith element big enough.
        # Get the indices of the sorted array
        sorted_indices = np.argsort(cand_arr)

        # Get the indices of the k smallest elements
        smallest_indices = sorted_indices[:k_nrst]

        # Get the first k smallest elements using these indices
        smallest_elements = cand_arr[smallest_indices]
        
        # The first must be zero. Disgard the first item.
        sig[i] = np.median(smallest_elements)
    
    # Standardize sig
    sig = sig / np.max(sig)
    
    weights = np.zeros((n,n))
    # for i in range(n):
    #     for j in range(i,n):
    #         weights[i,j] = np.exp(- np.linalg.norm(dataX[:,i] - dataX[:,j]) ** 2 / (sig[i] * sig[j]) )
    #         weights[j,i] = weights[i,j]
    
    for i in range(n):
        for j in range(n):
            weights[i,j] = np.exp(- np.linalg.norm(dataX[:,i] - dataX[:,j]) ** 2 / (sig[i] * sig[j]) )
            
    return weights
    
    

def find_k_nrst_neighbors_diff(X, W, k):
    """
    Find the k nearest neighbors for each column in a data matrix X using the diffusion distance matrix W.

    Parameters:
    X (numpy.ndarray): Data matrix with shape (p, n), where p is the number of rows and n is the number of columns.
    W (numpy.ndarray): Distance matrix with shape (n, n), where W_ij is the diffusion distance between x_i and x_j.
    k (int): Number of nearest neighbors to find for each column.

    Returns:
    neighbors (list of numpy.ndarray): List of arrays, where each array contains the indices of the k nearest neighbors 
                                        for each column.
    """

    n = X.shape[1]
    neighbors = []

    for i in range(n):
        # Get the distances from the i-th column to all other columns
        distances = W[i, :]
        # Exclude the i-th column itself by setting its distance to a very high value
        distances[i] = np.inf
        # Find the indices of the k smallest distances
        nearest_indices = np.argsort(distances)[:k]
        neighbors.append(nearest_indices)

    return neighbors


def build_incidence_matrix_diff(X, W, k):
    """
    Build the directed graph incidence matrix based on k-nearest neighbors for each column of X.

    Parameters:
    X (numpy.ndarray): Data matrix with shape (p, n), where p is the number of rows and n is the number of columns.
    W (numpy.ndarray): Distance matrix with shape (n, n), where W_ij is the diffusion distance between x_i and x_j.
    k (int): Number of nearest neighbors to find for each column.

    Returns:
    incidence_matrix (numpy.ndarray): 
    Directed graph incidence matrix with shape (num_edges, n), where num_edges is the 
    total number of edges, and each row represents an edge with 1 and -1.
        .
    """

    n = X.shape[1]
    neighbors = find_k_nrst_neighbors_diff(X, W, k)
    
    edges = []
    
    for i in range(n):
        for neighbor in neighbors[i]:
            edges.append((i, neighbor))

    num_edges = len(edges)
    incidence_matrix = np.zeros((num_edges, n), dtype=int)

    for idx, (i, j) in enumerate(edges):
        incidence_matrix[idx, i] = 1
        incidence_matrix[idx, j] = -1


    return incidence_matrix

# # Example usage:
# # X is your data matrix with shape (p, n)
# # W is your distance matrix with shape (n, n)
# # k is the number of nearest neighbors you want to find
# X = np.random.rand(5, 10)  # Example data matrix
# W = np.random.rand(10, 10)  # Example distance matrix
# k = 3  # Number of nearest neighbors

# incidence_matrix = build_incidence_matrix(X, W, k)
# print(incidence_matrix)


def extract_nonzero_elements(D, W):
    # Retrieve indices i and j of each row of incidence matrix D.
    # Record W_ij.
    # Output array with W_ij.
    
    m, n = D.shape
    weights_vec = []
    
    for row in D:
        nonzero_indices = np.nonzero(row)[0]
        if len(nonzero_indices) == 2:
            i, j = nonzero_indices
            weights_vec.append(W[i, j])
    
    return np.array(weights_vec)
    