"""
Project: Convex clustering with general Moreau envelope (GME-CC).

Function tools.


@author: Zheming Gao
"""

import numpy as np
from scipy.sparse import spdiags
import scipy as sp
import matplotlib.pyplot as plt
from scipy.stats import entropy


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
    
    # # Standardize sig
    # sig = sig / np.max(sig)
    
    weights = np.zeros((n,n))
    # for i in range(n):
    #     for j in range(i,n):
    #         weights[i,j] = np.exp(- np.linalg.norm(dataX[:,i] - dataX[:,j]) ** 2 / (sig[i] * sig[j]) )
    #         weights[j,i] = weights[i,j]
    
    for i in range(n):
        for j in range(n):
            weights[i,j] = np.exp(-E_dis[i,j] ** 2 / (sig[i] * sig[j]) )
            
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
    
    # m, n = D.shape
    weights_vec = []
    
    for row in D:
        nonzero_indices = np.nonzero(row)[0]
        if len(nonzero_indices) == 2:
            i, j = nonzero_indices
            weights_vec.append(W[i, j])
    
    return np.array(weights_vec)


def get_edge_indices(D):
    """
    Given an incidence matrix D, return a list of tuples representing the edges.
    
    Parameters:
    D (numpy.ndarray): The incidence matrix where rows represent edges and columns represent vertices.
    
    Returns:
    edges (list of tuples): A list of tuples where each tuple (i, j) represents an edge between vertex i and vertex j.
    """
    edges = []
    num_edges, num_vertices = D.shape
    
    for edge_index in range(num_edges):
        # Find the indices of non-zero entries in the row corresponding to the edge
        vertices = np.where(D[edge_index] != 0)[0]
        
        if len(vertices) == 2:
            edges.append(tuple(vertices))
    
    return edges

def gaussian_dist(dataX, sigma = 2):
    """
    Calculating Gaussian weights between x^i and x^j  
        w_ij = exp (-|| x^i - x^j ||^2 / sigma)

    input:  dataX --- data matrix (n by p), each column a point.

            
    output:
        weights --- weights matrix.
    """
    # find sigma_i for x^i.
    n = dataX.shape[0]
    
    # Calculate Euclidean distance between x^i and x^j 
    weights = np.zeros((n,n))
    for i in range(n):
        for j in range(i, n):
            norm_ij = np.linalg.norm(dataX[i,:] - dataX[j,:])
            weights[i,j] = np.exp(- norm_ij ** 2 / sigma)
            weights[j,i] = weights[i,j]
    
    return weights


def gaussian_log_dist(dataX, base = 2, sigma = 2):
    """
    Calculating Gaussian weights between x^i and x^j  
        w_ij = log(exp (-|| x^i - x^j ||^2 / sigma))

    input:  dataX --- data matrix (n by p), each column a point.

            
    output:
        log_weights --- log_weights matrix.
    """
    # find sigma_i for x^i.
    n = dataX.shape[0]
    
    # Calculate Euclidean distance between x^i and x^j 
    log_weights = np.zeros((n,n))
    for i in range(n):
        for j in range(i, n):
            norm_ij = np.linalg.norm(dataX[i,:] - dataX[j,:])
            log_weights[i,j] = base ** (- norm_ij ** 2 / sigma)
            log_weights[j,i] = log_weights[i,j]
    
    return log_weights



import networkx as nx

def find_clusters(A):
    """
    Eric's code: in R package "cvxclustr" for convex clustering
    
    Identifies the connected components of the adjacency graph.

    Parameters:
    - A (scipy.sparse matrix or numpy.ndarray): Adjacency matrix representing the graph.

    Returns:
    - dict: A dictionary containing:
        - 'cluster': An array where each entry indicates the cluster ID of each node.
        - 'size': An array with the sizes of each cluster.
    """
    # Convert the adjacency matrix A to a NetworkX graph
    G = nx.from_numpy_matrix(A, create_using=nx.Graph)

    # Initialize variables
    n = A.shape[0]
    cluster = np.zeros(n, dtype=int)
    k = 1

    # Use BFS to find connected components
    node_seen = np.zeros(n, dtype=bool)
    for i in range(n):
        if not node_seen[i]:
            connected_set = list(nx.bfs_tree(G, source=i).nodes)
            node_seen[connected_set] = True
            cluster[connected_set] = k
            k += 1

    # Count the number of clusters and sizes
    nClusters = k - 1
    size = np.array([np.sum(cluster == j) for j in range(1, nClusters + 1)])
    
    return {'cluster': cluster, 'size': size}

def check_weights(w_union):
    w_min = np.min(w_union)
    w_max = np.max(w_union)
    w_mean = np.mean(w_union)
    w_std = np.std(w_union) 
    # # Plot boxplot
    # plt.figure(figsize=(6, 4))
    # plt.boxplot(w_union, vert=False, patch_artist=True, boxprops=dict(facecolor="lightblue", color="black"), 
    #             whiskerprops=dict(color="black"), capprops=dict(color="black"), medianprops=dict(color="red"))
    # plt.title("Boxplot of w_union")
    # plt.xlabel("Values")
    # plt.grid(True, axis="x", linestyle="--", alpha=0.7)
    # # Show plot
    # plt.show()
    
    return w_min, w_max, w_mean, w_std

#%% Generate clustering data (Gaussian clusters)
import random

# Function to generate Gaussian clusters (your existing code)
def generate_gaussian_clusters(K, n, N, overlap_factor=0.05, random_seed=1):
    np.random.seed(random_seed)
    data = []
    labels = []
    means = []
    covariances = []

    for k in range(K):
        mean = np.random.uniform(-1, 1, n)
        means.append(mean)
        covariance = np.random.rand(1) * np.eye(n) * 0.5
        covariance *= overlap_factor
        covariances.append(covariance)
        cluster_data = np.random.multivariate_normal(mean, covariance, N)
        data.append(cluster_data)
        labels.extend([k] * N)

    data = np.vstack(data)
    labels = np.array(labels)
    return data, labels, means, covariances


def generate_gaussian_clusters_v2(K, n, N, overlap_factor=0.05, random_seed=1):
    """
    # Function to generate Gaussian clusters 
    # also generate the ground-truth means for each data.
    
    """
    np.random.seed(random_seed)
    data = []
    labels = []
    means = []
    GTmeans = []
    covariances = []

    for k in range(K):
        mean = np.random.uniform(-1, 1, n)
        means.append(mean)
        covariance = np.random.rand(1) * np.eye(n) * 0.5
        covariance *= overlap_factor
        covariances.append(covariance)
        cluster_data = np.random.multivariate_normal(mean, covariance, N)
        # GTmeans is a matrix n by N*K, same shape of data.
        GTmeans.append((mean.reshape([n,1]) @ np.ones([1,N])).T)
        data.append(cluster_data)
        labels.extend([k] * N)
        
    GTmeans = np.vstack(GTmeans)
    data = np.vstack(data)
    labels = np.array(labels)
    return data, labels, means, covariances, GTmeans


# Function to generate corner Gaussian clusters
def generate_gaussian_clusters_corner(K, n, N, overlap_factor=0.05, random_seed=1):
    from itertools import product
    """
    
    Generate Gaussian Clusters at the corners of unit boxes.
    K >= 2.
    
    """
    np.random.seed(random_seed)
    data = []
    labels = []
    selected_means = []
    covariances = []
    # Generate all combinations of 1 and -1 for p dimensions
    means = np.array(list(product([1, -1], repeat = n)))
    scale_para = overlap_factor   # scale the covariance.
    
    # randomly select K of corners, and generate Gaussian clusters.
    random.seed(random_seed)
    selected = random.sample(range(means.shape[0]), K)
    for k in selected:
        mean = means[k, :]  # find mean coordinate
        selected_means.append(mean)
        # covariance = np.random.rand(1) * np.eye(n)  # generate covariance
        covariance = scale_para * np.eye(n)    # generate identity covariance
        covariances.append(covariance)
        cluster_data = np.random.multivariate_normal(mean, covariance, N)
        data.append(cluster_data)
        labels.extend([k] * N)

    data = np.vstack(data)
    labels = np.array(labels)
    return data, labels, selected_means, covariances


# Function to generate Gaussian clusters 2d mean fixed.
def generate_gaussian_clusters_2dmean(K, n, N, overlap_factor=0.05, random_seed=1):
    np.random.seed(random_seed)
    data = []
    labels = []
    means = []
    covariances = []
    
    

    for k in range(K):
        if n == 2:
            mean = np.random.uniform(-1.5, 1.5, n)
        if n > 2:
            mean_2d = np.random.uniform(-1.5, 1.5, 2)
            mean = np.concatenate([mean_2d, np.zeros(n-2)])
        means.append(mean)
        covariance = np.random.rand(1) * np.eye(n)*0.5
        covariance *= overlap_factor
        covariances.append(covariance)
        cluster_data = np.random.multivariate_normal(mean, covariance, N)
        data.append(cluster_data)
        labels.extend([k] * N)

    data = np.vstack(data)
    labels = np.array(labels)
    return data, labels, means, covariances

# Function to generate Gaussian clusters 2d mean fixed.
def generate_gaussian_clusters_low4high_dim(K, p_low, p_high, N, overlap_factor=0.05, random_seed=1):
    """
    Generate low dim and project to high dim data by adding zeros.

    Parameters
    ----------
    K : TYPE
        DESCRIPTION.
    p_low : TYPE
        DESCRIPTION.
    p_high : TYPE
        DESCRIPTION.
    N : TYPE
        DESCRIPTION.
    overlap_factor : TYPE, optional
        DESCRIPTION. The default is 0.05.
    random_seed : TYPE, optional
        DESCRIPTION. The default is 1.

    Returns
    -------
    data : TYPE
        DESCRIPTION.
    labels : TYPE
        DESCRIPTION.
    means : TYPE
        DESCRIPTION.
    covariances : TYPE
        DESCRIPTION.

    """
    
    np.random.seed(random_seed)
    data = []
    labels = []
    means = []
    covariances = []
    n = p_high
    

    for k in range(K):
        mean_low = np.random.uniform(-1.5, 1.5, p_low)
        mean_high = np.concatenate([mean_low, np.zeros(p_high - p_low)])
        means.append(mean_high)
        covariance_low = np.random.rand(1) * np.eye(p_low)*0.5
        covariance_low *= overlap_factor
        covariances.append(covariance_low)
        cluster_data_low = np.random.multivariate_normal(mean_low, covariance_low, N)
        cluster_data_high = np.concatenate([cluster_data_low, np.zeros((N, p_high - p_low))], axis = 1)
        data.append(cluster_data_high)
        labels.extend([k] * N)
    data = np.vstack(data)
    labels = np.array(labels)       
        
        
        
        # mean = np.concatenate([mean_low, np.zeros(p_high - p_low)])
        # means.append(mean)
        # covariance = np.random.rand(1) * np.eye(n)*0.5
        # covariance *= overlap_factor
        # covariances.append(covariance)
        # cluster_data = np.random.multivariate_normal(mean, covariance, N)
        # data.append(cluster_data)
        # labels.extend([k] * N)

    # data = np.vstack(data)
    # labels = np.array(labels)
    return data, labels, means, covariances


def generate_2half_moons(num_points=8, cluster_points=7, cluster_radius=0.1, vertical_scale=0.5, random_seed = 1):
    """
    Generate a 2 half-moons dataset with moderately loose clusters.
    
    Parameters:
    num_points (int): Number of core points in each half-moon.
    cluster_points (int): Number of additional points per core point.
    cluster_radius (float): Radius for generating clusters around each core point.
    vertical_scale (float): Scaling factor for the vertical height of the moons.
    
    Returns:
    np.ndarray: Data matrix (p, n) where p=2 (2D points) and n is the total number of data points.
    """
    # Generate first half-moon
    theta1 = np.linspace(0, np.pi, num_points)
    x1 = np.cos(theta1)
    y1 = vertical_scale * np.sin(theta1)

    # Generate second half-moon
    theta2 = np.linspace(0, np.pi, num_points)
    x2 = 1 - np.cos(theta2)
    y2 = vertical_scale * (-np.sin(theta2) - 0.5)

    # Combine the two half-moons
    X1 = np.vstack((x1, y1)).T  # Shape (num_points, 2)
    X2 = np.vstack((x2, y2)).T  # Shape (num_points, 2)
    X = np.vstack((X1, X2))  # Shape (2 * num_points, 2)

    # Generate clusters around each point
    cluster_data = []
    for i in range(X.shape[0]):
        cluster = X[i, :] + cluster_radius * np.random.randn(cluster_points, 2)
        cluster_data.append(cluster)

    # Convert list of clusters into a single matrix
    cluster_data = np.vstack(cluster_data)  # Shape (num_points * 2 * cluster_points, 2)

    # Reshape into (p, n) format
    dataX = cluster_data.T  # Shape (2, n)

    return dataX

# # Generate dataset
# dataX = generate_2half_moons()

# # Print shape of the dataset
# print("Shape of dataX:", dataX.shape)  # Expected: (2, n)

    
def pred_labels(U):
    """
    Assign labels based on solution matrix U: p by n.
    """
    
    # Find unique columns and assign same label to the similar columns.
    
    col_tol = 1e-06
    n = U.shape[1]  # Number of columns
    y_pred = -np.ones(n, dtype=int)  # Initialize labels with -1
    current_label = 0

    for i in range(n):
        if y_pred[i] == -1:  # If the column is not labeled
            # Assign a new label
            y_pred[i] = current_label
            # Compare with other columns
            for j in range(i + 1, n):
                if y_pred[j] == -1:  # Check unlabeled columns
                    diff = np.linalg.norm(U[:, i] - U[:, j])
                    if diff < col_tol:  # If columns are similar
                        y_pred[j] = current_label
            current_label += 1

    return y_pred

# Variation of Information (VI)
def variation_of_information(true_labels, predicted_labels):
    contingency_matrix = np.histogram2d(true_labels, predicted_labels)[0]
    joint_distribution = contingency_matrix / np.sum(contingency_matrix)
    marginal_true = np.sum(joint_distribution, axis=1)
    marginal_pred = np.sum(joint_distribution, axis=0)
    
    h_true = entropy(marginal_true)
    h_pred = entropy(marginal_pred)
    mutual_info = np.sum(joint_distribution * np.log(joint_distribution / (marginal_true[:, None] @ marginal_pred[None, :] + 1e-10) + 1e-10))
    
    return h_true + h_pred - 2 * mutual_info

