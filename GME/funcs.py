"""
Project: Convex clustering with general Moreau envelope.

Function tools.


@author: Zheming Gao
"""

import numpy as np
from scipy.sparse import spdiags
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