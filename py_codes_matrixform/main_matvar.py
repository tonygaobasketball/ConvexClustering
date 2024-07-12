"""
Project: Convex clustering with general Moreau envelope.

main file (matrix variable)


@author: Zheming Gao

"""
import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse import eye, kron
from scipy.linalg import toeplitz
from fbs_gme3_cc_mat import *
from funcs import *

# Assuming the following functions are already defined:
# prox_l1_admm, identical_cols, sparse_convmtx, decompose_vec, fbs_gme3_cc, incidence_matrix

#%% Load data

### Run certain cell in "Load_data.py"
### Receive:
    # vec_x: data vector;
    # n: the # of data points
    # p: the dimension of data
    
mat_x = vec_x.reshape(n,p).T

# Plot the data
plt.figure()
plt.plot(mat_x[0,:], mat_x[1,:], '*')
plt.xlabel('x1')
plt.ylabel('x2')
plt.title(f'Two half moons data (n = {n})')
plt.grid(True)
plt.show()

#%% Matrix D and C. (Euclidean)

# Example usage
# k_nearest = n-1  # Number of nearest neighbors to find
k_nearest = 9
incidence_matrices = build_incidence_matrix(mat_x, k_nearest)  # Build the incidence matrices

# # Print the incidence matrices for each column
# for j, A_j in enumerate(incidence_matrices):
#     print(f"Incidence matrix for column {j+1}:\n{A_j}\n")
    
# Stack the matrices vertically
Stacked_incidence_matrices = np.vstack(incidence_matrices)
D = delete_redundant_rows(Stacked_incidence_matrices)
# print(f"Matrix D: \n{D}\n")
sig1_D = np.linalg.norm(D, 2)


# Define identity matrices
I_p = eye(p).toarray()
I_n = eye(n).toarray()

C = eye(D.shape[0]).toarray()  # Let matrix C be idenity.

# Singular value of Ctilde
sig1_C = np.linalg.norm(C, 2)

# #%% Generate C, D and calculate B.
# # Define parameters
# K = 10
# gamma = 2.5
# hh = np.ones(K) / K
# h = np.concatenate([np.zeros(K - 1), [1], np.zeros(K - 1)]) - np.convolve(hh, hh, mode='full')
# g = np.convolve(h, [1, -1], mode='full')
# G = sparse_convmtx(g, n - 1 - (len(g) - 1))
# c = (1 / np.sqrt(gamma)) * g
# C = (1 / np.sqrt(gamma)) * G


# # # Define matrix Dtilde
# # D = incidence_matrix(n)

# # Singular value of Ctilde
# sig1_C = np.linalg.norm(C.toarray(), 2)

#%% Matrix D and C. (weighted)
#--------------------
# Diffusion distance based D and weights.
#--------------------

# k_nearest = n-1  # Number of nearest neighbors to find
k_nearest = 9

diff_t = 1
diff_l = n-1

_, diff_dis = diff_map(mat_x, diff_t, diff_l)
incidence_matrices = build_incidence_matrix_diff(mat_x, diff_dis, k_nearest)  # Build the incidence matrices

# # Print the incidence matrices for each column
# for j, A_j in enumerate(incidence_matrices):
#     print(f"Incidence matrix for column {j+1}:\n{A_j}\n")
    
# Stack the matrices vertically
Stacked_incidence_matrices = np.vstack(incidence_matrices)
D = delete_redundant_rows(Stacked_incidence_matrices)
# print(f"Matrix D: \n{D}\n")
sig1_D = np.linalg.norm(D, 2)


# Define identity matrices
I_p = eye(p).toarray()
I_n = eye(n).toarray()

# Put Gaussian weights on edges. Equivalent to multiply weights
# on diagonals of C.
weights_mat = Gauss_weight_diff_k_nrst(mat_x, diff_t, diff_l, k_nearest) 
weights_vec = extract_nonzero_elements(D, weights_mat)

C = eye(D.shape[0]).toarray() @ np.diag(weights_vec) # Let matrix C be idenity.

# Singular value of Ctilde
sig1_C = np.linalg.norm(C, 2)

#%% Matrix D and C. (weighted)
#--------------------
# Euclidean distance based D and weights.
#--------------------

k_nearest = n-1  # Number of nearest neighbors to find
# k_nearest = 9

incidence_matrices = build_incidence_matrix(mat_x, k_nearest)  # Build the incidence matrices

# Stack the matrices vertically
Stacked_incidence_matrices = np.vstack(incidence_matrices)
D = delete_redundant_rows(Stacked_incidence_matrices)
# print(f"Matrix D: \n{D}\n")
sig1_D = np.linalg.norm(D, 2)


# Define identity matrices
I_p = eye(p).toarray()
I_n = eye(n).toarray()

# Put Gaussian weights on edges. Equivalent to multiply weights
# on diagonals of C.
weights_mat = Gauss_weight_k_nrst(mat_x, k_nearest) 
weights_vec = extract_nonzero_elements(D, weights_mat)

C = eye(D.shape[0]).toarray() @ np.diag(weights_vec) # Let matrix C be idenity.

# Singular value of Ctilde
sig1_C = np.linalg.norm(C, 2)
#%% Generate connected graph with data and the edges in D.
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx

def plot_graph(nodes, incidence_matrix):
    # Create an undirected graph
    G = nx.Graph()

    # Add nodes to the graph
    for i, (x, y) in enumerate(nodes):
        G.add_node(i, pos=(x, y))

    # Add edges to the graph using the incidence matrix
    num_edges = incidence_matrix.shape[1]
    for j in range(num_edges):
        start_nodes = np.where(incidence_matrix[:, j] == -1)[0]
        end_nodes = np.where(incidence_matrix[:, j] == 1)[0]
        
        if start_nodes.size > 0 and end_nodes.size > 0:
            start_node = start_nodes[0]
            end_node = end_nodes[0]
            if start_node in G.nodes and end_node in G.nodes:
                G.add_edge(start_node, end_node)

    # Get positions of nodes
    pos = nx.get_node_attributes(G, 'pos')

    # Draw the graph
    plt.figure(figsize=(8, 6))
    nx.draw(G, pos, with_labels = False, node_color='lightblue', edge_color='gray', node_size=200)
    
    # Add axes and grid
    ax = plt.gca()
    ax.set_axis_on()
    ax.tick_params(left=True, bottom=True, labelleft=True, labelbottom=True)
    plt.grid(True)
    
    plt.title(f"Undirected Graph with k = {k_nearest}")
    plt.show()
    
# Plot the graph with edges.
nodes = mat_x.T
incidence_matrix = D.T

plot_graph(nodes, incidence_matrix)


#%% Run FBS algorithms
# Set parameters
rho = 1
mu = 1
c_rate = 2**-1
# c_rate = 0.01

gamma_up = 1 / (sig1_D * sig1_C)**2
gamma = gamma_up * c_rate

# gamma = 0.08

U0 = mat_x       # initialize U

# Run the FBS algorithm
U_opt, V_opt, Z_opt, cput = fbs_gme3_cc_mat(D, C, sig1_C, mu, rho, gamma, U0, mat_x)

# # Decompose the solution vector
# U_opt = decompose_vec(U_opt, n)

# Plot the clustering result
plt.figure()
X = mat_x.T
plt.scatter(X[:, 0], X[:, 1], c='b', marker='*', label='Data Points')

# Tolerance for considering columns as identical
tol_uniqcol = 1e-2

# Extract the unique columns
Centroids, uniqueCols, Clusters_cell = identical_cols(U_opt, tol_uniqcol)

# Plot clusters
for k, cluster in enumerate(Clusters_cell):
    plt.scatter(Centroids[0, k], Centroids[1, k], edgecolor='r', facecolor='none', label='Centroids' if k == 0 else "")

plt.xlabel('X')
plt.ylabel('Y')
plt.legend()
plt.title('Scatter Plot of Clusters')
plt.show()


#%% Run FBS and generate interchange graphs.
import imageio
from matplotlib.colors import to_rgb


# Define the number of frames
numFrames = 7


# Initialize the GIF file
# filename = '2half_moons.gif'
# filename = '2half_moons_2nd.gif'
# filename = 'toy-3c.gif'
# filename = 'toy-6c.gif'
filename = 'temp.gif'

# **********************
# Loop through each frame
with imageio.get_writer(filename, mode='I', duration=0.5) as writer:
    for i in range(numFrames+1):
        # Clear the figure
        plt.figure()
            
        # Generating plots with different parameter gamma.
        # ===========
        # GME3-CC model
    
        c_rate = 2 ** (- numFrames + i - -2)
        # gamma = 1/(4 * sig1_C ** 2) * c_rate   # gamma <= 1/(4 * sig_max(Ctilde)^2)
        gamma_up = 1 / (sig1_D * sig1_C)**2
        gamma = gamma_up * c_rate
        
        U_opt, V_opt, Z_opt, cput = fbs_gme3_cc_mat(D, C, sig1_C, mu, rho, gamma, U0, mat_x)
        # U_opt = decompose_vec(u_opt, n)
    
        # Plot the clustering result.
        # figure
        # data matrix
        X = mat_x
        plt.scatter(X[0,:], X[1,:], c='b', marker='*', label='Data Points')
        
        # Tolerance for considering columns as identical
        tol_uniqcol = 1e-2
        
        # Extract the unique columns
        Centroids, uniqueCols, Clusters_cell  = identical_cols(U_opt, tol_uniqcol)
        
        
        # Plot clusters.
        for k, cluster in enumerate(Clusters_cell):
            plt.scatter(Centroids[0, k], Centroids[1, k], edgecolor='r', facecolor='none', label='Centroids' if k == 0 else "")
        plt.xlabel('X')
        plt.ylabel('Y')
        plt.legend()
        title_str = f'Data and Centroids: $\\gamma = \\frac{{1}}{{{int(1 / c_rate)}}} \\cdot \\gamma_{{ub}}$'
        plt.title(title_str, fontsize=10)
        
        # Capture the plot as an image
        plt.draw()
        frame = np.frombuffer(plt.gcf().canvas.tostring_rgb(), dtype='uint8')
        frame = frame.reshape(plt.gcf().canvas.get_width_height()[::-1] + (3,))
        writer.append_data(frame)
        plt.close()
    
print(f'GIF saved as {filename}')
    
    
