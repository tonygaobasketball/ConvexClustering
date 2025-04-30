"""
Project: Convex clustering with general Moreau envelope.

main file.


@author: Zheming Gao

"""
import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse import eye, kron
from scipy.linalg import toeplitz
from fbs_gme3_cc import fbs_gme3_cc
from funcs import *

# Assuming the following functions are already defined:
# prox_l1_admm, identical_cols, sparse_convmtx, decompose_vec, fbs_gme3_cc, incidence_matrix

#%% Load data

### Run certain cell in "Load_data.py"
### Receive:
    # vec_x: data vector;
    # n: the # of data points
    # p: the dimension of data
    


#%% Generate C and calculate D_tilde, C_tilde, B.
# Define parameters
K = 10
gamma = 2.5
hh = np.ones(K) / K
h = np.concatenate([np.zeros(K - 1), [1], np.zeros(K - 1)]) - np.convolve(hh, hh, mode='full')
g = np.convolve(h, [1, -1], mode='full')
G = sparse_convmtx(g, n - 1 - (len(g) - 1))
c = (1 / np.sqrt(gamma)) * g
C = (1 / np.sqrt(gamma)) * G

# Define identity matrices

I_p = eye(p).toarray()
I_n = eye(n).toarray()

# Define matrix Dtilde
D = incidence_matrix(n)
Dtilde = kron(D, I_p)
Ctilde = kron(C, I_p)

# Singular value of Ctilde
sig1_Ctilde = np.linalg.norm(Ctilde.toarray(), 2)


#%% Run FBS algorithms
# Set parameters
rho = 1
mu = 1
c_rate = 2**-12
gamma = 1 / (4 * sig1_Ctilde**2) * c_rate
u0 = vec_x

# Run the FBS algorithm
v_opt, z_opt, u_opt, cput = fbs_gme3_cc(Dtilde, Ctilde, mu, rho, gamma, u0, vec_x, p)

# Decompose the solution vector
U_opt = decompose_vec(u_opt, n)

# Plot the clustering result
plt.figure()
X = matrixData.T
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
numFrames = 10


# Initialize the GIF file
# filename = '2half_moons.gif'
filename = '2half_moons_2nd.gif'
# filename = 'toy-3c.gif'
# filename = 'toy-6c.gif'

# **********************
# Loop through each frame
with imageio.get_writer(filename, mode='I', duration=0.5) as writer:
    for i in range(numFrames):
        # Clear the figure
        plt.figure()
            
        # Generating plots with different parameter gamma.
        # ===========
        # GME3-CC model
    
        c_rate = 4 ** (-numFrames + i)
        gamma = 1/(4 * sig1_Ctilde ** 2) * c_rate   # gamma <= 1/(4 * sig_max(Ctilde)^2)
        v_opt, z_opt, u_opt, cput = fbs_gme3_cc(Dtilde, Ctilde, mu, rho, gamma, u0, vec_x, p)
        U_opt = decompose_vec(u_opt, n)
    
        # Plot the clustering result.
        # figure
        # data matrix
        X = matrixData
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
    
    
