"""
Project: Convex clustering with general Moreau envelope.

Loading data.


@author: Zheming Gao

"""
#%% Define functions.

import scipy.io
import numpy as np


import numpy as np
import matplotlib.pyplot as plt


def generate_clusters(p, n_k, K, clusterCenters):
    """
    Generate K clusters of p-dimensional data points. Each cluster has n_k data points.
    The cluster centers are given by the matrix clusterCenters, where each column is a cluster center.
    
    Parameters:
    p (int): Dimensionality of the data points.
    n_k (int): Number of data points per cluster.
    K (int): Number of clusters.
    clusterCenters (numpy.ndarray): Matrix of cluster centers with shape (p, K).
    
    Returns:
    dict: A dictionary with clusters, where each key corresponds to a cluster index and each value
          contains the data points for that cluster.
    """
    
    # Number of clusters
    numClusters = K
    
    # Number of points per cluster
    pointsPerCluster = n_k
    
    # Define a standard deviation for the clusters
    stdDev = 0.1
    
    # Initialize a list to hold clusters
    clusters = []
    
    # Generate data points for each cluster
    for i in range(numClusters):
        # Generate points around the cluster center
        clusterData = stdDev * np.random.randn(p, pointsPerCluster) + clusterCenters[:, i].reshape(p, 1)
        
        # Store the cluster data in the list
        clusters.append({'points': clusterData})
        # clusters[i] = clusterData
    
    return clusters

# # Example usage
# p = 2
# n_k = 50
# K = 3
# clusterCenters = np.array([[1, 3, 5], [2, 4, 6]])
# clusters = generate_clusters(p, n_k, K, clusterCenters)

# # Plot the data
# import matplotlib.pyplot as plt

# for i in range(K):
#     plt.scatter(clusters[i][0, :], clusters[i][1, :], label=f'Cluster {i+1}')

# plt.xlabel('Feature 1')
# plt.ylabel('Feature 2')
# plt.title('Generated Clustering Data')
# plt.legend()
# plt.grid(True)
# plt.show()



def generate_2half_moons():
    """
    Generate data for the 2 half moons example with moderately loose clusters.
    
    Returns:
    list of dict: List where each dictionary contains the points for each cluster.
    """
    # Number of points in each half moon
    num_points = 5

    # Vertical scaling factor
    vertical_scale = 0.5

    # Generate first half moon
    theta1 = np.linspace(0, np.pi, num_points)
    x1 = np.cos(theta1)
    y1 = vertical_scale * np.sin(theta1)

    # Generate second half moon
    theta2 = np.linspace(0, np.pi, num_points)
    x2 = 1 - np.cos(theta2)
    y2 = vertical_scale * (-np.sin(theta2) - 0.5)

    # Combine the two half moons
    X1 = np.vstack((x1, y1)).T
    X2 = np.vstack((x2, y2)).T
    X = np.vstack((X1, X2))

    # Parameters for clusters
    cluster_points = 10
    cluster_radius = 0.1  # Moderate cluster radius for balanced looseness

    # Initialize list to hold clusters
    clusters = []

    # Add small clusters around each point with moderate variance
    for i in range(X.shape[0]):
        # Create a cluster around each point
        cluster_points_data = X[i, :] + cluster_radius * np.random.randn(cluster_points, 2)
        clusters.append({'points': cluster_points_data})

    return clusters

# # Generate the 2 half moons data
# data = generate_2half_moons()

# # Plot the dataset with moderately loose small clusters
# plt.figure()
# for cluster in data:
#     plt.scatter(cluster['points'][:, 0], cluster['points'][:, 1], s=10)
# plt.title('Two Half Moons with Moderately Loose Clusters')
# plt.xlabel('X')
# plt.ylabel('Y')
# plt.axis('equal')
# plt.show()

# # Example usage
# numClusters = len(data)

# # Initialize an empty matrix to store the data
# matrixData = []  # Data matrix with each column a data point
# vec_x = []       # Long data vector with all data points stacked

# # Loop through each cluster and add its contents to the matrix
# for i in range(numClusters):
#     varData = data[i]['points']
    
#     # Append the numerical data to the matrix
#     matrixData.append(varData)

# # Convert the list of arrays to a single numpy array
# matrixData = np.vstack(matrixData)

# # Record the dimension of the data
# n = matrixData.shape[0]
# p = 2

# # Initialize an empty list to store the long data vector
# vec_x = []

# # Stack all data points
# for i in range(n):
#     vec_x.append(matrixData[i, :])

# # Convert the list to a numpy array
# vec_x = np.hstack(vec_x)

# # Example usage
# print("matrixData:\n", matrixData)
# print("vec_x:\n", vec_x)
# print("Number of data points (n):", n)
# print("Number of features (p):", p)

#%%
#---------------------------------------------------
# Load the data from the .mat file
data = scipy.io.loadmat("mammals.mat")['data']

# Initialize an empty list to store the data
matrixData = []

# Extract field names
variables = data.dtype.names

# Loop through each variable and add its contents to the matrix
for i in range(1, len(variables)):  # Start from 1 to skip the first variable
    varName = variables[i]
    varData = data[0, 0][i]
    
    # Append the numerical data to the matrix
    matrixData.append(varData)

# Convert the list to a numpy array and transpose it
matrixData = np.hstack(matrixData).T

# Convert the matrix to a double precision numpy array
matrixData = matrixData.astype(np.float64)

# Initialize an empty list to store the long data vector
vec_x = []

# Stack all data points
for i in range(matrixData.shape[1]):
    vec_x.append(matrixData[:, i])

# Convert the list to a numpy array
vec_x = np.hstack(vec_x)

# Record the dimension of the data.
n, p =  matrixData.T.shape

#---------------------------------------------------


#%%
#---------------------------------------------------
# Load the 2half_moons data
data = generate_2half_moons()
numClusters = len(data)

# Initialize an empty matrix to store the data
matrixData = []  # Data matrix with each column a data point
vec_x = []       # Long data vector with all data points stacked

# Loop through each cluster and add its contents to the matrix
for i in range(numClusters):
    varData = data[i]['points']
    
    # Append the numerical data to the matrix
    matrixData.append(varData)

# Convert the list of arrays to a single numpy array
matrixData = np.vstack(matrixData).T

# Initialize an empty list to store the long data vector
vec_x = []

# Stack all data points
for i in range(n):
    vec_x.append(matrixData[i, :])

# Convert the list to a numpy array
vec_x = np.hstack(vec_x)

# Record the dimension of the data.
n, p =  matrixData.T.shape


# Example usage
print("matrixData:\n", matrixData)
print("vec_x:\n", vec_x)
print("Number of data points (n):", n)
print("Number of features (p):", p)
#---------------------------------------------------


#%%
#---------------------------------------------------
import numpy as np
import matplotlib.pyplot as plt

# Generate toy-6c example

# Parameters
K = 6
n_k = 10
p = 2
cls_centers = np.array([[1, 0], [1, -1], [0, -1], [-1, 0], [-1, 1], [0, 1]]).T

# Generate the clusters
data = generate_clusters(p, n_k, K, cls_centers)
n = n_k * K

# Concatenate all cluster data into a single matrix
matrixData = np.hstack([data[i]['points'] for i in range(K)])

# Initialize an empty list to store the long data vector
vec_x = []

# Stack all data points
for i in range(matrixData.shape[1]):
    vec_x.append(matrixData[:, i])

# Convert the list to a numpy array
vec_x = np.hstack(vec_x)

# Record the dimension of the data.
n, p =  matrixData.T.shape


# Plot the data
plt.figure()
plt.plot(matrixData[0, :], matrixData[1, :], 'o')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.title('Generated Clustering Data')
plt.grid(True)
plt.show()

#---------------------------------------------------