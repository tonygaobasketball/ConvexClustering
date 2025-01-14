#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Build graph and assign weights.

Graph: MST + KNN

@author: Zheming Gao
"""
from sklearn.metrics import pairwise_distances
from sklearn.neighbors import NearestNeighbors
import networkx as nx
import numpy as np
import funcs
import diff_map_funcs
import matplotlib.pyplot as plt

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

# Function: Extract edges from incidence matrix and weights array
def extract_edges(incidence_matrix, weights):
    edges = []
    for idx, row in enumerate(incidence_matrix):
        u, v = np.where(row == 1)[0][0], np.where(row == -1)[0][0]
        edges.append((u, v, weights[idx]))
    return edges

##%% Building graph, and assigning weights.
### Function: Setting weights -----------**##


def assign_weights(matrixData, sigma_k_nrst, weights_type = 2):

    """
    INPUT: 
        Data matrix --- numpy array p by n
        sigma_k_nrst --- the parameter for calculating sigma_ij
        weights_type:
            0: Euclidean distance based weights;
            1: Gaussian distance based weights;
            2: Gaussian distance based with Euclidean calcuated sigma_ij [Chi et al. 2019]
        
    OUTPUT:
        weigh_mat --- numpy array n by n
    
    
    """
    
    if weights_type == 0:
        weight_mat = pairwise_distances(matrixData.T, matrixData.T)
    
    if weights_type == 1:
        base = 1.5  # default base is e.
        weight_mat = funcs.gaussian_log_dist(matrixData.T, base, sigma = 2)
    
    if weights_type == 2:
        # Gaussian distance based with Euclidean 
        # calcuated sigma_ij [Chi et al. 2019]
        base = np.e
        weight_mat = diff_map_funcs.gaussian_kernel_matrix_knrst(matrixData, sigma_k_nrst, base)
        
    return weight_mat



### Create weighted graph with MST and KNN -----------**##

# Function to create a graph from the data matrix and weights.
def create_graph_MST_KNN(matrixData, weight_mat, k_nrst, print_G = 'y'):    

    """
    INPUT: 
        matrixData --- numpy array p by n
        weigh_mat --- numpy array n by n
        k_nrst --- k_nearest parameter. numpy scalar. 
        print_G --- whether print out the graphs (default: yes)
        
    OUTPUT:
        G --- graph (using package networkx)    
        D --- incidence matrix. (np.array, m by m)
        weights_vec --- array of weights on each edges. (m-dim np.array)
    """
    # Create a graph
    Gr = nx.Graph()
    # weight_mat = assign_weights(matrixData)
    p, n = matrixData.shape
    for i in range(n):
        Gr.add_node(i, pos=(matrixData[0, i], matrixData[1, i]))

    # Add edges with distances as weights
    for i in range(n):
        for j in range(i + 1, n):
            Gr.add_edge(i, j, weight = weight_mat[i, j])


    # Compute the pairwise distances between nodes (dist_matrix)
    n_nodes = len(Gr.nodes)
    Dis_all = np.full((n_nodes, n_nodes), np.inf)
    # Populate dist_matrix with edge weights
    for u, v, data in Gr.edges(data=True):
        Dis_all[u, v] = data['weight']
        Dis_all[v, u] = data['weight']

    # Get positions for all nodes
    pos = nx.get_node_attributes(Gr, 'pos')

    ###-----------------------------------
    ### Generate graph with two parts:
        ### 1. Generate edges with minimum spanning tree (MST).
        ### 2. Generate edges with k-nearest neighbor (KNN).
    ### Then combine 1 and 2 to finalize graph with weighted edges.
    ###-----------------------------------



    ### ============
    ### Part 1. MST --------------------**##
    # (1) Create a new graph with inverse weights
    G_inverse = nx.Graph()
    
    for u, v, data in Gr.edges(data=True):
        inverse_weight = 1 / data['weight']
        G_inverse.add_edge(u, v, weight=inverse_weight)
    
    # (2) Generate the Minimum Spanning Tree (MST)
    mst = nx.minimum_spanning_tree(G_inverse)
    
    # When data is 2-d
    if p == 2 and print_G == 'y':
        # Visualize the MST on the original graph
        # Draw the original graph (light gray)
        plt.figure(figsize=(8, 6))
        nx.draw(Gr, pos, with_labels=False, node_size=100, node_color='lightgray', edge_color='lightgray')
        
        # Draw the MST (in blue)
        nx.draw(mst, pos, with_labels=False, node_size=100, edge_color='blue', width = 2)
        
        plt.title('Minimum Spanning Tree (MST)')
        plt.grid(False)
        plt.show()
       
    # When data is high-dim
    if p > 2 and print_G == 'y':
        # Project the high-dimensional data onto 2 dimensions using first 2 dimension
        Trans_mat = np.concatenate([np.eye(2), np.zeros([p-2,2])])  # p by 2 matrix
        data_2d = matrixData.T @ Trans_mat
        
        # Visualize the MST on the 2d
        Gr_2d = nx.Graph()
        for i in range(n):
            Gr_2d.add_node(i, pos=(data_2d[i, 0], data_2d[i, 1]))
            for j in range(i + 1, n):
                # Add edges with distances as weights
                Gr_2d.add_edge(i, j, weight = weight_mat[i, j])
        
        # Plot figure
        plt.figure(figsize=(8, 6))
        pos_2d = {i: (data_2d[i,0], data_2d[i,1]) for i in range(n)}
        Gr_2d_inverse = nx.Graph()
        for u, v, data in Gr_2d.edges(data=True):
            inverse_weight = 1 / data['weight']
            Gr_2d_inverse.add_edge(u, v, weight = inverse_weight)
        mst_2d = nx.minimum_spanning_tree(Gr_2d_inverse)
    
        # Draw the original graph (light gray)
        nx.draw(Gr_2d, pos_2d, with_labels=False, node_size=100, node_color='lightgray', edge_color='lightgray')
        
        # Draw the MST (in blue)
        nx.draw(mst_2d, pos_2d, with_labels=False, node_size=100, edge_color='blue', width=2)
        
        plt.title('Minimum Spanning Tree (MST) on Data (first 2d)')
        plt.grid(False)
        plt.show()
    
    
    ### Calculate the incidence matrix of the mst.
    n_nodes = len(Gr.nodes)
    n_edges_mst = len(mst.edges)
    D_mst = np.zeros((n_edges_mst, n_nodes))
    w_mst = np.zeros(n_edges_mst)

    for idx, (u, v, data) in enumerate(mst.edges(data=True)):
        if u < v:
            D_mst[idx, u] = 1
            D_mst[idx, v] = -1
        else:
            D_mst[idx, v] = 1
            D_mst[idx, u] = -1
        
        # Store the corresponding weight in the weight array
        w_mst[idx] = 1 / data['weight']  # Remember we used inverse weights for the MST        
        
    # # Display the incidence matrix
    # print("Incidence Matrix of the MST (D_mst):\n", D_mst)
    # print("\nWeight Array of the MST (w_mst):\n", w_mst)

    ### ============
    # Part 2: Generate edges with KNN.-------------**##
    # #####****** Select weights assignment method:
    # from scipy.sparse import eye, kron
    # from scipy.linalg import toeplitz
    # import diff_map_funcs
    
    
    ###**********++++++++++++++++++++++++++++++++
    # Calculate D and the singular values.
    incidence_matrices = build_incidence_matrix(matrixData, k_nrst)  # Build the incidence matrices
    # Stack the matrices vertically
    Stacked_incidence_matrices = np.vstack(incidence_matrices)
    D_knn = delete_redundant_rows(Stacked_incidence_matrices)
    # print(f"Matrix D: \n{D}\n")
    
    # Weights for KNN.
    edges = get_edge_indices(D_knn)
    w_knn = np.array([Dis_all[edges[k]] for k in range(D_knn.shape[0])])

    ### ============
    ####%% Part 3: Union of MST and KNN graph.---------**##

    D1 = D_knn   
    weights1 = w_knn
    D2 = D_mst
    weights2 = w_mst
    
    # Extract edges from D1 and D2
    edges1 = extract_edges(D1, weights1)
    edges2 = extract_edges(D2, weights2)

    # Combine edges from both incidence matrices
    edge_dict = {}
    for u, v, weight in edges2:
        edge_dict[(u, v)] = weight
    
    for u, v, weight in edges1:
        if (u, v) in edge_dict:
            if edge_dict[(u, v)] != weight:
                # raise ValueError(f"Edge ({u}, {v}) has different weights in D1 and D2: {edge_dict[(u, v)]} vs {weight}")
                print(f"Edge ({u}, {v}) has different weights in D1 and D2: {edge_dict[(u, v)]} vs {weight}")
                if edge_dict[(u, v)] < weight:
                    edge_dict[(u, v)] = weight
        else:
            edge_dict[(u, v)] = weight


    # Create the incidence matrix and weight array for the union graph
    num_nodes = n
    num_edges = len(edge_dict)
    D_union = np.zeros((num_edges, num_nodes))
    weights_union = np.zeros(num_edges)
    
    for idx, ((u, v), weight) in enumerate(edge_dict.items()):
        if u < v:  # Ensure that 1 is on the left of -1
            D_union[idx, u] = 1
            D_union[idx, v] = -1
        else:
            D_union[idx, v] = 1
            D_union[idx, u] = -1
        weights_union[idx] = weight


    # # Print the incidence matrix and weight array for the union graph
    # print("Union Incidence Matrix:")
    # print(D_union)
    # print("\nUnion Weights Array:")
    # print(weights_union)


    if p == 2 and print_G == 'y':
        """
        We may visualize the plot of the union graph G_u
        on 2-d case.
        """ 
        # Create a NetworkX graph
        G_u = nx.Graph()
        
        # Add edges to the graph with weights
        for (u, v), weight in edge_dict.items():
            G_u.add_edge(u, v, weight = weight)
        
        # # Visualize the graph using the original coordinates
        # pos = {i: matrixData[:, i] for i in range(n)}
        
        # Draw the nodes
        nx.draw_networkx_nodes(G_u, pos, node_size=10, node_color='lightblue')
        
        # Draw the edges with weights
        nx.draw_networkx_edges(G_u, pos, edgelist=G_u.edges(data=True), width=1)
        
        # # Draw the node labels
        # nx.draw_networkx_labels(G_u, pos, font_size=6, font_color='black')
        
        # Draw the edge labels (weights)
        edge_labels = {(u, v): f"{d['weight']:.2f}" for u, v, d in G_u.edges(data=True)}
        nx.draw_networkx_edge_labels(G_u, pos, edge_labels=edge_labels, font_size = 12, label_pos=0.5)
        
        # Show the plot
        plt.title("Union Graph Visualization")
        plt.show()
        
    if p > 2 and print_G == 'y':
        """
        We may visualize the plot of the union graph G_u
        on first 2-d.
        """ 
        # Create a NetworkX graph
        G_u_2d = nx.Graph()
        
        # Add edges to the graph with weights
        for (u, v), weight in edge_dict.items():
            G_u_2d.add_edge(u, v, weight = weight)
        
        # # Visualize the graph using the original coordinates
        # pos = {i: matrixData[:, i] for i in range(n)}
        
        # Draw the nodes
        nx.draw_networkx_nodes(G_u_2d, pos_2d, node_size=10, node_color='lightblue')
        
        # Draw the edges with weights
        nx.draw_networkx_edges(G_u_2d, pos_2d, edgelist = G_u_2d.edges(data=True), width=1)
        
        # # Draw the node labels
        # nx.draw_networkx_labels(G_u, pos, font_size=6, font_color='black')
        
        # Draw the edge labels (weights)
        edge_labels = {(u, v): f"{d['weight']:.2f}" for u, v, d in G_u_2d.edges(data=True)}
        nx.draw_networkx_edge_labels(G_u_2d, pos_2d, edge_labels=edge_labels, font_size=12, label_pos=0.5)
        
        # Show the plot
        plt.title("Union Graph Visualization (first 2d)")
        plt.show()    
    
    
    # Return union graph G_u
    
    # sig1_D = np.linalg.norm(D, 2)  # largest singular value of D matrix.
    return D_union, weights_union
