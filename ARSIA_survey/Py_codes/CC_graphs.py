"""
Graph constructions support file.

Author: Zheming Gao
"""


from sklearn.metrics import pairwise_distances
from sklearn.neighbors import NearestNeighbors
import networkx as nx
import numpy as np
# import funcs
# import diff_map_funcs
import matplotlib.pyplot as plt
from scipy.spatial.distance import pdist, squareform
from itertools import combinations
from sklearn.metrics import pairwise_distances

def gaussian_kernel_matrix_knrst(D, k_nearest = 3, base = np.e):
    """
    Computes the Gaussian kernel matrix for a given data matrix D.
    
    Parameters:
    - D (numpy.ndarray): Data matrix where each column is a data point.
    - k_nearest: 
        Calculate sigma_i as the median of Euclidean distances 
                between x_i to its k nearest neighbors.
                
                sigma_{ij} = sigma_i * sigma_j.
    
    
    Returns:
    - K (numpy.ndarray): The Gaussian kernel matrix.
    """
   
    # Compute pairwise distances (D is #features by #samples)
    pairwise_dist = pairwise_distances(D.T)
    
    # Compute the sigma 
    # distances to p * n-th nearest neighbor are used. 
    # Default value is p = .01
    # Determine the epsilon value
    
    n = pairwise_dist.shape[0]
    # n_neighbors = k_nearest
    
    sig = np.zeros(n)
    for i in range(n):
        cand_arr = pairwise_dist[:,i].copy()
        cand_arr[i] = np.inf # make the ith element big enough.
        # Get the indices of the sorted array
        sorted_indices = np.argsort(cand_arr)

        # Get the indices of the k smallest elements
        smallest_indices = sorted_indices[:k_nearest]

        # Get the first k smallest elements using these indices
        smallest_elements = cand_arr[smallest_indices]
        
        # The first must be zero. Disgard the first item.
        sig[i] = np.median(smallest_elements)
    
    # sigma = np.median(np.sort(pairwise_dist, axis=1)[:, n_neighbors])
    sig_mat = np.outer(sig, sig)
    
    # Compute the Gaussian kernel matrix
    # print(pairwise_dist)
    # print(sig_mat)
    K = base ** (- pairwise_dist ** 2 / sig_mat)
    
    return K

def compute_mu_ij(i, j, alpha, W, id_set):
    """
    Compute mu_ij
    """
    K = len(id_set)
    mu_val = 0.0
    for beta in range(K):
        if beta != alpha:
            w_i_beta = np.sum(W[i, id_set[beta]])
            w_j_beta = np.sum(W[j, id_set[beta]])
            mu_val += np.abs(w_i_beta - w_j_beta)
    return mu_val

def compute_gamma_min_max(AA, W, GT_means, id_set, q=2):
    """
    Compute gamma_min and gamma_max based on provided definitions.

    Parameters:
    - AA: ndarray (p, n) where each column is a_i
    - W: ndarray (n, n), where element (i,j) is w_ij
    - GT_means: ndarray (p, K), where each column is a^{alpha}
    - id_set: list of lists; id_set[k] contains indices in cluster k
    - q: norm type (e.g. 2 for Euclidean)

    Returns:
    - gamma_min: float
    - gamma_max: float
    - check: 1 if gamma_min < gamma_max, 0 otherwise.
    """
    # n = AA.shape[1]
    K = len(id_set)

    # Compute gamma_min
    gamma_min = -np.inf
    for alpha in range(K):
        I_alpha = id_set[alpha]
        n_alpha = len(I_alpha)
        for i in I_alpha:
            for j in I_alpha:
                if i != j:
                    norm_val = np.linalg.norm(AA[:, i] - AA[:, j], ord=q)
                    mu_ij = compute_mu_ij(i, j, alpha, W, id_set)
                    denom = n_alpha * W[i, j] - mu_ij  # denom > 0.
                    # if denom != 0:
                    ratio = norm_val / denom
                    gamma_min = max(gamma_min, ratio)

    # Compute gamma_max
    gamma_max = np.inf
    for alpha, beta in combinations(range(K), 2):
        a_alpha = GT_means[:, alpha]
        a_beta = GT_means[:, beta]
        numerator = np.linalg.norm(a_alpha - a_beta, ord=q)

        n_alpha = len(id_set[alpha])
        n_beta = len(id_set[beta])

        # w^{(alpha, l)} for all l != alpha
        sum_alpha = 0.0
        for l in range(K):
            if l != alpha:
                w_alpha_l = np.sum([np.sum(W[i, id_set[l]]) for i in id_set[alpha]])
                sum_alpha += w_alpha_l
        sum_alpha /= n_alpha

        # w^{(beta, l)} for all l != beta
        sum_beta = 0.0
        for l in range(K):
            if l != beta:
                w_beta_l = np.sum([np.sum(W[i, id_set[l]]) for i in id_set[beta]])
                sum_beta += w_beta_l
        sum_beta /= n_beta

        denom = sum_alpha + sum_beta
        if denom != 0:
            ratio = numerator / denom
            gamma_max = min(gamma_max, ratio)
            
        # Check if gamma_min < gamma_max
        
        if gamma_min < gamma_max:
            check = 1
        else:
            check = 0

    return gamma_min, gamma_max, check

# def gamma_bounds(X, W, GT_means, id_set):
#     """
#     Calculate the upper bound and lower bound of gamma for 
#     ref: Sun et al. 2021 Convex Clustering: Model, Theoretical 
#     Guarantee and Eﬃcient Algorithm. Page 10, formula (11).
#     (p = 2, q = 2)
    
#     Input: 
#         X --- Data matrix. p by n data array.
#         W --- weight matrix. m by m array, where m is the number 
#               of edges.
#         GT_means --- p by K centroids matrix. 
#         id_set --- set of indices in X for GT_means.
#                   i.e., id_set[k] is the index list of points in cluster k, 
#                   corresponding to centroid GT_means[:,k]. k = 1, ..., K_cls.
        
#     Output: 
#         ga_lb --- lower bound of gamma.
#         ga_ub --- upper bound of gamma.
            
#     """
    
#     p, n = X.shape
#     n_edges = W.shape[0]
#     K_cls = GT_means.shape[1]
    
#     # First, calculate mu.
#     for k in range(K_cls):
        
    
    
    


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
            # print(f'j = {j}')
            # print(f'idx = {idx}')
            # print(f'neighbor_idx = {neighbor_idx}')
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
        
        if len(vertices) != 2:
            print(tuple(vertices))
        
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


def assign_weights(matrixData, sigma_k_nrst = None, weights_type = 2):

    """
    INPUT: 
        Data matrix --- numpy array p by n
        sigma_k_nrst --- the parameter for calculating sigma_ij (only for Guassian weights)
        weights_type:
            0: Euclidean distance based weights;
            1: Gaussian distance based weights;
            2: Gaussian distance based with Euclidean calcuated sigma_ij [Chi et al. 2019]
            3: naive all-one weights.
    OUTPUT:
        weigh_mat --- numpy array n by n
    
    
    """
    
    if weights_type == 1:
        temp = pairwise_distances(matrixData.T, matrixData.T)
        weight_mat = np.zeros(temp.shape)
        for i in range(temp.shape[0]):
            for j in range(temp.shape[1]):
                if i == j:
                    weight_mat[i,j] = -999
                else:
                    weight_mat[i,j] = 1 / temp[i,j]
    # if weights_type == 1:
    #     base = 1.5  # default base is e.
    #     weight_mat = funcs.gaussian_log_dist(matrixData.T, base, sigma = 2)
    
    if weights_type == 2:
        # Gaussian distance based with Euclidean 
        # calcuated sigma_ij [Chi et al. 2019]
        base = np.e
        # if sigma_k_nrst == None:
        #     sigma_k_nrst = 3
        sigma_k_nrst = np.int(matrixData.shape[1] * 0.1)
        weight_mat = gaussian_kernel_matrix_knrst(matrixData, sigma_k_nrst, base)
    
    if weights_type == 3: # all equal weights.
        n = matrixData.shape[1]
        weight_mat = np.ones([n,n])
        
    if weights_type == 4: # w(r) = nu^{p+1} exp(-nu * r)
        # ref: 2022, Dunlap and Mourrat, 
        # LOCAL VERSIONS OF SUM-OF-NORMS CLUSTERING
        p, n = matrixData.shape
        nu = n ** (3 / (4 * p))  # suggested on page 5.
        temp = pairwise_distances(matrixData.T, matrixData.T)
        weight_mat = np.zeros(temp.shape)
        for i in range(temp.shape[0]):
            for j in range(temp.shape[1]):
                weight_mat[i,j] = (nu ** (p+1)) * np.exp(-nu * temp[i,j])
    
        
    return weight_mat



def conv_comb_weights(matrixData, sigma_k_nrst = None, al = .5):

    """
    Convex combination of uniform and Gaussian kernel weights.

    INPUT: 
        Data matrix --- numpy array p by n
        sigma_k_nrst --- the parameter for calculating sigma_ij (only for Guassian weights)
        al --- parameter of the convex combination
        
    OUTPUT:
        weigh_mat --- numpy array n by n
    
    
    """
    p, n = matrixData.shape
    # Gaussian distance based with Euclidean 
    # calcuated sigma_ij [Chi et al. 2019]
    base = np.e
    sigma_k_nrst = np.int(matrixData.shape[1] * 0.1)
    weight_Gau = gaussian_kernel_matrix_knrst(matrixData, sigma_k_nrst, base)
    weight_Uni = np.ones([n,n])
        
    weight_mat = al * weight_Gau + (1 - al) * weight_Uni
        
    return weight_mat

#%% MST graph.
def create_graph_MST(matrixData, weight_mat, print_G='y'):
    """
    INPUT: 
        matrixData --- numpy array p by n
        weight_mat --- numpy array n by n
        print_G --- whether to print out the graphs (default: 'y')
        
    OUTPUT:
        D --- incidence matrix (np.array, m by n)
        weights_vec --- array of weights on each edge (m-dimensional np.array)
    """
    p, n = matrixData.shape

    # Compute pairwise Euclidean distance matrix
    dist_matrix = squareform(pdist(matrixData.T, metric='euclidean'))

    # Create a graph with Euclidean distances
    Gr = nx.Graph()
    for i in range(n):
        for j in range(i + 1, n):
            Gr.add_edge(i, j, weight=dist_matrix[i, j])

    # Compute the Minimum Spanning Tree (MST) using Euclidean distances
    mst = nx.minimum_spanning_tree(Gr)

    # Visualize MST if required
    if print_G == 'y':
        if p == 2:  # Direct 2D plot
            pos = {i: (matrixData[0, i], matrixData[1, i]) for i in range(n)}
            plt.figure(figsize=(8, 6))
            nx.draw(Gr, pos, with_labels=False, node_size=100, node_color='lightgray', edge_color='lightgray', width=1)
            nx.draw(mst, pos, with_labels=False, node_size=100, edge_color='blue', width=1)
            # plt.title('MST using Euclidean Distance')
            plt.grid(False)
            plt.show()

        elif p > 2:  # Project high-dimensional data onto 2D
            data_2d = matrixData[:2, :]  # Take the first two dimensions
            pos_2d = {i: (data_2d[0, i], data_2d[1, i]) for i in range(n)}
            plt.figure(figsize=(8, 6))
            nx.draw(Gr, pos_2d, with_labels=False, node_size=100, node_color='lightgray', edge_color='lightgray', width=1)
            nx.draw(mst, pos_2d, with_labels=False, node_size=100, edge_color='blue', width=1)
            # plt.title('MST (Projected to 2D)')
            plt.grid(False)
            plt.show()

    # Compute the incidence matrix and update weights from weight_mat
    n_edges_mst = len(mst.edges)
    D_mst = np.zeros((n_edges_mst, n))  # m x n incidence matrix
    w_mst = np.zeros(n_edges_mst)  # m-dim weight vector

    for idx, (u, v) in enumerate(mst.edges()):
        D_mst[idx, u] = 1
        D_mst[idx, v] = -1
        w_mst[idx] = weight_mat[u, v]  # Use weight from weight_mat instead of Euclidean distance

    return D_mst, w_mst

#%% KNN graph.
def create_graph_KNN(matrixData, weight_mat, k_nrst, print_G='y'):
    """
    Constructs a k-Nearest Neighbors Graph (kNNG) using Euclidean distances for structure,
    assigns edge weights from a given weight matrix, and overlays a fully connected graph (FCG) for visualization.

    INPUT: 
        matrixData --- numpy array p by n
        weight_mat --- numpy array n by n
        k_nrst --- k-nearest neighbors parameter (int)
        print_G --- whether to print the graph (default: 'y')
        
    OUTPUT:
        D_knn --- incidence matrix (np.array, m by n)
        w_knn --- array of weights on each edge (m-dimensional np.array)
    """

    p, n = matrixData.shape

    # Step 1: Compute Incidence Matrix Using k-NN
    incidence_matrices = build_incidence_matrix(matrixData, k_nrst)  
    Stacked_incidence_matrices = np.vstack(incidence_matrices)
    D_knn = delete_redundant_rows(Stacked_incidence_matrices)

    # Step 2: Extract Edges from Incidence Matrix
    edges_knn = get_edge_indices(D_knn)

    # Step 3: Assign Weights from weight_mat
    w_knn = np.array([weight_mat[u, v] for u, v in edges_knn])

    # Step 4: Extract edges with weights for visualization
    weighted_edges = extract_edges(D_knn, w_knn)

    # Step 5: Compute Fully Connected Graph (FCG) for Visualization
    dist_matrix = squareform(pdist(matrixData.T, metric='euclidean'))  # Compute Euclidean distances
    Gr = nx.Graph()
    for i in range(n):
        for j in range(i + 1, n):
            Gr.add_edge(i, j, weight=dist_matrix[i, j])  # Use Euclidean distance for FCG visualization

    # Step 6: Visualization (if required)
    if print_G == 'y':
        kNNG = nx.Graph()
        kNNG.add_weighted_edges_from(weighted_edges)

        pos_knn = {i: matrixData[:, i] for i in range(n)} if p == 2 else {i: matrixData[:2, i] for i in range(n)}
        
        plt.figure(figsize=(8, 6))

        # Plot Fully Connected Graph (FCG) first (light gray)
        nx.draw(Gr, pos_knn, with_labels=False, node_size=100, node_color='lightgray', edge_color='lightgray', width=1)

        # Overlay k-NN Graph (kNNG) with custom settings
        nx.draw(kNNG, pos_knn, with_labels=False, node_size=100, edge_color='blue', width=1)

        # plt.title(f"k-NNG with k={k_nrst}")
        plt.show()  

    return D_knn, w_knn
    

#%%
### Create weighted graph with MST and KNN -----------**##
# Function to create a graph from the data matrix and weights.
def create_graph_MST_KNN(matrixData, weight_mat, k_nrst, print_G='y'):
    """
    Constructs a combined graph using both Minimum Spanning Tree (MST) and k-Nearest Neighbors Graph (kNNG).
    The MST is built using Euclidean distances, while kNNG is built using k-nearest neighbors.
    Edge weights are assigned from the given weight matrix.

    INPUT: 
        matrixData --- numpy array p by n
        weight_mat --- numpy array n by n
        k_nrst --- k-nearest neighbors parameter (int)
        print_G --- whether to print the graph (default: 'y')

    OUTPUT:
        D_union --- incidence matrix (np.array, m by n)
        weights_union --- array of weights on each edge (m-dimensional np.array)
    """
    
    p, n = matrixData.shape

    ### **Step 1: Compute MST**
    D_mst, w_mst = create_graph_MST(matrixData, weight_mat, print_G='n')

    ### **Step 2: Compute kNNG**
    D_knn, w_knn = create_graph_KNN(matrixData, weight_mat, k_nrst, print_G='n')

    ### **Step 3: Union of MST and kNNG**
    edges_mst = extract_edges(D_mst, w_mst)
    edges_knn = extract_edges(D_knn, w_knn)

    # Combine edges from both graphs while ensuring no duplication
    edge_dict = {}
    
    for u, v, weight in edges_mst:
        edge_dict[(u, v)] = weight  # Store MST edge
    
    for u, v, weight in edges_knn:
        if (u, v) in edge_dict:
            edge_dict[(u, v)] = max(edge_dict[(u, v)], weight)  # Keep max weight if duplicate
        else:
            edge_dict[(u, v)] = weight  # Store kNNG edge

    # Construct Union Incidence Matrix
    num_edges = len(edge_dict)
    D_union = np.zeros((num_edges, n))
    weights_union = np.zeros(num_edges)

    for idx, ((u, v), weight) in enumerate(edge_dict.items()):
        D_union[idx, u] = 1
        D_union[idx, v] = -1
        weights_union[idx] = weight
        
    # Step 4: Compute Fully Connected Graph (FCG) for Visualization
    dist_matrix = squareform(pdist(matrixData.T, metric='euclidean'))  # Compute Euclidean distances
    Gr = nx.Graph()
    for i in range(n):
        for j in range(i + 1, n):
            Gr.add_edge(i, j, weight=dist_matrix[i, j])  # Use Euclidean distance for FCG visualization


    ### **Step 5: Visualization**
    if print_G == 'y':
        pos_knn = {i: matrixData[:, i] for i in range(n)} if p == 2 else {i: matrixData[:2, i] for i in range(n)}

        # Create Graphs for Visualization
        # G_mst = nx.Graph()
        # G_mst.add_weighted_edges_from(edges_mst)

        # G_knn = nx.Graph()
        # G_knn.add_weighted_edges_from(edges_knn)
        
        
        
        G_union = nx.Graph()
        G_union.add_weighted_edges_from([(u, v, weight) for (u, v), weight in edge_dict.items()])

        plt.figure(figsize=(8, 6))
        
        # Draw Union Graph with Fully Connected Graph (FCG) (light gray)
        nx.draw(Gr, pos_knn, with_labels=False, node_size=100, node_color='lightgray', edge_color='lightgray', width=1)
        nx.draw(G_union, pos_knn, with_labels=False, node_size=100, edge_color = 'blue', width=1)

        # plt.title(f"Union of MST and k-NNG (k={k_nrst})")
        # plt.legend(["MST", "kNNG", "Union"])
        plt.show()

    return D_union, weights_union


#%% --------------------------------------------------------
# Generate DMSTs with t.

class UnionFind:
    """ Helper class to implement Union-Find with path compression. """
    def __init__(self, n):
        self.parent = list(range(n))
        self.rank = [0] * n
    
    def find(self, u):
        if self.parent[u] != u:
            self.parent[u] = self.find(self.parent[u])
        return self.parent[u]
    
    def union(self, u, v):
        root_u = self.find(u)
        root_v = self.find(v)
        if root_u != root_v:
            if self.rank[root_u] > self.rank[root_v]:
                self.parent[root_v] = root_u
            elif self.rank[root_u] < self.rank[root_v]:
                self.parent[root_u] = root_v
            else:
                self.parent[root_v] = root_u
                self.rank[root_u] += 1
            return True  # Successfully merged
        return False  # Already in the same component (cycle detected)

def build_DMST(X, W_mat, t=3, print_G='n'):
    """
    Construct Disjoint MSTs using Kruskal's algorithm without replacement and return the graph incidence matrix D.

    INPUT:
        X (np.ndarray): Data matrix (p, n), where p is the dimension, and n is the number of points.
        W_mat (np.ndarray): Weight matrix (n, n), where weights are assigned to edges.
        t (int): Number of MSTs to construct.
        print_G (str): 'y' or 'n' (default). Show the graph or not.

    OUTPUT:
        D (np.ndarray): Graph incidence matrix D (m, n), where m is the number of edges across t MSTs.
        weights_vec (np.ndarray): Weights of edges (m-dimensional).
    """

    p, n = X.shape

    # Step 1: Compute Pairwise Euclidean Distances (for graph structure)
    distances = squareform(pdist(X.T, metric='euclidean'))
    edges = [(i, j, distances[i, j]) for i in range(n) for j in range(i + 1, n)]
    
    # Step 2: Sort edges by increasing Euclidean distance
    edges.sort(key=lambda x: x[2])

    # Step 3: Build t MSTs using Kruskal's Algorithm without replacement
    used_edges = set()
    all_edges = []

    for tree_idx in range(t):
        uf = UnionFind(n)  # Reset Union-Find for each tree
        mst_edges = []

        for i, j, _ in edges:
            if (i, j) not in used_edges and uf.union(i, j):
                mst_edges.append((i, j))
                used_edges.add((i, j))  # Do not reuse edges for future MSTs
            if len(mst_edges) == n - 1:  # If MST has n-1 edges, stop
                break
        
        all_edges.extend(mst_edges)

    # Step 4: Construct the Graph Incidence Matrix D
    m = len(all_edges)  # Number of edges used across all MSTs
    D = np.zeros((m, n))

    for edge_idx, (i, j) in enumerate(all_edges):
        D[edge_idx, i] = 1
        D[edge_idx, j] = -1
    
    # Step 5: Assign Weights from W_mat (Consistent with D)
    weights_vec = np.array([W_mat[i, j] for i, j in all_edges])

    ### **Step 6: Visualization**
    if print_G == 'y':
        pos_knn = {i: X[:, i] for i in range(n)} if p == 2 else {i: X[:2, i] for i in range(n)}

        # Create Fully Connected Graph for Background (light gray)
        Gr = nx.Graph()
        Gr.add_weighted_edges_from([(i, j, distances[i, j]) for i in range(n) for j in range(i + 1, n)])

        # Create Disjoint MST Graph
        G_dmst = nx.Graph()
        G_dmst.add_weighted_edges_from([(i, j, W_mat[i, j]) for i, j in all_edges])

        plt.figure(figsize=(8, 6))

        # Draw Fully Connected Graph (light gray)
        nx.draw(Gr, pos_knn, with_labels=False, node_size=100, node_color='lightgray', edge_color='lightgray', width=1)

        # Draw Disjoint MSTs (blue)
        nx.draw(G_dmst, pos_knn, with_labels=False, node_size=100, edge_color='blue', width=1)

        # plt.title(f"DMSTs with t={t}")
        plt.show()  

    return D, weights_vec

#%% epsilon ball graph (eBG)

def build_eps_ball_graph(X, W_mat, epsilon=None, print_G='y'):
    """
    Constructs an epsilon-ball graph where each point connects to points within distance epsilon.
    The graph structure is based on Euclidean distances, but weights come from the given weight matrix.

    INPUT:
        X (np.ndarray): Data matrix (p, n), where p is the dimension, and n is the number of points.
        W_mat (np.ndarray): Weight matrix (n, n) with edge weights.
        epsilon (float): Threshold distance for connection. If None, set to the median of all pairwise distances.
        print_G (str): 'y' or 'n' (default). Show the graph or not.

    OUTPUT:
        D (np.ndarray): Graph incidence matrix D (m, n), where m is the number of edges.
        weights_vec (np.ndarray): Weights vector corresponding to the edges in D.
    """

    p, n = X.shape

    # Step 1: Compute Pairwise Euclidean Distances
    distances = squareform(pdist(X.T, metric='euclidean'))

    # Step 2: Set epsilon to median distance if not provided
    if epsilon is None:
        epsilon = np.median(distances[np.triu_indices(n, k=1)])

    # Step 3: Select edges where distance <= epsilon
    edge_indices = np.argwhere((distances <= epsilon) & (distances > 0))  # Exclude self-loops
    edges = [(i, j) for i, j in edge_indices if i < j]  # Ensure unique pairs

    # Step 4: Construct the Graph Incidence Matrix D
    m = len(edges)
    D = np.zeros((m, n))

    for edge_idx, (i, j) in enumerate(edges):
        D[edge_idx, i] = 1
        D[edge_idx, j] = -1

    # Step 5: Extract Weights from W_mat
    weights_vec = np.array([W_mat[i, j] for i, j in edges])

    ### **Step 6: Visualization**
    if print_G == 'y':
        pos = {i: X[:, i] for i in range(n)} if p == 2 else {i: X[:2, i] for i in range(n)}

        # Create Fully Connected Graph (FCG) for Background (light gray)
        Gr = nx.Graph()
        Gr.add_weighted_edges_from([(i, j, distances[i, j]) for i in range(n) for j in range(i + 1, n)])

        # Create Epsilon-Ball Graph
        G_eps = nx.Graph()
        G_eps.add_weighted_edges_from([(i, j, W_mat[i, j]) for i, j in edges])

        plt.figure(figsize=(8, 6))

        # Draw Fully Connected Graph (light gray)
        nx.draw(Gr, pos, with_labels=False, node_size=100, node_color='lightgray', edge_color='lightgray', width=1)

        # Draw Epsilon-Ball Graph (blue)
        nx.draw(G_eps, pos, with_labels=False, node_size=100, edge_color='blue', width=1)

        # plt.title(f"Epsilon-Ball Graph (ε={epsilon:.2f}) with Fully Connected Graph")
        plt.show()

    return D, weights_vec

#%% All connected graph.
from itertools import combinations
def create_graph_dense(matrixData, weight_mat, print_G='y'):
    """
    Constructs a fully connected (dense) graph where edges are based on Euclidean distances,
    but edge weights are assigned from the given weight matrix.

    Parameters:
    - matrixData (numpy.ndarray): p x n data matrix where each column is a data point.
    - weight_mat (numpy.ndarray): n x n matrix containing edge weights between nodes.
    - print_G (str, optional): Whether to visualize the graph (default: 'y').

    Returns:
    - D_dense (numpy.ndarray): Incidence matrix of the fully connected graph (edges x nodes).
    - w_dense (numpy.ndarray): Weights of the fully connected graph edges.
    """
    p, n = matrixData.shape

    # Step 1: Compute Pairwise Euclidean Distances (For Graph Structure)
    # distances = squareform(pdist(matrixData.T, metric='euclidean'))

    # Step 2: Generate All Possible Edges (Fully Connected Graph)
    edges = [(i, j) for i, j in combinations(range(n), 2)]  # Unique node pairs

    # Step 3: Construct Incidence Matrix D_dense
    m = len(edges)  # Number of edges in fully connected graph
    D_dense = np.zeros((m, n))

    for idx, (i, j) in enumerate(edges):
        D_dense[idx, i] = 1
        D_dense[idx, j] = -1

    # Step 4: Extract Weights from `weight_mat` (Consistent with `D_dense`)
    w_dense = np.array([weight_mat[i, j] for i, j in edges])

    ### **Step 5: Visualization**
    if print_G == 'y':
        pos = {i: matrixData[:, i] for i in range(n)} if p == 2 else {i: matrixData[:2, i] for i in range(n)}

        # Create Fully Connected Graph (FCG)
        Gr = nx.Graph()
        Gr.add_weighted_edges_from([(i, j, weight_mat[i, j]) for i, j in edges])

        plt.figure(figsize=(8, 6))

        # Draw Fully Connected Graph (blue)
        nx.draw(Gr, pos, with_labels=False, node_size=100, edge_color='blue', width=1, alpha=0.5)

        # plt.title("Fully Connected Graph (FCG)")
        plt.show()

    return D_dense, w_dense




#%% MST graph.
def create_graph_MST_labeled(matrixData, weight_mat, print_G='y'):
    """
    INPUT: 
        matrixData --- numpy array p by n
        weight_mat --- numpy array n by n
        print_G --- whether to print out the graphs (default: 'y')
        
    OUTPUT:
        D --- incidence matrix (np.array, m by n)
        weights_vec --- array of weights on each edge (m-dimensional np.array)
    """
    p, n = matrixData.shape

    # Compute pairwise Euclidean distance matrix
    dist_matrix = squareform(pdist(matrixData.T, metric='euclidean'))

    # Create a graph with Euclidean distances
    Gr = nx.Graph()
    for i in range(n):
        for j in range(i + 1, n):
            Gr.add_edge(i, j, weight=dist_matrix[i, j])

    # Compute the Minimum Spanning Tree (MST) using Euclidean distances
    mst = nx.minimum_spanning_tree(Gr)

    # Visualize MST if required
    if print_G == 'y':
        if p == 2:  # Direct 2D plot
            pos = {i: (matrixData[0, i], matrixData[1, i]) for i in range(n)}
            plt.figure(figsize=(8, 6))
            nx.draw(Gr, pos, with_labels=False, node_size=100, node_color='lightgray', edge_color='lightgray', width=1)
            nx.draw(mst, pos, with_labels=False, node_size=100, edge_color='blue', width=1)
            
            # Label edges with weight values
            edge_labels = {(u, v): f"{weight_mat[u, v]:.2f}" for u, v in mst.edges()}
            for (u, v), label in edge_labels.items():
                mid_x = (pos[u][0] + pos[v][0]) / 2
                mid_y = (pos[u][1] + pos[v][1]) / 2
                plt.text(mid_x, mid_y, label, fontsize=16, color='red', ha='center', va='center', 
                         bbox=dict(facecolor='white', alpha=0.7, edgecolor='gray', boxstyle='round,pad=0.2'))
    
            # plt.title('MST using Euclidean Distance')
            plt.grid(False)
            plt.show()

        elif p > 2:  # Project high-dimensional data onto 2D
            data_2d = matrixData[:2, :]  # Take the first two dimensions
            pos_2d = {i: (data_2d[0, i], data_2d[1, i]) for i in range(n)}
            plt.figure(figsize=(8, 6))
            nx.draw(Gr, pos_2d, with_labels=False, node_size=100, node_color='lightgray', edge_color='lightgray', width=1)
            nx.draw(mst, pos_2d, with_labels=False, node_size=100, edge_color='blue', width=1)
            # plt.title('MST (Projected to 2D)')
            plt.grid(False)
            plt.show()

    # Compute the incidence matrix and update weights from weight_mat
    n_edges_mst = len(mst.edges)
    D_mst = np.zeros((n_edges_mst, n))  # m x n incidence matrix
    w_mst = np.zeros(n_edges_mst)  # m-dim weight vector

    for idx, (u, v) in enumerate(mst.edges()):
        D_mst[idx, u] = 1
        D_mst[idx, v] = -1
        w_mst[idx] = weight_mat[u, v]  # Use weight from weight_mat instead of Euclidean distance

    return D_mst, w_mst