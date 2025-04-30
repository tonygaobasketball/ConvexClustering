#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Generate clustering data sets


"""
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
from scipy.spatial import distance_matrix
from sklearn.decomposition import PCA
import funcs
import scipy as sp
import diff_map_funcs
import random
import time

def stratified_sampling_preserve_stats(X, y, sample_fraction=0.3):
    """
    Perform stratified sampling while preserving the mean and variance of each class.

    Parameters:
        X (np.ndarray): Feature matrix (n_samples, n_features)
        y (np.ndarray): Class labels (n_samples,)
        sample_fraction (float): Fraction of data to sample from each class

    Returns:
        X_sampled (np.ndarray): Sampled feature matrix
        y_sampled (np.ndarray): Sampled class labels
    """
    unique_classes = np.unique(y)
    X_sampled_list = []
    y_sampled_list = []

    for cls in unique_classes:
        # Extract data for this class
        X_cls = X[y == cls]
        
        # Sample proportionally while maintaining mean & variance
        sample_size = max(1, int(len(X_cls) * sample_fraction))  # Ensure at least 1 sample
        X_cls_sampled, _ = train_test_split(X_cls, train_size=sample_size, random_state=42)

        # Check statistics
        print(f"Class {cls}: Original Mean = {X_cls.mean(axis=0)}, Sampled Mean = {X_cls_sampled.mean(axis=0)}")
        print(f"Class {cls}: Original Variance = {X_cls.var(axis=0)}, Sampled Variance = {X_cls_sampled.var(axis=0)}\n")

        X_sampled_list.append(X_cls_sampled)
        y_sampled_list.append(np.full(sample_size, cls))

    # Combine sampled data
    X_sampled = np.vstack(X_sampled_list)
    y_sampled = np.concatenate(y_sampled_list)

    return X_sampled.T, y_sampled

#%%%% 1. Synthetic data

### Generate Gaussian clusters graph.

# Generate Gaussian clusters.
"""
# Parameters
K = 5                 # number of clusters
p = 10                 # dimension
n_in_cluster = 5 * 2  # number of data points in each cluster
n = K * n_in_cluster  # total number of data points
rand_seed = 4

# k_nrst = n_in_cluster - 1
# k_nrst = n - 1
k_nrst = 5


# Generate the data
###--------------------
scale_para = 0.05
# GM_data, labels, means, covariances = funcs.generate_gaussian_clusters(K, p, n_in_cluster, random_seed=rand_seed)
GM_data, labels, means, covariances, GTmeans = funcs.generate_gaussian_clusters_v2(K, p, n_in_cluster, random_seed=rand_seed)
matrixData = GM_data.T

dataX = matrixData
y_true = labels
true_cls_centers = means

"""
#####------------------#------------------

#####------------------#------------------
# Half moons (synthetic)
"""
#------# OPTION 1: Use existing data set .mat file.
# Generate Gaussian clusters.
# Load the data from the 2half_moons.mat file

data = sio.loadmat("2half_moons.mat")['data']
V1 = np.vstack(data['V1'][0][0])  # Feature matrix
V2 = np.vstack(data['V2'][0][0])  # Additional feature matrix
V3 = np.vstack(data['V3'][0][0])  # Label or categorical data

# Combine into a single NumPy matrix
dataX = np.hstack((V1, V2)).T  # Stack features and labels
y_true = V3.flatten()



#------# OPTION 2: Generate new set
# # Generate dataset
# dataX = funcs.generate_2half_moons(random_seed = 1)

# # Record the dimension of the data.
# p, n =  dataX.shape

"""
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
#####------------------#------------------

#%% 2. Benchmark data (with true labels)
import pandas as pd
import re
from sklearn.preprocessing import normalize

"""
dataX: p by n np array
y_true: n-dim np array
"""

#####------------------#------------------
# 1) TCGA breast cancer data.
"""
# Step 1: Import the CSV file
file_path = 'Datasets/data/tcga_breast.csv'  # Replace with your file's path
raw_data = pd.read_csv(file_path)

# Step 2: Check the data structure
print(raw_data.head())

# Step 3: Convert group names to numerical labels

# Option 1: Using pandas.Categorical
raw_data['true_label'] = pd.Categorical(raw_data.iloc[:,0]).codes

# Step 4: Use the labels for your data
dataX = np.array(raw_data.iloc[:, 1:-1]).T   # p by n array.
# # row normalization. (Keep row norm = 1)
dataX = normalize(dataX, norm='l2', axis=1)
y_true = np.array(raw_data['true_label'])
"""


# PCA applied to data (run the original data section.)
"""
pca = PCA(n_components=.90)  # Change the number of components as needed
x_pca = pca.fit_transform(dataX.T)
Trans_mat = pca.components_.T

dataX = x_pca.T
"""
#####------------------#------------------

#####------------------#------------------
# 2) Lung 500
"""
# Step 1: Import the CSV file
file_path = 'Datasets/data/lung500.csv'  # Replace with your file's path
raw_data = pd.read_csv(file_path)

# Step 2: Check the data structure
print(raw_data.head())

# Step 3: Convert group names to numerical labels
# Extract the letters part using regex
raw_data['Extracted_Letters'] = raw_data.iloc[:,0].apply(lambda x: re.search(r'[a-z_]+$', x).group())
# Map unique letters to numerical labels
letter_to_label = {letter: idx for idx, letter in enumerate(raw_data['Extracted_Letters'].unique())}
raw_data['true_label'] = raw_data['Extracted_Letters'].map(letter_to_label)
# Display the result
print(raw_data)

# Step 4: Use the labels for your data
dataX = np.array(raw_data.iloc[:, 1:-2]).T   # p by n array.
# row normalization. (Keep row norm = 1)
dataX = normalize(dataX, norm='l2', axis=1)
y_true = np.array(raw_data['true_label'])
"""
#####------------------#------------------

#####------------------#------------------
# Lung 100
"""
# Step 1: Import the CSV file
file_path = 'Datasets/data/lung100.csv'  # Replace with your file's path
raw_data = pd.read_csv(file_path)

# Step 2: Check the data structure
print(raw_data.head())

# Step 3: Convert group names to numerical labels
# Extract the letters part using regex
raw_data['Extracted_Letters'] = raw_data.iloc[:,0].apply(lambda x: re.search(r'[a-z_]+$', x).group())
# Map unique letters to numerical labels
letter_to_label = {letter: idx for idx, letter in enumerate(raw_data['Extracted_Letters'].unique())}
raw_data['true_label'] = raw_data['Extracted_Letters'].map(letter_to_label)
# Display the result
print(raw_data)

# Step 4: Use the labels for your data
dataX = np.array(raw_data.iloc[:, 1:-2]).T   # p by n array.
# row normalization. (Keep row norm = 1)
dataX = normalize(dataX, norm='l2', axis=1)
y_true = np.array(raw_data['true_label'])
"""
#####------------------#------------------



#####------------------#------------------
# 3) Iris data for clustering.
"""
# Step 1: Import the CSV file
from sklearn.datasets import load_iris
import pandas as pd

# Load Iris dataset
iris = load_iris()

# Convert to pandas DataFrame
iris_df = pd.DataFrame(data=iris.data, columns=iris.feature_names)
iris_df['species'] = iris.target

# Map target integers to species names
iris_df['species'] = iris_df['species'].map({0: 'setosa', 1: 'versicolor', 2: 'virginica'})

# Step 2: Check the data structure
print(iris_df.head())

# Step 3: Convert group names to numerical labels
# Extract the letters part using regex
iris_df['true_label'] = pd.Categorical(iris_df['species']).codes

# Step 4: Use the labels for your data
dataX = np.array(iris_df.iloc[:, :-2]).T   # p by n array.
# row normalization. (Keep row norm = 1)
dataX = normalize(dataX, norm='l2', axis=1)
y_true = np.array(iris_df['true_label'])

##### Notice: duplicated columns detected. Delete one.
dataX = np.delete(dataX, 101, axis = 1)
y_true = np.delete(y_true, 101)
"""

#####------------------#------------------
# 4) AD data.
"""
# Step 1: Import the CSV file
file_path = 'Datasets/AD/data_numeric.csv'  # Replace with your file's path
raw_data = pd.read_csv(file_path)

# Step 2: Check the data structure
print(raw_data.head())

# Step 3: Use the labels for your data
dataX = np.array(raw_data.iloc[:, 1:-1]).T   # p by n array.
# row normalization. (Keep row norm = 1)
dataX = normalize(dataX, norm='l2', axis=1)
y_true = np.array(raw_data['Class'])
"""


# PCA applied to AD data (run the original data section.)
"""
pca = PCA(n_components=.90)  # Change the number of components as needed
x_pca = pca.fit_transform(dataX.T)
Trans_mat = pca.components_.T

dataX = x_pca.T
"""
#####------------------#------------------

#####------------------#------------------
# 5) Wholesale Customer
"""
file_path = 'Datasets/WholesaleCustomer_scaled.csv'  # Replace with your file's path
raw_data = pd.read_csv(file_path)

# Step 2: Check the data structure
print(raw_data.head())

# Step 3: Use the labels for your data
dataX = np.array(raw_data.iloc[:, :-1]).T   # p by n array.
# row normalization. (Keep row norm = 1)
dataX = normalize(dataX, norm='l2', axis=1)
y_true = (np.array(raw_data['Class']) + 1)/2
"""
#####------------------#------------------
# 6) UCI: HCV data
"""
file_path = 'Datasets/HCV_UCI/hcv_pro.csv'  # Replace with your file's path
raw_data = pd.read_csv(file_path)

# Step 2: Check the data structure
print(raw_data.head())

# Step 3: Use the labels for your data
dataX = np.array(raw_data.iloc[:, :-1]).T   # p by n array.
# row normalization. (Keep row norm = 1)
dataX = normalize(dataX, norm='l2', axis=1)
y_true = np.array(raw_data['Class'])
"""
#####------------------#------------------
# 7) UCI: Libras Movement data [Wang et al. 2018, Sparse convex clustering]
"""
# File path
file_path = "Datasets/libras+movement/movement_libras.data"  # Replace with the actual path if not in the current directory
# Read the file into a DataFrame
df = pd.read_csv(file_path, header=None)  # No header in the file

# Define feature names
feature_names = [
    f"{i//2 + 1} coordinate {'abcissa' if i % 2 == 0 else 'ordinate'}"
    for i in range(df.shape[1]-1)
] + ["Class"]

# Create a DataFrame
df.columns = feature_names

# Use the labels for your data
dataX = np.array(df.iloc[:, :-1]).T   # p by n array.
# row normalization. (Keep row norm = 1)
dataX = normalize(dataX, norm='l2', axis=1)

# Remove duplicate columns
_, unique_indices = np.unique(dataX, axis=1, return_index=True)
# Get the sorted indices for consistent order
unique_indices = np.sort(unique_indices)
# Get the filtered array with unique columns
dataX_unique = dataX[:, unique_indices]
# Find the removed column indices
all_indices = set(range(dataX.shape[1]))
removed_indices = list(all_indices - set(unique_indices))

y_true = np.array(df['Class'])
y_true_unique = np.delete(y_true, removed_indices)

dataX = dataX_unique
y_true = y_true_unique
"""
#####------------------#------------------

#####------------------#------------------
# 8) ds1_brca_tumor data (original))
"""
file_path = "Datasets/ds1_brca_tumor/data_combination.csv"  # Replace with the actual path if not in the current directory
# Read the file into a DataFrame
df = pd.read_csv(file_path)  # No header in the file
df_cleaned = df.dropna(subset = ['subtype_BRCA_Subtype_PAM50'])
df_X = df_cleaned.drop(columns = ['sample', 'time', 'days_to_last_follow_up', 'vital_status', 
                          'subtype_BRCA_Subtype_PAM50', 'status'])
dataX = np.array(df_X).T
dataX = normalize(dataX, norm = 'l2', axis = 1)
y_true_time = np.array(df_cleaned['time'])
y_true_status = np.array(df_cleaned['status'])
y_true_subtype = np.array(pd.Categorical(df_cleaned['subtype_BRCA_Subtype_PAM50']).codes)
###------ 

# smaller data set
file_path = "Datasets/ds1_brca_tumor/data_combination.csv"  # Replace with the actual path if not in the current directory
# Read the file into a DataFrame
df = pd.read_csv(file_path)  # No header in the file
df_cleaned = df.dropna(subset = ['subtype_BRCA_Subtype_PAM50'])
df_cleaned_small = df_cleaned.groupby('subtype_BRCA_Subtype_PAM50', group_keys=False).apply(
    lambda x: x.sample(frac=0.1, random_state=42)
    )
df_X = df_cleaned_small.drop(columns = ['sample', 'time', 'days_to_last_follow_up', 'vital_status', 
                          'subtype_BRCA_Subtype_PAM50', 'status'])


dataX = np.array(df_X).T
dataX = normalize(dataX, norm = 'l2', axis = 1)



# Create labels.
y_true_time = np.array(df_cleaned_small['time'])
y_true_status = np.array(df_cleaned_small['status'])
y_true_subtype = np.array(pd.Categorical(df_cleaned_small['subtype_BRCA_Subtype_PAM50']).codes)

y_true = y_true_status
# y_true = y_true_subtype
"""
#####------------------#------------------

#####------------------#------------------
# PCA applied to ds1_brca_tumor data (run the original data section.)
"""
pca = PCA(n_components=.90)  # Change the number of components as needed
x_pca = pca.fit_transform(dataX.T)
Trans_mat = pca.components_.T

dataX = x_pca.T
"""
#####------------------#------------------

#####------------------#------------------
# 9) Mammals data (Chi and Lange, 2015)
"""
# Load the data from the .mat file
import scipy.io as sio
data = sio.loadmat("mammals.mat")['data']

# Initialize an empty list to store the data
matrixData = []

# Extract field names
variables = data.dtype.names
mammals_names = data[0,0][0][:,0]
mammals_names_list = [mammals_names[j][0] for j in range(mammals_names.shape[0])]
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

dataX = matrixData
y_true = np.array(range(matrixData.shape[1]))
"""
#####------------------#------------------

#####------------------#------------------
# AD toy 1: 
"""
# File path
file_path = "Datasets/AD_toy1/AD_toy1.csv"  # Replace with the actual path if not in the current directory
# Read the file into a DataFrame
df = pd.read_csv(file_path, header=None)  # No header in the file


# Use the labels for your data
dataX = np.array(df.iloc[1:, 1:]).T.astype('float')   # p by n array.
# row normalization. (Keep row norm = 1)
dataX = normalize(dataX, norm='l2', axis=1)

# # Remove duplicate columns
# _, unique_indices = np.unique(dataX, axis=1, return_index=True)
# # Get the sorted indices for consistent order
# unique_indices = np.sort(unique_indices)
# # Get the filtered array with unique columns
# dataX_unique = dataX[:, unique_indices]
# # Find the removed column indices
# all_indices = set(range(dataX.shape[1]))
# removed_indices = list(all_indices - set(unique_indices))

y_true = np.array(df.iloc[1:, 0]).astype('float')
# y_true_unique = np.delete(y_true, removed_indices)

# dataX = dataX_unique
# y_true = y_true_unique
"""
#####------------------#------------------


#####------------------#------------------
# AD toy 2: 
"""
# File path
file_path = "Datasets/AD_toy2/AD_toy2.csv"  # Replace with the actual path if not in the current directory
# Read the file into a DataFrame
df = pd.read_csv(file_path)  # No header in the file


# Use the labels for your data
dataX = np.array(df.iloc[:, :-1]).T   # p by n array.
# row normalization. (Keep row norm = 1)
dataX = normalize(dataX, norm='l2', axis=1)

# # Remove duplicate columns
# _, unique_indices = np.unique(dataX, axis=1, return_index=True)
# # Get the sorted indices for consistent order
# unique_indices = np.sort(unique_indices)
# # Get the filtered array with unique columns
# dataX_unique = dataX[:, unique_indices]
# # Find the removed column indices
# all_indices = set(range(dataX.shape[1]))
# removed_indices = list(all_indices - set(unique_indices))

y_true = (np.array(df.iloc[:, -1]) + 1) / 2
# y_true_unique = np.delete(y_true, removed_indices)

# dataX = dataX_unique
# y_true = y_true_unique
"""
#####------------------#------------------


#####------------------#------------------
 # lymphography
"""
# File path
file_path = "Datasets/lymphography/data.csv"  # Replace with the actual path if not in the current directory
# Read the file into a DataFrame
df = pd.read_csv(file_path)  # No header in the file
df = df.drop_duplicates()

# p = 18, n = 148
# Use the labels for your data
dataX = np.array(df.iloc[:, 1:]).T   # p by n array.
# row normalization. (Keep row norm = 1)
dataX = normalize(dataX, norm='l2', axis=1)

y_true = np.array(df['Class']) # n array.
"""
#####------------------#------------------


#####------------------#------------------
# wine
"""
# File path
file_path = "Datasets/wine/wine.txt"  # Replace with the actual path if not in the current directory
# Read the file into a DataFrame
df = pd.read_csv(file_path, header=None)  # No header in the file
df = df.drop_duplicates()

# p = 13, n = 178
# Use the labels for your data
dataX = np.array(df.iloc[:, 1:]).T   # p by n array.
# row normalization. (Keep row norm = 1)
dataX = normalize(dataX, norm='l2', axis=1)

y_true = np.array(df.iloc[:, 0]) # n array.
"""
#####------------------#------------------

#####------------------#------------------
# land mines
"""
# File path
file_path = "Datasets/land_mines/land_mines.csv"  # Replace with the actual path if not in the current directory
# Read the file into a DataFrame
df = pd.read_csv(file_path)  # No header in the file
df = df.drop_duplicates()

# p = 13, n = 178
# Use the labels for your data
dataX = np.array(df.iloc[:, :-1]).T   # p by n array.
# row normalization. (Keep row norm = 1)
dataX = normalize(dataX, norm='l2', axis=1)

y_true = np.array(df.iloc[:, -1]) # n array.
"""
#####------------------#------------------

#####------------------#------------------
# 
# """

# """
#####------------------#------------------


#####------------------#------------------
# Hayes-Roth
"""
# File path
file_path = "Datasets/Hayes-Roth/data.txt"  # Replace with the actual path if not in the current directory
# Read the file into a DataFrame
df = pd.read_csv(file_path, header=None)  # No header in the file
df = df.drop_duplicates()

# p = 5, n = 132
# Use the labels for your data
dataX = np.array(df.iloc[:, :-1]).T   # p by n array.
# row normalization. (Keep row norm = 1)
dataX = normalize(dataX, norm='l2', axis=1)

# last column: class clumn [1, 2, 3]
y_true = np.array(df.iloc[:, -1]) # n array.
"""
#####------------------#------------------


#####------------------#------------------
# ecolidata
"""
file_path = "Datasets/Ecoli/ecolidata.txt"  # Replace with the actual path if not in the current directory
# Read the file into a DataFrame
df = pd.read_csv(file_path, delimiter=r"\s+", header=None, engine="python")
# Attribute Information 1. Sequence Name: Accession number for the SWISS-PROT database
df = df.iloc[:, 1:]

# transfer 8 class strings into numbers 1 to 8
class_col = df.columns[-1]
class_mapping = {label: i+1 for i, label in enumerate(df[class_col].unique())}
df[class_col] = df[class_col].map(class_mapping)

df = df.drop_duplicates()

# p = 7, n = 336
# Use the labels for your data
dataX = np.array(df.iloc[:, :-1]).T   # p by n array.
# row normalization. (Keep row norm = 1)
dataX = normalize(dataX, norm='l2', axis=1)

# last column: class clumn [1, 2, 3, 4, 5, 6, 7, 8]
y_true = np.array(df.iloc[:, -1]) # n array.
"""
#####------------------#------------------

#####------------------#------------------
# glass
"""
# File path
file_path = "Datasets/glass/glass.txt"  # Replace with the actual path if not in the current directory
# Read the file into a DataFrame
df = pd.read_csv(file_path, header=None)  # No header in the file
# Remove Column 1. ID
df = df.iloc[:, 1:]
df = df.drop_duplicates()

# p = 9, n = 214-1
# Use the labels for your data
dataX = np.array(df.iloc[:, :-1]).T   # p by n array.
# row normalization. (Keep row norm = 1)
dataX = normalize(dataX, norm='l2', axis=1)

# last column: class clumn [1, 2, 3, 4, 5, 6, 7]
y_true = np.array(df.iloc[:, -1]) # n array.
"""
#####------------------#------------------


# dermatology_data
#####------------------#------------------
"""
# File path
file_path = "Datasets/dermatology/dermatology_data.data"  # Replace with the actual path if not in the current directory
# Read the file into a DataFrame
df = pd.read_csv(file_path, header=None)  # No header in the file
df = df.drop_duplicates()

# p = 34, n = 366
# Missing Attribute Values: 8 (in Age attribute). Distinguished with '?'.
df = df.replace('?', np.nan)
df = df.dropna()
# OR replace by median of age? age is the 34-th attribute
# df = df.iloc[:, 33].fillna(df.iloc[:, 33].median(), inplace=True)


# p = 34, n = 358
# Use the labels for your data
dataX = np.array(df.iloc[:, :-1]).T   # p by n array.
# row normalization. (Keep row norm = 1)
dataX = normalize(dataX, norm='l2', axis=1)

# last column: class clumn [1, 2, 3, 4, 5, 6]
y_true = np.array(df.iloc[:, -1]) # n array.
"""
#####------------------#------------------


#####------------------#------------------
# careval
"""
file_path = "Datasets/careval/careval.csv"
df = pd.read_csv(file_path)
df = df.drop_duplicates()
# p = 6, n = 1728

dataX = np.array(df.iloc[:, :-1]).T   # p by n array.
# row normalization. (Keep row norm = 1)
dataX = normalize(dataX, norm='l2', axis=1)

# nn = dataX.shape[1]
# data_normalized = np.zeros(dataX.shape)
# for i in range(nn-1):
#     data_normalized[:,i] = (
#         dataX[:,i] - min(dataX[:,i]))/(max(dataX[:,i]) - min(dataX[:,i]))
# data_normalized[:,nn-1] = dataX[:,nn-1]-2

# class = [0, 1, 2, 3]
y_true = np.array(df['Class']) # n array.
"""
#####------------------#------------------

#####------------------#------------------
# BreastTissue
"""
# File path
file_path = "Datasets/BreastTissue/BT_3cls_normalized.csv"  # Replace with the actual path if not in the current directory
# Read the file into a DataFrame
df = pd.read_csv(file_path)
df = df.drop_duplicates()

# p = 9, n = 84-1
# Use the labels for your data
dataX = np.array(df.iloc[:, :-1]).T   # p by n array.
# row normalization. (Keep row norm = 1)
dataX = normalize(dataX, norm='l2', axis=1)

# class = [-1, 0, 1]
y_true = np.array(df['Class']) # n array.
"""
#####------------------#------------------

#####------------------#------------------
# Mall Customers (Kaggle)
"""
# File path
file_path = "Datasets/MallCustomers/Mall_customers.csv"  # Replace with the actual path if not in the current directory
# Read the file into a DataFrame
df = pd.read_csv(file_path)
df = df.drop_duplicates()
df['Class'] = pd.Categorical(df['Gender']).codes
# p = 9, n = 84-1
# Use the labels for your data
dataX = np.array(df.iloc[:, 2:-1]).T   # p by n array.
# row normalization. (Keep row norm = 1)
dataX = normalize(dataX, norm='l2', axis=1)

# class = [-1, 0, 1]
y_true = np.array(df['Class']) # n array.
"""
#####------------------#------------------


#%% mammals_milk data
"""
# File path
file_path = "Datasets/hartigan/file47_mammals_milk.csv"  # Replace with the actual path if not in the current directory
# Read the file into a DataFrame
df = pd.read_csv(file_path)

dataX = np.array(df.iloc[:, 1:]).T   # p by n array.
# row normalization. (Keep row norm = 1)
dataX = normalize(dataX, norm='l2', axis=1)



habitat_list = [
    -1,  # Horse (Land)
    -1,  # Orangutan (Land)
    -1,  # Monkey (Land)
    -1,  # Donkey (Land)
    0,  # Hippo (Land, but semi-aquatic)
    -1,  # Camel (Land)
    -1,  # Bison (Land)
    -1,  # Buffalo (Land)
    -1,  # Guinea Pig (Land)
    -1,  # Cat (Land)
    -1,  # Fox (Land)
    -1,  # Llama (Land)
    -1,  # Mule (Land)
    -1,  # Pig (Land)
    -1,  # Zebra (Land)
    -1,  # Sheep (Land)
    -1,  # Dog (Land)
    -1,  # Elephant (Land)
    -1,  # Rabbit (Land)
    -1,  # Rat (Land)
    -1,  # Deer (Land)
    -1,  # Reindeer (Land)
    1,   # Whale (Water)
    1,   # Seal (Water)
    1    # Dolphin (Water)
]

y_true_habitat = np.array(habitat_list) + 1
y_true = y_true_habitat

diet_list = [
    0,  # Horse (Herbivore)
    1,  # Orangutan (Omnivore) "红毛猩猩"
    1,  # Monkey (Omnivore)
    0,  # Donkey (Herbivore)
    0,  # Hippo (Herbivore)
    0,  # Camel (Herbivore)
    0,  # Bison (Herbivore)
    0,  # Buffalo (Herbivore)
    0,  # Guinea Pig (Herbivore)  "豚鼠"
    -1, # Cat (Carnivore)
    -1, # Fox (Carnivore)
    0,  # Llama (Herbivore)  "羊驼"
    0,  # Mule (Herbivore)   "骡子"
    1,  # Pig (Omnivore)
    0,  # Zebra (Herbivore)
    0,  # Sheep (Herbivore)
    -1, # Dog (Carnivore, but adaptable)
    0,  # Elephant (Herbivore)
    0,  # Rabbit (Herbivore)
    1,  # Rat (Omnivore)
    0,  # Deer (Herbivore)
    0,  # Reindeer (Herbivore)
    -1, # Whale (Mostly Carnivore, eats krill/fish)
    -1, # Seal (Carnivore)
    -1  # Dolphin (Carnivore)
]

y_true_diet = np.array(diet_list) + 1
y_true = y_true_diet
"""


