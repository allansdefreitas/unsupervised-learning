# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
#import skfuzzy
from sklearn.preprocessing import StandardScaler



def preprocess_dataset(dataframe):
    
    #pre-processing of dataset
    scaler = StandardScaler()
    data = scaler.fit_transform(dataframe.values)
    
    return data

    
"""
I. Considere os dados "Image Segmentation" do site uci machine learning
repository (https://archive.ics.uci.edu/ml/datasets/Image+Segmentation).

"""

PATH = 'https://raw.githubusercontent.com/allansdefreitas/unsupervised-learning/main/segmentation.data'
PATH2 = 'https://raw.githubusercontent.com/allansdefreitas/unsupervised-learning/main/segmentation.test'

dataset_original = pd.read_csv(PATH, sep=',').reset_index(drop=True)
dataset_original2 = pd.read_csv(PATH2, sep=',').reset_index(drop=True)

#concat datasets
frames = [dataset_original, dataset_original2]
dataset_original = pd.concat(frames)


""" Considere 3 datasets: """
#o primeiro considerando as variáveis 4 a 9 (shape) ----------

dataset_1 = dataset_original.iloc[:,3:9]
#pre-processing of dataset
X_dataset_1 = preprocess_dataset(dataset_1)


#o segundo considerando as variaveis 10 a 19 (rgb) ----------
dataset_2 = dataset_original.iloc[:,9:19]
#pre-processing of dataset
X_dataset_2 = preprocess_dataset(dataset_2)

#O terceiro considerando as variaveis 4 a 19 (shape + rgb) -------
dataset_3 = dataset_original.iloc[:,3:19]
#pre-processing of dataset
X_dataset_3 = preprocess_dataset(dataset_3)



""" Em cada dataset execute o algoritmo FCM com a distância de City-Block
50 vezes para obter uma partição fuzzy em 7 grupos e selecione o melhor
resultado segundo a função objetivo.


#PARA dataset_1_shape (features 4 a 9)

"""

def initialize_membership_matrix(n_samples, n_clusters):
    """
    Initializes the membership matrix for Fuzzy C-Means.

    Parameters:
        n_samples (int): Number of data points.
        n_clusters (int): Number of clusters.

    Returns:
        numpy.ndarray: Initial membership matrix.
    """
    membership_matrix = np.random.rand(n_samples, n_clusters)
    membership_matrix /= np.sum(membership_matrix, axis=1, keepdims=True)
    return membership_matrix


def update_membership_matrix(data, centroids, m, distance_metric):
    """
    Updates the membership matrix for Fuzzy C-Means.

    Parameters:
        data (numpy.ndarray): Input data points.
        centroids (numpy.ndarray): Current centroid positions.
        m (float): Fuzziness parameter.
        distance_metric (str): Distance metric to use ('cityblock' or 'euclidean').

    Returns:
        numpy.ndarray: Updated membership matrix.
    """
    n_samples, n_clusters = data.shape[0], centroids.shape[0]
    membership_matrix = np.zeros((n_samples, n_clusters))

    for i in range(n_samples):
        for j in range(n_clusters):
            if distance_metric == 'cityblock':
                dist = np.sum(np.abs(data[i] - centroids[j]))
            elif distance_metric == 'euclidean':
                dist = np.linalg.norm(data[i] - centroids[j])
            else:
                raise ValueError("Invalid distance metric.")

            membership_matrix[i, j] = 1 / np.sum((dist / np.abs(data[i] - centroids)) ** (2 / (m - 1)))

    membership_matrix /= np.sum(membership_matrix, axis=1, keepdims=True)
    return membership_matrix


def update_centroids(data, membership_matrix, m):
    """
    Updates the centroids for Fuzzy C-Means.

    Parameters:
        data (numpy.ndarray): Input data points.
        membership_matrix (numpy.ndarray): Current membership matrix.
        m (float): Fuzziness parameter.

    Returns:
        numpy.ndarray: Updated centroid positions.
    """
    n_clusters, n_features = membership_matrix.shape[1], data.shape[1]
    centroids = np.zeros((n_clusters, n_features))

    for j in range(n_clusters):
        membership_power = membership_matrix[:, j] ** m
        centroids[j] = np.sum(membership_power.reshape(-1, 1) * data, axis=0) / np.sum(membership_power)

    return centroids

def fuzzy_cmeans(data, n_clusters, m, distance_metric='cityblock', max_iter=100, tolerance=1e-4):
    """
    Fuzzy C-Means clustering algorithm.

    Parameters:
        data (numpy.ndarray): Input data points.
        n_clusters (int): Number of clusters.
        m (float): Fuzziness parameter (> 1).
        distance_metric (str): Distance metric to use ('cityblock' or 'euclidean').
        max_iter (int): Maximum number of iterations.
        tolerance (float): Convergence tolerance.

    Returns:
        numpy.ndarray: Final centroid positions.
        numpy.ndarray: Membership matrix.
        int: Number of iterations performed.
    """
    n_samples, n_features = data.shape
    membership_matrix = initialize_membership_matrix(n_samples, n_clusters)
    centroids = np.zeros((n_clusters, n_features))

    for iteration in range(max_iter):
        prev_centroids = centroids.copy()

        centroids = update_centroids(data, membership_matrix, m)
        membership_matrix = update_membership_matrix(data, centroids, m, distance_metric)

        if np.linalg.norm(centroids - prev_centroids) < tolerance:
            break

    return centroids, membership_matrix, iteration+1


def calculate_objective(data, centroids, membership_matrix, m, distance_metric):
    """
    Calculates the objective function value for Fuzzy C-Means.

    Parameters:
        data (numpy.ndarray): Input data points.
        centroids (numpy.ndarray): Current centroid positions.
        membership_matrix (numpy.ndarray): Current membership matrix.
        m (float): Fuzziness parameter.
        distance_metric (str): Distance metric to use ('cityblock' or 'euclidean').

    Returns:
        float: Objective function value.
    """
    objective = 0
    n_samples, n_clusters = data.shape[0], centroids.shape[0]

    for i in range(n_samples):
        for j in range(n_clusters):
            if distance_metric == 'cityblock':
                dist = np.sum(np.abs(data[i] - centroids[j]))
            elif distance_metric == 'euclidean':
                dist = np.linalg.norm(data[i] - centroids[j])
            else:
                raise ValueError("Invalid distance metric.")

            objective += (membership_matrix[i, j] ** m) * (dist ** 2)

    return objective

#Modified partition coefficient e partition entropy --------------################

#close to 1 values are better
def calculate_mpc(membership_matrix):
    """
    Calculates the Modified Partition Coefficient (MPC) for Fuzzy C-Means clustering.

    Parameters:
        membership_matrix (numpy.ndarray): Membership matrix of shape (n_samples, n_clusters).

    Returns:
        float: Modified Partition Coefficient value.
    """
    max_memberships = np.max(membership_matrix, axis=1)
    sum_memberships = np.sum(membership_matrix, axis=1)

    mpc = np.mean(max_memberships / sum_memberships)

    return mpc

#close to 0 values are better
def calculate_partition_entropy(membership_matrix):
    """
    Calculates the Partition Entropy for Fuzzy C-Means clustering.

    Parameters:
        membership_matrix (numpy.ndarray): Membership matrix of shape (n_samples, n_clusters).

    Returns:
        float: Partition Entropy value.
    """
    n_samples, n_clusters = membership_matrix.shape
    entropy = 0.0

    for i in range(n_samples):
        for j in range(n_clusters):
            if membership_matrix[i, j] > 0:
                entropy -= membership_matrix[i, j] * np.log2(membership_matrix[i, j])

    partition_entropy = entropy / n_samples

    return partition_entropy


"""## Em cada dataset execute o algoritmo FCM com a distância de City-Block 50 vezes para obter 
uma partição fuzzy em 7 grupos e selecione o melhor resultado segundo a função objetivo."""
def get_best_partition(data, n_clusters, m, distance_metric = 'cityblock', times_to_run=50):
    
    best_objective_value = 99999999999.9
    best_results = 0
    TIMES = times_to_run
    
    for i in range(TIMES):
    
      # Example usage
      #print("FCM: ", i + 1)
    
      centroids, membership_matrix, iterations = fuzzy_cmeans(data, n_clusters, m, distance_metric)
      objective_value = calculate_objective(data, centroids, membership_matrix, m, distance_metric)
      
      
      if(objective_value < best_objective_value):
        best_objective_value = objective_value
        best_results = centroids, membership_matrix, iterations
    
        #print("Centroids:")
        #print(centroids)
        #print("Membership matrix:")
        #print(membership_matrix)
        print("Objective value:", objective_value)
        #print("Iterations:", iterations, "\n")
    
    return best_results




data = X_dataset_1
n_clusters = 7
m = 2
distance_metric = 'cityblock'
best_objective_value = 99999999999.9
best_results = 0
times = 1 #50


datasets = [X_dataset_1, X_dataset_2, X_dataset_3]
best_results = 0


for dataset_i in datasets:
 
   best_results = get_best_partition(dataset_i, n_clusters, m, distance_metric = 'cityblock', times_to_run=times)
  



print(best_results[0])



"""Para cada dataset e partição fuzzy, calcule o Modified partition coefficient
e o Partition entropy. Comente."""

"""
NUM_DATASETS = 3

mpc_datasets = []
partition_entropy_datasets = []


for i in range(NUM_DATASETS):
    mpc = calculate_mpc(best_membership_matrix)
    
    
    

mpc_dataset_1 = 0
mpc_dataset_2 = 0
mpc_dataset_3 = 0

partition_entropy_dataset_1 = 0
partition_entropy_dataset_2 = 0
partition_entropy_dataset_3 = 0

    

calculate_mpc(best_membership_matrix)

calculate_partition_entropy(best_membership_matrix)

"""

""" 'TRASH' CODE ----------------------------------------------------------------####################"""

"""# ALGORTIMO FCM via scikit-fuzzy
!pip install -U scikit-fuzzy

import numpy as np
import matplotlib.pyplot as plt
import skfuzzy
from sklearn.preprocessing import StandardScaler

X_dataset_1_shape = dataset_1_shape.values

X_dataset_1_shape, X_dataset_1_shape.shape

scaler = StandardScaler()
X_dataset_1_shape = scaler.fit_transform(X_dataset_1_shape)
X_dataset_1_shape

R_dataset_1_shape = skfuzzy.cmeans(data = X_dataset_1_shape.T, c = 7, m = 2, error=0.005, maxiter= 1000, init=None)
R_dataset_1_shape

type(R_dataset_1_shape)

#matrix of probabilities
U = R_dataset_1_shape[1]
U

U.shape

U[6][0]

sum = 0
for i in range(7):
  sum +=  U[i][0]

print(sum)

#to which group each i-th element belongs
predictions = U.argmax(0)
predictions

type(X_dataset_1_shape), X_dataset_1_shape.shape

X_dataset_1_shape

X = X_dataset_1_shape
preds = predictions

#let's see
colors = ['blue', 'orange', 'green', 'red', 'brown', 'pink', 'gray']
for i in range(7):
  print(i)
  plt.scatter(X[preds == i, 0], X[preds == i, 1], s=100, c = colors[i], label='Cluster ' + str(i + 1))

R_dataset_1_shape = np.asarray(R_dataset_1_shape)
R_dataset_1_shape.shape, type(R_dataset_1_shape)

R_dataset_1_shape[0][6][0]

sum = 0
for i in range(5):
  sum = R_dataset_1_shape[0][i][0]

print(sum) """
