from sys import flags

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pyexpat import features

np.random.seed(2)


def add_noise(data):
    """
    :param data: dataset as numpy array of shape (n, 2)
    :return: data + noise, where noise~N(0,0.00001^2)
    """
    noise = np.random.normal(loc=0, scale=1e-5, size=data.shape)
    return data + noise


def choose_initial_centroids(data, k):
    """
    :param data: dataset as numpy array of shape (n, 2)
    :param k: number of clusters
    :return: numpy array of k random items from dataset
    """
    n = data.shape[0]
    indices = np.random.choice(range(n), k, replace=False)
    return data[indices]


# ====================
def transform_data(df, features):
    """
    Performs the following transformations on df:
        - selecting relevant features
        - scaling
        - adding noise
    :param df: dataframe as was read from the original csv.
    :param features: list of 2 features from the dataframe
    :return: transformed data as numpy array of shape (n, 2)
    """
    df_copy = df[features].copy()
    for col in features: #
        min = df_copy[col].min()
        max = df_copy[col].max()
        df_copy[col] = (df_copy[col] - min) / (max - min)

    arr = df_copy.values
    arr_with_nois = add_noise(arr)
    return arr_with_nois



def kmeans(data, k):

    """
    Running kmeans clustering algorithm.
    :param data: numpy array of shape (n, 2)
    :param k: desired number of cluster
    :return:
    * labels - numpy array of size n, where each entry is the predicted label (cluster number)
    * centroids - numpy array of shape (k, 2), centroid for each cluster.
    """

    flag = True
    priv_centroid = choose_initial_centroids(data, k)
    while flag:
        labels = assign_to_clusters(data, priv_centroid)
        current_centroids_np = recompute_centroids(data, labels,k)
        if(np.array_equal(current_centroids_np, priv_centroid)):
            flag = False
        priv_centroid = current_centroids_np.copy()


    return labels, current_centroids_np

def visualize_results(data, labels, centroids, path):
    """
    Visualizing results of the kmeans model, and saving the figure.
    :param data: data as numpy array of shape (n, 2)
    :param labels: the final labels of kmeans, as numpy array of size n
    :param centroids: the final centroids of kmeans, as numpy array of shape (k, 2)
    :param path: path to save the figure to.
    """
    colors = np.array(['red','blue','green', 'yellow', 'cyan'])
    plt.figure(figsize = [8,8])
    plt.scatter(data[:, 0], data[:, 1], c = colors[labels])

    for i in range(len(centroids)):
        plt.scatter(centroids[i,0], centroids[i,1], color='white', edgecolors='black', marker='*', linewidth=2, s=222, alpha=0.85,  label=f'Centroid' if i==0 else None)
        
    plt.xlabel('cnt')
    plt.ylabel('t1')
    plt.title(f'Results for kmeans with k = {len(centroids)}')
    plt.legend()
    plt.savefig(path)
    plt.show()
    #plt.savefig(path)

def dist(x, y):
    """
    Euclidean distance between vectors x, y
    :param x: numpy array of size n
    :param y: numpy array of size n
    :return: the euclidean distance
    """
    return np.array(np.linalg.norm(x - y, axis=1))
    # return distance

def assign_to_clusters(data, centroids):
    """
    Assign each data point to a cluster based on current centroids
    :param data: data as numpy array of shape (n, 2)
    :param centroids: current centroids as numpy array of shape (k, 2)
    :return: numpy array of size n
    """
    labels = np.array([np.argmin(dist(x,centroids)) for x in data])
    return labels

def recompute_centroids(data, labels, k):
    """
    Recomputes new centroids based on the current assignment
    :param data: data as numpy array of shape (n, 2)
    :param labels: current assignments to clusters for each data point, as numpy array of size n
    :param k: number of clusters
    :return: numpy array of shape (k, 2)
    """
    labels_column = labels[:, np.newaxis]
    combined_data = np.hstack((data, labels_column))
    temp_df = pd.DataFrame(combined_data)
    current_centroids_df = temp_df.groupby(2).mean()
    current_centroids_np = current_centroids_df.values
    return current_centroids_np

