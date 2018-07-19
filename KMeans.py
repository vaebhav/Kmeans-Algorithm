#!/usr/local/bin/python3

import math
import numpy as np
import pandas as pd
from sklearn import preprocessing
import collections as cl
import seaborn as sns
import sys
import matplotlib.pyplot as plt
import matplotlib.animation as animation

f = "/Users/vaebhav/Documents/Python/Machine Learning/Kmeans/durudataset.txt"

test_df=pd.read_csv(f,sep='\t',header=None)

data_final_cols = test_df.columns.values.tolist()


def euclidian(a, b):
        return np.linalg.norm(np.array(a)-np.array(b))

def plot(dataset, history_centroids, Hash):
    colors = ['r', 'g']

    fig, ax = plt.subplots()

    for instance_index in Hash:
        for key_index in Hash[instance_index]:
            ax.plot(dataset.iloc[key_index].values[0],dataset.iloc[key_index].values[1],(colors[instance_index] + 'o'))

    history_points = []
    for index, centroids in enumerate(history_centroids):
        for inner, item in enumerate(centroids):
            if index == 0:
                history_points.append(ax.plot(item[0], item[1], 'bo')[0])
            else:
                history_points[inner].set_data(item[0], item[1])
                plt.pause(1.0)

def cluster_points(dataframe, prototypes):
    Cluster_hash = cl.defaultdict(list)
    for index_instance, instance in enumerate(dataframe.values):
        centroid_index = min([(i[0], euclidian(instance,prototypes[i[0]]))
                for i in enumerate(prototypes)], key=lambda t:t[1])[0]
        try:
            Cluster_hash[centroid_index].append(index_instance)
        except KeyError:
            Cluster_hash[centroid_index] = [index_instance]
    return Cluster_hash

def update_centroid(dataframe,cluster):
    new_centroid = []
    for key in cluster:
        new_avg = np.mean(dataframe.iloc[cluster[key]],axis=0)
        new_centroid.append(new_avg)
    return new_centroid

def Kmeans(k,dataframe=None,epsilon=0, distance='euclidian'):
    history_centroids = []
    if distance == 'euclidian':
        dist_method = euclidian
    #if dataframe == None:
    #    sys.exit()
    num_instances, num_features = dataframe.shape
    idx=np.random.randint(0, num_instances - 1, size=k)
    #print(idx)
    prototypes = dataframe.iloc[idx].values
    #print("ProtoType--->{0}".format(prototypes))
    history_centroids.append(prototypes)
    prototypes_old = np.zeros(prototypes.shape)
    norm = dist_method(prototypes, prototypes_old)
    iteration = 0

    while norm > epsilon:
        iteration += 1
        norm = dist_method(prototypes, prototypes_old)
        prototypes_old = prototypes
        Cluster_hash = cluster_points(dataframe,prototypes)
        prototypes = update_centroid(dataframe,Cluster_hash)
    history_centroids.append(prototypes)

    plot(dataframe,history_centroids,Cluster_hash)

Kmeans(2,test_df)
