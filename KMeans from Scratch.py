# import libraries
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import pandas as pd
import random
from sklearn import datasets
np.seterr(divide='ignore', invalid='ignore')


# Dataset
iris = datasets.load_iris()
data = iris.data[:, :4]
number_of_rows = data.shape[0]      # number_of_rows = 150
number_of_features = data.shape[1]  # number_of_features = 4
total_distance_to_centroids_list = []

# Function to calculate 4-Dimension Euclidean Distance
def Euclidean_Distance_4d (a, b):
    distance = ((a[0] - b[0])**2 + (a[1] - b[1])**2 + (a[2] - b[2])**2 + (a[3] - b[3])**2)**0.5
    return distance

# Function to calculate Total Distance to Centroids
def total_distance_to_centroids_cal(number_of_clusters, number_of_iterations):

    # Initialization

    # Random pick data points as centroids initially
    tmp_list = []
    for i in range(number_of_clusters):
        rand=random.randint(0,number_of_rows-1)
        tmp_list.append(data[rand])
    centroids = np.array(tmp_list)

    # For each point, calculate the distance between it and those initial centroids
    # Assign them to nearest cluster by compairing distance between observation point and centroids
    distance_matrix= np.zeros(shape=(number_of_rows,number_of_clusters))
    for i in range(number_of_clusters):
        for j in range(number_of_rows):
            distance_matrix[j,i] = Euclidean_Distance_4d(data[j], centroids[i])
    assigned_cluster = np.argmin(distance_matrix,axis=1) + 1

    # Create a dictionary to hold points which share same clusters
    cluster_dict = {}
    for i in range(number_of_clusters):
        cluster_dict[i+1]=np.array([]).reshape(number_of_features,0)

    for i in range(number_of_rows):
        cluster_dict[assigned_cluster[i]]=np.c_[cluster_dict[assigned_cluster[i]],data[i]]

    for i in range(number_of_clusters):
        cluster_dict[i+1]=cluster_dict[i+1].T

    for i in range(number_of_clusters):
        centroids[i]=np.nanmean(cluster_dict[i+1],axis=0)


    # Iteration (Loop the above process again and again to make the centroids become stationary)
    for a in range(number_of_iterations):
        
        # Same process as above
        distance_matrix= np.zeros(shape=(number_of_rows,number_of_clusters))
        for b in range(number_of_clusters):
            for c in range(number_of_rows):
                distance_matrix[c,b] = Euclidean_Distance_4d(data[c], centroids[b])
        assigned_cluster = np.argmin(distance_matrix,axis=1) + 1
        
        cluster_dict = {}
        for b in range(number_of_clusters):
            cluster_dict[b+1]=np.array([]).reshape(number_of_features,0)

        for b in range(number_of_rows):
            cluster_dict[assigned_cluster[b]]=np.c_[cluster_dict[assigned_cluster[b]],data[b]]

        for b in range(number_of_clusters):
            cluster_dict[b+1]=cluster_dict[b+1].T

        for b in range(number_of_clusters):
            centroids[b]=np.nanmean(cluster_dict[b+1],axis=0)

    total_centroid_distance = 0
    for i in range(number_of_rows):
        total_centroid_distance += distance_matrix[i][assigned_cluster[i]-1]

    return total_centroid_distance


# Calculate total distance to centroids when number of clusters = 1 to 10
for i in range(1,11):
    total_distance_to_centroids_list.append(total_distance_to_centroids_cal(i, 75))


# Plot the graph (Total Distance to Centroids vs Number of Clusters)
plt.title('Relationship between Total Distance to Centroids vs Number of Clusters')
plt.xlabel('Number of Clusters')
plt.ylabel('Total Distance to Centroids')
plt.plot([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], total_distance_to_centroids_list)
plt.show()


# ****************************************************************************************
# After reading the graph of Total Distance to Centroids vs Number of Clusters,
# I think that 3 clusters is the best for this data
# Therefore, I will run the data and plot the centroids graph using 3 clusters on below
# ****************************************************************************************

# Parameters
number_of_clusters = 3 
number_of_iterations = 200

# Initialization

# Random pick data points as centroids initially
tmp_list = []
for i in range(number_of_clusters):
    rand=random.randint(0,number_of_rows-1)
    tmp_list.append(data[rand])
centroids = np.array(tmp_list)

# For each point, calculate the distance between it and those initial centroids
# Assign them to nearest cluster by compairing distance between observation point and centroids
distance_matrix= np.zeros(shape=(number_of_rows,number_of_clusters))
for i in range(number_of_clusters):
    for j in range(number_of_rows):
        distance_matrix[j,i] = Euclidean_Distance_4d(data[j], centroids[i])
assigned_cluster = np.argmin(distance_matrix,axis=1) + 1

# Create a dictionary to hold points which share same clusters
cluster_dict = {}
for i in range(number_of_clusters):
    cluster_dict[i+1]=np.array([]).reshape(number_of_features,0)

for i in range(number_of_rows):
    cluster_dict[assigned_cluster[i]]=np.c_[cluster_dict[assigned_cluster[i]],data[i]]

for i in range(number_of_clusters):
    cluster_dict[i+1]=cluster_dict[i+1].T

for i in range(number_of_clusters):
    centroids[i]=np.nanmean(cluster_dict[i+1],axis=0)


# Iteration (Loop the above process again and again to make the centroids become stationary)
for a in range(number_of_iterations):
    
    # Same process as above
    distance_matrix= np.zeros(shape=(number_of_rows,number_of_clusters))
    for b in range(number_of_clusters):
        for c in range(number_of_rows):
            distance_matrix[c,b] = Euclidean_Distance_4d(data[c], centroids[b])
    assigned_cluster = np.argmin(distance_matrix,axis=1) + 1
        
    cluster_dict = {}
    for b in range(number_of_clusters):
        cluster_dict[b+1]=np.array([]).reshape(number_of_features,0)

    for b in range(number_of_rows):
        cluster_dict[assigned_cluster[b]]=np.c_[cluster_dict[assigned_cluster[b]],data[b]]

    for b in range(number_of_clusters):
        cluster_dict[b+1]=cluster_dict[b+1].T

    for b in range(number_of_clusters):
        centroids[b]=np.nanmean(cluster_dict[b+1],axis=0)


# Plot graph for 3 clusters with centroids (first 2 features)
color=['blue','red','green']
labels=['cluster1','cluster2','cluster3']
for k in range(number_of_clusters):
    plt.scatter(cluster_dict[k+1][:,0],cluster_dict[k+1][:,1],c=color[k],label=labels[k])
plt.scatter(centroids[:,0],centroids[:,1],s=150,c='orange',label='Centroids')
plt.title('iris data first 2 features in 3 clusters')
plt.xlabel('feature1')
plt.ylabel('feature2')
plt.legend()
plt.show()


