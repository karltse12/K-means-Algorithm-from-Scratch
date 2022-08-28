# K-means Algorithm from Scratch

Implement the K-means algorithm and carry out experiments on the Iris
dataset (note that you are not allowed to use the libraries such as scikit-
learn to implement the algorithm itself, but you are free to compare your
results with such). The dataset can be accessed from scikit-learn library.
You may follow the instructions at the following link:
https://scikit-learn.org/stable/auto_examples/datasets/plot_iris_dataset.html

a) Plot the K-means clustering results by plotting the first 2 dimensions
of the input data as well as the converged centroids.

b) Provide some discussions about how you picked the value of K in the
K-means algorithm.

Note: You should only use the 4 input features in the Iris dataset to
cluster them, and not the labels. Also, similar to previous exercise, you
are asked to implement from scratch without using third-party implemen-
tations/ libraries.

# Major Challenge
The major challenge of this project is to build the TF-IDF from scratch. Third party libraries are NOT allowed for construction of TF-IDF

# Implementations and Results

a) I used Python to do the implementation of K-means algorithm from scratch on iris dataset.

1) First of all, stored the dataset in an array, and calculate the number of rows (150) and number of features (4)

2) Then, random picked several data points (e.g. 4 clusters) to be centroids initially.

3) Then, calculated the 4d Euclidean distance between each point and initial centroids to form a distance matrix. Then assign each point to the ‘nearest’ clusters/centroids.

4) Then, for data points in the same cluster, they will be used to calculate the centroids by using the average of 4-dimensions of those data points, then the new centroids formed for each cluster.

5) I created a function which can loop the above 3rd and 4th steps 75 times, to find the centroids for each clusters and related data points.

6) Also, in order to find the suitable K for the algorithm, I created a function to loop the above 2nd to 5th steps, and calculate the total distance to centroids (which has linear relationship with average distance to centroids), and then plotted a graph for “Relationship between Total Distance to Centroids vs Number of Clusters” (see below graph). And I found that using 3 clusters is the best.

![image](https://user-images.githubusercontent.com/57484350/187073257-d38f95f7-c785-4b0f-b9df-046606263165.png)

7) Finally, I plotted the graph for iris data in first 2 features in 3 clusters. (see below graph)

![image](https://user-images.githubusercontent.com/57484350/187073432-0e7dff9a-739b-4246-bcf7-fdf2f8487952.png)


b) The program I wrote can calculate the total distance to centroids (which has linear relationship with average distance to centroids) for K = 1 to 10, and it also plotted the graph of relationship between total centroid distance vs K (number of clusters). Total distance to centroids is the total sum of distance between centroids and their points in the same clusters. I used this distance because I saw the book use average distance to centroids, and the total distance to centroids also serve the same purpose.

On below graph, we can see the relationship between total distance to centroids and Number of Clusters, the total distance to centroids falls rapidly until K = 3, then it changes little, so I chose K =3. And then I used K =3 to plot the K-means clustering (above graph 2).

