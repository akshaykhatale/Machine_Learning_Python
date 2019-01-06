#K-means Clustering

#importing Libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#importing data
dataset= pd.read_csv('iris.csv')
x=dataset.iloc[:,[0,2]].values

#Feature Scaling
from sklearn.preprocessing import  StandardScaler
sc_x = StandardScaler()
x=sc_x.fit_transform(x)

#finding the optimal number of clusters(elbow method)
from sklearn.cluster import KMeans
wcss=[]
for i in range(1,11):
    kmeans=KMeans(n_clusters=i,init='k-means++',max_iter=100, n_init=10,random_state=0)
    kmeans.fit(x)
    wcss.append(kmeans.inertia_)
plt.plot(range(1,11),wcss)
plt.title('Finding optimal N Clusters')
plt.xlabel('Number of Clusters')
plt.ylabel('WCSS')
plt.legend()
plt.show()

#Apply k-means to the data set
kmeans=KMeans(n_clusters=3,init='k-means++',max_iter=100, n_init=10,random_state=0)
y_kmeans=kmeans.fit_predict(x)

#visualizing the clusters
plt.scatter(x[y_kmeans==0,0],x[y_kmeans==0,1],s=20,c='red',label='Iris-setosa')
plt.scatter(x[y_kmeans==1,0],x[y_kmeans==1,1],s=20,c='magenta',label='Iris-versicolor')
plt.scatter(x[y_kmeans==2,0],x[y_kmeans==2,1],s=20,c='cyan',label='Iris-virginica')
plt.scatter(kmeans.cluster_centers_[:,0],kmeans.cluster_centers_[:,1],s=100,c='blue',label='centroid')
plt.title('Clusters of flowers')
plt.legend()
plt.show()