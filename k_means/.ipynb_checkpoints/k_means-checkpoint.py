import matplotlib.pyplot as plt
import numpy as np 
import pandas as pd 
import seaborn as sns 
sns.set_style('darkgrid')
# IMPORTANT: DO NOT USE ANY OTHER 3RD PARTY PACKAGES
# (math, random, collections, functools, etc. are perfectly fine)


class KMeans:
    
    def __init__(self, k=15, maxIter = 200):
        self.k = k
        self.centroids = []
        self.y = []
        self.maxIter = maxIter
        self.firstCentroids = None
        # NOTE: Feel free add any hyperparameters 
        # (with defaults) as you see fit
        pass
        
    def fit(self, X, centroids = None): #centroids used for elbow
        """
        Estimates parameters for the classifier
        
        Args:
            X (array<m,n>): a matrix of floats with
                m rows (#samples) and n columns (#features)
        """
        # TODO: Implement
        Xt =  X
        X = np.array(X)
        temp = []
       # if centroids is None: used for elbow
        centroids= np.random.uniform(np.amin(X,axis =0), np.amax(X,axis = 0), size = (1,X.shape[1]))
            #self.firstCentroid = centroids used for elbow
        for i in range(self.k):
            max_dist  = 0.0
            for data in X:
                dist = euclidean_distance(data, centroids)
                if max(dist) > max_dist:
                    max_dist = np.argmax(dist)
                    if data not in centroids:
                        centroids = np.vstack((centroids, data))
                if(centroids.shape[0] >= self.k):
                    break
        for i in range(self.maxIter):
            temp = []
            for point in X:
                distance = euclidean_distance(point, centroids)
                cluster = np.argmin(distance)
                temp.append(cluster)
            temp = np.array(temp)        
           
            
            cluster_index=[]
            
            
            for j in range(centroids.shape[0]):
                cluster_index.append(np.argwhere(temp == j))
            cluster_center =[]
            
            for j,index in enumerate(cluster_index):
                if len(index) == 0:
                    cluster_center.append(self.centroids[i])
                else:
                    cluster_center.append(np.mean(X[index],axis=0)[0])
            
            if np.max(centroids -np.array(cluster_center))<0.001:
                break
            else:
                centroids = np.array(cluster_center)
               
     #   print(self.k,self.centroids.shape, self.centroids)
        self.y = temp
        self.centroids = centroids
    
    def predict(self, X):
        """
        Generates predictions
        
        Note: should be called after .fit()
        
        Args:
            X (array<m,n>): a matrix of floats with 
                m rows (#samples) and n columns (#features)
            
        Returns:
            A length m integer array with cluster assignments
            for each point. E.g., if X is a 10xn matrix and 
            there are 3 clusters, then a possible assignment
            could be: array([2, 0, 0, 1, 2, 1, 1, 0, 2, 2])
        """
        # TODO: Implement 
        # euclidean_silhouette elbow on piazza TA said is not necessary
        """
        clusterN = self.k
        max = 0.0
        prediction = []
        for i in range(1,clusterN):
            self.k = i
            y,tempCent = self.fit(X,self.firstCentroids)
            silhouette = euclidean_silhouette(X,y)
            prediction.append(silhouette)
            if(silhouette > max):
                max = silhouette
                self.y = y
                self.centroids = tempCent
        plt.plot(prediction)
        """
        return self.y
    
    def get_centroids(self):
        """
        Returns the centroids found by the K-mean algorithm
        
        Example with m centroids in an n-dimensional space:
        >>> model.get_centroids()
        numpy.array([
            [x1_1, x1_2, ..., x1_n],
            [x2_1, x2_2, ..., x2_n],
                    .
                    .
                    .
            [xm_1, xm_2, ..., xm_n]
        ])
        """

            
                
        return self.centroids
    
    
    
# --- Some utility functions 

def euclidean_distance(x, y):
    """
    Computes euclidean distance between two sets of points 
    
    Note: by passing "y=0.0", it will compute the euclidean norm
    
    Args:
        x, y (array<...,n>): float tensors with pairs of 
            n-dimensional points 
            
    Returns:
        A float array of shape <...> with the pairwise distances
        of each x and y point
    """
    return np.linalg.norm(x - y, ord=2, axis=-1)

def cross_euclidean_distance(x, y=None):
    """
    
    
    """
    y = x if y is None else y 
    assert len(x.shape) >= 2
    assert len(y.shape) >= 2
    return euclidean_distance(x[..., :, None, :], y[..., None, :, :])


def euclidean_distortion(X, z):
    """
    Computes the Euclidean K-means distortion
    
    Args:
        X (array<m,n>): m x n float matrix with datapoints 
        z (array<m>): m-length integer vector of cluster assignments
    
    Returns:
        A scalar float with the raw distortion measure 
    """
    X, z = np.asarray(X), np.asarray(z)
    assert len(X.shape) == 2
    assert len(z.shape) == 1
    assert X.shape[0] == z.shape[0]
    
    distortion = 0.0
    clusters = np.unique(z)
    for i, c in enumerate(clusters):
        Xc = X[z == c]
        mu = Xc.mean(axis=0)
        distortion += ((Xc - mu) ** 2).sum() #removed axis = 1
        
    return distortion


def euclidean_silhouette(X, z):
    """
    Computes the average Silhouette Coefficient with euclidean distance 
    
    More info:
        - https://www.sciencedirect.com/science/article/pii/0377042787901257
        - https://en.wikipedia.org/wiki/Silhouette_(clustering)
    
    Args:
        X (array<m,n>): m x n float matrix with datapoints 
        z (array<m>): m-length integer vector of cluster assignments
    
    Returns:
        A scalar float with the silhouette score
    """
    X, z = np.asarray(X), np.asarray(z)
    assert len(X.shape) == 2
    assert len(z.shape) == 1
    assert X.shape[0] == z.shape[0]
    
    # Compute average distances from each x to all other clusters
    clusters = np.unique(z)
    D = np.zeros((len(X), len(clusters)))
    for i, ca in enumerate(clusters):
        for j, cb in enumerate(clusters):
            in_cluster_a = z == ca
            in_cluster_b = z == cb
            d = cross_euclidean_distance(X[in_cluster_a], X[in_cluster_b])
            div = d.shape[1] - int(i == j)
            D[in_cluster_a, j] = d.sum(axis=1) / np.clip(div, 1, None)
    
    # Intra distance 
    a = D[np.arange(len(X)), z]
    # Smallest inter distance 
    inf_mask = np.where(z[:, None] == clusters[None], np.inf, 0)
    b = (D + inf_mask).min(axis=1)
    
    return np.mean((b - a) / np.maximum(a, b))
  