"""K-Means Clustering"""

# Author: James McQueen <jmcq@u.washington.edu>
#         Xiao Wang <wang19@u.washington.edu>
# LICENSE: Simplified BSD https://github.com/mmp2/megaman/blob/master/LICEN

import numpy as np
import random 

class Kmeans():
    def __init__(self, K):
        self.K = K

    def fit(data):
        self.labels_ = k_means_clustering(data, self.K)
        
    def fit_transform(data):
        self.fit(data)
        return self.labels_

def k_means_clustering(data,K):
    """
    K-means clustering is an algorithm that take a data set and 
    a number of clusters K and returns the labels which represents
    the clusters of data which are similar to others
    
    Parameters    
    --------------------
    data: array-like, shape= (m_samples,n_samples)
    K: integer
        number of K clusters   
    Returns
    -------
    labels: array-like, shape (1,n_samples)    
    """
    N = data.shape[0]
    centroids, data_norms = orthogonal_initialization(data,K)
    old_centroids= np.zeros((N,K))
    labels = []
    
    # Run the main k-means algorithm
    while not _has_converged(centroids, old_centroids):    
        labels = get_labels(data, centroids,K)                
        centroids = get_centroids(data,K,labels,centroids,data_norms)
        old_centroids = centroids
        
    return labels

def orthogonal_initialization(X,K):
    """
    Initialize the centrodis by orthogonal_initialization.
    Parameters    
    --------------------
    X(data): array-like, shape= (m_samples,n_samples)
    K: integer
        number of K clusters   
    Returns
    -------
    centroids: array-like, shape (K,n_samples)  
    data_norms: array-like, shape=(1,n_samples)     
    """
    N,M = X.shape
    centroids= X[np.random.randint(0, N-1,1),:] 
    data_norms = np.linalg.norm(X, axis = 1)# contains the norm of each data point, only do this once
         
    center_norms = np.linalg.norm(centroids, axis=1) # contains the norms of the centers, will need to be updated when new center added
        
    for k in range(1,K):    
        ## Here's where we compute the cosine of the angle between them:
        # Compute the dot (inner) product between each data point and each center
        new_center_index,new_center = new_orthogonal_center(X,data_norms,centroids,center_norms =center_norms)
        centroids = np.vstack((centroids,new_center))          
        center_norms = np.hstack((center_norms,data_norms[new_center_index]))   
    return centroids,data_norms

def new_orthogonal_center(X,data_norms,centroids,center_norms=None): 
    """
    Initialize the centrodis by orthogonal_initialization.
    Parameters    
    --------------------
    X(data): array-like, shape= (m_samples,n_samples)
    data_norms: array-like, shape=(1,n_samples)
    center_norms:array-like,shape=(centroids.shape[0])
    centroids: array-like, shape (K,n_samples)        
    Returns
    -------
    new_center: array-like, shape (1,n_samples)
    new_center_index: integer   
                        data index of the new center
    """
    if center_norms is None:
        center_norms = np.linalg.norm(centroids, axis=1)
    cosine = np.inner(X,centroids) # cosine[i, j] = np.dot(X[i, :],centroids[j,:])
    cosine = cosine/center_norms # divide each column by the center norm
    cosine = cosine / data_norms[:,np.newaxis] # divide each row by the data norm  
    max_cosine = np.abs(np.max(cosine, 1)) # the largest (absolute) cosine for each data point 

    # then we find the index of the new center:
    new_center_index = np.argmin(max_cosine) # the data index of the new center is the smallest max cosine
    new_center = X[new_center_index, :]       
    return new_center_index,new_center

def get_labels(data, centroids,K):
    """
    Returns a label for each piece of data in the dataset
    
    Parameters
    ------------
    data: array-like, shape= (m_samples,n_samples)
    K: integer
        number of K clusters  
    centroids: array-like, shape=(K, n_samples)     
    
    returns
    -------------
    labels: array-like, shape (1,n_samples)
    """
    distances = np.sqrt(((data - centroids[:, np.newaxis])**2).sum(axis=2))
    return np.argmin(distances, axis=0)
    
def get_centroids(data,k,labels,centroids,data_norms):
    """
    For each element in the dataset, choose the closest centroid
    
    Parameters
    ------------
    data: array-like, shape= (m_samples,n_samples)
    K: integer, number of K clusters  
    centroids: array-like, shape=(K, n_samples)     
    labels: array-like, shape (1,n_samples)
    returns
    -------------
    centroids: array-like, shape (K,n_samples)    
    """

    D = data.shape[1]    
    for j in range(k):
        cluster_points = np.where(labels == j)
        cluster_total = len(cluster_points)
        if cluster_total == 0:
            _, temp = new_orthogonal_center(data,data_norms,centroids)
        else:
            temp = np.mean(data[cluster_points,:],axis=1)      
        centroids[j,:] = temp
    return centroids       

def _has_converged(centroids, old_centroids):
    """
    Stop if centroids stop to update
    Parameters
    -----------
    centroids: array-like, shape=(K, n_samples)     
    old_centroids: array-like, shape=(K, n_samples)
    ------------    
    returns
    True: bool
    
    """
    return (set([tuple(a) for a in centroids]) == set([tuple(a) for a in old_centroids]))    