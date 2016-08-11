from sklearn import neighbors
import numpy as np 

from spectral_clustering import spectral_clustering

def test_spectral_clustering():

    K = 3
    num_per_cluster = 10
    c = np.array([[1,1,1], [2,2,2], [3,3,3]])
    X = np.repeat(c, np.repeat(num_per_cluster, K), axis = 0)
    model = neighbors.NearestNeighbors(algorithm='brute').fit(X)
    radius = 1
    mode='distance'
    D = model.radius_neighbors_graph(X, radius=radius, mode=mode)
    S = D.copy()
    data = S.data
    data **= 2
    data /= -radius ** 2
    np.exp(data, out=data)
    S.setdiag(1)
    #print S
    
    labels = spectral_clustering(S, K=K)
    print labels
    for k in range(K):
        
        cluster_labs = labels[range((k*num_per_cluster),((k+1)*num_per_cluster))] 
        first_lab = cluster_labs[0]
        assert(np.all(cluster_labs == first_lab))
        
if __name__ == '__main__':
    test_spectral_clustering()# run the above tests        