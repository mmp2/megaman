from sklearn import neighbors
import numpy as np 

from megaman.utils.spectral_clustering import SpectralClustering

def test_spectral_clustering():
    K = 3
    num_per_cluster = 50
    c = np.array([[1,0,0], [0,1,0], [0,0,1]])
    X = np.repeat(c, np.repeat(num_per_cluster, K), axis = 0)
    radius = 5 
    
    def check_labels(stabalize, renormalize):
        SC = SpectralClustering(K=K, radius=radius, stabalize=stabalize, renormalize=renormalize)
        labels = SC.fit_transform(X, input_type= 'data')
        for k in range(K):        
            cluster_labs = labels[range((k*num_per_cluster),((k+1)*num_per_cluster))] 
            first_lab = cluster_labs[0]
            assert(np.all(cluster_labs == first_lab))
            
    for stabalize in [True, False]:
        for renormalize in [True, False]:
            yield check_labels, stabalize, renormalize