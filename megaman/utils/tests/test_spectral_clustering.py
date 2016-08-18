from sklearn import neighbors
import numpy as np 

from megaman.utils.eigendecomp import EIGEN_SOLVERS
from megaman.utils.spectral_clustering import SpectralClustering

def test_spectral_clustering():
    K = 3
    num_per_cluster = 100
    c = np.array([[1,0,0], [0,1,0], [0,0,1]])
    X = np.repeat(c, np.repeat(num_per_cluster, K), axis = 0)
    radius = 5 
    rng = np.random.RandomState(36)
    def check_labels(stabalize, renormalize, eigen_solver):
        if eigen_solver in ['dense', 'auto']:
            solver_kwds = {}
        else:
            solver_kwds = {'maxiter':100000, 'tol':1e-5}
        SC = SpectralClustering(K=K, radius=radius, stabalize=stabalize, renormalize=renormalize,
                                eigen_solver = eigen_solver, solver_kwds=solver_kwds, random_state = rng,
                                additional_vectors = 0)
        labels = SC.fit_transform(X, input_type= 'data')
        for k in range(K):        
            cluster_labs = labels[range((k*num_per_cluster),((k+1)*num_per_cluster))] 
            first_lab = cluster_labs[0]
            assert(np.all(cluster_labs == first_lab))
            
    for stabalize in [True, False]:
        for renormalize in [True, False]:
            for solver in EIGEN_SOLVERS:
                yield check_labels, stabalize, renormalize, solver