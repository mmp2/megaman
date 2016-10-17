from scipy.spatial.distance import cdist, pdist, squareform
from megaman.geometry.adjacency import compute_adjacency_matrix
from megaman.geometry.complete_adjacency_matrix import complete_adjacency_matrix
import numpy as np
from numpy.testing import assert_allclose

def test_complete_adjacency():
    rand = np.random.RandomState(36)
    radius = 1.5
    X = rand.randn(10, 2)
    Xtest = rand.randn(4, 2)
    
    Xtotal = np.vstack([X, Xtest])
    D_true = squareform(pdist(Xtotal))
    D_true[D_true > radius] = 0
    
    adjacency_kwds = {'radius':radius}
    Dtrain = compute_adjacency_matrix(X, method='cyflann', radius = radius)
    this_D = complete_adjacency_matrix(Dtrain, X, Xtest, adjacency_kwds)
    
    assert_allclose(this_D.toarray(), D_true, rtol=1E-4)