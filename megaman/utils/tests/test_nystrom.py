import numpy as np
from numpy import absolute
from numpy.linalg import qr
from megaman.utils.nystrom_extension import nystrom_extension
from numpy.testing import assert_array_almost_equal


def test_nystrom_extension(seed=123):
    """ Test Nystrom Extension: low rank approximation is exact when
    G is itself low rank
    """
    n = 10
    s = 2
    rng = np.random.RandomState(seed)
    X = rng.randn(n, s)
    G = np.dot(X, X.T) # has rank s

    # find the linearly independent columns of 
    q = qr(G)[1] 
    q = absolute(q)
    sums = np.sum(q,axis=1)
    i = 0
    dims = list()
    while( i < n ): #dim is the matrix dimension
        if(sums[i] > 1.e-10):
            dims.append(i)
        i += 1
    
    # Find the eigendecomposition of the full rank portion:
    W = G[dims,:]
    W = W[:,dims]
    eval, evec = np.linalg.eigh(W)
    
    # pass the dims columns of G 
    C = G[:,dims]
    # Find the estimated eigendecomposition using Nystrom 
    eval_nystrom, evec_nystrom = nystrom_extension(C, evec, eval)
        
    # reconstruct G using Nystrom Approximatiuon 
    G_nystrom = np.dot(np.dot(evec_nystrom, np.diag(eval_nystrom)),evec_nystrom.T)
    # since rank(W) = rank(G) = s the nystrom approximation of G is exact:
    assert_array_almost_equal(G_nystrom, G)