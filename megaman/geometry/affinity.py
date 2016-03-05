from __future__ import division ## removes integer division
import numpy as np
from scipy import sparse
from scipy.spatial.distance import pdist
import subprocess, os, sys, warnings


def compute_affinity_matrix(adjacency_matrix, method, **kwargs):
    return affinity_matrix(adjacency_matrix, **kwargs)
    
def symmetrize_sparse(A):
    """
    Symmetrizes a sparse matrix in place (coo and csr formats only)

    NOTES:
    1. if there are values of 0 or 0.0 in the sparse matrix, this operation will DELETE them.
    """
    if A.getformat() is not "csr":
        A = A.tocsr()
    A = (A + A.transpose(copy = True))/2
    return A

def affinity_matrix(distances, radius=None, symmetrize = True):
    if radius is None:
        radius = 1./distances.shape[0]
    if radius <= 0.:
        raise ValueError('radius must be >0.')
    A = distances.copy()
    if sparse.isspmatrix( A ):
        A.data = A.data**2
        A.data = A.data/(-radius**2)
        np.exp( A.data, A.data )
        if symmetrize:
            A = symmetrize_sparse( A )  # converts to CSR; deletes 0's
        else:
            pass
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            # sparse will complain that this is faster with lil_matrix
            A.setdiag(1) # the 0 on the diagonal is a true zero
    else:
        A **= 2
        A /= (-radius**2)
        np.exp(A, A)
        if symmetrize:
            A = (A+A.T)/2
            A = np.asarray( A, order="C" )  # is this necessary??
        else:
            pass
    return A