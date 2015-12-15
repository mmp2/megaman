import numpy as np
import Mmani.embedding.geometry as geom
import sys
import scipy as sp
import scipy.sparse as sparse
import time
from sklearn import datasets
import warnings
from sklearn.neighbors import NearestNeighbors
from numpy.testing import assert_array_almost_equal

def _check_with_col_sign_flipping(A, B, tol=0.0):
    """ Check array A and B are equal with possible sign flipping on
    each columns"""
    sign = True
    for column_idx in range(A.shape[1]):
        sign = sign and ((((A[:, column_idx] -
                            B[:, column_idx]) ** 2).mean() <= tol ** 2) or
                         (((A[:, column_idx] +
                            B[:, column_idx]) ** 2).mean() <= tol ** 2))
        if not sign:
            return False
    return True

N = 10
X, color = datasets.samples_generator.make_s_curve(N, random_state=0)
n_components = 2
n_neighbors = 3

knn = NearestNeighbors(n_neighbors + 1).fit(X)

# Assign the geometry matrix to get the same answer since sklearn using k-neighbors instead of radius-neighbors
Geometry = geom.Geometry(X)
Geometry.assign_distance_matrix(knn.kneighbors_graph(X, mode = 'distance'))

from sklearn import manifold

# test LTSA with sklearn
sk_Y_ltsa = manifold.LocallyLinearEmbedding(n_neighbors, n_components, 
                                            method = 'ltsa', eigen_solver = 'arpack').fit_transform(X)
import Mmani.embedding.ltsa as ltsa
(mm_Y_ltsa, err) = ltsa.ltsa(Geometry, n_components, eigen_solver = 'arpack')
assert(_check_with_col_sign_flipping(sk_Y_ltsa, mm_Y_ltsa, 0.05))

# test LLE with sklearn
sk_Y_lle = manifold.LocallyLinearEmbedding(n_neighbors, n_components, method = 'standard').fit_transform(X)
import Mmani.embedding.locally_linear_ as lle
(mm_Y_lle, err) = lle.locally_linear_embedding(Geometry, n_components)
assert(_check_with_col_sign_flipping(sk_Y_ltsa, mm_Y_ltsa, 0.05))

# test Isomap with sklearn
sk_Y_iso = manifold.Isomap(n_neighbors, n_components, eigen_solver = 'arpack').fit_transform(X)
import Mmani.embedding.isomap_ as iso
mm_Y_iso = iso.isomap(Geometry, n_components)
assert(_check_with_col_sign_flipping(sk_Y_iso, mm_Y_iso, 0.05))
