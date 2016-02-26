import sys
sys.path.append('/homes/jmcq/megaman/') # this is stupid

import numpy as np
import megaman.geometry.geometry as geom
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
import megaman.embedding.ltsa as ltsa
(mm_Y_ltsa, err) = ltsa.ltsa(Geometry, n_components, eigen_solver = 'arpack')
assert(_check_with_col_sign_flipping(sk_Y_ltsa, mm_Y_ltsa, 0.05))

# test LLE with sklearn
sk_Y_lle = manifold.LocallyLinearEmbedding(n_neighbors, n_components, method = 'standard').fit_transform(X)
import megaman.embedding.locally_linear_ as lle
(mm_Y_lle, err) = lle.locally_linear_embedding(Geometry, n_components)
assert(_check_with_col_sign_flipping(sk_Y_ltsa, mm_Y_ltsa, 0.05))

# test Isomap with sklearn
sk_Y_iso = manifold.Isomap(n_neighbors, n_components, eigen_solver = 'arpack').fit_transform(X)
import megaman.embedding.isomap_ as iso
mm_Y_iso = iso.isomap(Geometry, n_components)
assert(_check_with_col_sign_flipping(sk_Y_iso, mm_Y_iso, 0.05))

from scipy.spatial.distance import pdist, squareform


Geometry = geom.Geometry(X, neighborhood_radius = 2, distance_method = 'brute')

distance_mat = Geometry.get_distance_matrix()
graph_distance_matrix = graph_shortest_path(distance_mat)
A = graph_distance_matrix.copy()
A **= 2
A = np.exp(-A)

centered_matrix = center_matrix(graph_distance_matrix)
center_matrix_A = center_matrix(A)

lambdas, diffusion_map = eigen_decomposition(centered_matrix, n_components,largest = True)
ind = np.argsort(lambdas); ind = ind[::-1] # sort largest
lambdas = lambdas[ind];
diffusion_map = diffusion_map[:, ind]
embedding = diffusion_map[:, 0:n_components] * np.sqrt(lambdas[0:n_components])

lambdas2, diffusion_map2 = eigen_decomposition(A, n_components,largest = True)
ind2 = np.argsort(lambdas2); ind2 = ind2[::-1] # sort largest
lambdas2 = lambdas2[ind2];
diffusion_map2 = diffusion_map2[:, ind2]
embedding2 = diffusion_map2[:, 0:n_components] * np.sqrt(lambdas2[0:n_components])
