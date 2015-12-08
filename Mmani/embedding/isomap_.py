"""ISOMAP"""

# Author: James McQueen <jmcq@u.washington.edu>
#
#         after the scikit-learn version by 
#         Jake Vanderplas  -- <vanderplas@astro.washington.edu>
# License: BSD 3 clause

import warnings
import numpy as np
import Mmani.embedding.geometry as geom
from Mmani.embedding.eigendecomp import eigen_decomposition
from scipy import sparse
from scipy.sparse.csgraph import shortest_path as graph_shortest_path
from Mmani.utils.validation import check_random_state
# debugging
import time 

def isomap(Geometry, n_components=8, eigen_solver=None,
           random_state=None, eigen_tol=0.0, path_method='auto'):
    """
    TO UPDATE!!!
    Parameters
    ----------        
    Geometry : a Geometry object from Mmani.embedding.geometry

    n_components : integer, optional
        The dimension of the projection subspace.

    eigen_solver : {None, 'arpack', 'lobpcg', or 'amg'}
        The eigenvalue decomposition strategy to use. AMG requires pyamg
        to be installed. It can be faster on very large, sparse problems,
        but may also lead to instabilities.

    random_state : int seed, RandomState instance, or None (default)
        A pseudo random number generator used for the initialization of the
        lobpcg eigen vectors decomposition when eigen_solver == 'amg'.
        By default, arpack is used.

    eigen_tol : float, optional, default=0.0
        Stopping criterion for eigendecomposition of the Laplacian matrix
        when using arpack eigen_solver.


    Returns
    -------
    embedding : array, shape=(n_samples, n_components)
        The reduced samples.

    Notes
    -----
    """

    random_state = check_random_state(random_state)    

    if not isinstance(Geometry, geom.Geometry):
        raise RuntimeError("Geometry object not Mmani.embedding.geometry Geometry class")
        
    # Step 1: 
    # Use geometry to calculate the distance matrix 
    distance_matrix = Geometry.get_distance_matrix()
        # Step 2:
    # use graph_shortest_path to construct D_G
    ## NOTE: D_G is a DENSE matrix!! 
    graph_distance_matrix = graph_shortest_path(distance_matrix,
                                                method=path_method,
                                                directed=False)
    # Step 3:
    # Let S = -1/2* D_g^2 and  N_1 = np.ones([N, N])/N 
    # Compute centred version: K = S - N_1*S - S*N_1  + N_1*S*N_1
    S = graph_distance_matrix ** 2 
    S *= -0.5
    N = S.shape[0]
    K = S.copy()
    row_sums = np.sum(S, axis=0)/N
    K -= row_sums
    K -= (np.sum(S, axis = 1)/N)[:, np.newaxis]
    K += np.sum(row_sums)/N
        
    # Step 4:
    # Compute d largest eigenvectors/values of K 
    lambdas, diffusion_map = eigen_decomposition(K, n_components, eigen_solver,
                                                 random_state, eigen_tol, largest = True)

    
    # Step 5: 
    # return Y = [sqrt(lambda_1)*V_1, ..., sqrt(lambda_d)*V_d]
    ind = np.argsort(lambdas); ind = ind[::-1] # sort largest 
    lambdas = lambdas[ind];
    diffusion_map = diffusion_map[:, ind]
    embedding = diffusion_map[:, 0:n_components] * np.sqrt(lambdas[0:n_components])
    return embedding


class Isomap():
    """
    TO UPDATE!!!
    Parameters
    -----------
    n_components : integer, default: 2
        The dimension of the projected subspace.

    eigen_solver : {None, 'arpack', 'lobpcg', or 'amg'}
        The eigenvalue decomposition strategy to use. AMG requires pyamg
        to be installed. It can be faster on very large, sparse problems,
        but may also lead to instabilities.

    random_state : int seed, RandomState instance, or None, default : None
        A pseudo random number generator used for the initialization of the
        lobpcg eigen vectors decomposition when eigen_solver == 'amg'.

    is_affinity : string or callable, default : "nearest_neighbors"
         - True : interpret X as precomputed affinity matrix

    radius : float, optional, default : 1/n_features
        Kernel coefficient for rbf kernel.

    Attributes
    ----------

    `embedding_` : array, shape = (n_samples, n_components)
        Spectral embedding of the training matrix.

    References
    ----------
    """

    """ 
    """
    def __init__(self, n_components=2, is_distance = False,
                 radius=None, random_state=None, eigen_solver=None,
                 use_flann = False, path_to_flann = None, cpp_distances = False,
                 path_method = 'auto'):
        self.n_components = n_components
        self.is_distance = is_distance
        self.radius = radius
        self.random_state = random_state
        self.eigen_solver = eigen_solver
        self.use_flann = use_flann
        self.path_to_flann = path_to_flann
        self.cpp_distances = cpp_distances
        self.path_method = path_method

    def fit(self, X, y=None):
        """Fit the model from data in X.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            Training vector, where n_samples in the number of samples
            and n_features is the number of features.

            If is_distance is True
            X : array-like, shape (n_samples, n_samples),
            Interpret X as precomputed pairwise distance graph computed from
            samples.

        Returns
        -------
        self : object
            Returns the instance itself.
        """
        random_state = check_random_state(self.random_state)
        if self.is_distance:
            Geometry = geom.Geometry(X, neighbors_radius = self.radius, 
                                        is_distance = True, use_flann = 
                                        self.use_flann, path_to_flann = 
                                        self.path_to_flann, cpp_distances = 
                                        self.cpp_distances)
        else:
            Geometry = geom.Geometry(X, neighbors_radius = self.radius, use_flann = 
                                        self.use_flann, path_to_flann = 
                                        self.path_to_flann, cpp_distances = 
                                        self.cpp_distances)
        self.embedding_ = isomap(Geometry, n_components=self.n_components,
                                 eigen_solver=self.eigen_solver,
                                 random_state=random_state, path_method = 
                                        self.path_method)
        return self

    def fit_transform(self, X, y=None):
        """Fit the model from data in X and transform X.

        Parameters
        ----------
        X: array-like, shape (n_samples, n_features)
            Training vector, where n_samples in the number of samples
            and n_features is the number of features.

            If affinity is "precomputed"
            X : array-like, shape (n_samples, n_samples),
            Interpret X as precomputed adjacency graph computed from
            samples.

        Returns
        -------
        X_new: array-like, shape (n_samples, n_components)
        """
        self.fit(X)
        return self.embedding_
