"""ISOMAP"""

# Author: James McQueen <jmcq@u.washington.edu>
#
# License: BSD 3 clause

import warnings

import numpy as np
from scipy import sparse
from scipy.sparse.csgraph import shortest_path as graph_shortest_path

from ..geometry import geometry as geom
from ..utils.eigendecomp import eigen_decomposition
from ..utils.validation import check_random_state


def center_matrix(G):
    # Let S = -1/2* D_g^2 and  N_1 = np.ones([N, N])/N
    # Compute centred version: K = S - N_1*S - S*N_1  + N_1*S*N_1
    S = G ** 2
    S *= -0.5
    N = S.shape[0]
    K = S.copy()
    row_sums = np.sum(S, axis=0)/N
    K -= row_sums
    K -= (np.sum(S, axis = 1)/N)[:, np.newaxis]
    K += np.sum(row_sums)/N
    return(K)

def isomap(Geometry, n_components=8, eigen_solver=None,
           random_state=None, eigen_tol=1e-12, path_method='auto',
           distance_matrix = None, graph_distance_matrix = None,
           centered_matrix = None):
    """
    Parameters
    ----------
    Geometry : a Geometry object from megaman.geometry.geometry

    n_components : integer, optional
        The dimension of the projection subspace.
    eigen_solver : {'auto', 'dense', 'arpack', 'lobpcg', or 'amg'}
        'auto' :
            algorithm will attempt to choose the best method for input data
        'dense' :
            use standard dense matrix operations for the eigenvalue decomposition.
            For this method, M must be an array or matrix type.  This method should be avoided for large problems.
        'arpack' :
            use arnoldi iteration in shift-invert mode. For this method,
            M may be a dense matrix, sparse matrix, or general linear operator.
            Warning: ARPACK can be unstable for some problems.  It is best to
            try several random seeds in order to check results.
        'lobpcg' :
            Locally Optimal Block Preconditioned Conjugate Gradient Method.
            A preconditioned eigensolver for large symmetric positive definite
            (SPD) generalized eigenproblems.
        'amg' :
            AMG requires pyamg to be installed. It can be faster on very large,
            sparse problems, but may also lead to instabilities.
    random_state : int seed, RandomState instance, or None (default)
        A pseudo random number generator used for the initialization of the
        lobpcg eigen vectors decomposition when eigen_solver == 'amg'.
        By default, arpack is used.
    eigen_tol : float, optional, default=0.0
        Stopping criterion for eigendecomposition of the Laplacian matrix
        when using arpack eigen_solver.
    path_method : string, method for computing graph shortest path. One of :
        'auto', 'D', 'FW', 'BF', 'J'. See scipy.sparse.csgraph.shortest_path
        for more information.
    distance_matrix : sparse Ndarray (n_obs, n_obs), optional. Pairwise distance matrix
        sparse zeros considered 'infinite'.
    graph_distance_matrix : Ndarray (n_obs, n_obs), optional. Pairwise graph distance
        matrix. Output of graph_shortest_path.
    centered_matrix : Ndarray (n_obs, n_obs), optional. Centered version of
        graph_distance_matrix

    Returns
    -------
    embedding : array, shape=(n_samples, n_components)
        The reduced samples.

    Notes
    -----
    """

    random_state = check_random_state(random_state)

    if not isinstance(Geometry, geom.Geometry):
        raise RuntimeError("Geometry object not megaman.embedding.geometry ",
                            "Geometry class")

    # Step 1: use geometry to calculate the distance matrix
    if ((distance_matrix is None) and (centered_matrix is None)):
        distance_matrix = Geometry.get_distance_matrix()

    # Step 2: use graph_shortest_path to construct D_G
    ## WARNING: D_G is an (NxN) DENSE matrix!!
    if ((graph_distance_matrix is None) and (centered_matrix is None)):
        graph_distance_matrix = graph_shortest_path(distance_matrix,
                                                    method=path_method,
                                                    directed=False)

    # Step 3: center graph distance matrix
    if centered_matrix is None:
        centered_matrix = center_matrix(graph_distance_matrix)


    # Step 4: compute d largest eigenvectors/values of centered_matrix
    lambdas, diffusion_map = eigen_decomposition(centered_matrix, n_components, eigen_solver,
                                                 random_state, eigen_tol,
                                                 largest = True)
    # Step 5:
    # return Y = [sqrt(lambda_1)*V_1, ..., sqrt(lambda_d)*V_d]
    ind = np.argsort(lambdas); ind = ind[::-1] # sort largest
    lambdas = lambdas[ind];
    diffusion_map = diffusion_map[:, ind]
    embedding = diffusion_map[:, 0:n_components] * np.sqrt(lambdas[0:n_components])
    return embedding

class Isomap():
    """
    Parameters
    -----------
    n_components : integer, default: 2
        The dimension of the projected subspace.
    eigen_solver : {'auto', 'dense', 'arpack', 'lobpcg', or 'amg'}
        'auto' :
            algorithm will attempt to choose the best method for input data
        'dense' :
            use standard dense matrix operations for the eigenvalue decomposition.
            For this method, M must be an array or matrix type.  This method should be avoided for large problems.
        'arpack' :
            use arnoldi iteration in shift-invert mode. For this method,
            M may be a dense matrix, sparse matrix, or general linear operator.
            Warning: ARPACK can be unstable for some problems.  It is best to
            try several random seeds in order to check results.
        'lobpcg' :
            Locally Optimal Block Preconditioned Conjugate Gradient Method.
            A preconditioned eigensolver for large symmetric positive definite
            (SPD) generalized eigenproblems.
        'amg' :
            AMG requires pyamg to be installed. It can be faster on very large,
            sparse problems, but may also lead to instabilities.
    random_state : int seed, RandomState instance, or None, default : None
        A pseudo random number generator used for the initialization of the
        lobpcg eigen vectors decomposition when eigen_solver == 'amg'.
    eigen_tol : float, optional. Tolerance for 'arpack' solver.
    path_method : string, optionl. method for computing graph shortest path.
        One of :
        'auto', 'D', 'FW', 'BF', 'J'. See scipy.sparse.csgraph.shortest_path
        for more information.
    neighborhood_radius : scalar, passed to distance_matrix. Value such that all
        distances beyond neighborhood_radius are considered infinite.
    affinity_radius : scalar, passed to affinity_matrix. 'bandwidth' parameter
        used in Guassian kernel for affinity matrix
    distance_method : string, one of 'auto', 'brute', 'cython', 'pyflann', 'cyflann'.
        method for computing pairwise radius neighbors graph.
    input_type : string, one of: 'data', 'distance', 'affinity'.
        The values of input data X.
    path_to_flann : string. full file path location of FLANN if not installed to root or
        FLANN_ROOT set to path location. Used for importing pyflann from a
        different location.
    Geometry : a Geometry object from megaman.geometry.geometry

    Returns
    ----------
    embedding_ : array, shape = (n_samples, n_components)
        Spectral embedding of the training matrix.

    References
    ----------
    * Tenenbaum, J.B.; De Silva, V.; & Langford, J.C.
      A global geometric framework for nonlinear dimensionality reduction.
      Science 290 (5500)
    """
    def __init__(self, n_components=2, eigen_solver=None, random_state=None,
                 eigen_tol = 1e-12, path_method = 'auto',
                 neighborhood_radius = None, affinity_radius = None,
                 distance_method = 'auto', input_type = 'data',
                 path_to_flann = None, Geometry = None):
        # embedding parameters:
        self.n_components = n_components
        self.random_state = random_state
        self.eigen_solver = eigen_solver
        self.eigen_tol = eigen_tol
        self.path_method = path_method

        # Geometry parameters:
        self.Geometry = Geometry
        self.neighborhood_radius = neighborhood_radius
        self.affinity_radius = affinity_radius
        self.distance_method = distance_method
        self.input_type = input_type
        self.path_to_flann = path_to_flann

        # intermediary steps for storage
        self.distance_matrix = None
        self.graph_distance_matrix = None
        self.centered_matrix = None

    def fit_geometry(self, X):
        self.Geometry = geom.Geometry(X,
                                      neighborhood_radius = self.neighborhood_radius,
                                      affinity_radius = self.affinity_radius,
                                      distance_method = self.distance_method,
                                      input_type = self.input_type,
                                      path_to_flann = self.path_to_flann)

    def fit(self, X, eigen_solver = None, input_type = 'data', n_components=None):
        """Fit the model from data in X.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            Training vector, where n_samples in the number of samples
            and n_features is the number of features.

        If self.input_type is 'distance', or 'affinity':

        X : array-like, shape (n_samples, n_samples),
            Interpret X as precomputed distance or adjacency graph
            computed from samples.

        input_type : string, one of: 'data', 'distance', 'affinity'.
            The values of input data X. (default = 'data')

        eigen_solver : {None, 'arpack', 'lobpcg', or 'amg'}
            The eigenvalue decomposition strategy to use. AMG requires pyamg
            to be installed. It can be faster on very large, sparse problems,
            but may also lead to instabilities.

        Returns
        -------
        self : object
            Returns the instance itself.
        """
        if input_type is not None:
            self.input_type = input_type
        if not isinstance(self.Geometry, geom.Geometry):
            self.fit_geometry(X)
        # might want to change the eigen solver
        if ((eigen_solver is not None) and (eigen_sovler != self.eigen_solver)):
            self.eigen_solver = eigen_solver
        # we also might want to change the # of components:
        if ((n_components is not None) and (n_components != self.n_components)):
            self.n_components = n_components

        # don't re-compute these if it's already been done.
        # This might be the case if an eigendecompostion fails and a different sovler is selected
        if self.distance_matrix is None:
            self.distance_matrix = self.Geometry.get_distance_matrix()
        if self.graph_distance_matrix is None:
            self.graph_distance_matrix = graph_shortest_path(self.distance_matrix,
                                                             method = self.path_method,
                                                             directed = False)
        if self.centered_matrix is None:
            self.centered_matrix = center_matrix(self.graph_distance_matrix)

        random_state = check_random_state(self.random_state)
        self.embedding_ = isomap(self.Geometry, n_components=self.n_components,
                                 eigen_solver=self.eigen_solver,
                                 random_state=random_state,
                                 eigen_tol = self.eigen_tol,
                                 path_method = self.path_method,
                                 distance_matrix = self.distance_matrix,
                                 graph_distance_matrix = self.graph_distance_matrix,
                                 centered_matrix = self.centered_matrix)
        return self

    def fit_transform(self, X):
        """Fit the model from data in X and transform X.

        Parameters
        ----------
        X: array-like, shape (n_samples, n_features)
            Training vector, where n_samples in the number of samples
            and n_features is the number of features.

        If self.input_type is 'distance', or 'affinity':

        X : array-like, shape (n_samples, n_samples),
            Interpret X as precomputed distance or adjacency graph
            computed from samples.

        Returns
        -------
        X_new: array-like, shape (n_samples, n_components)
        """
        self.fit(X)
        return self.embedding_
