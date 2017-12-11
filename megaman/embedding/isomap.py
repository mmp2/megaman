"""ISOMAP"""

# Author: James McQueen <jmcq@u.washington.edu>
# LICENSE: Simplified BSD https://github.com/mmp2/megaman/blob/master/LICENSE

from __future__ import division
import numpy as np
from scipy import sparse
from scipy.sparse.csgraph import shortest_path as graph_shortest_path

from ..utils.eigendecomp import eigen_decomposition
from ..embedding.base import BaseEmbedding

def center_matrix(G):
    # Let S = -1/2* D_g^2 and  N_1 = np.ones([N, N])/N
    # Compute centred version: K = S - N_1*S - S*N_1  + N_1*S*N_1
    S = G.copy()
    S = S ** 2
    S *= -0.5
    N = S.shape[0]
    K = S.copy()
    row_sums = np.sum(S, axis=0)/N
    K -= row_sums
    K -= (np.sum(S, axis = 1)/N)[:, np.newaxis]
    K += np.sum(row_sums)/N
    return(K)

def isomap(geom, n_components=8, eigen_solver='auto',
           random_state=None, path_method='auto',
           distance_matrix=None, graph_distance_matrix = None,
           centered_matrix=None, solver_kwds=None):
    """
    Parameters
    ----------
    geom : a Geometry object from megaman.geometry.geometry
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
    path_method : string, method for computing graph shortest path. One of :
        'auto', 'D', 'FW', 'BF', 'J'. See scipy.sparse.csgraph.shortest_path
        for more information.
    distance_matrix : sparse Ndarray (n_obs, n_obs), optional. Pairwise distance matrix
        sparse zeros considered 'infinite'.
    graph_distance_matrix : Ndarray (n_obs, n_obs), optional. Pairwise graph distance
        matrix. Output of graph_shortest_path.
    centered_matrix : Ndarray (n_obs, n_obs), optional. Centered version of
        graph_distance_matrix
    solver_kwds : any additional keyword arguments to pass to the selected eigen_solver

    Returns
    -------
    embedding : array, shape=(n_samples, n_components)
        The reduced samples.

    Notes
    -----
    """
    # Step 1: use geometry to calculate the distance matrix
    if ((distance_matrix is None) and (centered_matrix is None)):
        if geom.adjacency_matrix is None:
            distance_matrix = geom.compute_adjacency_matrix()
        else:
            distance_matrix = geom.adjacency_matrix

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
    lambdas, diffusion_map = eigen_decomposition(centered_matrix, n_components,
                                                 largest=True,
                                                 eigen_solver=eigen_solver,
                                                 random_state=random_state,
                                                 solver_kwds=solver_kwds)
    # Step 5:
    # return Y = [sqrt(lambda_1)*V_1, ..., sqrt(lambda_d)*V_d]
    ind = np.argsort(lambdas); ind = ind[::-1] # sort largest
    lambdas = lambdas[ind];
    diffusion_map = diffusion_map[:, ind]
    embedding = diffusion_map[:, 0:n_components] * np.sqrt(lambdas[0:n_components])
    return embedding

class Isomap(BaseEmbedding):
    """Isomap Embedding

    Non-linear dimensionality reduction through Isometric Mapping

    Parameters
    -----------

    n_components : integer
        number of coordinates for the manifold.
    radius : float (optional)
        radius for adjacency and affinity calculations. Will be overridden if
        either is set in `geom`
    geom : dict or megaman.geometry.Geometry object
        specification of geometry parameters: keys are
        ["adjacency_method", "adjacency_kwds", "affinity_method",
         "affinity_kwds", "laplacian_method", "laplacian_kwds"]
    eigen_solver : {'auto', 'dense', 'arpack', 'lobpcg', or 'amg'}
        'auto' :
            algorithm will attempt to choose the best method for input data
        'dense' :
            use standard dense matrix operations for the eigenvalue
            decomposition. Uses a dense data array, and thus should be avoided
            for large problems.
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
    random_state : numpy.RandomState or int, optional
        The generator or seed used to determine the starting vector for arpack
        iterations.  Defaults to numpy.random.RandomState
    path_method : string, optionl. method for computing graph shortest path.
        One of ['auto', 'D', 'FW', 'BF', 'J'].
        See `scipy.sparse.csgraph.shortest_path` for more information.
    solver_kwds : any additional keyword arguments to pass to the selected eigen_solver

    Attributes
    ----------
    embedding_ : array, shape = (n_samples, n_components)
        Spectral embedding of the training matrix.

    References
    ----------

    .. [1] Tenenbaum, J.B.; De Silva, V.; & Langford, J.C. A global geometric
           framework for nonlinear dimensionality reduction. Science 290 (5500)
    """
    def __init__(self, n_components=2, radius=None, geom=None,
                 eigen_solver='auto', random_state=None,
                 path_method='auto', solver_kwds=None):
        self.n_components = n_components
        self.radius = radius
        self.geom = geom
        self.eigen_solver = eigen_solver
        self.random_state = random_state
        self.path_method = path_method
        self.solver_kwds = solver_kwds

    def fit(self, X, y=None, input_type='data'):
        """Fit the model from data in X.

        Parameters
        ----------
        input_type : string, one of: 'data', 'distance'.
            The values of input data X. (default = 'data')

        X : array-like, shape (n_samples, n_features)
            Training vector, where n_samples in the number of samples
            and n_features is the number of features.

        If self.input_type is 'distance':

        X : array-like, shape (n_samples, n_samples),
            Interpret X as precomputed distance or adjacency graph
            computed from samples.

        eigen_solver : {None, 'arpack', 'lobpcg', or 'amg'}
            The eigenvalue decomposition strategy to use. AMG requires pyamg
            to be installed. It can be faster on very large, sparse problems,
            but may also lead to instabilities.

        Returns
        -------
        self : object
            Returns the instance itself.
        """

        X = self._validate_input(X, input_type)
        self.fit_geometry(X, input_type)

        if not hasattr(self, 'distance_matrix'):
            self.distance_matrix = None
        if not hasattr(self, 'graph_distance_matrix'):
            self.graph_distance_matrix = None
        if not hasattr(self, 'centered_matrix'):
            self.centered_matrix = None

        # don't re-compute these if it's already been done.
        # This might be the case if an eigendecompostion fails and a different sovler is selected
        if (self.distance_matrix is None and self.geom_.adjacency_matrix is None):
            self.distance_matrix = self.geom_.compute_adjacency_matrix()
        elif self.distance_matrix is None:
            self.distance_matrix = self.geom_.adjacency_matrix
        if self.graph_distance_matrix is None:
            self.graph_distance_matrix = graph_shortest_path(self.distance_matrix,
                                                             method = self.path_method,
                                                             directed = False)
        if self.centered_matrix is None:
            self.centered_matrix = center_matrix(self.graph_distance_matrix)

        self.embedding_ = isomap(self.geom_, n_components=self.n_components,
                                 eigen_solver=self.eigen_solver,
                                 random_state=self.random_state,
                                 path_method = self.path_method,
                                 distance_matrix = self.distance_matrix,
                                 graph_distance_matrix = self.graph_distance_matrix,
                                 centered_matrix = self.centered_matrix,
                                 solver_kwds = self.solver_kwds)
        return self
