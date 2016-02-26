"""Local Tangent Space Alignment"""

# Author: James McQueen -- <jmcq@u.washington.edu>
#
#
# After the sci-kit learn version by:
#         Fabian Pedregosa -- <fabian.pedregosa@inria.fr>
#         Jake Vanderplas  -- <vanderplas@astro.washington.edu>
# License: BSD 3 clause (C) INRIA 2011

import warnings
import numpy as np
import scipy.sparse as sparse
from scipy.linalg import eigh, svd, qr, solve
from scipy.sparse import eye, csr_matrix

from ..geometry import geometry as geom
from ..utils.validation import check_random_state, check_array
from ..utils.eigendecomp import null_space


def ltsa(Geometry, n_components, eigen_solver='auto', tol=1e-6,
         max_iter=100,random_state=None):
    """
        Perform a Locally Linear Embedding analysis on the data.

        Parameters
        ----------
        n_components : integer
            number of coordinates for the manifold.
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
        tol : float, optional
            Tolerance for 'arpack' method
            Not used if eigen_solver=='dense'.
        max_iter : integer
            maximum number of iterations for the arpack solver.
        random_state : numpy.RandomState or int, optional
            The generator or seed used to determine the starting vector for arpack
            iterations.  Defaults to numpy.random.

        Returns
        -------
        embedding : array-like, shape [n_samples, n_components]
            Embedding vectors.
        squared_error : float
            Reconstruction error for the embedding vectors. Equivalent to
            ``norm(Y - W Y, 'fro')**2``, where W are the reconstruction weights.

        References
        ----------
        * Zhang, Z. & Zha, H. Principal manifolds and nonlinear
          dimensionality reduction via tangent space alignment.
          Journal of Shanghai Univ.  8:406 (2004)
    """
    if Geometry.X is None:
        raise ValueError("Must pass data matrix X to Geometry")
    X = Geometry.X
    (N, d_in) = X.shape

    if eigen_solver not in ('auto', 'arpack', 'dense', 'amg', 'lobpcg'):
        raise ValueError("unrecognised eigen_solver '%s'" % eigen_solver)
    if n_components > d_in:
        raise ValueError("output dimension must be less than or equal "
                         "to input dimension")

    # get the distance matrix and neighbors list
    distance_matrix = Geometry.get_distance_matrix()
    (rows, cols) = distance_matrix.nonzero()

    M_sparse = (eigen_solver != 'dense')
    if M_sparse:
        M = sparse.csr_matrix((N, N))
    else:
        M = np.zeros((N, N))

    for i in range(N):
        neighbors_i = cols[rows == i]
        n_neighbors_i = len(neighbors_i)
        use_svd = (n_neighbors_i > d_in)
        Xi = X[neighbors_i]
        Xi -= Xi.mean(0)

        # compute n_components largest eigenvalues of Xi * Xi^T
        if use_svd:
            v = svd(Xi, full_matrices=True)[0]
        else:
            Ci = np.dot(Xi, Xi.T)
            v = eigh(Ci)[1][:, ::-1]

        Gi = np.zeros((n_neighbors_i, n_components + 1))
        Gi[:, 1:] = v[:, :n_components]
        Gi[:, 0] = 1. / np.sqrt(n_neighbors_i)
        GiGiT = np.dot(Gi, Gi.T)

        nbrs_x, nbrs_y = np.meshgrid(neighbors_i, neighbors_i)
        with warnings.catch_warnings():
            # sparse will complain this is better with lil_matrix but it doesn't work
            warnings.simplefilter("ignore")
            M[nbrs_x, nbrs_y] -= GiGiT
            M[neighbors_i, neighbors_i] += 1

    return null_space(M, n_components, k_skip=1, eigen_solver=eigen_solver,
                      tol=tol, max_iter=max_iter, random_state=random_state)

class LTSA():
    """
    Local Tangent Space Alignment

    Parameters
    ----------
    n_components : integer
        number of coordinates for the manifold.
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
    tol : float, optional
        Tolerance for 'arpack' method
        Not used if eigen_solver=='dense'.
    max_iter : integer
        maximum number of iterations for the arpack solver.
    random_state : numpy.RandomState or int, optional
        The generator or seed used to determine the starting vector for arpack
        iterations.  Defaults to numpy.random.
    neighborhood_radius : scalar, passed to distance_matrix. Value such that all
        distances beyond neighborhood_radius are considered infinite.
    affinity_radius : scalar, passed to affinity_matrix. 'bandwidth' parameter
        used in Guassian kernel for affinity matrix
    distance_method : string, one of 'auto', 'brute', 'cython', 'pyflann', 'cyflann'.
        method for computing pairwise radius neighbors graph.
    input_type : string, one of: 'data', 'distance', 'affinity'.
        The values of input data X.
    path_to_flann : string. full file path location of FLANN if not installed to
        root or to set FLANN_ROOT set to path location. Used for importing pyflann
        from a different location.
    Geometry : a Geometry object from megaman.geometry.geometry

    References
    ----------
    * Zhang, Z. & Zha, H. Principal manifolds and nonlinear
      dimensionality reduction via tangent space alignment.
      Journal of Shanghai Univ.  8:406 (2004)
    """
    def __init__(self, n_components=2, eigen_solver=None, random_state=None,
                 tol = 1e-6, max_iter=100, neighborhood_radius = None,
                 affinity_radius = None,  distance_method = 'auto',
                 input_type = 'data', path_to_flann = None, Geometry = None):
        # embedding parameters:
        self.n_components = n_components
        self.random_state = random_state
        self.eigen_solver = eigen_solver
        self.tol = tol
        self.max_iter = max_iter

        # Geometry parameters:
        self.Geometry = Geometry
        self.neighborhood_radius = neighborhood_radius
        self.affinity_radius = affinity_radius
        self.distance_method = distance_method
        self.input_type = input_type
        self.path_to_flann = path_to_flann

    def fit_geometry(self, X):
        self.Geometry = geom.Geometry(X, neighborhood_radius = self.neighborhood_radius,
                                      affinity_radius = self.affinity_radius,
                                      distance_method = self.distance_method,
                                      input_type = self.input_type,
                                      path_to_flann = self.path_to_flann)

    def fit(self, X, y=None):
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

        Returns
        -------
        self : object
            Returns the instance itself.
        """
        if not isinstance(self.Geometry, geom.Geometry):
            self.fit_geometry(X)
        random_state = check_random_state(self.random_state)
        (self.embedding_, self.error_) = ltsa(self.Geometry,n_components=self.n_components,
                                                eigen_solver=self.eigen_solver, tol = self.tol,
                                                random_state=random_state, max_iter = self.max_iter)
        return self

    def fit_transform(self, X, y=None):
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
