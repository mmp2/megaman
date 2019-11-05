"""Local Tangent Space Alignment"""

# Author: James McQueen -- <jmcq@u.washington.edu>
# LICENSE: Simplified BSD https://github.com/mmp2/megaman/blob/master/LICENSE
#
#
# After the sci-kit learn version by:
#         Fabian Pedregosa -- <fabian.pedregosa@inria.fr>
#         Jake Vanderplas  -- <vanderplas@astro.washington.edu>
# License: BSD 3 clause (C) INRIA 2011

from __future__ import division
import warnings
import numpy as np
import scipy.sparse as sparse
from scipy.linalg import eigh, svd, qr, solve
from scipy.sparse import eye, csr_matrix

from ..embedding.base import BaseEmbedding
from ..utils.validation import check_random_state, check_array
from ..utils.eigendecomp import null_space, check_eigen_solver


def ltsa(geom, n_components, eigen_solver='auto',
         random_state=None, solver_kwds=None):
    """
    Perform a Local Tangent Space Alignment analysis on the data.

    Parameters
    ----------
    geom : a Geometry object from megaman.geometry.geometry
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
    random_state : numpy.RandomState or int, optional
        The generator or seed used to determine the starting vector for arpack
        iterations.  Defaults to numpy.random.
    solver_kwds : any additional keyword arguments to pass to the selected eigen_solver

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
    if geom.X is None:
        raise ValueError("Must pass data matrix X to Geometry")
    (N, d_in) = geom.X.shape
    if n_components > d_in:
        raise ValueError("output dimension must be less than or equal "
                         "to input dimension")
    # get the distance matrix and neighbors list
    if geom.adjacency_matrix is None:
        geom.compute_adjacency_matrix()
    (rows, cols) = geom.adjacency_matrix.nonzero()
    eigen_solver, solver_kwds = check_eigen_solver(eigen_solver, solver_kwds,
                                                   size=geom.adjacency_matrix.shape[0],
                                                   nvec=n_components + 1)
    if eigen_solver != 'dense':
        M = sparse.dok_matrix((N, N))
    else:
        M = np.zeros((N, N))
    for i in range(N):
        neighbors_i = cols[rows == i]
        n_neighbors_i = len(neighbors_i)
        use_svd = (n_neighbors_i > d_in)
        Xi = geom.X[neighbors_i]
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
        # with warnings.catch_warnings():
        #     # sparse will complain this is better with lil_matrix but it doesn't work
        #     warnings.simplefilter("ignore")
        M[nbrs_x, nbrs_y] -= GiGiT
        M[neighbors_i, neighbors_i] += 1
    M = sparse.csr_matrix(M)
    return null_space(M, n_components, k_skip=1, eigen_solver=eigen_solver,
                      random_state=random_state,solver_kwds=solver_kwds)


class LTSA(BaseEmbedding):
    """
    Local Tangent Space Alignment

    Parameters
    ----------
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
    solver_kwds : any additional keyword arguments to pass to the selected eigen_solver

    References
    ----------
    .. [1] Zhang, Z. & Zha, H. Principal manifolds and nonlinear
        dimensionality reduction via tangent space alignment.
        Journal of Shanghai Univ.  8:406 (2004)
    """
    def __init__(self, n_components=2, radius=None, geom=None,
                 eigen_solver='auto', random_state=None,
                 tol=1e-6, max_iter=100, solver_kwds=None):
        self.n_components = n_components
        self.radius = radius
        self.geom = geom
        self.eigen_solver = eigen_solver
        self.random_state = random_state
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

        If self.input_type is 'distance', or 'affinity':

        X : array-like, shape (n_samples, n_samples),
            Interpret X as precomputed distance or adjacency graph
            computed from samples.

        Returns
        -------
        self : object
            Returns the instance itself.
        """
        X = self._validate_input(X, input_type)
        self.fit_geometry(X, input_type)
        random_state = check_random_state(self.random_state)
        (self.embedding_, self.error_) = ltsa(self.geom_,
                                              n_components=self.n_components,
                                              eigen_solver=self.eigen_solver,
                                              random_state=random_state,
                                              solver_kwds = self.solver_kwds)
        return self
