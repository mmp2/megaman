"""Locally Linear Embedding"""

# Author: James McQueen -- <jmcq@u.washington.edu>
# LICENSE: Simplified BSD https://github.com/mmp2/megaman/blob/master/LICENSE
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
from ..embedding.base import BaseEmbedding
from ..utils.validation import check_array, check_random_state
from ..utils.eigendecomp import null_space, check_eigen_solver

def barycenter_graph(distance_matrix, X, reg=1e-3):
    """
    Computes the barycenter weighted graph for points in X

    Parameters
    ----------
    distance_matrix: sparse Ndarray, (N_obs, N_obs) pairwise distance matrix.
    X : Ndarray (N_obs, N_dim) observed data matrix.
    reg : float, optional
        Amount of regularization when solving the least-squares
        problem. Only relevant if mode='barycenter'. If None, use the
        default.

    Returns
    -------
    W : sparse matrix in CSR format, shape = [n_samples, n_samples]
        W[i, j] is assigned the weight of edge that connects i to j.
    """
    (N, d_in) = X.shape
    (rows, cols) = distance_matrix.nonzero()
    W = sparse.lil_matrix((N, N)) # best for W[i, nbrs_i] = w/np.sum(w)
    for i in range(N):
        nbrs_i = cols[rows == i]
        n_neighbors_i = len(nbrs_i)
        v = np.ones(n_neighbors_i, dtype=X.dtype)
        C = X[nbrs_i] - X[i]
        G = np.dot(C, C.T)
        trace = np.trace(G)
        if trace > 0:
            R = reg * trace
        else:
            R = reg
        G.flat[::n_neighbors_i + 1] += R
        w = solve(G, v, sym_pos = True)
        W[i, nbrs_i] = w / np.sum(w)
    return W


def locally_linear_embedding(geom, n_components, reg=1e-3, max_iter=100,
                            eigen_solver='auto', tol=1e-6,  random_state=None):
    """
    Perform a Locally Linear Embedding analysis on the data.

    Parameters
    ----------
    geom : a Geometry object from megaman.geometry.geometry
    n_components : integer
        number of coordinates for the manifold.
    reg : float
        regularization constant, multiplies the trace of the local covariance
        matrix of the distances.
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
    Y : array-like, shape [n_samples, n_components]
        Embedding vectors.
    squared_error : float
        Reconstruction error for the embedding vectors. Equivalent to
        ``norm(Y - W Y, 'fro')**2``, where W are the reconstruction weights.

    References
    ----------

    .. [1] Roweis, S. & Saul, L. Nonlinear dimensionality reduction
        by locally linear embedding.  Science 290:2323 (2000).
    """
    if geom.X is None:
        raise ValueError("Must pass data matrix X to Geometry")
    if geom.adjacency_matrix is None:
        geom.compute_adjacency_matrix()
    W = barycenter_graph(geom.adjacency_matrix, geom.X, reg=reg)
    # we'll compute M = (I-W)'(I-W)
    # depending on the solver, we'll do this differently
    eigen_solver = check_eigen_solver(eigen_solver,
                                      size=W.shape[0],
                                      nvec=n_components + 1)
    if eigen_solver != 'dense':
        M = eye(*W.shape, format=W.format) - W
        M = (M.T * M).tocsr()
    else:
        M = (W.T * W - W.T - W).toarray()
        M.flat[::M.shape[0] + 1] += 1  # W = W - I = W - I
    return null_space(M, n_components, k_skip=1, eigen_solver=eigen_solver,
                      tol=tol, max_iter=max_iter, random_state=random_state)


class LocallyLinearEmbedding(BaseEmbedding):
    """
    Locally Linear Embedding

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
    tol : float, optional
        Tolerance for 'arpack' method
        Not used if eigen_solver=='dense'.
    max_iter : integer, optional
        maximum number of iterations for the arpack solver.
    reg : float, optional
        regularization constant, multiplies the trace of the local covariance
        matrix of the distances.

    References
    ----------
    .. [1] Roweis, S. & Saul, L. Nonlinear dimensionality reduction
        by locally linear embedding.  Science 290:2323 (2000).
    """
    def __init__(self, n_components=2, radius=None, geom=None,
                 eigen_solver='auto', random_state=None,
                 tol=1e-6, max_iter=100, reg=1e3):
        self.n_components = n_components
        self.radius = radius
        self.geom = geom
        self.eigen_solver = eigen_solver
        self.random_state = random_state
        self.tol = tol
        self.max_iter = max_iter
        self.reg = reg

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

        Returns
        -------
        self : object
            Returns the instance itself.
        """
        X = self._validate_input(X, input_type)
        self.fit_geometry(X, input_type)
        random_state = check_random_state(self.random_state)
        self.embedding_, self.error_ = locally_linear_embedding(self.geom_,
                                                                n_components=self.n_components,
                                                                eigen_solver=self.eigen_solver,
                                                                tol=self.tol,
                                                                random_state=self.random_state,
                                                                reg=self.reg,
                                                                max_iter=self.max_iter)
        return self
