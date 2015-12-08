"""Locally Linear Embedding"""

# Author: James McQueen -- <jmcq@u.washington.edu>
#
#
# After the sci-kit learn version by:
#         Fabian Pedregosa -- <fabian.pedregosa@inria.fr>
#         Jake Vanderplas  -- <vanderplas@astro.washington.edu>
# License: BSD 3 clause (C) INRIA 2011

import numpy as np
import scipy.sparse as sparse
from scipy.linalg import eigh, svd, qr, solve
from scipy.sparse import eye, csr_matrix
from Mmani.utils.validation import check_random_state, check_array
from Mmani.embedding.eigendecomp import null_space

def barycenter_graph(distance_matrix, X, reg=1e-3):
    """
        Computes the barycenter weighted graph for points in X
        Parameters
        ----------
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
    W = sparse.csr_matrix((N, N))
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

def locally_linear_embedding(Geometry, n_components, reg=1e-3, max_iter=100,
                            eigen_solver='auto', tol=1e-6,  random_state=None):
    """
        Perform a Locally Linear Embedding analysis on the data.
        Read more in the :ref:`User Guide <locally_linear_embedding>`.
        Parameters
        ----------
        X : {array-like, sparse matrix, BallTree, KDTree, NearestNeighbors}
            Sample data, shape = (n_samples, n_features), in the form of a
            numpy array, sparse array, precomputed tree, or NearestNeighbors
            object.
        n_neighbors : integer
            number of neighbors to consider for each point.
        n_components : integer
            number of coordinates for the manifold.
        reg : float
            regularization constant, multiplies the trace of the local covariance
            matrix of the distances.
        eigen_solver : string, {'auto', 'arpack', 'dense'}
            auto : algorithm will attempt to choose the best method for input data
            arpack : use arnoldi iteration in shift-invert mode.
                        For this method, M may be a dense matrix, sparse matrix,
                        or general linear operator.
                        Warning: ARPACK can be unstable for some problems.  It is
                        best to try several random seeds in order to check results.
            dense  : use standard dense matrix operations for the eigenvalue
                        decomposition.  For this method, M must be an array
                        or matrix type.  This method should be avoided for
                        large problems.
        tol : float, optional
            Tolerance for 'arpack' method
            Not used if eigen_solver=='dense'.
        max_iter : integer
            maximum number of iterations for the arpack solver.
        random_state: numpy.RandomState or int, optional
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
        .. [1] `Roweis, S. & Saul, L. Nonlinear dimensionality reduction
            by locally linear embedding.  Science 290:2323 (2000).`
        .. [2] `Donoho, D. & Grimes, C. Hessian eigenmaps: Locally
            linear embedding techniques for high-dimensional data.
            Proc Natl Acad Sci U S A.  100:5591 (2003).`
        .. [3] `Zhang, Z. & Wang, J. MLLE: Modified Locally Linear
            Embedding Using Multiple Weights.`
            http://citeseerx.ist.psu.edu/viewdoc/summary?doi=10.1.1.70.382
        .. [4] `Zhang, Z. & Zha, H. Principal manifolds and nonlinear
            dimensionality reduction via tangent space alignment.
            Journal of Shanghai Univ.  8:406 (2004)`
    """
    if eigen_solver not in ('auto', 'arpack', 'dense', 'amg', 'lobpcg'):
        raise ValueError("unrecognized eigen_solver '%s'" % eigen_solver)
    if Geometry.X is None:
        raise ValueError("Must pass data matrix X to Geometry")
    X = Geometry.X    
    M_sparse = (eigen_solver != 'dense')
    W = barycenter_graph(Geometry.get_distance_matrix(), X, reg=reg)
    # we'll compute M = (I-W)'(I-W)
    # depending on the solver, we'll do this differently
    if M_sparse:
        M = eye(*W.shape, format=W.format) - W
        M = (M.T * M).tocsr()
    else:
        M = (W.T * W - W.T - W).toarray()
        M.flat[::M.shape[0] + 1] += 1  # W = W - I = W - I
    return null_space(M, n_components, k_skip=1, eigen_solver=eigen_solver,
                      tol=tol, max_iter=max_iter, random_state=random_state)

class LocallyLinearEmbedding():
    """
        Locally Linear Embedding
        Read more in the :ref:`User Guide <locally_linear_embedding>`.
        Parameters
        ----------
        n_neighbors : integer
            number of neighbors to consider for each point.
        n_components : integer
            number of coordinates for the manifold
        reg : float
            regularization constant, multiplies the trace of the local covariance
            matrix of the distances.
        eigen_solver : string, {'auto', 'arpack', 'dense'}
            auto : algorithm will attempt to choose the best method for input data
            arpack : use arnoldi iteration in shift-invert mode.
                        For this method, M may be a dense matrix, sparse matrix,
                        or general linear operator.
                        Warning: ARPACK can be unstable for some problems.  It is
                        best to try several random seeds in order to check results.
            dense  : use standard dense matrix operations for the eigenvalue
                        decomposition.  For this method, M must be an array
                        or matrix type.  This method should be avoided for
                        large problems.
        tol : float, optional
            Tolerance for 'arpack' method
            Not used if eigen_solver=='dense'.
        max_iter : integer
            maximum number of iterations for the arpack solver.
            Not used if eigen_solver=='dense'.
        method : string ('standard', 'hessian', 'modified' or 'ltsa')
            standard : use the standard locally linear embedding algorithm.  see
                       reference [1]
            hessian  : use the Hessian eigenmap method. This method requires
                       ``n_neighbors > n_components * (1 + (n_components + 1) / 2``
                       see reference [2]
            modified : use the modified locally linear embedding algorithm.
                       see reference [3]
            ltsa     : use local tangent space alignment algorithm
                       see reference [4]
        hessian_tol : float, optional
            Tolerance for Hessian eigenmapping method.
            Only used if ``method == 'hessian'``
        modified_tol : float, optional
            Tolerance for modified LLE method.
            Only used if ``method == 'modified'``
        neighbors_algorithm : string ['auto'|'brute'|'kd_tree'|'ball_tree']
            algorithm to use for nearest neighbors search,
            passed to neighbors.NearestNeighbors instance
        random_state: numpy.RandomState or int, optional
            The generator or seed used to determine the starting vector for arpack
            iterations.  Defaults to numpy.random.
        Attributes
        ----------
        embedding_vectors_ : array-like, shape [n_components, n_samples]
            Stores the embedding vectors
        reconstruction_error_ : float
            Reconstruction error associated with `embedding_vectors_`
        nbrs_ : NearestNeighbors object
            Stores nearest neighbors instance, including BallTree or KDtree
            if applicable.
        References
        ----------
        .. [1] `Roweis, S. & Saul, L. Nonlinear dimensionality reduction
            by locally linear embedding.  Science 290:2323 (2000).`
        .. [2] `Donoho, D. & Grimes, C. Hessian eigenmaps: Locally
            linear embedding techniques for high-dimensional data.
            Proc Natl Acad Sci U S A.  100:5591 (2003).`
        .. [3] `Zhang, Z. & Wang, J. MLLE: Modified Locally Linear
            Embedding Using Multiple Weights.`
            http://citeseerx.ist.psu.edu/viewdoc/summary?doi=10.1.1.70.382
        .. [4] `Zhang, Z. & Zha, H. Principal manifolds and nonlinear
            dimensionality reduction via tangent space alignment.
            Journal of Shanghai Univ.  8:406 (2004)`
    """

    def __init__(self, n_neighbors=5, n_components=2, reg=1E-3,
                 eigen_solver='auto', tol=1E-6, max_iter=100,
                 method='standard', hessian_tol=1E-4, modified_tol=1E-12,
                 neighbors_algorithm='auto', random_state=None):

        self.n_neighbors = n_neighbors
        self.n_components = n_components
        self.reg = reg
        self.eigen_solver = eigen_solver
        self.tol = tol
        self.max_iter = max_iter
        self.method = method
        self.hessian_tol = hessian_tol
        self.modified_tol = modified_tol
        self.random_state = random_state
        self.neighbors_algorithm = neighbors_algorithm

    def _fit_transform(self, X):
        self.nbrs_ = NearestNeighbors(self.n_neighbors,
                                      algorithm=self.neighbors_algorithm)

        random_state = check_random_state(self.random_state)
        X = check_array(X)
        self.nbrs_.fit(X)
        self.embedding_, self.reconstruction_error_ = \
            locally_linear_embedding(
                self.nbrs_, self.n_neighbors, self.n_components,
                eigen_solver=self.eigen_solver, tol=self.tol,
                max_iter=self.max_iter, method=self.method,
                hessian_tol=self.hessian_tol, modified_tol=self.modified_tol,
                random_state=random_state, reg=self.reg)

    def fit(self, X, y=None):
        """Compute the embedding vectors for data X
        Parameters
        ----------
        X : array-like of shape [n_samples, n_features]
            training set.
        Returns
        -------
        self : returns an instance of self.
        """
        self._fit_transform(X)
        return self

    def fit_transform(self, X, y=None):
        """Compute the embedding vectors for data X and transform X.
        Parameters
        ----------
        X : array-like of shape [n_samples, n_features]
            training set.
        Returns
        -------
        X_new: array-like, shape (n_samples, n_components)
        """
        self._fit_transform(X)
        return self.embedding_