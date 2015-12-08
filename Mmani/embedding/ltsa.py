"""Local Tangent Space Alignment"""

# Author: James McQueen -- <jmcq@u.washington.edu>
#
#
# After the sci-kit learn version by:
#         Fabian Pedregosa -- <fabian.pedregosa@inria.fr>
#         Jake Vanderplas  -- <vanderplas@astro.washington.edu>
# License: BSD 3 clause (C) INRIA 2011

import numpy as np
from scipy.linalg import eigh, svd, qr, solve
from scipy.sparse import eye, csr_matrix
from Mmani.utils.validation import check_random_state, check_array
from Mmani.embedding.eigendecomp import null_space
import Mmani.embedding.geometry as geom
import scipy.sparse as sparse


def ltsa(Geometry, n_components, eigen_solver='auto', tol=1e-6,
        max_iter=100,random_state=None):
    """
        Perform a Locally Linear Embedding analysis on the data.
        Read more in the :ref:`User Guide <locally_linear_embedding>`.
        Parameters
        ----------
        X : {array-like, sparse matrix, BallTree, KDTree, NearestNeighbors}
            Sample data, shape = (n_samples, n_features), in the form of a
            numpy array, sparse array, precomputed tree, or NearestNeighbors
            object.
        n_components : integer
            number of coordinates for the manifold.
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
        .. [1] `Zhang, Z. & Zha, H. Principal manifolds and nonlinear
            dimensionality reduction via tangent space alignment.
            Journal of Shanghai Univ.  8:406 (2004)`
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
        M[nbrs_x, nbrs_y] -= GiGiT
        M[neighbors_i, neighbors_i] += 1

    return null_space(M, n_components, k_skip=1, eigen_solver=eigen_solver,
                      tol=tol, max_iter=max_iter, random_state=random_state)

class LTSA():
    def __init__(self, radius=None, n_components=2,
                 eigen_solver='auto', tol=1E-6, max_iter=100,
                 random_state=None):
        # update parameters 
        self.radius = radius
        self.n_components = n_components
        self.eigen_solver = eigen_solver
        self.tol = tol
        self.max_iter = max_iter
        self.random_state = random_state
        # GEOMETRY ARGUMENTS

    def _fit_transform(self, X):
        self.Geometry = geom.Geometry(X) # add other arguments
        random_state = check_random_state(self.random_state)
        X = check_array(X)
        self.nbrs_.fit(X)
        self.embedding_, self.reconstruction_error_ = \
                ltsa(
                self.Geometry,self.n_components, maxiter = self.max_iter,
                eigen_solver=self.eigen_solver, tol=self.tol,
                random_state=random_state)

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