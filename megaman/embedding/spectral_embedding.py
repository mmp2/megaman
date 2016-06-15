"""Spectral Embedding"""

# Author: Marina Meila <mmp@stat.washington.edu>
#         James McQueen <jmcq@u.washington.edu>
# LICENSE: Simplified BSD https://github.com/mmp2/megaman/blob/master/LICENSE
#
#         after the scikit-learn version by
#         Gael Varoquaux <gael.varoquaux@normalesup.org>
#         Wei LI <kuantkid@gmail.com>
#
#         diffusion maps portion after:
#         Satrajit Ghosh <satra@mit.edu> https://github.com/satra/mapalign/blob/master/mapalign/embed.py
# License: BSD 3 clause

import warnings
import numpy as np
from scipy import sparse
from scipy.sparse.csgraph import connected_components
from ..embedding.base import BaseEmbedding
from ..utils.validation import check_random_state
from ..utils.eigendecomp import eigen_decomposition, check_eigen_solver


def _graph_connected_component(graph, node_id):
    """
    Find the largest graph connected components the contains one
    given node

    Parameters
    ----------
    graph : array-like, shape: (n_samples, n_samples)
        adjacency matrix of the graph, non-zero weight means an edge
        between the nodes

    node_id : int
        The index of the query node of the graph

    Returns
    -------
    connected_components : array-like, shape: (n_samples,)
        An array of bool value indicates the indexes of the nodes
        belong to the largest connected components of the given query
        node
    """
    connected_components = np.zeros(shape=(graph.shape[0]), dtype=np.bool)
    connected_components[node_id] = True
    n_node = graph.shape[0]
    for i in range(n_node):
        last_num_component = connected_components.sum()
        _, node_to_add = np.where(graph[connected_components] != 0)
        connected_components[node_to_add] = True
        if last_num_component >= connected_components.sum():
            break
    return connected_components


def _graph_is_connected(graph):
    """
    Return whether the graph is connected (True) or Not (False)

    Parameters
    ----------
    graph : array-like or sparse matrix, shape: (n_samples, n_samples)
        adjacency matrix of the graph, non-zero weight means an edge
        between the nodes

    Returns
    -------
    is_connected : bool
        True means the graph is fully connected and False means not
    """
    if sparse.isspmatrix(graph):
        # sparse graph, find all the connected components
        n_connected_components, _ = connected_components(graph)
        return n_connected_components == 1
    else:
        # dense graph, find all connected components start from node 0
        return _graph_connected_component(graph, 0).sum() == graph.shape[0]


def spectral_embedding(geom, n_components=8, eigen_solver='auto',
                       random_state=None, drop_first=True,
                       diffusion_maps = False, diffusion_time = 0, solver_kwds = None):
    """
    Project the sample on the first eigen vectors of the graph Laplacian.

    The adjacency matrix is used to compute a normalized graph Laplacian
    whose  principal eigenvectors (associated to the
    smallest eigen values) represent the embedding coordinates of the data.

    The ``adjacency`` variable is not strictly the adjacency matrix of a graph but more generally
    an affinity or similarity matrix between samples (for instance the
    heat kernel of a euclidean distance matrix or a k-NN matrix).
    The Laplacian must be symmetric so that the eigen vector decomposition works as expected. 
    This is ensured by the default setting (for more details,
    see the documentation in geometry.py).
    
    The data and generic geometric parameters are passed via a Geometry object, which also
    computes the Laplacian. By default, the 'geometric' Laplacian (or "debiased", or "renormalized" with
    alpha=1) is used. This is the Laplacian construction defined in [Coifman and Lafon, 2006] (see also 
    documentation in laplacian.py). Thus, with diffusion_maps=False, spectral embedding is a modification
    of the Laplacian Eigenmaps algorithm of [Belkin and Nyiogi, 2002], with diffusion_maps=False, geom.laplacian_method
    ='symmetricnormalized' it is exactly the Laplacian Eigenmaps, with diffusion_maps=True, diffusion_time>0 it
    is the Diffusion Maps algorithm of [Coifman and Lafon 2006]; diffusion_maps=True and diffusion_time=0 is the same
    as diffusion_maps=False and default geom.laplacian_method.

    Parameters
    ----------
    geom : a Geometry object from megaman.embedding.geometry
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
    drop_first : bool, optional, default=True
        Whether to drop the first eigenvector. For spectral embedding, this
        should be True as the first eigenvector should be constant vector for
        connected graph, but for spectral clustering, this should be kept as
        False to retain the first eigenvector.
    diffusion_map : boolean, optional. Whether to return the diffusion map
        version by re-scaling the embedding coordinate by the eigenvalues to the power 
        diffusion_time.
    diffusion_time: if diffusion_map=True, the eigenvectors of the Laplacian are rescaled by 
        (1-lambda)^diffusion_time, where lambda is the corresponding eigenvalue. 
        diffusion_time has the role of scale parameter. One of the main ideas of diffusion framework is
        that running the diffusion forward in time (taking larger and larger
        powers of the Laplacian/transition matrix) reveals the geometric structure of X at larger and
        larger scales (the diffusion process).
        diffusion_time = 0 empirically provides a reasonable balance from a clustering
        perspective. Specifically, the notion of a cluster in the data set
        is quantified as a region in which the probability of escaping this
        region is low (within a certain time t).
        Credit to Satrajit Ghosh (http://satra.cogitatum.org/) for description
    solver_kwds : any additional keyword arguments to pass to the selected eigen_solver

    Returns
    -------
    embedding : array, shape=(n_samples, n_components)
        The reduced samples.

    Notes
    -----
    Spectral embedding is most useful when the graph has one connected
    component. If there graph has many components, the first few eigenvectors
    will simply uncover the connected components of the graph.

    References
    ----------
    * http://en.wikipedia.org/wiki/LOBPCG

    * Toward the Optimal Preconditioned Eigensolver: Locally Optimal
      Block Preconditioned Conjugate Gradient Method
      Andrew V. Knyazev
      http://dx.doi.org/10.1137%2FS1064827500366124
    """
    random_state = check_random_state(random_state)

    if geom.affinity_matrix is None:
        geom.compute_affinity_matrix()
    if not _graph_is_connected(geom.affinity_matrix):
        warnings.warn("Graph is not fully connected: "
                      "spectral embedding may not work as expected.")

    if geom.laplacian_matrix is None:
        laplacian = geom.compute_laplacian_matrix(copy=False,
                                                  return_lapsym=True)
    else:
        laplacian = geom.laplacian_matrix

    n_nodes = laplacian.shape[0]
    lapl_type = geom.laplacian_method
    eigen_solver, solver_kwds = check_eigen_solver(eigen_solver,solver_kwds,
                                                   size=laplacian.shape[0],
                                                   nvec=n_components + 1)
    re_normalize = False
    PD_solver = False
    if eigen_solver in ['amg', 'lobpcg']: # these methods require a symmetric positive definite matrix!
        epsilon = 2
        PD_solver = True
        if lapl_type not in ['symmetricnormalized', 'unnormalized']:
            re_normalize = True
            # If lobpcg (or amg with lobpcg) is chosen and
            # If the Laplacian is non-symmetric then we need to extract:
            # the w (weight) vector from geometry
            # and the symmetric Laplacian = S.
            # The actual Laplacian is L = W^{-1}S  (Where W is the diagonal matrix of w)
            # Which has the same spectrum as: L* = W^{-1/2}SW^{-1/2} which is symmetric
            # We calculate the eigen-decomposition of L*: [D, V]
            # then use W^{-1/2}V  to compute the eigenvectors of L
            # See (Handbook for Cluster Analysis Chapter 2 Proposition 1).
            # However, since we censor the affinity matrix A at a radius it is not guaranteed
            # to be positive definite. But since L = W^{-1}S has maximum eigenvalue 1 (stochastic matrix)
            # and L* has the same spectrum it also has largest e-value of 1.
            # therefore if we look at I - L* then this has smallest eigenvalue of 0 and so
            # must be positive semi-definite. It also has the same spectrum as L* but
            # lambda(I - L*) = 1 - lambda(L*).
            # Finally, since we want positive definite not semi-definite we use (1+epsilon)*I
            # instead of I to make the smallest eigenvalue epsilon.
            if geom.laplacian_weights is None: # a laplacian existed but it wasn't called with return_lapsym = True
                geom.compute_laplacian_matrix(copy = False, return_lapsym = True)
            w = np.array(geom.laplacian_weights)
            symmetrized_laplacian = geom.laplacian_symmetric.copy()
            if sparse.isspmatrix(symmetrized_laplacian):
                symmetrized_laplacian.data /= np.sqrt(w[symmetrized_laplacian.row])
                symmetrized_laplacian.data /= np.sqrt(w[symmetrized_laplacian.col])
                symmetrized_laplacian = (1+epsilon)*sparse.identity(n_nodes) - symmetrized_laplacian
            else:
                symmetrized_laplacian /= np.sqrt(w)
                symmetrized_laplacian /= np.sqrt(w[:,np.newaxis])
                symmetrixed_laplacian = (1+epsilon)*np.identity(n_nodes) - symmetrized_laplacian
        else: # using a symmetric laplacian but adjust to avoid positive definite errors
            symmetrized_laplacian = geom.laplacian_matrix.copy()
            if sparse.isspmatrix(symmetrized_laplacian):
                symmetrized_laplacian = (1+epsilon)*sparse.identity(n_nodes) - symmetrized_laplacian
            else:
                symmetrixed_laplacian = (1+epsilon)*np.identity(n_nodes) - symmetrized_laplacian

    if PD_solver: # then eI - L was used, fix the eigenvalues
        lambdas, diffusion_map = eigen_decomposition(symmetrized_laplacian, n_components+1, eigen_solver=eigen_solver,
                                                     random_state=random_state, drop_first=drop_first, largest = False,
                                                     solver_kwds=solver_kwds)
        lambdas = -lambdas + epsilon
    else:
        lambdas, diffusion_map = eigen_decomposition(laplacian, n_components+1, eigen_solver=eigen_solver,
                                                     random_state=random_state, drop_first=drop_first, largest = True,
                                                     solver_kwds=solver_kwds)
    if re_normalize:
        diffusion_map /= np.sqrt(w[:, np.newaxis]) # put back on original Laplacian space
        diffusion_map /= np.linalg.norm(diffusion_map, axis = 0) # norm 1 vectors
    # sort the eigenvalues 
    ind = np.argsort(lambdas); ind = ind[::-1]
    lambdas = lambdas[ind]; lambdas[0] = 0
    diffusion_map = diffusion_map[:, ind]
    if diffusion_maps:
        """ Credit to Satrajit Ghosh (http://satra.cogitatum.org/) for final steps """
        # Check that diffusion maps is using the correct laplacian, warn otherwise
        if lapl_type not in ['geometric', 'renormalized']:
            warnings.warn("for correct diffusion maps embedding use laplacian type 'geometric' or 'renormalized'.")
        # Step 5 of diffusion maps:
        vectors = diffusion_map
        psi = vectors/vectors[:,[0]]
        diffusion_times = diffusion_time
        if diffusion_time == 0:
            lambdas = np.abs(lambdas)
            diffusion_times = np.exp(1. -  np.log(1 - lambdas[1:])/np.log(lambdas[1:]))
            lambdas = lambdas / (1 - lambdas)
        else:
            lambdas = np.abs(lambdas)
            lambdas = lambdas ** float(diffusion_time)
        diffusion_map = psi * lambdas
    if drop_first:
        embedding = diffusion_map[:, 1:(n_components+1)]
    else:
        embedding = diffusion_map[:, :n_components]
    return embedding


class SpectralEmbedding(BaseEmbedding):
    """
    Spectral embedding for non-linear dimensionality reduction.

    Forms an affinity matrix given by the specified function and
    applies spectral decomposition to the corresponding graph laplacian.
    The resulting transformation is given by the value of the
    eigenvectors for each data point.

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
    drop_first : bool, optional, default=True
        Whether to drop the first eigenvector. For spectral embedding, this
        should be True as the first eigenvector should be constant vector for
        connected graph, but for spectral clustering, this should be kept as
        False to retain the first eigenvector.
    diffusion_map : boolean, optional. Whether to return the diffusion map
        version by re-scaling the embedding by the eigenvalues.
    solver_kwds : any additional keyword arguments to pass to the selected eigen_solver

    References
    ----------
    .. [1] A Tutorial on Spectral Clustering, 2007
        Ulrike von Luxburg
        http://citeseerx.ist.psu.edu/viewdoc/summary?doi=10.1.1.165.9323

    .. [2] On Spectral Clustering: Analysis and an algorithm, 2011
        Andrew Y. Ng, Michael I. Jordan, Yair Weiss
        http://citeseerx.ist.psu.edu/viewdoc/summary?doi=10.1.1.19.8100

    .. [3] Normalized cuts and image segmentation, 2000
        Jianbo Shi, Jitendra Malik
        http://citeseer.ist.psu.edu/viewdoc/summary?doi=10.1.1.160.2324
    """
    def __init__(self, n_components=2, radius=None, geom=None,
                 eigen_solver='auto', random_state=None,
                 drop_first=True, diffusion_maps=False, diffusion_time=0,solver_kwds=None):
        self.n_components = n_components
        self.radius = radius
        self.geom = geom
        self.eigen_solver = eigen_solver
        self.random_state = random_state
        self.drop_first = drop_first
        self.diffusion_maps = diffusion_maps
        self.diffusion_time = diffusion_time
        self.solver_kwds = solver_kwds

    def fit(self, X, y=None, input_type='data'):
        """
        Fit the model from data in X.
        
        Parameters
        ----------
        input_type : string, one of: 'data', 'distance' or 'affinity'.
            The values of input data X. (default = 'data')
        X : array-like, shape (n_samples, n_features)
            Training vector, where n_samples in the number of samples
            and n_features is the number of features.

        If self.input_type is distance, or affinity:
        
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
        self.embedding_ = spectral_embedding(self.geom_,
                                             n_components = self.n_components,
                                             eigen_solver = self.eigen_solver,
                                             random_state = random_state,
                                             drop_first = self.drop_first,
                                             diffusion_maps = self.diffusion_maps,
                                             diffusion_time = self.diffusion_time,
                                             solver_kwds = self.solver_kwds)
        self.affinity_matrix_ = self.geom_.affinity_matrix
        self.laplacian_matrix_ = self.geom_.laplacian_matrix
        self.laplacian_matrix_type_ = self.geom_.laplacian_method
        return self
