import warnings
import numpy as np
from scipy import sparse
from scipy.linalg import eigh, eig
from scipy.sparse.linalg import lobpcg, eigs, eigsh

from .validation import check_random_state
from .validation import check_array

eigen_solvers = ['auto', 'dense', 'arpack', 'lobpcg', 'amg']

def _is_symmetric(M, tol = 1e-8):
    if sparse.isspmatrix(M):
        conditions = np.abs((M - M.T).data) < tol
    else:
        conditions = np.abs((M - M.T)) < tol
    return(np.all(conditions))

def eigen_decomposition(G, n_components=8, eigen_solver=None,
                       random_state=None, eigen_tol=0.0,
                       drop_first=True, largest = True):
    """
    Function to compute the eigendecomposition of a square matrix.

    Parameters
    ----------
    G : 2d numpy/scipy array. Potentially sparse. The matrix to find the eigendecomposition of
    n_components : integer, optional
        The number of eigenvectors to return
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
        Stopping criterion for eigendecomposition when using arpack eigen_solver

    Returns
    -------
    lambdas, diffusion_map : eigenvalues, eigenvectors
    """

    n_nodes = G.shape[0]
    if eigen_solver is None:
        eigen_solver = 'auto'
    elif not eigen_solver in eigen_solvers:
        raise ValueError("Unknown value for eigen_solver: '%s'."
                         "Should be: '%s'"
                         % eigen_solver, eigen_solvers)
    if eigen_solver == 'auto':
        if G.shape[0] > 200:
            eigen_solver = 'arpack'
        else:
            eigen_solver = 'dense'

    # Check eigen_solver method
    try:
        from pyamg import smoothed_aggregation_solver
    except ImportError:
        if eigen_solver == "amg":
            raise ValueError("The eigen_solver was set to 'amg', but pyamg is "
                             "not available.")
    # Check input values
    if not isinstance(largest, bool):
        raise ValueError("largest should be True if you want largest eigenvalues otherwise False")
    random_state = check_random_state(random_state)
    if drop_first:
        n_components = n_components + 1
    # Check for symmetry
    is_symmetric = _is_symmetric(G)
    # Convert G to best type for eigendecomposition
    if sparse.issparse(G):
        if G.getformat() is not 'csr':
            G.tocsr()
    G = G.astype(np.float)

    if ((eigen_solver == 'lobpcg') and (n_nodes < 5 * n_components + 1)):
        warnings.warn("lobpcg has problems with small number of nodes. Using dense eigh")
        eigen_solver = 'dense'

    # Try Eigen Methods:
    if eigen_solver == 'arpack':
        if is_symmetric:
            if largest:
                which = 'LM'
            else:
                which = 'SM'
            lambdas, diffusion_map = eigsh(G, k=n_components, which=which,tol=eigen_tol)
        else:
            if largest:
                which = 'LR'
            else:
                which = 'SR'
            lambdas, diffusion_map = eigs(G, k=n_components, which=which,tol=eigen_tol)
        lambdas = np.real(lambdas)
        diffusion_map = np.real(diffusion_map)
    elif eigen_solver == 'amg':
        if not is_symmetric:
            raise ValueError("lobpcg requires symmetric matrices.")
        if not sparse.issparse(G):
            warnings.warn("AMG works better for sparse matrices")
        # Use AMG to get a preconditioner and speed up the eigenvalue problem.
        ml = smoothed_aggregation_solver(check_array(G, accept_sparse = ['csr']))
        M = ml.aspreconditioner()
        n_find = min(n_nodes, 5 + 2*n_components)
        X = random_state.rand(n_nodes, n_find)
        X[:, 0] = (G.diagonal()).ravel()
        lambdas, diffusion_map = lobpcg(G, X, M=M, largest=largest)
        sort_order = np.argsort(lambdas)
        if largest:
            lambdas = lambdas[sort_order[::-1]]
            diffusion_map = diffusion_map[:, sort_order[::-1]]
        else:
            lambdas = lambdas[sort_order]
            diffusion_map = diffusion_map[:, sort_order]
        lambdas = lambdas[:n_components]
        diffusion_map = diffusion_map[:, :n_components]
    elif eigen_solver == "lobpcg":
        if not is_symmetric:
            raise ValueError("lobpcg requires symmetric matrices.")
        n_find = min(n_nodes, 5 + 2*n_components)
        X = random_state.rand(n_nodes, n_find)
        lambdas, diffusion_map = lobpcg(G, X, largest=largest)
        sort_order = np.argsort(lambdas)
        if largest:
            lambdas = lambdas[sort_order[::-1]]
            diffusion_map = diffusion_map[:, sort_order[::-1]]
        else:
            lambdas = lambdas[sort_order]
            diffusion_map = diffusion_map[:, sort_order]
        lambdas = lambdas[:n_components]
        diffusion_map = diffusion_map[:, :n_components]
    elif eigen_solver == 'dense':
        if sparse.isspmatrix(G):
            G = G.todense()
        if is_symmetric:
            lambdas, diffusion_map = eigh(G)
        else:
            lambdas, diffusion_map = eig(G)
        if largest:# eigh always returns eigenvalues in ascending order
            lambdas = lambdas[::-1] # reverse order the e-values
            diffusion_map = diffusion_map[:, ::-1] # reverse order the vectors
        lambdas = lambdas[:n_components]
        diffusion_map = diffusion_map[:, :n_components]
    return (lambdas, diffusion_map)

def null_space(M, k, k_skip=1, eigen_solver='arpack', tol=1E-6, max_iter=100,
               random_state=None):
    """
    Find the null space of a matrix M: eigenvectors associated with 0 eigenvalues

    Parameters
    ----------
    M : {array, matrix, sparse matrix, LinearOperator}
        Input covariance matrix: should be symmetric positive semi-definite
    k : integer
        Number of eigenvalues/vectors to return
    k_skip : integer, optional
        Number of low eigenvalues to skip.
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
        Tolerance for 'arpack' method.
        Not used if eigen_solver=='dense'.
    max_iter : maximum number of iterations for 'arpack' method
        not used if eigen_solver=='dense'
    random_state: numpy.RandomState or int, optional
        The generator or seed used to determine the starting vector for arpack
        iterations.  Defaults to numpy.random.

    Returns
    -------
    null_space : estimated k vectors of the null space
    error : estimated error (sum of eigenvalues)
    """

    if eigen_solver == 'auto':
        if M.shape[0] > 200 and k + k_skip < 10:
            eigen_solver = 'arpack'
        else:
            eigen_solver = 'dense'

    if eigen_solver == 'arpack':
        random_state = check_random_state(random_state)
        v0 = random_state.rand(M.shape[0])
        try:
            eigen_values, eigen_vectors = eigsh(M, k + k_skip, sigma=0.0,
                                                tol=tol, maxiter=max_iter,
                                                v0=v0)
        except RuntimeError as msg:
            raise ValueError("Error in determining null-space with ARPACK. "
                             "Error message: '%s'. "
                             "Note that method='arpack' can fail when the "
                             "weight matrix is singular or otherwise "
                             "ill-behaved.  method='dense' is recommended. "
                             "See online documentation for more information."
                             % msg)

        return eigen_vectors[:, k_skip:], np.sum(eigen_values[k_skip:])
    elif eigen_solver == 'dense':
        if hasattr(M, 'toarray'):
            M = M.toarray()
        eigen_values, eigen_vectors = eigh(M, eigvals=(0, k+k_skip),overwrite_a=True)
        index = np.argsort(np.abs(eigen_values))
        eigen_vectors = eigen_vectors[:, index]
        eigen_values = eigen_values[index]
        return eigen_vectors[:, k_skip:k+1], np.sum(eigen_values[k_skip:k+1])
        # eigen_values, eigen_vectors = eigh(
            # M, eigvals=(k_skip, k + k_skip - 1), overwrite_a=True)
        # index = np.argsort(np.abs(eigen_values))
        # return eigen_vectors[:, index], np.sum(eigen_values)
    elif (eigen_solver == 'amg' or eigen_solver == 'lobpcg'):
        # M should be positive semi-definite. Add 1 to make it pos. def.
        try:
            M = sparse.identity(M.shape[0]) + M
            n_components = min(k + k_skip + 10, M.shape[0])
            eigen_values, eigen_vectors = eigen_decomposition(M, n_components,
                                                              eigen_solver = eigen_solver,
                                                              drop_first = False,
                                                              largest = False)
            eigen_values = eigen_values -1
            index = np.argsort(np.abs(eigen_values))
            eigen_values = eigen_values[index]
            eigen_vectors = eigen_vectors[:, index]
            return eigen_vectors[:, k_skip:k+1], np.sum(eigen_values[k_skip:k+1])
        except np.linalg.LinAlgError: # try again with bigger increase
            warnings.warn("LOBPCG failed the first time. Increasing Pos Def adjustment.")
            M = 2.0*sparse.identity(M.shape[0]) + M
            n_components = min(k + k_skip + 10, M.shape[0])
            eigen_values, eigen_vectors = eigen_decomposition(M, n_components,
                                                              eigen_solver = eigen_solver,
                                                              drop_first = False,
                                                              largest = False)
            eigen_values = eigen_values - 2
            index = np.argsort(np.abs(eigen_values))
            eigen_values = eigen_values[index]
            eigen_vectors = eigen_vectors[:, index]
            return eigen_vectors[:, k_skip:k+1], np.sum(eigen_values[k_skip:k+1])
    else:
        raise ValueError("Unrecognized eigen_solver '%s'" % eigen_solver)
