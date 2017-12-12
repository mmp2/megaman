# LICENSE: Simplified BSD https://github.com/mmp2/megaman/blob/master/LICENSE

import warnings
import numpy as np
from scipy import sparse
from scipy.linalg import eigh, eig
from scipy.sparse.linalg import lobpcg, eigs, eigsh
from sklearn.utils.validation import check_random_state

from .validation import check_array


EIGEN_SOLVERS = ['auto', 'dense', 'arpack', 'lobpcg']
BAD_EIGEN_SOLVERS = {}
AMG_KWDS = ['strength', 'aggregate', 'smooth', 'max_levels', 'max_coarse']

try:
    from pyamg import smoothed_aggregation_solver
    PYAMG_LOADED = True
    EIGEN_SOLVERS.append('amg')
except ImportError:
    PYAMG_LOADED = False
    BAD_EIGEN_SOLVERS['amg'] = """The eigen_solver was set to 'amg',
    but pyamg is not available. Please either
    install pyamg or use another method."""


def check_eigen_solver(eigen_solver, solver_kwds, size=None, nvec=None):
    """Check that the selected eigensolver is valid

    Parameters
    ----------
    eigen_solver : string
        string value to validate
    size, nvec : int (optional)
        if both provided, use the specified problem size and number of vectors
        to determine the optimal method to use with eigen_solver='auto'

    Returns
    -------
    eigen_solver : string
        The eigen solver. This only differs from the input if
        eigen_solver == 'auto' and `size` is specified.
    """
    if eigen_solver in BAD_EIGEN_SOLVERS:
        raise ValueError(BAD_EIGEN_SOLVERS[eigen_solver])
    elif eigen_solver not in EIGEN_SOLVERS:
        raise ValueError("Unrecognized eigen_solver: '{0}'."
                         "Should be one of: {1}".format(eigen_solver,
                                                        EIGEN_SOLVERS))

    if size is not None and nvec is not None:
        # do some checks of the eigensolver
        if eigen_solver == 'lobpcg' and size < 5 * nvec + 1:
            warnings.warn("lobpcg does not perform well with small matrices or "
                          "with large numbers of vectors. Switching to 'dense'")
            eigen_solver = 'dense'
            solver_kwds = None

        elif eigen_solver == 'auto':
            if size > 200 and nvec < 10:
                if PYAMG_LOADED:
                    eigen_solver = 'amg'
                    solver_kwds = None
                else:
                    eigen_solver = 'arpack'
                    solver_kwds = None
            else:
                eigen_solver = 'dense'
                solver_kwds = None

    return eigen_solver, solver_kwds


def _is_symmetric(M, tol = 1e-8):
    if sparse.isspmatrix(M):
        conditions = np.abs((M - M.T).data) < tol
    else:
        conditions = np.abs((M - M.T)) < tol
    return(np.all(conditions))


def eigen_decomposition(G, n_components=8, eigen_solver='auto',
                        random_state=None,
                        drop_first=True, largest=True, solver_kwds=None):
    """
    Function to compute the eigendecomposition of a square matrix.

    Parameters
    ----------
    G : array_like or sparse matrix
        The square matrix for which to compute the eigen-decomposition.
    n_components : integer, optional
        The number of eigenvectors to return
    eigen_solver : {'auto', 'dense', 'arpack', 'lobpcg', or 'amg'}
        'auto' :
            attempt to choose the best method for input data (default)
        'dense' :
            use standard dense matrix operations for the eigenvalue decomposition.
            For this method, M must be an array or matrix type.
            This method should be avoided for large problems.
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
            Algebraic Multigrid solver (requires ``pyamg`` to be installed)
            It can be faster on very large, sparse problems, but may also lead
            to instabilities.
    random_state : int seed, RandomState instance, or None (default)
        A pseudo random number generator used for the initialization of the
        lobpcg eigen vectors decomposition when eigen_solver == 'amg'.
        By default, arpack is used.
    solver_kwds : any additional keyword arguments to pass to the selected eigen_solver

    Returns
    -------
    lambdas, diffusion_map : eigenvalues, eigenvectors
    """
    n_nodes = G.shape[0]
    if drop_first:
        n_components = n_components + 1

    eigen_solver, solver_kwds = check_eigen_solver(eigen_solver, solver_kwds,
                                                   size=n_nodes,
                                                   nvec=n_components)
    random_state = check_random_state(random_state)

    # Convert G to best type for eigendecomposition
    if sparse.issparse(G):
        if G.getformat() is not 'csr':
            G.tocsr()
    G = G.astype(np.float)

    # Check for symmetry
    is_symmetric = _is_symmetric(G)

    # Try Eigen Methods:
    if eigen_solver == 'arpack':
        # This matches the internal initial state used by ARPACK
        v0 = random_state.uniform(-1, 1, G.shape[0])
        if is_symmetric:
            if largest:
                which = 'LM'
            else:
                which = 'SM'
            lambdas, diffusion_map = eigsh(G, k=n_components, which=which,
                                           v0=v0,**(solver_kwds or {}))
        else:
            if largest:
                which = 'LR'
            else:
                which = 'SR'
            lambdas, diffusion_map = eigs(G, k=n_components, which=which,
                                          **(solver_kwds or {}))
        lambdas = np.real(lambdas)
        diffusion_map = np.real(diffusion_map)
    elif eigen_solver == 'amg':
        # separate amg & lobpcg keywords:
        if solver_kwds is not None:
            amg_kwds = {}
            lobpcg_kwds = solver_kwds.copy()
            for kwd in AMG_KWDS:
                if kwd in solver_kwds.keys():
                    amg_kwds[kwd] = solver_kwds[kwd]
                    del lobpcg_kwds[kwd]
        else:
            amg_kwds = None
            lobpcg_kwds = None
        if not is_symmetric:
            raise ValueError("lobpcg requires symmetric matrices.")
        if not sparse.issparse(G):
            warnings.warn("AMG works better for sparse matrices")
        # Use AMG to get a preconditioner and speed up the eigenvalue problem.
        ml = smoothed_aggregation_solver(check_array(G, accept_sparse = ['csr']),**(amg_kwds or {}))
        M = ml.aspreconditioner()
        n_find = min(n_nodes, 5 + 2*n_components)
        X = random_state.rand(n_nodes, n_find)
        X[:, 0] = (G.diagonal()).ravel()
        lambdas, diffusion_map = lobpcg(G, X, M=M, largest=largest,**(lobpcg_kwds or {}))
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
        lambdas, diffusion_map = lobpcg(G, X, largest=largest,**(solver_kwds or {}))
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
            lambdas, diffusion_map = eigh(G,**(solver_kwds or {}))
        else:
            lambdas, diffusion_map = eig(G,**(solver_kwds or {}))
            sort_index = np.argsort(lambdas)
            lambdas = lambdas[sort_index]
            diffusion_map[:,sort_index]
        if largest:# eigh always returns eigenvalues in ascending order
            lambdas = lambdas[::-1] # reverse order the e-values
            diffusion_map = diffusion_map[:, ::-1] # reverse order the vectors
        lambdas = lambdas[:n_components]
        diffusion_map = diffusion_map[:, :n_components]
    return (lambdas, diffusion_map)


def null_space(M, k, k_skip=1, eigen_solver='arpack',
               random_state=None, solver_kwds=None):
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
    random_state: numpy.RandomState or int, optional
        The generator or seed used to determine the starting vector for arpack
        iterations.  Defaults to numpy.random.
    solver_kwds : any additional keyword arguments to pass to the selected eigen_solver

    Returns
    -------
    null_space : estimated k vectors of the null space
    error : estimated error (sum of eigenvalues)

    Notes
    -----
    dense solver key words: see
        http://docs.scipy.org/doc/scipy-0.14.0/reference/generated/scipy.linalg.eigh.html
        for symmetric problems and
        http://docs.scipy.org/doc/scipy-0.14.0/reference/generated/scipy.linalg.eig.html#scipy.linalg.eig
        for non symmetric problems.
    arpack sovler key words: see
        http://docs.scipy.org/doc/scipy-0.14.0/reference/generated/scipy.sparse.linalg.eigsh.html
        for symmetric problems and http://docs.scipy.org/doc/scipy-0.14.0/reference/generated/scipy.sparse.linalg.eigs.html#scipy.sparse.linalg.eigs
        for non symmetric problems.
    lobpcg solver keywords: see
        http://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.linalg.lobpcg.html
    amg solver keywords: see
        http://pyamg.googlecode.com/svn/branches/1.0.x/Docs/html/pyamg.aggregation.html#module-pyamg.aggregation.aggregation
        (Note amg solver uses lobpcg and also accepts lobpcg keywords)
    """
    eigen_solver, solver_kwds = check_eigen_solver(eigen_solver, solver_kwds,
                                                   size=M.shape[0],
                                                   nvec=k + k_skip)
    random_state = check_random_state(random_state)

    if eigen_solver == 'arpack':
        # This matches the internal initial state used by ARPACK
        v0 = random_state.uniform(-1, 1, M.shape[0])
        try:
            eigen_values, eigen_vectors = eigsh(M, k + k_skip, sigma=0.0,
                                                v0=v0,**(solver_kwds or {}))
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
        eigen_values, eigen_vectors = eigh(M, eigvals=(0, k+k_skip),overwrite_a=True,
                                           **(solver_kwds or {}))
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
                                                              largest = False,
                                                              random_state=random_state,
                                                              solver_kwds=solver_kwds)
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
                                                              largest = False,
                                                              random_state=random_state,
                                                              solver_kwds=solver_kwds)
            eigen_values = eigen_values - 2
            index = np.argsort(np.abs(eigen_values))
            eigen_values = eigen_values[index]
            eigen_vectors = eigen_vectors[:, index]
            return eigen_vectors[:, k_skip:k+1], np.sum(eigen_values[k_skip:k+1])
    else:
        raise ValueError("Unrecognized eigen_solver '%s'" % eigen_solver)
