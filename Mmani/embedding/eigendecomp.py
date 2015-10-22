import warnings
import numpy as np
from scipy import sparse
from scipy.sparse.linalg import lobpcg
from scipy.sparse.linalg.eigen.lobpcg.lobpcg import symeig
from sklearn.utils import check_random_state
from sklearn.utils.validation import check_array
from sklearn.utils.arpack import eigsh, eigs

def _is_symmetric(M, tol = 1e-10):
    if sparse.isspmatrix(M):
        conditions = (M - M.T).data < tol 
    else:
        conditions = (M - M.T) < tol
    return(np.all(conditions))
    

def eigen_decomposition(G, n_components=8, eigen_solver=None,
                       random_state=None, eigen_tol=0.0, 
                       drop_first=True):
    """
    G : 2d numpy/scipy array. Potentially sparse.
        The matrix to find the eigendecomposition of 
    n_components : integer, optional
        The number of eigenvectors to return 

    eigen_solver : {None, 'arpack', 'lobpcg', or 'amg'}
        The eigenvalue decomposition strategy to use. AMG requires pyamg
        to be installed. It can be faster on very large, sparse problems,
        but may also lead to instabilities.

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
    
    # Check eigen_solver method
    try:
        from pyamg import smoothed_aggregation_solver
    except ImportError:
        if eigen_solver == "amg":
            raise ValueError("The eigen_solver was set to 'amg', but pyamg is "
                             "not available.")
    if eigen_solver is None:
        eigen_solver = 'arpack'
    elif not eigen_solver in ('arpack', 'lobpcg', 'amg'):
        raise ValueError("Unknown value for eigen_solver: '%s'."
                         "Should be 'amg', 'arpack', or 'lobpcg'"
                         % eigen_solver)

    # Initialize random state
    random_state = check_random_state(random_state)
    n_nodes = G.shape[0]
    
    is_symmetric = _is_symmetric(laplacian)
    
    if drop_first:
        n_components = n_components + 1       
    
    # Try Eigen Methods:
    if (eigen_solver == 'arpack'
        or eigen_solver != 'lobpcg' and
            (not sparse.isspmatrix(G)
             or n_nodes < 5 * n_components)):
        # lobpcg used with eigen_solver='amg' has bugs for low number of nodes
        # for details see the source code in scipy:
        # https://github.com/scipy/scipy/blob/v0.11.0/scipy/sparse/linalg/eigen
        # /lobpcg/lobpcg.py#L237
        # or matlab:
        # http://www.mathworks.com/matlabcentral/fileexchange/48-lobpcg-m

        # Here we'll use shift-invert mode for fast eigenvalues
        # (see http://docs.scipy.org/doc/scipy/reference/tutorial/arpack.html
        # for a short explanation of what this means)
        # Because the normalized Laplacian has eigenvalues between 0 and 2,
        # I - L has eigenvalues between -1 and 1.  ARPACK is most efficient
        # when finding eigenvalues of largest magnitude (keyword which='LM')
        # and when these eigenvalues are very large compared to the rest.
        # For very large, very sparse graphs, I - L can have many, many
        # eigenvalues very near 1.0.  This leads to slow convergence.  So
        # instead, we'll use ARPACK's shift-invert mode, asking for the
        # eigenvalues near 1.0.  This effectively spreads-out the spectrum
        # near 1.0 and leads to much faster convergence: potentially an
        # orders-of-magnitude speedup over simply using keyword which='LA'
        try:
            if is_symmetric:
                lambdas, diffusion_map = eigsh(-G, k=n_components, sigma=1.0, 
                                                which='LM',tol=eigen_tol)
            else:
                lambdas, diffusion_map = eigs(-G, k=n_components, sigma=1.0, 
                                                which='LM',tol=eigen_tol)            
        except RuntimeError:
            # When submatrices are exactly singular, an LU decomposition fails
            eigen_solver = "lobpcg"
            warnings.warn("submatrices exactlt singular, LU decomposition failed. Reverting to lobpcg")
    if eigen_solver == 'amg':
        if not is_symmetric:
            raise ValueError("lobpcg requires symmetric matrices.")
        if not sparse.issparse(G):
            warnings.warn("AMG works better for sparse matrices")
        # Use AMG to get a preconditioner and speed up the eigenvalue problem.
        G = G.astype(np.float)  # lobpcg needs native floats
        ml = smoothed_aggregation_solver(check_array(G, accept_sparse = ['csr']))
        M = ml.aspreconditioner()
        X = random_state.rand(n_nodes, n_components + 1)
        X[:, 0] = (G.diagonal()).ravel()
        lambdas, diffusion_map = lobpcg(G, X, M=M, tol=1.e-12, largest=False)    
    elif eigen_solver == "lobpcg":
        if not is_symmetric:
            raise ValueError("lobpcg requires symmetric matrices.")
        G = G.astype(np.float)  # lobpcg needs native floats
        if (n_nodes < 5 * n_components + 1):
            # lobpcg has problems with small number of nodes
            # lobpcg will fallback to symeig, so we short circuit it
            if sparse.isspmatrix(G):
                G = G.todense()
            lambdas, diffusion_map = symeig(G)
        else:            
            G = G.astype(np.float) # lobpcg needs native floats
            # We increase the number of eigenvectors requested, as lobpcg
            # doesn't behave well in low dimension
            X = random_state.rand(n_nodes, n_components + 1)
            lambdas, diffusion_map = lobpcg(G, X, tol=1e-15, largest=False, 
                                            maxiter=2000)
    return (lambdas, diffusion_map)