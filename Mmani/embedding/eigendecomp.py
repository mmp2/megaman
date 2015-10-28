import warnings
import numpy as np
from scipy import sparse
from scipy.sparse.linalg import lobpcg, eigs, eigsh
from scipy.sparse.linalg.eigen.lobpcg.lobpcg import symeig
from sklearn.utils import check_random_state
from sklearn.utils.validation import check_array

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
    n_nodes = G.shape[0]
    
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
    # Check input values
    if not isinstance(largest, bool):
        raise ValueError("largest should be True if you want largest eigenvalues otherwise False")
    random_state = check_random_state(random_state)
    if drop_first:
        n_components = n_components + 1     
    # Check for symmetry
    is_symmetric = _is_symmetric(G)
    # Convert G to best type for eigendecomposition 
    if G.getformat() is not 'csr':
        G.tocsr()
    G = G.astype(np.float)
    
    # Try Eigen Methods:
    if eigen_solver == 'arpack':
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
        X = random_state.rand(n_nodes, n_components)
        X[:, 0] = (G.diagonal()).ravel()
        lambdas, diffusion_map = lobpcg(G, X, M=M, tol=1.e-12, largest=largest)    
    elif eigen_solver == "lobpcg":
        if not is_symmetric:
            raise ValueError("lobpcg requires symmetric matrices.")
        if (n_nodes < 5 * n_components + 1):
            # lobpcg has problems with small number of nodes
            if sparse.isspmatrix(G):
                G = G.todense()
            lambdas, diffusion_map = symeig(G)
        else:            
            X = random_state.rand(n_nodes, n_components)
            lambdas, diffusion_map = lobpcg(G, X, tol=1.e-12, largest=largest)
    return (lambdas, diffusion_map)
