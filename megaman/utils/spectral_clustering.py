"""Spectral Clustering"""

# Author: James McQueen <jmcq@u.washington.edu>
#         Xiao Wang <wang19@u.washington.edu>
# LICENSE: Simplified BSD https://github.com/mmp2/megaman/blob/master/LICENSE


import numpy as np
import warnings
from megaman.utils.eigendecomp import eigen_decomposition
from scipy.sparse import isspmatrix, identity
from megaman.utils.k_means_clustering import k_means_clustering
from megaman.embedding.base import BaseEmbedding
from megaman.utils.validation import check_random_state

class SpectralClustering(BaseEmbedding):
    """
    Spectral clustering for find K clusters by using the eigenvectors of a 
    matrix which is derived from a set of similarities S.

    Parameters
    -----------
    K: integer
        number of K clusters
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
        iterations.  Defaults to numpy.random.RandomState    
    solver_kwds : any additional keyword arguments to pass to the selected eigen_solver    
    renormalize : (bool) whether or not to set the rows of the eigenvectors to have norm 1 
                 this can improve label quality
    stabalize : (bool) whether or not to compute the (more stable) eigenvectors of L = D^-1/2*S*D^-1/2
                instead of P = D^-1*S 
    """    
    def __init__(self,K,eigen_solver='auto', 
                 random_state=None, solver_kwds = None,
                 geom = None, radius = None, renormalize = True, stabalize = True,
                 additional_vectors=0):
        self.eigen_solver = eigen_solver
        self.random_state = random_state             
        self.K = K
        self.solver_kwds = solver_kwds 
        self.geom = geom
        self.radius = radius
        self.renormalize = renormalize
        self.stabalize = stabalize
        self.additional_vectors = 0
     
    def fit(self, X, y=None, input_type='affinity'):
        """
        Fit the model from data in X.

        Parameters
        ----------
        input_type : string, one of: 'similarity', 'distance' or 'data'.
            The values of input data X. (default = 'data')
        X : array-like, shape (n_samples, n_features)
            Training vector, where n_samples in the number of samples
            and n_features is the number of features.

        If self.input_type is similarity:
        X : array-like, shape (n_samples, n_samples),
            copy the similarity matrix X to S.    
        """
        X = self._validate_input(X, input_type)
        self.fit_geometry(X, input_type)
        random_state = check_random_state(self.random_state)            
        self.embedding_, self.eigen_vectors_, self.P_ = spectral_clustering(self.geom_, K = self.K, 
                                                                   eigen_solver = self.eigen_solver, 
                                                                   random_state = self.random_state, 
                                                                   solver_kwds = self.solver_kwds,
                                                                   renormalize = self.renormalize,
                                                                   stabalize = self.stabalize,
                                                                   additional_vectors = self.additional_vectors)   
                
def spectral_clustering(geom, K, eigen_solver = 'dense', random_state = None, solver_kwds = None, 
                        renormalize = True, stabalize = True, additional_vectors = 0):
    """
    Spectral clustering for find K clusters by using the eigenvectors of a 
    matrix which is derived from a set of similarities S.

    Parameters
    -----------
    S: array-like,shape(n_sample,n_sample)
        similarity matrix 
    K: integer
        number of K clusters
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
        iterations.  Defaults to numpy.random.RandomState    
    solver_kwds : any additional keyword arguments to pass to the selected eigen_solver 
    renormalize : (bool) whether or not to set the rows of the eigenvectors to have norm 1 
                 this can improve label quality
    stabalize : (bool) whether or not to compute the (more stable) eigenvectors of L = D^-1/2*S*D^-1/2
                instead of P = D^-1*S 
    additional_vectors : (int) compute additional eigen vectors when computing eigen decomposition.
        When eigen_solver = 'amg' or 'lopcg' often if a small number of eigen values is sought the
        largest eigenvalue returned is *not* equal to 1 (it should be). This can usually be fixed
        by requesting more than K eigenvalues until the first eigenvalue is close to 1 and then
        omitted. The remaining K-1 eigenvectors should be informative. 
    Returns
    -------
    labels: array-like, shape (1,n_samples)
    """ 
    # Step 1: get similarity matrix
    if geom.affinity_matrix is None:
        S = geom.compute_affinity_matrix()
    else:
        S = geom.affinity_matrix
        
    # Check for stability method, symmetric solvers require this
    if eigen_solver in ['lobpcg', 'amg']:
        stabalize = True
    if stabalize:
        geom.laplacian_type = 'symmetricnormalized'
        return_lapsym = True
    else:
        geom.laplacian_type = 'randomwalk'
        return_lapsym = False
    
    # Step 2: get the Laplacian matrix
    P = geom.compute_laplacian_matrix(return_lapsym = return_lapsym)
    # by default the Laplacian is subtracted from the Identify matrix (this step may not be needed)
    P += identity(P.shape[0])        
    
    # Step 3: Compute the top K eigenvectors and drop the first 
    if eigen_solver in ['auto', 'amg', 'lobpcg']:
        n_components = 2*int(np.log(P.shape[0]))*K + 1
        n_components += int(additional_vectors)
    else:
        n_components = K
    n_components = min(n_components, P.shape[0])
    (lambdas, eigen_vectors) = eigen_decomposition(P, n_components=n_components, eigen_solver=eigen_solver, 
                                                   random_state=random_state, drop_first = True,
                                                   solver_kwds=solver_kwds)
    # the first vector is usually uninformative 
    if eigen_solver in ['auto', 'lobpcg', 'amg']:
        if np.abs(lambdas[0] - 1) > 1e-4:
            warnings.warn("largest eigenvalue not equal to 1. Results may be poor. Try increasing additional_vectors parameter")
    eigen_vectors = eigen_vectors[:, 1:K]
    lambdas = lambdas[1:K]
    
    # If stability method chosen, adjust eigenvectors
    if stabalize:
        w = np.array(geom.laplacian_weights)
        eigen_vectors /= np.sqrt(w[:,np.newaxis])
        eigen_vectors /= np.linalg.norm(eigen_vectors, axis = 0)    
    
    # If renormalize: set each data point to unit length
    if renormalize:
        norms = np.linalg.norm(eigen_vectors, axis=1)
        eigen_vectors /= norms[:,np.newaxis]
    
    # Step 4: run k-means clustering
    labels =  k_means_clustering(eigen_vectors,K)    
    return labels, eigen_vectors, P