"""Spectral Clustering"""

# Author: James McQueen <jmcq@u.washington.edu>
#         Xiao Wang <wang19@u.washington.edu>
# LICENSE: Simplified BSD https://github.com/mmp2/megaman/blob/master/LICENSE


import numpy as np
from eigendecomp import eigen_decomposition
from scipy.sparse import isspmatrix 
from sklearn.cluster import KMeans
from megaman.clustering.k_means import k_means_clustering

'''
* Implement P and L matrices in laplacian
* Use geometry class for inputting data -- copy from embedding functions
* Implement eigenvector rotation 
'''

class SpectralClustering():
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
    """    
    def __init__(self,K,eigen_solver='auto', 
                 random_state=None, solver_kwds = None):
        self.eigen_solver = eigen_solver
        self.random_state = random_state             
        self.K = K
        self.solver_kwds = solver_kwds 
     
     
    def fit(self, X, y=None, input_type='similarity'):
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
        if input_type == 'data':
            print('not working')
        elif input_type == 'distance':
            print('not working')
        elif input_type == 'similarity':
            S = X
        else:
            print('input_type is not recognized')
            
        spectral_clustering(S, K = self.K, eigen_solver = self.eigen_solver, random_state = self.random_state, solver_kwds = self.solver_kwds)   
                
def spectral_clustering(S, K, eigen_solver = 'auto', random_state = None, solver_kwds = None, renormalize = False):
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
    renormalize : bool
    Returns
    -------
    labels: array-like, shape (1,n_samples)
    """    
# *Afterwards*: step 0: compute S matrix given a data set X

# Step 1:
    
    d = np.asarray(S.sum(1)).squeeze()
    P = S.copy()
    if isspmatrix(P):
        P = P.tocoo()
        P.data /= d[P.row]
    else:
        P /= d[:, np.newaxis]
        
    # Step 2 & 3: 
    (lambdas, eigen_vectors) = eigen_decomposition(P, n_components=K-1, eigen_solver=eigen_solver, random_state=random_state, solver_kwds=solver_kwds)
    if renormalize:
        norms = np.linalg.norm(eigen_vectors, axis=1)
        eigen_vectors /= norms
    # Step 4: *afterwards*: write own K-means w/ orthogonal initialization
    labels =  k_means_clustering(eigen_vectors,K)
    
    return labels 