"""Embed with Riemannian Metric"""

# Author: Marina Meila <mmp@stat.washington.edu>
# License: BSD 3 clause

import numpy as np
from .geometry import *
from .rmetric import *
from .spectral_embedding_ import SpectralEmbedding, spectral_embedding

def embed_with_rmetric( X, n_dim = 2, neighbors_radius=None, what_input = "points",  embedding = "spectral_embedding", invert_h = False ):
    """
    Parameters:
    ----------
    what_input: string, , optional. type of input data, 
           can be one of "points", "distance_matrix", "similarity_matrix"
    X: array-like or sparse array or matrix
       if what_input = "points",  shape: (n_samples, n_dim_data )
       input data is a sample of size n_samples
       if what_input = "distance_matrix",  shape: (n_samples, n_samples )
       input data is a set of distances between the n_samples points. 
       can be sparse.
       if what_input = "similarity_matrix",  shape: (n_samples, n_samples )
       input data is similarity matrix  between the n_samples points. 
       can be sparse.
       X is read-only
    n_dim : integer, optional
        The dimension of the embedding subspace.
    embedding: string, optional
        The embedding method to use. Currently only "spectral_embedding"
        available
     neighbors_radius = float
        the neighborhood radius parameter used in 
        - computing the Laplacian (required)
        - computing the similarity matrix 
        - computing the neighborhood graph
    invert_h: boolean, optional
        if True, G will be returned

    Returns
    -------
    distance_matrix: 2D array-like,  shape: (n_samples, n_samples )
       distances between the n_samples points.
       what_input = "points", distance_matrix computed internally, sparse.
        uses 1.5*neighbors_radius
       what_input = "distance_matrix", distance_matrix=X
       what_input = "similarity_matrix", distance_matrix is None
    similarity_matrix: 2D array-like,  shape: (n_samples, n_samples )
       similarities between the n_samples points.
       if what_input = "similarity_matrix": similarity_matrix is X
       else: similarity_matrix computed internally, sparse
          uses neighbors_radius 
    laplacian:  2D array-like, sparse, shape: (n_samples, n_samples )
       geometric Laplacian of the data, uses neighbors_radius
    Y: 2D array-like, shape: (n_samples, n_dim)
       embedding of the data 
    H: 3D array, shape: (n_samples, n_dim, n_dim )
       H[ i, :,: ] contains the inverse of the Riemannian metric at point i
       H is called the dual metric
    G: 3D array, optional. shape: (n_samples, n_dim, n_dim )
       G[ i, :,: ] contains the Riemannian metric at point i
       this is calculated by inverting each H matrix
    """

    n_samples = X.shape[0]

    # Compute distances
    if what_input is "points":
        dX = DistanceMatrix( X )
        distance_matrix = dX.get_distance_matrix( neighbors_radius = 1.5*neighbors_radius )
    elif what_input is "distance_matrix":
        distance_matrix = X
        if (distance_matrix.shape[1] != n_samples ):
            raise ValueError(("X must be square to be a distance matrix, X.shape =" %X.shape))

    # Compute similarities 
    if what_input is "similarity_matrix":
        similarity_matrix = X
        distance_matrix = None
        if (similarity_matrix.shape[1] != n_samples ):
            raise ValueError(("X must be square to be a similarity matrix, X.shape =" %X.shape))
    else:
        similarity_matrix = affinity_matrix( distance_matrix, neighbors_radius )
        if not np.all((similarity_matrix - similarity_matrix.T).data < 1e-10):
            dum = (similarity_matrix - similarity_matrix.T).data
            print( dum.shape )
            print( similarity_matrix.shape )
            maxdiff = max( dum )
            print( "similarity_matrix not symmetric! maxdiff", maxdiff )
            # i put this test here because spectral embedding complains 
            # about the similarity matrix

    # Embedding
    model = SpectralEmbedding(n_components = n_dim,
                              neighbors_radius=neighbors_radius,
                              affinity="precomputed")
    Y = model.fit_transform(similarity_matrix)
    # wasteful... computes the Laplacian internally. 
    # to rewrite spectral_embedding in the future

    # Laplacian and Riemannian metric
    laplacian = graph_laplacian( similarity_matrix, scaling_epps=neighbors_radius)
    RM = RiemannMetric( Y, laplacian = laplacian )
    if invert_h:
        H, G = RM.get_dual_rmetric( invert_h = True )
    else:
        H = RM.get_dual_rmetric()
        return distance_matrix, similarity_matrix, laplacian, Y, H
