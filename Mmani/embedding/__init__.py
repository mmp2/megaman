"""
MMP: TO UPDATE THIS
The :mod:`sklearn.manifold` module implements data embedding techniques.
"""

from .locally_linear import locally_linear_embedding, LocallyLinearEmbedding
from .spectral_embedding_ import SpectralEmbedding, spectral_embedding
from .geometry import graph_laplacian, distance_matrix, DistanceMatrix,  riemann_metric, affinity_matrix
#from .isomap import Isomap
#from .mds import MDS

#__all__ = ['locally_linear_embedding', 'LocallyLinearEmbedding', 'Isomap',
__all__ = ['locally_linear_embedding', 'LocallyLinearEmbedding', 'spectral_embedding', 'SpectralEmbedding', 'graph_laplacian', 'riemann_metric', 'distance_matrix','DistanceMatrix','affinity_matrix']
