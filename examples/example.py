import sys
import numpy as np
import scipy as sp
import scipy.sparse as sparse
from megaman.geometry import Geometry
from sklearn import datasets
from megaman.embedding import (Isomap, LocallyLinearEmbedding,
                               LTSA, SpectralEmbedding)

# Generate an example data set
N = 10
X, color = datasets.samples_generator.make_s_curve(N, random_state=0)

# Geometry is the main class that will Cache things like distance, affinity, and laplacian.
# you instantiate the Geometry class with the parameters & methods for the three main components:
# Adjacency: an NxN (sparse) pairwise matrix indicating neighborhood regions
# Affinity an NxN (sparse) pairwise matrix insicated similarity between points 
# Laplacian an NxN (sparse) pairwsie matrix containing geometric manifold information

radius = 5
adjacency_method = 'cyflann'
adjacency_kwds = {'radius':radius} # ignore distances above this radius
affinity_method = 'gaussian'
affinity_kwds = {'radius':radius} # A = exp(-||x - y||/radius^2) 
laplacian_method = 'geometric'
laplacian_kwds = {'scaling_epps':radius} # scaling ensures convergence to Laplace-Beltrami operator

geom  = Geometry(adjacency_method=adjacency_method, adjacency_kwds=adjacency_kwds,
				 affinity_method=affinity_method, affinity_kwds=affinity_kwds,
				 laplacian_method=laplacian_method, laplacian_kwds=laplacian_kwds)

# You can/should also use the set_data_matrix, set_adjacency_matrix, set_affinity_matrix
# to send your data set (in whichever form it takes) this way.
geom.set_data_matrix(X)

# You can get the distance, affinity etc with e.g: Geometry.get_distance_matrix()
	# you can update the keyword arguments passed inially using these functions
adjacency_matrix = geom.compute_adjacency_matrix()
# by defualt this is pass-by-reference. Use copy=True to get a copied version.

# If you don't want to pre-compute a Geometry you can pass a dictionary or geometry
# arguments to one of the embedding classes.
geom  = {'adjacency_method':adjacency_method, 'adjacency_kwds':adjacency_kwds,
		 'affinity_method':affinity_method, 'affinity_kwds':affinity_kwds,
		 'laplacian_method':laplacian_method, 'laplacian_kwds':laplacian_kwds}
	

# an example follows for creating each embedding into 2 dimensions.
n_components = 2

# LTSA
ltsa =LTSA(n_components=n_components, eigen_solver='arpack',
			geom=geom)
embed_ltsa = ltsa.fit_transform(X)

# LLE
lle = LocallyLinearEmbedding(n_components=n_components, eigen_solver='arpack',
							 geom=geom)
embed_lle = lle.fit_transform(X)

# Isomap
isomap = Isomap(n_components=n_components, eigen_solver='arpack',
				geom=geom)
embed_isomap = isomap.fit_transform(X)

# Spectral Embedding 
spectral = SpectralEmbedding(n_components=n_components, eigen_solver='arpack',
							 geom=geom)
embed_spectral = spectral.fit_transform(X)