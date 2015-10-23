#!/usr/bin/env python
# import matplotlib.mlab as mlab
# import matplotlib.pyplot as plt
import sys
import time
import numpy as np
import Mmani.embedding.geometry as geom

# Make some artificial Data:
rad = 0.05
n_samples = 1000

X = np.random.random((n_samples, 2))
thet = X[:,0]
X1 = np.array( 3*thet*np.sin(2*thet ))
X2 = np.array( 3*thet*np.cos(2*thet ))
X = np.array( (X1, X2, X[:,1]) )
X = X.T

### THE GEOMETRY OBJECT
## Computing Distance Matrices:
# 1.  sklearn radius_neighbors_graph (small data sets)
geoX = geom.Geometry(X)
distance_matrix = geoX.get_distance_matrix()

# pyflann -- moderate to large data with pyflann installed
#         -- use path_to_flann if FLANN is installed to a specific location 
path_to_flann =  '/homes/jmcq/flann-1.8.4-src/src/python'
geoX = geom.Geometry(X, use_flann = True, path_to_flann = path_to_flann)
distance_matrix = geoX.get_distance_matrix()

# C++ executable (Currently only works on UNIX based systems)
geoX = geom.Geometry(X, cpp_distances = True)
distance_matrix = geoX.get_distance_matrix()

## Computing Affinity Matrices:
# 1. When it's already passed, returns itself
geoX = geom.Geometry(np.random.random((n_samples, n_samples)), 
                    neighbors_radius = rad, is_affinity = True)
affinity_matrix = geoX.get_affinity_matrix()
# 2. when distance matrix already computed, only computes affinity:
geoX = geom.Geometry(X, neighbors_radius = rad, cpp_distances = True)
distance_matrix = geoX.get_distance_matrix()
affinity_matrix = geoX.get_affinity_matrix()
# 3. distance matrix passed
geoX = geom.Geometry(np.random.random((n_samples, n_samples)), 
                    neighbors_radius = rad, is_distance = True)
affinity_matrix = geoX.get_affinity_matrix()
# 4. distance matrix not computed
geoX = geom.Geometry(X, neighbors_radius = rad, cpp_distances = True)
affinity_matrix = geoX.get_affinity_matrix()
geoX.distance_matrix # now distance matrix exists

## Computing Laplacian Matrices:
# 1. Laplacian matrix has been assigned 
geoX = geom.Geometry(X, neighbors_radius = rad, cpp_distances = True)
Lapl = np.random.random((n_samples, n_samples))
geoX.assign_laplacian_matrix(Lapl)
laplacian = geoX.get_laplacian_matrix()

# 2. affinity matrix is passed or already computeed 
geoX = geom.Geometry(X, neighbors_radius = rad, cpp_distances = True)
affinity_matrix = geoX.get_affinity_matrix() # distance_matrix also exists
laplacian = geoX.get_laplacian_matrix()

# 3. distance matrix is passed or already computed
geoX = geom.Geometry(X, neighbors_radius = rad, cpp_distances = True)
distance_matrix = geoX.get_distance_matrix()
laplacian = geoX.get_laplacian_matrix()

# 4. raw data is passed
geoX = geom.Geometry(X, neighbors_radius = rad, cpp_distances = True)
laplacian = geoX.get_laplacian_matrix() # distance and affinity matrices exist


### Performing a Spectral Embedding 
import numpy as np
import Mmani.embedding.geometry as geom
import Mmani.embedding.spectral_embedding as se

rad = 5
n_samples = 50

X = np.random.random((n_samples, 2))
thet = X[:,0]
X1 = np.array( 3*thet*np.sin(2*thet ))
X2 = np.array( 3*thet*np.cos(2*thet ))
X = np.array( (X1, X2, X[:,1]) )
X = X.T
Geometry = geom.Geometry(X, neighbors_radius = rad, cpp_distances = True)

d1 = se.spectral_embedding(Geometry, n_components = 3, eigen_solver = 'amg') # symmetry required
d2 = np.real(se.spectral_embedding(Geometry, n_components = 3, eigen_solver = 'arpack'))  # symmetry not required

