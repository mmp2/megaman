import sys
sys.path.append('/homes/jmcq/megaman/') # this is stupid

import numpy as np
import megaman.geometry.geometry as geom
import scipy as sp
import scipy.sparse as sparse
from sklearn import datasets

# Generate an example data set
N = 10
X, color = datasets.samples_generator.make_s_curve(N, random_state=0)
n_components = 2
n_radius = 5 # ignore distances above this radius
a_radius = 5 # A = exp(-||x - y||/a_radius^2) (if not passed a_radius = n_radius)

# Geometry is the main class that will Cache things like distance, affinity, and laplacian.

Geometry = geom.Geometry(X, distance_method = 'brute',
                        neighborhood_radius = n_radius, affinity_radius = a_radius,
                        laplacian_type = 'geometric')

# You can get the distance, affinity etc with e.g: Geometry.get_distance_matrix()
# you can also use the assign_distance, assign_affinity, and assign_laplacian

# If you pass X to the embedding functions it will create a Geometry object.
# But if you're planning on trying multiple embeddings it's best to pass it directly

# LTSA
import megaman.embedding.ltsa as ltsa
LTSA = ltsa.LTSA(n_components = n_components, eigen_solver = 'arpack',
                Geometry = Geometry)
(embed_ltsa, err) = LTSA.fit_transform(X)

# LLE
import megaman.embedding.locally_linear as lle
LLE = lle.LocallyLinearEmbedding(n_components = n_components, eigen_solver = 'arpack',
                                Geometry = Geometry)
(embed_lle, err) = LLE.fit_transform(X)

# Isomap
import megaman.embedding.isomap as iso
ISOMAP = iso.Isomap(n_components = n_components, eigen_solver = 'arpack',
                    Geometry = Geometry)
embed_isomap = ISOMAP.fit_transform(X)

# Spectral Embedding (
import megaman.embedding.spectral_embedding as se
Spectral = se.SpectralEmbedding(n_components = n_components, eigen_solver = 'arpack',
                                Geometry = Geometry)
embed_spectral = Spectral.fit_transform(X)

# Visulize them in a 4x4 plot:

import matplotlib.pyplot as plt
f, axarr = plt.subplots(2, 2)
axarr[0, 0].plot(embed_ltsa[:,0], embed_ltsa[:,1])
axarr[0, 0].set_title('LTSA')
axarr[0, 1].scatter(embed_lle[:,0], embed_lle[:,1])
axarr[0, 1].set_title('LLE')
axarr[1, 0].plot(embed_isomap[:,0], embed_isomap[:,1])
axarr[1, 0].set_title('Isomap')
axarr[1, 1].scatter(embed_spectral[:,0], embed_spectral[:,1])
axarr[1, 1].set_title('Spectral')
plt.show()
