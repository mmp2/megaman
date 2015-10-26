### Performing a Spectral Embedding 
import numpy as np
import Mmani.embedding.geometry as geom
import Mmani.embedding.spectral_embedding as se
import sys

rad = 0.05
if len(sys.argv) > 1:
    n_samples = int(sys.argv[1])
else:
    n_samples = 10

X = np.random.random((n_samples, 2))
thet = X[:,0]
X1 = np.array( 3*thet*np.sin(2*thet ))
X2 = np.array( 3*thet*np.cos(2*thet ))
X = np.array( (X1, X2, X[:,1]) )
X = X.T

#X = np.random.random((n_samples, 3))

Geometry = geom.Geometry(X, neighbors_radius = rad, cpp_distances = True)
#Geometry = geom.Geometry(X, neighbors_radius = rad)


d2 = se.spectral_embedding(Geometry, n_components = 3, eigen_solver = 'arpack')  # symmetry not required
d1 = se.spectral_embedding(Geometry, n_components = 3, eigen_solver = 'amg') # symmetry required


#print d1
#print d2
