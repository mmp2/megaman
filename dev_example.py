### Performing a Spectral Embedding 
import numpy as np
import Mmani.embedding.geometry as geom
import Mmani.embedding.spectral_embedding as se
import Mmani.embedding.eigendecomp as ed
import sys
from math import pi 
import scipy as sp
import time
from sklearn import datasets


if len(sys.argv) > 1:
    n_samples = int(sys.argv[1])
else:
    n_samples = 10

# n_samples = 50
if n_samples < 50:
    t = np.linspace(0, pi, n_samples)
    X = np.column_stack((t, np.cos(t), np.sin(t)))
else:
    X, color = datasets.samples_generator.make_s_curve(n_samples, random_state=0)

D = X.shape[1]
n_components = 2
rad = D/2.

Geometry = geom.Geometry(X, neighbors_radius = rad, cpp_distances = True)

t0 = time.time()
d1 = se.spectral_embedding(Geometry, n_components, eigen_solver = 'amg')  # symmetry not required
t1 = time.time()
print 'total time amg: ' + str(t1 - t0 ) + " seconds."
print d1[:5, :]

t0 = time.time()
d2 = se.spectral_embedding(Geometry, n_components, eigen_solver = 'arpack')  # symmetry not required
t1 = time.time()
print 'total time arpack: ' + str(t1 - t0 ) + " seconds."
print d2[:5, :]

