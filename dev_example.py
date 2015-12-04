### Performing a Spectral Embedding 
import numpy as np
import Mmani.embedding.geometry as geom
import sys
import scipy as sp
import scipy.sparse as sparse
import time
from sklearn import datasets
import warnings
from scipy.sparse import isspmatrix

n_samples = 10
X, color = datasets.samples_generator.make_s_curve(n_samples, random_state=0)

D = X.shape[1]
n_components = 2
rad = D/2.
seed = 36
random_state = np.random.RandomState(seed)


import Mmani.embedding.isomap_ as iso

Geometry = geom.Geometry(X, neighbors_radius = rad)

# These all agree for sufficiently small data sets
embed1 = iso.isomap(Geometry, 2, 'arpack', random_state)
embed2 = iso.isomap(Geometry, 2, 'amg', random_state)
embed3 = iso.isomap(Geometry, 2, 'lobpcg', random_state)

## In the eigendecomp we should catch the lobpcg non pos def error:
    # If it happens once try again with a G + I 
    # if it happens after G + I then suggest they try arpack or amg

## In geometry might want to offer 'dist_method' = ['base', 'pyflann', 'Cpp']

# Let's time it on some big data
n_samples = 100000
X, color = datasets.samples_generator.make_s_curve(n_samples, random_state=0)

D = X.shape[1]
n_components = 2
rad = D/2.
seed = 36
random_state = np.random.RandomState(seed)

isom = iso.Isomap(n_components, radius = rad,
                  random_state = random_state,
                  eigen_solver = 'amg', 
                  cpp_distances = True)
t0 = time.time()
embed = isom.fit_transform(X)
t1 = time.time()

print str(t1 - t0) + " seconds"

