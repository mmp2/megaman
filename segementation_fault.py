import numpy as np
from scipy import sparse
from sklearn.datasets.samples_generator import make_swiss_roll

rng = np.random.RandomState(123)
ns = 3000
X, t = make_swiss_roll( ns, noise = 0.0, random_state = rng)
X = np.asarray( X, order="C" )
nf = 750 
rad0 = 2.5
dim = 2

rad = rad0/ns**(1./(dim+6))  #check the scaling
n_noisef = nf - 3
noise_rad_frac = 0.1
noiserad = rad/np.sqrt(n_noisef) * noise_rad_frac
Xnoise = rng.rand(ns, n_noisef) * noiserad
X = np.hstack((X, Xnoise))
rad = rad*(1+noise_rad_frac) # add a fraction for noisy dimensions

from Mmani.geometry.distance import distance_matrix
dmat = distance_matrix(X, method = 'cython', radius = rad)