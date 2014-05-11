#!/usr/bin/env python
import matplotlib.mlab as mlab
import matplotlib.pyplot as plt
import numpy as np
from Mmani.embedding.embed_with_rmetric import embed_with_rmetric
from covar_plotter import plot_cov_ellipse

# Same as first_example.py but using embed_with_rmetric()

rad = 0.2
n_samples = 1000
X = np.random.random((n_samples, 2))
thet = X[:,0]
X1 = np.array( 3*thet*np.sin(2*thet ))
X2 = np.array( 3*thet*np.cos(2*thet ))
X = np.array( (X1, X2, X[:,1]) )
X = X.T
X = np.asarray( X, order="C" )
#print( X.flags )

distance_matrix, similarity_matrix, laplacian, Y, H = embed_with_rmetric( X,2, rad )

# Plot the results

n_samplot = np.minimum( 500, n_samples ) # subsample the data
iisamples = np.random.randint( 0, n_samples, size=n_samplot )

ax = plt.gca()
plt.plot( Y[iisamples,0], Y[ iisamples,1 ], marker='.',linestyle='None',color='red',label='Y' )
plt.xlabel('0')
plt.ylabel('1')
plt.legend()

for i in range(n_samplot):
    ii = iisamples[i]
    cov = H[ ii, :, : ].squeeze()
    plot_cov_ellipse( cov/1000, Y[ii,:], nstd=2, ax=ax, edgecolor='none', facecolor=(thet[ii],0,1-thet[ii]))
    
plt.show(block=True)
