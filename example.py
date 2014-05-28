#!/usr/bin/env python
import matplotlib.mlab as mlab
import matplotlib.pyplot as plt
import numpy as np
from Mmani.embedding.embed_with_rmetric import embed_with_rmetric
from covar_plotter import plot_cov_ellipse
from sklearn.datasets.samples_generator import make_swiss_roll


# Same as first_example.py but using embed_with_rmetric()

rad = 0.12
n_samples = 1000
if False:
    X = np.random.random((n_samples, 2))
    thet = X[:,0]
    X1 = np.array( 3*thet*np.sin(2*thet ))
    X2 = np.array( 3*thet*np.cos(2*thet ))
    X = np.array( (X1, X2, X[:,1]) )
    X = X.T
else:
    X, thet = make_swiss_roll( n_samples, noise = 0.03 )
    X /= 10.
    thet -= thet.min()
    thet /= thet.max()  # normalize thet between [0,1] 
#    print( "max,min(thet)", max( thet), min(thet))
#    print( X.max(0), X.min(0))
    
X = np.asarray( X, order="C" )

#print( X.flags )
#print( X.shape, type(X))


distance_matrix, similarity_matrix, laplacian, Y, H = embed_with_rmetric( X,3, rad ) ## evectors 0,2 !!
#distance_matrix, similarity_matrix, laplacian, Y, H = embed_with_rmetric( X,3, rad )

# Plot the results

n_samplot = np.minimum( 500, n_samples ) # subsample the data
iisamples = np.random.randint( 0, n_samples, size=n_samplot )

ax = plt.gca()
#plt.plot( Y[iisamples,0], Y[ iisamples,1 ], marker='.',linestyle='None',color='red',label='Y' )
plt.plot( Y[iisamples,0], Y[ iisamples,2 ], marker='.',linestyle='None',color='red',label='Y' ) ## evectors 0,2 !!
plt.xlabel('0')
plt.ylabel('1')
plt.legend()

detH = np.linalg.det(H)
ineg = np.nonzero( detH <= 0 )
if ineg[0].size > 0:
    print( ineg[0].size, ' negative or singular covariance matrices' )
    plt.plot( detH )
    plt.title( 'detH' )
    plt.show()


for i in range(0):
#for i in range(n_samplot):
    ii = iisamples[i]
    cov = H[ ii, :2, :2 ].squeeze() ## evectors 0,2 !!
    plot_cov_ellipse( cov/1000, Y[ii,:], nstd=2, ax=ax, edgecolor='none', facecolor=(thet[ii],0,1-thet[ii]))
    
plt.show(block=True)
