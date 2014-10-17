#!/usr/bin/env python
import matplotlib.mlab as mlab
import matplotlib.pyplot as plt
import numpy as np
from Mmani.embedding import * 

rad = 0.1
n_samples = 1000
X = np.random.random((n_samples, 2))
thet = X[:,0]
X1 = np.array( 3*thet*np.sin(2*thet ))
X2 = np.array( 3*thet*np.cos(2*thet ))
X = np.array( (X1, X2, X[:,1]) )
X = X.T
X = np.asarray( X, order="C" )
#print( X.flags )

print( "....Computing affinity matrix..." )
dX = DistanceMatrix( X )
csdistanceX = dX.get_distance_matrix( neighbors_radius = rad )
A = affinity_matrix( csdistanceX, rad )

isSP = True
#from sklearn.manifold import LocallyLinearEmbedding
if isSP:
    from Mmani.embedding import SpectralEmbedding
    print( "....Spectral embedding...." )
    model = SpectralEmbedding(neighbors_radius=rad, affinity="precomputed")
else:
    from Mmani.embedding import LocallyLinearEmbedding
    model = LocallyLinearEmbedding(15, 2)

Y = model.fit_transform(A)

print( Y.shape )
print( X.shape )

print( "....Plot embedding...." )
#plt.plot( X[:,0], X[ :,1 ], marker='.',linestyle='None',label='X' )
n_samplot = np.minimum( 1000, n_samples )
iisamples = np.random.randint( 0, n_samples, size=n_samplot )
plt.plot( Y[iisamples,0], Y[ iisamples,1 ], marker='.',linestyle='None',color='red',label='Y' )
plt.xlabel('0')
plt.ylabel('1')
plt.legend()
#plt.axis([-1, 9, 0, 1])
#plt.grid(True)
plt.show(block=False)


print( "....Riemannian metric...." )

# Rmetric now
from scipy.sparse.linalg.eigen.lobpcg.lobpcg import symeig
from sklearn.utils.arpack import eigsh
#lambdas, diffusion_map = eigsh(-laplacian, k=n_components,
#                                           sigma=1.0, which='LM',
#                                           tol=eigen_tol)
            
A = model.get_affinity_matrix()
#la, va = eigsh(A,k=5,which='LM')
#print( la )
L = graph_laplacian( A, scaling_epps=rad)
#ll, vl = eigsh(L,k=5,which='LM')
#print( ll )
h,g, dum1, dum2, dum3 = riemann_metric(Y, laplacian=L, n_dim = 2 )

print( "h.shape=", h.shape )
print( h[:4,:,:] )

print( "....Plotting the dual rmetric...." )

from covar_plotter import plot_cov_ellipse

ax = plt.gca()
for i in range(n_samplot):
    ii = iisamples[i]
    cov = h[ ii, :, : ].squeeze()
    plot_cov_ellipse( cov/1000, Y[ii,:], nstd=2, ax=ax, edgecolor='none', facecolor=(thet[ii],0,1-thet[ii]))

plt.show(block=True)
