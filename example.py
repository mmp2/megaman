#!/usr/bin/env python
import matplotlib.mlab as mlab
import matplotlib.pyplot as plt
import numpy as np
from Mmani.embedding import * 

rad = 0.05
n_samples = 5000
X = np.random.random((n_samples, 2))
thet = X[:,0]
X1 = np.array( thet*np.sin(2*thet ))
X2 = np.array( thet*np.cos(2*thet ))
X = np.array( (X1, X2, X[:,1]) )
X = X.T

#X = np.concatenate( (X,X),axis=1)
print X.shape

isSP = True
#from sklearn.manifold import LocallyLinearEmbedding
if isSP:
    from Mmani.embedding import SpectralEmbedding
    model = SpectralEmbedding(neighbors_radius=rad)
else:
    from Mmani.embedding import LocallyLinearEmbedding
    model = LocallyLinearEmbedding(15, 2)

Y = model.fit_transform(X)

print Y.shape
print X.shape

#plt.plot( X[:,0], X[ :,1 ], marker='.',linestyle='None',label='X' )
plt.plot( Y[:,0], Y[ :,1 ], marker='.',linestyle='None',color='red',label='Y' )
plt.xlabel('0')
plt.ylabel('1')
plt.legend()
#plt.axis([-1, 9, 0, 1])
#plt.grid(True)
plt.show(block=False)


# Rmetric now
from scipy.sparse.linalg.eigen.lobpcg.lobpcg import symeig
from sklearn.utils.arpack import eigsh
#lambdas, diffusion_map = eigsh(-laplacian, k=n_components,
#                                           sigma=1.0, which='LM',
#                                           tol=eigen_tol)
            
A = model.get_affinity_matrix()
la, va = eigsh(A,k=5,which='LM')
print la
L = graph_laplacian( A, scaling_epps=rad)
ll, vl = eigsh(L,k=5,which='LM')
print ll
h,g,duml = riemann_metric(Y, 2, laplacian=L)

print h.shape
print h[:4,:,:]

from junkelli import plot_cov_ellipse

ax = plt.gca()
for i in range(n_samples):
    cov = h[ i, :, : ].squeeze()
    plot_cov_ellipse( cov/1000, Y[i,:], nstd=2, ax=ax, edgecolor='none', facecolor=(thet[i],0,1-thet[i]))

plt.show()
