#!/usr/bin/env python

import matplotlib.mlab as mlab
import matplotlib.pyplot as plt
import numpy as np

import sys
import os
path = os.path.abspath('..')
sys.path.append(path)
path = os.path.abspath('../..')
sys.path.append(path)
from covar_plotter import plot_cov_ellipse
from Mmani.embedding.embed_with_rmetric import *
from Mmani.embedding.spectral_embedding_ import _graph_is_connected


""" Small demo with sdss_corrected_spectra """
rad = 2000    # from analyze_spectra, rad in [1000, 2000] gives dim = 1.9
              # number neighbors 3..12
remove_outliers = True
degmin = 5

compute_H = True
mdimY = 6

save_fig = True

# Load spectra

data = np.load('spectra.npz')
wavelengths = data['wavelengths']
X = data['spectra']
print( X.shape )
n_samples = X.shape[0]

flann = FLANN()
params = flann.build_index(X)
dists = distance_matrix( X, flindex = flann, mode='radius_neighbors', 
                         neighbors_radius=rad*1.5 )
A = affinity_matrix( dists, rad )

#plt.imshow(A.toarray())
#plt.show()

# Preprocessing

if remove_outliers:
    degrees = np.asarray(A.sum(axis=1)).squeeze()
    deg_mean = np.mean( degrees )
    deg_std = np.std( degrees )
    iikeep = np.nonzero( degrees >= degmin )
    iikeep = iikeep[ 0 ]
    nodes_removed = n_samples - iikeep.size
    print( '---Removing ' + repr(nodes_removed) + ' nodes with degree <' + repr(degmin))
    X = X[ iikeep, : ]
    A = A.toarray()[ iikeep, iikeep ]
    dists = dists.toarray()[ iikeep, iikeep ]
    n_samples = iikeep.size

# Embeddings

# geometric embedding
distance_matrix, similarity_matrix, laplacian, Y, H = embed_with_rmetric( X, mdimY, rad )

# Plot the results

n_samplot = np.minimum( 1000, n_samples ) # subsample the data
iisamples = np.random.randint( 0, n_samples, size=n_samplot )


"""
detH = np.linalg.det(H[:,:2,:2])
ineg = np.nonzero( detH <= 0 )
if ineg[0].size > 0:
    print( ineg[0].size, ' negative or singular covariance matrices' )
    plt.plot( detH )
    plt.title( 'detH' )
    plt.show()
"""

# plot the evectors

#plt.plot( Y, marker='.', markersize=0.4,linestyle='None' )
#plt.title( 'evectors')
#plt.show()

ax0 = 1
ax1 = 3
iax = [ax0, ax1]
ax = plt.gca()
plt.plot( Y[iisamples,ax0], Y[ iisamples, ax1], marker='.', markersize=2,linestyle='None',label='Y' )
#plt.show()

if compute_H:
    degrees = np.asarray(similarity_matrix.sum(axis=1)).squeeze()
    degmax = np.max( degrees )
    cov0 = np.eye(2)/1.e4
    for i in range(n_samplot):
        ii = iisamples[i]
        cov = H[ ii, (1,3), (1,3) ].squeeze()
        if i in [0,3,100]:
            print( cov )
        if np.linalg.det(cov)>0:
            plot_cov_ellipse( cov*rad*5, Y[ii,(1,3)], nstd=2, ax=ax, edgecolor='none', facecolor=[ 0, degrees[ii]/degmax, 0])
#        plot_cov_ellipse( cov0, Y[ii,:2], nstd=2, ax=ax, edgecolor='none', facecolor='pink')
#    print( cov, np.linalg.det( cov ) )
    if save_fig:
        plt.savefig( "spectra-emb"+".png", format='png' )
    plt.show()





