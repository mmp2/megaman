#!/usr/bin/env python

import matplotlib.mlab as mlab
import matplotlib.pyplot as plt
import numpy as np
from pylab import polyfit

#from sklearn.manifold.spectral_embedding_ import SpectralEmbedding as SE0
import sys
import os
path = os.path.abspath('..')
sys.path.append(path)
path = os.path.abspath('../..')
sys.path.append(path)
from covar_plotter import plot_cov_ellipse
from Mmani.embedding.geometry import *
#from Mmani.embedding.embed_with_rmetric import *
#from Mmani.embedding.spectral_embedding_ import _graph_is_connected
from pyflann import *

# Load spectra

data = np.load('spectra.npz')
wavelengths = data['wavelengths']
X = data['spectra']
print( X.shape )
n_samples = X.shape[0]

is_neighborhoods = False  # dimension = rate of growth of neighborhoods
mdimY = 6
nradii = 8

is_outliers = True   # find outliers
rad = 2000

if is_outliers:
    flann = FLANN()
    params = flann.build_index(X)
    dists = distance_matrix( X, flindex = flann, mode='radius_neighbors', 
                             neighbors_radius=rad*1.5 )
    avgd = np.asarray(dists.sum(axis=1)).squeeze()/n_samples
    d_mean = np.mean( avgd )
    d_std = np.std( avgd )
    print( 'radius=' + repr(rad) +': average distances mean (std)=' + repr(d_mean) + '(' + repr(d_std) +')')
    
    A = affinity_matrix( dists, rad )
    degrees = np.asarray(A.sum(axis=1)).squeeze()
    deg_mean = np.mean( degrees )
    deg_std = np.std( degrees )
    print( ' average degrees mean (std)=' + repr(deg_mean) + '(' + repr(deg_std) +')')
    
    plt.plot( np.sort( degrees ) )
    plt.title( 'degrees' )
    plt.grid(b=True)
    plt.show()


if is_neighborhoods:
# Neighborhoods

    radii = np.logspace(3,3.3,nradii)
    avg_neighbors = np.zeros( nradii )

    flann = FLANN()
    params = flann.build_index(X)
    for ii in range( nradii ):
        rad = radii[ ii ]
        dists = distance_matrix( X, flindex = flann, mode='radius_neighbors', 
                         neighbors_radius=rad, symmetrize = True, n_neighbors=0 )
        avg_neighbors[ii] = dists.data.shape[0]/1./n_samples
        print( avg_neighbors[ii]*2)

    m,b = polyfit( np.log(radii), np.log(avg_neighbors), 1 )
    plt.plot( radii, avg_neighbors, color='red' )
    plt.plot( radii, np.exp(b)*radii**m, color='blue' )
    plt.yscale( 'log' )
    plt.xscale( 'log' )
    plt.xlabel( 'radius' )
    plt.ylabel( 'neighbors' )
    plt.title( 'spectra data dim='+repr(m)[:4] )
    plt.grid(b=True,which='minor')
    print('dim=', m )
    plt.savefig( "neighbors"+".png", format='png' )
    plt.show()



