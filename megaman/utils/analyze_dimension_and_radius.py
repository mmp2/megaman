#!/usr/bin/env python
from __future__ import division
try:
    import matplotlib
    MATPLOTLIB_LOADED = True
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
except ImportError:
    MATPLOTLIB_LOADED = False

import sys, time
import numpy as np

from numpy import polyfit

from megaman.geometry.adjacency import compute_adjacency_matrix

def compute_largest_radius_distance(X, rad, adjacency_method, adjacency_kwds):
    print('computing distance matrix...')
    adjacency_kwds['radius'] = rad
    t0 = time.time()
    dists = compute_adjacency_matrix(X,adjacency_method,**adjacency_kwds)
    print("Symmetrizing distance matrix...")
    dists = (dists + dists.T) # symmetrize, removes zero on diagonal
    dists.data = 0.5 * dists.data
    print("largest distance found: " + str(np.max(dists.data)))
    return(dists)

def neighborhood_analysis(dists, radii):
    n_samples = dists.shape[0]; nradii = len(radii)
    avg_neighbors = np.zeros(nradii); num_no_nbrs = np.zeros(nradii);
    max_neighbors = np.zeros(nradii); min_neighbors = np.zeros(nradii)
    dists = dists.tocsr()
    for ii in range(nradii):
        print("=========================================")
        print("Step " + str(ii +1) + " of " + str(nradii))
        ii = nradii - 1 - ii
        print(ii)
        rad = radii[ii]
        print("radius: " + str(radii[ii]))
        print("censoring distance matrix...")
        dists.data[dists.data > rad] = 0.0
        dists.eliminate_zeros()
        print(dists.nnz)
        avg_neighbors[ii] = dists.nnz/n_samples
        print('finding neighbors per point...')
        nbrs_per_row = np.diff(dists.indptr)
        min_neighbors[ii] = np.min(nbrs_per_row)
        max_neighbors[ii] = np.max(nbrs_per_row)
        num_no_nbrs[ii] = len(np.where(nbrs_per_row == 0)[0])
        print("average # nbrs: " + str(avg_neighbors[ii]))
        print("min # nbrs: " + str(min_neighbors[ii]))
        print("max # nbrs: " + str(max_neighbors[ii]))
        print("There are " + str(num_no_nbrs[ii]) + " points with no neighbors.")
        print("calculating affinity matrix...")
        print("finding connected components...")
        ii = nradii - 1 - ii
    results = {'avg_neighbors':avg_neighbors,
               'min_neighbors':min_neighbors,
               'max_neighbors':max_neighbors,
               'num_no_neighbors':num_no_nbrs,
               'radii':radii}
    return(results)

def find_dimension_plot(avg_neighbors, radii, fit_range, savefig=False, fname='dimension_plot.png'):
    tickrange = np.append(np.arange(0, len(radii)-1, 10), len(radii)-1)
    try:
        m,b = polyfit(np.log(radii[fit_range]), np.log(avg_neighbors[fit_range]), 1)
    except:
        m = 0
        b = 0
    if MATPLOTLIB_LOADED:
        plt.scatter(radii, avg_neighbors)
        plt.plot(radii, avg_neighbors, color='red')
        plt.plot(radii[fit_range], np.exp(b)*radii[fit_range]**m, color='blue')
        plt.yscale('log')
        plt.xscale('log')
        plt.xlabel('radius')
        plt.ylabel('neighbors')
        plt.title('data dim='+repr(m)[:4] + "\nRadius = [" + str(np.min(radii)) + ", " + str(np.max(radii)) + "]")
        plt.xlim([np.min(radii), np.max(radii)])
        plt.xticks(np.round(radii[tickrange], 1), np.round(radii[tickrange], 1))
        plt.grid(b=True,which='minor')
        print('dim=', m )
        plt.show()
        if savefig:
            plt.savefig(fname, format='png')
    return(m)

def run_analyze_dimension_and_radius(data, rmin, rmax, nradii, adjacency_method='brute', adjacency_kwds = {},
                                     fit_range=None, savefig=False, plot_name = 'dimension_plot.png'):
    """
    This function is used to estimate the doubling dimension (approximately equal to the intrinsic
    dimension) by computing a graph of neighborhood radius versus average number of neighbors.

    The "radius" refers to the truncation constant where all distances greater than
    a specified radius are taken to be infinite. This is used for example in the
    truncated Gaussian kernel in estimate_radius.py


    Parameters
    ----------
    data : numpy array,
        Original data set for which we are estimating the bandwidth
    rmin : float,
        smallest radius to consider
    rmax : float,
        largest radius to consider
    nradii : int,
        number of radii between rmax and rmin to consider
    adjacency_method : string,
        megaman adjacency method to use, default 'brute' see geometry.py for details
    adjacency_kwds : dict,
        dictionary of keywords for adjacency method
    fit_range : list of ints,
        range of radii to consider default is range(nradii), i.e. all of them
    savefig: bool,
        whether to save the radius vs. neighbors figure
    plot_name: string,
        filename of the figure to be saved as.

    Returns
    -------
    results : dictionary
        contains the radii, average nieghbors, min and max number of neighbors and number
        of points with no neighbors.
    dim : float,
        estimated doubling dimension (used as an estimate of the intrinsic dimension)
    """
    n, D = data.shape
    radii = 10**(np.linspace(np.log10(rmin), np.log10(rmax), nradii))
    dists = compute_largest_radius_distance(data, rmax, adjacency_method, adjacency_kwds)
    results = neighborhood_analysis(dists, radii)
    avg_neighbors = results['avg_neighbors'].flatten()
    radii = results['radii'].flatten()
    if fit_range is None:
        fit_range = range(len(radii))
    dim = find_dimension_plot(avg_neighbors, radii, fit_range, savefig, plot_name)
    return(results, dim)
