#!/usr/bin/env python
from __future__ import division 
import sys, time, pickle
import numpy as np
import scipy.io, scipy
from scipy.sparse.csgraph import connected_components
import cPickle

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pylab import polyfit
from large_sparse_functions import *

from megaman.geometry.adjacency import compute_adjacency_matrix

def compute_largest_radius_distance(X, rad, num_checks, num_trees, fbase, nparts):
    print('computing distance matrix...')
    # adjacency_method = 'cyflann'
    # cyflann_kwds = {'index_type':'kdtrees', 'num_trees':num_trees, 'num_checks':num_checks}
    # kwds = {'radius':rad, 'cyflann_kwds':cyflann_kwds} 
    print('using brute force...')
    adjacency_method = 'brute'
    kwds = {'radius':rad} 
    t0 = time.time()
    dists = compute_adjacency_matrix(X,adjacency_method,**kwds)
    print("Symmetrizing distance matrix...")
    dists = (dists + dists.T) # symmetrize, removes zero on diagonal
    dists.data = 0.5 * dists.data 
    print("largest distance found: " + str(np.max(dists.data)))
    # fname = fbase + 'dists_radius' + str(rad) + '_num_trees_' + str(num_trees)+'_num_checks_' + str(num_checks)
    fname = fbase + 'dists_radius' + str(rad) + '_brute_force'
    # save in parts: 
    print("saving distance matrix...")
    save_sparse_in_k_parts(dists, fname, nparts)
    return(dists)

def neighborhood_analysis(dists, radii, fbase):
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
    scipy.io.savemat(fbase + 'results_find_dim_and_radius.mat', results)
    return(results)

def find_dimension_plot(avg_neighbors, radii, fit_range, fname):
    tickrange = np.append(np.arange(0, len(radii)-1, 10), len(radii)-1)
    try:
        m,b = polyfit(np.log(radii[fit_range]), np.log(avg_neighbors[fit_range]), 1)
    except:
        m = 0
        b = 0
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
    plt.savefig(fname, format='png')  
    return(m)

if __name__ == '__main__':
    # turn this into a main function 
    fbase = '/homes/jmcq/molecule_analysis/clean/results/normed_subset_'
    print('reading in data...')
    fname = 'chembl_22_clean_1575727_ipam_2K_embedding_csc_sparse_col_normed_subset.pkl'
    with open(fname, 'rb') as f:
        X = cPickle.load(f).toarray()
    n, D = X.shape
    rmin = 2
    rmax = 200
    radii = 10**(np.linspace(np.log10(rmin), np.log10(rmax)))
    # num_trees = 8
    # num_checks = 2048
    # print("performing radius analysis...")
    # nparts = 5
    # dists = compute_largest_radius_distance(X, rmax, num_checks, num_trees, fbase, nparts)
    # results = neighborhood_analysis(dists, radii, fbase)
    results = scipy.io.loadmat('/homes/jmcq/molecule_analysis/clean/results/normed_subset_results_find_dim_and_radius.mat')
    avg_neighbors = results['avg_neighbors'].flatten()
    radii = results['radii'].flatten()
    fit_range = range(len(radii))
    fit_range = np.where((radii > 20) & (radii < 30))[0]
    fname = fbase + 'neighbors_dimension_rad_' + str(rmin) + '_' + str(rmax) + '.png'
    print("building dimension plot...")
    dim = find_dimension_plot(avg_neighbors, radii, fit_range, fname)
    print("job complete.")