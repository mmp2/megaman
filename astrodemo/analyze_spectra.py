#!/usr/bin/env python
import sys
import h5py
import time
import scipy
import numpy as np
import scipy.io
from scipy.sparse.csgraph import connected_components

# Plotting imports:
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pylab import polyfit

# Mmani imports:
sys.path.append('/home/jmcq/GitHub/Mmani')
from Mmani.geometry.geometry import affinity_matrix
from Mmani.geometry.distance import distance_matrix
from Mmani.geometry.cyflann.index import Index

print("Loading Data...")
fname = '/home/jmcq/GitHub/Mmani/astrodemo/spectra100000_clean.hdf5'
file = h5py.File(fname, 'r')
wavelengths = np.array(file['wavelengths'])
X = np.array(file['spectra'])
print(X.shape)
n_samples = X.shape[0]

is_neighborhoods = True  # dimension = rate of growth of neighborhoods
nradii = 15
backwards = True

is_outliers = False   # find outliers
rad = 500 #~100 spectra with no neighbors 

is_connected = False
rad = 1500

if is_neighborhoods:
    print("evaluating radii...")
    radii = np.logspace(2,5.2,nradii)
    avg_neighbors = np.zeros(nradii)
    min_neighbors = np.zeros(nradii)
    num_no_nbrs = np.zeros(nradii)
    num_components = np.zeros(nradii)
    largest_component_size = np.zeros(nradii)
    print("Building index...")
    cyindex = Index(X)
    for ii in range(nradii):
        print("=========================================")
        print("Step " + str(ii +1) + " of " + str(nradii))
        t0 = time.time()
        if backwards:
            ii = nradii - 1 - ii
            print(ii)
            rad = radii[ii]
            print("radius: " + str(radii[ii]))
            if ii == nradii - 1:
                print('computing distance matrix...')
                dists = distance_matrix(X, method='cython', radius=rad, cyindex=cyindex)
                print("Symmetrizing distance matrix...")
                dists = dists + dists.T # symmetrize, removes zero on diagonal        
            else:
                print("censoring distance matrix...")
                dists.data[dists.data > rad] = 0
                dists.eliminate_zeros()   
        else:
            rad = radii[ii]
            print("radius: " + str(radii[ii]))
            dists = distance_matrix(X, method='cython', radius=rad, cyindex=cyindex)
            print("Symmetrizing distance matrix...")
            dists = dists + dists.T # symmetrize, removes zero on diagonal
        print(dists.nnz)
        avg_neighbors[ii] = dists.data.shape[0]/float(n_samples)
        print('finding neighbors per point...')
        nonzero_per_row = np.split(dists.indices, dists.indptr[1:-1])
        nbrs_per_row = [len(item) for item in nonzero_per_row]
        min_neighbors[ii] = min(nbrs_per_row)
        num_no_nbrs[ii] = nbrs_per_row.count(0)
        print("average # nbrs: " + str(avg_neighbors[ii]))
        print("min # nbrs: " + str(min_neighbors[ii]))
        print("There are " + str(num_no_nbrs[ii]) + " spectra with no neighbors.")
        print("calculating affinity matrix...")
        A = affinity_matrix(dists, rad)
        print("finding connected components...")
        (n_components, labels) = connected_components(A)
        num_components[ii] = n_components
        print("There are: " + str(n_components) + " connected components.")
        num_per_comp = np.zeros(n_components)
        for i in range(n_components):
            num_per_comp[i] = list(labels).count(i)
        largest_component_size[ii] = np.max(num_per_comp)
        print("the largest connected component contains %d of the %d spectra" % (largest_component_size[ii], n_samples))
        if backwards:
            ii = nradii - 1 - ii
        t1 = time.time()
        print("iteration took: " + str(t1 - t0) + " seconds.")
    try:
        m,b = polyfit(np.log(radii), np.log(avg_neighbors), 1)
    except:
        m = 0
        b = 0
    results = {'avg_neighbors':avg_neighbors,
               'min_neighbors':min_neighbors,
               'num_no_neighbors':num_no_nbrs,
               'radii':radii,
               'dim':m,
               'num_components':num_components,
               'largest_component_size':largest_component_size}
    scipy.io.savemat('results_find_dim_and_radius.mat', results)
    plt.scatter(radii, avg_neighbors)
    plt.plot(radii, avg_neighbors, color='red')
    plt.plot(radii, np.exp(b)*radii**m, color='blue')
    plt.plot(radii, min_neighbors, color='green')
    plt.yscale('log')
    plt.xscale('log')
    plt.xlabel('radius')
    plt.ylabel('neighbors')
    plt.title('spectra data dim='+repr(m)[:4])
    plt.grid(b=True,which='minor')
    print('dim=', m )
    plt.savefig("neighbors"+".png", format='png')
    
if is_outliers:
    print("evaluating graph degrees & outliers...")
    print("building index...")
    cyindex = Index(X)
    print("calculating distance...")
    dists = distance_matrix( X, method='cython', radius=rad, cyindex = cyindex )    
    print("symmetrizing distance...")
    dists = dists + dists.T
    print("Average of non-zero distance values: " + str(np.mean(dists.data)))
    avgd = np.asarray(dists.sum(axis=1)).squeeze()/n_samples
    d_mean = np.mean(avgd)
    d_std = np.std(avgd)
    print('radius=' + repr(rad) +': average distances mean (std)=' + 
          repr(d_mean) + '(' + repr(d_std) +')')    
    print("calculating affinity matrix...")
    A = affinity_matrix(dists, rad)
    degrees = np.asarray(A.sum(axis=1)).squeeze()
    deg_mean = np.mean(degrees)
    deg_std = np.std(degrees)
    outlier_results = {'avgd':avgd, 'degrees':degrees,'radius':rad}
    scipy.io.savemat('outlier_results.mat', outlier_results)
    print('average degrees mean (std)=' + repr(deg_mean) + '(' + 
          repr(deg_std) +')')    
    plt.plot(np.sort(degrees))
    plt.title('degrees')
    plt.grid(b=True)
    plt.savefig("outlier_degrees"+".png", format='png')
    
if is_connected:
    print("evaluating connected components...")
    print("building index...")
    cyindex = Index(X)
    print("calculating distance...")
    dists = distance_matrix( X, method='cython', radius=rad, cyindex = cyindex )    
    print("symmetrizing distance...")
    dists = dists + dists.T
    print("calculating affinity matrix...")
    A = affinity_matrix(dists, rad)
    print("finding connected components...")
    (n_components, labels) = connected_components(A)
    print("There are: " + str(n_components) + " connected components.")
    num_per_comp = np.zeros(n_components)
    for i in range(n_components):
        num_per_comp[i] = list(labels).count(i)
        if num_per_comp[i] > 100:
            print("there are " + str(num_per_comp[i]) + " spectra in the " + 
                  str(i) + "th component")
    print("the largest connected component contains %d of the %d spectra" % (np.max(num_per_comp), n_samples))
    components_results = {'n_components':n_components, 'labels':labels, 
                          'num_per_comp':num_per_comp, 'radius':rad}
    scipy.io.savemat('connected_components.mat', components_results)