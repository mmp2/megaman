#!/usr/bin/env python
"""
Benchmarks of geometry functions (distance_matrix, affinity_matrix,
graph_laplacian) in sparse vs dense representation.

First, we fix a training set and increase the number of
samples. Then we plot the computation time as function of
the number of samples.

In the second benchmark, we increase the number of dimensions of the
training set. Then we plot the computation time as function of
the number of dimensions.

TODO: generate the data -- what should it be
      with precomputing of the index?
      
      (todo next: benchmarks for rmetric, for spectral embedding, for other methods like isomap)

"""
import gc  #the garbage collector
from time import time
import numpy as np

from sklearn.datasets.samples_generator import make_swiss_roll

def compute_bench(n_samples, n_features, rad0, dim, quiet = False):

    dense_d_results = []
    dense_a_results = []
    dense_l_results = []
    sparse_fl_results = []
    sparse_d_results = []
    sparse_a_results = []
    sparse_l_results = []

    it = 0
    
    for ns in n_samples:
        # make a dataset
        X, t = make_swiss_roll( ns, noise = 0.0 )
        X = np.asarray( X, order="C" )

        for nf in n_features:
            it += 1
            rad = rad0/ns**(1./(dim+6))  #check the scaling
            if not quiet:
                print('==================')
                print('Iteration %s of %s' % (it, max(len(n_samples),
                                              len(n_features))))
                print('==================')
                print( 'rad=', rad, 'ns=', ns )

            if nf < 3:
                raise ValueError('n_features must be at least 3 for swiss roll')
            else:
                # add noise dimensions up to n_features
                n_noisef = nf - 3
                noise_rad_frac = 0.1
                noiserad = rad/np.sqrt(n_noisef) * noise_rad_frac
                Xnoise = np.random.random((ns, n_noisef)) * noiserad 
                X = np.hstack((X, Xnoise))
                rad = rad*(1+noise_rad_frac) # add a fraction for noisy dimensions

            gc.collect()
            if not quiet:
                print("- benchmarking sparse")
            tstart = time()
            flann = FLANN()
#            params = flann.build_index(X,algorithm='autotuned',target_precision=0.7,sample_fraction=min(10000./ns, 0.1))
            params = flann.build_index(X,algorithm='kmeans', target_precision=0.7)
            sparse_fl_results.append(time() - tstart)
            dists = distance_matrix(X, flindex = flann, mode='radius_neighbors',
                                    neighbors_radius=rad*1.5 )
            sparse_d_results.append(time() - tstart)
            A = affinity_matrix( dists, rad )
            sparse_a_results.append(time() - tstart)
            lap = graph_laplacian(A, normed='geometric', symmetrize=False, scaling_epps=rad, return_lapsym=False)
            sparse_l_results.append(time() - tstart)
            gc.collect()
            if not quiet:
                print("- benchmarking dense")
            tstart = time()
            dists = distance_matrix(X, flindex = None, mode='radius_neighbors',
                                    neighbors_radius=rad*1.5 )
            dense_d_results.append(time() - tstart)
            A = affinity_matrix( dists, rad )
            dense_a_results.append(time() - tstart)
            lap = graph_laplacian(A, normed='geometric', symmetrize=False, scaling_epps=rad, return_lapsym=False)
            dense_l_results.append(time() - tstart)
            gc.collect()
    return sparse_fl_results, sparse_d_results, sparse_a_results, sparse_l_results,  dense_d_results, dense_a_results, dense_l_results

"""
UNFINISHED. Won't work unless I write a distance_matrix that can work
on slices of X.

def compute_bench_precomputed_index(n_samples, n_features, rad0, dim, quiet = False):

    sparse_index_results =[]
    dense_d_results = []
    dense_a_results = []
    dense_l_results = []
    sparse_d_results = []
    sparse_a_results = []
    sparse_l_results = []

    it = 0
    
    nsmax = max( n_samples )
    # make a dataset once, use slices of it to benchmark
    X, t = make_swiss_roll( ns, noise = 0.0 )
    X = np.asarray( X, order="C" )
    rad = rad0/mean( n_samples )**(1./(dim+6)) # a compromise solution
    if n_features < 3:
        raise ValueError('n_features must be at least 3 for swiss roll')
    else:
        # add noise dimensions up to n_features
        n_noisef = n_features - 3
        noise_rad_frac = 0.1
        noiserad = rad/np.sqrt(n_noisef) * noise_rad_frac
        Xnoise = np.random.random((nsmax, n_noisef)) * noiserad 
        Xall = np.hstack((X, Xnoise))
        rad = rad*(1+noise_rad_frac) # add a fraction for noisy dimensio

        flann = FLANN()
        params = flann.build_index(Xall)

    for ns in n_samples:
        it += 1
        if not quiet:
            print('==================')
            print('Iteration %s of %s, ns=%d' % (it,len(n_samples), ns))
            print('==================')
            print("- benchmarking sparse")
        gc.collect()
        tstart = time()
        dists = distance_matrix(X, flindex = flann, mode='radius_neighbors',
                                neighbors_radius=rad*1.5 )
        sparse_d_results.append(time() - tstart)
        A = affinity_matrix( dists, rad )
        sparse_a_results.append(time() - tstart)
        lap = graph_laplacian(A, normed='geometric', symmetrize=False, scaling_epps=rad, return_lapsym=False)
        sparse_l_results.append(time() - tstart)
        gc.collect()
        if not quiet:
            print("- benchmarking dense")
        tstart = time()
        dists = distance_matrix(X, flindex = None, mode='radius_neighbors',
                                neighbors_radius=rad*1.5 )
        dense_d_results.append(time() - tstart)
        A = affinity_matrix( dists, rad )
        dense_a_results.append(time() - tstart)
        lap = graph_laplacian(A, normed='geometric', symmetrize=False, scaling_epps=rad, return_lapsym=False)
        dense_l_results.append(time() - tstart)
        gc.collect()
    return sparse_d_results, sparse_a_results, sparse_l_results,  dense_d_results, dense_a_results, dense_l_results
"""

if __name__ == '__main__':
    import sys
    import os
    path = os.path.abspath('..')
    sys.path.append(path)
#    path = os.path.abspath('../..')
#    sys.path.append(path)
    from Mmani.embedding.geometry import *
    import pylab as pl
    import scipy.io

    is_save = True
    if sys.argv.__len__() > 1:
        is_save = bool(sys.argv[1])

    rad0 = 2.5
    dim = 2
    n_features = 10
    list_n_samples = np.linspace(500, 1000, 2).astype(np.int)
#    list_n_samples = np.logspace(4, 6, 7).astype(np.int)
    sparse_fl_results, sparse_d_results, sparse_a_results, sparse_l_results,  dense_d_results, dense_a_results, dense_l_results = compute_bench(list_n_samples,
                                            [n_features], rad0, dim, quiet=False)

    save_dict = { 'ns_sparse_fl_results':sparse_fl_results, 'ns_sparse_d_results':sparse_d_results, 'ns_sparse_a_results':sparse_a_results,'ns_sparse_l_results':sparse_l_results, 'ns_dense_d_results':dense_d_results, 'ns_dense_a_results':dense_a_results, 'ns_dense_l_results':dense_l_results }
    if is_save:
        scipy.io.savemat( 'results_bench_laplacian_sparse_vs_dense.mat', save_dict )


    pl.figure('Mmani.embedding benchmark results sparse vs dense')
    pl.subplot(211)
    pl.plot(list_n_samples, sparse_fl_results, 'r--',
                            label='flann index')
    pl.plot(list_n_samples, sparse_d_results, 'r-',
                            label='sparse distance matrix')
    pl.plot(list_n_samples, sparse_a_results, 'r:',
                            label='sparse affinity matrix')
    pl.plot(list_n_samples, sparse_d_results, 'r-.',
                            label='sparse laplacian')
    pl.plot(list_n_samples, dense_d_results, 'b-',
                            label='dense distance matrix')
    pl.plot(list_n_samples, dense_a_results, 'b:',
                            label='dense affinity matrix')
    pl.plot(list_n_samples, dense_l_results, 'b-.',
                            label='dense laplacian')
    pl.title('data index built every step, %d features' % (n_features))
    pl.legend(loc='upper left')
    pl.xlabel('number of samples')
    pl.ylabel('Time (s)')
    pl.axis('tight')
    pl.yscale( 'log' )
    pl.xscale( 'log' )

    n_samples = 2000
    list_n_features = np.linspace(50, 1000, 2).astype(np.int)
    sparse_fl_results, sparse_d_results, sparse_a_results, sparse_l_results, dense_d_results, dense_a_results, dense_l_results = compute_bench([n_samples], list_n_features, rad0, dim, quiet=False)
    nf_dict = {'nf_sparse_fl_results':sparse_fl_results, 'nf_sparse_d_results':sparse_d_results, 'nf_sparse_a_results':sparse_a_results, 'nf_sparse_l_results':sparse_l_results, 'nf_dense_d_results':dense_d_results, 'nf_dense_a_results':dense_a_results, 'nf_dense_l_results':dense_l_results }

    save_dict.update( nf_dict )
    if is_save:
        scipy.io.savemat( 'results_bench_laplacian_sparse_vs_dense.mat', save_dict )

    pl.subplot(212)
    pl.plot(list_n_features, sparse_d_results, 'r-',
                            label='sparse distance matrix')
    pl.plot(list_n_features, sparse_a_results, 'r:',
                            label='sparse affinity matrix')
    pl.plot(list_n_features, sparse_l_results, 'r-.',
                            label='sparse laplacian')
    pl.plot(list_n_features, dense_d_results, 'b-',
                            label='dense distance matrix')
    pl.plot(list_n_features, dense_a_results, 'b:',
                            label='dense affinity matrix')
    pl.plot(list_n_features, dense_l_results, 'b-.',
                            label='dense laplacian')
    pl.title('data index built every step, %d samples' % (n_samples))
    pl.legend(loc='upper left')
    pl.xlabel('number of featureses')
    pl.ylabel('Time (s)')
    pl.axis('tight')
    pl.yscale( 'log' )
    pl.xscale( 'log' )

    if is_save:
        pl.savefig('results_bench_laplacian_sparse_vs_dense'+'.png', format='png')
    else:
        pl.show()

