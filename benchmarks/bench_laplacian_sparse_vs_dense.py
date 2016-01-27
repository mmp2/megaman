#!/usr/bin/env python
"""
First, we fix a training set and increase the number of
samples. Then we plot the computation time as function of
the number of samples.

In the second benchmark, we increase the number of dimensions of the
training set. Then we plot the computation time as function of
the number of dimensions.
"""
import gc  #the garbage collector
from time import time
import numpy as np
from scipy import sparse

from sklearn.datasets.samples_generator import make_swiss_roll

def compute_bench_sparse(n_samples, n_features, rad0, dim, quiet = False):
    sparse_d_results = []
    sparse_a_results = []
    sparse_l_results = []
    sparse_e_results = []
    sparse_r_results = []
    it = 0
    rng = np.random.RandomState(123)
    for ns in n_samples:
        # make a dataset
        X, t = make_swiss_roll( ns, noise = 0.0, random_state = rng)
        X = np.asarray( X, order="C" )

        for nf in n_features:
            it += 1
            rad = rad0/ns**(1./(dim+6))  #check the scaling
            if not quiet:
                print('==================')
                print('Iteration %s of %s' % (it, max(len(n_samples),
                                              len(n_features))))
                print('==================')

            if nf < 3:
                raise ValueError('n_features must be at least 3 for swiss roll')
            else:
                # add noise dimensions up to n_features
                n_noisef = nf - 3
                noise_rad_frac = 0.1
                noiserad = rad/np.sqrt(n_noisef) * noise_rad_frac
                Xnoise = rng.rand(ns, n_noisef) * noiserad
                # Xnoise = np.random.random((ns, n_noisef)) * noiserad 
                X = np.hstack((X, Xnoise))
                rad = rad*(1+noise_rad_frac) # add a fraction for noisy dimensions
            gc.collect()
            if not quiet:
                print("- benchmarking sparse")
                print( 'rad=', rad, 'ns=', ns, 'nf=', nf )
            Geom = Geometry(X, neighborhood_radius = 1.5*rad, affinity_radius = 1.5*rad, 
                            distance_method = 'cython', input_type = 'data', 
                            laplacian_type = 'symmetricnormalized')
            tstart = time()
            dists = Geom.get_distance_matrix(copy=False)
            sparse_d_results.append(time() - tstart)
            A = Geom.get_affinity_matrix(copy = False, symmetrize = True)
            sparse_a_results.append(time() - tstart)
            lap = Geom.get_laplacian_matrix(scaling_epps=rad*1.5, return_lapsym=True,
                                            copy = True)
            sparse_l_results.append(time() - tstart)
            gc.collect()
            embed = spectral_embedding(Geom, n_components = 2, eigen_solver = 'amg')
            sparse_e_results.append(time() - tstart)
            gc.collect()
    return sparse_d_results, sparse_a_results, sparse_l_results, sparse_e_results

def compute_bench_dense(n_samples, n_features, rad0, dim, quiet = False):
    dense_d_results = []
    dense_a_results = []
    dense_l_results = []
    dense_e_results = []
    dense_r_results = []
    it = 0
    rng = np.random.RandomState(123)
    for ns in n_samples:
        # make a dataset
        X, t = make_swiss_roll( ns, noise = 0.0, random_state = rng )
        X = np.asarray( X, order="C" )

        for nf in n_features:
            it += 1
            rad = rad0/ns**(1./(dim+6))  #check the scaling
            if not quiet:
                print('==================')
                print('Iteration %s of %s' % (it, max(len(n_samples),
                                              len(n_features))))
                print('==================')

            if nf < 3:
                raise ValueError('n_features must be at least 3 for swiss roll')
            else:
                # add noise dimensions up to n_features
                n_noisef = nf - 3
                noise_rad_frac = 0.1
                noiserad = rad/np.sqrt(n_noisef) * noise_rad_frac
                Xnoise = rng.rand(ns, n_noisef) * noiserad
                # Xnoise = np.random.random((ns, n_noisef)) * noiserad 
                X = np.hstack((X, Xnoise))
                rad = rad*(1+noise_rad_frac) # add a fraction for noisy dimensions
            gc.collect()
            if not quiet:
                print("- benchmarking dense")
                print( 'rad=', rad, 'ns=', ns, 'nf=', nf )
            Geom = Geometry(X, neighborhood_radius = 1.5*rad, affinity_radius = 1.5*rad, 
                            distance_method = 'brute', input_type = 'data', 
                            laplacian_type = 'symmetricnormalized')
            tstart = time()
            dists = Geom.get_distance_matrix(copy=False)
            dense_d_results.append(time() - tstart)
            A = Geom.get_affinity_matrix(copy = False, symmetrize = True)
            dense_a_results.append(time() - tstart)
            if sparse.isspmatrix(A):
                A.todense()
                Geom.assign_affinity_matrix(A, affinity_radius = rad*1.5)
            lap = Geom.get_laplacian_matrix(scaling_epps=rad*1.5, return_lapsym=True,
                                            copy = False)
            dense_l_results.append(time() - tstart)
            gc.collect()
            embed = spectral_embedding(Geom, n_components = 2, eigen_solver = 'dense')
            dense_e_results.append(time() - tstart)
            gc.collect()
    return dense_d_results, dense_a_results, dense_l_results, dense_e_results
    
def compute_bench_sklearn(n_samples, n_features, rad0, dim, quiet = False):
    sklearn_d_results = []
    sklearn_a_results = []
    sklearn_e_results = []
    it = 0
    rng = np.random.RandomState(123)
    for ns in n_samples:
        # make a dataset
        X, t = make_swiss_roll( ns, noise = 0.0, random_state = rng )
        X = np.asarray( X, order="C" )

        for nf in n_features:
            it += 1
            rad = rad0/ns**(1./(dim+6))  #check the scaling
            if not quiet:
                print('==================')
                print('Iteration %s of %s' % (it, max(len(n_samples),
                                              len(n_features))))
                print('==================')
                print( 'rad=', rad, 'ns=', ns, 'nf=', nf )

            if nf < 3:
                raise ValueError('n_features must be at least 3 for swiss roll')
            else:
                # add noise dimensions up to n_features
                n_noisef = nf - 3
                noise_rad_frac = 0.1
                noiserad = rad/np.sqrt(n_noisef) * noise_rad_frac
                # Xnoise = np.random.random((ns, n_noisef)) * noiserad 
                Xnoise = rng.rand(ns, n_noisef) * noiserad
                X = np.hstack((X, Xnoise))
                rad = rad*(1+noise_rad_frac) # add a fraction for noisy dimensions
            gc.collect()
            if not quiet:
                print("- benchmarking sklearn")
            tstart = time()           
            dists = radius_neighbors_graph(X, radius = rad*1.5, mode = 'distance')
            dists = 0.5 * (dists + dists.T)
            sklearn_d_results.append(time() - tstart)
            # taken from sklearn.metrics.pairwise.rbf_kernel()
            gamma = -1.0/(rad*1.5)
            A = dists.copy()
            A.data = A.data**2
            A.data = A.data/(-(rad*1.5)**2)
            np.exp(A.data,A.data)
            sklearn_a_results.append(time() - tstart)
            embed = se(A, n_components = 2, eigen_solver = 'amg')
            sklearn_e_results.append(time() - tstart)
            gc.collect()
    return sklearn_d_results, sklearn_a_results, sklearn_e_results

if __name__ == '__main__':
    import sys
    import os
    path = '/homes/jmcq/Mmani'
    sys.path.append(path)
    from Mmani.geometry.geometry import *
    from Mmani.geometry.distance import *
    from Mmani.embedding.spectral_embedding import *
    from sklearn.manifold.spectral_embedding_ import spectral_embedding as se
    from sklearn.neighbors.graph import radius_neighbors_graph
    import pylab as pl
    import scipy.io
    
    is_save = True
    vary = 'D'
    print ("D method -- yet even more D")
    
    if vary == 'n':
        if sys.argv.__len__() > 1:
            is_save = bool(sys.argv[1])
        rad0 = 2.5
        dim = 2
        n_features = 10
        # n index parameters:
        start = 500
        stop_1 = 10000
        start_2 = np.int(stop_1*(1.2))
        stop_2 = 100000
        start_3 = np.int(stop_2*(1.2))
        stop_3 = 1000000
        list_n_samples_dense = np.linspace(start, stop_1, 4).astype(np.int)
        list_n_samples_sklearn = np.concatenate((list_n_samples_dense, np.linspace(start_2, stop_2, 4).astype(np.int)),axis=0)
        list_n_samples_sparse = np.concatenate((list_n_samples_sklearn, np.linspace(start_3, stop_3, 5).astype(np.int)),axis=0)
        
        list_n_samples_sklearn = list_n_samples_sparse
        
        dense_d_results, dense_a_results, dense_l_results, dense_e_results = compute_bench_dense(list_n_samples_dense,
                                                [n_features], rad0, dim, quiet=False)
        sklearn_d_results, sklearn_a_results, sklearn_e_results = compute_bench_sklearn(list_n_samples_sklearn,
                                                [n_features], rad0, dim, quiet=False)
        sparse_d_results, sparse_a_results, sparse_l_results, sparse_e_results = compute_bench_sparse(list_n_samples_sparse,
                                                [n_features], rad0, dim, quiet=False)
        
        save_dict = { 'ns_sparse_d_results':sparse_d_results, 'ns_sparse_a_results':sparse_a_results,
                      'ns_sparse_l_results':sparse_l_results, 'ns_sparse_e_results': sparse_e_results,
                      'ns_dense_d_results':dense_d_results, 'ns_dense_a_results':dense_a_results, 
                      'ns_dense_l_results':dense_l_results, 'ns_dense_e_results':dense_e_results,
                      'ns_sklearn_d_results':sklearn_d_results, 'ns_sklearn_a_results':sklearn_a_results, 
                      'ns_sklearn_e_results':sklearn_e_results,
                      'list_n_samples':list_n_samples_sparse}
        if is_save:
            scipy.io.savemat( 'results_bench_N_sparse_vs_dense_vs_sklearn.mat', save_dict )
        
        pl.figure('Mmani.embedding benchmark results sparse vs dense')
        pl.subplot(111)
        pl.plot(list_n_samples_sparse, sparse_d_results, 'r-',
                                label='sparse distance matrix')
        pl.plot(list_n_samples_sparse, sparse_e_results, 'r--',
                                label='sparse embedding')
        pl.plot(list_n_samples_dense, dense_d_results, 'b-',
                                label='dense distance matrix')
        pl.plot(list_n_samples_dense, dense_e_results, 'b--',
                                label='dense embedding')
        pl.plot(list_n_samples_sklearn, sklearn_d_results, 'k-',
                                label='sklearn distance matrix')
        pl.plot(list_n_samples_sklearn, sklearn_e_results, 'k--',
                                label='sklearn embedding')
        pl.title('data index built every step, %d features' % (n_features))
        pl.legend(loc='lower right', prop={'size':5})
        pl.xlabel('number of samples')
        pl.ylabel('Time (s)')
        pl.axis('tight')
        pl.yscale( 'log' )
        pl.xscale( 'log' )
        if is_save:
            pl.savefig('results_bench_N_sparse_vs_dense_vs_sklearn'+'.png', format='png')
        else:
            pl.show()
    else:    
        rad0 = 2.5
        dim = 2
        n_samples = 100000
        list_n_features_dense = np.linspace(50, 1000, 4).astype(np.int)
        list_n_features_sklearn = np.concatenate((list_n_features_dense, np.linspace(1500, 3000, 3).astype(np.int)),axis=0)
        list_n_features_sparse = np.concatenate((list_n_features_sklearn, np.linspace(4000, 100000, 6).astype(np.int)),axis=0)
        list_n_features_sklearn = list_n_features_sparse
        dense_d_results = []; dense_a_results = []; dense_e_results = []; dense_l_results = [];
        sparse_d_results, sparse_a_results, sparse_l_results, sparse_e_results = compute_bench_sparse([n_samples], list_n_features_sparse, rad0, dim, quiet=False)
        # dense_d_results, dense_a_results, dense_l_results, dense_e_results = compute_bench_dense([n_samples], list_n_features_dense, rad0, dim, quiet=False)
        sklearn_d_results, sklearn_a_results, sklearn_e_results = compute_bench_sklearn([n_samples], list_n_features_sklearn, rad0, dim, quiet=False)
        save_dict = {'nf_sparse_d_results':sparse_d_results, 'nf_sparse_a_results':sparse_a_results, 
                    'nf_sparse_l_results':sparse_l_results, 'nf_sparse_e_results':sparse_e_results,
                    'nf_dense_d_results':dense_d_results, 'nf_dense_a_results':dense_a_results, 
                    'nf_dense_l_results':dense_l_results, 'nf_dense_e_results':dense_e_results,
                    'nf_sklearn_d_results':sklearn_d_results, 'nf_sklearn_a_results':sklearn_a_results, 
                    'nf_sklearn_e_results':sklearn_e_results,
                    'list_n_features': list_n_features_sparse}
        
        if is_save:
            scipy.io.savemat( 'results_bench_D_sparse_vs_dense_vs_sklearn.mat', save_dict )
        
        pl.subplot(111)
        pl.plot(list_n_features_sparse, sparse_d_results, 'r-',
                                label='sparse distance matrix')
        pl.plot(list_n_features_sparse, sparse_e_results, 'r--',
                                label='sparse embedding')
        pl.plot(list_n_features_dense, dense_d_results, 'b-',
                                label='dense distance matrix')
        pl.plot(list_n_features_dense, dense_e_results, 'b--',
                                label='dense embedding')
        pl.plot(list_n_features_sklearn, sklearn_d_results, 'k-',
                                label='sklearn distance matrix')
        pl.plot(list_n_features_sklearn, sklearn_e_results, 'k--',
                                label='sklearn embedding')
        pl.title('data index built every step, %d samples' % (n_samples))
        pl.legend(loc='lower right', prop={'size':7})
        pl.xlabel('number of featureses')
        pl.ylabel('Time (s)')
        pl.axis('tight')
        pl.yscale( 'log' )
        pl.xscale( 'log' )
        
        if is_save:
            pl.savefig('results_bench_D_sparse_vs_dense_vs_sklearn'+'.png', format='png')
        else:
            pl.show()