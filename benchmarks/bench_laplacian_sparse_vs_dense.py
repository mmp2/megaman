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

def generate_data(ns, nf, rng, rad0):
    X, t = make_swiss_roll( ns, noise = 0.0, random_state = rng)
    X = np.asarray( X, order="C" )
    rad = rad0/ns**(1./(dim+6))  #check the scaling
    method = 2
    if nf < 3:
        raise ValueError('n_features must be at least 3 for swiss roll')
    else:
        if method == 1:
            # Method one: noise dimensions adjust 
            # add noise dimensions up to n_features
            n_noisef = nf - 3
            noise_rad_frac = 0.1
            noiserad = rad/np.sqrt(n_noisef) * noise_rad_frac
            Xnoise = rng.rand(ns, n_noisef) * noiserad / nf
            X = np.hstack((X, Xnoise))
            rad = rad*(1+noise_rad_frac) # add a fraction for noisy dimensions
        else:
            n_noisef = nf - 3
            Xnoise = rng.rand(ns, n_noisef)
            X = np.hstack((X, Xnoise))
            # rad += (np.sqrt(np.sqrt(n_noisef)) - 2)*1.5
            rad += (np.sqrt(np.sqrt(n_noisef)) - 2)
            if nf > 2000:
                rad = rad*1.5
            if nf > 5000:
                rad = rad*1.5
        # Method two: noise dimensions do not adjust 
    return X, rad

def compute_bench(n_samples, n_features, rad0, dim, method, quiet = False):
    d_results = []
    a_results = []
    l_results = []
    e_results = []
    r_results = []
    it = 0
    rng = np.random.RandomState(123)
    for ns in n_samples:
        for nf in n_features:
            it += 1
            if not quiet:
                print('==================')
                print('Iteration %s of %s' % (it, max(len(n_samples),
                                              len(n_features))))
                print('==================')
            (X, rad) = generate_data(ns, nf, rng, rad0)
            gc.collect()
            if not quiet:
                print( 'rad=', rad, 'ns=', ns, 'nf=', nf )
            
            # Set up method:
            if method == 'sparse':
                if not quiet:
                    print("- benchmarking sparse")
                Geom = Geometry(X, neighborhood_radius = 1.5*rad, affinity_radius = 1.5*rad, 
                                distance_method = 'cython', input_type = 'data', 
                                laplacian_type = 'symmetricnormalized')
            elif method == 'dense':
                if not quiet:
                    print("- benchmarking dense")
                Geom = Geometry(X, neighborhood_radius = 1.5*rad, affinity_radius = 1.5*rad, 
                                distance_method = 'brute', input_type = 'data', 
                                laplacian_type = 'symmetricnormalized')
            else:
                if not quiet:
                    print("- benchmarking sklearn")
            
            # Distance matrix:
            tstart = time()
            if method == 'sparse' or method == 'dense':
                dists = Geom.get_distance_matrix(copy=False)
            else: 
                dists = radius_neighbors_graph(X, radius = rad*1.5, mode = 'distance')
                dists = 0.5 * (dists + dists.T)
            d_results.append(time() - tstart)
            print(d_results[len(d_results)-1])
                
            # Affinity Matrix
            if method == 'sparse' or method == 'dense':
                print("skipping A")
                # A = Geom.get_affinity_matrix(copy = False, symmetrize = True)
            else:
                gamma = -1.0/(rad*1.5)
                A = dists.copy()
                A.data = A.data**2
                A.data = A.data/(-(rad*1.5)**2)
                np.exp(A.data,A.data)                
            a_results.append(time() - tstart)
            
            # Laplacian Matrix: 
            if method == 'sparse' or method == 'dense':
                # lap = Geom.get_laplacian_matrix(scaling_epps=rad*1.5, return_lapsym=True,
                                                # copy = True)
                print("Skipping L")
            l_results.append(time() - tstart)
            gc.collect()
            
            # Embedding:
            if not quiet:
                print("embedding...")
            if method == 'sparse':
                embed = spectral_embedding(Geom, n_components = 2, eigen_solver = 'amg')
            elif method == 'dense':
                embed = spectral_embedding(Geom, n_components = 2, eigen_solver = 'dense')
            else:
                embed = se(A, n_components = 2, eigen_solver = 'amg')
            e_results.append(time() - tstart)
            gc.collect()
    return d_results, a_results, l_results, e_results

if __name__ == '__main__':
    import sys
    import os
    path = '/home/jmcq/GitHub/Mmani'
    sys.path.append(path)
    from Mmani.geometry.geometry import *
    from Mmani.geometry.distance import *
    from Mmani.embedding.spectral_embedding import *
    from sklearn.manifold.spectral_embedding_ import spectral_embedding as se
    from sklearn.neighbors.graph import radius_neighbors_graph
    import matplotlib
    matplotlib.use('Agg')
    import pylab as pl
    
    import scipy.io
    
    is_save = True
    rad0 = 3
    rad0 = 2
    dim = 2
    
    if sys.argv > 1:
        try:
            input_vary = str(sys.argv[1])
            if input_vary in ['D', 'n']:
                vary = input_vary
            elif input_vary == 'N':
                vary = 'n'
            else:
                vary = 'n'
        except:
            vary = 'n'
            
    print(vary)
    
    if vary == 'n':
        print("N = 5,000,000 sparse, 1,000,000 sklearn;  D = 100, sparse only")
        n_features = 100
        rad0 = 2
        list_n_samples_dense = []
        list_n_samples_sparse = []
        list_n_samples_sklearn = []
        save_dict = {'n_features':n_features}
        
        if len(list_n_samples_sparse) > 0:
            rad0 = 20
            sparse_d_results, sparse_a_results, sparse_l_results, sparse_e_results = compute_bench(list_n_samples_sparse,
                                                    [n_features], rad0, dim, method = 'sparse', quiet=False)
            sparse_dict = { 'ns_sparse_d_results':sparse_d_results, 'ns_sparse_a_results':sparse_a_results,
                            'ns_sparse_l_results':sparse_l_results, 'ns_sparse_e_results': sparse_e_results,
                            'sparse_n_samples':list_n_samples_sparse}
            scipy.io.savemat('results_bench_N_sparse.mat', sparse_dict)
            save_dict.update(sparse_dict)        
        if len(list_n_samples_dense) > 0:
            dense_d_results, dense_a_results, dense_l_results, dense_e_results = compute_bench(list_n_samples_dense,
                                                    [n_features], rad0, dim, method = 'dense', quiet=False)
            dense_dict = {'ns_dense_d_results':dense_d_results, 'ns_dense_a_results':dense_a_results, 
                          'ns_dense_l_results':dense_l_results, 'ns_dense_e_results':dense_e_results,
                          'dense_n_samples':list_n_samples_dense}
            scipy.io.savemat('results_bench_N_dense.mat', dense_dict)
            save_dict.update(dense_dict)        
        if len(list_n_samples_sklearn) > 0:
            rad0 = 20
            sklearn_d_results, sklearn_a_results, sklearn_l_results, sklearn_e_results = compute_bench(list_n_samples_sklearn,
                                                    [n_features], rad0, dim, method = 'sklearn', quiet=False)
            sklearn_dict = {'ns_sklearn_d_results':sklearn_d_results, 'ns_sklearn_a_results':sklearn_a_results, 
                            'ns_sklearn_l_results':sklearn_l_results, 'ns_sklearn_e_results':sklearn_e_results,
                            'sklearn_n_samples':list_n_samples_sklearn}
            scipy.io.savemat('results_bench_N_sklearn.mat', sklearn_dict)
            save_dict.update(sklearn_dict)
        
        
        # if is_save:
            # scipy.io.savemat( 'results_bench_N_sparse_vs_dense_vs_sklearn.mat', save_dict )
        
        # pl.figure('Mmani.embedding benchmark results sparse vs dense')
        # pl.subplot(111)
        # if len(list_n_samples_sparse) > 0:
            # pl.plot(list_n_samples_sparse, sparse_e_results, 'r--',
                                    # label='sparse embedding')
        # if len(list_n_samples_dense) > 0:
            # pl.plot(list_n_samples_dense, dense_e_results, 'b--',
                                    # label='dense embedding')
        # if len(list_n_samples_sklearn) > 0:
            # pl.plot(list_n_samples_sklearn, sklearn_e_results, 'k--',
                                    # label='sklearn embedding')
        # pl.title('data index built every step, %d features' % (n_features))
        # pl.legend(loc='lower right', prop={'size':5})
        # pl.xlabel('number of samples')
        # pl.ylabel('Time (s)')
        # pl.axis('tight')
        # pl.yscale( 'log' )
        # pl.xscale( 'log' )
        # if is_save:
            # pl.savefig('results_bench_N_sparse_vs_dense_vs_sklearn'+'.png', format='png')
        # else:
            # pl.show()
    else:
        print("Benchmark D. N = 50,000. D = 10 - 4,000 sparse only smaller radius")
        rad0 = 2.5
        rad0 = 2
        rad0 = 20
        n_samples = 50000
        list_n_features_dense = []
        list_n_features_sparse = [10000]
        list_n_features_sklearn = [10000]
        save_dict = {'n_samples':n_samples}
        
        if len(list_n_features_dense) > 0:
            dense_d_results, dense_a_results, dense_l_results, dense_e_results = compute_bench([n_samples],
                                                    list_n_features_dense, rad0, dim, method = 'dense', quiet=False)
            dense_dict = {'nf_dense_d_results':dense_d_results, 'nf_dense_a_results':dense_a_results, 
                          'nf_dense_l_results':dense_l_results, 'nf_dense_e_results':dense_e_results,
                          'dense_nf_samples':list_n_features_dense}
            scipy.io.savemat('results_bench_D_dense.mat', dense_dict)
            save_dict.update(dense_dict)
        if len(list_n_features_sparse) > 0:
            rad0 = 20
            sparse_d_results, sparse_a_results, sparse_l_results, sparse_e_results = compute_bench([n_samples],
                                                    list_n_features_sparse, rad0, dim, method = 'sparse', quiet=False)
            sparse_dict = { 'ns_sparse_d_results':sparse_d_results, 'ns_sparse_a_results':sparse_a_results,
                            'ns_sparse_l_results':sparse_l_results, 'ns_sparse_e_results': sparse_e_results,
                            'sparse_n_samples':list_n_features_sparse}
            scipy.io.savemat('results_bench_D_sparse.mat', sparse_dict)
            save_dict.update(sparse_dict)
        if len(list_n_features_sklearn) > 0:
            rad0 = 2
            sklearn_d_results, sklearn_a_results, sklearn_l_results, sklearn_e_results = compute_bench([n_samples],
                                                    list_n_features_sklearn, rad0, dim, method = 'sklearn', quiet=False)
            sklearn_dict = {'ns_sklearn_d_results':sklearn_d_results, 'ns_sklearn_a_results':sklearn_a_results, 
                            'ns_sklearn_l_results':sklearn_l_results, 'ns_sklearn_e_results':sklearn_e_results,
                            'sklearn_n_samples':list_n_features_sklearn}
            scipy.io.savemat('results_bench_D_sklearn.mat', sklearn_dict)
            save_dict.update(sklearn_dict)
        
        if is_save:
            scipy.io.savemat( 'results_bench_D_sparse_vs_dense_vs_sklearn.mat', save_dict )
        
        pl.figure('Mmani.embedding benchmark results sparse vs dense')
        pl.subplot(111)
        if len(list_n_features_sparse) > 0:
            pl.plot(list_n_features_sparse, sparse_e_results, 'r--',
                                    label='sparse embedding')
        if len(list_n_features_dense) > 0:
            pl.plot(list_n_features_dense, dense_e_results, 'b--',
                                    label='dense embedding')
        if len(list_n_features_sklearn) > 0:
            pl.plot(list_n_features_sklearn, sklearn_e_results, 'k--',
                                    label='sklearn embedding')        
        pl.title('data index built every step, %d samples' % (n_samples))
        pl.legend(loc='lower right', prop={'size':5})
        pl.xlabel('number of features')
        pl.ylabel('Time (s)')
        pl.axis('tight')
        pl.yscale( 'log' )
        pl.xscale( 'log' )
        
        if is_save:
            pl.savefig('results_bench_D_sparse_vs_dense_vs_sklearn'+'.png', format='png')
        else:
            pl.show()
