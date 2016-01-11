#!/usr/bin/env python
import sys
import os
import time
import argparse

import numpy as np
sys.path.append('/homes/jmcq/Mmani/') # this is stupid 
from Mmani.geometry.distance import distance_matrix

BENCH_FUNCTIONS = ['distance_cy', 'distance_py', 'distance_brute',
                    'spectral', 'isomap', 'ltsa', 'lle']
VARY_PARAMETERS = ['N', 'D', 'd', 'radius']
SEED = 36
D_FIX = 100
d_FIX = 2
N_FIX = 5000
RAD_FIX = 1.3

def generate_data(random_state, N, D):
    # generate a data set uniformly distributed on a unit (D-1)-sphere
    # e.g. D = 2 gives a circle, D = 3 gives a sphere etc.
    X = random_state.randn(N, D)
    dist = np.sqrt(np.sum(X**2, 1))
    X = X / dist[:, None]
    return X

def time_function(X, method, radius, d = None, random_state = None):
    if method == 'distance_cy':
        t0 = time.time()
        dmat = distance_matrix(X, method = 'cyflann', radius = radius)
        t1 = time.time()
        return (t1 - t0)
    if method == 'distance_py':
        path_to_flann = '/homes/jmcq/flann-1.8.4-src/src/python'
        sys.path.insert(0, path_to_flann)
        import pyflann as pyf
        flindex = pyf.FLANN()
        flparams = flindex.build_index(X, algorithm = 'kmeans', target_precision = 0.9)
        t0 = time.time()
        dmat = distance_matrix(X, method = 'pyflann', radius = radius, flindex = flindex)
        t1 = time.time()
        return (t1 - t0)
    if method == 'distance_brute':
        t0 = time.time()
        dmat = distance_matrix(X, method = 'brute', radius = radius)
        t1 = time.time()
        return (t1 - t0)
    elif method == 'spectral':
        from Mmani.embedding.spectral_embedding import SpectralEmbedding
        t0 = time.time()
        SE = SpectralEmbedding(n_components=d, eigen_solver='amg',random_state=random_state,
                                neighborhood_radius=radius, distance_method='cyflann',
                                input_type='data',laplacian_type='geometric')
        embedding = SE.fit_transform(X)
        t1 = time.time()
        return (t1 - t0)
    elif method == 'isomap':
        from Mmani.embedding.isomap import Isomap
        t0 = time.time()
        try:
            isomap = Isomap(n_components=d, eigen_solver='amg',random_state=random_state,
                            neighborhood_radius=radius, distance_method='cyflann',
                            input_type='data')
            embedding = isomap.fit_transform(X)
        except:
            isomap = Isomap(n_components=d, eigen_solver='arpack',random_state=random_state,
                            neighborhood_radius=radius, distance_method='cyflann',
                            input_type='data')
            embedding = isomap.fit_transform(X)        
        t1 = time.time()
        return (t1 - t0)
    elif method == 'ltsa':
        from Mmani.embedding.ltsa import LTSA
        t0 = time.time()
        ltsa = LTSA(n_components=d, eigen_solver='amg',random_state=random_state,
                    neighborhood_radius=radius, distance_method='cyflann',
                    input_type='data')
        embedding = ltsa.fit_transform(X)
        t1 = time.time()
        return (t1 - t0)
    elif method == 'lle':
        from Mmani.embedding.locally_linear import LocallyLinearEmbedding
        t0 = time.time()
        lle = LocallyLinearEmbedding(n_components=d, eigen_solver='amg',random_state=random_state,
                                    neighborhood_radius=radius, distance_method='cyflann',
                                    input_type='data')
        embedding = lle.fit_transform(X)
        t1 = time.time()
        return (t1 - t0)
    else:
        raise ValueError('benchmark method: ' + str(method) + ' not found.')
        return None

def run_single_benchmark(bench_function, vary_parameter, par_min, par_max, par_len, seed):
    # base parameters
    head_line = 'Benchmarking function: ' + bench_function + '\n' 
    
    # set-up benchmarking parameters depending on which parameter is varying:
    if vary_parameter == 'N':
        fixed_par_line = 'Fixed D: ' + str(D_FIX) + '. Fixed radius: ' + str(RAD_FIX) +  '. Fixed d: ' + str(d_FIX) + '.\n'
        Ns = np.array(np.exp(np.linspace(np.log(int(par_min)), np.log(int(par_max)), num = par_len)), dtype = 'int')
        Ds = np.repeat(D_FIX, par_len)
        radii = np.repeat(RAD_FIX, par_len)
        ds = np.repeat(d_FIX, par_len)
    if vary_parameter == 'D':
        fixed_par_line = 'Fixed N: ' + str(N_FIX) + '. Fixed radius: ' + str(RAD_FIX) +  '. Fixed d: ' + str(d_FIX) + '.\n'
        Ns = np.repeat(N_FIX, par_len)
        Ds = np.array(np.exp(np.linspace(np.log(int(par_min)), np.log(int(par_max)), num = par_len)), dtype = 'int')
        radii = np.repeat(RAD_FIX, par_len)
        ds = np.repeat(d_FIX, par_len)
    if vary_parameter == 'radius':
        fixed_par_line = 'Fixed D: ' + str(D_FIX) + '. Fixed N: ' + str(N_FIX) +  '. Fixed d: ' + str(d_FIX) + '.\n'
        Ns = np.repeat(N_FIX, par_len)
        Ds = np.repeat(D_FIX, par_len)
        radii = np.linspace(par_min, par_max, num = par_len)
        ds = np.repeat(d_FIX, par_len)
    if vary_parameter == 'd':
        fixed_par_line = 'Fixed D: ' + str(par_max + 1) + '. Fixed radius: ' + str(RAD_FIX) +  '. Fixed N: ' + str(N_FIX) + '.\n'
        Ns = np.repeat(N_FIX, par_len)
        Ds = np.repeat(par_max + 1, par_len) # must be at least max(ds)
        radii = np.repeat(RAD_FIX, par_len)
        ds = np.linspace(int(par_min), int(par_max), num = par_len, dtype = 'int')        
    
    random_state = np.random.RandomState(int(seed))
    # output file:
    fname = "output_time_" + bench_function + "_" + vary_parameter + "_" + str(par_min) + "_" + \
            str(par_max) + "_" + str(par_len) + "_" + str(seed) + ".out"
    # run the benchmarking
    with open(fname, 'wb') as output:
        output.write(head_line) # what are we benchmarking
        output.write(fixed_par_line) # what are the fixed parameters
        for i in xrange(par_len):
            try:
                N = int(Ns[i])
                D = int(Ds[i])
                radius = radii[i]
                d = int(ds[i])
                print 'generating data'
                X = generate_data(random_state, N, D)
                print 'running function'
                ftime = time_function(X, method = bench_function, radius = radius, d = d, random_state = random_state)
                if vary_parameter == 'N':
                    current_param = int(N)
                elif vary_parameter == 'D':
                    current_param = int(D)
                elif vary_parameter == 'radius':
                    current_param = radius
                else:
                    current_param = int(d)
                line = vary_parameter + ': ' + str(current_param) + ', Time: ' + str(ftime) + '\n'
            except:
                e = sys.exc_info()[0]
                line = str(N) + ': failed with error ' + str(e) + '\n'
            output.write(line)
            output.flush()
            os.fsync(output)# do arg parsing


def check_par_lengths(args):
    vals = args.values()
    # print vals
    lens = [len(val) for val in vals if val is not None]
    max_inputs = max(lens)
    checked_args = {}
    for key in args.keys():
        if args[key] is None:
            if key == 'seed':
                checked_args[key] = list(np.repeat(SEED, max_inputs))
            else:
                raise ValueError('unknown None parameter:' + key)
        elif len(args[key]) == max_inputs:
            checked_args[key] = args[key]
        else:
            checked_args[key] = list(np.repeat(args[key][0], max_inputs))
    return checked_args
            
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description = 'Run benchmarker on Mmani package')
    parser.add_argument('-function', nargs = '*', choices = BENCH_FUNCTIONS,
                        help = 'which function to benchmark', required = True)
    parser.add_argument('-parameter', nargs = '*', choices = VARY_PARAMETERS, 
                        help = 'which parameter to vary', required = True)
    parser.add_argument('-par_min', nargs = '*', type = float, 
                        help = 'parameter minimum', required = True)
    parser.add_argument('-par_max', nargs = '*', type = float, 
                        help = 'parameter maximum', required = True)
    parser.add_argument('-par_len', nargs = '*', type = int, required = True,
                        help = 'number of parameters to test between minimum and maximum')
    parser.add_argument('-seed', nargs = '*', type = int,
                        help = 'seed for random number generating')
    args = vars(parser.parse_args())
    checked_args = check_par_lengths(args)
    
    functions = checked_args['function']
    parameters = checked_args['parameter']
    mins = checked_args['par_min']
    maxs = checked_args['par_max']
    lens = checked_args['par_len']
    seeds = checked_args['seed']
    num_to_test = len(functions)
    
    for i in range(num_to_test):
        # call run_single_benchmark for each bench_function passed 
        run_single_benchmark(functions[i], parameters[i], int(mins[i]), int(maxs[i]), lens[i], seeds[i])    