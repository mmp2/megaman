import numpy as np
import cPickle
import sys
from megaman.geometry.rmetric import riemann_metric_lazy
from megaman.geometry.affinity import compute_affinity_matrix
from megaman.geometry.laplacian import compute_laplacian_matrix
from large_sparse_functions import * 
from scipy.io import loadmat, mmread
from scipy.sparse import issparse, csr_matrix
import multiprocessing as mp


def compute_laplacian_by_row(A, sample, radius):
    d = A.sum(0)
    nbrs = np.split(A.indices, A.indptr[1:-1])
    L = [laplacian_i_row(A, d, nbrs, i) for i in sample]
    return(L)
    
def laplacian_i_row(A, d, nbrs, i):
    dt = np.sum((A[i,nbrs]/d[nbrs]))
    Li = [compute_Lij(A[i, j], d[j], dt) for j in nbrs]
    
def compute_Lij(Aij, dj, dt):
    Lij = Aij / (dj * dt)
    return(Lij)

def compute_nbr_wts(A, sample):
    Ps = list()
    nbrs = list()
    for ii in range(len(sample)):
        w = np.array(A[sample[ii],:].todense()).flatten()
        p = w / np.sum(w)
        nbrs.append(np.where(p > 0)[0])
        Ps.append(p[nbrs[ii]])
    return(Ps, nbrs)

def project_tangent(X, p, nbr, d):
    Z = (X[nbr, :] - np.dot(p, X[nbr, :]))*p[:, np.newaxis]
    sig = np.dot(Z.transpose(), Z)
    e_vals, e_vecs = np.linalg.eigh(sig)
    j = e_vals.argsort()[::-1]  # Returns indices that will sort array from greatest to least.
    e_vec = e_vecs[:, j]
    e_vec = e_vec[:, :d]  # Gets d largest eigenvectors.
    X_tangent = np.dot(X[nbr,:], e_vec)  # Project local points onto tangent plane
    return(X_tangent)

def distortion(X, L, sample, PS, nbrs, n, d):
    n = len(sample)
    dist = 0.0
    nsum = n
    for i in range(n):
        p = PS[i]
        nbr = nbrs[i]
        if len(nbr) > 1:
            X_t = project_tangent(X, p, nbr, d)
            X_tangent = X.copy()
            X_tangent = X_tangent[:,range(d)]
            X_tangent[nbr,:] = X_t # rmetric only depends on nbrs 
            H = riemann_metric_lazy(X_tangent,sample,L,d)[0]
            dist += np.linalg.norm(H[i, :, :] - np.eye(d))
        else:
            nsum -= 1
    if nsum > 0:
        distortion = dist/nsum
    else:
        distortion = 'Inf' 
    return distortion
    
def evaluate_radius(radius, d, sample):
    global Dists
    global X
    t0 = time.time()
    D = Dists.copy()
    (n, dim) = X.shape
    D.data[D.data > radius] = 0.0
    D.eliminate_zeros()    
    r = radius / 3.0
    affinity_kwds = {'radius':r}
    A = compute_affinity_matrix(D, 'gaussian', **affinity_kwds)
    (PS, nbrs) = compute_nbr_wts(A, sample)
    L = compute_laplacian_matrix(A, method = 'geometric', scaling_epps = r)
    L = L.tocsr()
    e_dist = distortion(X, L, sample, PS, nbrs, n, d)
    t1 = time.time()
    print("for radius: " + str(radius) + " distortion is: " + str(e_dist))
    print("for radius: " + str(radius) + " analysis took: " + str(t1-t0) + " seconds\n")
    return(radius, e_dist)
    
def radius_search(d, sample, rmin, rmax, ntry):
    print("performing radius search...\n")
    radii = np.linspace(rmin, rmax, ntry)
    results = np.array([evaluate_radius(r, d, sample) for r in radii])
    return(results)    
    
def multi_process_radius_search(d, sample, rmin, rmax, ntry, processes):
    print("performing parallel radius search...\n")
    radii = np.linspace(rmin, rmax, ntry)
    pool = mp.Pool(processes = processes)
    results = [pool.apply_async(evaluate_radius, args = (r,d,sample)) for r in radii]
    results = [p.get() for p in results]
    results.sort()
    return(results)
    
if __name__ == '__main__':
    # turn this into a main function 
    
    import time
    from numpy.random import RandomState
    
    # see the random choice
    rng = RandomState(98107)
    
    # parameters:
    d = 3
    rmin = 40
    rmax = 400
    ntry = 20
    multi = True
    fbase = '/homes/jmcq/astro/'
    
    nsam = int(200)
    print('using real data...')
    fname = fbase + 'combined_spectra_distance_radius_35000_trees_100.mat'
    print("loading in distances...")
    Dists = loadmat(fname)
    Dists = Dists['distmatrix']
    Dists = Dists.tocsr()
    Dists.data[Dists.data > rmax] = 0.0
    Dists.eliminate_zeros()    
    sample_info = loadmat('spectra_pc_data.mat')
    pc_points = sample_info['pc_points_all_index'].flatten()
    # sample = np.random.choice(pc_points, nsam, replace = False) # choose from principal curves 
    sample = rng.choice(pc_points, nsam, replace = False) # choose from principal curves 
    print("reading in data...")
    fname = 'combined_spectra_clean.mtx'
    X = mmread(fname)
    if issparse(X):
        X = X.todense()
    n, dim = X.shape
    
    '''
    print('using artificial data...')
    X = rng.randn(1000, 10)
    from scipy.spatial.distance import squareform, pdist
    Dists = csr_matrix(squareform(pdist(X)))
    sample = range(100)
    '''
    print('using d = ' + str(d))
    t0 = time.time()
    if multi:
        ncpu = mp.cpu_count() # 32 for Newton
        processes = int(min(ntry, ncpu))
        print('using ' + str(processes) + ' processes to perform asynchronous parallel radius search')
        results = multi_process_radius_search(d, sample, rmin, rmax, ntry, processes)
    else:
        results = radius_search(d, sample, rmin, rmax, ntry)
    t1 = time.time()
    print('analysis took: ' + str(t1 - t0) + ' seconds to complete.')
    print(results)
    cPickle.dump(results, open(fbase + 'radius_search_d_3.p', 'wb'))