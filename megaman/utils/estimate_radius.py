from __future__ import division
import numpy as np
import sys, time
from megaman.geometry.rmetric import riemann_metric_lazy
from megaman.geometry.affinity import compute_affinity_matrix
from megaman.geometry.laplacian import compute_laplacian_matrix
import multiprocessing as mp
from multiprocessing.pool import ThreadPool

# GLOBALS
DIST = None
X = None

def compute_laplacian_by_row(A, sample, radius):
    d = A.sum(0)
    nbrs = np.split(A.indices, A.indptr[1:-1])
    L = [laplacian_i_row(A, d, nbrs, i) for i in sample]
    return L

def laplacian_i_row(A, d, nbrs, i):
    dt = np.sum((A[i,nbrs]/d[nbrs]))
    Li = [compute_Lij(A[i, j], d[j], dt) for j in nbrs]

def compute_Lij(Aij, dj, dt):
    Lij = Aij / (dj * dt)
    return Lij

def compute_nbr_wts(A, sample):
    Ps = list()
    nbrs = list()
    for ii in range(len(sample)):
        w = np.array(A[sample[ii],:].todense()).flatten()
        p = w / np.sum(w)
        nbrs.append(np.where(p > 0)[0])
        Ps.append(p[nbrs[ii]])
    return Ps, nbrs

def project_tangent(X, p, nbr, d):
    Z = (X[nbr, :] - np.dot(p, X[nbr, :]))*p[:, np.newaxis]
    sig = np.dot(Z.transpose(), Z)
    e_vals, e_vecs = np.linalg.eigh(sig)
    j = e_vals.argsort()[::-1]  # Returns indices that will sort array from greatest to least.
    e_vec = e_vecs[:, j]
    e_vec = e_vec[:, :d]  # Gets d largest eigenvectors.
    X_tangent = np.dot(X[nbr, :], e_vec)  # Project local points onto tangent plane
    return X_tangent

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
            X_tangent = X_tangent[:, :d]
            X_tangent[nbr, :] = X_t # rmetric only depends on nbrs
            H = riemann_metric_lazy(X_tangent, sample, L, d)[0]
            dist += np.linalg.norm(H[i, :, :] - np.eye(d), ord=2)
        else:
            nsum -= 1
    if nsum > 0:
        distortion = dist / nsum
    else:
        distortion = float('Inf')
    return distortion

def evaluate_radius(radius, d, sample, rad_bw_ratio=3.0):
    global DIST
    global X

    t0 = time.time()
    D = DIST.copy()
    (n, dim) = X.shape
    D.data[D.data > radius] = 0.0
    D.eliminate_zeros()
    h = radius / rad_bw_ratio
    A = compute_affinity_matrix(D, 'gaussian', radius=h)
    (PS, nbrs) = compute_nbr_wts(A, sample)
    L = compute_laplacian_matrix(A, method='geometric', scaling_epps=h)
    L = L.tocsr()
    e_dist = distortion(X, L, sample, PS, nbrs, n, d)
    t1 = time.time()
    print("for radius: " + str(radius) + " distortion is: " + str(e_dist))
    print("for radius: " + str(radius) + " analysis took: " + str(t1-t0) + " seconds\n")
    return radius, e_dist

def radius_search(d, sample, rmin, rmax, ntry, search_space='linspace', rad_bw_ratio=3.0):
    print("performing radius search...\n")
    if search_space == "linspace":
        radii = np.linspace(rmin, rmax, ntry)
    elif search_space == "logspace":
        radii = np.logspace(np.log10(rmin),np.log10(rmax), ntry)
    else:
        raise ValueError("search_space can only be logspace or linspace")
    results = np.array([evaluate_radius(r, d, sample, rad_bw_ratio) for r in radii])
    return results

def multi_process_radius_search(d, sample, rmin, rmax, ntry, processes,
                                search_space='linspace', rad_bw_ratio=3.0):
    print("performing parallel radius search...\n")
    if search_space == "linspace":
        radii = np.linspace(rmin, rmax, ntry)
    elif search_space == "logspace":
        radii = np.logspace(np.log10(rmin),np.log10(rmax), ntry)
    else:
        raise ValueError("search_space can only be logspace or linspace")
    pool = ThreadPool(processes=processes)
    results = [pool.apply_async(evaluate_radius,
                                args=(r, d, sample, rad_bw_ratio))
               for r in radii]
    pool.close()
    pool.join()
    results = [p.get() for p in results]
    results.sort()
    return(np.array(results))

def run_estimate_radius(data, dists, sample, d, rmin, rmax, ntry, run_parallel,
                        search_space='linspace', rad_bw_ratio=3.0, max_cpus=None):
    """
    This function is used to estimate the bandwidth, h, of the Gaussian Kernel:
        exp(-||x_i - x_j||/h^2)


    The "radius" refers to the truncation constant which we take to be
    rad_bw_ratio * h and this is the parameter over which we will iterate.

    We use the method of: https://arxiv.org/abs/1406.0118

    Parameters
    ----------
    data : numpy array,
        Original data set for which we are estimating the bandwidth
    dists : scipy.csr_matrix,
        A CSR matrix containing pairwise distances from data up to rmax
    sample : np.array,
        subset of data points over which to evaluate the radii
    d : int,
        dimension over which to evaluate the radii (smaller usually better)
    rmin : float,
        smallest radius ( = rad_bw_ratio * bandwidth) to consider
    rmax : float,
        largest radius ( = rad_bw_ratio * bandwidth) to consider
    ntry : int,
        number of radii between rmax and rmin to try
    run_parallel : bool,
        whether to run the analysis in parallel over radii
    search_space : str,
        either 'linspace' or 'logspace', choose to search in log or linear space
    rad_bw_ratio : float,
        the ratio of radius and kernel bandwidth, default to be 3 (radius = 3*h)
    max_cpus : int,
        the maximum cpus to use when run_parallel is True, ignore if False

    Returns
    -------
    results : np.array (2d)
        first column is the set of radii and the second column is the distortion
        (smaller is better)

    """
    # declare global variables
    global DIST
    global X

    # process distance matrix
    dists = dists.tocsr()
    dists.data[dists.data > rmax] = 0.0
    dists.eliminate_zeros()

    # Assign global variables
    DIST = dists
    X = data

    n, dim = X.shape
    t0 = time.time()
    if run_parallel:
        ncpu = mp.cpu_count() # 32 for Newton
        processes = int(min(ntry, ncpu))
        if max_cpus is not None:
            processes = int(min(processes, max_cpus))
        print('using ' + str(processes) + ' processes to perform asynchronous parallel radius search')
        results = multi_process_radius_search(
            d, sample, rmin, rmax, ntry, processes, search_space, rad_bw_ratio)
    else:
        results = radius_search(
            d, sample, rmin, rmax, ntry, search_space, rad_bw_ratio)
    t1 = time.time()
    print('analysis took: ' + str(t1 - t0) + ' seconds to complete.')
    return(results)
