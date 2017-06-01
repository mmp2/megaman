import numpy as np
from numpy.random import RandomState
from scipy.spatial.distance import squareform, pdist
from megaman.utils.estimate_radius import run_estimate_radius
from scipy.sparse import csr_matrix
from numpy.testing import assert_array_almost_equal

def test_radius_serial_vs_parallel(seed=1234):
    rng = RandomState(seed)
    X = rng.randn(100, 10)
    dists = csr_matrix(squareform(pdist(X)))
    sample = range(100)
    d = 3
    rmin = 2
    rmax = 10.0
    ntry = 10
    run_parallel = True
    results_parallel = run_estimate_radius(X, dists, sample, d, rmin, rmax, ntry, run_parallel)
    print(results_parallel)
    results_serial = run_estimate_radius(X, dists, sample, d, rmin, rmax, ntry, False)
    print(results_serial)
    assert_array_almost_equal(results_parallel, results_serial)