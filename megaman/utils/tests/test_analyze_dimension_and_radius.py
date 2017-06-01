import numpy as np
from numpy.random import RandomState
from scipy.spatial.distance import squareform, pdist
import megaman.utils.analyze_dimension_and_radius as adar
from scipy.sparse import csr_matrix
from numpy.testing import assert_array_almost_equal

def test_dim_distance_passed_vs_computed(seed=1234):
    rng = RandomState(seed)
    X = rng.randn(100, 10)
    dists = csr_matrix(squareform(pdist(X)))
    rmin = 2
    rmax = 10.0
    nradii = 10
    radii = 10**(np.linspace(np.log10(rmin), np.log10(rmax), nradii))

    results_passed = adar.neighborhood_analysis(dists, radii)
    avg_neighbors = results_passed['avg_neighbors'].flatten()
    radii = results_passed['radii'].flatten()
    fit_range = range(len(radii))
    dim_passed = adar.find_dimension_plot(avg_neighbors, radii, fit_range)
    results_computed, dim_computed = adar.run_analyze_dimension_and_radius(X, rmin, rmax, nradii)
    assert(dim_passed == dim_computed)
    assert_array_almost_equal(results_passed['avg_neighbors'], results_computed['avg_neighbors'])