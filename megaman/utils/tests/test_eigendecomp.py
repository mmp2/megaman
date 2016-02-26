from megaman.utils.eigendecomp import eigen_decomposition, null_space
from numpy.testing import assert_array_almost_equal
import numpy as np

spd_solvers = ['auto', 'dense', 'arpack', 'amg', 'lobpcg']
non_spd_solvers = ['auto', 'dense', 'arpack']

def _check_with_col_sign_flipping(A, B, tol=0.0):
    """ Check array A and B are equal with possible sign flipping on
    each columns"""
    sign = True
    for column_idx in range(A.shape[1]):
        sign = sign and ((((A[:, column_idx] -
                            B[:, column_idx]) ** 2).mean() <= tol ** 2) or
                         (((A[:, column_idx] +
                            B[:, column_idx]) ** 2).mean() <= tol ** 2))
        if not sign:
            return False
    return True

def _test_all_solvers(solvers_to_test, S):
    for largest in [True, False]:
        Lambdas = {};
        for eigen_solver in solvers_to_test:
            lambdas, diffusion_map = eigen_decomposition(S, n_components = 3,
                                                        eigen_solver = eigen_solver,
                                                        largest = largest, drop_first = False)
            Lambdas[eigen_solver] = np.sort(lambdas)
        # pairwise comparison:
        for i in range(len(solvers_to_test)):
            for j in range(i+1, len(solvers_to_test)):
                print largest
                print(str(solvers_to_test[i]) + " + " + str(solvers_to_test[j]))
                assert_array_almost_equal(Lambdas[solvers_to_test[i]],
                                        Lambdas[solvers_to_test[j]])

def _test_all_null_solvers(solvers_to_test, S):
    for largest in [True, False]:
        Null_Space = {};
        for eigen_solver in solvers_to_test:
            nullspace, errors = null_space(S, k = 3, eigen_solver = eigen_solver)
            Null_Space[eigen_solver] = nullspace
        # pairwise comparison:
        for i in range(len(solvers_to_test)):
            for j in range(i+1, len(solvers_to_test)):
                print largest
                print(str(solvers_to_test[i]) + " + " + str(solvers_to_test[j]))
                _check_with_col_sign_flipping(Null_Space[solvers_to_test[i]],
                                        Null_Space[solvers_to_test[j]], 0.05)
def test_sym_pos_def_agreement():
    solvers_to_test = spd_solvers
    rng = np.random.RandomState(0)
    X = rng.uniform(size=(100, 40))
    S = np.dot(X.T, X)
    _test_all_solvers(solvers_to_test, S)

def test_null_space_sym_pos_def_agreement():
    solvers_to_test = spd_solvers
    solvers_to_test = spd_solvers
    rng = np.random.RandomState(0)
    X = rng.uniform(size=(100, 100))
    S = np.dot(X.T, X)
    _test_all_null_solvers(solvers_to_test, S)

def test_null_space_sym_agreement():
    solvers_to_test = non_spd_solvers
    solvers_to_test = non_spd_solvers
    rng = np.random.RandomState(0)
    X = rng.uniform(size=(10, 10))
    S = X + X.T
    _test_all_null_solvers(solvers_to_test, S)

def test_null_space_non_sym_agreement():
    solvers_to_test = non_spd_solvers
    rng = np.random.RandomState(0)
    S = rng.uniform(size=(10, 10))
    _test_all_null_solvers(solvers_to_test, S)
