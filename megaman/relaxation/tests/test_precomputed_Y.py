from megaman.relaxation.precomputed import *
from .utils import generate_toy_laplacian

import numpy as np
import numpy.testing
import scipy as sp
from functools import wraps


class BaseTestLkNeighbors(object):
    def generate_laplacian(self):
        raise NotImplementedError()
    def generate_subset(self,n):
        raise NotImplementedError()
    def setup_message(self):
        raise NotImplementedError()

    def setUp(self):
        self.generate_laplacian_and_subset()
        self.setup_message()
        self.Lk_tensor, self.nbk, self.si_map = compute_Lk(self.laplacian, self.n, self.subset)

    def generate_laplacian_and_subset(self):
        self.laplacian = self.generate_laplacian()
        self.n = self.laplacian.shape[0]
        self.subset = self.generate_subset(self.n)

    def get_rk(self,k):
        idx_lk_space = self.si_map[k]
        return (self.nbk[idx_lk_space] == k).nonzero()[0]

    def _test_array_close_deco(func):
        @wraps(func)
        def wrapper(self):
            test_func, true_func, err_msg = func(self)

            def properties_to_test():
                for k in self.subset:
                    Lk = self.Lk_tensor[self.si_map[k]]
                    yield test_func(Lk, k)
            def properties_is_true():
                for k in self.subset:
                    yield true_func(self.laplacian, k)

            testing_list = [item for item in properties_to_test()]
            correct_list = [item for item in properties_is_true()]
            try:
                np.testing.assert_allclose(testing_list, correct_list, err_msg=err_msg)
            except TypeError:
                for idx, totest in enumerate(testing_list):
                    np.testing.assert_allclose(
                        totest, correct_list[idx],
                        err_msg=err_msg + ", fails at k == {}".format(idx)
                    )
        return wrapper

    @_test_array_close_deco
    def test_nonzero_counts(self):
        def test_func(Lk,k):
            return Lk.nonzero()[0].shape[0]
        def true_func(laplacian,k):
            nonzero_count = laplacian[k,:].nonzero()[0].shape[0]
            return 3*nonzero_count-2 if nonzero_count != 0 else 0
        err_msg = 'The nonzero count should be the same for Lk and laplacian[k,:]'
        return test_func, true_func, err_msg

    @_test_array_close_deco
    def test_diagonal_Lk(self):
        def test_func(Lk,k):
            return Lk.diagonal()
        def true_func(laplacian,k):
            nnz_axis = laplacian[k,:].nonzero()
            rk = self.get_rk(k)
            true_Lk = np.squeeze(np.asarray(laplacian[k,:][nnz_axis]))
            true_Lk[rk] *= -1
            return true_Lk
        err_msg = 'The diagonal of Lk should be the same as nonzeros laplacian[k,:] term'
        return test_func, true_func, err_msg

    @_test_array_close_deco
    def test_row_Lk(self):
        def test_func(Lk,k):
            rk = self.get_rk(k)
            return np.squeeze(Lk[rk,:].toarray())
        def true_func(laplacian,k):
            nnz_axis = laplacian[k,:].nonzero()
            return np.squeeze(np.array(-laplacian[k,:][nnz_axis]))
        err_msg = 'The kth row of Lk should be the same as laplacian[k,:] term.'
        return test_func, true_func, err_msg

    @_test_array_close_deco
    def test_col_Lk(self):
        def test_func(Lk,k):
            rk = self.get_rk(k)
            return np.squeeze(Lk[:,rk].toarray())
        def true_func(laplacian,k):
            nnz_axis = laplacian[k,:].nonzero()
            return np.squeeze(np.array(-laplacian[k,:][nnz_axis]))
        err_msg = 'The kth col of Lk should be the same as laplacian[k,:] term.'
        return test_func, true_func, err_msg

    def test_Lk_symmetric(self):
        if_symmetric = [ np.allclose(Lk.toarray(),Lk.T.toarray()) for Lk in self.Lk_tensor]
        assert np.all(if_symmetric), 'One or more Lk is not symmetric'

    def _test_other_zeros(self):
        pass
        # You do not need this since if nonzero terms is 3n-2 and Lk row Lk col Lk diag are close.
        # Then this is automatically become true.

    def test_neighbors(self):
        true_neighbors = [self.laplacian[k,:].nonzero()[1] for k in self.subset]
        for idx, nb in enumerate(self.nbk):
            np.testing.assert_array_equal(
                nb, true_neighbors[idx],
                err_msg='calculated nbk should be the same as the non zero term in laplacian'
            )

    def test_si_map_index(self):
        keys, values = zip(*self.si_map.items())
        sorted_key = np.sort(keys)
        sorted_subset = np.sort(self.subset)
        np.testing.assert_array_equal(
            sorted_key, sorted_subset,
            err_msg='The index in subset should be identical to that in si_map'
        )

    def test_Lk_nbk_size_match(self):
        if_size_match = [Lk.shape[0] == self.nbk[k].shape[0] for k,Lk in enumerate(self.Lk_tensor)]
        assert np.all(if_size_match), 'One or more size of Lk and nbk does not match'

class TestLkWithWholeSubset(BaseTestLkNeighbors):
    def generate_laplacian(self):
        return generate_toy_laplacian()
    def generate_subset(self,n):
        return np.arange(n)
    def setup_message(self):
        print ('Testing Lk properties for whole subset')

class TestLkWithHalfIncrementalSubset(TestLkWithWholeSubset):
    def generate_subset(self,n):
        return np.arange(0,n,2)
    def setup_message(self):
        print ('Testing Lk properties for half incremental subsets')

class TestLkWithQuarterRandomSubset(TestLkWithWholeSubset):
    def generate_subset(self,n):
        size = int(round(n/4))
        return np.random.choice(np.arange(n), size, replace=False)
    def setup_message(self):
        print ('Testing Lk properties with a quarter random generated subset')
