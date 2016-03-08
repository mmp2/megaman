# LICENSE: Simplified BSD https://github.com/mmp2/megaman/blob/master/LICENSE

import warnings

from itertools import product

import numpy as np
from numpy.testing import assert_array_equal
import scipy.sparse as sp
from nose.tools import assert_raises, assert_true, assert_false, assert_equal

from megaman.utils.validation import check_array, check_symmetric, DataConversionWarning
from megaman.utils.testing import (assert_no_warnings, assert_warns,
                                ignore_warnings, assert_raise_message)

@ignore_warnings
def test_check_array():
    # accept_sparse == None
    # raise error on sparse inputs
    X = [[1, 2], [3, 4]]
    X_csr = sp.csr_matrix(X)
    assert_raises(TypeError, check_array, X_csr)
    # ensure_2d
    assert_warns(DeprecationWarning, check_array, [0, 1, 2])
    X_array = check_array([0, 1, 2])
    assert_equal(X_array.ndim, 2)
    X_array = check_array([0, 1, 2], ensure_2d=False)
    assert_equal(X_array.ndim, 1)
    # don't allow ndim > 3
    X_ndim = np.arange(8).reshape(2, 2, 2)
    assert_raises(ValueError, check_array, X_ndim)
    check_array(X_ndim, allow_nd=True)  # doesn't raise
    # force_all_finite
    X_inf = np.arange(4).reshape(2, 2).astype(np.float)
    X_inf[0, 0] = np.inf
    assert_raises(ValueError, check_array, X_inf)
    check_array(X_inf, force_all_finite=False)  # no raise
    # nan check
    X_nan = np.arange(4).reshape(2, 2).astype(np.float)
    X_nan[0, 0] = np.nan
    assert_raises(ValueError, check_array, X_nan)
    check_array(X_inf, force_all_finite=False)  # no raise

    # dtype and order enforcement.
    X_C = np.arange(4).reshape(2, 2).copy("C")
    X_F = X_C.copy("F")
    X_int = X_C.astype(np.int)
    X_float = X_C.astype(np.float)
    Xs = [X_C, X_F, X_int, X_float]
    dtypes = [np.int32, np.int, np.float, np.float32, None, np.bool, object]
    orders = ['C', 'F', None]
    copys = [True, False]

    for X, dtype, order, copy in product(Xs, dtypes, orders, copys):
        X_checked = check_array(X, dtype=dtype, order=order, copy=copy)
        if dtype is not None:
            assert_equal(X_checked.dtype, dtype)
        else:
            assert_equal(X_checked.dtype, X.dtype)
        if order == 'C':
            assert_true(X_checked.flags['C_CONTIGUOUS'])
            assert_false(X_checked.flags['F_CONTIGUOUS'])
        elif order == 'F':
            assert_true(X_checked.flags['F_CONTIGUOUS'])
            assert_false(X_checked.flags['C_CONTIGUOUS'])
        if copy:
            assert_false(X is X_checked)
        else:
            # doesn't copy if it was already good
            if (X.dtype == X_checked.dtype and
                    X_checked.flags['C_CONTIGUOUS'] == X.flags['C_CONTIGUOUS']
                    and X_checked.flags['F_CONTIGUOUS'] == X.flags['F_CONTIGUOUS']):
                assert_true(X is X_checked)

    # allowed sparse != None
    X_csc = sp.csc_matrix(X_C)
    X_coo = X_csc.tocoo()
    X_dok = X_csc.todok()
    X_int = X_csc.astype(np.int)
    X_float = X_csc.astype(np.float)

    Xs = [X_csc, X_coo, X_dok, X_int, X_float]
    accept_sparses = [['csr', 'coo'], ['coo', 'dok']]
    for X, dtype, accept_sparse, copy in product(Xs, dtypes, accept_sparses,
                                                 copys):
        with warnings.catch_warnings(record=True) as w:
            X_checked = check_array(X, dtype=dtype,
                                    accept_sparse=accept_sparse, copy=copy)
        if (dtype is object or sp.isspmatrix_dok(X)) and len(w):
            message = str(w[0].message)
            messages = ["object dtype is not supported by sparse matrices",
                        "Can't check dok sparse matrix for nan or inf."]
            assert_true(message in messages)
        else:
            assert_equal(len(w), 0)
        if dtype is not None:
            assert_equal(X_checked.dtype, dtype)
        else:
            assert_equal(X_checked.dtype, X.dtype)
        if X.format in accept_sparse:
            # no change if allowed
            assert_equal(X.format, X_checked.format)
        else:
            # got converted
            assert_equal(X_checked.format, accept_sparse[0])
        if copy:
            assert_false(X is X_checked)
        else:
            # doesn't copy if it was already good
            if (X.dtype == X_checked.dtype and X.format == X_checked.format):
                assert_true(X is X_checked)

    # other input formats
    # convert lists to arrays
    X_dense = check_array([[1, 2], [3, 4]])
    assert_true(isinstance(X_dense, np.ndarray))
    # raise on too deep lists
    assert_raises(ValueError, check_array, X_ndim.tolist())
    check_array(X_ndim.tolist(), allow_nd=True)  # doesn't raise

def test_check_array_dtype_stability():
    # test that lists with ints don't get converted to floats
    X = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
    assert_equal(check_array(X).dtype.kind, "i")
    assert_equal(check_array(X, ensure_2d=False).dtype.kind, "i")


def test_check_array_dtype_warning():
    X_int_list = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
    X_float64 = np.asarray(X_int_list, dtype=np.float64)
    X_float32 = np.asarray(X_int_list, dtype=np.float32)
    X_int64 = np.asarray(X_int_list, dtype=np.int64)
    X_csr_float64 = sp.csr_matrix(X_float64)
    X_csr_float32 = sp.csr_matrix(X_float32)
    X_csc_float32 = sp.csc_matrix(X_float32)
    X_csc_int32 = sp.csc_matrix(X_int64, dtype=np.int32)
    y = [0, 0, 1]
    integer_data = [X_int64, X_csc_int32]
    float64_data = [X_float64, X_csr_float64]
    float32_data = [X_float32, X_csr_float32, X_csc_float32]
    for X in integer_data:
        X_checked = assert_no_warnings(check_array, X, dtype=np.float64,
                                       accept_sparse=True)
        assert_equal(X_checked.dtype, np.float64)

        X_checked = assert_warns(DataConversionWarning, check_array, X,
                                 dtype=np.float64,
                                 accept_sparse=True, warn_on_dtype=True)
        assert_equal(X_checked.dtype, np.float64)

    for X in float64_data:
        X_checked = assert_no_warnings(check_array, X, dtype=np.float64,
                                       accept_sparse=True, warn_on_dtype=True)
        assert_equal(X_checked.dtype, np.float64)
        X_checked = assert_no_warnings(check_array, X, dtype=np.float64,
                                       accept_sparse=True, warn_on_dtype=False)
        assert_equal(X_checked.dtype, np.float64)

    for X in float32_data:
        X_checked = assert_no_warnings(check_array, X,
                                       dtype=[np.float64, np.float32],
                                       accept_sparse=True)
        assert_equal(X_checked.dtype, np.float32)
        assert_true(X_checked is X)

        X_checked = assert_no_warnings(check_array, X,
                                       dtype=[np.float64, np.float32],
                                       accept_sparse=['csr', 'dok'],
                                       copy=True)
        assert_equal(X_checked.dtype, np.float32)
        assert_false(X_checked is X)

    X_checked = assert_no_warnings(check_array, X_csc_float32,
                                   dtype=[np.float64, np.float32],
                                   accept_sparse=['csr', 'dok'],
                                   copy=False)
    assert_equal(X_checked.dtype, np.float32)
    assert_false(X_checked is X_csc_float32)
    assert_equal(X_checked.format, 'csr')


def test_check_array_min_samples_and_features_messages():
    # empty list is considered 2D by default:
    msg = "0 feature(s) (shape=(1, 0)) while a minimum of 1 is required."
    assert_raise_message(ValueError, msg, check_array, [[]])

    # If considered a 1D collection when ensure_2d=False, then the minimum
    # number of samples will break:
    msg = "0 sample(s) (shape=(0,)) while a minimum of 1 is required."
    assert_raise_message(ValueError, msg, check_array, [], ensure_2d=False)

    # Invalid edge case when checking the default minimum sample of a scalar
    msg = "Singleton array array(42) cannot be considered a valid collection."
    assert_raise_message(TypeError, msg, check_array, 42, ensure_2d=False)

    # But this works if the input data is forced to look like a 2 array with
    # one sample and one feature:
    X_checked = assert_warns(DeprecationWarning, check_array, [42],
                             ensure_2d=True)
    assert_array_equal(np.array([[42]]), X_checked)

def test_check_symmetric():
    arr_sym = np.array([[0, 1], [1, 2]])
    arr_bad = np.ones(2)
    arr_asym = np.array([[0, 2], [0, 2]])

    test_arrays = {'dense': arr_asym,
                   'dok': sp.dok_matrix(arr_asym),
                   'csr': sp.csr_matrix(arr_asym),
                   'csc': sp.csc_matrix(arr_asym),
                   'coo': sp.coo_matrix(arr_asym),
                   'lil': sp.lil_matrix(arr_asym),
                   'bsr': sp.bsr_matrix(arr_asym)}

    # check error for bad inputs
    assert_raises(ValueError, check_symmetric, arr_bad)

    # check that asymmetric arrays are properly symmetrized
    for arr_format, arr in test_arrays.items():
        # Check for warnings and errors
        assert_warns(UserWarning, check_symmetric, arr)
        assert_raises(ValueError, check_symmetric, arr, raise_exception=True)

        output = check_symmetric(arr, raise_warning=False)
        if sp.issparse(output):
            assert_equal(output.format, arr_format)
            assert_array_equal(output.toarray(), arr_sym)
        else:
            assert_array_equal(output, arr_sym)
