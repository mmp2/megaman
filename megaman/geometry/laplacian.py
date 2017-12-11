# LICENSE: Simplified BSD https://github.com/mmp2/megaman/blob/master/LICENSE
from __future__ import division
import numpy as np
from scipy.sparse import isspmatrix
from sklearn.utils.validation import check_array

from .utils import RegisterSubclasses


def compute_laplacian_matrix(affinity_matrix, method='auto', **kwargs):
    """Compute the laplacian matrix with the given method"""
    if method == 'auto':
        method = 'geometric'
    return Laplacian.init(method, **kwargs).laplacian_matrix(affinity_matrix)


def laplacian_methods():
    """Return the list of valid laplacian methods"""
    return ['auto'] + list(Laplacian.methods())


class Laplacian(RegisterSubclasses):
    """Base class for computing laplacian matrices

    Notes
    -----
    The methods here all return the negative of the standard
    Laplacian definition.
    """
    symmetric = False

    def __init__(self, symmetrize_input=True,
                 scaling_epps=None, full_output=False):
        self.symmetrize_input = symmetrize_input
        self.scaling_epps = scaling_epps
        self.full_output = full_output

    @staticmethod
    def _symmetrize(A):
        # TODO: make this more efficient?
        return 0.5 * (A + A.T)

    @classmethod
    def symmetric_methods(cls):
        for method in cls.methods():
            if cls.get_method(method).symmetric:
                yield method

    @classmethod
    def asymmetric_methods(cls):
        for method in cls.methods():
            if not cls.get_method(method).symmetric:
                yield method

    def laplacian_matrix(self, affinity_matrix):
        affinity_matrix = check_array(affinity_matrix, copy=False, dtype=float,
                                      accept_sparse=['csr', 'csc', 'coo'])
        if self.symmetrize_input:
            affinity_matrix = self._symmetrize(affinity_matrix)

        if isspmatrix(affinity_matrix):
            affinity_matrix = affinity_matrix.tocoo()
        else:
            affinity_matrix = affinity_matrix.copy()

        lap, lapsym, w = self._compute_laplacian(affinity_matrix)

        if self.scaling_epps is not None and self.scaling_epps > 0.:
            if isspmatrix(lap):
                lap.data *= 4 / (self.scaling_epps ** 2)
            else:
                lap *= 4 / (self.scaling_epps ** 2)

        if self.full_output:
            return lap, lapsym, w
        else:
            return lap

    def _compute_laplacian(self, lap):
        raise NotImplementedError()


class UnNormalizedLaplacian(Laplacian):
    name = 'unnormalized'
    symmetric = True

    def _compute_laplacian(self, lap):
        w = _degree(lap)
        _subtract_from_diagonal(lap, w)
        return lap, lap, w


class GeometricLaplacian(Laplacian):
    name = 'geometric'
    symmetric = False

    def _compute_laplacian(self, lap):
        _normalize_laplacian(lap, symmetric=True)
        lapsym = lap.copy()

        w, nonzero = _normalize_laplacian(lap, symmetric=False)
        _subtract_from_diagonal(lap, nonzero)

        return lap, lapsym, w


class RandomWalkLaplacian(Laplacian):
    name = 'randomwalk'
    symmetric = False

    def _compute_laplacian(self, lap):
        lapsym = lap.copy()
        w, nonzero = _normalize_laplacian(lap, symmetric=False)
        _subtract_from_diagonal(lap, nonzero)
        return lap, lapsym, w


class SymmetricNormalizedLaplacian(Laplacian):
    name = 'symmetricnormalized'
    symmetric = True

    def _compute_laplacian(self, lap):
        w, nonzero = _normalize_laplacian(lap, symmetric=True, degree_exp=0.5)
        _subtract_from_diagonal(lap, nonzero)
        return lap, lap, w


class RenormalizedLaplacian(Laplacian):
    name = 'renormalized'
    symmetric = False

    def __init__(self, symmetrize_input=True,
                 scaling_epps=None,
                 full_output=False,
                 renormalization_exponent=1):
        self.symmetrize_input = symmetrize_input
        self.scaling_epps = scaling_epps
        self.full_output = full_output
        self.renormalization_exponent = renormalization_exponent

    def _compute_laplacian(self, lap):
        _normalize_laplacian(lap, symmetric=True,
                             degree_exp=self.renormalization_exponent)
        lapsym = lap.copy()
        w, nonzero = _normalize_laplacian(lap, symmetric=False)
        _subtract_from_diagonal(lap, nonzero)

        return lap, lapsym, w


# Utility routines: these operate in-place and assume either coo matrix or
# dense array

def _degree(lap):
    return np.asarray(lap.sum(1)).squeeze()


def _divide_along_rows(lap, vals):
    if isspmatrix(lap):
        lap.data /= vals[lap.row]
    else:
        lap /= vals[:, np.newaxis]


def _divide_along_cols(lap, vals):
    if isspmatrix(lap):
        lap.data /= vals[lap.col]
    else:
        lap /= vals


def _normalize_laplacian(lap, symmetric=False, degree_exp=None):
    w = _degree(lap)
    w_nonzero = (w != 0)
    w[~w_nonzero] = 1

    if degree_exp is not None:
        w **= degree_exp

    if symmetric:
        _divide_along_rows(lap, w)
        _divide_along_cols(lap, w)
    else:
        _divide_along_rows(lap, w)

    return w, w_nonzero


def _subtract_from_diagonal(lap, vals):
    if isspmatrix(lap):
        lap.data[lap.row == lap.col] -= vals
    else:
        lap.flat[::lap.shape[0] + 1] -= vals
