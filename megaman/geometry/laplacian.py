import numpy as np
from scipy.sparse import isspmatrix
from sklearn.utils.validation import check_array

from .utils import RegisterSubclasses


def compute_laplacian_matrix(affinity_matrix, method='auto', **kwargs):
    """Compute the laplacian matrix with the given method"""
    if method == 'auto':
        method = 'squared_exponential'
    return Affinity.init(method, **kwargs).affinity_matrix(X)


class Laplacian(RegisterSubclasses):
    """Base class for computing affinity matrices"""
    def __init__(self, symmetrize=True):
        self.symmetrize = symmetrize

    @staticmethod
    def _symmetrize(A):
        # TODO: make this more efficient?
        return 0.5 * (A + A.T)

    @staticmethod
    def _degrees(lap, remove_zeros=False):
        degrees = np.asarray(lap.sum(1)).squeeze()
        if remove_zeros:
            degrees[degrees == 0] = 1
        return degrees

    @staticmethod
    def _divide_along_rows(lap, vals):
        if isspmatrix(lap):
            lap.data /= w[lap.row]
        else:
            lap /= w[:, np.newaxis]

    @staticmethod
    def _divide_along_cols(laplacian, vals):
        if isspmatrix(lap):
            lap.data /= w[lap.col]
        else:
            lap /= w

    @staticmethod
    def _subtract_from_diagonal(laplacian, vals):
        if isspmatrix(laplacian):
            lap.data[lap.row == lap.col] -= vals
        else:
            lap.flat[::lap.shape[0] + 1] -= vals

    def laplacian_matrix(self, affinity_matrix, return_symmetrized=False):
        affinity_matrix = check_array(affinity_matrix, copy=False, dtype=float,
                                      accept_sparse=['csr', 'csc', 'coo'])
        if self.symmetrize:
            affinity_matrix = self._symmetrize(affinity_matrix)

        if isspmatrix(affinity_matrix):
            affinity_matrix = affinity_matrix.tocoo()
        else:
            affinity_matrix = affinity_matrix.copy()

        lap, lapsym, w = self._compute_laplacian(affinity_matrix)

        if return_symmetrized:
            return lap, lapsym, w
        else:
            return lap

    def _compute_laplacian(self, lap):
        raise NotImplementedError()


class UnNormalizedLaplacian(Laplacian):
    name = 'unnormalized'

    def _compute_laplacian(self, lap):
        w = self._degrees(lap)
        self._subtract_from_diagonal(lap, w)
        return lap, lap.copy(), w


class GeometricLaplacian(Laplacian):
    name = 'geometric'

    def _compute_laplacian(self, lap):
        # normalize symmetrically by degree
        w = self._degrees(lap, remove_zeros=True)
        self._divide_along_cols(lap, w)
        self._divide_along_rows(lap, w)
        lapsym = lap.copy()

        #normalize again asymmetrically
        w = self._degrees(lap, remove_zeros=True)
        self._divide_along_rows(lap, w)
        self._subtract_from_diagonal(lap, 1)  # XXX check for w=0

        return lap, lapsym, w


class RandomWalkLaplacian(Laplacian):
    name = 'randomwalk'

    def _compute_laplacian(self, lap):
        lapsym = lap.copy()
        w = self._degrees(lap)
        self._divide_along_rows(lap, w)
        self._subtract_from_diagonal(lap, 1)
        return lap, lapsym, w


class SymmetricNormalizedLaplacian(Laplacian):
    name = 'symmetricnormalized'

    def _compute_laplacian(self, lap):
        w = np.sqrt(self._degrees(lap, remove_zeros=True))
        self._divide_along_cols(lap, w)
        self._divide_along_rows(lap, w)
        self._subtract_from_diagonal(lap, 1)
        return lap, lap.copy(), w


class RenormalizedLaplacian(Laplacian):
    name = 'renormalized'

    def __init__(self, symmetrize=True, renormalization_exponent=1):
        self.symmetrize = symmetrize
        self.renormalization_exponent = renormalization_exponent

    def _compute_laplacian(self, lap):
        w = self._degrees(lap, remove_zeros=True)
        w **= self.renormalization_exponent

        # same as GeometricLaplacian from here on
        self._divide_along_cols(lap, w)
        self._divide_along_rows(lap, w)

        lapsym = lap.copy()

        #normalize again asymmetrically
        w = self._degrees(lap, remove_zeros=True)
        self._divide_along_rows(lap, w)
        self._subtract_from_diagonal(lap, 1)  # XXX check for w=0

        return lap, lapsym, w
