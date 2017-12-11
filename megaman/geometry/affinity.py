# LICENSE: Simplified BSD https://github.com/mmp2/megaman/blob/master/LICENSE

from __future__ import division
import numpy as np
from scipy.sparse import isspmatrix
from sklearn.utils.validation import check_array

from .utils import RegisterSubclasses


def compute_affinity_matrix(adjacency_matrix, method='auto', **kwargs):
    """Compute the affinity matrix with the given method"""
    if method == 'auto':
        method = 'gaussian'
    return Affinity.init(method, **kwargs).affinity_matrix(adjacency_matrix)


def affinity_methods():
    """Return the list of valid affinity methods"""
    return ['auto'] + list(Affinity.methods())


class Affinity(RegisterSubclasses):
    """Base class for computing affinity matrices"""
    def __init__(self, radius=None, symmetrize=True):
        if radius is None:
            raise ValueError("must specify radius for affinity matrix")
        self.radius = radius
        self.symmetrize = symmetrize

    def affinity_matrix(self, adjacency_matrix):
        raise NotImplementedError()


class GaussianAffinity(Affinity):
    name = "gaussian"

    @staticmethod
    def _symmetrize(A):
        # TODO: make this more efficient?
        # Also, need to maintain explicit zeros!
        return 0.5 * (A + A.T)

    def affinity_matrix(self, adjacency_matrix):
        A = check_array(adjacency_matrix, dtype=float, copy=True,
                        accept_sparse=['csr', 'csc', 'coo'])

        if isspmatrix(A):
            data = A.data
        else:
            data = A

        # in-place computation of
        # data = np.exp(-(data / radius) ** 2)
        data **= 2
        data /= -self.radius ** 2
        np.exp(data, out=data)

        if self.symmetrize:
            A = self._symmetrize(A)

        # for sparse, need a true zero on the diagonal
        # TODO: make this more efficient?
        if isspmatrix(A):
            A.setdiag(1)

        return A
