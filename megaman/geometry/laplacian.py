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
    def _symmetrize(self, A):
        # TODO: make this more efficient?
        return 0.5 * (A + A.T)

    def laplacian_matrix(self, affinity_matrix):
        affinity_matrix = check_array(affinity_matrix, copy=True, dtype=float,
                                      accept_sparse=['csr', 'csc', 'coo'])

        if self.symmetrize:
            affinity_matrix = self._symmetrize(affinity_matrix)

        if isspmatrix(affinity_matrix):
            return self.laplacian_matrix_sparse(affinity_matrix)
        else:
            return self.laplacian_matrix_dense(affinity_matrix)

    def laplacian_matrix_sparse(self, affinity_matrix):
        raise NotImplementedError()

    def laplacian_matrix_dense(self, affinity_matrix):
        raise NotImplementedError()


class UnNormalizedLaplacian(Laplacian):
    name = 'unnormalized'

    def laplacian_matrix_sparse(self, affinity_matrix):
        raise NotImplementedError()

    def laplacian_matrix_dense(self, affinity_matrix):
        raise NotImplementedError()


class GeometricLaplacian(Laplacian):
    name = 'geometric'

    def laplacian_matrix_sparse(self, affinity_matrix):
        raise NotImplementedError()

    def laplacian_matrix_dense(self, affinity_matrix):
        raise NotImplementedError()


class RandomWalkLaplacian(Laplacian):
    name = 'randomwalk'

    def laplacian_matrix_sparse(self, affinity_matrix):
        raise NotImplementedError()

    def laplacian_matrix_dense(self, affinity_matrix):
        raise NotImplementedError()


class SymmetricNormalizedLaplacian(Laplacian):
    name = 'symmetricnormalized'

    def laplacian_matrix_sparse(self, affinity_matrix):
        raise NotImplementedError()

    def laplacian_matrix_dense(self, affinity_matrix):
        raise NotImplementedError()


class RenormalizedLaplacian(Laplacian):
    name = 'renormalized'

    def laplacian_matrix_sparse(self, affinity_matrix):
        raise NotImplementedError()

    def laplacian_matrix_dense(self, affinity_matrix):
        raise NotImplementedError()

    """
    lap = csgraph.copy()
    if symmetrize:
        if lap.format is not 'csr':
            lap.tocsr()
        lap = (lap + lap.T)/2.
    if lap.format is not 'coo':
        lap = lap.tocoo()
    diag_mask = (lap.row == lap.col)  # True/False
    degrees = np.asarray(lap.sum(axis=1)).squeeze()

    if normed == 'symmetricnormalized':
        w = np.sqrt(degrees)
        w_zeros = (w == 0)
        w[w_zeros] = 1
        lap.data /= w[lap.row]
        lap.data /= w[lap.col]
        lap.data[diag_mask] -= 1.
        if return_lapsym:
            lapsym = lap.copy()

    elif normed == 'geometric':
        w = degrees.copy()     # normzlize one symmetrically by d
        w_zeros = (w == 0)
        w[w_zeros] = 1
        lap.data /= w[lap.row]
        lap.data /= w[lap.col]
        w = np.asarray(lap.sum(axis=1)).squeeze() #normalize again asymmetricall
        if return_lapsym:
            lapsym = lap.copy()
        lap.data /= w[lap.row]
        lap.data[diag_mask] -= 1.

    elif normed == 'renormalized':
        w = degrees**renormalization_exponent;
        # same as 'geometric' from here on
        w_zeros = (w == 0)
        w[w_zeros] = 1
        lap.data /= w[lap.row]
        lap.data /= w[lap.col]
        w = np.asarray(lap.sum(axis=1)).squeeze() #normalize again asymmetricall
        if return_lapsym:
            lapsym = lap.copy()
        lap.data /= w[lap.row]
        lap.data[diag_mask] -= 1.

    elif normed == 'unnormalized':
        lap.data[diag_mask] -= degrees
        if return_lapsym:
            lapsym = lap.copy()

    elif normed == 'randomwalk':
        w = degrees.copy()
        if return_lapsym:
            lapsym = lap.copy()
        lap.data /= w[lap.row]
        lap.data[diag_mask] -= 1.

    if scaling_epps > 0.:
        lap.data *= 4/(scaling_epps**2)

    if return_diag:
        if return_lapsym:
            return lap, lap.data[diag_mask], lapsym, w
        else:
            return lap, lap.data[diag_mask]

    elif return_lapsym:
        return lap, lapsym, w
    else:
        return lap


def _laplacian_dense(csgraph, normed='geometric', symmetrize=True,
                     scaling_epps=0., renormalization_exponent=1,
                     return_diag=False, return_lapsym=False):
    n_nodes = csgraph.shape[0]
    if symmetrize:
        lap = (csgraph + csgraph.T)/2.
    else:
        lap = csgraph.copy()
    degrees = np.asarray(lap.sum(axis=1)).squeeze()
    di = np.diag_indices( lap.shape[0] )  # diagonal indices

    if normed == 'symmetricnormalized':
        w = np.sqrt(degrees)
        w_zeros = (w == 0)
        w[w_zeros] = 1
        lap /= w
        lap /= w[:, np.newaxis]
        di = np.diag_indices( lap.shape[0] )
        lap[di] -= (1 - w_zeros).astype(lap.dtype)
        if return_lapsym:
            lapsym = lap.copy()
    elif normed == 'geometric':
        w = degrees.copy()     # normalize once symmetrically by d
        w_zeros = (w == 0)
        w[w_zeros] = 1
        lap /= w
        lap /= w[:, np.newaxis]
        w = np.asarray(lap.sum(axis=1)).squeeze() #normalize again asymmetricall
        if return_lapsym:
            lapsym = lap.copy()
        lap /= w[:, np.newaxis]
        lap[di] -= (1 - w_zeros).astype(lap.dtype)
    elif normed == 'renormalized':
        w = degrees**renormalization_exponent;
        # same as 'geometric' from here on
        w_zeros = (w == 0)
        w[w_zeros] = 1
        lap /= w
        lap /= w[:, np.newaxis]
        w = np.asarray(lap.sum(axis=1)).squeeze() #normalize again asymmetricall
        if return_lapsym:
            lapsym = lap.copy()
        lap /= w[:, np.newaxis]
        lap[di] -= (1 - w_zeros).astype(lap.dtype)
    elif normed == 'unnormalized':
        dum = lap[di]-degrees[np.newaxis,:]
        lap[di] = dum[0,:]
        if return_lapsym:
            lapsym = lap.copy()
    elif normed == 'randomwalk':
        w = degrees.copy()
        if return_lapsym:
            lapsym = lap.copy()
        lap /= w[:,np.newaxis]
        lap -= np.eye(lap.shape[0])

    if scaling_epps > 0.:
        lap *= 4/(scaling_epps**2)

    if return_diag:
        diag = np.array( lap[di] )
        if return_lapsym:
            return lap, diag, lapsym, w
        else:
            return lap, diag
    elif return_lapsym:
        return lap, lapsym, w
    else:
        return lap
    """
