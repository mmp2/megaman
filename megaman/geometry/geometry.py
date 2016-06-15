"""
Scalable Manifold learning utilities and algorithms.

Graphs are represented with their weighted adjacency matrices, preferably using
sparse matrices.

A note on symmetrization and internal sparse representations
------------------------------------------------------------

For performance, this code uses the FLANN library to compute
approximate neighborhoods efficiently. This means that (1) the
adjacency matrix produced is NOT GUARANTEED to be symmetric and
(2) compute_adjacency_matrix returns a sparse matrix called
adjacency_matrix. adjacency_matrix has 0.0 on the diagonal,
as it should. Implicitly, the missing entries are infinity not
0 for this matrix. But (1) and (2) mean that if one tries to
symmetrize adjacency_matrix, the scipy.sparse code eliminates
the 0.0 entries from adjacency_matrix hence in the affinity
matrix we explicitly set the diagonal to 1.0 for sparse matrices.

We adopted the following convention:
   * adjacency_matrix will NOT BE GUARANTEED symmetric
   * affinity_matrix will perform a symmetrization by default
   * laplacian performs symmetrization 
     only if symmetrize_input=True (the default setting), and DOES NOT check symmetry
   * these conventions are the same for dense matrices, for consistency
"""

# Authors: Marina Meila <mmp@stat.washington.edu>
#         James McQueen <jmcq@u.washington.edu>
# LICENSE: Simplified BSD https://github.com/mmp2/megaman/blob/master/LICENSE
from __future__ import division ## removes integer division
import numpy as np
from scipy import sparse
from scipy.special import gammaln
from .adjacency import compute_adjacency_matrix
from .affinity import compute_affinity_matrix
from .laplacian import compute_laplacian_matrix
from ..utils.validation import check_array

sparse_formats = ['csr', 'coo', 'lil', 'bsr', 'dok', 'dia']
distance_error_msg = ("No data matrix exists. "
                      "Adjacency matrix cannot be computed.")


class Geometry(object):
    """
    The Geometry class stores the data, distance, affinity and laplacian
    matrices used by the various embedding methods and is the primary
    object passed to embedding functions.

    The Geometry class contains functions to compute the aforementioned
    matrices and allows for re-computation whenever necessary.

    Parameters
    ----------
    adjacency_method : string {'auto', 'brute', 'pyflann', 'cyflann'}
        method for computing pairwise radius neighbors graph.
    adjacency_kwds : dict
        dictionary containing keyword arguments for adjacency matrix.
        see distance.py docmuentation for arguments for each method.
        If new kwargs are passed to compute_adjacency_matrix then this
        dictionary will be updated.
    affinity_method : string {'auto', 'gaussian'}
        method of computing affinity matrix
    affinity_kwds : dict
        dictionary containing keyword arguments for affinity matrix.
        see affinity.py docmuentation for arguments for each method.
        If new kwargs are passed to compute_affinity_matrix then this
        dictionary will be updated.
    laplacian_method : string,
        type of laplacian to be computed. Possibilities are
        {'symmetricnormalized', 'geometric', 'renormalized',
        'unnormalized', 'randomwalk'} see laplacian.py for more information.
    laplacian_kwds : dict
        dictionary containing keyword arguments for Laplacian matrix.
        see laplacian.py docmuentation for arguments for each method.
        If new kwargs are passed to compute_laplacian_matrix then this
        dictionary will be updated.
    **kwargs :
        additional arguments will be parsed and used to override values in
        the above dictionaries. For example:
        - `affinity_radius` will override `affinity_kwds['radius']`
        - `adjacency_n_neighbors` will override `adjacency_kwds['n_neighbors']`
        etc.
    """
    def __init__(self, adjacency_method='auto', adjacency_kwds=None,
                 affinity_method='auto', affinity_kwds=None,
                 laplacian_method='auto',laplacian_kwds=None, **kwargs):
        self.adjacency_method = adjacency_method
        self.adjacency_kwds = dict(**(adjacency_kwds or {}))
        self.affinity_method = affinity_method
        self.affinity_kwds = dict(**(affinity_kwds or {}))
        self.laplacian_method = laplacian_method
        self.laplacian_kwds = dict(**(laplacian_kwds or {}))

        # map extra keywords: e.g. affinity_radius -> affinity_kwds['radius']
        dicts = dict(adjaceny=self.adjacency_kwds,
                     affinity=self.affinity_kwds,
                     laplacian=self.laplacian_kwds)
        for key, val in kwargs.items():
            keysplit = key.split('_')
            if keysplit[0] not in dicts:
                raise ValueError('key `{0}` not valid'.format(key))
            dicts[keysplit[0]]['_'.join(keysplit[1:])] = val

        self.X = None
        self.adjacency_matrix = None
        self.affinity_matrix = None
        self.laplacian_matrix = None
        self.laplacian_symmetric = None
        self.laplacian_weights = None

    def set_radius(self, radius, override=True, X=None, n_components=2):
        """Set the radius for the adjacency and affinity computation

        By default, this will override keyword arguments provided on
        initialization.

        Parameters
        ----------
        radius : float
            radius to set for adjacency and affinity.
        override : bool (default: True)
            if False, then only set radius if not already defined in
            `adjacency_args` and `affinity_args`.
        X : ndarray or sparse (optional)
            if provided, estimate a suitable radius from this data.
        n_components : int (default=2)
            the number of components to use when estimating the radius
        """
        if radius < 0:
            raise ValueError("radius must be non-negative")

        if override or ('radius' not in self.adjacency_kwds and
                        'n_neighbors' not in self.adjacency_kwds):
            self.adjacency_kwds['radius'] = radius

        if override or ('radius' not in self.affinity_kwds):
            self.affinity_kwds['radius'] = radius

    def set_matrix(self, X, input_type):
        """Set the data matrix given the (string) input_type"""
        if input_type == 'data':
            self.set_data_matrix(X)
        elif input_type == 'adjacency':
            self.set_adjacency_matrix(X)
        elif input_type == 'affinity':
            self.set_affinity_matrix(X)
        else:
            raise ValueError("Unrecognized input_type: {0}".format(input_type))


    def compute_adjacency_matrix(self, copy=False, **kwargs):
        """
        This function will compute the adjacency matrix.
        In order to acquire the existing adjacency matrix use
        self.adjacency_matrix as comptute_adjacency_matrix() will re-compute
        the adjacency matrix.

        Parameters
        ----------
        copy : boolean, whether to return a copied version of the adjacency matrix
        **kwargs : see distance.py docmuentation for arguments for each method.

        Returns
        -------
        self.adjacency_matrix : sparse matrix (N_obs, N_obs)
            Non explicit 0.0 values should be considered not connected.
        """
        if self.X is None:
            raise ValueError(distance_error_msg)

        kwds = self.adjacency_kwds.copy()
        kwds.update(kwargs)
        self.adjacency_matrix = compute_adjacency_matrix(self.X,
                                                         self.adjacency_method,
                                                         **kwds)
        if copy:
            return self.adjacency_matrix.copy()
        else:
            return self.adjacency_matrix

    def compute_affinity_matrix(self, copy=False, **kwargs):
        """
        This function will compute the affinity matrix. In order to
        acquire the existing affinity matrix use self.affinity_matrix as
        comptute_affinity_matrix() will re-compute the affinity matrix.

        Parameters
        ----------
        copy : boolean
            whether to return a copied version of the affinity matrix
        **kwargs :
            see affinity.py docmuentation for arguments for each method.

        Returns
        -------
        self.affinity_matrix : sparse matrix (N_obs, N_obs)
            contains the pairwise affinity values using the Guassian kernel
            and bandwidth equal to the affinity_radius
        """
        if self.adjacency_matrix is None:
            self.compute_adjacency_matrix()

        kwds = self.affinity_kwds.copy()
        kwds.update(kwargs)
        self.affinity_matrix = compute_affinity_matrix(self.adjacency_matrix,
                                                       self.affinity_method,
                                                       **kwds)
        if copy:
            return self.affinity_matrix.copy()
        else:
            return self.affinity_matrix

    def compute_laplacian_matrix(self, copy=True, return_lapsym=False, **kwargs):
        """
        Note: this function will compute the laplacian matrix. In order to acquire
            the existing laplacian matrix use self.laplacian_matrix as
            comptute_laplacian_matrix() will re-compute the laplacian matrix.

        Parameters
        ----------
        copy : boolean, whether to return copied version of the self.laplacian_matrix
        return_lapsym : boolean, if True returns additionally the symmetrized version of
            the requested laplacian and the re-normalization weights.
        **kwargs : see laplacian.py docmuentation for arguments for each method.

        Returns
        -------
        self.laplacian_matrix : sparse matrix (N_obs, N_obs).
            The requested laplacian.
        self.laplacian_symmetric : sparse matrix (N_obs, N_obs)
            The symmetric laplacian.
        self.laplacian_weights : ndarray (N_obs,)
            The renormalization weights used to make
            laplacian_matrix from laplacian_symmetric
        """
        if self.affinity_matrix is None:
            self.compute_affinity_matrix()

        kwds = self.laplacian_kwds.copy()
        kwds.update(kwargs)
        kwds['full_output'] = return_lapsym
        result = compute_laplacian_matrix(self.affinity_matrix,
                                          self.laplacian_method,
                                          **kwds)
        if return_lapsym:
            (self.laplacian_matrix,
             self.laplacian_symmetric,
             self.laplacian_weights) = result
        else:
            self.laplacian_matrix = result

        if copy:
            return self.laplacian_matrix.copy()
        else:
            return self.laplacian_matrix

    def set_data_matrix(self, X):
        """
        Parameters
        ----------
        X : ndarray (N_obs, N_features)
            The original data set to input.
        """
        X = check_array(X, accept_sparse=sparse_formats)
        self.X = X

    def set_adjacency_matrix(self, adjacency_mat):
        """
        Parameters
        ----------
        adjacency_mat : sparse matrix (N_obs, N_obs)
            The adjacency matrix to input.
        """
        adjacency_mat = check_array(adjacency_mat, accept_sparse=sparse_formats)
        if adjacency_mat.shape[0] != adjacency_mat.shape[1]:
            raise ValueError("adjacency matrix is not square")
        self.adjacency_matrix = adjacency_mat

    def set_affinity_matrix(self, affinity_mat):
        """
        Parameters
        ----------
        affinity_mat : sparse matrix (N_obs, N_obs).
            The adjacency matrix to input.
        """
        affinity_mat = check_array(affinity_mat, accept_sparse=sparse_formats)
        if affinity_mat.shape[0] != affinity_mat.shape[1]:
            raise ValueError("affinity matrix is not square")
        self.affinity_matrix = affinity_mat

    def set_laplacian_matrix(self, laplacian_mat):
        """
        Parameters
        ----------
        laplacian_mat : sparse matrix (N_obs, N_obs).
            The Laplacian matrix to input.
        """
        laplacian_mat = check_array(laplacian_mat, accept_sparse = sparse_formats)
        if laplacian_mat.shape[0] != laplacian_mat.shape[1]:
            raise ValueError("Laplacian matrix is not square")
        self.laplacian_matrix = laplacian_mat

    def delete_data_matrix(self):
        self.X = None

    def delete_adjacency_matrix(self):
        self.adjacency_matrix = None

    def delete_affinity_matrix(self):
        self.affinity_matrix = None

    def delete_laplacian_matrix(self):
        self.laplacian_matrix = None
