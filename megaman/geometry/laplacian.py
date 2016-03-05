from __future__ import division ## removes integer division
import numpy as np
from scipy import sparse
from scipy.spatial.distance import pdist
import subprocess, os, sys, warnings


def compute_laplacian_matrix(affinity_matrix, method, **kwargs):
    return graph_laplacian(affinity_matrix, normed = method, **kwargs)

###############################################################################
# Graph laplacian
# Code adapted from the Matlab function laplacian.m of Dominique Perrault-Joncas
def graph_laplacian(csgraph, normed = 'geometric', symmetrize = False,
                    scaling_epps = 0., renormalization_exponent = 1,
                    return_diag = False, return_lapsym = False):
	"""
	Return the Laplacian matrix of an undirected graph.

	Computes a consistent estimate of the Laplace-Beltrami operator L
	from the similarity matrix A . See "Diffusion Maps" (Coifman and
	Lafon, 2006) and "Graph Laplacians and their Convergence on Random
	Neighborhood Graphs" (Hein, Audibert, Luxburg, 2007) for more
	details.

	A is the similarity matrix from the sampled data on the manifold M.
	Typically A is obtained from the data X by applying the heat kernel
	A_ij = exp(-||X_i-X_j||^2/EPPS). The bandwidth EPPS of the kernel is
	need to obtained the properly scaled version of L. Following the usual
	convention, the laplacian (Laplace-Beltrami operator) is defined as
	div(grad(f)) (that is the laplacian is taken to be negative
	semi-definite).

	Note that the Laplacians defined here are the negative of what is
	commonly used in the machine learning literature. This convention is used
	so that the Laplacians converge to the standard definition of the
	differential operator.

	notation: A = csgraph, D=diag(A1) the diagonal matrix of degrees
	L = lap = returned object, EPPS = scaling_epps**2

	Parameters
	----------
	csgraph : array_like or sparse matrix, 2 dimensions
		compressed-sparse graph, with shape (N, N).
	normed : string, optional
		if 'renormalized':
			compute renormalized Laplacian of Coifman & Lafon
			L = D**-alpha A D**-alpha
			T = diag(L1)
			L = T**-1 L - eye()
		if 'symmetricnormalized':
		   compute normalized Laplacian
			L = D**-0.5 A D**-0.5 - eye()
		if 'unnormalized': compute unnormalized Laplacian.
			L = A-D
		if 'randomwalks': compute stochastic transition matrix
			L = D**-1 A
	symmetrize: bool, optional
		if True symmetrize adjacency matrix (internally) before computing lap
	scaling_epps: float, optional
		if >0., it should be the same neighbors_radius that was used as kernel
		width for computing the affinity. The Laplacian gets the scaled by
		4/np.sqrt(scaling_epps) in order to ensure consistency in the limit
		of large N
	return_diag : bool, optional (kept for compatibility)
		If True, then return diagonal as well as laplacian.
	return_lapsym : bool, optional
		If normed in { 'geometric', 'renormalized' } then a symmetric matrix
		lapsym, and a row normalization vector w are also returned. Having
		these allows us to compute the laplacian spectral decomposition
		as a symmetric matrix, which has much better numerical properties.

	Returns
	-------
	lap : ndarray
		The N x N laplacian matrix of graph.
	diag : ndarray (obsolete, for compatibiility)
		The length-N diagonal of the laplacian matrix.
		diag is returned only if return_diag is True.

	Notes
	-----
	There are a few differences from the sklearn.spectral_embedding laplacian
	function.

	1. normed='unnormalized' and 'symmetricnormalized' correspond respectively
	   to normed=False and True in the latter. (Note also that normed was changed
	   from bool to string.
	2. the signs of this laplacians are changed w.r.t the original
	3. the diagonal of lap is no longer set to 0; also there is no checking if
	   the matrix has zeros on the diagonal. If the degree of a node is 0, this
	   is handled graciuously (by not dividing by 0).
	4. if csgraph is not symmetric the out-degree is used in the
	   computation and no warning is raised. However, it is not recommended to
	   use this function for directed graphs.
	"""
	if csgraph.ndim != 2 or csgraph.shape[0] != csgraph.shape[1]:
		raise ValueError('csgraph must be a square matrix or array')

	normed_options = ['unnormalized', 'geometric', 'randomwalk',
					  'symmetricnormalized', 'renormalized']
	normed = normed.lower()
	if normed == 'auto':
		normed = 'geometric'
	if normed.lower() not in normed_options:
		raise ValueError('normed must be one of {0}'.format(normed_options))

	if np.issubdtype(csgraph.dtype, np.integer):
		csgraph = csgraph.astype(np.float)

	if sparse.isspmatrix(csgraph):
		return _laplacian_sparse(csgraph, normed=normed,
								 symmetrize=symmetrize,
								 scaling_epps=scaling_epps,
								 renormalization_exponent=renormalization_exponent,
								 return_diag=return_diag,
								 return_lapsym=return_lapsym)

	else:
		return _laplacian_dense(csgraph, normed=normed,
								symmetrize=symmetrize,
								scaling_epps=scaling_epps,
								renormalization_exponent=renormalization_exponent,
								return_diag=return_diag,
								return_lapsym=return_lapsym)


def _laplacian_sparse(csgraph, normed='geometric', symmetrize=True,
                      scaling_epps=0., renormalization_exponent=1,
                      return_diag=False, return_lapsym=False):
    n_nodes = csgraph.shape[0]
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