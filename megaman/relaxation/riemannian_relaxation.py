# Author: Yu-Chia Chen <yuchaz@uw.edu>
# LICENSE: Simplified BSD https://github.com/mmp2/megaman/blob/master/LICENSE

from __future__ import division
import numpy as np
import scipy as sp
from scipy import sparse

from megaman.geometry import RiemannMetric
from megaman.geometry.utils import RegisterSubclasses

from .trace_variable import TracingVariable
from .precomputed import *
from .optimizer import init_optimizer
from .utils import *

import os

def run_riemannian_relaxation(laplacian, initial_guess,
                              intrinsic_dim, relaxation_kwds):
    """Helper function for creating a RiemannianRelaxation class."""
    n, s = initial_guess.shape
    relaxation_kwds = initialize_kwds(relaxation_kwds, n, s, intrinsic_dim)
    if relaxation_kwds['save_init']:
        directory = relaxation_kwds['backup_dir']
        np.save(os.path.join(directory, 'Y0.npy'),initial_guess)
        sp.io.mmwrite(os.path.join(directory, 'L_used.mtx'),
                      sp.sparse.csc_matrix(laplacian))

    lossf = relaxation_kwds['lossf']
    return RiemannianRelaxation.init(lossf, laplacian, initial_guess,
                                     intrinsic_dim, relaxation_kwds)

class RiemannianRelaxation(RegisterSubclasses):
    """
    The RiemannianRelaxation class is an interface for reimannian relaxation,
    which stores loss function, tracing variables and optimizers.

    Parameters
    ----------
    laplacian : (n x n) sparse matrix
        The laplacian matrix of the embedding genereated from
        other manifold learning algorithms.
    initial_guess : (n x s) array
        embedding genereated by other manifold learning algorithms
    intrinsic_dim : int
        intrinsic dimension of the manifold
    relaxation_kwds : dict
        Contains keyword arguments for RiemannianRelaxation,
        see relaxation/utils.py for arguments for each method.
    """
    def __init__(self,laplacian,initial_guess,intrinsic_dim,relaxation_kwds):
        laplacian = sp.sparse.csc_matrix(laplacian)
        self.laplacian_matrix = laplacian
        self.n, self.s = initial_guess.shape
        self.d = intrinsic_dim

        self.relaxation_kwds = relaxation_kwds
        self.Y = initial_guess
        self.H = np.zeros((self.n, self.s, self.s))

        self._init_precomp()

        self.Id = np.identity(self.d)

        self.trace_var = TracingVariable(self.n, self.s, self.relaxation_kwds,
                                         self.precomputed_kwds)
        self.eta = 0

        optimizer_kwargs, self.relaxation_kwds = \
            split_kwargs(self.relaxation_kwds)
        self.optimizer = init_optimizer(**optimizer_kwargs)

    def _init_precomp(self):
        raise NotImplementedError()

    def relax_isometry(self):
        """Main function for doing riemannian relaxation."""
        for ii in range(self.relaxation_kwds['niter']):
            self.H = self.compute_dual_rmetric()

            self.loss = self.rieman_loss()
            self.trace_var.update(ii,self.H,self.Y,self.eta,self.loss)
            self.trace_var.print_report(ii)
            self.trace_var.save_backup(ii)

            self.compute_gradient()

            self.make_optimization_step(first_iter=(ii == 0))

        self.H = self.compute_dual_rmetric()

        self.trace_var.update(-1,self.H,self.Y,self.eta,self.loss)
        self.trace_var.print_report(ii)
        tracevar_path = os.path.join(self.trace_var.backup_dir, 'results.pyc')
        TracingVariable.save(self.trace_var,tracevar_path)

    def calc_loss(self, embedding):
        """Helper function to calculate rieman loss given new embedding"""
        Hnew = self.compute_dual_rmetric(Ynew=embedding)
        return self.rieman_loss(Hnew=Hnew)

    def compute_dual_rmetric(self,Ynew=None):
        """Helper function to calculate the """
        usedY = self.Y if Ynew is None else Ynew
        rieman_metric = RiemannMetric(usedY, self.laplacian_matrix)
        return rieman_metric.get_dual_rmetric()

    def rieman_loss(self,Hnew=None):
        raise NotImplementedError()

    def compute_gradient(self):
        self.grad = self.relaxation_kwds['alpha']*self.grad
        orig_grad = np.copy(self.grad)
        for idx in self.relaxation_kwds['subset']:
            dLk = self._compute_dLk(idx)
            dLk_full = np.zeros((self.grad.shape[0],dLk.shape[1]))
            sidx = self._compute_sidx(idx)
            dLk_full[self.precomputed_kwds['nbk'][sidx],:] += dLk
            dLk_full = dLk_full * self.relaxation_kwds['weights'][idx] \
                if self.relaxation_kwds['weights'].shape[0] == self.n \
                else dLk_full / self.n
            self.grad += dLk_full
        return self.grad - orig_grad

    def _compute_dLk(self,k):
        raise NotImplementedError()

    def _compute_sidx(self,idx):
        raise NotImplementedError()

    def _part_dLk(self,Vk,nbk):
        raise NotImplementedError()

    def _get_Vk_nbk(self,k):
        raise NotImplementedError()

    def make_optimization_step(self,first_iter=False):
        self.optimizer.apply_optimization(
            lambda **kwargs: self.update_embedding_with(**kwargs),
            self.grad,
            calc_loss=lambda embedding: self.calc_loss(embedding),
            loss=self.loss,
            first_iter=first_iter,
        )
        self.eta = self.optimizer.eta

    def update_embedding_with(self,**kwargs):
        delta = kwargs.get('delta', None)
        new_embedding = kwargs.get('new_embedding', None)
        if delta is not None and new_embedding is None:
            return self._update_with_delta(delta,copy=kwargs.get('copy', False))
        elif new_embedding is not None and delta is None:
            return self._update_with_embedding(new_embedding)
        else:
            raise ValueError()

    def _update_with_delta(self,delta,copy=False):
        raise NotImplementedError()
    def _update_with_embedding(self,embedding):
        raise NotImplementedError()

class ProjectedClass(RiemannianRelaxation):
    name_prefix='projected'
    def _init_precomp(self):
        self.precomputed_kwds = precompute_optimzation_S(
            self.laplacian_matrix, self.n, self.relaxation_kwds)
        print ('Finish computing S')
        self.S = self.precomputed_kwds['A'].dot(self.Y)
        self.n_S = self.S.shape[0]
        self.grad = np.zeros((self.n_S, self.s))

    def _get_Vk_nbk(self,k):
        return self.precomputed_kwds['RK'][k], self.precomputed_kwds['nbk'][k]

    def _update_with_delta(self,delta,copy=False):
        ST = self.S + delta
        YT = np.asarray(
            self.precomputed_kwds['ATAinv']
            .dot(self.precomputed_kwds['A'].T.dot(ST))
        )
        if not copy:
            self.S = ST
            self.Y = YT
        return YT

    def _update_with_embedding(self,embedding):
        self.Y = embedding
        self.S = self.precomputed_kwds['A'].dot(self.Y)

    def _compute_sidx(self,idx):
        return idx

class NonProjectedClass(RiemannianRelaxation):
    name_prefix='nonprojected'
    def _init_precomp(self):
        self.precomputed_kwds = precompute_optimzation_Y(
            self.laplacian_matrix,self.n,self.relaxation_kwds)
        self.grad = np.zeros((self.n, self.s))
        self.Y -= np.mean(self.Y, axis=0)

    def _get_Vk_nbk(self,k):
        try:
            sidx = self.precomputed_kwds['si_map'][k]
        except Exception as e:
            raise ValueError('the index k should be in the subset.')
        return self.precomputed_kwds['Lk'][sidx], \
               self.precomputed_kwds['nbk'][sidx]

    def _update_with_delta(self,delta,copy=False):
        YT = self.Y + delta
        if not copy:
            self.Y = YT
        return YT

    def _update_with_embedding(self,embedding):
        self.Y = embedding
        return embedding

    def _compute_sidx(self,idx):
        return self.precomputed_kwds['si_map'][idx]

class EpsilonRiemannianRelaxation(RiemannianRelaxation):
    name_suffix = 'epsilon'
    def __init__(self,laplacian,initial_guess,intrinsic_dim,relaxation_kwds):
        RiemannianRelaxation.__init__(self, laplacian, initial_guess,
                                      intrinsic_dim, relaxation_kwds)
        self.epsI = self.relaxation_kwds['eps_orth']*np.identity(self.s)
        self.H = self.compute_dual_rmetric()
        self.UU, self.IUUEPS = compute_principal_plane(self.H, self.epsI, self.d)
        self.HUU = np.zeros((self.n,self.s,self.s))

    def rieman_loss(self,Hnew=None):
        used_H = self.H if Hnew is None else Hnew
        subset = self.relaxation_kwds['subset']
        err = np.zeros(self.n)
        err[subset] = np.linalg.norm(
            np.matmul(self.IUUEPS[subset],np.matmul(
                used_H[subset]-self.UU[subset],self.IUUEPS[subset])),
            axis=(1,2),ord=2)
        loss = np.mean(err) \
               if not self.relaxation_kwds['weights'].shape[0] == self.n \
               else self.relaxation_kwds['weights'].T.dot(err)
        return loss

    def _compute_dLk(self,k):
        Uk = principal_space(self.H[k], self.d)
        self.UU[k], self.IUUEPS[k], self.HUU[k] = \
            epsilon_norm(self.H[k],Uk,self.epsI)
        Vk, nbk = self._get_Vk_nbk(k)
        argmax_eig_vec, fact, lambda_max_abs_v = matrix_derivative(self.HUU[k])
        v = self.IUUEPS[k].dot(argmax_eig_vec)
        dLK = self._part_dLk(Vk,nbk).dot( np.tensordot(v, v,axes=0) )*fact
        if self.relaxation_kwds['sqrd']:
            dLK = 2*lambda_max_abs_v*dLK
        return dLK

class ProjectedEpsilonRiemannianRelaxation(EpsilonRiemannianRelaxation,
                                           ProjectedClass):
    name='{}_{}'.format(ProjectedClass.name_prefix,
                        EpsilonRiemannianRelaxation.name_suffix)
    def _part_dLk(self,Vk,nbk):
        return np.multiply(Vk.reshape(-1,1),self.S[nbk,:])

class NonprojectedEpsilonRiemannianRelaxation(EpsilonRiemannianRelaxation,
                                              NonProjectedClass):
    name='{}_{}'.format(NonProjectedClass.name_prefix,
                        EpsilonRiemannianRelaxation.name_suffix)
    def _part_dLk(self,Vk,nbk):
        return Vk.dot(self.Y[nbk,:])

class RLossRiemannianRelaxation(RiemannianRelaxation):
    name_suffix = 'rloss'
    def __init__(self,laplacian,initial_guess,intrinsic_dim,relaxation_kwds):
        RiemannianRelaxation.__init__(self, laplacian, initial_guess,
                                      intrinsic_dim,relaxation_kwds)

    def rieman_loss(self,Hnew=None):
        used_H = self.H if Hnew is None else Hnew
        subset = self.relaxation_kwds['subset']
        err = np.zeros(self.n)
        err[subset] = np.linalg.norm(
            used_H[subset] - self.Id[None,...], axis=(1,2), ord=2)
        # for k in subset:
        #     U = used_H[k] - self.Id
        #     err[k] = np.linalg.norm(U,ord=2)
        loss = np.mean(err) \
               if self.relaxation_kwds['weights'].shape[0] != self.n \
               else self.relaxation_kwds['weights'].T.dot(err)
        return loss

    def _compute_dLk(self,k):
        argmax_eig_vec, fact, lambda_max_abs_v = \
            matrix_derivative(self.H[k] - self.Id)
        Vk, nbk = self._get_Vk_nbk(k)
        dLk = self._part_dLk(Vk,nbk)\
              .dot( np.tensordot(argmax_eig_vec, argmax_eig_vec,axes=0) )*fact
        if self.relaxation_kwds['sqrd']:
            dLk = 2*lambda_max_abs_v*dLk
        return dLk

class ProjectedRLossRiemannianRelaxation(RLossRiemannianRelaxation,
                                         ProjectedClass):
    name='{}_{}'.format(ProjectedClass.name_prefix,
                        RLossRiemannianRelaxation.name_suffix)
    def _part_dLk(self,Vk,nbk):
        return np.multiply(Vk.reshape(-1,1),self.S[nbk,:])

class NonprojectedRLossRiemannianRelaxation(RLossRiemannianRelaxation,
                                            NonProjectedClass):
    name='{}_{}'.format(NonProjectedClass.name_prefix,
                        RLossRiemannianRelaxation.name_suffix)
    def _part_dLk(self,Vk,nbk):
        return Vk.dot(self.Y[nbk,:])


def matrix_derivative(U):
    eig_vals, eigen_vecs = np.linalg.eigh(U)
    argmax_eig = np.argmax(np.absolute(eig_vals))
    fact = -1 if eig_vals[argmax_eig] < 0 else 1
    argmax_eig_vec = eigen_vecs[:,argmax_eig]
    return argmax_eig_vec,fact,np.absolute(eig_vals[argmax_eig])

def epsilon_norm(Hk,Uk,epsI):
    UUk = Uk.dot(Uk.T)
    # TODO: why inverse here?
    iUUepk = np.linalg.inv(sqrtm_psd(UUk+epsI))
    HUUK = iUUepk.dot(Hk - UUk).dot(iUUepk)
    return UUk, iUUepk, HUUK

def compute_principal_plane(H, epsI, intrinsic_dim):
    n_samples, n_components = H.shape[:2]
    UU = np.zeros((n_samples, n_components, n_components))
    IUUEPS = np.zeros((n_samples, n_components, n_components))
    for k in range(n_samples):
        Uk = principal_space(H[k],intrinsic_dim)
        UU[k] = np.dot(Uk,Uk.T)
        IUUEPS[k] = np.linalg.inv(sqrtm_psd(UU[k]+epsI))
    return UU, IUUEPS

def principal_space(Hk,intrinsic_dim):
    eig_vals, eigen_vecs = np.linalg.eigh(Hk)
    eig_vals = eig_vals[::-1]
    Uk = eigen_vecs[:,:-1-intrinsic_dim:-1]
    return Uk

def sqrtm_psd(M):
    vals, vecs = np.linalg.eigh(M)
    vals = np.sqrt(np.maximum(vals,0))
    return vecs.dot(np.diag(vals)).dot(vecs.conj().T)
