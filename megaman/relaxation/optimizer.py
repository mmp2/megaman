# Author: Yu-Chia Chen <yuchaz@uw.edu>
# LICENSE: Simplified BSD https://github.com/mmp2/megaman/blob/master/LICENSE

from __future__ import division
from megaman.geometry.utils import RegisterSubclasses

def init_optimizer(**kwargs):
    optimizer = kwargs.get('step_method', 'fixed')
    return BaseOptimizer.init(optimizer, **kwargs)

class BaseOptimizer(RegisterSubclasses):
    """
    Base class for the optimizer.

    BaseOptimizer creates the common interface to the optimzer class
    as well as providing a common apply_optimization() which can be used
    in RiemannianRelaxation class to update the embeddings.

    Parameters
    ----------
    linesearch : bool
        If use linesearch to search for optima eta.
    eta_max : float
        (Linesearch mode) The maximum learning rate (eta) to start search with.
    eta : float
        (Non linesearch mode) The fixed learning rate (eta) to use.
    linesearch_first : bool
        (Linesearch mode)  If do linesearch at first iteration.
    """
    def __init__(self, linesearch=False, eta_max=None, eta=None,
                 linesearch_first=False, **kwargs):
        self.linesearch = linesearch
        if self.linesearch:
            self.linesearch_first = linesearch_first
            if eta_max is not None:
                self.eta_max = eta_max
                self.eta_min = 2**-10
            else:
                raise ValueError('Should provide eta_max keyword '
                                 'when linesearch method is used.')
        else:
            if eta is not None:
                self.eta = eta
            else:
                raise ValueError('Should provide eta keyword '
                                 'when fixed method is used.')

    def apply_optimization(self, update_embedding_with, grad, **kwargs):
        """
        Calculating (Obtaining) the learning rate (eta) and apply optimizations
        on the embedding states by the specified method.

        Parameters
        ----------
        update_embedding_with : function
            Function used to update the state of RiemannianRelaxation
            class (Y or S).

        grad : (n x s) array
            Gradients used in updating the embedding.

        calc_loss : function (used by its child function)
            Function used to calculated the loss from the temperary state of
            RiemannianRelaxation instance. (YT or ST)

        loss : float (used by its child function)
            Loss of the current state of RiemannianRelaxation instance.
        """
        if self.linesearch:
            return self._apply_linesearch_optimzation(update_embedding_with,
                                                      grad, **kwargs)
        else:
            return self._apply_fixed_optimization(update_embedding_with,
                                                  grad, **kwargs)

    def _apply_linesearch_optimzation(self, update_embedding_with, grad,
                                      calc_loss, loss, **kwargs):
        self.eta = self.eta_max
        if kwargs.get('first_iter',False) and not self.linesearch_first:
            self.eta = kwargs.get('eta_first',1)
        loss_diff = 1
        while loss_diff > 0:
            loss_diff, temp_embedding, delta = self._linesearch_once(
                update_embedding_with,grad,calc_loss,loss,**kwargs)
            if self.eta <= self.eta_min and loss_diff > 0:
                loss_diff, temp_embedding, delta = self._linesearch_once(
                    update_embedding_with,grad,calc_loss,loss,**kwargs)
                loss_diff = -1
        self.eta *= 2
        update_embedding_with(new_embedding=temp_embedding)
        return delta

    def _linesearch_once(self, update_embedding_with, grad,
                         calc_loss, loss, **kwargs):
        delta = self._calc_delta(grad)
        temp_embedding = update_embedding_with(delta=delta,copy=True)
        loss_diff = calc_loss(temp_embedding) - loss
        self.eta /= 2
        return loss_diff, temp_embedding, delta

    def _apply_fixed_optimization(self,update_embedding_with,grad,**kwargs):
        delta = self._calc_delta(grad)
        update_embedding_with(delta=delta)
        return delta

    def _calc_delta(self,grad,**kwargs):
        raise NotImplementedError()

class FixedOptimizer(BaseOptimizer):
    """Optimizer for fixed (non-momentum) method."""
    name='fixed'
    def _calc_delta(self,grad,**kwargs):
        return -self.eta*grad

class MomentumOptimizer(BaseOptimizer):
    """Optimizer for momentum method."""
    name='momentum'
    def __init__(self,momentum,**kwargs):
        BaseOptimizer.__init__(**kwargs)
        self.momentum = momentum
        self.last_delta = 0

    def _calc_delta(self,grad,**kwargs):
        return -self.eta * grad + self.momentum * self.last_delta

    def apply_optimization(self,update_embedding_with,grad,**kwargs):
        self.last_delta = BaseOptimizer.apply_optimization(
            self,update_embedding_with,grad,**kwargs)
        return self.last_delta
