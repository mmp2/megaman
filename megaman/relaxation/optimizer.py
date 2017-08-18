from megaman.geometry.utils import RegisterSubclasses

def init_optimizer(**kwargs):
    optimizer = kwargs.get('step_method', 'fixed')
    return BaseOptimizer.init(optimizer, **kwargs)

class BaseOptimizer(RegisterSubclasses):
    def __init__(self,linesearch=False, eta_max=None, eta=None, linesearch_first=False, **kwargs):
        self.linesearch = linesearch
        if self.linesearch:
            self.linesearch_first = linesearch_first
            if eta_max is not None:
                self.eta_max = eta_max
                self.eta_min = 2**-10
            else:
                raise ValueError('Should given eta_max when specified linesearch')
        else:
            if eta is not None:
                self.eta = eta
            else:
                raise ValueError('Should given eta when specified fixed')

    def apply_optimization(self,update_embedding_with,grad,**kwargs):
        if self.linesearch:
            return self._apply_linesearch_optimzation(update_embedding_with,grad,**kwargs)
        else:
            return self._apply_fixed_optimization(update_embedding_with,grad,**kwargs)

    def _apply_fixed_optimization(self,update_embedding_with,grad,**kwargs):
        delta = self._calc_delta(grad)
        update_embedding_with(delta=delta)
        return delta

    def _apply_linesearch_optimzation(self,update_embedding_with,grad,calc_loss,loss,**kwargs):
        self.eta = self.eta_max
        if kwargs.get('first_iter',False) and not self.linesearch_first:
            self.eta = kwargs.get('eta_first',1)
        loss_diff = 1
        while loss_diff > 0:
            loss_diff, temp_embedding, delta = self._linesearch_once(update_embedding_with,grad,calc_loss,loss,**kwargs)
            if self.eta <= self.eta_min and loss_diff > 0:
                loss_diff, temp_embedding, delta = self._linesearch_once(update_embedding_with,grad,calc_loss,loss,**kwargs)
                loss_diff = -1
        self.eta *= 2
        update_embedding_with(new_embedding=temp_embedding)
        return delta

    def _linesearch_once(self,update_embedding_with,grad,calc_loss,loss,**kwargs):
        delta = self._calc_delta(grad)
        temp_embedding = update_embedding_with(delta=delta,copy=True)
        loss_diff = calc_loss(temp_embedding) - loss
        self.eta /= 2
        return loss_diff, temp_embedding, delta

    def _calc_delta(self,grad,**kwargs):
        raise NotImplementedError()

class FixedOptimizer(BaseOptimizer):
    name='fixed'
    def _calc_delta(self,grad,**kwargs):
        return -self.eta*grad

class MomentumOptimizer(BaseOptimizer):
    name='momentum'
    def __init__(self,momentum,**kwargs):
        BaseOptimizer.__init__(**kwargs)
        self.momentum = momentum
        self.last_delta = 0 # TODO: need to change it a little bit.

    def _calc_delta(self,grad,**kwargs):
        return -self.eta * grad + self.momentum * self.last_delta

    def apply_optimization(self,update_embedding_with,grad,**kwargs):
        self.last_delta = BaseOptimizer.apply_optimization(self,update_embedding_with,grad,**kwargs)
        return self.last_delta
