from megaman.relaxation import *
from functools import wraps

import numpy as np
import numpy.testing

from .utils import gen_data, Bunch
import shutil

def _regression_test(if_epsilon):
    def _test_deco(func):
        @wraps(func)
        def wrapper():
            test_dict = func()
            var = Bunch(test_dict)

            rr = run_riemannian_relaxation(var.laplacian, var.Y_list[0], var.d, var.relaxation_kwds)

            calculated_loss_list = []
            calculated_DL_list = []
            calculated_Y_list = []

            for idx,Y in enumerate(var.Y_list):
                rr.Y = Y
                rr.H = np.copy(var.H_list[idx])
                if if_epsilon and idx >= 1:
                    rr.UU, rr.IUUEPS = compute_principal_plane(var.H_list[idx-1],rr.epsI,var.d)
                calculated_loss_list.append(rr.rieman_loss())

            for idx,H in enumerate(var.H_list):
                rr.H = H
                rr.Y = np.copy(var.Y_list[idx])
                calculated_DL_list.append(rr.compute_gradient())

            for idx,grad in enumerate(var.grad_list):
                rr.grad = grad
                rr.Y = np.copy(var.Y_list[idx])
                rr.loss = var.loss_list[idx]
                if if_epsilon:
                    rr.H = rr.compute_dual_rmetric()
                    rr.UU, rr.IUUEPS = compute_principal_plane(rr.H,rr.epsI,var.d)
                rr.make_optimization_step(first_iter=(idx == 0))
                calculated_Y_list.append(rr.Y)

            np.testing.assert_allclose(
                calculated_loss_list, var.loss_list,
                err_msg='Loss calculated from matlab should be similar to that calculated from python, in {}'.format(__name__)
            )
            np.testing.assert_allclose(
                calculated_DL_list[:-1], var.DL_list,
                err_msg='gradient difference calculated from matlab should be similar to that calculated from python, in {}'.format(__name__)
            )
            np.testing.assert_allclose(
                calculated_Y_list, var.Y_list[1:],
                err_msg='Y calculated from linesearch should be similar, in {}'.format(__name__)
            )

        return wrapper
    return _test_deco

@_regression_test(True)
def test_whole_eps():
    return gen_data('eps_halfdome','whole_eps')

@_regression_test(False)
def test_whole_rloss():
    return gen_data('rloss_halfdome','whole_eps')

@_regression_test(True)
def test_half_eps():
    return gen_data('eps_halfdome','half_eps')

@_regression_test(False)
def test_half_rloss():
    return gen_data('rloss_halfdome','half_eps')

@_regression_test(True)
def test_weight_eps():
    return gen_data('eps_halfdome','weight_eps')

@_regression_test(False)
def test_weight_rloss():
    return gen_data('rloss_halfdome','weight_eps')

@_regression_test(True)
def test_half_weight_eps():
    return gen_data('eps_halfdome','half_weight_eps')

@_regression_test(False)
def test_half_weight_rloss():
    return gen_data('rloss_halfdome','half_weight_eps')

if __name__ == '__main__':
    test_weight_rloss()

def tearDownModule():
    tmp_dir = '/tmp/test_backup'
    if os.path.exists(tmp_dir):
        shutil.rmtree(tmp_dir)
