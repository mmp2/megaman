from __future__ import division
from .utils import generate_toy_laplacian
from megaman.relaxation.trace_variable import TracingVariable as tv
from megaman.relaxation import *
import shutil, os

def test_copy():
    n, s, d = 1000, 3, 2
    niter = 10
    niter_trace = niter//2
    ltrace = 2*niter_trace+1
    L = generate_toy_laplacian(n)
    Y0 = np.zeros((n,s))
    rr = run_riemannian_relaxation(L, Y0, d, dict(niter=niter, niter_trace=niter_trace))
    copied_tv = rr.trace_var.copy()
    copied_tv.H = copied_tv.H[::2,:,:]
    assert rr.trace_var.H.shape[0] == ltrace, 'The original size of H should not be affected by downsamping'
    assert copied_tv.H.shape[0] == round(ltrace / 2), 'The size of copied H should be downsampled by 2'

def tearDownModule():
    tmp_dir = '/tmp/test_backup'
    if os.path.exists(tmp_dir):
        shutil.rmtree(tmp_dir)
