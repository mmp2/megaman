from megaman.relaxation.utils import *
from nose.tools import assert_raises
import numpy as np
import numpy.testing
import shutil, warnings

n, s, d = 1000, 3, 2

basic_kwds = {
    'verbose': False,
    'niter': 2000,
    'niter_trace': 0,
    'presave': False,
    'sqrd': True,
    'alpha': 0,
    'projected': False,
    'saveiter': 10,
    'printiter': 1,
}

nonprojected_epsilon_test = {
    'lossf': 'nonprojected_epsilon',
    'projected': False,
    'eps_orth': 0.1,
}

tmp_dir = '/tmp/test_backup'
def _initialize_kwds(kwds,n,s,d):
    kwds['backup_base_dir'] = tmp_dir
    return initialize_kwds(kwds,n,s,d)

def test_default_keywords():
    calculated_kwds = _initialize_kwds({},n,s,d)
    for k,v in basic_kwds.items():
        assert calculated_kwds[k] == v, 'keyword {} do not initialized correctly.'.format(k)

    assert calculated_kwds['weights'].shape[0] == 0, 'initialized weights is not zero.'
    np.testing.assert_allclose(
        calculated_kwds['subset'], np.arange(n),
        err_msg='initialized subset should be arange(n).'
    )

def test_normalize_weights():
    weights = np.array([1,4])
    calculated_kwds = _initialize_kwds(dict(weights=weights),n,s,d)
    np.testing.assert_allclose(
        calculated_kwds['weights'], [0.2,0.8],
        err_msg='The weights should be normalized'
    )

def test_default_lossf():
    calculated_kwds = _initialize_kwds({},n,s,d)
    for k,v in nonprojected_epsilon_test.items():
        assert calculated_kwds[k] == v, 'keyword {} do not initialized correctly.'.format(k)

    calculated_kwds = _initialize_kwds(dict(projected=True),n,s,d)
    assert calculated_kwds['lossf'] == 'projected_epsilon', 'lossf should be projected_epsilon when projected is True'

    calculated_kwds = _initialize_kwds({},n,d,d)
    assert calculated_kwds['lossf'] == 'nonprojected_rloss', 'lossf should be nonprojected_rloss for default'

    calculated_kwds = _initialize_kwds(dict(projected=True),n,d,d)
    assert calculated_kwds['lossf'] == 'projected_rloss', 'lossf should be projected_epsilon when projected is True'

def test_update_lossf():
    calculated_kwds = _initialize_kwds(dict(eps_orth=0.55),n,s,d)
    assert calculated_kwds['eps_orth'] == 0.55, 'eps_orth should be updated to 0.55.'

def test_raise_lossf_error():
    assert_raises(ValueError, _initialize_kwds, dict(lossf='rloss'),n,s,d)
    assert_raises(ValueError, _initialize_kwds, dict(lossf='epsilon'),n,d,d)
    assert_raises(ValueError, _initialize_kwds, dict(projected=True, subset=np.arange(0,n,5)),n,s,d)

def test_default_momentum():
    calculated_kwds = _initialize_kwds(dict(step_method='momentum',linesearch=False),n,s,d)
    test_momentum_kwds = {
        'm': 0.05,
        'eta': 1.0
    }
    for k,v in test_momentum_kwds.items():
        assert calculated_kwds[k] == v, 'keyword {} do not initialized correctly.'.format(k)

def test_default_fixed():
    calculated_kwds = _initialize_kwds(dict(step_method='fixed',linesearch=False),n,s,d)
    assert calculated_kwds['eta'] == 1.0, 'Default eta does not match'

def test_default_linsearch():
    calculated_kwds = _initialize_kwds(dict(projected=True),n,s,d)
    test_kwds = {
        'linesearch_first': False,
        'eta_max': 2**11,
    }
    for k,v in test_kwds.items():
        assert calculated_kwds[k] == v, 'keyword {} do not initialized correctly.'.format(k)

    calculated_kwds = _initialize_kwds(dict(projected=False),n,s,d)
    assert calculated_kwds['eta_max'] == 2**4, 'eta_max should be 2**4 if projected == False'

def test_backup_dir_function():
    tmp_dir = '/tmp/test_backup'
    calculated_kwds = initialize_kwds(dict(backup_base_dir=tmp_dir),n,s,d)
    assert 'backup_dir' in calculated_kwds
    backup_dir = calculated_kwds['backup_dir']
    assert tmp_dir in backup_dir
    assert os.path.exists(tmp_dir)

def test_not_int_warnings():
    with warnings.catch_warnings(record=True) as w:
        calculated_kwds = initialize_kwds(dict(printiter=1.3),n,s,d)
        assert issubclass(w[-1].category, RuntimeWarning), \
               'Should raise RuntimeWarning when input is not integer'

def tearDownModule():
    tmp_dir = '/tmp/test_backup'
    if os.path.exists(tmp_dir):
        shutil.rmtree(tmp_dir)
