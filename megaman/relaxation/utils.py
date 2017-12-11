# Author: Yu-Chia Chen <yuchaz@uw.edu>
# LICENSE: Simplified BSD https://github.com/mmp2/megaman/blob/master/LICENSE

from __future__ import division
import numpy as np
import time, os, warnings

default_basedir = os.path.join(os.getcwd(), 'backup')

def split_kwargs(relaxation_kwds):
    """Split relaxation keywords to keywords for optimizer and others"""
    optimizer_keys_list = [
        'step_method',
        'linesearch',
        'eta_max',
        'eta',
        'm',
        'linesearch_first'
    ]
    optimizer_kwargs = { k:relaxation_kwds.pop(k) for k in optimizer_keys_list if k in relaxation_kwds }
    if 'm' in optimizer_kwargs:
        optimizer_kwargs['momentum'] = optimizer_kwargs.pop('m')
    return optimizer_kwargs, relaxation_kwds


def initialize_kwds(relaxation_kwds, n_samples, n_components, intrinsic_dim):
    """
    Initialize relaxation keywords.

    Parameters
    ----------
    relaxation_kwds : dict
        weights : numpy array, the weights
        step_method : string { 'fixed', 'momentum' }
            which optimizers to use
        linesearch : bool
            whether to do linesearch in search for eta in optimization
        verbose : bool
            whether to print reports to I/O when doing relaxation
        niter : int
            number of iterations to run.
        niter_trace : int
            number of iterations to be traced.
        presave : bool
            whether to store precomputed keywords to files or not.
        sqrd : bool
            whether to use squared norm in loss function. Default : True
        alpha : float
            shrinkage rate for previous gradient. Default : 0
        projected : bool
            whether or not to optimize via projected gradient descent on differences S
        lossf : string { 'epsilon', 'rloss' }
            which loss function to optimize.
            Default : 'rloss' if n == d, otherwise 'epsilon'
        subset : numpy array
            Subset to do relaxation on.
        sub_dir : string
            sub_dir used to store the outputs.
        backup_base_dir : string
            base directory used to store outputs
            Final path will be backup_base_dir/sub_dir
        saveiter : int
            save backup on every saveiter iterations
        printiter : int
            print report on every printiter iterations
        save_init : bool
            whether to save Y0 and L before running relaxation.
    """
    new_relaxation_kwds = {
        'weights': np.array([],dtype=np.float64),
        'step_method': 'fixed',
        'linesearch': True,
        'verbose': False,
        'niter': 2000,
        'niter_trace': 0,
        'presave': False,
        'sqrd': True,
        'alpha': 0,
        'projected': False,
        'lossf': 'epsilon' if n_components > intrinsic_dim else 'rloss',
        'subset': np.arange(n_samples),
        'sub_dir': current_time_str(),
        'backup_base_dir': default_basedir,
        'saveiter': 10,
        'printiter': 1,
        'save_init': False,
    }

    new_relaxation_kwds.update(relaxation_kwds)

    backup_dir = os.path.join(new_relaxation_kwds['backup_base_dir'], new_relaxation_kwds['sub_dir'])
    new_relaxation_kwds['backup_dir'] = backup_dir
    create_output_dir(backup_dir)

    new_relaxation_kwds = convert_to_int(new_relaxation_kwds)

    if new_relaxation_kwds['weights'].shape[0] != 0:
        weights = np.absolute(new_relaxation_kwds['weights']).astype(np.float64)
        new_relaxation_kwds['weights'] = weights / np.sum(weights)

    if new_relaxation_kwds['lossf'] == 'epsilon':
        new_relaxation_kwds.setdefault('eps_orth', 0.1)

    if n_components != intrinsic_dim and new_relaxation_kwds['lossf'] == 'rloss':
        raise ValueError('loss function rloss is for n_components equal intrinsic_dim')

    if n_components == intrinsic_dim and new_relaxation_kwds['lossf'] == 'epsilon':
        raise ValueError('loss function rloss is for n_components equal intrinsic_dim')

    if new_relaxation_kwds['projected'] and new_relaxation_kwds['subset'].shape[0] < n_samples:
        raise ValueError('Projection derivative not working for subset methods.')

    prefix = 'projected' if new_relaxation_kwds['projected'] else 'nonprojected'
    new_relaxation_kwds['lossf'] = '{}_{}'.format(prefix,new_relaxation_kwds['lossf'])
    step_method = new_relaxation_kwds['step_method']

    if new_relaxation_kwds['linesearch'] == True:
        new_relaxation_kwds.setdefault('linesearch_first', False)
        init_eta_max = 2**11 if new_relaxation_kwds['projected'] else 2**4
        new_relaxation_kwds.setdefault('eta_max',init_eta_max)
    else:
        new_relaxation_kwds.setdefault('eta', 1.0)

    if step_method == 'momentum':
        new_relaxation_kwds.setdefault('m', 0.05)

    return new_relaxation_kwds


def convert_to_int(kwds):
    keys_to_convert = ['printiter', 'saveiter', 'niter', 'niter_trace']
    for k in keys_to_convert:
        if type(kwds[k]) != int:
            kwds[k] = int(round(kwds[k]))
            warnings.warn('The input value of key {} should be an integer, ' + \
                          'will round to integer automatically'.format(k),
                          RuntimeWarning )
    return kwds

def current_time_str():
    rand_str = generate_random_str()
    return time.strftime('%m-%d-%y_%H:%M:%S', time.localtime() ) + '_{}'.format(rand_str)

def generate_random_str(N=5):
    import random, string
    return ''.join(random.choice(string.ascii_lowercase + string.digits) for _ in range(N))

def create_output_dir(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)
    return directory
