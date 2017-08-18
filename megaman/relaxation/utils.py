import numpy as np
import time, os

default_basedir = os.path.join(os.getcwd(), 'backup')

def split_kwargs(relaxation_kwds):
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


# TODO: It should be make into OOP but not now.
def initialize_kwds(relaxation_kwds, n_samples, n_components, intrinsic_dim):
    new_relaxation_kwds = {
        'weights': np.array([],dtype=np.float64),
        'step_method': 'fixed',
        'linesearch': True,
        'verbose': False,
        'niter': 2000,
        'niter_trace': 0,
        'preload': False,
        'presave': False,
        'sqrd': True,
        'alpha': 0,
        'projected': False,
        'lossf': 'epsilon' if n_components > intrinsic_dim else 'rloss',
        'subset': np.arange(n_samples),
        'sub_dir': current_time_str(),
        'saveiter': 10,
        'printiter': 1,
        'savebackup': True,
        'backup_base_dir': default_basedir,
        'save_init': False,
    }

    # TODO: add to choose not save!
    new_relaxation_kwds.update(relaxation_kwds)
    # new_results_dir = os.path.join(new_relaxation_kwds['base_dir'], new_relaxation_kwds['sub_dir'])
    # new_relaxation_kwds.setdefault('results_dir', new_results_dir)

    backup_dir = os.path.join(new_relaxation_kwds['backup_base_dir'], new_relaxation_kwds['sub_dir'])
    new_relaxation_kwds['backup_dir'] = backup_dir
    create_output_dir(backup_dir)

    if new_relaxation_kwds['weights'].shape[0] != 0:
        new_relaxation_kwds['weights'] = np.absolute(new_relaxation_kwds['weights'])
        weight_norm = np.sum(new_relaxation_kwds['weights'])
        new_relaxation_kwds['weights'] = new_relaxation_kwds['weights'] / weight_norm

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
