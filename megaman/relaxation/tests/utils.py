import numpy as np
import scipy as sp
import scipy.sparse
import h5py
import copy, os

def generate_toy_laplacian(n=1000):
    neighbor_counts = 10
    adjacency_mat = np.zeros((n,n))
    for i in range(n):
        x = np.ones(neighbor_counts,dtype=np.int32)*i
        y = np.random.choice(n, neighbor_counts, replace=False)
        adjacency_mat[(x,y)] = 1

    np.fill_diagonal(adjacency_mat,0)
    adjacency_mat = (adjacency_mat.T + adjacency_mat) / 2
    degree = np.sum(adjacency_mat,axis=1)
    degree_mat = np.diag(degree)

    return sp.sparse.csc_matrix(degree_mat - adjacency_mat)

def process_test_data():
    namelist = ['rloss_halfdome', 'eps_halfdome']
    return { name: process_one_loss_test_data(name) for name in namelist }

def process_one_loss_test_data(name):
    file_dir = os.path.dirname(os.path.abspath(__file__))
    path = os.path.join(file_dir,'{}.mat'.format(name))
    f = h5py.File(path)
    laplacian_ref = f['/{}/L'.format(name)]
    laplacian = sp.sparse.csc_matrix((laplacian_ref['data'], laplacian_ref['ir'], laplacian_ref['jc']))
    opts_list = ['whole_eps','half_eps','weight_eps','half_weight_eps']
    processed_data = { opts:process_one_test_data(f,name,opts) for opts in opts_list }
    processed_data['L'] = laplacian
    processed_data['d'] = 2
    return processed_data

def process_one_test_data(f, name, opts):
    Y_ref_list = f['/{}/{}/trace/Y'.format(name,opts)]
    Y_list = np.array([ f[Y_ref_list[idx,0]] for idx in range(Y_ref_list.shape[0]) ])
    Y_list = np.swapaxes(Y_list, 1, 2)

    H_ref_list = f['/{}/{}/trace/H'.format(name,opts)]
    H_list = np.array([ f[H_ref_list[idx,0]] for idx in range(H_ref_list.shape[0]) ])

    DL_ref_list = f['/{}/{}/trace/DL'.format(name,opts)]
    DL_list = np.array([ f[DL_ref_list[idx,0]] for idx in range(DL_ref_list.shape[0]-1) ])
    DL_list = np.swapaxes(DL_list, 1, 2)

    grad_ref_list = f['/{}/{}/trace/grad'.format(name,opts)]
    grad_list = np.array([ f[grad_ref_list[idx,0]] for idx in range(grad_ref_list.shape[0]-1) ])
    grad_list = np.swapaxes(grad_list, 1, 2)

    loss_list = np.squeeze(np.array(f['/{}/{}/loss'.format(name,opts)]))
    etas_list = np.squeeze(np.array(f['/{}/{}/etas'.format(name,opts)]))

    rk_h5py = f['/{}/{}/opts'.format(name,opts)]
    relaxation_kwds = {
        'alpha': rk_h5py['alpha'][0,0],
        'lossf': u''.join(chr(c) for c in rk_h5py['lossf']),
        'step_method': 'fixed',
        'linsearch': u''.join(chr(c) for c in rk_h5py['step_method']) == u'linesearch',
        'projected': rk_h5py['projected'][0,0],
        'eta_max': rk_h5py['eta_max'][0,0],
        'backup_base_dir': '/tmp/test_backup',
    }
    if 'weight' in opts:
        weights = np.squeeze(np.array(rk_h5py['w']))
        relaxation_kwds['weights'] = weights

    if 'half' in opts:
        relaxation_kwds['subset'] = np.arange(0,1000,2)

    if 'epsorth' in rk_h5py:
        relaxation_kwds['eps_orth'] = rk_h5py['epsorth'][0,0]
    if 'sqrd' in rk_h5py:
        relaxation_kwds['sqrd'] = rk_h5py['sqrd'][0,0] == 1
    return dict(
        Y_list=Y_list, H_list=H_list, DL_list=DL_list, grad_list=grad_list,
        loss_list=loss_list, etas_list=etas_list, relaxation_kwds=relaxation_kwds
    )

class Bunch(object):
    def __init__(self, adict):
        self.__dict__.update(adict)

data = process_test_data()
def gen_data(name, opts):
    test_data = copy.deepcopy(data[name])
    test_dict = test_data[opts]
    test_dict['laplacian'] = test_data['L']
    test_dict['d'] = test_data['d']
    return test_dict
