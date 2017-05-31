import numpy as np
import scipy as sp
import cPickle
from scipy.io import loadmat, savemat
from scipy.sparse import coo_matrix, dia_matrix, identity

def save_sparse_in_2_parts(A, name):
    # mat and coo format easily readable into MATLAB
    nz = len(A.data)
    A = A.tocoo()
    A_1 = {'I1':A.row[xrange(0, int(nz/2))],
           'J1':A.col[xrange(0, int(nz/2))],
           'V1':A.data[xrange(0, int(nz/2))]}
    savemat(name + '_part_1.mat', A_1)

    A_2 = {'I2':A.row[xrange(int(nz/2), nz)],
           'J2':A.col[xrange(int(nz/2), nz)],
           'V2':A.data[xrange(int(nz/2), nz)]}
    savemat(name + '_part_2.mat', A_2)
    return(None)
    
def load_sparse_in_2_parts(f1, f2, n):
    A_1 = loadmat(f1)
    A_2 = loadmat(f2)
    row = np.append(A_1['I1'], A_2['I2'])
    col = np.append(A_1['J1'], A_2['J2'])
    data = np.append(A_1['V1'], A_2['V2'])
    A = coo_matrix((data, (row, col)), shape = (n, n))
    return(A)
    
    
def save_sparse_in_k_parts(A, name, k):
    nz = len(A.data)
    A = A.tocoo()
    nk = 0 
    nper = int(nz / k)
    for ii in range(k):
        fname = name + '_part_' + str(ii+1) + '.mat'
        nkp1 = nk + nper 
        if ii == k-1:
            nkp1 = nz 
        A_k = {'I':A.row[xrange(nk, nkp1)],
               'J':A.col[xrange(nk, nkp1)],
               'V':A.data[xrange(nk, nkp1)]}
        savemat(fname, A_k)
        nk = nkp1
    return(None)
    
def load_sparse_in_k_parts(name, k, n):
    row = np.array([])
    col = np.array([])
    data = np.array([])
    for ii in range(k):
        fname = name + '_part_' + str(ii+1) + '.mat'
        A_k = loadmat(fname)
        row = np.append(row, A_k['I'])
        col = np.append(col, A_k['J'])
        data = np.append(data, A_k['V'])
    A = coo_matrix((data, (row, col)), shape = (n, n))
    return(A)
    
def dump_array_in_k_parts(A, name, k):
    n = A.shape[0]
    nk = 0 
    nper = int(n / k)
    for ii in range(k):
        fname = name + '_part_' + str(ii+1) + '.p'
        nkp1 = nk + nper 
        if ii == k-1:
            nkp1 = n
        A_k = A[range(nk, nkp1)]
        cPickle.dump(A_k, open(fname, 'wb'), -1)
        nk = nkp1
    return(None)

def load_array_in_k_parts(name, k):
    for ii in range(k):
        fname = name + '_part_' + str(ii+1) + '.p'
        A_k = cPickle.load(open(fname, 'rb'))
        if ii == 0:
            A = A_k.copy()
        else:
            A = np.vstack((A, A_k))
    return(A)
    
def set_sparse_diag_to_one(mat):
    # appears to implicitly convert to csr which might be a problem 
    (n, n) = mat.shape
    # copy the matrix, subtract the diagonal values, add identity matrix 
    # see http://nbviewer.jupyter.org/gist/Midnighter/9992103 for speed testing
    cpy = mat - dia_matrix((mat.diagonal()[sp.newaxis, :], [0]), shape=(n, n)) + identity(n)
    return(cpy)
    
def set_coo_diag_to_one(mat):
    # this function takes a coo matrix and sets diagonal to one
    (n, n) = mat.shape
    off_diag = np.where(mat.row != mat.col)[0]
    row = np.append(mat.row[off_diag], range(n))
    col = np.append(mat.col[off_diag], range(n))
    data = np.append(mat.data[off_diag], np.ones(n))
    cpy = coo_matrix((data, (row, col)), shape = (n, n))
    return(cpy)