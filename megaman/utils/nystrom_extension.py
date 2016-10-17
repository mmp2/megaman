# LICENSE: Simplified BSD https://github.com/mmp2/megaman/blob/master/LICENSE
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 21 11:11:40 2016

@author: wang1
"""
from __future__ import division 
import numpy as np
import warnings
from scipy.sparse import isspmatrix
def nystrom_extension(C, e_vec, e_val):
    """
    Parameters
    ----------
    C: array-like, shape = (n, l)
      Stacking the training and testing data where n
      is the total number of data and l is the number of 
      training data.
    e_val: array, shape = (1,s)
      If W equals to C[0:l, :], then e_val are the largest s
      eig values of W
    e_vec: array-like, shape = (l, s)
      These are the corresponding eig vectors to e_val
    
    Returns
    -------
    eval_nystrom: array-like, shape = (1,s)
      These are the estimated largest s eig values of the matrix where C is the 
      first l columns.
    evec_nystrom: arrau-like, shape = (n, s)
      These are the corresponding eig vectors to eval_nystrom
      
    """
    n,l = C.shape
    W = C[0:l, :]
    eval_nystrom = (n/l)*e_val
    eval_inv = e_val.copy()
    e_nonzero = np.where(e_val != 0)
    # e_nonzero = [i for i, e in enumerate(e_val) if e != 0] #np.nonzero(a)[0]
    eval_inv[e_nonzero] = 1.0/e_val[e_nonzero]
    
    if isspmatrix(C):
        evec_nystrom = np.sqrt(l/n)*C.dot(e_vec)*eval_inv
    else:
        evec_nystrom = np.sqrt(l/n)*np.dot(C,e_vec)*eval_inv
    return eval_nystrom,evec_nystrom
    
    
    
    
    
    
    
    
    