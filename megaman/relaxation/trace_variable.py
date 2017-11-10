# Author: Yu-Chia Chen <yuchaz@uw.edu>
# LICENSE: Simplified BSD https://github.com/mmp2/megaman/blob/master/LICENSE

import numpy as np
import os, pickle, pprint, copy

from .utils import *

class TracingVariable(object):
    """
    The TracingVariable is the class to store the variables to trace and
    print relaxation reports in each 'printiter' iteration.
    """
    def __init__(self,n,s,relaxation_kwds,precomputed_kwds,**kwargs):
        self.niter_trace = relaxation_kwds['niter_trace']
        self.niter = relaxation_kwds['niter']
        self.ltrace = 2*self.niter_trace+1

        self.loss = np.zeros(self.niter+1)
        self.etas = np.zeros(self.niter+1)
        self.H = np.zeros((self.ltrace,n,s,s))
        self.Y = np.zeros((self.ltrace,n,s))
        self.lmin = np.finfo(np.float64).max

        self.verbose = relaxation_kwds['verbose']
        self.printiter = relaxation_kwds['printiter']
        self.saveiter = relaxation_kwds['saveiter']
        self.backup_dir = relaxation_kwds['backup_dir']

        create_output_dir(self.backup_dir)
        self.report_and_save_keywords(relaxation_kwds,precomputed_kwds)

    def copy(self):
        return copy.deepcopy(self)

    def report_and_save_keywords(self,relaxation_kwds,precomputed_kwds):
        """Save relaxation keywords to .txt and .pyc file"""
        report_name = os.path.join(self.backup_dir,'relaxation_keywords.txt')
        pretty_relax_kwds = pprint.pformat(relaxation_kwds,indent=4)
        with open(report_name,'w') as wf:
            wf.write(pretty_relax_kwds)
        wf.close()

        origin_name = os.path.join(self.backup_dir,'relaxation_keywords.pyc')
        with open(origin_name,'wb') as ro:
            pickle.dump(relaxation_kwds,ro,protocol=pickle.HIGHEST_PROTOCOL)
        ro.close()

        if relaxation_kwds['presave']:
            precomp_kwds_name = os.path.join(self.backup_dir,
                                             'precomputed_keywords.pyc')
            with open(precomp_kwds_name, 'wb') as po:
                pickle.dump(precomputed_kwds, po,
                            protocol=pickle.HIGHEST_PROTOCOL)
            po.close()

    def update(self,iiter,H,Y,eta,loss):
        """Update the trace_var in new iteration"""
        if iiter <= self.niter_trace+1:
            self.H[iiter] = H
            self.Y[iiter] = Y
        elif iiter >self.niter - self.niter_trace + 1:
            self.H[self.ltrace+iiter-self.niter-1] = H
            self.Y[self.ltrace+iiter-self.niter-1] = Y

        self.etas[iiter] = eta
        self.loss[iiter] = loss
        if self.loss[iiter] < self.lmin:
            self.Yh = Y
            self.lmin = self.loss[iiter]
            self.miniter = iiter if not iiter == -1 else self.niter + 1

    def print_report(self,iiter):
        if self.verbose and iiter % self.printiter == 0:
            print ('Iteration number: {}'.format(iiter))
            print ('Last step size eta: {}'.format(self.etas[iiter]))
            print ('current loss (before gradient step): {}'
                   .format(self.loss[iiter]))
            print ('minimum loss: {}, at iteration: {}\n'
                   .format(self.lmin, self.miniter))

    def save_backup(self,iiter):
        if iiter % self.saveiter == 0 and iiter != 0:
            backup_name = os.path.join(self.backup_dir,'backup_trace.pyc')
            TracingVariable.save(self,backup_name)
            print ('Save backup at iteration: {}\n'.format(iiter))

    @classmethod
    def correct_file_extension(cls,filename):
        return os.path.splitext(filename)[0]+'.pyc'

    @classmethod
    def save(cls,instance,filename):
        """Class method save for saving TracingVariable."""
        filename = cls.correct_file_extension(filename)
        try:
            with open(filename,'wb') as f:
                pickle.dump(instance,f,protocol=pickle.HIGHEST_PROTOCOL)
        except MemoryError as e:
            print ('{} occurred, will downsampled the saved file by 20.'
                   .format(type(e).__name__))
            copy_instance = instance.copy()
            copy_instance.H = copy_instance.H[::20,:,:]
            copy_instance.Y = copy_instance.Y[::20,:]
            with open(filename,'wb') as f:
                pickle.dump(copy_instance,f,protocol=pickle.HIGHEST_PROTOCOL)

    @classmethod
    def load(cls,filename):
        """Load from stored files"""
        filename = cls.correct_file_extension(filename)
        with open(filename,'rb') as f:
            return pickle.load(f)
