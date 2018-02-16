from petsc4py import PETSc
from slepc4py import SLEPc
import numpy as np
from scipy.sparse import csr_matrix
Print = PETSc.Sys.Print
__author__ = 'Murat Keceli'
__date__ = 'February 13, 2017'
"""
You can install slepc4py and all its dependencies with conda:
conda install -c conda-forge slepc4py
More info on PETSc and SLEPc:
https://www.mcs.anl.gov/petsc/
http://slepc.upv.es/
http://slepc.upv.es/documentation/slepc.pdf
"""
def get_petsc_matrix(Acsr,comm=PETSc.COMM_WORLD):
    """
    Given a scipy csr matrix returns PETSC AIJ matrix on a given mpi communicator.
    Parameters
    ----------
    Acsr: Array like, scipy CSR formated sparse matrix
    comm: MPI communicator
    Returns
    ----------
    PETSc AIJ Mat
    """
    A = PETSc.Mat().createAIJ(Acsr.shape[0])
    A.setUp()
    rstart, rend = A.getOwnershipRange()
    return A.createAIJ(size=Acsr.shape[0],
                       csr=(Acsr.indptr[rstart:rend+1] - Acsr.indptr[rstart],
                            Acsr.indices[Acsr.indptr[rstart]:Acsr.indptr[rend]],
                            Acsr.data[Acsr.indptr[rstart]:Acsr.indptr[rend]]),
                       comm=comm) 

def get_numpy_array(xr):
    """
    Convert a distributed PETSc Vec into a sequential numpy array.
    Parameters:
    -----------
    xr: PETSc Vec
    Returns:
    --------
    Array
    """
    xr_size = xr.getSize()
    seqx = PETSc.Vec()
    seqx.createSeq(xr_size,comm=PETSc.COMM_SELF)
    seqx.setFromOptions()
   # seqx.set(0.0)
    fromIS = PETSc.IS().createGeneral(range(xr_size),comm=PETSc.COMM_SELF)
    toIS = PETSc.IS().createGeneral(range(xr_size),comm=PETSc.COMM_SELF)
    sctr=PETSc.Scatter().create(xr,fromIS,seqx,toIS)
    sctr.begin(xr,seqx,addv=PETSc.InsertMode.INSERT,mode=PETSc.ScatterMode.FORWARD)
    sctr.end(xr,seqx,addv=PETSc.InsertMode.INSERT,mode=PETSc.ScatterMode.FORWARD)
    return seqx.getArray()


def get_eigenpairs(A, npairs=8, largest=True, comm=PETSc.COMM_WORLD):
    """
    Parameters:
    -----------
    A: scipy CSR matrix, or numpy array or PETSc mat 
    npairs: int, number of eigenpairs required
    largest: boolean, True (False) if largest (smallest) magnitude eigenvalues are requried
    comm: MPI communicator
    Returns:
    --------
    evals: 1D array of npairs, eigenvalues from largest to smallest
    evecs: 2D array of (n, npairs) where n is size of A matrix
    """
    matrixtype = str(type(A))
    if 'scipy' in matrixtype:
        n = A.shape[0]
        A = get_petsc_matrix(A, comm)
    elif 'petsc' in matrixtype:
        n = A.getSize()
    elif 'numpy' in matrixtype:
        A = csr_matrix(A)
        n = A.shape[0]
        A = get_petsc_matrix(A, comm)
    else:
        Print('Matrix type {} is not compatible. Use scipy CSR or PETSc AIJ type.'.format(type(A)))
    mpisize = comm.size
    Print('Matrix size: {}'.format(n))
    eps = SLEPc.EPS()
    eps.create()
    eps.setOperators(A)
    eps.setFromOptions() #Use command line options
    eps.setProblemType(SLEPc.EPS.ProblemType.HEP)
    if largest:
        eps.setWhichEigenpairs(SLEPc.EPS.Which.LARGEST_REAL)
    else:
        eps.setWhichEigenpairs(SLEPc.EPS.Which.SMALLEST_REAL)
    eps.setDimensions(nev=8,ncv=SLEPc.DECIDE,mpd=SLEPc.DECIDE)
    eps.solve()
    nconv = eps.getConverged()
    vr = A.getVecLeft()
    vi = A.getVecLeft()
    if nconv < npairs:
        Print("{} eigenvalues required, {} converged.".format(npairs,nconv))
        npairs = nconv
    evals = np.zeros(npairs)
    evecs = np.zeros((n,npairs))
    for i in range(npairs):
        k = eps.getEigenpair(i,vr,vi)
        if abs(k.imag) > 1.e-10:
            Print("Imaginary eigenvalue: {} + {}j".format(k.real,k.imag))
            Print("Error: {}".format(eps.computeError(i)))
        if mpisize > 1:
            evecs[:,i] = get_numpy_array(vr)
        else:
            evecs[:,i] = vr.getArray()
        evals[i] = k.real
    return evals, evecs    
