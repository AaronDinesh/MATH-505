from mpi4py import MPI
import numpy as np


def f(x, mu):
    return (np.sin(10*(mu+x))) / (np.cos(100*(mu-x)) + 1.1) 

def gen_matrix(rows, columns):
    A = np.zeros((rows, columns), dtype=np.float64)
    
    for i in range(rows):
        for j in range(columns):
            A[i, j] = f((i-1)/(rows-1), (j-1)/(columns-1))

    return A


def mat_vec_mul(A, x):
    return A @ x

def parallel_cholesky(A_local, comm):
    G = np.empty((A_local.shape[1], A_local.shape[1]), dtype=np.float64)
    G_local = A_local.T @ A_local 
    
    #Turns out you can send variables from within functions to other functions
    G = comm.allreduce(G_local)
    R = np.linalg.cholesky(G)
    Q_local = A_local @ np.linalg.inv(R.T)
    assert Q_local.shape == A_local.shape
    return Q_local, R.T
    
    
###########################

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

#These seem to be the values that result in a PSD matrix
matrix_rows = 12   # Can go up to 2**7
matrix_columns = 5 # Can go up to 2**6

blocks = int(matrix_rows/size)

#Even though A is only used in process 0, we need to define it in all processors to stop
#python from complaining
A = np.empty((matrix_rows, matrix_columns), dtype=np.float64)
A_local = np.empty((blocks, matrix_columns), dtype=np.float64)
Q = np.empty((matrix_rows, matrix_columns), dtype=np.float64) 

if rank == 0:
    A = gen_matrix(matrix_rows, matrix_columns)
    assert np.all(np.linalg.eigvals(A.T @ A) > 0), "The A.T@A is not a positive semi-definite matrix"

#Split A among all the nodes
comm.Scatterv(A, A_local, root=0)

#Compute cholesky in parallel
Q_local, R = parallel_cholesky(A_local, comm)

#Gather all the local Q's from the nodes to node 0
comm.Gatherv(Q_local, Q, root=0)


if rank == 0:
    print(A)
    print(Q@R)
    print(np.abs(A - Q @ R))

