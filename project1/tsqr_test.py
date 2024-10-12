import numpy as np
from mpi4py import MPI

import sys

#Set this to true to get a CSV friendly output
CSV_OUT = True


# put this somewhere but before calling the asserts
sys_excepthook = sys.excepthook
def mpi_excepthook(type, value, traceback): 
    sys_excepthook(type, value, traceback) 
    if MPI.COMM_WORLD.size > 1:
        MPI.COMM_WORLD.Abort(1) 
#This will kill all processes when an assertion error is triggered
sys.excepthook = mpi_excepthook 

def f(x, mu):
    return (np.sin(10*(mu+x))) / (np.cos(100*(mu-x)) + 1.1) 

def isPowOfTwo(x):
    return (x & (x-1)) == 0


def gen_matrix(rows, columns):
    A = np.zeros((rows, columns), dtype=np.float64)
    
    for i in range(rows):
        for j in range(columns):
            A[i, j] = f((i-1)/(rows-1), (j-1)/(columns-1))

    return A


def tsqr(A_local, comm, matrix_rows, matrix_cols):
    rank = comm.Get_rank()
    Y_local_arr = []

    Y_local, R_local = np.linalg.qr(A_local)
    Y_local_arr.append(Y_local)
    step = 1
    while step < comm.Get_size():
        if rank % 2**step == 0:
            if rank +step < size:
                print(rank, " recieving from ", rank+step)
                neighbor_r = np.zeros_like(R_local, dtype=np.float64)
                comm.Recv(neighbor_r, source=rank+step, tag=1)
                print(rank, " finished receiving")
                Y_local, R_local = np.linalg.qr(np.vstack((R_local, neighbor_r)))
                Y_local_arr.append(Y_local)
        else:
            print(rank, " sending to ", rank-step)
            comm.Send(R_local, dest=rank-step, tag=1)
            print(rank, " finished Sending")
            break
        comm.Barrier()
	step*=2

    return Y_local_arr, R_local

#################

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

assert isPowOfTwo(size), "The number of nodes must be a power of 2" 

matrix_rows = 4**4
matrix_columns = 4

assert matrix_rows > matrix_columns, "The matrix is not tall is skinny. Number of rows must be greater than columns"
assert matrix_rows % size == 0, "The matrix cannot be evenly row distributed"

blocks = int(matrix_rows/size)

#Even though A is only used in process 0, we need to define it in all processors to stop
#python from complaining
A = np.empty((matrix_rows, matrix_columns), dtype=np.float64)
A_local = np.empty((blocks, matrix_columns), dtype=np.float64)
Q = np.empty((matrix_rows, matrix_columns), dtype=np.float64) 

if rank == 0:
    # Machine precision for double is 10^-16
    A = gen_matrix(matrix_rows, matrix_columns)

start = MPI.Wtime()
comm.Scatterv(A, A_local, root=0)
Y_arr, R = tsqr(A_local, comm, matrix_rows, matrix_columns)
if rank == 0:
    assert np.allclose(R, np.triu(R)), "R is not upper triangular"
    print(R.shape)
    Q_test, R_test = np.linalg.qr(A)

    print(R)
    print(R_test)
