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

    #Locally compute QR at the leaves of the binomial tree
    Y_local, R_local = np.linalg.qr(A_local)
    R_recv_buff = None
    
    tree_levels = np.log2(comm.Get_size()) - 1

    #Now communicate the local R matrix up the tree to the root
    for k in np.arange(tree_levels, -1, -1):
        
        #This should drop out nodes that have already sent stuff
        if rank > np.power(2, k+1):
            break
        
        #I need to send to the neighbor to my left. In a 4 node tree, 1 -> 0, 3 -> 2, 2 -> 0 
        #Note this calculation may be negative but it is ok
        neighbor_id = ((rank+1) + 2**k) % 2**(k+1)
            
        print(k, rank, " sends to ", neighbor_id) 
        


        if rank > neighbor_id:
            comm.Send(R_local, dest=neighbor_id, tag=1)
        else:
            R_recv_buff = np.zeros_like(R_local, dtype=np.float64)
            comm.Recv(R_recv_buff, source=neighbor_id, tag=1)
            Y_local, R_local = np.linalg.qr(np.vstack([R_local, R_recv_buff]))
            Y_local_arr.append(Y_local)
        

        #Allow for sync
        comm.Barrier()


    return Y_local_arr, R_local



#################

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

assert isPowOfTwo(size), "The number of nodes must be a power of 2" 

matrix_rows = 4
matrix_columns = 4**4

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
    print(R)
