import numpy as np
from mpi4py import MPI

import sys

#Set this to true to get a CSV friendly output
CSV_OUT = True
SPARSE_MATRIX_USE = False


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
    size = comm.Get_size()

    Y_local, R_local = np.linalg.qr(A_local, mode='reduced')
    Q_matrices = [Y_local] 
    
    step = 1
    while step < comm.Get_size():
        partner = rank ^ step
        if partner < size:
            if rank < partner:
                print("Processor ", rank, ": Recieving from ", partner)
                neighbor_r = np.zeros_like(R_local, dtype=np.float64)
                comm.Recv(neighbor_r, source=partner, tag=0)
                print("Processor ", rank, ": Finished receiving.")
                Y_local, R_local = np.linalg.qr(np.vstack((R_local, neighbor_r)), mode='reduced')
                Q_matrices.append(Y_local)
            else:
                comm.Send(R_local, dest=partner, tag=0)
                
                #After we finish sending we can break the while loop since we are finished computing.
                break
        
        step*=2

    #print("Processor ", rank, ": Finished. Dropping out.")
    
    #if(rank == 0):
        #print("Number of Q matrices in 0: ", len(Q_matrices))

    return Q_matrices, R_local

def tsqr_applyQ(Q_matrices, B_local):
    for i in range(len(Q_matrices)):
        B_local = np.dot(Q_matrices[i], B_local)
    return B_local


#################

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

assert isPowOfTwo(size), "The number of nodes must be a power of 2" 

if SPARSE_MATRIX_USE:
    from scipy.io import mmread
    sparse_mat = mmread("c-67b.mtx")
    matrix_rows = 2**13
    matrix_columns = 190
else:
    matrix_rows = 2**10
    matrix_columns = 50

assert matrix_rows > matrix_columns, "The matrix is not tall is skinny. Number of rows must be greater than columns"
assert matrix_rows % size == 0, "The matrix cannot be evenly row distributed"

blocks = int(matrix_rows/size)

#Even though A is only used in process 0, we need to define it in all processors to stop
#python from complaining
A = np.empty((matrix_rows, matrix_columns), dtype=np.float64)
A_local = np.empty((blocks, matrix_columns), dtype=np.float64)
 

if rank == 0:
    # Machine precision for double is 10^-16
    if SPARSE_MATRIX_USE:
        A[:,:] = np.delete(sparse_mat.todense(), np.arange(matrix_columns, sparse_mat.shape[1]), 1)
    else:
        A = gen_matrix(matrix_rows, matrix_columns)

start = MPI.Wtime()
comm.Scatterv(A, A_local, root=0)
Q_matrices, R = tsqr(A_local, comm, matrix_rows, matrix_columns)

localIden = np.zeros_like(A_local)
Imn = np.eye(A.shape[0], A.shape[1])

comm.Scatter(Imn, localIden, root=0)
partialQ = tsqr_applyQ(Q_matrices, localIden)
Q = np.zeros_like(A)
comm.Gather(partialQ, Q, root=0)
A_reconstructed = Q @ R

if rank == 0:
    if CSV_OUT:
        end = MPI.Wtime() - start
        print(f"{end},{np.linalg.norm((A - A_reconstructed))},{np.linalg.cond(A)},{np.linalg.norm((np.eye(Q.shape[1]) - Q.T@Q))}")
    else:
        print("Total execution time: ", MPI.Wtime() - start)
        print("Accuracy of Factorisation: ", np.linalg.norm(A - A_reconstructed))
        print("Condition Number: ", np.linalg.cond(A))
        print("Loss of Orthogonality: ", np.linalg.norm((np.eye(Q.shape[1]) - Q.T@Q)))
     
