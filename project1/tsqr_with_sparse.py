import numpy as np
from mpi4py import MPI

import sys

#Set this to true to get a CSV friendly output
CSV_OUT = True
SPARSE_MATRIX_USE = False
COMPUTE_Q = True

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


def tsqr(A_local, comm, matrix_rows, matrix_cols, COMPUTE_Q):
    rank = comm.Get_rank()
    size = comm.Get_size()
    Q_matrix = None
    Y_local, R_local = np.linalg.qr(A_local, mode='reduced')
    Q_matrices = [Y_local] 
    step = 1
    while step < comm.Get_size():
        partner = rank ^ step
        if partner < size:
            if rank < partner:
                #print("Processor ", rank, ": Recieving from ", partner)
                neighbor_r = np.zeros_like(R_local, dtype=np.float64)
                comm.Recv(neighbor_r, source=partner, tag=0)
                Y_local, R_local = np.linalg.qr(np.vstack((R_local, neighbor_r)), mode='reduced')
                Q_matrices.append(Y_local)
            else:
                #print("Processor ", rank, ": Sending R to ", rank-step)
                comm.Send(R_local, dest=partner, tag=0)
                #print("Processor ", rank, ": Finished sending R")
                
                #After we finish sending we can break the while loop since we are finished computing.
                break
        
        step*=2
    
    if COMPUTE_Q:
        for j in range(int(np.log2(size))-1, -1, -1):
            step = 2 ** j
        
            partner = rank ^ step
                #if i am the lower rank I need to send. Else I need to receive
            if partner < size and (rank % step == 0):

                if rank < partner:
                    q_pop = Q_matrices.pop()
                    first_half, second_half = np.split(q_pop, 2, axis=0)
                    comm.Send(second_half, dest=partner, tag=2)
                    Q_matrices[-1] = Q_matrices[-1] @ first_half
                else:
                    tmp_holding_var = np.zeros(((Q_matrices[-1].shape[1]), Q_matrices[-1].shape[1]))
                    comm.Recv(tmp_holding_var, source=partner, tag=2)
                    Q_matrices[-1] = Q_matrices[-1] @ tmp_holding_var
    
    
        Q_matrix = np.zeros((matrix_rows, matrix_cols), dtype=np.float64)
        comm.Gatherv(Q_matrices[0], Q_matrix, root=0)
            
    #print("Processor ", rank, ": Finished. Dropping out.")
    
    #if(rank == 0):
        #print("Number of Q matrices in 0: ", len(Q_matrices))

    return Q_matrix, R_local




#################

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

assert isPowOfTwo(size), "The number of nodes must be a power of 2" 

if SPARSE_MATRIX_USE:
    from scipy.io import mmread
    sparse_mat = mmread("c-67b.mtx")
    matrix_rows = sparse_mat.shape[0]
    matrix_columns = 20
else:
    matrix_rows = 2**16
    matrix_columns = int(sys.argv[1])

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

comm.Scatterv(A, A_local, root=0)
start = MPI.Wtime()
globalQ,  R = tsqr(A_local, comm, matrix_rows, matrix_columns, COMPUTE_Q)
end = MPI.Wtime() - start


if rank == 0:
    if CSV_OUT:
        if COMPUTE_Q:
            A_reconstructed = globalQ @ R
            print(f"{end},{np.linalg.norm((A - A_reconstructed))},{np.linalg.cond(A)},{np.linalg.norm((np.eye(globalQ.shape[1]) - globalQ.T@globalQ))},{np.linalg.cond(globalQ)}")
        else:
            print(f"{end}")


    else:
        print("Total execution time: ", MPI.Wtime() - start)
        print("Accuracy of Factorisation: ", np.linalg.norm(A - A_reconstructed))
        print("Condition Number: ", np.linalg.cond(A))
        print("Loss of Orthogonality: ", np.linalg.norm((np.eye(globalQ.shape[1]) - globalQ.T@globalQ)))
     
