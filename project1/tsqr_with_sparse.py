import numpy as np
from mpi4py import MPI
from scipy.sparse import coo_matrix, csr_matrix, identity



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

    Y_local, R_local = np.linalg.qr(A_local, mode='complete')
    Q_matrices = [] 
    Q_matrices = comm.gather(Y_local, root=0)
    
    step = 1
    while step < comm.Get_size():
        partner = rank ^ step
        if partner < size:
            if rank < partner:
                #print("Processor ", rank, ": Recieving from ", partner)
                neighbor_r = np.zeros_like(R_local, dtype=np.float64)
                comm.Recv(neighbor_r, source=partner, tag=0)
                #print("Processor ", rank, ": Finished receiving.")
                Y_local, R_local = np.linalg.qr(np.vstack((R_local, neighbor_r)), mode='complete')
                
                if(rank == 0):
                    #First append the one from 0 and then append the remaining ones.
                    Q_matrices.append(Y_local)
                    
                    #The list comprehension determines the rank of the remaining processors
                    #The for loop guarantees that we receive it in order.
                    for i in [x for x in np.arange(1, size) if x % (step*2) == 0]:
                        other_Q_mat = np.empty_like(Y_local ,dtype=np.float64) 
                        #receive from the remaining matrices
                        comm.Recv(other_Q_mat, source=i, tag=1)
                        #print("Processor 0: Finished receving from ", i)
                        Q_matrices.append(other_Q_mat)
                else:
                    #Send the Y_local to 0
                    comm.Send(Y_local, dest=0, tag=1)
                    #print("Processor ", rank, ": Finished sending Q to 0")
            else:
                #print("Processor ", rank, ": Sending R to ", rank-step)
                comm.Send(R_local, dest=partner, tag=0)
                #print("Processor ", rank, ": Finished sending R")
                
                #After we finish sending we can break the while loop since we are finished computing.
                break
        
        step*=2

    #print("Processor ", rank, ": Finished. Dropping out.")
    
    #if(rank == 0):
        #print("Number of Q matrices in 0: ", len(Q_matrices))

    return Q_matrices, R_local




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
    matrix_rows = 2**13
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

start = MPI.Wtime()
comm.Scatterv(A, A_local, root=0)
Q_matrices, R = tsqr(A_local, comm, matrix_rows, matrix_columns)

if rank == 0:
    assert np.allclose(R, np.triu(R)), "R is not upper triangular"
    # Now we need to assemble the proper Q matrix. They need to be placed on the diagonals of a matrix. 
    # The number of matricies that need to be placed depends on the nodes at each level of the binary tree
    # The matrices are also ordered by their position in the binary tree.
    


    globalQ_rows = Q_matrices[0].shape[0]*size
    globalQ_cols = Q_matrices[0].shape[1]*size
    globalQ = identity(globalQ_rows, dtype=np.float64, format='csr')
    q_mat_vec_pos_offset = 0

    #from log2(size) -> 0
    for k in range(int(np.log2(size)), -1, -1):
        curr_level_node_count = 2**k
        
        curr_level_mat_data = np.array([])
        curr_level_mat_row_idx = np.array([])
        curr_level_mat_col_idx = np.array([])
        
        #Generate the required data in the COO format
        for j in range(curr_level_node_count):
            q_rows = Q_matrices[q_mat_vec_pos_offset + j].shape[0]
            q_cols = Q_matrices[q_mat_vec_pos_offset + j].shape[1]
            
            curr_level_mat_data = np.concatenate((curr_level_mat_data, Q_matrices[q_mat_vec_pos_offset + j].flatten()))
            row_idx, col_idx = np.meshgrid(np.arange(j*q_rows,j*q_rows + q_rows), np.arange(j*q_cols,j*q_cols + q_cols), indexing='ij')
            curr_level_mat_row_idx = np.concatenate((curr_level_mat_row_idx, row_idx.flatten()))
            curr_level_mat_col_idx = np.concatenate((curr_level_mat_col_idx, col_idx.flatten()))


        curr_level_q_hat = coo_matrix((curr_level_mat_data, (curr_level_mat_row_idx, curr_level_mat_col_idx)),shape=(globalQ_rows, globalQ_cols), dtype=np.float64)
        globalQ = globalQ @ curr_level_q_hat.tocsr()
        q_mat_vec_pos_offset += curr_level_node_count

    A_reconstructed = globalQ.todense() @ R

    if CSV_OUT:
        end = MPI.Wtime() - start
        print(f"{end},{np.linalg.norm((A - A_reconstructed))},{np.linalg.cond(A)},{np.linalg.norm((np.eye(globalQ.shape[1]) - globalQ.T@globalQ))}")
    else:
        print("Total execution time: ", MPI.Wtime() - start)
        print("Accuracy of Factorisation: ", np.linalg.norm(A - A_reconstructed))
        print("Condition Number: ", np.linalg.cond(A))
        print("Loss of Orthogonality: ", np.linalg.norm((np.eye(globalQ.shape[1]) - globalQ.T@globalQ)))
     
