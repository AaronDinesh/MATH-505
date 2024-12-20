from mpi4py import MPI
import numpy as np

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

def gen_matrix(rows, columns):
    A = np.zeros((rows, columns), dtype=np.float64)
    
    for i in range(rows):
        for j in range(columns):
            A[i, j] = f((i-1)/(rows-1), (j-1)/(columns-1))

    return A


def parallel_cgs(A_local, comm, matrix_rows, matrix_cols):
    R = np.zeros((matrix_cols, matrix_cols), dtype=np.float64)
    local_Q = np.zeros((A_local.shape[0], matrix_cols), dtype=np.float64)
    beta_local = np.power(np.linalg.norm(A_local[:, 0]), 2)


    #This all reduce will calculate the norm-squared of the first columns
    beta = comm.allreduce(beta_local)
    R[0,0] = np.sqrt(beta)
    
    local_Q[:,0] = A_local[:, 0] / R[0, 0]

    for j in range(1, matrix_cols):
        r_local = local_Q[:, 0:j-1].T @ A_local[:, j]
        r = comm.allreduce(r_local)
        
        R[0:j-1, j] = r

        local_Q[:, j] = A_local[:, j] - local_Q[:, 0:j-1] @ R[0:j-1, j]
        beta_local = np.power(np.linalg.norm(local_Q[:, j]), 2)
        beta = comm.allreduce(beta_local)
        R[j, j] = np.sqrt(beta)
        local_Q[:, j] = local_Q[:, j] / R[j, j]
        
        #tst = np.empty((matrix_rows, j), dtype=np.float64)
        #comm.Gatherv(np.ascontiguousarray(local_Q[:, :j]), tst, root=0)
        #if comm.rank == 0:
            #print(f"[{j}] Loss of Orthogonality: ", np.linalg.norm((np.eye(local_Q.shape[1]) - local_Q.T@local_Q)), "Condition Number: ", np.linalg.cond(local_Q))   
            #print(f"{np.linalg.norm((np.eye(tst.shape[1]) - tst.T@tst))},{np.linalg.cond(tst)}")   
    return local_Q, R

#################

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

if SPARSE_MATRIX_USE:
    from scipy.io import mmread
    sparse_mat = mmread("c-67b.mtx")
    matrix_rows = sparse_mat.shape[0]
    matrix_columns = 190
else:
    matrix_rows = 2**16
    matrix_columns = int(sys.argv[1])

assert matrix_rows % size == 0, "The matrix cannot be evenly row distributed"

blocks = int(matrix_rows/size)

#Even though A is only used in process 0, we need to define it in all processors to stop
#python from complaining
A = np.empty((matrix_rows, matrix_columns), dtype=np.float64)
A_local = np.empty((blocks, matrix_columns), dtype=np.float64)
Q = np.empty((matrix_rows, matrix_columns), dtype=np.float64) 

if rank == 0:
    # Machine precision for double is 10^-16
    if SPARSE_MATRIX_USE:
        A = np.delete(np.delete(sparse_mat.todense(), np.arange(matrix_columns, sparse_mat.shape[1]), 1), np.arange(matrix_rows, sparse_mat.shape[0]), 0)
    else:
        A = gen_matrix(matrix_rows, matrix_columns)

comm.Scatterv(A, A_local, root=0)

start = MPI.Wtime()
Q_local, R = parallel_cgs(A_local, comm, matrix_rows, matrix_columns)
end = MPI.Wtime() - start
comm.Gatherv(Q_local, Q, root=0)


if rank == 0:
    if CSV_OUT:
        print(f"{end},{np.linalg.norm((A - Q@R))},{np.linalg.cond(A)},{np.linalg.norm((np.eye(Q.shape[1]) - Q.T@Q))},{np.linalg.cond(Q)}")
    else:
        print("Time Taken: ", end)
        print("Accuracy of Factorisation: ", np.linalg.norm((A - Q@R)))
        print("Condition Number of A: ", np.linalg.cond(A))
        print("Loss of Orthogonality: ", np.linalg.norm((np.eye(Q.shape[1]) - Q.T@Q)))
