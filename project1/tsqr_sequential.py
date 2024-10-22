from mpi4py import MPI
import numpy as np
import sys

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

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

assert isPowOfTwo(size), "The number of nodes must be a power of 2" 

matrix_rows = 2**15
matrix_columns = int(sys.argv[1])

assert matrix_rows > matrix_columns, "The matrix is not tall is skinny. Number of rows must be greater than columns"
assert matrix_rows % size == 0, "The matrix cannot be evenly row distributed"

blocks = int(matrix_rows/size)

#Even though A is only used in process 0, we need to define it in all processors to stop
#python from complaining
A = np.empty((matrix_rows, matrix_columns), dtype=np.float64)
A = gen_matrix(matrix_rows, matrix_columns)

start = MPI.Wtime()
globalQ, R = np.linalg.qr(A, mode='complete')
A_reconstructed = globalQ @  R
end = MPI.Wtime() - start

print(f"{end},{np.linalg.norm((A - A_reconstructed))},{np.linalg.cond(A)},{np.linalg.norm((np.eye(globalQ.shape[1]) - globalQ.T@globalQ))}")
