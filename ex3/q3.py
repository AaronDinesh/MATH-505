from mpi4py import MPI
import numpy as np

def mat_vec_mul(A, x):
    return A@x

comm = MPI.COMM_WORLD
size = comm.Get_size()
rank = comm.Get_rank()

wt = MPI.Wtime()

cols = 4
rows = 8
#How many rows of Matrix A should go to each process
blocks = int(rows/size)

A = None
A_partial = np.empty((blocks, cols), dtype='int')
x = None
result = np.empty((rows, 1), dtype='int')



#Let rank 0 be the initalizer
if rank == 0:
    A = np.array([[1, 2, 3, 4],
                  [5, 6, 7, 8],
                  [9, 10, 11, 12],
                  [13, 14, 15, 16],
                  [17, 18, 19, 20],
                  [21, 22, 23, 24,],
                  [25, 26, 27, 28],
                  [29, 30, 31, 32]])
    x = np.array([7, 8, 9, 10])

comm.Scatterv(A, A_partial ,root=0)

x = comm.bcast(x, root=0)
if rank == 1:
    print(x)
partial_result = mat_vec_mul(A_partial, x)
comm.Gatherv(partial_result, result, root=0)


if rank == 0:
    time_taken = MPI.Wtime() - wt
    print("Result: ", result, "\nTime Taken: ",wt)
