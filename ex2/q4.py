#Writing a matrix multiply with mpi

from mpi4py import MPI
import numpy as np


comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

if rank == 0:
    result = None
    N = 4
    A = np.array([[10, 10, 10, 10],
                 [20, 20, 20, 20],
                 [30, 30, 30, 30],
                 [40, 40, 40, 40]])
    v = np.array([1,2,3,4])
    assert A.shape[1] == v.shape[0]
else:
    v = None
    A = None


v = comm.bcast(v, root=0)
A = comm.scatter(A, root=0)

#print("Rank: ", rank, " v: ", v, " A: ", A, "\n")

prod = A @ v
result = comm.gather(prod, root=0)

if rank == 0:
    print(result)





    

