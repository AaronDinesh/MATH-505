from mpi4py import MPI
import numpy as np

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()


senddata = rank*np.ones(size, dtype=np.int32)

global_result1 = comm.allreduce(senddata, op=MPI.SUM)
global_result2 = comm.allreduce(rank, op=MPI.MAX)

print("Process ", rank, " sending ", senddata)
print("Process ", rank, "Reduction Operation 1: ", global_result1, "\nReduction Operation 2: ", global_result2)
