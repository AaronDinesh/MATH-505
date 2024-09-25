from mpi4py import MPI
import numpy as np


comm = MPI.COMM_WORLD
size = comm.Get_size()
rank = comm.Get_rank()

senddata = rank*np.ones(size, dtype=np.int32)
recvdata = comm.alltoall(senddata)
scatterdata = comm.scatter(senddata, root=1)
print("Process ", rank, " sending ", senddata, " receiving ", recvdata, " scatter ", scatterdata)
