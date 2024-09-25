from mpi4py import MPI
import numpy as np


comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()


#For scatter we need to make use the first dim is == Size
if rank == 0:
    vector = np.random.rand(4, 2)
else:
    vector = None


#Broadcast is send the data from the root to all other nodes
data1 = comm.bcast(vector, root=0)

#Scatter means cyclically split the data between the nodes
data2 = comm.scatter(vector, root=0)
print("Rank: ", rank, " data1: ", data1, " data2: ", data2, "\n")


