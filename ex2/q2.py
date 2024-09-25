from mpi4py import MPI
import numpy as np

comm = MPI.COMM_WORLD
rank = comm.Get_rank()

if rank == 0:
    data = {'a': 7, 'b': 3.14}
    print("From Process: ", rank, "\n data sent: ", data, "\n")
    comm.isend(data, dest=1, tag=11)
elif rank == 1:
    #irecv returns a request handeler that we need to wait on
    recv_handle = comm.irecv(source=0, tag=11)
    data = recv_handle.wait()
    print("From Process: ", rank, "\n data received: ", data, "\n")
elif rank == 2:
    data = np.ones((5,))
    comm.isend(data, dest=3, tag=66)
    print("From Process: ", rank, "\n data sent: ", data, "\n")
else:
    recv_handle = comm.irecv(source=2, tag=66)
    data = recv_handle.wait()
    print("From Process: ", rank, "\n data received: ", data, "\n")
