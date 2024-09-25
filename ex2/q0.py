from mpi4py import MPI


com = MPI.COMM_WORLD
rank = com.Get_rank()
size = com.Get_size()

if(rank == 0):
    print("The number of processors used: ", size)

print("Rnak: ", rank)


