from mpi4py import MPI
import numpy as np

b = np.array([1, 2, 3, 4])
c = np.array([5, 6, 7, 8])
a = np.zeros_like(b)
d = np.zeros_like(b)

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

print("I am rank: ", rank)


#The elif is added to protect against starting the script
#with more than 2 processes
if(rank == 0):
    for i in range(4):
        a[i] = b[i] + c[i]

    #Once this computation is done, send it
    #process 1 to continue computation
    comm.Send(a, dest=1, tag=77)
elif (rank == 1):
    #All other processes should run this code.
    #Since we only have one other processes only
    #processor with rank 1 will run this
    
    #We need to receive the data from 0
    comm.Recv(a, source=0, tag=77)
    #We can receive in a since in the context of
    #process 1, a is not populated. We also explicitly
    #filter for the tag id of 77
    for i in range(4):
        d[i] = a[i] + b[i]
    
    #When the computation is done, we want this process
    #to print out the result
    print("d: ", d)
else:
    print("I am rank: ", rank, ". I have no work to do.")
    pass

