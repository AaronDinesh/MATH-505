from mpi4py import MPI
import numpy as np

com = MPI.COMM_WORLD
rank = com.Get_rank()
print("I am rank = ", rank)
