import numpy as np
from mpi4py import MPI

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

START = 0
END = 3
NUM_INTERVALS = 100000 * size

if rank == 0:
    print("Num_Intervals: ", NUM_INTERVALS)

def f(x):
    return np.power(x, 2)

def midpoint_integrator(start, end, num_intervals=500):
    h = (end - start) / num_intervals
    x_domain = np.arange(start + (h/2) + h*rank, end - h/2, h*size)
    f_partial_integral = np.sum(f(x_domain)*h)
    return f_partial_integral


partial_result = midpoint_integrator(START, END, NUM_INTERVALS)

result = comm.reduce(partial_result, root=0)

if rank == 0:
    print('{:20.15f}'.format(result))
    

