import numpy as np
import matplotlib.pyplot as plt
from mpi4py import MPI

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

# general
length = 1  # m, rod length
nx = 256  # number of points => these points need to be distributed between 0 and 1 with the spacing dx. Done partially for each process to run certain spatial points, All seperated by dx

# parallel params
chunk = int(nx / size) #creating the number of chunk sizes 

# space
dx = length / (nx - 1)  # delta x
partial_x = dx * np.arange(rank * chunk, (rank + 1) * chunk) #creating partial points with the spacing dx

if rank == 0:
    full_x = dx * np.arange(nx) #full list of points seperated by dx from 0-1

# coefficient
alpha = 2.3e-4  # m^2/s, aluminum diffusion coefficient

# time
t_final = 60  # s, length of simulation
dt = 0.01  # delta time

F = alpha * dt / dx**2

# snapshots
snap = [0, 5, 10, 20, 30, 60]  # s, seconds at which to take a snapshot

# initial function
func = lambda x: 20 + (30 * np.exp(-100 * (-0.5 + x) ** 2))

# init temperature vector. +2 to have a buffer for communication
partial_T = np.empty([chunk + 2]) #adding the buffers for communication which is why the plus two
# Fill with 20 for an implicit boundary condition.
partial_T.fill(20) #fill the empty array with the implicit boundary conditions
for i, x_val in enumerate(partial_x): #include the index and the value from partial_x
    # i+1 to avoid replacing the first and last values because they are buffers/boundary conditions
    partial_T[i + 1] = func(x_val) #the function is not applied to those points so the boundary condition is roughly the same. Since the first and last index will never be reached

# Courant Friedrichs Lewy condition to assure that time steps are small enough
if rank == 0:
    if F >= 0.5:
        raise Exception( #if F is too great this exception is raised because it doesn't pass the Courant Friedrichs Lewy condition
            "CFL condition failed. CFL must be less than 0.5. Current value: {}".format(
                F
            )
        )

# Main program. This is where I run the appropriate communication to solve the diffusion equation in parallel
for j in np.arange(0, t_final+dt, dt): #must include the plus dt here to get the snap shot at 60 
    # communication
    if rank != 0: #if rank is not zero means there is a rank behind
        comm.send(partial_T[1], dest=rank - 1, tag=11)
        partial_T[0] = comm.recv(source=rank - 1, tag=12)

    if rank != size - 1: #if rank is not end it means theres a rank infront
        comm.send(partial_T[-2], dest=rank + 1, tag=12)
        partial_T[-1] = comm.recv(source=rank + 1, tag=11)

    # calculation
    partial_T[1:-1] = partial_T[1:-1] + F * (
        partial_T[0:-2] - 2 * partial_T[1:-1] + partial_T[2:]
    )

    # plotting
    if j in snap:

        full_T = None
        if rank == 0:
            full_T = np.empty(nx)
        comm.Gather(partial_T[1:-1], full_T, root=0)

        if rank == 0:
            #plt.figure()
            plt.plot(full_x, full_T)
            plt.axis([0, length, 20, 50])
            plt.xlabel("Rod length (m)")
            plt.ylabel("Temperature (C)")
            plt.savefig("snapshot{}.png".format(j), dpi=200)
            plt.clf()
