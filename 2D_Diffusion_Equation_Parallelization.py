import numpy as np
import matplotlib.pyplot as plt
from mpi4py import MPI

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()
# plate size, mm
w = h = 20.48 #heat is diffusing on a square plate

# grid size
nx = ny = n = int(2**10)

#setting up the sizes for each process
chunk_x = chunk_y = chunk = int(n / size)

# intervals in x-, y- directions, mm
dx, dy = w / (nx - 1), h / (ny - 1) #step size in each direction
dx2, dy2 = dx * dx, dy * dy

#must setup a partial x vector and a partial y vector to create a partial temp array. Both of these partial vectors are equal since chunk is the same and dx and dy are the same
partial_nx = dx * np.arange(nx) #rows have all info and split only accross y   
partial_ny = dy * np.arange(rank*chunk, (rank+1)*chunk) #creating the split array, split along y 


# Thermal diffusivity of steel, mm2/s
D = 4.2 #thermal diffusivity of the material in questions

#printing grid and plate size and diffusivity. Same for each proccess.
if rank == 0:
    print("grid size:", nx, "x", ny)
    print("plate width:", w, "plate height", h, "mm")
    print("thermal diffusivity:", D)

# time
nsteps = 101
dt = dx2 * dy2 / (2 * D * (dx2 + dy2))  # Using CFL to calculate the largest dt (Courant-Friedrichs-Lewy Stability Condition)
F = D * dt
print("dt:", dt)

# plot
plot_ts = [0, 10, 25, 50, 100]

# array
partial_Tarray = np.zeros((chunk+2, n))  # (rows, columns) => need to add a buffer

# Initialization - circle of radius r centred at (cx,cy) (mm)
Tcool, Thot = 300, 2000
r, cx, cy = 5.12, w / 2, h / 2
r2 = r**2
for j in range(chunk): #initializing
    for i in range(nx):
        x = i*dx #getting the x and y position, required for initialization (setting up grid points at time t = 0)
        y = (rank*chunk + j)*dy #required to do this to line up the y position with the correct x poisiton.
        p2 = (x - cx) ** 2 + (y - cy) ** 2 #checking distance from the center of the circle
        if p2 < r2: #comparing it with the radius and if within the radius the scaled as such below and if it is not then outside so temp is cool
            radius = np.sqrt(p2)
            partial_Tarray[j+1, i] = Thot * np.cos(4 * radius) ** 4
        else:
            partial_Tarray[j+1, i] = Tcool


for m in range(nsteps):  
    #non blocking + buffer => definitely the fastest
#Parallelization code    
    if rank != 0: #if rank is not first it means there is a rank behind
        recieved_data_head = np.empty(nx) #provide and existing empty array for which MPI will right data too.
        sendReq_head = comm.Isend(partial_Tarray[1, :], dest = rank-1, tag=5)
        recvReq_head = comm.Irecv(recieved_data_head, source = rank-1, tag=6) 
        
        sendReq_head.wait()
        recvReq_head.wait()
        partial_Tarray[0, :] = recieved_data_head
    
    if rank != size - 1: #if rank is not end it means there is a rank infront
        recieved_data_tail = np.empty(nx)
        sendReq_tail = comm.Isend(partial_Tarray[-2, :], dest = rank+1, tag=6)
        recvReq_tail = comm.Irecv(recieved_data_tail, source = rank+1, tag=5)
            
        sendReq_tail.wait()
        recvReq_tail.wait()
        partial_Tarray[-1, :] = recieved_data_tail
    
    
    #non blocking => This version works!!
    #if rank != 0:
        #sendReq_head = comm.isend(partial_Tarray[1, :], dest = rank-1, tag=5)
        #recvReq_head = comm.irecv(source = rank-1, tag=6) 
        
        #sendReq_head.wait()
        #partial_Tarray[0, :] = recvReq_head.wait()
    

    #if rank != size - 1:
        #sendReq_tail = comm.isend(partial_Tarray[-2, :], dest = rank+1, tag=6)
        #recvReq_tail = comm.irecv(source = rank+1, tag=5)
            
        #sendReq_tail.wait()
        #partial_Tarray[-1, :] = recvReq_tail.wait()
        
    #blocking 
    # if rank != 0:
    #     comm.send(partial_Tarray[1, :], dest = rank-1, tag=5)
    #     partial_Tarray[0, :] = comm.recv(source = rank-1, tag=6) 
    # if rank != size - 1:
    #     comm.send(partial_Tarray[-2, :], dest = rank+1, tag=6)
    #     partial_Tarray[-1, :] = comm.recv(source = rank+1, tag=5)
    
    #finite differencing
            
    partial_Tarray[1:-1, 1:-1] = partial_Tarray[1:-1, 1:-1] + F * (
        (partial_Tarray[2:, 1:-1] - 2 * partial_Tarray[1:-1, 1:-1] + partial_Tarray[:-2, 1:-1]) / dy2
        + (partial_Tarray[1:-1, 2:] - 2 * partial_Tarray[1:-1, 1:-1] + partial_Tarray[1:-1, :-2]) / dx2
    ) #2D diffusion equation from finite differencing. Dirichlet (implicit) boundary conditions are imposed though initial conditions by not updating the first and last value!

    
    if m in plot_ts: #this sendbuf is required due to the contiguous array error!
        sendbuf = np.empty((chunk, n)) #the size of the buff is chunk(rows) and n(columns) 
        sendbuf[:] = partial_Tarray[1:-1, :] #removing ghost rows as that data belongs to neighbouring processes

        if rank == 0:
            Tarray = np.empty((ny, nx))
        else:
            Tarray = None

        comm.Gather(sendbuf, Tarray, root=0) #if at the root process we are gathering the data from each process and putting into Tarray for plotting!!
        if rank == 0:
            fig = plt.figure(1)
            im = plt.imshow(Tarray, cmap=plt.get_cmap("hot"), vmin=Tcool, vmax=Thot)
            plt.title("{:.1f} ms".format((m + 1) * dt * 1000))
            cbar_ax = fig.add_axes([0.9, 0.15, 0.03, 0.7])
            cbar_ax.set_xlabel("K", labelpad=20)
            fig.colorbar(im, cax=cbar_ax)
            plt.savefig("iter_{}.png".format(m), dpi=200)
            plt.clf()
            
# time mpirun -np 4 --oversubscribe python parallel_2d_diffusion_equation1.py

#if __name__ == "__main__":
    