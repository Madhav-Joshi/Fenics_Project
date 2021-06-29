from fenics import *
import matplotlib.pyplot as plt

plt.rcParams['image.cmap'] = 'jet'
fig = plt.figure()
fig.show()
set_log_active(False)

mesh = RectangleMesh(Point(0,0),Point(1,2),25,50,diagonal='crossed')
V = FunctionSpace(mesh,'P',1)
c = Function(V)

# Load and plot solution
for i in range(50):
    try:
        input_file = HDF5File(mesh.mpi_comm(), f"./Topology_Opt/Output_Files/charfunc_{i}.h5", "r")
        input_file.read(c, "solution")
        input_file.close()
        fig.clear()
        p = plot(c)
        fig.colorbar(p)
        plt.pause(0.1)
    except Exception as e:
        break

plt.show()