# Heat conduction problem

import matplotlib.pyplot as plt
import numpy as np
from fenics import *

## Global settings
#plt.ion()
#plt.show()
plt.rcParams['image.cmap'] = 'jet'
'''fig = plt.figure()
fig.show()'''

set_log_active(False)

# Variables
tf = 3           # final time in seconds
num_steps = 100     # number of time steps
dt = tf / num_steps # time step size
alpha = 18.7 # Thermal diffusivity mm^2/s
k = 63.9e3 # Thermal conductivity W/(mm-K)
vel = 8.33 # Velocity in mm/sec
qmax = 113.35e6 # Max power per unit area supplied to the material, Power of laser = integral of q over laser circle
r_b = 2 # Radius of the beam in mm where intensity becomes 1/e 
sigma_z = 1e-5 # Penetration depth in z direction in mm
T0 = Constant(25) # Ambient temperature in Kelvin or deg C anything will work

tol = 1e-6

# Define mesh 
L = 25.0 # Length of cube/cuboid
n = 10 # Number of cells in x,y,z direction

x0,y0,z0 = -L/4,-L/2,0
x1,y1,z1 = 3*L/4,L/2,-L

mesh = BoxMesh(Point(x0,y0,z0),Point(x1,y1,z1),int(n*abs(x1-x0)/L),int(n*abs(y1-y0)/L),int(n*abs(z1-z0)/L))

# Refine region of mesh
cell = MeshFunction('bool',mesh,0)
cell.set_all(False)

w = 2 # Refinement width as a multiple of L

class Ref(SubDomain):
    def inside(self, x, on_boundary):
        return abs(x[1])<w*r_b and x[2]>-w*r_b 

ref = Ref()
ref.mark(cell, True) 

mesh = refine(mesh,cell)
mesh = refine(mesh,cell)
#mesh = refine(mesh,cell)

'''mshco = mesh.coordinates()
x = mshco[:,0]
y = mshco[:,1]
z = mshco[:,2]
print(x)
print(y)
print(z)

fig = plt.figure()
ax = fig.add_subplot(111,projection='3d')
ax.plot_trisurf(x,y,z)'''
print(mesh.hmax(),mesh.hmin())
plot(mesh)
plt.show()