from fenics import *
import numpy as np
import matplotlib.pyplot as plt

plt.rcParams['image.cmap'] = 'jet'
fig = plt.figure()
fig.show()

# Mesh
mesh = RectangleMesh(Point(0,0),Point(2,1),8,4,diagonal='left/right')

plot(mesh)
# plt.pause(0.1)
# fig.clear()

tol = 1e-6

## Mark facets for identifying BCs
facets = MeshFunction('size_t', mesh, 1) # Defining the type of the mesh function allowed i.e. bool,size_t,int,double; given the mesh and the topology dimension (codimension = problem/total dim - geometric dimension) (2-1=1 in this case) of the mesh
facets.set_all(0) # Set values at all facets to zero

ds = Measure('ds', domain=mesh, subdomain_data=facets)

## Specify bottom face as heat source
class Heat_source(SubDomain):
    def inside(self, x, on_boundary):
        return (
            on_boundary and 
            near(x[0], 0.0, tol)
        )

heat = Heat_source()
heat.mark(facets, 1)

k0 = 267 # W/(m-K)
q0 = 6 # J
h = 50 # W/(m^2-K)
T_env = 298 # K

V = FunctionSpace(mesh,'P',1)
k = interpolate(Constant(k0),V) 
q = Expression('x[1] <= 0. + tol ? q0 : 0', degree=0, tol=tol, q0=q0)

T = TrialFunction(V)
v = TestFunction(V)

F = k*dot(grad(T), grad(v))*dx + q*v*dx + h*(T-T_env)*v*ds(0)
a, L = lhs(F), rhs(F)

T = Function(V)

solve(a==L,T)

p = plot(T)
fig.colorbar(p)
plt.show()