from fenics import *
import numpy as np
import matplotlib.pyplot as plt

plt.rcParams['image.cmap'] = 'jet'
fig = plt.figure()
fig.show()

# Mesh
mesh = RectangleMesh(Point(0,0),Point(1,2),8,16,diagonal='left/right')

# plot(mesh)
# plt.pause(0.1)
# fig.clear()

tol = 1e-6

## Mark facets for identifying BCs
facets = MeshFunction('size_t', mesh, 1) # Defining the type of the mesh function allowed i.e. bool,size_t,int,double; given the mesh and the topology dimension (codimension = problem/total dim - geometric dimension) (2-1=1 in this case) of the mesh
facets.set_all(0) # Set values at all facets to zero

ds = Measure('ds', domain=mesh, subdomain_data=facets)

# Mark boundary condition domain
## Specify bottom face as heat source
class Heat_source(SubDomain):
    def inside(self, x, on_boundary):
        return (
            on_boundary and 
            near(x[1], 0.0, tol)
        )

heat = Heat_source()
heat.mark(facets, 1)

# Problem Variables
k = 267 # W/(m-K)
q_flux = 1000 # J
h = 50 # W/(m^2-K)
T_env = 298 # K

V = FunctionSpace(mesh,'P',1)

# Primal problem
v = TestFunction(V)

# -div(k*grad(T)) = 0 with -k*dT/dn|(y=0 or ds(1)) = q_flux
# and -k*dT/dn|(ds(0)) = h*(T(x)-T_env)
T = TrialFunction(V)
F = k*dot(grad(T), grad(v))*dx - q_flux*v*ds(1) + h*(T-T_env)*v*ds(0)
a, L = lhs(F), rhs(F)
#a = k*dot(grad(T), grad(v))*dx + h*T*v*ds(0)
#L = q_flux*v*ds(1) + h*T_env*v*ds(0)
T = Function(V)
solve(a==L,T)

# p = plot(T)
p = plot(grad(T))
fig.colorbar(p)
plt.show()