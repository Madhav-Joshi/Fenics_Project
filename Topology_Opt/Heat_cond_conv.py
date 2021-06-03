from fenics import *
import numpy as np
import matplotlib.pyplot as plt

plt.rcParams['image.cmap'] = 'jet'
fig = plt.figure()
fig.show()

# Mesh
mesh = RectangleMesh(Point(0,0),Point(1,2),4,8,diagonal='left/right') # left, right, left/right, crossed

plot(mesh)
# plt.pause(0.1)
# fig.clear()

tol = 1e-6

## Mark facets for identifying BCs
facets = MeshFunction('size_t', mesh, 1) # Defining the type of the mesh function allowed i.e. bool,size_t,int,double; given the mesh and the topology dimension (codimension = problem/total dim - geometric dimension) (2-1=1 in this case) of the mesh
facets.set_all(0) # Set values at all facets to zero

ds = Measure('ds', domain=mesh, subdomain_data=facets)

## Specify left face as heat in
class Heat_in(SubDomain):
    def inside(self, x, on_boundary):
        return (
            on_boundary and 
            near(x[0], 0.0, tol)
        )

heatin = Heat_in()
heatin.mark(facets, 1)

## Specify right face as heat out
class Heat_out(SubDomain):
    def inside(self, x, on_boundary):
        return (
            on_boundary and 
            near(x[0], 1.0, tol)
        )

heatout = Heat_out()
heatout.mark(facets, 2)

## There is insulation on upper and bottom walls so Neumann BC with q/A=-k*dT/dn=0

k = 267 # W/(m-K)
h = 60 # W/(m^2-K)
T_b1 = 400 # K
T_b2 = 300 # K

V = FunctionSpace(mesh,'P',1)
# k = interpolate(Constant(k),V) 

T = TrialFunction(V)
v = TestFunction(V)

# klaplace(T) = 0 with -k*dT/dn|(x=0 or ds(1)) = -h*(T(x)-T_b1)
# and -k*dT/dn|(x=1 or ds(2)) = h*(T(x)-T_b2) 
F = k*dot(grad(T), grad(v))*dx + h*(T_b1-T)*v*ds(1) + h*(T-T_b2)*v*ds(2)
a, L = lhs(F), rhs(F)

T = Function(V)

solve(a==L,T)

p = plot(T) # ,mode='warp' --> for 3D plot
fig.colorbar(p)
plt.show()