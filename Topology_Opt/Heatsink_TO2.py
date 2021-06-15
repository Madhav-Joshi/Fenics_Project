# Maximizing the heat flux through the surface

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

tol = 1e-6 # for using near function of FEniCS

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
km = 267 # W/(m-K) Thermal conductivity of metal
ka = 0.03 # W/(m-K) Thermal conductivity of air
q_flux = 1000 # J
h = 50 # W/(m^2-K)
T_env = 298 # K

V = FunctionSpace(mesh,'P',1)

c0 = 0.2 # characteristic function 
c = interpolate(Constant(c0),V) # k = c*km+(1-c)*ka
cfrac = assemble(c*dx(mesh))

## Interpolating function
def zeta(t):
    return t*t*t

def d_zeta(t):
    return 3*t*t

## Derivative of objective function with respect to charac fun
def d_obj(T, p, c):
    return d_zeta(c)*(km-ka)*inner(grad(T),grad(p))

# Primal problem
v = TestFunction(V)
L = q_flux*v*ds(1) + h*T_env*v*ds(0)

def primal(c):
    # -div(k*grad(T)) = 0 with -k*dT/dn|(y=0 or ds(1)) = q_flux
    # and -k*dT/dn|(ds(0)) = h*(T(x)-T_env)
    T = TrialFunction(V)
    a = (zeta(c)*(km-ka)+ka)*dot(grad(T), grad(v))*dx + h*T*v*ds(0)
    T = Function(V)
    solve(a==L,T)
    return T

# Adjoint Problem
La = -h*v*ds(0) #(-v/(assemble(interpolate(Constant(1),V)*dx(mesh))))*dx 

def adjoint(c):
    p = TrialFunction(V)
    a = (zeta(c)*(km-ka)+ka)*dot(grad(p),grad(v))*dx + h*p*v*ds(0)
    p = Function(V)
    solve(a==La,p)
    return p

## Regularization for c
alfah = 0.06 ############is this value fine############ Doubt
cr = TrialFunction(V) # Charac func c's regularization r
ac = ((alfah**2)*dot(grad(cr),grad(v))+ cr*v)*dx # LHS a for charac func c

def regularize_c(c):  
    L = c*v*dx
    cr = Function(V)
    solve(ac == L, cr)
    return cr

## Utility functions
def Max(a, b):
    return (a + b + abs(a - b))/2

def Min(a, b):
    return (a + b - abs(a - b))/2

## Bisection algorithm variables
cmin = 0
cmax = 1
#### Lagrange multipliers
l0 = -0.5*c0
l1 = 1.5*c0
dl = 0.5*c0
lerr = 1e-3


## Optimization loop
dt = 2.0 #0.25
max_iter = 1

for iter in range(max_iter):
    #### Solve primal and adjoint problems
    T = primal(c)
    p = adjoint(c)

    #### Compute gradient of objective function
    dJ = d_obj(T,p,c)

    #### Update c
    c = c + dt*dJ ###################### + because MAXIMIZE heat flux

    #### Enforce volume and Max,Min constraints by projection
    ###### Choose initial values of l0 and l1 for Bisection algorithm
    proj0 = assemble(Max(cmin, Min(cmax, c + l0))*dx(mesh))
    proj1 = assemble(Max(cmin, Min(cmax, c + l1))*dx(mesh))

    while proj0 > cfrac:
        l0 -= dl
        proj0 = assemble(Max(cmin, Min(cmax, c + l0))*dx(mesh))

    while proj1 < cfrac:
        l1 += dl
        proj1 = assemble(Max(cmin, Min(cmax, c + l1))*dx(mesh))

    ###### Bisection algorithm
    while (l1 - l0) > lerr:
        lmid = (l0 + l1)/2
        projmid = assemble(Max(cmin, Min(cmax, c + lmid))*dx(mesh))

        if projmid < cfrac:
            l0 = lmid
            proj0 = projmid
        else:
            l1 = lmid
            proj1 = projmid

    c = Max(cmin, Min(cmax, c + lmid))
    # c = Max(cmin, Min(cmax, c))
    # c = project(c,V)
    c = regularize_c(c)

    #### Dump volume fraction
    # cvol = assemble(c*dx(mesh))

    #### Dump objective function
    J = assemble(h*(T-T_env)*ds(0)) # assemble(T*dx(mesh))/assemble(interpolate(Constant(1),V)*dx(mesh))

    print(f'Iteration {iter + 1}: {J}')

    fig.clear()
    p = plot(c)
    fig.colorbar(p)
    plt.pause(1)

    c_arr = np.array(c.vector())
    c_max_value = np.max(c_arr)
    c_min_value = np.min(c_arr)

plt.show()