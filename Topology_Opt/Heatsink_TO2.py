# Minimizing the mean temperature in part of the metal

############################################################################
## Structure of the code
##1 Setting plotting environment
############################################################################
##2 Problem Variables
##      Bisection algorithm variables
##         Lagrange multipliers
##      Optimization loop variables
############################################################################
##3 Mesh 
##      Mark domain over which avg Temp will be minimized
##      Mark facets for identifying BCs
##      Specify bottom face as heat source
##      Define function space and calculate volume fraction
############################################################################
##4 Functions 
##      Interpolating function
##      Derivative of objective function with respect to charac fun
##      Primal problem
##      Adjoint Problem
##      Regularization for c
##      Utility functions
############################################################################
##5 Optimization Loop
##      Solve primal and adjoint problems
##      Compute gradient of objective function
##      Update c
##      Enforce volume and max,min constraints by projection
##          Choose initial values of l0 and l1 for Bisection algorithm
##          Bisection algorithm to find lmid such that cfrac is satisfied
##      Regularize
##      Calculate volume fraction
##      Calculate objective function
##      Plot Mollifier and Objective function
############################################################################



from fenics import *
import numpy as np
import matplotlib.pyplot as plt

############################################################################

#1 Setting plotting environment
plt.rcParams['image.cmap'] = 'jet'
fig = plt.figure()
ax2 = fig.add_subplot(122)
ax2.set(title = 'Objective Func vs Iteration number',xlabel='Iteration number')
fig.show()
set_log_active(False)

############################################################################

#2 Problem Variables
km = 267 # W/(m-K) Thermal conductivity of metal
ka = 0.03 # W/(m-K) Thermal conductivity of air
q_flux = 500 # J
h = 50 # W/(m^2-K)
T_env = 298 # K

c0 = 0.2 # characteristic function 
tol = 1e-6 # for using near function of FEniCS

## Bisection algorithm variables
cmin = 0
cmax = 1
#### Lagrange multipliers
l0 = -2*c0
l1 = 2*c0
dl = 3*c0
lerr = 1e-3

## Optimization loop variables
dt = 0.01
max_iter = 50

############################################################################

#3 Mesh
msh = RectangleMesh(Point(0,0),Point(1,2),25,50,diagonal='crossed')

## Mark domain over which avg Temp will be minimized
cell = MeshFunction('size_t',msh,0)
cell.set_all(0)

dx = Measure('dx', domain=msh, subdomain_data=cell)

class MinTemp(SubDomain):
    def inside(self,x,on_boundary):
        return (
            x[0]>0.4 and 
            x[0]<0.6 and 
            x[1]<0.1
        )
    
    # For center uncomment the bottom inside function and comment out the above one
    '''
    def inside(self,x,on_boundary):
        return (
            x[0]>0.4 and 
            x[0]<0.6 and 
            x[1]>0.9 and
            x[1]<1.1
        )
    '''

minT_mesh = MinTemp()
minT_mesh.mark(cell,1)

## Mark facets for identifying BCs
facets = MeshFunction('size_t', msh, 1) # Defining the type of the msh function allowed i.e. bool,size_t,int,double; given the msh and the topology dimension (codimension = problem/total dim - geometric dimension) (2-1=1 in this case) of the msh
facets.set_all(0) # Set values at all facets to zero

ds = Measure('ds', domain=msh, subdomain_data=facets)

class Heat_source(SubDomain):
    def inside(self, x, on_boundary):
        return (
            on_boundary and 
            near(x[1], 0.0, tol)
        )

heat = Heat_source()
heat.mark(facets, 1)

## Define function space and calculate volume fraction
V = FunctionSpace(msh,'P',1)
c0 = Constant(c0)
c = interpolate(c0,V) # k = c*km+(1-c)*ka
cfrac = assemble(c*dx(msh))
print(f'Volume fraction cfrac = {cfrac}')

############################################################################

#4 Functions
## Interpolating function
def zeta(t):
    return t*t*t

def d_zeta(t):
    return 3*t*t

## Derivative of objective function with respect to charac fun
def d_obj(T, p, c):
    return d_zeta(c)*(km-ka)*dot(grad(T),grad(p))


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
La = (-v/assemble(interpolate(Constant(1),V)*dx(1)))*dx(1)

def adjoint(c):
    p = TrialFunction(V)
    a = (zeta(c)*(km-ka)+ka)*dot(grad(p),grad(v))*dx(msh) + h*p*v*ds(0)
    p = Function(V)
    solve(a==La,p)
    return p


## Regularization for c
alfah = 0.03 
cr = TrialFunction(V) # Charac func c's regularization r
ac = ((alfah**2)*dot(grad(cr),grad(v))+ cr*v)*dx # LHS a for charac func c

def regularize_c(c):  
    L = c*v*dx
    cr = Function(V)
    solve(ac == L, cr)
    return cr

## Utility functions
def max(a, b):
    return (a + b + abs(a - b))/2

def min(a, b):
    return (a + b - abs(a - b))/2

############################################################################

#5 Optimization Loop
for iter in range(max_iter):
    print(f'Iteration {iter + 1}') 
    #### Solve primal and adjoint problems
    T = primal(c)
    p = adjoint(c)

    #### Compute gradient of objective function
    dJ = d_obj(T,p,c)
    dJ_print = np.array(project(dJ,V).vector())
    print('Max and min values of dJ')
    print(np.max(dJ_print),np.min(dJ_print))

    #### Update c
    c = c - dt*dJ # dJ/np.max(dJ_print) 

    #### Enforce volume and max,min constraints by projection
    ###### Choose initial values of l0 and l1 for Bisection algorithm
    proj0 = assemble(max(cmin, min(cmax, c + l0))*dx(msh))
    proj1 = assemble(max(cmin, min(cmax, c + l1))*dx(msh))

    while proj0 > cfrac:
        l0 -= dl
        proj0 = assemble(max(cmin, min(cmax, c + l0))*dx(msh))

    while proj1 < cfrac:
        l1 += dl
        proj1 = assemble(max(cmin, min(cmax, c + l1))*dx(msh))

    ###### Bisection algorithm to fund lmid such that cfrac is satisfied
    while (l1 - l0) > lerr:
        lmid = (l0 + l1)/2
        projmid = assemble(max(cmin, min(cmax, c + lmid))*dx(msh))

        if projmid < cfrac:
            l0 = lmid
            proj0 = projmid
        else:
            l1 = lmid
            proj1 = projmid

    c = max(cmin, min(cmax, c + lmid))
    cprint = np.array(project(c,V).vector())
    print('Max and min values of characteristic function are')
    print(np.max(cprint),np.min(cprint))
    cvol = assemble(c*dx(msh))
    print(f'cvol_pre_regularization = {cvol}')

    # c = max(cmin, min(cmax, c))
    c = regularize_c(c)

    #### Dump volume fraction
    cvol = assemble(c*dx(msh))
    print(f'cvol_post_regularization = {cvol}')

    #### Dump objective function
    J = assemble(T*dx(1))/assemble(interpolate(Constant(1),V)*dx(1))
    print(f'Objective function: {J}')

    if iter!=0: 
        ax1.remove()
        cb.remove()

    ax1 = fig.add_subplot(121)
    ax1.set(title='Mollifier Plot')
    pl = plot(c)
    cb = fig.colorbar(pl)
    ax2.scatter(iter+1,J)
    plt.pause(0.0001)

    '''#### Save solution
    output_file = HDF5File(msh.mpi_comm(), f"./Topology_Opt/Output_Files/charfunc_{iter}.h5", "w")
    output_file.write(c, "solution")
    output_file.close()'''

    print()

plt.show()

############################################################################