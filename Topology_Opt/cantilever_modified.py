# ensity based topology optimization of cantilever

import numpy as np
from fenics import *
import matplotlib.pyplot as plt

## Global settings
plt.rcParams['image.cmap'] = 'jet'
fig = plt.figure()
fig.show()

set_log_active(False)

TOL = 1e-4

## Constants
dim = 2

Lx = 4.0
Ly = 1.0

load_width = 0.05*Ly

E_ = 1.0
nu_ = 0.33

lambda_ = E_/(2.0*(1 + nu_))
mu_     = E_*nu_/((1 + nu_)*(1 - 2*nu_))

eps_void = 1e-3

## External loads
tx = 0.0
ty = -1e-1

bx = 0.0
by = 0.0

## Build mesh
Nx = 100
Ny = 25
mesh = RectangleMesh(
    Point(0.0,0.0), Point(Lx,Ly), Nx, Ny, "crossed"
)
# plot(mesh)

## Initialize finite element spaces
Vs = FunctionSpace(mesh, 'P', 1)
V = VectorFunctionSpace(mesh, 'P', 1)

## Dummy test functions
v = TestFunction(V)
vs = TestFunction(Vs)

## Mark facets for identifying BCs
facets = MeshFunction('size_t', mesh, 1) # Defining the type of the mesh function allowed i.e. bool,size_t,int,double; given the mesh and the topology dimension (codimension = problem/total dim - geometric dimension) (2-1=1 in this case) of the mesh
facets.set_all(0) # Set values at all facets to zero

ds = Measure('ds', domain=mesh, subdomain_data=facets) # If first arg is dx then it is over domain and if ds then over boundary of the domain

## Set up Dirichlet boundary condition
class Clamped(SubDomain):
    def inside(self, x, on_boundary):
        return (
            on_boundary and 
            near(x[0], 0.0, TOL)
        )

clamped = Clamped()
clamped.mark(facets, 1) # Mark all those facets/boundaries = 1 who are clamped which we know by the clamped class, as the child class Clamped inherits from the SubDomain parent class which has the method mark. 
dbc = DirichletBC(V, Constant((0.0, 0.0)), clamped)

## Set up external load
class Load(SubDomain):
    def inside(self, x, on_boundary):
        return (
            on_boundary and
            near(x[0], Lx, TOL) and
            near(x[1], 0.5*Ly, load_width)
        ) 

load = Load()
load.mark(facets, 2)

## Thickness distribution
h0 = 0.2
h = interpolate(Constant(h0), Vs)
hfrac = assemble(h*dx(mesh))

hmin = 0.0
hmax = 1.0

## Lagrange multipliers
l0 = -0.5*h0
l1 = 1.5*h0
dl = 0.1*h0
lerr = 1e-3

## Interpolating function
def zeta(t):
    return t*t*t

def d_zeta(t):
    return 3*t*t

## Linear elastic strain
def epsilon(u):
    return (0.5*(grad(u) + grad(u).T))

## Linear elastic stress
def sigma(u, h):
    damage = (zeta(h) + (1.0 - zeta(h))*eps_void)
    return damage*(lambda_*tr(epsilon(u))*Identity(dim) + 2*mu_*epsilon(u))

## Derivative of objective function with respect to thickness
def d_obj(u, p, h):
    epsu = epsilon(u)
    epsp = epsilon(p)
    sig = lambda_*tr(epsu)*Identity(dim) + 2*mu_*epsu
    hfact = (1.0 - eps_void)*d_zeta(h)
    return hfact*inner(sig,epsp)

## Primal problem
T = Constant((tx,ty))
b = Constant((bx,by))
L = dot(b,v)*dx + dot(T,v)*ds(2)

def primal(h):
    u = TrialFunction(V)
    a = inner(sigma(u, h), epsilon(v))*dx
    u = Function(V)
    solve(a == L, u, dbc)
    return u

# ## Adjoint problem
# Ta = Constant((-tx,-ty))
# ba = Constant((-bx,-by))
# La = dot(ba,v)*dx + dot(Ta,v)*ds(2)

# def adjoint(h):
#     u = TrialFunction(V)
#     a = inner(sigma(u, h), epsilon(v))*dx
#     u = Function(V)
#     solve(a == La, u, dbc)
#     return u

## Regularization for h
alfah = 0.01
hr = TrialFunction(Vs)
ah = ((alfah**2)*dot(grad(hr),grad(vs))+ hr*vs)*dx 

def regularize_h(h):  
    L = h*vs*dx
    hr = Function(Vs)
    solve(ah == L, hr)
    return hr

## Utility functions
def max(a, b):
    return (a + b + abs(a - b))/2

def min(a, b):
    return (a + b - abs(a - b))/2

'''## Open files to record output
fobj = open("objective_fn_pgd.dat", 'w')
fh = open("volume_pgd.dat", 'w')

#### Dump volume fraction
hvol = assemble(h*dx(mesh))
fh.write('%d\t%f\n' % (0, (hvol/hfrac)))

#### Dump objective function
u = primal(h)
J = assemble(dot(b,u)*dx + dot(T,u)*ds(2))
fobj.write('%d\t%f' % (0, J))'''

## Optimization loop
dt = 1.0 #0.25
max_iter = 100
skip = 20

'''u_vtk = File('cantilever_deflection_pgd.pvd')
h_vtk = File('cantilever_pgd.pvd')'''

for iter in range(max_iter + 1):
    #### Solve primal and adjoint problems
    u = primal(h)
    # p = adjoint(h)

    #### Compute gradient of objective function
    # dJ = d_obj(u,p,h)
    dJ = -d_obj(u,u,h)

    #### Update h
    h = h - dt*dJ

    #### Enforce constraints by projection
    ###### Choose initial values of l0 and l1
    proj0 = assemble(max(hmin, min(hmax, h + l0))*dx(mesh))
    proj1 = assemble(max(hmin, min(hmax, h + l1))*dx(mesh))

    while proj0 > hfrac:
        l0 -= dl
        proj0 = assemble(max(hmin, min(hmax, h + l0))*dx(mesh))

    while proj1 < hfrac:
        l1 += dl
        proj1 = assemble(max(hmin, min(hmax, h + l1))*dx(mesh))

    ###### Bisection algorithm
    while (l1 - l0) > lerr:
        lmid = (l0 + l1)/2
        projmid = assemble(max(hmin, min(hmax, h + lmid))*dx(mesh))

        if projmid < hfrac:
            l0 = lmid
            proj0 = projmid
        else:
            l1 = lmid
            proj1 = projmid

    h = max(hmin, min(hmax, h + lmid))

    # h = max(hmin, min(hmax, h))
    h = regularize_h(h)

    #### Dump volume fraction
    hvol = assemble(h*dx(mesh))
    # fh.write('%d\t%f\n' % ((iter + 1), (hvol/hfrac))) ##########################
    # hvol = assemble(h*dx(mesh))/(Lx*Ly)
    # fh.write('%d\t%f\n' % ((iter + 1), hvol))

    #### Dump objective function
    J = assemble(dot(b,u)*dx + dot(T,u)*ds(2))
    # fobj.write('%d\t%f\n' % ((iter + 1), J)) ###########################

    print(f'Iteration {iter + 1}: {J}')

    fig.clear()
    p = plot(h)
    fig.colorbar(p)
    plt.pause(0.0001)


    '''if iter % skip == 0:
        u.rename('u','u')
        h.rename('h','h')
        u_vtk << (u, iter)
        h_vtk << (h, iter)'''
plt.show()
'''## Close files
fobj.close()
fh.close()'''