# Solving for menbrane thickness
# Minimizing J(h) = int2d_D (fu) dx

from fenics import *
import numpy as np 
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d

# For showing plots continuously
plt.ion()
fig = plt.figure()
ax1 = fig.add_subplot(2,2,2,projection = '3d')
ax2 = fig.add_subplot(2,2,1)
ax2.set_title('Objective Function Plot')
ax2.set_xlabel('Iteration number')
ax2.set_ylabel('J(h)')
ax3 = fig.add_subplot(2,2,3,projection = '3d')
ax3.set(xlabel='X',ylabel='Y',zlabel='Load (f)')
ax4 = fig.add_subplot(2,2,4,projection = '3d')
plt.show()

# Create mesh and function space
mesh = Mesh('fenics_trial/Square.xml')
mesh = refine(mesh)
mesh = refine(mesh)
mesh = refine(mesh)
#mesh = refine(mesh)

mshco = mesh.coordinates()
mshcox = mshco[:,0]
mshcoy = mshco[:,1]

V = FunctionSpace(mesh, 'P', 1)

# Boundary condition for displacement
def boundary(x, on_boundary):
    return on_boundary
bc = DirichletBC(V, 0, boundary)

# Define variables
u = TrialFunction(V) # Displacement
p = TrialFunction(V) 
v = TestFunction(V) 
h0 = 0.25 # Thickness
h = Constant(h0)
h = interpolate(h,V)
hmin = 0.1
hmax = 1
l0 = 0 # Bisecetion starting guess
l1 = 1 # Bisection starting guess 
lerr = 1e-3
hfrac = assemble(h0*dx(mesh))
F = 5
r =0.1
xc = 1.5
yc = 1

f = Expression('F*exp(-((pow(x[0]-xc, 2) + pow(x[1] - yc, 2))/(2*pow(r,2))))', degree=1, F=F, xc=xc, yc=yc, r=r)
f_int = interpolate(f,V)
f_mesh = f_int.compute_vertex_values(mesh)
# plot(f_int)
ax3.plot_trisurf(mshcox,mshcoy,f_mesh,cmap = plt.cm.jet)

def Max(a,b):
    return (a+b+abs(a-b))/2

def Min(a,b):
    return (a+b-abs(a-b))/2

def regularization(h):
    e = 0.01 # regularization factor
    hr = TrialFunction(V)
    a = (e**2*dot(grad(hr),grad(v))+hr*v)*dx
    L = dot(h,v)*dx
    hr = Function(V)
    solve(a==L,hr)
    return hr

max_iter = 10
dt = 0.2 # Gradient desccent variable
J_arr = np.array([])


for i in range(max_iter):
    u = TrialFunction(V)
    a1 = h*dot(grad(u),grad(v))*dx
    L1 = dot(f,v)*dx
    u = Function(V)
    solve(a1==L1,u,bc)

    p = TrialFunction(V)
    a2 = h*dot(grad(p),grad(v))*dx
    L2 = dot(-f,v)*dx
    p = Function(V)
    solve(a2==L2,p,bc)

    dJ = dot(grad(u),grad(p))
    print(f'dJ = {np.array(project(dJ,V).vector())}')
    ddt = dt/np.max(np.abs(np.array(h.vector())))

    h = h - ddt*dJ
    
    # proj0 = assemble(max(hmin, min(hmax,(h + l0)))*dx(mesh))
    # proj1 = assemble(max(hmin, min(hmax,(h + l1)))*dx(mesh))

    proj0 = assemble(Max(hmin, Min(hmax,(h + l0)))*dx(mesh))
    proj1 = assemble(Max(hmin,Min(hmax,(h + l1)))*dx(mesh))

    # Bisection algorithm
    ## Choose appropriate starting l0 and l1
    while (proj0>hfrac):
        l0 -= 0.1
        proj0 = assemble(Max(hmin,Min(hmax,h+l0))*dx(mesh))
    while (proj1<hfrac):
        l1 += 0.1
        proj1 = assemble(Max(hmin,Min(hmax,(h+l1)))*dx(mesh))

    ## Bisection algorithm
    while l1-l0 > lerr:
        lmid = (l0+l1)/2
        projmid = assemble(Max(hmin, Min(hmax,h+lmid))*dx(mesh))
        if projmid<hfrac:
            l0 = lmid
            proj0 = projmid
        else:
            l1 = lmid
            proj1 = projmid
    
    h = Max(hmin, Min(hmax, h + lmid))

    h = regularization(h)

    # plot(h)
    vertex_values_h = h.compute_vertex_values(mesh)
    ax1.clear()
    ax1.set(xlabel='X',ylabel='Y',zlabel='Thickness (h)')
    ax1.plot_trisurf(mshcox,mshcoy,vertex_values_h,cmap = plt.cm.jet)

    # plot(u)
    vertex_values_u = u.compute_vertex_values(mesh)
    ax4.clear()
    ax4.set(xlabel='X',ylabel='Y',zlabel='Displacement (u)')
    ax4.plot_trisurf(mshcox,mshcoy,vertex_values_u,cmap = plt.cm.jet)

    # plot objective function
    J = assemble(f_int*u*dx(mesh))
    J_arr = np.append(J_arr,J)
    print(f'Objective Function = {J}')
    ax2.scatter(i,J)

    if i==max_iter-1:
        plt.pause(10)
    else:
        plt.pause(0.001)

# ax1.set_title('Thickness (h) Plot')
# ax1.set(xlabel='X',ylabel='Y',zlabel='Thickness')
# ax1.plot_trisurf(mshcox,mshcoy,vertex_values_h,cmap = plt.cm.jet)
# 
# x_J = np.linspace(1,len(J_arr),len(J_arr))
# plt.scatter(x_J,J_arr)

plt.show()



f_plot = interpolate(f,V)
# plot(f_plot)
# plt.show()
plot(mesh)
plot(h)
plt.show()