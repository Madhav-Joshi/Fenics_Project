# Minimize J(n) = int_2D(nfu)dx over n
## n = mollifier
## f = force
## u = displacement
## That satisfy -div(dot((nh+(1-n)eh),grad(u)))=f on Omega(2D)
## with BC's --> u = 0 on Boundary of Omega  
## h = Thickness = Constant
## e = small number tending to zero


from fenics import *
import numpy as np 
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d

plt.ion()
fig = plt.figure()
ax1 = fig.add_subplot(1,2,2,projection = '3d')
ax2 = fig.add_subplot(1,2,1)
ax2.set_title('Objective Function Plot')
ax2.set_xlabel('Iteration number')
ax2.set_ylabel('J(h)')


# Steps
## Define mesh, force(f), thickness(h)
mesh = Mesh('fenics_trial/Square.xml')
mesh = refine(mesh)
mesh = refine(mesh)
mesh = refine(mesh)

mshco = mesh.coordinates()
mshcox = mshco[:,0]
mshcoy = mshco[:,1]

V = FunctionSpace(mesh, 'P', 1)

def boundary(x, on_boundary):
    return on_boundary
bc = DirichletBC(V, 0, boundary)

F = 5
r =0.1
xc = 1.5
yc = 1
f = Expression('F*exp(-((pow(x[0]-xc, 2) + pow(x[1] - yc, 2))/(2*pow(r,2))))', degree=1, F=F, xc=xc, yc=yc, r=r)
f_int = interpolate(f,V)

h = 0.25 # Constant Thickness
n = interpolate(Constant(1),V) # Mollifier
n = project(n,V)
e = 1e-5
v = TestFunction(V)
dt = 0.1 # Gradient descent update variable

max_iter = 20

for i in range(max_iter):
    u = TrialFunction(V) # Displacement
    a1 = (n*h+(1-n)*e*h)*dot(grad(u),grad(v))*dx
    L1 = dot(f,v)*dx
    u = Function(V)
    solve(a1==L1,u,bc)

    p = TrialFunction(V) # Adjoint Variable
    a2 = (n*h+(1-n)*e*h)*dot(grad(p),grad(v))*dx
    L2 = dot(-n*f,v)*dx
    p = Function(V)
    solve(a2==L2,p,bc)

    dJ = h*(1-e)*dot(grad(u),grad(p)) # dJ wrt n

    n_arr = np.array(n.vector())
    h_new = n_arr*h+(1-n_arr)*e*h
    ddt = dt/np.max(np.abs(h_new))

    n = n - ddt*dJ
    n = project(n,V)
    # plot(n)
    vertex_values_n = n.compute_vertex_values(mesh)
    ax1.clear()
    ax1.set(xlabel='X',ylabel='Y',zlabel='Thickness (h)')
    ax1.plot_trisurf(mshcox,mshcoy,vertex_values_n,cmap = plt.cm.jet)

    # Plot objective function
    J = assemble(n*f_int*u*dx(mesh))
    ax2.scatter(i,J)

    if i==max_iter-1:
        plt.pause(5)
    else:
        plt.pause(0.001)
