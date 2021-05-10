from fenics import *
import numpy as np
import matplotlib.pyplot as plt 

fig = plt.figure()
fig.show()

T = 2.0            # final time
num_steps = 30     # number of time steps
dt = T / num_steps # time step size
alpha = 3          # parameter alpha
beta = 1.2         # parameter beta
gamma = 2

# Create mesh and define function space
nx = ny = 3
mesh = UnitSquareMesh(nx, ny)
V = FunctionSpace(mesh, 'P', 1)

# Define boundary condition
u_D = Expression('1 + x[0]*x[0] + alpha*x[1]*x[1] + beta*t*t',
                 degree=2, alpha=alpha, beta=beta, t=0)

def boundary(x, on_boundary):
    return on_boundary

bc = DirichletBC(V, u_D, boundary)

# Define initial value
u_n = interpolate(u_D, V)
#u_n = project(u_D, V)

# Define variational problem
u = TrialFunction(V)
v = TestFunction(V)
f = Expression('2*t*beta/gamma - 2 - 2*alpha',degree = 2, beta=beta,alpha=alpha,gamma=gamma,t=0)

F = u*v*dx + gamma*dt*dot(grad(u), grad(v))*dx - (u_n + gamma*dt*f)*v*dx
a, L = lhs(F), rhs(F)

# Time-stepping
u = Function(V)
t = 0
for n in range(num_steps):

    # Update current time
    t += dt
    u_D.t = t
    f.t = t 

    # Compute solution
    solve(a == L, u, bc)

    # Plot solution
    fig.clear()
    plot(u,mode='warp')
    fig.canvas.draw()
    plt.pause(0.2)

    # Compute error at vertices
    u_e = interpolate(u_D, V)
    error = np.max(np.abs(np.array(u_e.vector()) - np.array(u.vector())))
    print('t = %.2f: error = %.3g' % (t, error))

    # Update previous solution
    u_n.assign(u)

# Hold plot
plt.show()