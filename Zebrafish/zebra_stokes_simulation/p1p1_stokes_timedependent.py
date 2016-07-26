from dolfin import *

mesh = Mesh("new_coarser.xml")

"""
The mesh is constructed such that the openings are orthogonal to the
coordinate axis. Then, marking the boundaries are easy since they 
are simply given by planes in max/min-values of the mesh coordinates.

x = interpolate(Expression("x[0]"),V)
y = interpolate(Expression("x[1]"),V)
z = interpolate(Expression("x[2]"),V)

x =  x.vector().array()
y =  y.vector().array()
z =  z.vector().array()

print 'Min and max x-value: ' , min(x) ,'  ', max(x) 
print 'Min and max y-value: ' , min(y) ,'  ', max(y) 
print 'Min and max z-value: ' , min(z) ,'  ', max(z) 
"""

# Mark boundaries
class NoSlip(SubDomain): 				
	def inside(self, x, on_boundry):
		return on_boundry

class Inlet(SubDomain): 				# y-max surface
	def inside(self, x, on_boundry):
		return (x[1] > 367.467-0.01) and on_boundry

class Passive(SubDomain): 				# x-min surface
	def inside(self, x, on_boundry):
		return (x[0] < 296.37+0.01) and on_boundry

class Outlet(SubDomain): 				# x-max surface, y < 330
	def inside(self, x, on_boundry):
		return (x[0] > 325.491-4) and (x[1] < 330.) and on_boundry

mf = FacetFunction("size_t", mesh)
mf.set_all(4)

NoSlip().mark(mf,0)
Inlet().mark(mf,1)
Outlet().mark(mf,2)
Passive().mark(mf, 3)
#plot(mf,interactive=True)

# Define spaces and test/trial functions
V = VectorFunctionSpace(mesh, "CG", 1)
Q = FunctionSpace(mesh, "CG", 1)
W = V*Q
w = Function(W)

(u, p) = TrialFunctions(W)
(v, q) = TestFunctions(W)

# Physical parameters
mu = 3.5*1e-9 						# kg/(micrometer*s)
p_atm = 101e-3 						# kg/(micrometem*s^2).
distance_between_openings = 58 		# micrometer
pressure_gradient = 2.2e-7			# (kg/(micrometer*s^2))/micrometer

# Set boundary conditions
p_0 = p_atm
pressure_difference = distance_between_openings*pressure_gradient

inlet1_pressure = Expression(("p_0 + 20*pressure_gradient*sin(5*pi*t)"),p_0=p_0,pressure_gradient=pressure_gradient, t=0)
outlet2_pressure = Constant(Constant(p_0+pressure_difference))
outlet3_pressure = Constant(Constant(p_0+0.75*pressure_difference)) 
noslip = DirichletBC(W.sub(0), Constant((0,0,0)), mf, 0)

bcs = [noslip]

# Define variational problem
h = CellSize(mesh)
beta  = 0.1
epsilon = beta*h*h
f = Constant((0,0,0))

n = FacetNormal(mesh)
ds = ds[mf]

time_steps = 100
dt = 0.01 
T = dt*time_steps

a = (mu*inner(grad(v), grad(u)) + div(v)*p + q*div(u) - epsilon*inner(grad(q), grad(p)))*dx 
L = inner(v + epsilon*grad(q), f)*dx + inner(v,inlet1_pressure*n)*ds(1) + \
	inner(v,outlet2_pressure*n)*ds(2) + inner(v,outlet3_pressure*n)*ds(3)

ufile = File('results/velocity.pvd')
pfile = File('results/pressure.pvd')

t = 0.
for i in range(0,time_steps):
	print 'Progress(time): ', t,'/',T, '  ', 100.*(t/T), '%'
	# Update BC
	inlet1_pressure.t = t

	# Compute solution
	solve(a == L, w, bcs)
	(u, p) = w.split()

	ufile << u
	pfile << p

	t += dt


plot(u,title='velocity')
plot(p,title='pressure')
interactive()
