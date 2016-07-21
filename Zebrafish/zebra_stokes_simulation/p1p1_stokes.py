from dolfin import *

mesh = Mesh("new_coarser.xml")

"""
The mesh is constructed such that the openings are orthogonal to the
coordinate axis. Then, marking the boundaries are easy since they 
are simply given by planes in max/min-values of the mesh coordinates.

V = FunctionSpace(mesh, "CG", 1)
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

# Physical parameters
mu = 3.5*1e-9 		# kg/(micrometer*s)

# Define spaces and test/trial functions
V = VectorFunctionSpace(mesh, "CG", 1)
Q = FunctionSpace(mesh, "CG", 1)
W = V*Q
w = Function(W)

(u, p) = TrialFunctions(W)
(v, q) = TestFunctions(W)

# Set boundary conditions

p_in = 80*1e-7
p_in = p_in - 0.1*p_in 

inlet1_pressure = Constant(p_in)
outlet2_pressure = Constant(Constant(p_in+ 0.3*p_in))
outlet3_pressure = Constant(Constant(p_in)) 
noslip = DirichletBC(W.sub(0), Constant((0,0,0)), mf, 0)
bcs = [noslip]

# Define variational problem
h = CellSize(mesh)
beta  = 0.1
epsilon = beta*h*h
f = Constant((0,0,0))

n = FacetNormal(mesh)
ds = ds[mf]

a = (mu*inner(grad(v), grad(u)) + div(v)*p + q*div(u) - epsilon*inner(grad(q), grad(p)))*dx 
L = inner(v + epsilon*grad(q), f)*dx + inner(v,inlet1_pressure*n)*ds(0) + \
	inner(v,outlet2_pressure*n)*ds(2) + inner(v,outlet3_pressure*n)*ds(3)

# Compute solution
solve(a == L, w, bcs)

(u, p) = w.split()

file = File('velocity.pvd')
file << u

file = File('pressure.pvd')
file << p

plot(u,title='velocity')
interactive()
