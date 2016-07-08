from dolfin import *

mesh = Mesh("orthogonal_surface_zebra_mesh.xml")

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

class Passive(SubDomain): 				# y-max surface
	def inside(self, x, on_boundry):
		return (x[1] > 373.639-0.01) and on_boundry

class Outlet(SubDomain): 				# x-min surface
	def inside(self, x, on_boundry):
		return (x[0] < 282.272+0.01) and on_boundry

class Inlet(SubDomain): 				# x-max surface, y < 330
	def inside(self, x, on_boundry):
		return (x[0] > 325.545-4) and (x[1] < 330.) and on_boundry

noslip = NoSlip()
passive_boundary = Passive()
inlet = Inlet()
outlet = Outlet()

mf = FacetFunction("size_t", mesh)
mf.set_all(4)

noslip.mark(mf,0)
passive_boundary.mark(mf, 1)
inlet.mark(mf,2)
outlet.mark(mf,3)
plot(mf,interactive=True)

# Define spaces and test/trial functions
V = VectorFunctionSpace(mesh, "CG", 2)
Q = FunctionSpace(mesh, "CG", 1)
W = V * Q
w = Function(W)

(u, p) = TrialFunctions(W)
(v, q) = TestFunctions(W)

# Set boundary conditions
inlet_pressure = Constant(0.1)
outlet_pressure = Constant(0)
p_in = DirichletBC(W.sub(1), inlet_pressure, mf, 2)
p_out = DirichletBC(W.sub(1), outlet_pressure, mf, 3)
noslip = DirichletBC(W.sub(0), Constant((0,0,0)), mf, 0)
bcs = [p_in, p_out, noslip]

# Define variational problem
a = (inner(grad(u), grad(v)) - div(v)*p + q*div(u))*dx
L = rhs(a) 

# Compute solution
solve(a == L, w, bcs)

(u, p) = w.split()

plot(u,title='velocity')
plot(p,title='pressure')
interactive()
