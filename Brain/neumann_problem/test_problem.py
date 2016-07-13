
"""
This is a demo for the pure traction problem in linear elasticity.
The example problem is a unit cube with a traction force on the side.
First, the problem is solved my clamping one side.
Second, the problem is solved by removing the nullspace of RM. 
"""

from dolfin import *

# Create mesh
mesh = UnitCubeMesh(10,10,10)

# Set up function spaces
V = VectorFunctionSpace(mesh, "Lagrange", 1)
u = TrialFunction(V)
v = TestFunction(V)
u_ = Function(V)

# Mark boundaries
class Neumann_boundary(SubDomain):
	def inside(self, x, on_boundry):
		return on_boundry

class Clamped(SubDomain):
	def inside(self, x, on_boundry):
		return (x[0] < 1e-6) and on_boundry

class Traction_boundary(SubDomain):
	def inside(self, x, on_boundry):
		return (x[0]> 1-1e-6) and on_boundry

mf = FacetFunction("size_t", mesh)
mf.set_all(3)

Neumann_boundary().mark(mf, 0)
Clamped().mark(mf, 1)
Traction_boundary().mark(mf, 2) # imposed weakly 
ds = ds[mf]

#plot(mf, interactive=True) 		# check that marking is correct

zero_displacement = Constant(("0.0", "0.0", "0.0"))
bc1 = DirichletBC(V, zero_displacement, mf, 1)

# Continuum mechanics
E, nu = 10.0, 0.3
mu, lambda_ = Constant(E/(2*(1 + nu))), Constant(E*nu/((1 + nu)*(1 - 2*nu)))
epsilon = lambda u: sym(grad(u))

T = Constant((-.5, 0, 0))
f = Constant((0, 0, 0))

a = 2*mu*inner(epsilon(u),epsilon(v))*dx + lambda_*inner(div(u),div(v))*dx
L = inner(f, v)*dx + inner(T,v)*ds(2)

def a_fun(u,v):
	return 2*mu*inner(epsilon(u),epsilon(v))*dx + lambda_*inner(div(u),div(v))*dx

def L_fun(v):
	return inner(f, v)*dx + inner(T,v)*ds(2)

solve(a==L, u_, bcs=[bc1])
plot(u_, title="Clamped_displacement", mode="displacement")

"""
So far we have not considered a pure traction problem. So let's do it:
By simply removing the bc1 we have such a problem and the solver does not 
converge. The way we now will fix it is by removing the nullspace of rigid
motions removed by Lagrange multipliers.
"""

def Neumann_problem_solver(a,L,V):
	V = VectorFunctionSpace(mesh, 'Lagrange', 1) # space for displacement 
	R = FunctionSpace(mesh, 'R', 0)        # space for one Lagrange multiplier
	M = MixedFunctionSpace([R]*6)          # space for all multipliers
	W = MixedFunctionSpace([V, M])
	u, mus = TrialFunctions(W)
	v, nus = TestFunctions(W)

	# Establish a basis for the nullspace of RM
	e0 = Constant((1, 0, 0))				# translations
	e1 = Constant((0, 1, 0))
	e2 = Constant((0, 0, 1))

	e3 = Expression(('-x[1]', 'x[0]', '0')) # rotations
	e4 = Expression(('-x[2]', '0', 'x[0]'))
	e5 = Expression(('0', '-x[2]', 'x[1]'))
	basis_vectors = [e0, e1, e2, e3, e4, e5]

	a = a_fun(u,v)
	L = L_fun(v)

	# Lagrange multipliers contrib to a
	for i, e in enumerate(basis_vectors):
		mu = mus[i]
		nu = nus[i]
		a += mu*inner(v, e)*dx + nu*inner(u, e)*dx

	# Assemble the system
	A = PETScMatrix()
	b = PETScVector()
	assemble_system(a, L, A_tensor=A, b_tensor=b)

	# Solve
	uh = Function(W)
	solver = PETScLUSolver('mumps') # NOTE: we use direct solver for simplicity
	solver.set_operator(A)
	solver.solve(uh.vector(), b)

	# Split displacement and multipliers. Plot
	u, ls = uh.split(deepcopy=True) 
	plot(u, mode='displacement', title='Neumann_displacement', interactive=True)


Neumann_problem_solver(a,L,V)

