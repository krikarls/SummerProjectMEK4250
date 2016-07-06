from dolfin import *

mesh = Mesh("orthogonal_surface_zebra_mesh.xml")
V = FunctionSpace(mesh,'CG',1)

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

class Passive(SubDomain): 				# y-max surface
	def inside(self, x, on_boundry):
		return (x[1] > 373.639-0.01) and on_boundry

class Inlet(SubDomain): 				# x-min surface
	def inside(self, x, on_boundry):
		return (x[0] < 282.272+0.01) and on_boundry

class Outlet(SubDomain): 				# x-max surface
	def inside(self, x, on_boundry):
		return (x[0] > 325.545-4) and (x[1] < 330.) and on_boundry

passive_boundary = Passive()
inlet = Inlet()
outlet = Outlet()

mf = FacetFunction("size_t", mesh)
mf.set_all(0)

passive_boundary.mark(mf, 1)
inlet.mark(mf,2)
outlet.mark(mf,3)

plot(mf,interactive=True)
