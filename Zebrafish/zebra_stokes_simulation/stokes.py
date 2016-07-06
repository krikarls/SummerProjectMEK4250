from dolfin import *

mesh = Mesh("lin_zebra_mesh.xml.gz")
plot(mesh,interactive=True)