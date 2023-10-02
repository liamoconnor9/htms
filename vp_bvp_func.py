import numpy as np
import dedalus.public as d3
from mpi4py import MPI
CW = MPI.COMM_WORLD
import logging
logger = logging.getLogger(__name__)

dtype = np.float64
ncpu = MPI.COMM_WORLD.size
Nx, Ny, Nz = 8, 8, 8
Lx, Ly, Lz = 1., 1., 1.

# Bases
coords = d3.CartesianCoordinates('y', 'z', 'x')
dist = d3.Distributor(coords, dtype=dtype)
dealias = 3/2

xbasis = d3.ChebyshevT(coords['x'], size=Nx, bounds=(-Lx / 2.0, Lx / 2.0), dealias=dealias)
ybasis = d3.RealFourier(coords['y'], size=Ny, bounds=(0, Ly), dealias=dealias)
zbasis = d3.RealFourier(coords['z'], size=Nz, bounds=(0, Lz), dealias=dealias)

y = dist.local_grid(ybasis)
z = dist.local_grid(zbasis)
x = dist.local_grid(xbasis)

# Fields
phi = dist.Field(name='phi', bases=(ybasis,zbasis,xbasis))
A = dist.VectorField(coords, name='A', bases=(ybasis,zbasis,xbasis))
b = dist.VectorField(coords, name='b', bases=(ybasis,zbasis,xbasis))

tauphi = dist.Field(name='tauphi', bases=(ybasis,zbasis))
tau1A = dist.VectorField(coords, name='tau1A', bases=(ybasis,zbasis))

ey = dist.VectorField(coords, name='ey')
ez = dist.VectorField(coords, name='ez')
ex = dist.VectorField(coords, name='ex')

ey['g'][0] = 1
ez['g'][1] = 1
ex['g'][2] = 1

Ay = A @ ey
Az = A @ ez
Ax = A @ ex

dy = lambda A: d3.Differentiate(A, coords['y'])
dz = lambda A: d3.Differentiate(A, coords['z'])
dx = lambda A: d3.Differentiate(A, coords['x'])

integ = lambda A: d3.Integrate(d3.Integrate(d3.Integrate(A, 'y'), 'z'), 'x')

lift_basis = xbasis.derivative_basis(1) # First derivative basis
lift = lambda A: d3.Lift(A, lift_basis, -1)
grad_A = d3.grad(A) + ex*lift(tau1A) # First-order reduction
grad_phi = d3.grad(phi) + ex*lift(tauphi)

b = d3.Curl(A).evaluate()

logger.info('solving bvp for vector potential A given b')
problem = d3.LBVP(variables=[A, phi, tau1A, tauphi], namespace=locals())

problem.add_equation((d3.trace(grad_A), 0))
problem.add_equation((d3.curl(A) + grad_phi + lift(tau1A), b))

problem.add_equation("Ay(x='left') = 0", condition="(ny!=0) or (nz!=0)")
problem.add_equation("Az(x='left') = 0", condition="(ny!=0) or (nz!=0)")
problem.add_equation("Ay(x='right') = 0", condition="(ny!=0) or (nz!=0)")
problem.add_equation("Az(x='right') = 0", condition="(ny!=0) or (nz!=0)")

problem.add_equation("Ax(x='left') = 0", condition="(ny==0) and (nz==0)")
problem.add_equation("Ay(x='left') = 0", condition="(ny==0) and (nz==0)")
problem.add_equation("Az(x='left') = 0", condition="(ny==0) and (nz==0)")
problem.add_equation("phi(x='left') = 0", condition="(ny==0) and (nz==0)")

# Build solver
solver = problem.build_solver()
solver.solve()
logger.info('bvp solved.')
