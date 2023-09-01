import numpy as np
import dedalus.public as d3
from mpi4py import MPI
CW = MPI.COMM_WORLD
import logging
logger = logging.getLogger(__name__)

dtype = np.complex128
coords = d3.CartesianCoordinates('y', 'z', 'x')
dist = d3.Distributor(coords, dtype=dtype)

Lx = 1
Lz = 1
Nx = 16
Nz = 16

zbasis = d3.RealFourier(coords['z'], size=Nz, bounds=(0, Lz))
xbasis = d3.ChebyshevT(coords['x'], size=Nx, bounds=(-Lx / 2.0, Lx / 2.0))
bases = (zbasis, xbasis)
fbases = (zbasis,)

z = dist.local_grid(zbasis)
x = dist.local_grid(xbasis)

# nccs
A0 = dist.VectorField(coords, name='A0', bases=xbasis)

# Fields
p = dist.Field(name='p', bases=bases)
phi = dist.Field(name='phi', bases=bases)
u = dist.VectorField(coords, name='u', bases=bases)
A = dist.VectorField(coords, name='A', bases=bases)

taup = dist.Field(name='taup')

tau1u = dist.VectorField(coords, name='tau1u', bases=fbases)
tau2u = dist.VectorField(coords, name='tau2u', bases=fbases)

tau1A = dist.VectorField(coords, name='tau1A', bases=fbases)
tau2A = dist.VectorField(coords, name='tau2A', bases=fbases)

# operations
b = d3.Curl(A)
B0 = d3.Curl(A0)

ex = dist.VectorField(coords, name='ex')
ex['g'][2] = 1
integ = lambda A: d3.Integrate(d3.Integrate(A, 'z'), 'x')

lift_basis = xbasis.derivative_basis(1) # First derivative basis
lift = lambda A: d3.Lift(A, lift_basis, -1)
grad_u = d3.grad(u) + ex*lift(tau1u) # First-order reduction
grad_A = d3.grad(A) + ex*lift(tau1A) # First-order reduction

problem = d3.IVP([p, phi, u, A, taup, tau1u, tau2u, tau1A, tau2A], namespace=locals())
problem.add_equation("trace(grad_u) + taup = 0")
problem.add_equation("trace(grad_A) = 0")
problem.add_equation("dt(u) - div(grad_u) + grad(p) + lift(tau2u) = 0")
problem.add_equation("dt(A) + grad(phi) - div(grad_A) + lift(tau2A) - cross(u, B0) = 0")

# no-slip BCs
problem.add_equation("u(x='left') = 0")
problem.add_equation("u(x='right') = 0")

# vacuum BCs
problem.add_equation("A(x='left') = 0")
problem.add_equation("A(x='right') = 0")

# Pressure gauge
problem.add_equation("integ(p) = 0") 

solver = problem.build_solver(d3.RK111)
CFL = d3.CFL(solver, initial_dt=0.1, cadence=10, safety=0.3, threshold=0.05,
             max_change=1.5, min_change=0.5)
CFL.add_velocity(u)