"""
3D cartesian MRI initial value problem using vector potential formulation
Usage:
    mri.py <config_file>
"""
from unicodedata import decimal
from docopt import docopt
from configparser import ConfigParser
import time
from pathlib import Path
import numpy as np
import os
import sys
import h5py
import gc
import pickle
import dedalus.public as d3
from mpi4py import MPI
import pickle

CW = MPI.COMM_WORLD
import logging
import pathlib
logger = logging.getLogger(__name__)
sys.path.append('..')
    
args = docopt(__doc__)
filename = Path(args['<config_file>'])

config = ConfigParser()
config.read(str(filename))

from read_config import ConfigEval
try:
    args = docopt(__doc__)
    filename = Path(args['<config_file>'])
except:
    filename = path + '/config.cfg'
config = ConfigEval(filename)
locals().update(config.execute_locals())

logger.info('Running mri.py with the following parameters:')
param_str = config.items('parameters')
logger.info(param_str)

f =  R/np.sqrt(q)
eta = nu / Pm

# Evolution params
dtype = np.complex128

coords = d3.CartesianCoordinates('y', 'z', 'x')
dist = d3.Distributor(coords, dtype=dtype)

# Bases
Ly, Lz, Lx = eval(str(Ly)), eval(str(Lz)), eval(str(Lx))
Nz = 2
zbasis = d3.ComplexFourier(coords['z'], size=Nz, bounds=(0, Lz))
xbasis = d3.ChebyshevT(coords['x'], size=Nx, bounds=(-Lx / 2.0, Lx / 2.0))
bases = (zbasis, xbasis)
fbases = (zbasis,)

z = dist.local_grid(zbasis)
x = dist.local_grid(xbasis)

if not is2D:
    ybasis = d3.RealFourier(coords['y'], size=Ny, bounds=(0, Ly))
    y = dist.local_grid(ybasis)
    bases = (ybasis,) + bases
    fbases = (ybasis,) + fbases

# nccs
U0 = dist.VectorField(coords, name='U0', bases=xbasis)
S = 1e0
U0['g'][0] = S * x

A0 = dist.VectorField(coords, name='A0', bases=xbasis)

fz_hat = dist.VectorField(coords, name='fz_hat', bases=xbasis)
fz_hat['g'][1] = f

# Fields
omega = dist.Field(name='omega')
dt = lambda A: -1j*omega*A

p = dist.Field(name='p', bases=bases)
phi = dist.Field(name='phi', bases=bases)
u = dist.VectorField(coords, name='u', bases=bases)
A = dist.VectorField(coords, name='A', bases=bases)

taup = dist.Field(name='taup')
tauphi = dist.Field(name='tauphi')
# tauphi = dist.Field(name='tauphi', bases=fbases)

tau1u = dist.VectorField(coords, name='tau1u', bases=fbases)
tau2u = dist.VectorField(coords, name='tau2u', bases=fbases)

tau1b = dist.VectorField(coords, name='tau1b', bases=fbases)
tau2b = dist.VectorField(coords, name='tau2b', bases=fbases)

tau1A = dist.VectorField(coords, name='tau1A', bases=fbases)
tau2A = dist.VectorField(coords, name='tau2A', bases=fbases)

# operations

ey = dist.VectorField(coords, name='ey')
ez = dist.VectorField(coords, name='ez')
ex = dist.VectorField(coords, name='ex')

ey['g'][0] = 1
ez['g'][1] = 1
ex['g'][2] = 1

Ay = A @ ey
Az = A @ ez
Ax = A @ ex

# Ay['g'], Az['g'], Ax['g'] = vp_bvp_func(by, bz, bx, Ay, Ax, Az, phi, coords)

dy = lambda A: d3.Differentiate(A, coords['y'])
dz = lambda A: d3.Differentiate(A, coords['z'])
dx = lambda A: d3.Differentiate(A, coords['x'])

integ = lambda A: d3.Integrate(d3.Integrate(A, 'z'), 'x')
if not is2D:
    integ = lambda A: d3.Integrate(integ(A), 'y')

lift_basis = xbasis.derivative_basis(1) # First derivative basis
lift = lambda A: d3.Lift(A, lift_basis, -1)
b = d3.curl(A).evaluate()
B0 = d3.curl(A0)
grad_u = d3.grad(u) + ex*lift(tau1u) # First-order reduction
grad_A = d3.grad(A) + ex*lift(tau1A) # First-order reduction
grad_phi = d3.grad(phi) + ex*lift(tauphi)
grad_b = d3.grad(b) + ex*lift(tau1b)
grad_B0 = d3.grad(B0)

A.change_scales(1)
A0.change_scales(1)
A['g'] = 0.0

A0['g'][0] = eval(str(Ayg))
A0['g'][1] = eval(str(Azg))
A0['g'][2] = eval(str(Axg))
    
problem = d3.EVP([p, u, b, taup, tau1u, tau2u, tau1b, tau2b], namespace=locals(), eigenvalue=omega)
problem.add_equation("trace(grad_u) + taup = 0")
# problem.add_equation("trace(grad_b) + tauphi = 0")
problem.add_equation("dt(u) + dot(u,grad(U0)) + dot(U0,grad_u) - dot(b, grad(B0)) - dot(B0,grad_b) - nu*div(grad_u) + grad(p) + cross(fz_hat, u) + lift(tau2u) = 0")
problem.add_equation("dt(b) - eta*div(grad_b) + lift(tau2b) - curl(cross(u, B0)) - curl(cross(U0, b)) = 0")

if (isNoSlip):
    # no-slip BCs
    problem.add_equation("u(x='left') = 0")
    problem.add_equation("u(x='right') = 0")
else:
    # stress-free BCs
    problem.add_equation("dot(u, ex)(x='left') = 0")
    problem.add_equation("dot(u, ex)(x='right') = 0")
    problem.add_equation("dot(dx(u), ey)(x='left') = 0")
    problem.add_equation("dot(dx(u), ey)(x='right') = 0")
    problem.add_equation("dot(dx(u), ez)(x='left') = 0")
    problem.add_equation("dot(dx(u), ez)(x='right') = 0")

# Pressure gauge
problem.add_equation("integ(p) = 0") 

problem.add_equation("dot(b, ex)(x='left')=0")
problem.add_equation("dot(b, ex)(x='right')=0")

problem.add_equation("dot(curl(b), ex)(x='left')=0")
problem.add_equation("dot(curl(b), ex)(x='right')=0")

use divergence condition
problem.add_equation("dot(dx(curl(b)), ex)(x='left')=0")
problem.add_equation("dot(dx(curl(b)), ex)(x='right')=0")

solver = problem.build_solver(entry_cutoff=0)
solver.solve_sparse(solver.subproblems[1], NEV, target=0)
print('kz = {}, omega = {}'.format(kz, np.max(solver.eigenvalues.imag)))