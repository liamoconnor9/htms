"""
Usage:
    vp_bvp_func.py <config_file>
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
with open("status.txt", "w") as file:
    file.write(str(param_str) + "\n")

f =  R/np.sqrt(q)
eta = nu / Pm

# Evolution params
wall_time = 60. * 60. * wall_time_hr
dtype = np.float64

ncpu = MPI.COMM_WORLD.size
log2 = np.log2(ncpu)
if log2 == int(log2):
    mesh = [int(2**np.ceil(log2/2)),int(2**np.floor(log2/2))]
else:
    logger.error("pretty sure this shouldn't happen... log2(ncpu) is not an int?")
    
logger.info("running on processor mesh={}".format(mesh))

# Bases
coords = d3.CartesianCoordinates('y', 'z', 'x')
dist = d3.Distributor(coords, dtype=dtype, mesh=mesh)
dealias = 3/2

xbasis = d3.ChebyshevT(coords['x'], size=Nx, bounds=(-Lx / 2.0, Lx / 2.0), dealias=dealias)
ybasis = d3.RealFourier(coords['y'], size=Ny, bounds=(0, Ly), dealias=dealias)
zbasis = d3.RealFourier(coords['z'], size=Nz, bounds=(0, Lz), dealias=dealias)

y = dist.local_grid(ybasis)
z = dist.local_grid(zbasis)
x = dist.local_grid(xbasis)

# nccs
U0 = dist.VectorField(coords, name='U0', bases=xbasis)
S = 1e0
U0['g'][0] = S * x

fz_hat = dist.VectorField(coords, name='fz_hat', bases=xbasis)
fz_hat['g'][1] = f

# Fields
p = dist.Field(name='p', bases=(ybasis,zbasis,xbasis))
phi = dist.Field(name='phi', bases=(ybasis,zbasis,xbasis))
u = dist.VectorField(coords, name='u', bases=(ybasis,zbasis,xbasis))
A = dist.VectorField(coords, name='A', bases=(ybasis,zbasis,xbasis))
b = dist.VectorField(coords, name='b', bases=(ybasis,zbasis,xbasis))

taup = dist.Field(name='taup')
tauphi = dist.Field(name='tauphi', bases=(ybasis,zbasis))

tau1u = dist.VectorField(coords, name='tau1u', bases=(ybasis,zbasis))
tau2u = dist.VectorField(coords, name='tau2u', bases=(ybasis,zbasis))

tau1A = dist.VectorField(coords, name='tau1A', bases=(ybasis,zbasis))
tau2A = dist.VectorField(coords, name='tau2A', bases=(ybasis,zbasis))

# operations
b = d3.Curl(A)
b.store_last = True

ey = dist.VectorField(coords, name='ey')
ez = dist.VectorField(coords, name='ez')
ex = dist.VectorField(coords, name='ex')

ey['g'][0] = 1
ez['g'][1] = 1
ex['g'][2] = 1

by = (b@ey).evaluate()
bz = (b@ez).evaluate()
bx = (b@ex).evaluate()

# Initial conditions
lshape = dist.grid_layout.local_shape(u.domain, scales=1)
noise_coeff = 1e-3

rand = np.random.RandomState(seed=23 + CW.rank)
noise = x * (Lx - x) * noise_coeff * rand.standard_normal(lshape)
u['g'][0] = noise

rand = np.random.RandomState(seed=23 + CW.rank)
noise = noise_coeff * rand.standard_normal(lshape)
u['g'][1] = noise

rand = np.random.RandomState(seed=23 + CW.rank)
noise = noise_coeff * rand.standard_normal(lshape)
u['g'][2] = noise

Ay = A @ ey
Az = A @ ez
Ax = A @ ex

# Ay['g'], Az['g'], Ax['g'] = vp_bvp_func(by, bz, bx, Ay, Ax, Az, phi, coords)

dy = lambda A: d3.Differentiate(A, coords['y'])
dz = lambda A: d3.Differentiate(A, coords['z'])
dx = lambda A: d3.Differentiate(A, coords['x'])

integ = lambda A: d3.Integrate(d3.Integrate(d3.Integrate(A, 'y'), 'z'), 'x')

lift_basis = xbasis.derivative_basis(1) # First derivative basis
lift = lambda A: d3.Lift(A, lift_basis, -1)
grad_u = d3.grad(u) + ex*lift(tau1u) # First-order reduction
grad_A = d3.grad(A) + ex*lift(tau1A) # First-order reduction
grad_phi = d3.grad(phi) + ex*lift(tauphi)
grad_b = d3.grad(b)

b = d3.Curl(A).evaluate()

A.change_scales(1)
b.change_scales(1)
try:
    b['g'][0] = byg
except:
    b['g'][0] = eval(byg)
try:
    b['g'][1] = bzg
except:
    b['g'][1] = eval(bzg)
try:
    b['g'][2] = bxg
except:
    b['g'][2] = eval(bxg)
try:
    A['g'][0] = Ayg
except:
    A['g'][0] = eval(Ayg)
try:
    A['g'][1] = Azg
except:
    A['g'][1] = eval(Azg)
try:
    A['g'][2] = Axg
except:
    A['g'][2] = eval(Axg)

logger.info('solving bvp for vector potential A given b')
problem = d3.LBVP(variables=[A, phi, tau1A, tauphi], namespace=locals())

problem.add_equation("trace(grad_A) = 0")
problem.add_equation("curl(A) + grad_phi + lift(tau1A) = b")

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
