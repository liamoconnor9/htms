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
if CW.rank == 0:
    with open(suffix + "/status.txt", "w") as file:
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


zbasis = d3.RealFourier(coords['z'], size=Nz, bounds=(0, Lz), dealias=dealias)
xbasis = d3.ChebyshevT(coords['x'], size=Nx, bounds=(-Lx / 2.0, Lx / 2.0), dealias=dealias)
bases = (zbasis, xbasis)
fbases = (zbasis,)

z = dist.local_grid(zbasis)
x = dist.local_grid(xbasis)

if not is2D:
    ybasis = d3.RealFourier(coords['y'], size=Ny, bounds=(0, Ly), dealias=dealias)
    y = dist.local_grid(ybasis)
    bases = (ybasis,) + bases
    fbases = (ybasis,) + fbases

# nccs
U0 = dist.VectorField(coords, name='U0', bases=xbasis)
S = 1e0
U0['g'][0] = S * x

fz_hat = dist.VectorField(coords, name='fz_hat', bases=xbasis)
fz_hat['g'][1] = f



# Fields
p = dist.Field(name='p', bases=bases)
phi = dist.Field(name='phi', bases=bases)
u = dist.VectorField(coords, name='u', bases=bases)
A = dist.VectorField(coords, name='A', bases=bases)

taup = dist.Field(name='taup')
tauphi = dist.Field(name='tauphi', bases=fbases)

tau1u = dist.VectorField(coords, name='tau1u', bases=fbases)
tau2u = dist.VectorField(coords, name='tau2u', bases=fbases)

tau1A = dist.VectorField(coords, name='tau1A', bases=fbases)
tau2A = dist.VectorField(coords, name='tau2A', bases=fbases)

# operations
b = d3.Curl(A)

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

integ = lambda A: d3.Integrate(d3.Integrate(A, 'z'), 'x')
if not is2D:
    integ = lambda A: d3.Integrate(integ(A), 'y')

lift_basis = xbasis.derivative_basis(1) # First derivative basis
lift = lambda A: d3.Lift(A, lift_basis, -1)
grad_u = d3.grad(u) + ex*lift(tau1u) # First-order reduction
grad_A = d3.grad(A) + ex*lift(tau1A) # First-order reduction
grad_phi = d3.grad(phi) + ex*lift(tauphi)
grad_b = d3.grad(b)

# b = d3.Curl(A).evaluate()

# b.change_scales(1)
# b['g'][0] = eval(str(byg))
# b['g'][1] = eval(str(bzg))
# b['g'][2] = eval(str(bxg))


A.change_scales(1)
A['g'][0] = eval(str(Ayg))
A['g'][1] = eval(str(Azg))
A['g'][2] = eval(str(Axg))

problem = d3.IVP([p, phi, u, A, taup, tau1u, tau2u, tau1A, tau2A], namespace=locals())
problem.add_equation("trace(grad_u) + taup = 0")
problem.add_equation("trace(grad_A) = 0")
problem.add_equation("dt(u) + dot(u,grad(U0)) + dot(U0,grad(u)) - nu*div(grad_u) + grad(p) + lift(tau2u) = dot(b, grad_b) - dot(u,grad(u)) - cross(fz_hat, u)")
problem.add_equation("dt(A) + grad(phi) - eta*div(grad_A) + lift(tau2A) = cross(u, b) + cross(U0, b)")

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

problem.add_equation("dot(A, ey)(x='left') = 0")
problem.add_equation("dot(A, ez)(x='left') = 0")
problem.add_equation("dot(A, ey)(x='right') = 0")
problem.add_equation("dot(A, ez)(x='right') = 0")
problem.add_equation("phi(x='left') = 0")
problem.add_equation("phi(x='right') = 0")

# Solver
solver = problem.build_solver(d3.SBDF2)
solver.stop_sim_time = stop_sim_time

# sys.exit()
# A['g'][0], A['g'][1], A['g'][2] = vp_bvp_func(by, bz, bx)

fh_mode = 'overwrite'

checkpoint = solver.evaluator.add_file_handler(suffix + '/checkpoint', max_writes=1, sim_dt=checkpoint_dt)
checkpoint.add_tasks(solver.state, layout='g')

slicepoints = solver.evaluator.add_file_handler(suffix + '/slicepoints', sim_dt=slicepoint_dt, max_writes=50, mode=fh_mode)

for field, field_name in [(b, 'b'), (u, 'v')]:
    for d2, unit_vec in zip(('x', 'y', 'z'), (ex, ey, ez)):
        slicepoints.add_task(d3.dot(field, unit_vec)(x = 'center'), name = "{}{}_mid{}".format(field_name, d2, 'x'))
        slicepoints.add_task(d3.dot(field, unit_vec)(z = 'center'), name = "{}{}_mid{}".format(field_name, d2, 'z'))
        if not is2D:
            slicepoints.add_task(d3.dot(field, unit_vec)(y = 'center'), name = "{}{}_mid{}".format(field_name, d2, 'y'))

scalars = solver.evaluator.add_file_handler(suffix + '/scalars', sim_dt=scalar_dt, max_writes=50, mode=fh_mode)
scalars.add_task(integ(np.sqrt(d3.dot(u, u)) + np.sqrt(d3.dot(b, b))), name='x_l2')
scalars.add_task(integ(np.sqrt(d3.dot(u, u))), name='u_l2')
scalars.add_task(integ(np.sqrt(d3.dot(b, b))), name='b_l2')

CFL = d3.CFL(solver, initial_dt=init_timestep, cadence=10, safety=0.3, threshold=0.05,
             max_change=1.5, min_change=0.5)
# CFL.add_velocity(u)
# CFL.add_velocity(b)

# Flow properties
flow = d3.GlobalFlowProperty(solver, cadence=1)
flow.add_property(np.sqrt(d3.dot(u, u)) + np.sqrt(d3.dot(b, b)), name='x_l2')
flow.add_property(np.sqrt(d3.dot(u, u)), name='u_l2')
flow.add_property(np.sqrt(d3.dot(b, b)), name='b_l2')

# Main loop
# print(flow.properties.tasks)
solver.evaluator.evaluate_handlers((flow.properties, ))
# metrics = 'Iteration, Time, dt, x_l2, u_l2, b_l2'
# with open(suffix + "/data.csv", "w") as file:
    # file.write(metrics + "\n")

try:
    logger.info('Starting main loop')
    while solver.proceed:

        timestep = CFL.compute_timestep()
        if (solver.iteration-1) % 100 == 0:
            x_l2 = flow.volume_integral('x_l2')
            u_l2 = flow.volume_integral('u_l2')
            b_l2 = flow.volume_integral('b_l2')
            # status = 'Iteration=%i, Time=%e, dt=%e, x_l2%f, u_l2%f, b_l2=%f' %(solver.iteration, solver.sim_time, timestep, x_l2, u_l2, b_l2)
            nums = [solver.iteration, solver.sim_time, timestep, x_l2, u_l2, b_l2]
            status = 'Iteration={}, Time={}, dt={}, x_l2={}, u_l2={}, b_l2={}'.format(*nums)
            logger.info(status)
            if CW.rank == 0:
                with open(suffix + "/status.txt", "a") as file:
                    file.write(status + "\n")
            # with open(suffix + "/data.csv", "a") as file:
            #     file.write(str(nums)[1:-1] + "\n")

        solver.step(timestep)
except:
    logger.error('Exception raised, triggering end of main loop.')
    raise


# finally:
    # solver.log_stats()
