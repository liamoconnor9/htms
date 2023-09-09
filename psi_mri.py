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

# Evolution params
wall_time = 60. * 60. * wall_time_hr
# dtype = np.float64

ncpu = MPI.COMM_WORLD.size
log2 = np.log2(ncpu)
if log2 == int(log2):
    mesh = [int(2**np.ceil(log2/2)),int(2**np.floor(log2/2))]
else:
    logger.error("pretty sure this shouldn't happen... log2(ncpu) is not an int?")
    
logger.info("running on processor mesh={}".format(mesh))
dealias = 3/2

if (solveEVP):
    dtype = np.complex128
else:
    dtype = np.float64

coords = d3.CartesianCoordinates('z', 'y', 'x')
dist = d3.Distributor(coords, dtype=dtype, mesh=mesh)

# Bases
Ly, Lz, Lx = eval(str(Ly)), eval(str(Lz)), eval(str(Lx))
if (solveEVP):
    Nz = 2
    zbasis = d3.ComplexFourier(coords['z'], size=Nz, bounds=(0, Lz))
else:
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
eta = nu / Pm


# Fields
if (solveEVP):
    omega = dist.Field(name='omega')
    dt = lambda A: -1j*omega*A

b = dist.Field(name='b', bases=bases)
a = dist.Field(name='a', bases=bases)
v = dist.Field(name='v', bases=bases)
psi = dist.Field(name='psi', bases=bases)

taub1 = dist.Field(name='taub1', bases=fbases)
taub2 = dist.Field(name='taub2', bases=fbases)

taua1 = dist.Field(name='taua1', bases=fbases)
taua2 = dist.Field(name='taua2', bases=fbases)

tauv1 = dist.Field(name='tauv1', bases=fbases)
tauv2 = dist.Field(name='tauv2', bases=fbases)

taupsi1 = dist.Field(name='taupsi1', bases=fbases)
taupsi2 = dist.Field(name='taupsi2', bases=fbases)

taulapsi1 = dist.Field(name='taulapsi1', bases=fbases)
taulapsi2 = dist.Field(name='taulapsi2', bases=fbases)

ey = dist.VectorField(coords, name='ey')
ez = dist.VectorField(coords, name='ez')
ex = dist.VectorField(coords, name='ex')

ez['g'][0] = 1
ey['g'][1] = 1
ex['g'][2] = 1

# Initial conditions
# noise = (x * (Lx - x) + 1e-4*rand.standard_normal(lshape)) * noise_coeff 
psi.fill_random()
psi.low_pass_filter(scales=0.125)
psi['g'] *= noise_coeff * x * (Lx - x)

v.fill_random()
v.low_pass_filter(scales=0.125)
v['g'] *= noise_coeff * x * (Lx - x)

a.fill_random()
a.low_pass_filter(scales=0.125)
a['g'] *= noise_coeff * x * (Lx - x)

b.fill_random()
b.low_pass_filter(scales=0.125)
b['g'] *= noise_coeff * x * (Lx - x)

dy = lambda A: d3.Differentiate(A, coords['y'])
dz = lambda A: d3.Differentiate(A, coords['z'])
dx = lambda A: d3.Differentiate(A, coords['x'])
J = lambda P, Q: (dx(P)*dz(Q) - dz(P)*dx(Q))

integ = lambda A: d3.Integrate(d3.Integrate(A, 'z'), 'x')
if not is2D:
    integ = lambda A: d3.Integrate(integ(A), 'y')

lift_basis = xbasis.derivative_basis(1) # First derivative basis
lift = lambda A: d3.Lift(A, lift_basis, -1)

grad_a = d3.grad(a) + ex*lift(taua1)
lapa = d3.div(grad_a) + lift(taua2)

grad_b = d3.grad(b) + ex*lift(taub1)
grad_v = d3.grad(v) + ex*lift(tauv1)
grad_psi = d3.grad(psi) + ex*lift(taupsi1)
lapsi = d3.div(grad_psi) + lift(taupsi2)

grad_lapsi = d3.grad(lapsi) + ex*lift(taulapsi1)
laplapsi = d3.div(grad_lapsi) + lift(taulapsi2)

vx = -dz(psi)
vy = v
vz = dx(psi)

bx = -dz(a)
by = b
bz = dx(a)

u = ex*vx + ey*vy + ez*vz
bvec = ex*bx + ey*by + ez*bz

B = 1
f = 2
S = 1*np.pi**2*B**2 / (f*Lx**2)

problem = d3.IVP([b, a, v, psi, taub1, taub2, taua1, taua2, tauv1, tauv2, taupsi1, taupsi2, taulapsi1, taulapsi2], namespace=locals())
    
problem.add_equation("dt(b) - B*dz(v) + S*dz(a) - eta*div(grad_b) + lift(taub2)= -J(psi, b) + J(a, v)")
problem.add_equation("dt(a) - B*dz(psi) - eta*lapa = -J(psi, a)")
problem.add_equation("dt(v) - (f + S)*dz(psi) - B*dz(b) - nu*div(grad_v) + lift(tauv2) = -J(psi, v) + J(a, b)")
problem.add_equation("dt(lapsi) + f*dz(v) - B*dz(lapa) - nu*laplapsi = -J(psi, lapsi) + J(a, lapa)")


problem.add_equation("psi(x='left') = 0")
problem.add_equation("psi(x='right') = 0")

problem.add_equation("lapsi(x='left') = 0")
problem.add_equation("lapsi(x='right') = 0")

problem.add_equation("b(x='left') = 0")
problem.add_equation("b(x='right') = 0")

problem.add_equation("a(x='left') = 0")
problem.add_equation("a(x='right') = 0")

problem.add_equation("v(x='left') = 0")
problem.add_equation("v(x='right') = 0")

# problem.add_equation("vx(x='left') = 0")
# problem.add_equation("vx(x='right') = 0")

# problem.add_equation("vy(x='left') = 0")
# problem.add_equation("vy(x='right') = 0")

# problem.add_equation("vz(x='left') = 0")
# problem.add_equation("vz(x='right') = 0")

# problem.add_equation("bx(x='left') = 0")
# problem.add_equation("bx(x='right') = 0")

# problem.add_equation("bz(x='left') = 0")
# problem.add_equation("bz(x='right') = 0")

# problem.add_equation("bz(x='left') = 0")
# problem.add_equation("bz(x='right') = 0")

# if (isNoSlip):
#     # no-slip BCs
#     problem.add_equation("u(x='left') = 0")
#     problem.add_equation("u(x='right') = 0")
# else:
#     # stress-free BCs
#     problem.add_equation("dot(u, ex)(x='left') = 0")
#     problem.add_equation("dot(u, ex)(x='right') = 0")
#     problem.add_equation("dot(dx(u), ey)(x='left') = 0")
#     problem.add_equation("dot(dx(u), ey)(x='right') = 0")
#     problem.add_equation("dot(dx(u), ez)(x='left') = 0")
#     problem.add_equation("dot(dx(u), ez)(x='right') = 0")

# # Pressure gauge
# problem.add_equation("integ(p) = 0") 

# problem.add_equation("dot(A, ey)(x='left') = 0")
# problem.add_equation("dot(A, ez)(x='left') = 0")
# problem.add_equation("dot(A, ey)(x='right') = 0")
# problem.add_equation("dot(A, ez)(x='right') = 0")
# problem.add_equation("phi(x='left') = 0")
# problem.add_equation("phi(x='right') = 0")

if solveEVP:
    solver = problem.build_solver(entry_cutoff=0)
    solver.solve_sparse(solver.subproblems[1], NEV, target=0)
    print('kz = {}, omega = {}'.format(kz, np.max(solver.eigenvalues.imag)))
    sys.exit()
   

# Solver
solver = problem.build_solver(timestepper)
solver.stop_sim_time = stop_sim_time

# sys.exit()
# A['g'][0], A['g'][1], A['g'][2] = vp_bvp_func(by, bz, bx)

# fh_mode = 'overwrite'

# checkpoint = solver.evaluator.add_file_handler(suffix + '/checkpoint', max_writes=1, sim_dt=checkpoint_dt)
# checkpoint.add_tasks(solver.state, layout='g')

# slicepoints = solver.evaluator.add_file_handler(suffix + '/slicepoints', sim_dt=slicepoint_dt, max_writes=50, mode=fh_mode)

# for field, field_name in [(b, 'b'), (u, 'v')]:
#     for d2, unit_vec in zip(('x', 'y', 'z'), (ex, ey, ez)):
#         slicepoints.add_task(d3.dot(field, unit_vec)(x = 'center'), name = "{}{}_mid{}".format(field_name, d2, 'x'))
#         slicepoints.add_task(d3.dot(field, unit_vec)(z = 'center'), name = "{}{}_mid{}".format(field_name, d2, 'z'))
#         if not is2D:
#             slicepoints.add_task(d3.dot(field, unit_vec)(y = 'center'), name = "{}{}_mid{}".format(field_name, d2, 'y'))

# scalars = solver.evaluator.add_file_handler(suffix + '/scalars', sim_dt=scalar_dt, max_writes=50, mode=fh_mode)
# scalars.add_task(integ(np.sqrt(d3.dot(u, u)) + np.sqrt(d3.dot(b, b))), name='x_l2')
# scalars.add_task(integ(np.sqrt(d3.dot(u, u))), name='u_l2')
# scalars.add_task(integ(np.sqrt(d3.dot(b, b))), name='b_l2')

CFL = d3.CFL(solver, initial_dt=init_timestep, cadence=10, safety=0.3, threshold=0.05,
             max_change=1.5, min_change=0.5, max_dt=init_timestep)
CFL.add_velocity(u)
# CFL.add_velocity(b)

# Flow properties
flow = d3.GlobalFlowProperty(solver, cadence=1)
flow.add_property(np.sqrt(d3.dot(u, u)) + np.sqrt(d3.dot(bvec, bvec)), name='x_l2')
flow.add_property(np.sqrt(d3.dot(u, u)), name='u_l2')
flow.add_property(np.sqrt(d3.dot(bvec, bvec)), name='b_l2')

# # Main loop
# # print(flow.properties.tasks)
# solver.evaluator.evaluate_handlers((flow.properties, ))
# # metrics = 'Iteration, Time, dt, x_l2, u_l2, b_l2'
# # with open(suffix + "/data.csv", "w") as file:
    # file.write(metrics + "\n")
timestep = init_timestep
try:
    logger.info('Starting main loop')
    while solver.proceed:

        timestep = CFL.compute_timestep()
        if (solver.iteration-1) % 10 == 0:
            x_l2 = flow.volume_integral('x_l2')
            u_l2 = flow.volume_integral('u_l2')
            b_l2 = flow.volume_integral('b_l2')
            # status = 'Iteration=%i, Time=%e, dt=%e, x_l2%f, u_l2%f, b_l2=%f' %(solver.iteration, solver.sim_time, timestep, x_l2, u_l2, b_l2)
            nums = [solver.iteration, solver.sim_time, timestep, x_l2, u_l2, b_l2]
            status = 'Iteration={}, Time={}, dt={}, x_l2={}, u_l2={}, b_l2={}'.format(*nums)
            logger.info(status)
            # if CW.rank == 0:
            #     with open(suffix + "/status.txt", "a") as file:
            #         file.write(status + "\n")
            # with open(suffix + "/data.csv", "a") as file:
            #     file.write(str(nums)[1:-1] + "\n")

        solver.step(timestep)
except:
    logger.error('Exception raised, triggering end of main loop.')
    raise


# finally:
    # solver.log_stats()
