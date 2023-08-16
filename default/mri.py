"""
3D cartesian MRI initial value problem using vector potential formulation
Usage:
    mri_vp.py <config_file> <dir> <run_suffix>
"""
from unicodedata import decimal
from docopt import docopt
import time
from configparser import ConfigParser
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

def vp_bvp_func(by, bz, bx, Ay, Ax, Az, phi, coords):

    # problem.parameters['by'] = by
    # problem.parameters['bz'] = bz
    # problem.parameters['bx'] = bx

    dy = lambda A: d3.Differentiate(A, coords['y'])
    dz = lambda A: d3.Differentiate(A, coords['z'])
    dx = lambda A: d3.Differentiate(A, coords['x'])

    problem = d3.LBVP(variables=[Ax, Ay, Az, phi], namespace=locals())

    problem.add_equation("dx(Ax) + dy(Ay) + dz(Az) = 0")
    problem.add_equation("dy(Az) - dz(Ay) + dx(phi) = bx")
    problem.add_equation("dz(Ax) - dx(Az) + dy(phi) = by")
    problem.add_equation("dx(Ay) - dy(Ax) + dz(phi) = bz")

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

    # Plot solution
    Ay = solver.state['Ay']
    Az = solver.state['Az']
    Ax = solver.state['Ax']
    # phi = solver.state['phi']

    return Ay['g'].copy(), Az['g'].copy(), Ax['g'].copy()
    
args = docopt(__doc__)
filename = Path(args['<config_file>'])
script_dir = args['<dir>']
run_suffix = args['<run_suffix>']

config = ConfigParser()
config.read(str(filename))

logger.info('Running mri_vp.py with the following parameters:')
param_str = config.items('parameters')
logger.info(param_str)
with open("status.txt", "w") as file:
    file.write(str(param_str) + "\n")
restart = config.getboolean('parameters','restart')

Ny = config.getint('parameters','Ny')
Ly = eval(config.get('parameters','Ly'))

Nz = config.getint('parameters','Nz')
Lz = eval(config.get('parameters','Lz'))

Nx = config.getint('parameters','Nx')
Lx = eval(config.get('parameters','Lx'))

B = config.getfloat('parameters','B')

R      =  config.getfloat('parameters','R')
q      =  config.getfloat('parameters','q')

nu = config.getfloat('parameters','nu')
Pm = config.getfloat('parameters','Pm')
eta = nu / Pm

S      = -R*np.sqrt(q)
f      =  R/np.sqrt(q)

tau = config.getfloat('parameters','tau')
isNoSlip = config.getboolean('parameters','isNoSlip')

ary = Ly / Lx
arz = Lz / Lx

# Evolution params
init_timestep = config.getfloat('parameters', 'dt')
stop_sim_time = config.getfloat('parameters', 'stop_sim_time')
wall_time = 60. * 60. * config.getfloat('parameters', 'wall_time_hr')
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
# tauphi = dist.Field(name='tauphi')

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
noise_coeff = 1e-5

rand = np.random.RandomState(seed=23 + CW.rank)
noise = x * (Lx - x) * noise_coeff * rand.standard_normal(lshape)
u['g'][0] = noise

rand = np.random.RandomState(seed=23 + CW.rank)
noise = noise_coeff * rand.standard_normal(lshape)
u['g'][1] = noise

rand = np.random.RandomState(seed=23 + CW.rank)
noise = noise_coeff * rand.standard_normal(lshape)
u['g'][2] = noise

# bz['g'] = 1.0

# Ay = (A@ey).evaluate()
# Az = (A@ez).evaluate()
# Ax = (A@ex).evaluate()

# Ay['g'], Az['g'], Ax['g'] = vp_bvp_func(by, bz, bx, Ay, Ax, Az, phi, coords)

A.change_scales(1)
A['g'][0] = 1e1*np.cos(x*np.pi / Lx)

dy = lambda A: d3.Differentiate(A, coords['y'])
dz = lambda A: d3.Differentiate(A, coords['z'])
dx = lambda A: d3.Differentiate(A, coords['x'])

integ = lambda A: d3.Integrate(d3.Integrate(d3.Integrate(A, 'y'), 'z'), 'x')

lift_basis = xbasis.derivative_basis(1) # First derivative basis
lift = lambda A: d3.Lift(A, lift_basis, -1)
grad_u = d3.grad(u) + ex*lift(tau1u) # First-order reduction
grad_A = d3.grad(A) + ex*lift(tau1A) # First-order reduction
grad_b = d3.grad(b)

# Problem
# First-order form: "div(f)" becomes "trace(grad_f)"
# First-order form: "lap(f)" becomes "div(grad_f)"
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

checkpoint = solver.evaluator.add_file_handler('checkpoint_{}'.format(run_suffix), max_writes=1, sim_dt=10.0)
checkpoint.add_tasks(solver.state, layout='g')

slicepoints = solver.evaluator.add_file_handler('slicepoints_' + run_suffix, sim_dt=0.1, max_writes=50, mode=fh_mode)

for field, field_name in [(b, 'b'), (u, 'v')]:
    for d2, unit_vec in zip(('x', 'y', 'z'), (ex, ey, ez)):
        slicepoints.add_task(d3.dot(field, unit_vec)(x = 'center'), name = "{}{}_mid{}".format(field_name, d2, 'x'))
        slicepoints.add_task(d3.dot(field, unit_vec)(y = 'center'), name = "{}{}_mid{}".format(field_name, d2, 'y'))
        slicepoints.add_task(d3.dot(field, unit_vec)(z = 'center'), name = "{}{}_mid{}".format(field_name, d2, 'z'))

CFL = d3.CFL(solver, initial_dt=init_timestep, cadence=10, safety=0.3, threshold=0.05,
             max_change=1.5, min_change=0.5)
CFL.add_velocity(u)
CFL.add_velocity(b)

# Flow properties
flow = d3.GlobalFlowProperty(solver, cadence=1)
flow.add_property(np.sqrt(d3.dot(u, u)) + np.sqrt(d3.dot(b, b)), name='x_l2')
flow.add_property(np.sqrt(d3.dot(u, u)), name='u_l2')
flow.add_property(np.sqrt(d3.dot(b, b)), name='b_l2')

# Main loop
# print(flow.properties.tasks)
solver.evaluator.evaluate_handlers((flow.properties, ))
metrics = 'Iteration, Time, dt, x_l2, u_l2, b_l2'
with open("data.csv", "w") as file:
    file.write(metrics + "\n")

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
            with open("status.txt", "a") as file:
                file.write(status + "\n")
            with open("data.csv", "a") as file:
                file.write(str(nums)[1:-1] + "\n")
            # with open('flow.pick', 'wb') as file:
            #     pickle.dump(flow, file)

        solver.step(timestep)
except:
    logger.error('Exception raised, triggering end of main loop.')
    raise


# finally:
    # solver.log_stats()
