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

param_str = config.items('parameters')
logger.info(param_str)

coords = d3.CartesianCoordinates('z', 'x')
dtype = np.float64
dealias = 3/2
dist = d3.Distributor(coords, dtype=dtype)

# Bases
Lz, Lx = eval(str(Lz)), eval(str(Lx))
zbasis = d3.RealFourier(coords['z'], size=Nz, bounds=(0, Lz), dealias=dealias)
xbasis = d3.ChebyshevT(coords['x'], size=Nx, bounds=(-Lx / 2.0, Lx / 2.0), dealias=dealias)
bases = (xbasis, zbasis)
fbases = (zbasis, )

z = dist.local_grid(zbasis)
x = dist.local_grid(xbasis)

# nccs
eta = nu / Pm

# Fields

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

ez = dist.VectorField(coords, name='ez')
ex = dist.VectorField(coords, name='ex')

ez['g'][0] = 1
ex['g'][1] = 1


dz = lambda A: d3.Differentiate(A, coords['z'])
# dz = lambda A: 0*A
dx = lambda A: d3.Differentiate(A, coords['x'])
J = lambda P, Q: 1*(dx(P)*dz(Q) - dz(P)*dx(Q))

integ = lambda A: d3.Integrate(d3.Integrate(A, 'z'), 'x')

lift_basis = xbasis.derivative_basis(1) # First derivative basis
lift = lambda A: d3.Lift(A, lift_basis, -1)

grad_a = d3.grad(a) + ex*lift(taua1)
lapa = d3.div(grad_a) + lift(taua2)

grad_b = d3.grad(b) + ex*lift(taub1)
grad_v = d3.grad(v) + ex*lift(tauv1)
grad_psi = d3.grad(psi) + ex*lift(taupsi1)

lapb = dz(dz(b)) + dx(dx(b) + lift(taub1)) + lift(taub2)
lapa = dz(dz(a)) + dx(dx(a) + lift(taua1)) + lift(taua2)
lapv = dz(dz(v)) + dx(dx(v) + lift(tauv1)) + lift(tauv2)
lapsi = dz(dz(psi)) + dx(dx(psi) + lift(taupsi1)) + lift(taupsi2)

grad_lapsi = d3.grad(lapsi) + ex*lift(taulapsi1)
laplapsi = d3.div(grad_lapsi) + lift(taulapsi2)

vx = -dz(psi)
vy = v
vz = dx(psi)
u = ex*vx + ez*vz

bx = -dz(a)
by = b
bz = dx(a)
bvec = ex*bx + ez*bz

B = 1
f = 2
S = -1

problem = d3.IVP([b, a, v, psi, taub1, taub2, taua1, taua2, tauv1, tauv2, taupsi1, taupsi2, taulapsi1, taulapsi2], namespace=locals())
    
problem.add_equation("dt(b) - B*dz(v) + S*dz(a) - eta*lapb= J(a, v) - J(psi, b)")
problem.add_equation("dt(a) - B*dz(psi) - eta*lapa = -J(psi, a)")
problem.add_equation("dt(v) - (f + S)*dz(psi) - B*dz(b) - nu*lapv = J(a, b) - J(psi, v)")
problem.add_equation("dt(lapsi) + f*dz(v) - B*dz(lapa) - nu*laplapsi = J(a, lapa) - J(psi, lapsi)")

problem.add_equation("dx(vy)(x='left') = 0")
problem.add_equation("dx(vy)(x='right') = 0")

problem.add_equation("dx(vz)(x='left') = 0")
problem.add_equation("dx(vz)(x='right') = 0")

# impenetrable (flow)
problem.add_equation("psi(x='left') = 0")
problem.add_equation("psi(x='right') = 0")

# problem.add_equation("bx(x='left') = 0", condition="nz != 0")
# problem.add_equation("bx(x='right') = 0", condition="nz != 0")

# no magnetic flux into or out of boundaries
problem.add_equation("a(x='left') = 0")
problem.add_equation("a(x='right') = 0")

# x-derivative of x-component of current density = 0
# problem.add_equation("dx((dy(bz) - dz(by)))(x='left') = 0")
# problem.add_equation("dx((dy(bz) - dz(by)))(x='right') = 0")
problem.add_equation("dx((b))(x='left') = 0")
problem.add_equation("dx((b))(x='right') = 0")

   
# Solver
solver = problem.build_solver(timestepper)
solver.stop_sim_time = stop_sim_time



psi.fill_random(seed=1)
psi.low_pass_filter(scales=0.25)
psi['g'] += np.sin(x) * np.cos(z*2*np.pi / Lz)
psi['g'] *= noise_coeff
# psi['g'] += np.cos(z*2*np.pi / Lz)

a.fill_random(seed=1)
a.low_pass_filter(scales=0.25)
a['g'] += np.sin(x) * np.cos(z*2*np.pi / Lz)
a['g'] *= noise_coeff
# a['g'] += np.cos(z*2*np.pi / Lz)

# sys.exit()
# A['g'][0], A['g'][1], A['g'][2] = vp_bvp_func(by, bz, bx)

fh_mode = 'overwrite'

# checkpoint = solver.evaluator.add_file_handler(suffix + '/checkpoint', max_writes=1, sim_dt=checkpoint_dt)
# checkpoint.add_tasks(solver.state, layout='g')

# slicepoints = solver.evaluator.add_file_handler(suffix + '/slicepoints', sim_dt=slicepoint_dt, max_writes=50, mode=fh_mode)

# for field, field_name in [(b, 'b'), (u, 'v')]:
#     for d2, unit_vec in zip(('x', 'y', 'z'), (ex, ey, ez)):
#         slicepoints.add_task(d3.dot(field, unit_vec)(x = 'center'), name = "{}{}_mid{}".format(field_name, d2, 'x'))
#         slicepoints.add_task(d3.dot(field, unit_vec)(z = 'center'), name = "{}{}_mid{}".format(field_name, d2, 'z'))
#         if not is2D:
#             slicepoints.add_task(d3.dot(field, unit_vec)(y = 'center'), name = "{}{}_mid{}".format(field_name, d2, 'y'))


CFL = d3.CFL(solver, initial_dt=init_timestep, cadence=10, safety=0.1, threshold=0.01,
             max_change=2, min_change=0.2, max_dt=init_timestep)
CFL.add_velocity(u)
CFL.add_velocity(bvec)

# Flow properties
flow = d3.GlobalFlowProperty(solver, cadence=1)

ke = np.sqrt(vz**2 + vx**2 + vy**2)
be = np.sqrt(bz**2 + bx**2 + by**2)
flow.add_property(ke + be, name='x_l2')
flow.add_property(ke, name='u_l2')
flow.add_property(be, name='b_l2')

scalars = solver.evaluator.add_file_handler(suffix + '/scalars', sim_dt=scalar_dt, max_writes=50, mode=fh_mode)
scalars.add_task(integ(ke + be), name='x_l2')
scalars.add_task(integ(ke), name='u_l2')
scalars.add_task(integ(be), name='b_l2')

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
        # timestep = 1
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
