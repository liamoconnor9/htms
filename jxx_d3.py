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

coords = d3.CartesianCoordinates('z', 'x')
dtype = np.complex128
dist = d3.Distributor(coords, dtype=dtype)

# Bases
Lz, Lx = eval(str(Lz)), eval(str(Lx))
Nz = 2
# zbasis = d3.ComplexFourier(coords['z'], size=Nz, bounds=(0, Lz))
xbasis = d3.ChebyshevT(coords['x'], size=Nx, bounds=(-Lx / 2.0, Lx / 2.0))
bases = (xbasis, )
fbases = None

# z = dist.local_grid(zbasis)
x = dist.local_grid(xbasis)

# nccs
eta = nu / Pm


# Fields
omega = dist.Field(name='omega')
dt = lambda A: -1j*omega*A

p = dist.Field(name='p', bases=bases)

vx = dist.Field(name='vx', bases=bases)
vy = dist.Field(name='vy', bases=bases)
vz = dist.Field(name='vz', bases=bases)

wy = dist.Field(name='wy', bases=bases)
wz = dist.Field(name='wz', bases=bases)

bx = dist.Field(name='bx', bases=bases)
by = dist.Field(name='by', bases=bases)
bz = dist.Field(name='bz', bases=bases)

jxx = dist.Field(name='jxx', bases=bases)
xf = dist.Field(name='xf', bases=bases)
xf['g'] = x

tau1 = dist.Field(name='tau1', bases=fbases)
tau2 = dist.Field(name='tau2', bases=fbases)
tau3 = dist.Field(name='tau3', bases=fbases)
tau4 = dist.Field(name='tau4', bases=fbases)
tau5 = dist.Field(name='tau5', bases=fbases)
tau6 = dist.Field(name='tau6', bases=fbases)
tau7 = dist.Field(name='tau7', bases=fbases)
tau8 = dist.Field(name='tau8', bases=fbases)
tau9 = dist.Field(name='tau9', bases=fbases)
tau10 = dist.Field(name='tau10', bases=fbases)

ez = dist.VectorField(coords, name='ez')
ex = dist.VectorField(coords, name='ex')

ez['g'][0] = 1
ex['g'][1] = 1


dx = lambda A: d3.Differentiate(A, coords['x'])
dy = lambda A: 1j*ky*A
dz = lambda A: 1j*kz*A
Dt = lambda A: -1j*omega*A + S*xf*dy(A)

integ = lambda A: d3.Integrate(d3.Integrate(A, 'z'), 'x')

lift_basis = xbasis.derivative_basis(1) # First derivative basis
lift = lambda A: d3.Lift(A, lift_basis, -1)

wx = dy(vz) - dz(vy)
jx = dy(bz) - dz(by)
jy = dz(bx) - dx(bz)
jz = dx(by) - dy(bx)

B = 1
f = 2
S = -1

fx = 0
fz = f

problem = d3.EVP([p, vx, vy, vz, wy, wz, bx, by, bz, jxx, tau1, tau2, tau3, tau4, tau5, tau6, tau7, tau8, tau9, tau10], namespace=locals(), eigenvalue=omega)

# Hydro equations: p, vx, vy, vz, wy, wz

problem.add_equation("dx(vx) + dy(vy) + dz(vz) + lift(tau1) = 0")
problem.add_equation("Dt(vx) - fz*vy   + dx(p) - B*dz(bx) + nu*(dy(wz) - dz(wy)) + lift(tau2) = 0")
problem.add_equation("Dt(vy) + (fz+S)*vx - fx*vz + dy(p) - B*dz(by) + nu*(dz(wx) - dx(wz)) + lift(tau3) = 0")
problem.add_equation("Dt(vz)    + fx*vy + dz(p) - B*dz(bz) + nu*(dx(wy) - dy(wx)) + lift(tau4) = 0")

problem.add_equation("wy - dz(vx) + dx(vz) + lift(tau5) = 0")
problem.add_equation("wz - dx(vy) + dy(vx) + lift(tau6) = 0")

# MHD equations: bx, by, bz, jxx

problem.add_equation("dx(bx) + dy(by) + dz(bz) + lift(tau7) = 0")

problem.add_equation("Dt(bx) - B*dz(vx)   + eta*( dy(jz) - dz(jy) ) + lift(tau8) = 0")
problem.add_equation("Dt(jx) - B*dz(wx) + S*dz(bx) - eta*( dx(jxx) + dy(dy(jx)) + dz(dz(jx)) ) + lift(tau9) = 0")

problem.add_equation("jxx - dx(jx) + lift(tau10) = 0")

# Boundary Conditions: stress-free, perfect-conductor

problem.add_equation("vx(x='left') = 0")
problem.add_equation("wy(x='left') = 0")
problem.add_equation("wz(x='left') = 0")
problem.add_equation("bx(x='left') = 0")
problem.add_equation("jxx(x='left') = 0")
problem.add_equation("vx(x='right') = 0")
problem.add_equation("wy(x='right') = 0")
problem.add_equation("wz(x='right') = 0")
problem.add_equation("bx(x='right') = 0")
problem.add_equation("jxx(x='right') = 0")
solver = problem.build_solver(entry_cutoff=0)

solver.solve_sparse(solver.subproblems[0], NEV, target=0.0)
# solver.solve_dense(solver.subproblems[1])
# print(.tolist())
maxomega=-1e10
for val in solver.eigenvalues.imag:
   if val > 1e6:
     continue
   elif maxomega < val:
     maxomega = val

print('kz = {}, omega = {}'.format(kz, maxomega))
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
#   for d2, unit_vec in zip(('x', 'y', 'z'), (ex, ey, ez)):
#      slicepoints.add_task(d3.dot(field, unit_vec)(x = 'center'), name = "{}{}_mid{}".format(field_name, d2, 'x'))
#      slicepoints.add_task(d3.dot(field, unit_vec)(z = 'center'), name = "{}{}_mid{}".format(field_name, d2, 'z'))
#      if not is2D:
#         slicepoints.add_task(d3.dot(field, unit_vec)(y = 'center'), name = "{}{}_mid{}".format(field_name, d2, 'y'))

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
        #   with open(suffix + "/status.txt", "a") as file:
        #      file.write(status + "\n")
          # with open(suffix + "/data.csv", "a") as file:
            #     file.write(str(nums)[1:-1] + "\n")

        solver.step(timestep)
except:
    logger.error('Exception raised, triggering end of main loop.')
    raise


# finally:
    # solver.log_stats()
