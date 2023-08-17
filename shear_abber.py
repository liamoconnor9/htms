"""
Usage:
    shear_abber.py <config_file>
    shear_abber.py <config_file> <SBI_config>
"""
from distutils.command.bdist import show_formats
import os
from ast import For
from contextlib import nullcontext
from turtle import backward
import numpy as np
import pickle
path = os.path.dirname(os.path.abspath(__file__))
import sys
sys.path.append(path + "/..")
from OptimizationContext import OptimizationContext
from Tracker import Tracker
from Euler import Euler
from JacLayout import JacLayout
from Plot2D import Plot2D
import h5py
import gc
import dedalus.public as d3
from dedalus.core.domain import Domain
from mpi4py import MPI
CW = MPI.COMM_WORLD
import logging
import pathlib
logger = logging.getLogger(__name__)
import ForwardShear
import BackwardShear
import matplotlib.pyplot as plt
from docopt import docopt
from pathlib import Path
from configparser import ConfigParser
from scipy import optimize
from datetime import datetime
from dedalus.core import domain
from collections import OrderedDict
from clean_div import clean_div
from dedalus.core.future import FutureField

from ConfigEval import ConfigEval
try:
    args = docopt(__doc__)
    filename = Path(args['<config_file>'])
except:
    filename = path + '/default.cfg'
config = ConfigEval(filename)
locals().update(config.execute_locals())
logger.info('objective_overwrite = {}'.format(objective_overwrite))
if (not os.path.isdir(path + '/' + suffix)):
    logger.error('target state not found')
    sys.exit()

doSBI = False
if (args['<SBI_config>'] != None):
    logger.info('SBI config provided. Overwriting default config for simple backward integration...')
    SBI_config = Path(args['<SBI_config>'])
    sbi_dict = config.SBI_dictionary(SBI_config)
    doSBI = True
    logger.info('Localizing SBI settings: {}'.format(sbi_dict))
    locals().update(sbi_dict)
logger.info('doSBI = {}'.format(doSBI))

# Simulation Parameters

dealias = 3/2
coords = d3.CartesianCoordinates('x', 'z')
dist = d3.Distributor(coords, dtype=np.float64)
coords.name = coords.names

xbasis = d3.RealFourier(coords['x'], size=Nx, bounds=(0, Lx), dealias=dealias)
zbasis = d3.RealFourier(coords['z'], size=Nz, bounds=(-Lz/2, Lz/2), dealias=dealias)
bases = [xbasis, zbasis]
x, z = dist.local_grids(bases[0], bases[1])
ex, ez = coords.unit_vector_fields(dist)

domain = domain.Domain(dist, bases)
dist = domain.dist
slices = dist.layout_references[opt_layout].slices(domain, scales=1)
slices_grid = dist.grid_layout.slices(domain, scales=1)

forward_problem = ForwardShear.build_problem(domain, coords, Reynolds)
backward_problem = BackwardShear.build_problem(domain, coords, Reynolds, abber)

# forward, and corresponding adjoint variables (fields)
u = forward_problem.variables[0]
s = forward_problem.variables[1]

u_t = backward_problem.variables[0]
s_t = backward_problem.variables[1]
obj_t = backward_problem.variables[3]

lagrangian_dict = {u : u_t, s : s_t}

forward_solver = forward_problem.build_solver(forward_timestepper)
backward_solver = backward_problem.build_solver(backward_timestepper)

Features = (Tracker,)
if (method == "euler"):
    Features = (Euler,) + Features
Features += (JacLayout,)
if (show):
    Features += (Plot2D,)
logger.info('features = {}'.format(Features))

class Optimization2D(*Features):

    def write_txt(self, tag='', scales=1):
        self.ic['u'].change_scales(scales)
        approx = self.ic['u'].allgather_data(layout=self.dist_layout).flatten().copy()
        savedir = path + '/' + suffix + '/checkpoints/{}write{:06}.txt'.format(tag, self.loop_index)
        if (CW.rank == 0):
            np.savetxt(savedir, approx)

        # logger.info(savedir)

    def checkpoint(self):
        checkpoints = self.forward_solver.evaluator.add_file_handler(self.run_dir + '/' + self.suffix + '/checkpoints/checkpoint_loop'  + str(self.loop_index), max_writes=1, sim_dt=self.T, mode='overwrite')
        checkpoints.add_tasks(self.forward_solver.state, layout='g')

    def reshape_soln(self, x, slices=None, scales=None):
        
        if (slices == None):
            slices = self.slices
        if (scales == None):
            scales = self.opt_scales

        if self.opt_layout == 'c':
            return x.reshape((2,) + self.domain.coeff_shape)[:, slices[0], slices[1]]
        else:
            return x.reshape((2,) + self.domain.grid_shape(scales=scales))[:, slices[0], slices[1]]

    def loop_message(self):
        # loop_message = 'loop index = {}; '.format(self.loop_index)
        # loop_message += 'Rsqrd = {}; '.format(self.tracker['Rsqrd'][-1])
        # loop_message += 'objective = {}; '.format(self.objective_norm)
        # loop_message += 'objectivet = {}; '.format(self.tracker['objectivet'])
        # loop_message += 'objectiveT = {}; '.format(self.objectiveT_norm)
        # loop_message += 'objectivet = {}; '.format(self.objectivet_norm)
        # loop_message += 'obj_approx = {}; '.format(self.tracker['obj_approx'][-1])
        # loop_message += 's_error = {}; '.format(self.tracker['s_error'][-1])
        # loop_message += 'omega_error = {}; '.format(self.tracker['omega_error'][-1])
        # loop_message += 'time = {}; '.format(self.tracker['time'][-1])
        loop_message = ""
        for metname in self.tracker.keys():
            loop_message += '{} = {}; '.format(metname, self.tracker[metname][-1])
        # loop_message += 'obj_approx = {}; '.format(self.tracker['obj_approx'][-1])
        for metric_name in self.metricsT_norms.keys():
            loop_message += '{} = {}; '.format(metric_name, self.metricsT_norms[metric_name])
        for metric_name in self.metrics0_norms.keys():
            loop_message += '{} = {}; '.format(metric_name, self.metrics0_norms[metric_name])
        logger.info(loop_message)
        return loop_message

    def before_fullforward_solve(self):
        if ('benchmark' in suffix):
            global Nz
            self.write_txt(tag='N{}_'.format(Nz), scales=1)
            if (Nz == 1024):
                self.ic['u'].change_scales(0.5)
                self.write_txt(tag='N512_')
                self.ic['u'].change_scales(0.25)
                self.write_txt(tag='N256_')
                self.ic['u'].change_scales(1)
            elif (Nz == 512):
                self.ic['u'].change_scales(0.5)
                self.write_txt(tag='N256_')
                self.ic['u'].change_scales(1)

        else:
            self.write_txt()

            
        for Feature in Features:
            Feature.before_fullforward_solve(self)

        # if self.add_handlers and self.loop_index % self.handler_loop_cadence == 0:
            # checkpoints = self.forward_solver.evaluator.add_file_handler(self.run_dir + '/' + self.suffix + '/checkpoints/checkpoint_loop'  + str(self.loop_index), max_writes=1, sim_dt=self.T, mode='overwrite')
            # checkpoints.add_tasks(self.forward_solver.state, layout='g')
            # ex, ez = self.coords.unit_vector_fields(self.domain.dist)
        # logger.info('before forward handlers = {}'.format(self.backward_solver.evaluator.handlers[0].__dict__))

    def during_fullforward_solve(self):
        for Feature in Features:
            Feature.during_fullforward_solve(self)

    def after_fullforward_solve(self):
        for Feature in Features:
            Feature.after_fullforward_solve(self)

    def before_backward_solve(self):
        for Feature in Features:
            Feature.before_backward_solve(self)

    def during_backward_solve(self):
        for Feature in Features:
            Feature.during_backward_solve(self)

    def after_backward_solve(self):
        # Tracker.after_backward_solve(self)
        for Feature in Features:
            Feature.after_backward_solve(self)
        if (CW.rank == 0  or self.domain.dist.comm == MPI.COMM_SELF):
            msg = self.loop_message()
            with open(self.run_dir + '/' + self.suffix + '/output.txt', 'a') as f:
                f.write(msg)
                f.write('\n')

opt = Optimization2D(domain, coords, forward_solver, backward_solver, lagrangian_dict, None, suffix)
opt.set_time_domain(T, num_cp, timestep)
opt.opt_iters = opt_iters
opt.opt_layout = opt_layout
opt.opt_scales = opt_scales
opt.add_handlers = add_handlers
opt.handler_loop_cadence = handler_loop_cadence

opt.init_layout(u)

opt._obj_coeff = obj_coeff
opt._jac_coeff = jac_coeff

logger.info("opt._obj_coeff = {}".format(opt._obj_coeff))
logger.info("opt.obj_coeff() = {}".format(opt.obj_coeff()))

logger.info("opt._jac_coeff = {}".format(opt._jac_coeff))
logger.info("opt.jac_coeff() = {}".format(opt.jac_coeff()))

# opt.checkpoint()

opt.dist_layout = dist.layout_references[opt_layout]
opt.slices = opt.dist_layout.slices(domain, scales=opt_scales)
# opt.slices_coeff = dist.coeff_layout.slices(domain)
opt.show = show
opt.show_loop_cadence = show_loop_cadence
opt.show_iter_cadence = show_iter_cadence

# Populate U with end state of known initial condition
U = dist.VectorField(coords, name='U', bases=bases)
S = dist.Field(name='S', bases=bases)

objt_ic = dist.Field(name='objt_ic', bases=bases)
objt_ic['g'] = 0.0
end_state_path = path + '/' + suffix + '/checkpoint_target/checkpoint_target_s1.h5'
with h5py.File(end_state_path) as f:
    U['g'] = f['tasks/u'][-1, :, :][:, slices_grid[0], slices_grid[1]]
    S['g'] = f['tasks/s'][-1, :, :][slices_grid[0], slices_grid[1]]
    logger.info('loading target {}: t = {}'.format(end_state_path, f['scales/sim_time'][-1]))

dx = lambda A: d3.Differentiate(A, coords['x'])
dz = lambda A: d3.Differentiate(A, coords['z'])
opt.dx = dx
opt.dz = dz
ux = u @ ex
uz = u @ ez
w = dx(uz) - dz(ux)
Ux = U @ ex
Uz = U @ ez
W = (dx(Uz) - dz(Ux)).evaluate()

try:
    objectiveT = u_weight*0.5*d3.dot(u - U, u - U)
    logger.info('using ic gain u_weight: {}'.format(u_weight))
except:
    objectiveT = 0.5*d3.dot(u - U, u - U)
try:
    opt.set_objectiveT(objectiveT)
    logger.info('set_objectiveT succeeded!!')
    if (abber != 0):
        opt.set_objectivet(obj_t)
        logger.info('set_objectivet succeeded!!')
    opt.backward_ic['obj_t'] = objt_ic
except:
    logger.info('set_objectiveT failed')
    opt.objectiveT = objectiveT
    opt.backward_ic = OrderedDict()
    opt.backward_ic['u_t'] = -(u - U)
if (abber > 0.0):
    logger.info('objectivet not implemented')

opt.skew_gradgrad = False
opt.W = W
if (omega_weight == 1):
    logger.info('overwriting objective and adjoint initial condition! using vorticity objective...')
    opt.objectiveT = 0.5*omega_weight*(w - W)**2
    opt.backward_ic['u_t'] = omega_weight*d3.skew(d3.grad((w - W)))
elif (omega_weight == -1):
    logger.info('applying omega objective with SBI at initial time...')
    logger.info('opt.skew_gradgrad = True')
    opt.skew_gradgrad = True

# 'bar' quantities refer to the target initial condition (ubar is the minimizer we want to approximate)
sbar = dist.Field(name='sbar', bases=bases)
ubar = dist.VectorField(coords, name='ubar', bases=bases)
ubar['g'][0] = 1/2 + 1/2 * (np.tanh((z-0.5)/0.1) - np.tanh((z+0.5)/0.1))
sbar['g'] = ubar['g'][0]
ubar['g'][1] += 0.1 * np.sin(2*np.pi*x/Lx) * np.exp(-(z-0.5)**2/0.01)
ubar['g'][1] += 0.1 * np.sin(2*np.pi*x/Lx) * np.exp(-(z+0.5)**2/0.01)
uc_data = clean_div(domain, coords, ubar['g'].copy())
ubar.change_scales(1)
ubar['g'] = uc_data.copy()
wbar = opt.wbar = (dx(ubar @ ez) - dz(ubar @ ex)).evaluate()


# assume we are given the initial tracer distribution
opt.ic['s']['g'] = sbar['g'].copy()
tracker_dir = path + '/' + suffix + '/tracker.pick'

# # cadence = opt_iters - 1
cadence = show_loop_cadence
if (load_state):
    try:
        opt.load_tracker(tracker_dir, 1, cadence)
        logger.info('SUCCESSFULLY LOADED TRACKER AT DIR {}'.format(tracker_dir))
    except:
        logger.info('FAILED LOADING TRACKER AT DIR {}'.format(tracker_dir))
        logger.info("(setting load_state = False)")
        logger.info('BUILDING TRACKER AT DIR {}'.format(tracker_dir))
        load_state = False
        opt.build_tracker(tracker_dir, cadence)
else:
    logger.info('BUILDING TRACKER AT DIR {}'.format(tracker_dir))
    opt.build_tracker(tracker_dir, cadence)

# opt.add_metric('x_lst', True, 10, opt.ic['u'])
# opt.add_metric('objT_lst', False, 1, opt.objectiveT)
# opt.add_metric('objt_lst', False, 1, opt.objectiveT)
# opt.add_metric('obj_lst', False, 1, opt.objective)

opt.add_metric('Rsqrd', True, 1, 0.5*d3.dot(opt.ic['u'] - ubar, opt.ic['u'] - ubar), integrate=True)
opt.add_metric('omega_0', True, 1, 0.5*((dx(opt.ic['u'] @ ez) - dz(opt.ic['u'] @ ex)) - wbar)**2, integrate=True)
opt.add_metric('objective', False, 1, opt.objective, integrate=True)
opt.add_metric('objectiveT', False, 1, opt.objectiveT, integrate=True)
opt.add_metric('objectivet', False, 1, opt.objectivet, integrate=bool(abber))
opt.add_metric('obj_approx', False, 1, 0.5*d3.dot(u_t, u_t), integrate=True)

proj_num = d3.Integrate(0.5*d3.dot(opt.ic['u'] - ubar, u_t))
proj_den = np.sqrt( d3.Integrate(opt.get_metric('Rsqrd').quantity) * d3.Integrate(opt.get_metric('obj_approx').quantity) )
opt.add_metric('proj', False, 1, proj_num / proj_den, integrate=False)

opt.add_metric('u_error', False, 1, 0.5*d3.dot(u - U, u - U), integrate=True)
opt.add_metric('s_error', False, 1, 0.5*(s - S)**2, integrate=True)
opt.add_metric('omega_error', False, 1, 0.5*(w - W)**2, integrate=True)
opt.add_metric('time', False, 1, datetime.now)

opt.objective_overwrite = objective_overwrite
if objective_overwrite != 'default':
    try:
        if isinstance(eval(objective_overwrite), FutureField):
            opt.objective_overwrite = eval(objective_overwrite)
    except:
        logger.warning('Objective overwrite cannot be evaluated')
        raise


# loading state from existing run or from Simple Backward Integration (SBI)
if (load_state):
    try:
        try:
            write_fn = path + '/' + suffix + '/checkpoints/write{:06}.txt'.format(opt.loop_index)
            loadu = np.loadtxt(write_fn).copy()
            logger.info('loaded state in alignment with tracker: {}'.format(write_fn))
        except:
            logger.info('couldnt find state in alignment with tracker: {}'.format(write_fn))
            write_names = [name for name in os.listdir(path + '/' + suffix + '/checkpoints/') if '.txt' in name]
            from natsort import natsorted
            write_fn = path + '/' + suffix + '/checkpoints/' + natsorted(write_names)[-1]
            loadu = np.loadtxt(write_fn).copy()
            logger.info('loaded most recent state: {}'.format(write_fn))
        opt.ic['u']['c'] = opt.reshape_soln(loadu, slices=slices, scales=1)
    except:
        logger.info('load state failed, using SBI or default')

elif ('X' in suffix):
    qrm200 = np.loadtxt(path + '/qrm200.txt').copy()
    opt.ic['u'][opt_layout] = opt.reshape_soln(qrm200, slices=slices, scales=1)
    logger.info('initial guess loaded from qrm200.txt')
    
else:

    try:
        sbi = np.loadtxt(path + '/' + suffix  + '/SBI.txt').copy()
        opt.ic['u'][opt_layout] = opt.reshape_soln(sbi, slices=slices, scales=1)
        logger.info('initial guess loaded from SBI')
    except Exception as e:
        logger.info(e)
        logger.info('no SBI guess provided.')
        if (guide_coeff > 0):
            logger.info('initializating optimization loop with guide coefficient {}'.format(guide_coeff))
            opt.ic['u']['c'] = guide_coeff*ubar['c'].copy()
            logger.info('initial guess set to guide_coeff*ubar')

if (method == "euler"):
    method = opt.descend
    opt.set_euler_params(gamma_init, euler_safety)

startTime = datetime.now()
if (method == "fixedpt"):
    # options = {'maxiter' : opt_iters, 'gtol' : tol}
    opt.ic['u'].change_scales(opt_scales)
    opt.jac_layout.change_scales(1)
    opt.jac_layout['g'] = opt.ic['u']['g'].copy()
    opt.jac_layout['c']
    x0 = opt.jac_layout.allgather_data().flatten().copy()  # Initial guess.
    CW.barrier()
    logger.info('all procs entering optimization loop with # d.o.f. = {}'.format(np.shape(x0)))

    def fixed_iter(x2):
        opt.loop(x2.copy())
        return x2 - 1e0*opt.jac(x2)

    res1 = optimize.fixed_point(fixed_iter, x0)
    logger.info('scipy message {}'.format(res1.message))

if ("root" in suffix):
    def jac_loop(x2):
        opt.loop(x2.copy())
        return -opt.jac(x2)
    tol = 1e-14
    opt.ic['u'].change_scales(opt_scales)
    opt.jac_layout.change_scales(1)
    opt.jac_layout['g'] = opt.ic['u']['g'].copy()
    opt.jac_layout['c']
    x0 = opt.jac_layout.allgather_data().flatten().copy()  # Initial guess.
    CW.barrier()
    logger.info('all procs entering optimization loop with # d.o.f. = {}'.format(np.shape(x0)))

    # sys.exit()
    res1 = optimize.root(jac_loop, x0, method=method, tol=tol)
    logger.info('scipy message {}'.format(res1.message))
try:
    tol = 1e-10
    options = {'maxiter' : opt_iters, 'gtol' : tol}
    opt.ic['u'].change_scales(opt_scales)
    opt.jac_layout.change_scales(1)
    opt.jac_layout['g'] = opt.ic['u']['g'].copy()
    opt.jac_layout['c']
    x0 = opt.jac_layout.allgather_data().flatten().copy()  # Initial guess.
    CW.barrier()
    logger.info('all procs entering optimization loop with # d.o.f. = {}'.format(np.shape(x0)))

    # sys.exit()
    res1 = optimize.minimize(opt.loop, x0, jac=opt.jac, method=method, tol=tol, options=options)
    logger.info('scipy message {}'.format(res1.message))

except opt.LoopIndexException as e:
    details = e.args[0]
    logger.info(details["message"])
except opt.NanNormException as e:
    details = e.args[0]
    logger.info(details["message"])
logger.info('####################################################')
logger.info('COMPLETED OPTIMIZATION RUN')
logger.info('TOTAL TIME {}'.format(datetime.now() - startTime))
logger.info('####################################################')

if (True):
    logger.info('doSBI = {}'.format(doSBI))
    x0 = opt.ic['u'].allgather_data(layout=opt.dist_layout).flatten().copy()
    if (CW.rank == 0):
        np.savetxt(path + '/' + suffix + '/SBI.txt', x0)
    logger.info(path + '/' + suffix + '/SBI.txt')