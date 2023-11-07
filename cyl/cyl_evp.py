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
    
R0, R1 = 0.8, 1.0
Lr = R1 - R0
Lz = 2
Nr = 128
Nz = 256
f = 2
nu = 1e-3
eta = nu
stop_sim_time = 100
init_timestep = 1e-4


coords = d3.CartesianCoordinates('z', 'r')
dtype = np.float64
dealias = 3/2
dist = d3.Distributor(coords, dtype=dtype)

Lz, Lr = eval(str(Lz)), eval(str(Lr))
zbasis = d3.RealFourier(coords['z'], size=Nz, bounds=(0, Lz), dealias=dealias)
rbasis = d3.ChebyshevT(coords['r'], size=Nr, bounds=(R0, R1), dealias=dealias)

bases = (zbasis, rbasis)
fbases = (zbasis, )

z = dist.Field(name='z', bases=(zbasis, ))
zg = dist.local_grid(zbasis)
z['g'] = zg

r = dist.Field(name='r', bases=(rbasis, ))
rg = dist.local_grid(rbasis)
r['g'] = rg

# fields
ur = dist.Field(name='ur', bases=bases)
uth = dist.Field(name='uth', bases=bases)
uz = dist.Field(name='uz', bases=bases)
Ar = dist.Field(name='br', bases=bases)
Ath = dist.Field(name='bth', bases=bases)
Az = dist.Field(name='bz', bases=bases)
p = dist.Field(name='p', bases=bases)
phi = dist.Field(name='phi', bases=bases)

# tau terms
tau_p = dist.Field(name='tau_p')
tau_phi = dist.Field(name='tau_phi')

tau_ur1 = dist.Field(name='tau_ur1', bases=(zbasis, ))
tau_uth1 = dist.Field(name='tau_uth1', bases=(zbasis, ))
tau_uz1 = dist.Field(name='tau_uz1', bases=(zbasis, ))

tau_ur2 = dist.Field(name='tau_ur2', bases=(zbasis, ))
tau_uth2 = dist.Field(name='tau_uth2', bases=(zbasis, ))
tau_uz2 = dist.Field(name='tau_uz2', bases=(zbasis, ))

tau_Ar1 = dist.Field(name='tau_br1', bases=(zbasis, ))
tau_Ath1 = dist.Field(name='tau_bth1', bases=(zbasis, ))
tau_Az1 = dist.Field(name='tau_bz1', bases=(zbasis, ))

tau_Ar2 = dist.Field(name='tau_br2', bases=(zbasis, ))
tau_Ath2 = dist.Field(name='tau_bth2', bases=(zbasis, ))
tau_Az2 = dist.Field(name='tau_bz2', bases=(zbasis, ))

tau_dict = {
    p   : (tau_p, ),
    # phi   : (tau_phi, ),
    ur  : (tau_ur1, tau_ur2),
    uth : (tau_uth1, tau_uth2),
    uz  : (tau_uz1, tau_uz2),
    Ar  : (tau_Ar1, tau_Ar2),
    Ath : (tau_Ath1, tau_Ath2),
    Az  : (tau_Az1, tau_Az2),
}

lift_basis = rbasis.derivative_basis(1) # First derivative basis
lift = lambda A: d3.Lift(A, lift_basis, -1)

taus = tuple()
for value in tau_dict.values():
    taus += value

# operators
dr = lambda A: d3.Differentiate(A, coords['r'])
dz = lambda A: d3.Differentiate(A, coords['z'])
integ = lambda A: d3.Integrate(d3.Integrate(A, 'z'), 'r')

# substitutions
br = -dz(Ath)
bth = dz(Ar) - dr(Az)
bz = Ath / r + dr(Ath)

ez = dist.VectorField(coords, name='ez')
er = dist.VectorField(coords, name='er')

ez['g'][0] = 1
er['g'][1] = 1

uz_vec = uz * ez
ur_vec = ur * er

bz_vec = bz * ez
br_vec = br * er

lap_r  = lambda A: dz(dz(A)) + r**(-2) * (r*dr(A) - A) + dr(dr(A) + lift(tau_dict[A][0])) + lift(tau_dict[A][1])

lap_th = lambda A: dz(dz(A)) + r**(-2) * (r*dr(A) - A) + dr(dr(A) + lift(tau_dict[A][0])) + lift(tau_dict[A][1])

lap_z  = lambda A: dz(dz(A)) + r**(-1) * dr(A)         + dr(dr(A) + lift(tau_dict[A][0])) + lift(tau_dict[A][1])

# advective terms 
u_dg_u_r = -uth**2 / r + uz * dz(ur) + ur * dr(ur)
u_dg_u_th = ur * uth / r + uz * dz(uth) + ur * dr(uth)
u_dg_u_z = uz * dz(uz) + ur * dr(uz)

# A . grad B = 
# # r : Ar*dr(Br)  + Ath/r*dth(Br)  + Az * dz(Br)  - Ath*Bth/r
# # th: Ar*dr(Bth) + Ath/r*dth(Bth) + Az * dz(Bth) + Ath*Br/r
# # z : Ar*dr(Bz)  + Ath/r*dth(Bz)  + Az * dz(Bz)

# lorenz terms 
b_dg_b_r = -bth**2 / r + bz * dz(br) + br * dr(br)
b_dg_b_th = br * bth / r + bz * dz(bth) + br * dr(bth)
b_dg_b_z = bz * dz(bz) + br * dr(bz)

# inductive terms 
u_cross_br  = bz  * uth - bth * uz
u_cross_bth = br  * uz  - bz  * ur
u_cross_bz  = bth * ur  - br  * uth

u_cross_b_r  = lambda UR, UTH, UZ, BR, BTH, BZ:      BZ * UTH - BTH * UZ
u_cross_b_th = lambda UR, UTH, UZ, BR, BTH, BZ:      BR * UZ  - BZ  * UR
u_cross_b_z  = lambda UR, UTH, UZ, BR, BTH, BZ:      BTH * UR - BR * UTH

# curl_u_cross_br = -uz * dz(br) + ur * dz(bz) + bz * dz(ur) - br * dz(uz)
# curl_u_cross_bth = uth * dz(bz) - uz * dz(bth) - bth * dz(uz) + bz * dz(uth) + uth * dz(br) - ur * dr(bth) - bth * dr(ur) + br * dr(uth)
# curl_u_cross_bz = (br * uz - bz * ur) / r + uz * dr(br) - ur * dr(bz) - bz * dr(ur) + br * dr(uz)

ur_r = dist.Field(name='ur_r', bases=bases)
uth_r = dist.Field(name='uth_r', bases=bases)
uz_r = dist.Field(name='uz_r', bases=bases)

Ar_r = dist.Field(name='Ar_r', bases=bases)
Ath_r = dist.Field(name='Ath_r', bases=bases)
Az_r = dist.Field(name='Az_r', bases=bases)

U0r = dist.Field(name='U0r', bases=(rbasis, ))
U0th = dist.Field(name='U0th', bases=(rbasis, ))
U0z = dist.Field(name='U0z', bases=(rbasis, ))

A0r = dist.Field(name='A0r', bases=(rbasis, ))
A0th = dist.Field(name='A0th', bases=(rbasis, ))
A0z = dist.Field(name='A0z', bases=(rbasis, ))

B0r = dist.Field(name='B0r', bases=(rbasis, ))
B0th = dist.Field(name='B0th', bases=(rbasis, ))
B0z = dist.Field(name='B0z', bases=(rbasis, ))

# A0th['g'] = (rg**5 - 1) / (5*rg**2)

# B0r  = (-dz(A0th)).evaluate()
# B0th = ( dz(A0r) - dr(A0z)).evaluate()
# B0z  = ( A0th / r + dr(A0th)).evaluate()

U0th['g'] = 1 / rg
B0z['g'] = 1.0 * (1 + 4*rg**5) / (5*rg**3)

C_dg_D_r  = lambda Cr, Cth, Cz, Dr, Dth, Dz:      Cr * dr(Dr)  + Cz * dz(Dr)  - Cth * Dth / r
C_dg_D_th = lambda Cr, Cth, Cz, Dr, Dth, Dz:      Cr * dr(Dth) + Cz * dz(Dth) + Cth * Dr  / r
C_dg_D_z  = lambda Cr, Cth, Cz, Dr, Dth, Dz:      Cr * dr(Dz)  + Cz * dz(Dz)

b_dg_b_r  = C_dg_D_r(B0r, B0th, B0z, br, bth, bz)  + C_dg_D_r(br, bth, bz, B0r, B0th, B0z)
b_dg_b_th = C_dg_D_th(B0r, B0th, B0z, br, bth, bz) + C_dg_D_th(br, bth, bz, B0r, B0th, B0z)
b_dg_b_z  = C_dg_D_z(B0r, B0th, B0z, br, bth, bz)  + C_dg_D_z(br, bth, bz, B0r, B0th, B0z)

u_dg_u_r  = C_dg_D_r(U0r, U0th, U0z, ur, uth, uz)  + C_dg_D_r(ur, uth, uz, U0r, U0th, U0z)
u_dg_u_th = C_dg_D_th(U0r, U0th, U0z, ur, uth, uz) + C_dg_D_th(ur, uth, uz, U0r, U0th, U0z)
u_dg_u_z  = C_dg_D_z(U0r, U0th, U0z, ur, uth, uz)  + C_dg_D_z(ur, uth, uz, U0r, U0th, U0z)

uz.fill_random()
uz.low_pass_filter(scales=1/16)
uz['g'] *= 1e-4

problem = d3.IVP([ur, uth, uz, Ar, Ath, Az, ur_r, uth_r, uz_r, Ar_r, Ath_r, Az_r, p, phi] + list(taus), namespace=locals())

# first order reductions

problem.add_equation("ur_r - dr(ur) + lift(tau_ur1) = 0")
problem.add_equation("uth_r - dr(uth) + lift(tau_uth1) = 0")
problem.add_equation("uz_r - dr(uz) + lift(tau_uz1) = 0")

problem.add_equation("Ar_r - dr(Ar) + lift(tau_Ar1) = 0")
problem.add_equation("Ath_r - dr(Ath) + lift(tau_Ath1) = 0")
problem.add_equation("Az_r - dr(Az) + lift(tau_Az1) = 0")

# incomp.
problem.add_equation("ur + r*dz(uz) + r*ur_r + tau_p     = 0") 

# momentum-r
problem.add_equation("r**2 * dt(ur) + r**2 * dr(p) - r**2 * f * uth + nu * (ur  - r*(ur_r  + r*(dr(ur_r)  + dz(dz(ur)))))  + lift(tau_ur2)  - r**2 * (b_dg_b_r - u_dg_u_r) = 0")

# momentum-theta
problem.add_equation("r**2 * dt(uth)   + r**2 * f * ur + nu * (uth - r*(uth_r + r*(dr(uth_r) + dz(dz(uth))))) + lift(tau_uth2) - r**2 * (b_dg_b_th - u_dg_u_th) = 0")

# momentum-z
problem.add_equation("r    * dt(uz) + r    * dz(p) - nu * (r*dz(dz(uz)) + uz_r + r*dr(uz_r)) + lift(tau_uz2)               - r    * (b_dg_b_z - u_dg_u_z) = 0")


# coulomb gauge.
problem.add_equation("Ar + r*dz(Az) + r*dr(Ar) = 0") 

# induction-r
problem.add_equation("r**2 * dt(Ar) + r**2 * dr(phi) + eta * (Ar  - r*(Ar_r  + r*(dr(Ar_r)  + dz(dz(Ar)))))  + lift(tau_Ar2) - r**2 * (u_cross_b_r(U0r, U0th, U0z, br, bth, bz) + u_cross_b_r(ur, uth, uz, B0r, B0th, B0z) ) = 0")

# induction-theta
problem.add_equation("r**2 * dt(Ath)                 + eta * (Ath - r*(Ath_r + r*(dr(Ath_r) + dz(dz(Ath))))) + lift(tau_Ath2) - r**2 * (u_cross_b_th(U0r, U0th, U0z, br, bth, bz) + u_cross_b_th(ur, uth, uz, B0r, B0th, B0z) ) = 0")

# induction-z
problem.add_equation("r    * dt(Az) + r    * dz(phi) - eta * (r*dz(dz(Az)) + Az_r + r*dr(Az_r)) + lift(tau_Az2) - r**2 * (u_cross_b_z(U0r, U0th, U0z, br, bth, bz) + u_cross_b_z(ur, uth, uz, B0r, B0th, B0z) ) = 0")

# pressure gauge
problem.add_equation("integ(p) = 0")
# problem.add_equation("integ(phi) = 0")

# stress free
problem.add_equation("ur(r='left') = 0")
problem.add_equation("dr(uth)(r='left') = 0")
problem.add_equation("dr(uz)(r='left') = 0")

problem.add_equation("ur(r='right') = 0")
problem.add_equation("dr(uth)(r='right') = 0")
problem.add_equation("dr(uz)(r='right') = 0")


problem.add_equation("phi(r='left') = 0")
problem.add_equation("Ath(r='left') = 0")
problem.add_equation("Az(r='left')  = 0")

problem.add_equation("phi(r='right') = 0")
problem.add_equation("Ath(r='right') = 0")
problem.add_equation("Az(r='right')  = 0")

solver = problem.build_solver(d3.RK222)
solver.stop_sim_time = stop_sim_time


# measurements
cadence = 100
flow = d3.GlobalFlowProperty(solver, cadence=cadence)

ke = ur**2 + uth**2 + uz**2
be = br**2 + bth**2 + bz**2

flow.add_property(ke, name='ke')
flow.add_property(be, name='be')


flow.add_property(ke + be, name='x_l2')
flow.add_property(ke, name='u_l2')
flow.add_property(be, name='b_l2')

# scalars = solver.evaluator.add_file_handler('scalars', sim_dt=0.01, max_writes=1000, mode='overwrite')

# scalars.add_task(integ(ke + be) / Lr / Lz, name='x_l2')
# scalars.add_task(integ(ke) / Lr / Lz, name='u_l2')
# scalars.add_task(integ(be) / Lr / Lz, name='b_l2')

# scalars.add_task(integ(uz**2) / Lr / Lz, name='uz_l2')
# scalars.add_task(integ(uth**2) / Lr / Lz, name='uth_l2')
# scalars.add_task(integ(ur**2) / Lr / Lz, name='ur_l2')

# scalars.add_task(integ(bz**2) / Lr / Lz, name='bz_l2')
# scalars.add_task(integ(bth**2) / Lr / Lz, name='bth_l2')
# scalars.add_task(integ(br**2) / Lr / Lz, name='br_l2')

# slicepoints = solver.evaluator.add_file_handler('slicepoints', sim_dt=0.001, max_writes=50, mode='overwrite')

# slicepoints.add_task(uz, name="uz")
# slicepoints.add_task(uth, name="uth")
# slicepoints.add_task(ur, name="ur")

# slicepoints.add_task(bz, name="bz")
# slicepoints.add_task(bth, name="bth")
# slicepoints.add_task(br, name="br")


CFL = d3.CFL(solver, initial_dt=init_timestep, cadence=10, safety=0.3, threshold=0.01,
             max_change=2, min_change=0.2, max_dt=init_timestep)

CFL.add_velocity(uz_vec)
CFL.add_velocity(ur_vec)

CFL.add_velocity(bz_vec)
CFL.add_velocity(br_vec)

try:
    logger.info('entering main solver loop')
    while solver.proceed:
        ts = CFL.compute_timestep()
        solver.step(ts)
        if (solver.iteration-1) % cadence == 0:
            msg = ""
            msg += "time = {:.2e}; ".format(solver.sim_time)
            msg += "dt = {:.2e}; ".format(ts)
            msg += "avg(ke) = {:.2e}; ".format(flow.volume_integral('ke') / Lz / Lr)
            msg += "avg(be) = {:.2e}; ".format(flow.volume_integral('be') / Lz / Lr)

            logger.info(msg)

except:
    logger.error('Exception raised, triggering end of main loop.')
    raise
finally:
    solver.log_stats()
