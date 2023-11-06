import numpy as np
import matplotlib.pyplot as plt
import dedalus.public as d3
import logging
logger = logging.getLogger(__name__)
import sys

def mode_aspects(q,R,z=0,iter=20):
    """For q=S/f, R=f*S/ωa**2, where ωa = π/Ηx,
       this gives the fastest-growing mode
        aspect ratio: Lz/Hx
        growth rate:  γ/ωa.
        amplitude: max|v|/B
    """
    for _ in range(iter):
        n = (R-1)*(4*q+R*(q-1)**2)+q*(12+R*(q-2))*z**2+8*q*z**3
        d = 2*(R+6*q*(1+z)**2+q*R*(q-4+(q-2)*z))
        z = n/d
        
    n = z*(R*(1-R+2*z)+q*(R**2+2*(1+z)**2-R*(3+2*z)))
    d = q*(R*(2+z)-2*(1 + z)**2)
    s = n/d

    n = 4*R*(R-1-z)*((q-1)*s+q*z)**2
    d = (s+z)*(q*s*(2+R-2*z)+R*z+q*(2+R-2*z)*z-R*s)
    
    return 2 / z**(1/2), s**(1/2), (n/d)**(1/2)
    
# Parameters
π = np.pi

H  = π
B  = 1
ωa = B*π/H
# The sun: Ω ~ S ~ ωa --> q = 1/2, R = 2
q = 1/2
R = 2.0
f  = ωa * (R/q)**(1/2)
S  = ωa * (R*q)**(1/2)

print(f)
print(S)
print(B)
ratio , rate, vmax = mode_aspects(q,R)
vmax = 1.06221

k0 = 1 # fastest waves in box
L = k0 * H * ratio

η = ν = 1e-3 * H**2 * ωa

Nx = 64
Nz = k0 * Nx
time_step     = 5e-3 / (rate*ωa)
stop_sim_time = 500 / ωa
timestepper   = d3.RK222

#setup
coords = d3.CartesianCoordinates('z',  'x')
dist   = d3.Distributor(coords, dtype=np.float64)

# Bases
zbasis = d3.RealFourier(coords['z'], size=Nz, bounds=(0,L), dealias=3/2)
xbasis = d3.ChebyshevT( coords['x'], size=Nx, bounds=(0,H), dealias=3/2)

z = dist.local_grid(zbasis)
x = dist.local_grid(xbasis)

# operators
dz   = lambda A: d3.Differentiate(A, coords['z'])
dx   = lambda A: d3.Differentiate(A, coords['x'])
J    = lambda P, Q: dx(P) * dz(Q) - dz(P) * dx(Q)
Δ    = lambda A: d3.lap(A)

# fields
fields = lambda bases, n: [dist.Field(bases=bases) for _ in range(n)]

a, v, b, ψ = fields((zbasis,xbasis), 4)

j, ζ = -Δ(a), -Δ(ψ)

# residuals
def lift(*args):
    l = lambda a, i: d3.Lift(a,xbasis.derivative_basis(i+1),-1-i)
    return [sum(l(b,i) for i,b in enumerate(a)) for a in args]

τa, σa, τv, σv, τb, σb, τψ, σψ, τζ, σζ = fields(zbasis, 10)
Fa, Fv, Fb, Fζ = lift((τa,σa),(τv,σv),(τb,σb),(τψ,σψ,τζ,σζ))

# problem
problem = d3.IVP([a, v, b, ψ, τa, σa, τv, σv, τb, σb, τψ, σψ, τζ, σζ], namespace=locals())

problem.add_equation("dt(a) - B*dz(ψ)               - η*Δ(a) + Fa = J(a,ψ)          ")
problem.add_equation("dt(v) - B*dz(b) - (f-S)*dz(ψ) - ν*Δ(v) + Fv = J(v,ψ) + J(a,b) ")
problem.add_equation("dt(b) - B*dz(v) -     S*dz(a) - η*Δ(b) + Fb = J(b,ψ) + J(a,v) ")
problem.add_equation("dt(ζ) - B*dz(j) -     f*dz(v) - ν*Δ(ζ) + Fζ = J(ζ,ψ) + J(a,j) ")

for field in ('ψ','a','v','b','ζ'):
    problem.add_equation(f"{field}(x='left')  = 0")
    problem.add_equation(f"{field}(x='right') = 0")

# Solver
solver = problem.build_solver(timestepper)
solver.stop_sim_time = stop_sim_time

# Initial conditions
cut = 1/16
amp = 1e-5  * np.sin(π*x/H) * np.cos(2*π*k0*z/L)
for field in (a,v,b,ψ):
    field.fill_random()
    field.low_pass_filter(scales=1/16)
    field['g'] *= amp

# Flow properties
cadence = 100
flow = d3.GlobalFlowProperty(solver, cadence=cadence)
flow.add_property(np.abs(v), name='v')
max_v = []

# Main loop
try:
    logger.info('Starting main loop')
    while solver.proceed:
        solver.step(time_step)
        if (solver.iteration-1) % cadence == 0:
            max_v += [flow.max('v')]
            logger.info(f'Time={solver.sim_time:.2e} ({100*solver.sim_time/stop_sim_time:.1f}%), '+
                        f'max|v|={max_v[-1]:.2e}')
except:
    logger.error('Exception raised, triggering end of main loop.')
    raise
finally:
    solver.log_stats()
    
max_v = np.array(max_v)
time = np.linspace(0,stop_sim_time,len(max_v))
plt.plot(time,max_v/vmax)
plt.ylabel('$\max|v|\, /\, v_{\max}$');
plt.xlabel('time');
plt.xlabel('time');
plt.title(f'R={R}, q={q}, v_max={vmax:.2f}');
plt.savefig('vmax.png')
