import dedalus.public as d3

def vp_bvp_func(b, A, phi, coords):

    # problem.parameters['by'] = by
    # problem.parameters['bz'] = bz
    # problem.parameters['bx'] = bx

    dy = lambda A: d3.Differentiate(A, coords['y'])
    dz = lambda A: d3.Differentiate(A, coords['z'])
    dx = lambda A: d3.Differentiate(A, coords['x'])

    problem = d3.LBVP(variables=[A, phi], namespace=locals())

    # problem.add_equation("dx(Ax) + dy(Ay) + dz(Az) = 0")
    # problem.add_equation("dy(Az) - dz(Ay) + dx(phi) = bx")
    # problem.add_equation("dz(Ax) - dx(Az) + dy(phi) = by")
    # problem.add_equation("dx(Ay) - dy(Ax) + dz(phi) = bz")



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
