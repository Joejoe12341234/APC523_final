"""

Description: code runs the code and outputs the last timestep to a 2D files for u,w,p for each time step and spatial scheme. 
The file also saves the time taken to run each scheme combination. 

Run using: python Output_data_for_error_analy.py

"""


''' Initial parameters. '''

# Density
rho_0 = 1
# Number of grid points
N = 51
import pandas as pd
perturbation = 'constant'
reynolds_numbers = [10, 100]
timestep_methods = [ 'leapfrog', 'explicit_euler', 'projection']
 spatial_methods = ['QUICK', 'cdf2']
results = [] ## save timings 

# Numerical packages
import numpy as np, xarray as xr
# Visualization packages
import matplotlib, matplotlib.pyplot as plt
import matplotlib.animation
# Utility packages
import time


#### Derivatives
def ddx(data, dx, method='cdf2'):
    ''' 
    Partial derivative with respect to x. 
    Method options are 'cdf2' for second-order central difference and 'QUICK' for third-order upwind differencing. 
    Note: GR modified JL scheme to get QUICK working on first derivatives by making scheme pointwise.
    '''
    
    # Central-difference, 2nd-order
    if method == 'cdf2':
        data = data.copy()
        d_dx = np.zeros_like(data)
        d_dx[1:-1, 1:-1] = (data[1:-1, 2:] - data[1:-1, :-2])/(2*dx)
    # QUICK, 3rd-order (reverts to CDF2 at an edge row/column)
    elif method == 'QUICK':
        d_dx = np.zeros_like(data)
        # Iterate over rows: index from 1 to -1 to avoid edges
        for i in range(1, d_dx.shape[0]-1):
            # Iterate over columns: index from 1 to -1 to avoid edges
            for j in range(1, d_dx.shape[1]-1):
                # If index sufficient, use QUICK
                if j > 1:
                    # GR: Lockwood scheme modified to pointwise form (can definitely be more efficient)
                    # Consider d_dx per QUICK to be d_dx = A + B + C
                    A = (data[i, j+1] - data[i, j-1])/(2*dx)
                    B = (data[i, j+1] - 3*data[i, j] + 3*data[i, j-1] - data[i, j-2])/(8*dx)
                    C = dx**2*(1/8 - 1/6)*(data[i, j+1] - 3*data[i, j] + 3*data[i, j-1] - data[i, j-2])/dx**3
                    d_dx[i, j] = A + B + C
                # Else, revert to CDF2
                else:
                    d_dx[i, j] = (data[i, j+1] - data[i, j-1])/(2*dx)
    
    return d_dx

def d2dx2(data, dx, method='cdf2'):
    ''' 
    Second derivative with respect to x. 
    Method options are 'cdf2' for second-order central difference. 
    '''
    
    if method == 'cdf2':
        data = data.copy()
        d2_dx2 = np.zeros_like(data)
        d2_dx2[1:-1, 1:-1] = (data[1:-1, 2:] - 2*data[1:-1, 1:-1] + data[1:-1, :-2])/(dx**2)
    else:
        data = data.copy()
        d2_dx2 = np.zeros_like(data)
        d2_dx2[1:-1, 1:-1] = (data[1:-1, 2:] - 2*data[1:-1, 1:-1] + data[1:-1, :-2])/(dx**2)
    
    return d2_dx2

def ddz(data, dz, method='cdf2'):
    ''' 
    Partial derivative with respect to z. 
    Method options are 'cdf2' for second-order central difference and 'QUICK' for third-order upwind differencing. 
    Note: GR modified JL scheme to get QUICK working on first derivatives by making scheme pointwise.
    '''
    
    # Central-difference, 2nd-order
    if method == 'cdf2':
        data = data.copy()
        d_dz = np.zeros_like(data)
        d_dz[1:-1, 1:-1] = (data[2:, 1:-1] - data[:-2, 1:-1])/(2*dz)
    # QUICK, 3rd-order (reverts to CDF2 at an edge row/column)
    elif method == 'QUICK':
        d_dz = np.zeros_like(data)
        # Iterate over rows: index from 1 to -1 to avoid edges
        for i in range(1, d_dz.shape[0]-1):
            # Iterate over columns: index from 1 to -1 to avoid edges
            for j in range(1, d_dz.shape[1]-1):
                # If index sufficient, use QUICK
                if i > 1:
                    # GR: Lockwood scheme modified to pointwise form (can definitely be more efficient)
                    # Consider d_dx per QUICK to be d_dx = A + B + C
                    A = (data[i+1, j] - data[i-1, j])/(2*dz)
                    B = (data[i+1, j] - 3*data[i, j] + 3*data[i-1, j] - data[i-2, j])/(8*dz)
                    C = dx**2*(1/8 - 1/6)*(data[i+1, j] - 3*data[i, j] + 3*data[i-1, j] - data[i-2, j])/dz**3
                    d_dz[i, j] = A + B + C
                # Else, revert to CDF2
                else:
                    d_dz[i, j] = (data[i+1, j] - data[i-1, j])/(2*dz)
    
    return d_dz

def d2dz2(data, dz, method='cdf2'):
    ''' 
    Second derivative with respect to x. 
    Method options are 'cdf2' for second-order central difference. 
    '''
    
    if method == 'cdf2':
        data = data.copy()
        d2_dz2 = np.zeros_like(data)
        d2_dz2[1:-1, 1:-1] = (data[2:, 1:-1] - 2*data[1:-1, 1:-1] + data[:-2, 1:-1])/(dz**2)
    else:
        d2_dz2 = np.zeros_like(data)
        d2_dz2[1:-1, 1:-1] = (data[2:, 1:-1] - 2*data[1:-1, 1:-1] + data[:-2, 1:-1])/(dz**2)
    
    return d2_dz2


# Helper functions for time derivatives of momentum (u and w)
def F_u(u_in, w_in, method='cdf2'):
    return -(u_in * ddx(u_in, dx, method=method) + w_in * ddz(u_in, dz, method=method) - nu*(d2dx2(u_in, dx) + d2dz2(u_in, dz)))
def F_w(u_in, w_in, method='cdf2'):
    return -(u_in * ddx(w_in, dx, method=method) + w_in * ddz(w_in, dz, method=method) - nu*(d2dx2(w_in, dx) + d2dz2(w_in, dz)))
def F_up(u_in, w_in, p_in, method='cdf2'):
    return -(u_in * ddx(u_in, dx, method=method) + w_in * ddz(u_in, dz, method=method) - ddx(p_in, dx, method=method) - nu*(d2dx2(u_in, dx) + d2dz2(u_in, dz)))
def F_wp(u_in, w_in, p_in, method='cdf2'):
    return -(u_in * ddx(w_in, dx, method=method) + w_in * ddz(w_in, dz, method=method) - ddz(p_in, dz, method=method) - nu*(d2dx2(w_in, dx) + d2dz2(w_in, dz)))

#### Data saving for error  
def dump(data, model_time, dt, Re, spatial_method, timestep_method):
    '''
    Method to save xArray data for velocity (u, w) and pressure (p) for a given Reynolds number (Re) 
    from a given space (spatial_method) and time (timestep_method) scheme to .npy format for a given time (model_time, not time index).
    
    Data is saved to the 'assets' subdirectory.
    
    Example 'dump' command at model_time = 3s:  dump(data, 3, dt, Re, spatial_method, timestep_method)
    Example load command for saved data:        np.load('assets/u-3s-Re_100.0-space_cdf2-time_projection.npy')
    '''
    
    for field in ['u', 'w', 'p']:
        # Index corresponding to timestep
        time_index = int(np.floor(model_time/dt))-1
        # Output array
        array_out = data[field].values[-1,:,:]
        # Filename
        np.save('output_error/{0}-{1}s-Re_{2}-space_{3}-time_{4}.npy'.format(field, model_time, Re, spatial_method, timestep_method), array_out)
        print('Save: ', {'output_error/{0}-{1}s-Re_{2}-space_{3}-time_{4}.npy'})
        

#### Initialization function
def init(N, U_0, dt=0.01, rho_0=1, time_max=10):
    
    # Define time axis
    time = range(0, time_max)

    # Define the basis vector (x-axis, z-axis)
    x, z = [np.arange(min(bounds_z), max(bounds_x), dx),
            np.arange(min(bounds_z), max(bounds_z), dz)]
    # Crete meshgrid
    X, Z = np.meshgrid(x, z)

    ''' Initial values. '''
    # Density
    rho_init = np.full(shape=X.shape, fill_value=rho_0)
    rho = np.full(shape=(len(time), X.shape[0], X.shape[1]), fill_value=np.nan)
    rho[0, :, :] = rho_init
    # Pressure
    p_init = 0
    p = np.full(shape=(len(time), X.shape[0], X.shape[1]), fill_value=np.nan)
    p[0, :, :] = p_init
    # Horizontal velocity
    u_init = np.full(shape=(len(time), X.shape[0], X.shape[1]), fill_value=0, dtype=float)
    u_init[0, 0, :] = U_0
    # Vertical velocity
    w_init = np.full(shape=(len(time), X.shape[0], X.shape[1]), fill_value=0, dtype=float)

    # Build and populate xArray Dataset
    data = xr.Dataset(coords={'x': (['x'], x), 'z': (['z'], z),'t': (['t'], time)},
                      data_vars={'p': (['t', 'z', 'x'], p, {'long_name': 'pressure', 'units': 'Pa'}),
                                 'rho': (['t', 'z', 'x',], rho, {'long_name': 'density', 'units': 'kg m^{-3}'})})
    data['u'] = (['t', 'z', 'x'], u_init, {'long_name': 'horizontal velocity', 'units': 'm s^{-1}'})
    data['w'] = (['t', 'z', 'x'], w_init, {'long_name': 'vertical velocity', 'units': 'm s^{-1}'})
    
    return N, x, z, data



#### Direct inversion method

def direct_inversion(u, w, p, N, dx, dz, dt, U=None, nu=None, mode='explicit_euler'):
    
    '''
    Direct inversion for a linear matrix system to solve the Poisson equation for pressure, Ap = b.
    A is an M x M matrix (M = N_x * N_z), b is an M-sized vector. We're solving for pressure (p).
    '''

    # Initialize Jacobian
    M = N
    A = np.identity(M**2, dtype=float) # 5-point stencil
    b = np.full(shape=M**2, fill_value=0, dtype=float) # Equals the right-hand side of the differential (commonly called f)
    
    ''' First derivatives (2nd-order CDF). '''
    du_dx = ddx(u.values, dx, method='cdf2')
    du_dz = ddz(u.values, dz, method='cdf2')
    dw_dx = ddx(w.values, dx, method='cdf2')
    dw_dz = ddz(w.values, dz, method='cdf2')
    
    ''' Second derivatives (2nd-order CDF). '''
    d2u_dx2 = d2dx2(u.values, dx, method='cdf2')
    d2u_dz2 = d2dz2(u.values, dz, method='cdf2')
    d2w_dx2 = d2dx2(w.values, dx, method='cdf2')
    d2w_dz2 = d2dz2(w.values, dz, method='cdf2')
    
    row = 0
    # Iterate over each point (each point corresponds to a row in matrix A)
    for i in range(0, N): # iterate over rows
        for j in range(0, N): # iterate over columns
            
            # Capture the iterand values
            u_, w_ = u.isel(x=j, z=i), w.isel(x=j, z=i)
            
            # Fill in identity diagonal
            A[row, row] = -4
            # Get indices for +/- x and y values
            ym, yp, xm, xp = row - M, row + M, row - 1, row + 1
            
            ''' Set boundary conditions. '''
                
            # Handle corners
            # Top left
            if (row // M == 0) and (row % M == 0):
                # print('Top left')
                A[xp, row] = 1
                A[yp, row] = 1
                b[row] = 0 + p[i, j] # Dirichlet top, Neumann left BC
            # Top right
            elif (row // M == 0) and ((row % M) == (M-1)):
                # print('Top right')
                A[xm, row] = 1
                A[yp, row] = 1
                b[row] = 0 + p[i, j] # Dirichlet top, Neumann right BC
            # Bottom left
            elif (row // M == (M-1)) and (row % M == 0):
                # print('Bottom left')
                A[xp, row] = 1
                A[ym, row] = 1
                b[row] = p[i, j] + p[i, j] # Neumann bottom BC, Neumann left BC
            # Bottom right
            elif (row // M == (M-1)) and ((row % M) == (M-1)):
                # print('Bottom right')
                A[xm, row] = 1
                A[ym, row] = 1
                b[row] = p[i, j] + p[i, j] # Neumann bottom BC, Neumann right BC
            # Top center
            elif (row // M == 0) and ((row % M != 0) and ((row % M) != (M-1))):
                # print('Top center')
                A[xm, row] = 1
                A[xp, row] = 1
                A[yp, row] = 1
                b[row] = 0 # Dirichlet top BC
            # Bottom center
            elif (row // M == (M-1)) and ((row % M != 0) and ((row % M) != (M-1))):
                # print('Bottom center')
                A[xm, row] = 1
                A[xp, row] = 1
                A[ym, row] = 1
                b[row] = p[i, j] # Neumann bottom BC
            # Left center
            elif (row % M == 0) and ((row // M != 0) and (row // M != (M-1))):
                # print('Left center')
                A[xp, row] = 1
                A[ym, row] = 1
                A[yp, row] = 1
                b[row] = p[i, j] # Neumann left BC
            # Right center
            elif ((row % M) == (M-1)) and ((row // M != 0) and (row // M != (M-1))):
                # print('Right center')
                A[xm, row] = 1
                A[ym, row] = 1
                A[yp, row] = 1
                b[row] = p[i, j] # Neumann right BC
            # Non-edge
            else:
                # print('Non-edge')
                A[xm, row] = 1
                A[xp, row] = 1
                A[ym, row] = 1
                A[yp, row] = 1
                
                if not nu:
                    nu = 0.1
                if not U:
                    U = 1
                    
                Re = U/nu
                d3u_dx3  = (d2u_dx2[i, j+1] - d2u_dx2[i, j-1])/(2*dx)
                d3u_dxz2 = (d2u_dz2[i, j+1] - d2u_dz2[i, j-1])/(2*dx)
                d3w_dx2z = (d2w_dx2[i+1, j] - d2w_dx2[i-1, j])/(2*dz)
                d3w_dz3  = (d2w_dz2[i+1, j] - d2w_dz2[i-1, j])/(2*dz)
                
                d2u_dxz = (du_dz[i, j+1] - du_dz[i, j-1])/(2*dx)
                d2w_dxz = (dw_dx[i+1, j] - dw_dx[i-1, j])/(2*dz)
                
                u_, w_ = u.isel(x=j, z=i), w.isel(x=j, z=i)
                
                if mode == 'explicit_euler':
                    b[row] = dx**2*((1/dt)*(du_dx[i, j] + dw_dz[i, j]) + (1/Re)*(d3u_dx3 + d3u_dxz2 + d3w_dx2z + d3w_dz3) - (du_dx[i, j]**2 + dw_dz[i, j]**2 + 2*dw_dx[i, j]*du_dz[i, j]))
                elif mode == 'rk_proj':
                    x_comp = du_dx[i, j] + dt*(-du_dx[i, j]**2 - u_*d2u_dx2[i, j] - dw_dx[i, j]*du_dz[i, j] - w_*d2u_dxz + nu*(d3u_dx3 + d3u_dxz2))
                    z_comp = dw_dz[i, j] + dt*(-dw_dx[i, j]*du_dz[i, j] - u_*d2w_dxz - dw_dz[i, j]**2 - w_*d2w_dz2[i, j] + nu*(d3w_dx2z + d3w_dz3))
                    b[row] = dx**2*(x_comp + z_comp)/dt
                                               
            row += 1

    # print(A, '\n\n', b, '\n')
    p = (np.linalg.inv(A) @ b).reshape(M, M)
    return p


#### Iterative point-by-point Poisson solver

def pressure_solver(p, u, w, dx, dz, dt, method='cdf2'):
    
    ''' Iterative solver for the Poisson equation for pressure. '''
    
    p = p.copy().values
    u = u.copy().values
    w = w.copy().values
    
    pn = p.copy()
    
    # Get LHS factor
    B = -2*(1/dx**2 + 1/dz**2)
    
    # Define derivatives
    du_dx = ddx(u, dx, method=method)
    du_dz = ddz(u, dz, method=method)
    dw_dx = ddx(w, dx, method=method)
    dw_dz = ddz(w, dz, method=method)
    
    # Get divergence of v
    div_v = (du_dx + dw_dz)/dt
    # Get the pressure residual
    b = -du_dx**2 - 2*dw_dx*du_dz - dw_dz**2
    
    for q in range(100):
        pn = p.copy()
        
        # Get pressure terms from the LHS (these are not derivatives)
        p_x = np.zeros_like(pn)
        p_x[1:-1, 1:-1] = (pn[1:-1, 2:] + pn[1:-1, :-2])/(dx**2)
        p_z = np.zeros_like(pn)
        p_z[1:-1, 1:-1] = (pn[2:, 1:-1] + pn[:-2, 1:-1])/(dz**2)
        # Sum them
        p[1:-1, 1:-1] = (1/B)*(-p_x[1:-1, 1:-1] - p_z[1:-1, 1:-1] + div_v[1:-1, 1:-1] + dx*dz*b[1:-1, 1:-1])

        # Impose boundary conditions
        p[:, -1] = p[:, -2]  # dp/dx = 0 at x = 2
        p[-1, :] = p[-2, :]  # dp/dy = 0 at y = 0
        p[:, 0] = p[:, 1]    # dp/dx = 0 at x = 0
        p[0, :] = 0          # p = 0 at y = 2
        
    return p

#### Run script

for RE in reynolds_numbers:
    for timestep_method in timestep_methods:
        for spatial_method in spatial_methods:
            print(f'Simulation start for RE={RE}, timestep_method={timestep_method}, spatial_method={spatial_method} ')
            start_time = time.time()
            
            if RE == 10:
                nu = 0.05
                U = 0.5
            elif RE == 100:
                nu = 0.01
                U = 100

            # Define number of grid points
            N_x, N_z = N, N
            # Define the bounds for the domain
            bounds_x, bounds_z = [0, 1], [0, 1]
            # Define grid spacing
            dx = (max(bounds_x) - min(bounds_x))/N_x
            dz = (max(bounds_z) - min(bounds_x))/N_z

            # CFL number
            cfl = 0.01
            # Timestep as a function of the CFL number
            dt = cfl/(U/dx + U/dz)

            # End time for simulation (shouldn't exceed for steady-state)
            end_time = 0.75
            # Maximum timesteps
            time_max = int(np.ceil(end_time/dt))

            # Get initial data
            N, x, z, data = init(N, U, dt, rho_0=rho_0, time_max=time_max)
            data['time'] = (['t'], np.empty(len(data.t)))

 
            # Step through time
            for iter_count, time_index in enumerate(range(1, time_max)):

                # Print outputs every M steps
                M = 1000
                if time_index % M == 0:
                    print('Timestep {0} at time {1:.3f}s, Re = {2:.2f}'.format(time_index, time_index*dt, RE))

                # Get previous timestep values
                p_prev = data['p'].isel(t=time_index-1)
                u_prev = data['u'].isel(t=time_index-1)
                w_prev = data['w'].isel(t=time_index-1)

                ''' Update each prognostic field. '''

                # Explicit Euler timestepping method
                if timestep_method == 'explicit_euler':
                    # Horizontal velocity (u); [u_(n+1) = u_n - dt*F], where F is some function 
                    u_next = u_prev + dt*(F_u(u_prev.values, w_prev.values, method=spatial_method) - ddx(p_prev.values, dx, method=spatial_method))
                    # Vertical velocity (w); [u_(n+1) = u_n - dt*F], where F is some function 
                    w_next = w_prev + dt*(F_w(u_prev.values, w_prev.values, method=spatial_method) - ddz(p_prev.values, dz, method=spatial_method))
                    # Solve for pressure
                    # p = direct_inversion(u_prev, w_prev, p_prev,  N, dx, dz, dt=dt)
                    p = pressure_solver(p_prev, u_prev, w_prev, dx, dz, dt)

                # Projection timestepping method
                elif timestep_method == 'projection':
                    # Copy values
                    u, w = u_prev.copy().values, w_prev.copy().values
                    # First derivatives
                    du_dx = ddx(u, dx, method=spatial_method)
                    du_dz = ddz(u, dz, method=spatial_method)
                    dw_dx = ddx(w, dx, method=spatial_method)
                    dw_dz = ddz(w, dz, method=spatial_method)
                    # Second derivatives
                    d2u_dx2 = d2dx2(u, dx, method=spatial_method)
                    d2u_dz2 = d2dz2(u, dz, method=spatial_method)
                    d2w_dx2 = d2dx2(w, dx, method=spatial_method)
                    d2w_dz2 = d2dz2(w, dz, method=spatial_method)

                    # Get intermediate velocity step (u* in the Chorin method)
                    u_ = u_prev + dt*(-u*du_dx - w*du_dz + nu*(d2u_dx2 + d2u_dz2))
                    w_ = w_prev + dt*(-u*dw_dx - w*dw_dz + nu*(d2w_dx2 + d2w_dz2))
                    # Update pressure and get spatial derivatives
                    p = pressure_solver(p_prev, u_, w_, dx, dz, dt)
                    dp_dx = ddx(p, dx, method=spatial_method)
                    dp_dz = ddz(p, dz, method=spatial_method)
                    # Update velocity to the next timestep
                    u_next = u_ - dt*dp_dx
                    w_next = w_ - dt*dp_dz

                # Leapfrog timestepping method
                elif timestep_method == 'leapfrog':
                    if time_index < 3:
                        # Horizontal velocity (u); [u_(n+1) = u_n - dt*F], where F is some function 
                        u_next = u_prev + dt*(F_u(u_prev.values, w_prev.values, method=spatial_method) - ddx(p_prev.values, dx, method=spatial_method))
                        # Vertical velocity (w); [u_(n+1) = u_n - dt*F], where F is some function 
                        w_next = w_prev + dt*(F_w(u_prev.values, w_prev.values, method=spatial_method) - ddz(p_prev.values, dz, method=spatial_method))
                        # Solve for pressure
                        p = direct_inversion(u_prev, w_prev, p_prev, N, dx, dz, dt=dt)
                    else:
                        # Get previous timesteps: u(t-2) and w(t-2)
                        um = data['u'].isel(t=time_index-2)
                        wm = data['w'].isel(t=time_index-2)
                        # Get updated timestep
                        u_ = um + 2*dt*F_u(u_prev.values, w_prev.values, method=spatial_method)
                        w_ = wm + 2*dt*F_w(u_prev.values, w_prev.values, method=spatial_method)
                        # Get pressure and pressure gradient
                        p = pressure_solver(p_prev, u_, w_, dx, dz, dt)
                        dp_dx = np.zeros_like(u_)
                        dp_dx[1:-1, 1:-1] = (p[1:-1, 2:] - p[1:-1, :-2])/(2*dx)
                        dp_dz = np.zeros_like(u_)
                        dp_dz[1:-1, 1:-1] = (p[2:, 1:-1] - p[:-2, 1:-1])/(2*dz)
                        # Update velocities
                        u_next = u_ - 2*dt*dp_dx
                        w_next = w_ - 2*dt*dp_dz

                ''' Flow cases. '''
                u_next[0, :] = U
                    
                # Update velocities
                data['u'][{'t': time_index}] = u_next
                data['w'][{'t': time_index}] = w_next
                data['p'][{'t': time_index}] = p
                data['time'][{'t': time_index}] = time_index*dt

                if time_index == time_max-1:   
                    time_max =  np.round(time_index*dt,2)
                    dump(data, time_max, dt, RE, spatial_method, timestep_method)

                    # End timer
                    elapsed_time = time.time() - start_time

                    # Collect results
                    results.append({
                        'Reynolds Number': RE,
                        'Timestep Method': timestep_method,
                        'Spatial Method': spatial_method,
                        'Execution Time (s)': elapsed_time
                    })

                    print(f'Simulation done for RE={RE}, timestep_method={timestep_method}, spatial_method={spatial_method}, Time: {elapsed_time:.2f}s')


# Create a DataFrame
results_df = pd.DataFrame(results)

# Save DataFrame to CSV file
results_df.to_csv('output_error/simulation_results_timings.csv', index=False)

# Optionally, display the DataFrame
print(results_df)




 