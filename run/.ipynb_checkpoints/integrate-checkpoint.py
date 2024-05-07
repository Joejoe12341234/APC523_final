##################################################################################
# Name:      Run script
# Location:  ~/run/integrate.py
# Objective: Runs the model and integrates over time for the APC 523 final project
##################################################################################

''' Imports. '''
# Numerical packages
import numpy as np, xarray as xr
# Utility packages
import time

from run import initialize, integrate
from utils import storage, helper, math, solver

def run(data, time_max, timestep_method, spatial_method, dx, dz, dt, U, nu):
    
    # Begin timer for performance profiling
    start_time = time.time()
    
    # Step through time
    for iter_count, time_index in enumerate(range(1, time_max)):
        # Print outputs every M steps
        M = 100
        if time_index % M == 0:
            print('Timestep {0} at time {1:.3f}s, Re = {2:.2f}'.format(time_index, time_index*dt, U/nu))

        # Get previous timestep values
        p_prev = data['p'].isel(t=time_index-1)
        u_prev = data['u'].isel(t=time_index-1)
        w_prev = data['w'].isel(t=time_index-1)

        ''' Update each prognostic field. '''

        # Explicit Euler timestepping method
        if timestep_method == 'explicit_euler':
            # Horizontal velocity (u); [u_(n+1) = u_n - dt*F], where F is some function 
            u_next = u_prev + dt*(helper.F_u(u_prev.values, w_prev.values, nu, dx, dz, method=spatial_method) - math.ddx(p_prev.values, dx, method=spatial_method))
            # Vertical velocity (w); [u_(n+1) = u_n - dt*F], where F is some function 
            w_next = w_prev + dt*(helper.F_w(u_prev.values, w_prev.values, nu, dx, dz, method=spatial_method) - math.ddz(p_prev.values, dz, method=spatial_method))
            # Solve for pressure
            p = solver.pressure_solver(p_prev, u_prev, w_prev, dx, dz, dt, spatial_method)

        # Projection timestepping method
        elif timestep_method == 'projection':
            # Copy values
            u, w = u_prev.copy().values, w_prev.copy().values
            # First derivatives
            du_dx = math.ddx(u, dx, method=spatial_method)
            du_dz = math.ddz(u, dz, method=spatial_method)
            dw_dx = math.ddx(w, dx, method=spatial_method)
            dw_dz = math.ddz(w, dz, method=spatial_method)
            # Second derivatives
            d2u_dx2 = math.d2dx2(u, dx, method=spatial_method)
            d2u_dz2 = math.d2dz2(u, dz, method=spatial_method)
            d2w_dx2 = math.d2dx2(w, dx, method=spatial_method)
            d2w_dz2 = math.d2dz2(w, dz, method=spatial_method)

            # Get intermediate velocity step (u* in the Chorin method)
            u_ = u_prev + dt*(-u*du_dx - w*du_dz + nu*(d2u_dx2 + d2u_dz2))
            w_ = w_prev + dt*(-u*dw_dx - w*dw_dz + nu*(d2w_dx2 + d2w_dz2))
            # Update pressure and get spatial derivatives
            p = solver.pressure_solver(p_prev, u_, w_, dx, dz, dt, spatial_method)
            dp_dx = math.ddx(p, dx, method=spatial_method)
            dp_dz = math.ddz(p, dz, method=spatial_method)
            # Update velocity to the next timestep
            u_next = u_ - dt*dp_dx
            w_next = w_ - dt*dp_dz

        # Leapfrog timestepping method
        elif timestep_method == 'leapfrog':
            if time_index < 3:
                # Horizontal velocity (u); [u_(n+1) = u_n - dt*F], where F is some function 
                u_next = u_prev + dt*(helper.F_u(u_prev.values, w_prev.values, nu, dx, dz, method=spatial_method) - math.ddx(p_prev.values, dx, method=spatial_method))
                # Vertical velocity (w); [u_(n+1) = u_n - dt*F], where F is some function 
                w_next = w_prev + dt*(helper.F_w(u_prev.values, w_prev.values, nu, dx, dz, method=spatial_method) - math.ddz(p_prev.values, dz, method=spatial_method))
                # Solve for pressure
                p = solver.direct_inversion(u_prev, w_prev, p_prev, N, dx, dz, dt=dt)
            else:
                # Get previous timesteps: u(t-2) and w(t-2)
                um = data['u'].isel(t=time_index-2)
                wm = data['w'].isel(t=time_index-2)
                # Get updated timestep
                u_ = um + 2*dt*F_u(u_prev.values, w_prev.values, nu, dx, dz, method=spatial_method)
                w_ = wm + 2*dt*F_w(u_prev.values, w_prev.values, nu, dx, dz, method=spatial_method)
                # Get pressure and pressure gradient
                p = solver.pressure_solver(p_prev, u_, w_, dx, dz, dt, spatial_method)
                dp_dx = np.zeros_like(u_)
                dp_dx[1:-1, 1:-1] = (p[1:-1, 2:] - p[1:-1, :-2])/(2*dx)
                dp_dz = np.zeros_like(u_)
                dp_dz[1:-1, 1:-1] = (p[2:, 1:-1] - p[:-2, 1:-1])/(2*dz)
                # Update velocities
                u_next = u_ - 2*dt*dp_dx
                w_next = w_ - 2*dt*dp_dz

        # Set perturbation condition for next iteration
        u_next[0, :] = U
        # Update velocities
        data['u'][{'t': time_index}] = u_next
        data['w'][{'t': time_index}] = w_next
        data['p'][{'t': time_index}] = p
        data['time'][{'t': time_index}] = time_index*dt
        
    
    # End timer for performance profiling
    elapsed_time = time.time() - start_time
    # Get number of iterations
    iterations = time_max
    
    return data, elapsed_time, iterations
    