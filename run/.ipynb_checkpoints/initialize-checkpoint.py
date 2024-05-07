##############################################################################
# Name:      Initialization function
# Location:  ~/run/initialize.py
# Objective: Handle model initialization for APC 523 final project
##############################################################################

''' Imports. '''
# Numerical packages
import numpy as np, xarray as xr
# Utility packages
import time
# Internal imports

def init(N, U, rho_0, end_time, cfl):
    
    # Define number of grid points
    N_x, N_z = N, N
    # Define the bounds for the domain
    bounds_x, bounds_z = [0, 1], [0, 1]
    # Define grid spacing
    dx = (max(bounds_x) - min(bounds_x))/N_x
    dz = (max(bounds_z) - min(bounds_x))/N_z
    # Timestep as a function of the CFL number
    dt = cfl/(U/dx + U/dz)
    # Maximum timesteps
    time_max = int(np.ceil(end_time/dt))
    
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
    u_init[0, 0, :] = U
    # Vertical velocity
    w_init = np.full(shape=(len(time), X.shape[0], X.shape[1]), fill_value=0, dtype=float)

    # Build and populate xArray Dataset
    data = xr.Dataset(coords={'x': (['x'], x), 'z': (['z'], z),'t': (['t'], time)},
                      data_vars={'p': (['t', 'z', 'x'], p, {'long_name': 'pressure', 'units': 'Pa'}),
                                 'rho': (['t', 'z', 'x',], rho, {'long_name': 'density', 'units': 'kg m^{-3}'})})
    data['u'] = (['t', 'z', 'x'], u_init, {'long_name': 'horizontal velocity', 'units': 'm s^{-1}'})
    data['w'] = (['t', 'z', 'x'], w_init, {'long_name': 'vertical velocity', 'units': 'm s^{-1}'})
    
    return N, x, z, dx, dz, dt, time_max, data