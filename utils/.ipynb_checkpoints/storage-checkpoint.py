##############################################################################
# Name:      Data storage functions
# Location:  ~/utils/storage.py
# Objective: Handle storage functions for APC 523 final project
##############################################################################

''' Imports. '''
# Numerical packages
import numpy as np, xarray as xr
# Utility packages
import time

# Internal imports
from run import initialize, integrate
from utils import storage, helper, math, solver, visualization

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
        array_out = data[field].isel(t=time_index).values
        # Filename
        try:
            np.save('assets/{0}-{1}s-Re_{2}-space_{3}-time_{4}.npy'.format(field, model_time, Re, spatial_method, timestep_method), array_out)
            print('File saved successfully to assets/{0}-{1}s-Re_{2}-space_{3}-time_{4}.npy'.format(field, model_time, Re, spatial_method, timestep_method))
        except:
            print('File could not save ):')