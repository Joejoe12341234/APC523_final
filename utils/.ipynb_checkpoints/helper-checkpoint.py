##############################################################################
# Name:      Helper function
# Location:  ~/utils/helper.py
# Objective: Handle helper functions for APC 523 final project
##############################################################################

''' Imports. '''
# Numerical packages
import numpy as np, xarray as xr
# Utility packages
import time

# Internal imports
from run import initialize, integrate
from utils import storage, helper, math, solver, visualization

# Helper functions for time derivatives of momentum (u and w), without the pressure gradient term
def F_u(u_in, w_in, nu, dx, dz, method='cdf2'):
    return -(u_in * math.ddx(u_in, dx, method=method) + w_in * math.ddz(u_in, dz, method=method) - nu*(math.d2dx2(u_in, dx) + math.d2dz2(u_in, dz)))

def F_w(u_in, w_in, nu, dx, dz, method='cdf2'):
    return -(u_in * math.ddx(w_in, dx, method=method) + w_in * math.ddz(w_in, dz, method=method) - nu*(math.d2dx2(w_in, dx) + math.d2dz2(w_in, dz)))

# Helper functions for time derivatives of momentum (u and w), with the pressure gradient term
def F_up(u_in, w_in, p_in, nu, dx, dz, method='cdf2'):
    return -(u_in * math.ddx(u_in, dx, method=method) + w_in * math.ddz(u_in, dz, method=method) - math.ddx(p_in, dx, method=method) - nu*(math.d2dx2(u_in, dx) + math.d2dz2(u_in, dz)))

def F_wp(u_in, w_in, p_in, nu, dx, dz, method='cdf2'):
    return -(u_in * math.ddx(w_in, dx, method=method) + w_in * math.ddz(w_in, dz, method=method) - math.ddz(p_in, dz, method=method) - nu*(math.d2dx2(w_in, dx) + d2dz2(w_in, dz)))