##############################################################################
# Name:      Math functions
# Location:  ~/utils/math.py
# Objective: Handle mathematical functions for APC 523 final project
##############################################################################

''' Imports. '''
# Numerical packages
import numpy as np, xarray as xr
# Utility packages
import time

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
                    C = dz**2*(1/8 - 1/6)*(data[i+1, j] - 3*data[i, j] + 3*data[i-1, j] - data[i-2, j])/dz**3
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