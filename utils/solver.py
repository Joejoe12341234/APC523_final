##############################################################################
# Name:      Solver functions
# Location:  ~/utils/solver.py
# Objective: Handle iterative solvers for the APC 523 final project
##############################################################################

''' Imports. '''
# Numerical packages
import numpy as np, xarray as xr
# Utility packages
import time
# Internal imports

from run import initialize, integrate
from utils import storage, helper, math

def direct_inversion(u, w, p, N, dx, dz, dt, U=None, nu=None, spatial_method='explicit_euler'):
    
    '''
    Direct inversion for a linear matrix system to solve the Poisson equation for pressure, Ap = b.
    A is an M x M matrix (M = N_x * N_z), b is an M-sized vector. We're solving for pressure (p).
    '''

    # Initialize Jacobian
    M = N
    A = np.identity(M**2, dtype=float) # 5-point stencil
    b = np.full(shape=M**2, fill_value=0, dtype=float) # Equals the right-hand side of the differential (commonly called f)
    
    ''' First derivatives (2nd-order CDF). '''
    du_dz = math.ddz(u.values, dz, method=spatial_method)
    dw_dx = math.ddx(w.values, dx, method=spatial_method)
    dw_dz = math.ddz(w.values, dz, method=spatial_method)
    
    ''' Second derivatives (2nd-order CDF). '''
    d2u_dx2 = math.d2dx2(u.values, dx, method=spatial_method)
    d2u_dz2 = math.d2dz2(u.values, dz, method=spatial_method)
    d2w_dx2 = math.d2dx2(w.values, dx, method=spatial_method)
    d2w_dz2 = math.d2dz2(w.values, dz, method=spatial_method)
    
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

def pressure_solver(p, u, w, dx, dz, dt, spatial_method):
    
    ''' Iterative solver for the Poisson equation for pressure. '''
    
    p = p.copy().values
    u = u.copy().values
    w = w.copy().values
    
    pn = p.copy()
    
    # Get LHS factor
    B = -2*(1/dx**2 + 1/dz**2)
    
    # Define derivatives
    du_dx = math.ddx(u, dx, method=spatial_method)
    du_dz = math.ddz(u, dz, method=spatial_method)
    dw_dx = math.ddx(w, dx, method=spatial_method)
    dw_dz = math.ddz(w, dz, method=spatial_method)
    
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