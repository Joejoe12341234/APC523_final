##############################################################################
# Name:      Visualization functions
# Location:  ~/utils/visualization.py
# Objective: Handle visualization functions for APC 523 final project
##############################################################################

''' Imports. '''
# Numerical packages
import numpy as np, xarray as xr
# Visualization packages
import matplotlib, matplotlib.pyplot as plt
import matplotlib.animation
# Utility packages
import time

def field_plots(data, timestep, dt, U, nu, Re):
    
    '''
    Method to plot velocity magnitude, horizontal velocity, vertical velocity, and pressure at a given timestep.
    The Reynolds number should be passed for plot metadata printing.
    '''

    fig = plt.figure(figsize=(12, 3))
    gs = matplotlib.gridspec.GridSpec(nrows=1, ncols=4)
    
    # Get number of grid cells
    N = len(data['x'].values)
    
    # Get timestep numeric
    time_index = int(np.floor(timestep/dt) - 1)

    # Velocity magnitude
    X, Z = np.meshgrid(data['x'], data['z'])
    norm_U = matplotlib.colors.Normalize(vmin=0, vmax=0.75*U)
    cmap_U = 'plasma'
    ax_U = fig.add_subplot(gs[0, 0])
    u_norm = np.sqrt(data['u'].isel(t=time_index)**2 + data['w'].isel(t=time_index)**2)
    ax_U.pcolormesh(data['x'], data['z'], u_norm, norm=norm_U, cmap=cmap_U)
    ax_U.streamplot(X, Z, data['u'].isel(t=time_index), data['w'].isel(t=time_index), color=(1, 1, 1, 0.5))
    cax_U = ax_U.inset_axes([0, 1.025, 1, 0.03])
    colorbar = fig.colorbar(matplotlib.cm.ScalarMappable(norm_U, cmap_U), cax=cax_U, orientation='horizontal')
    colorbar.set_label('Velocity magnitude [m s$^{-1}$]', labelpad=10)
    cax_U.xaxis.set_ticks_position('top')
    cax_U.xaxis.set_label_position('top')

    # Horizontal velocity
    norm_u = matplotlib.colors.CenteredNorm()
    cmap_u = 'bwr'
    ax_u = fig.add_subplot(gs[0, 1])
    ax_u.pcolormesh(data['x'], data['z'], data['u'].isel(t=time_index), norm=norm_u, cmap=cmap_u)
    cax_u = ax_u.inset_axes([0, 1.025, 1, 0.03])
    colorbar = fig.colorbar(matplotlib.cm.ScalarMappable(norm_u, cmap_u), cax=cax_u, orientation='horizontal')
    colorbar.set_label('Horizontal velocity [m s$^{-1}$]', labelpad=10)
    cax_u.xaxis.set_ticks_position('top')
    cax_u.xaxis.set_label_position('top')

    # Vertical velocity
    norm_w = matplotlib.colors.CenteredNorm()
    cmap_w = 'bwr'
    ax_w = fig.add_subplot(gs[0, 2])
    ax_w.pcolormesh(data['x'], data['z'], data['w'].isel(t=time_index), norm=norm_w, cmap=cmap_w)
    cax_w = ax_w.inset_axes([0, 1.025, 1, 0.03])
    colorbar = fig.colorbar(matplotlib.cm.ScalarMappable(norm_w, cmap_w), cax=cax_w, orientation='horizontal')
    colorbar.set_label('Vertical velocity [m s$^{-1}$]', labelpad=10)
    cax_w.xaxis.set_ticks_position('top')
    cax_w.xaxis.set_label_position('top')

    # Pressure
    norm_p = matplotlib.colors.CenteredNorm()
    cmap_p = 'bwr'
    ax_p = fig.add_subplot(gs[0, 3])
    ax_p.pcolormesh(data['x'], data['z'], data['p'].isel(t=time_index), norm=norm_p, cmap=cmap_p)
    cax_p = ax_p.inset_axes([0, 1.025, 1, 0.03])
    colorbar = fig.colorbar(matplotlib.cm.ScalarMappable(norm_p, cmap_p), cax=cax_p, orientation='horizontal')
    colorbar.set_label('Pressure [Pa]', labelpad=10)
    cax_p.xaxis.set_ticks_position('top')
    cax_p.xaxis.set_label_position('top')

    # Universal subplot formatting
    for i, ax in enumerate(fig.axes):
        ax.set_xlim([0, 1])
        ax.set_ylim([0, 1])
        ax.set_aspect('equal')
        if i > 0:
            ax.set_yticklabels([])

    # Define title formatting
    title_str = 'Time: {0:.3f} s; N = {1}; Re = {2:.2f}'.format(timestep, N, Re)
    fig.suptitle(title_str, ha='center', y=1.2)

    filename = 'field_plot-Re_{0}-t_{1:.2f}s.png'.format(Re, timestep)
    try:
        plt.savefig('figs/{0}'.format(filename), dpi=300, bbox_inches='tight')
        print('Figure saved to figs/{0}'.format(filename))
    except:
        print('Figure could not save ):')