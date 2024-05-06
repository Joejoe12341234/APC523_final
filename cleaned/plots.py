def diagnostic_plots(data, time_index):
    
    '''
    Takes 'data' as an xArray Dataset and plots u, w, and p at timestep 'time_index'.
    '''
    
    fig, axes = plt.subplots(figsize=(9, 3), ncols=3, sharey=True)

    # Plot horizontal velocity
    ax_u = axes[0]
    ax_u.pcolormesh(x, z, data['u'].isel(t=time_index), norm=matplotlib.colors.CenteredNorm(), cmap='bwr')
    ax_u.set_title('u(t = {0})\nmin: {1:.2f}, max: {2:.2f}'.format(time_index, np.nanmin(data['u'].isel(t=time_index)[1:-1, 1:-1]), 
                                                                   np.nanmax(data['u'].isel(t=time_index)[1:-1, 1:-1])), fontsize=9)

    # Plot vertical velocity
    ax_z = axes[1]
    ax_z.pcolormesh(x, z, data['w'].isel(t=time_index), 
                       norm=matplotlib.colors.CenteredNorm(), cmap='bwr')
    ax_z.set_title('w(t = {0})\nmin: {1:.2f}, max: {2:.2f}'.format(time_index, np.nanmin(data['w'].isel(t=time_index)[1:-1, 1:-1]), 
                                                                   np.nanmax(data['w'].isel(t=time_index)[1:-1, 1:-1])), fontsize=9)

    # Plot pressure
    ax_pres = axes[2]
    ax_pres.pcolormesh(x, z, data['p'].isel(t=time_index), 
                       norm=matplotlib.colors.CenteredNorm(), cmap='bwr')
    ax_pres.set_title('p(t = {0})\nmin: {1:.2f}, max: {2:.2f}'.format(time_index, np.nanmin(data['p'].isel(t=time_index)[1:-1, 1:-1]), 
                                                                      np.nanmax(data['p'].isel(t=time_index)[1:-1, 1:-1])), fontsize=9)
    
    plt.show()
    
def field_plots(data, timestep, Re=None):
    
    '''
    Method to plot velocity magnitude, horizontal velocity, vertical velocity, and pressure at a given
    timestep with a Reynolds number (Re) from the model run. 
    
    Takes 'data' as an xArray Dataset, timestep as an integer, and Re as any numeric or str.
    '''

    fig = plt.figure(figsize=(12, 3))
    gs = matplotlib.gridspec.GridSpec(nrows=1, ncols=4)
    
    # Get number of grid cells
    N = len(data['x'].values)

    # Velocity magnitude
    X, Z = np.meshgrid(data['x'], data['z'])
    norm_U = matplotlib.colors.Normalize(vmin=0, vmax=0.75*U)
    cmap_U = 'plasma'
    ax_U = fig.add_subplot(gs[0, 0])
    u_norm = np.sqrt(data['u'].isel(t=timestep)**2 + data['w'].isel(t=timestep)**2)
    ax_U.pcolormesh(data['x'], data['z'], u_norm, norm=norm_U, cmap=cmap_U)
    ax_U.streamplot(X, Z, data['u'].isel(t=timestep), data['w'].isel(t=timestep), color=(1, 1, 1, 0.5))
    cax_U = ax_U.inset_axes([0, 1.025, 1, 0.03])
    colorbar = fig.colorbar(matplotlib.cm.ScalarMappable(norm_U, cmap_U), cax=cax_U, orientation='horizontal')
    colorbar.set_label('Velocity magnitude [m s$^{-1}$]', labelpad=10)
    cax_U.xaxis.set_ticks_position('top')
    cax_U.xaxis.set_label_position('top')

    # Horizontal velocity
    norm_u = matplotlib.colors.CenteredNorm()
    cmap_u = 'bwr'
    ax_u = fig.add_subplot(gs[0, 1])
    ax_u.pcolormesh(data['x'], data['z'], data['u'].isel(t=timestep), norm=norm_u, cmap=cmap_u)
    cax_u = ax_u.inset_axes([0, 1.025, 1, 0.03])
    colorbar = fig.colorbar(matplotlib.cm.ScalarMappable(norm_u, cmap_u), cax=cax_u, orientation='horizontal')
    colorbar.set_label('Horizontal velocity [m s$^{-1}$]', labelpad=10)
    cax_u.xaxis.set_ticks_position('top')
    cax_u.xaxis.set_label_position('top')

    # Vertical velocity
    norm_w = matplotlib.colors.CenteredNorm()
    cmap_w = 'bwr'
    ax_w = fig.add_subplot(gs[0, 2])
    ax_w.pcolormesh(data['x'], data['z'], data['w'].isel(t=timestep), norm=norm_w, cmap=cmap_w)
    cax_w = ax_w.inset_axes([0, 1.025, 1, 0.03])
    colorbar = fig.colorbar(matplotlib.cm.ScalarMappable(norm_w, cmap_w), cax=cax_w, orientation='horizontal')
    colorbar.set_label('Vertical velocity [m s$^{-1}$]', labelpad=10)
    cax_w.xaxis.set_ticks_position('top')
    cax_w.xaxis.set_label_position('top')

    # Pressure
    norm_p = matplotlib.colors.CenteredNorm()
    cmap_p = 'bwr'
    ax_p = fig.add_subplot(gs[0, 3])
    ax_p.pcolormesh(data['x'], data['z'], data['p'].isel(t=timestep), norm=norm_p, cmap=cmap_p)
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
    title_str = 'Time: {0:.2f} s; N = {1}; Re = {2:.2f}'.format(time_index*dt, N, Re)
    fig.suptitle(title_str, ha='center', y=1.2)