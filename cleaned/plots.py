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