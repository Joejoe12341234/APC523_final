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
