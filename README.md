## APC523-AST523-MAE507-CSE523_S2024 Numerical Algorithms for Scientific Computing: Final Project
__Authors__: Joseph Lockwood and Gabriel Rios (gr7610) \
__Location on `adroit`__: `/home/gr7610/apc_523/APC523_final`

### Summary
This set of scripts simulates a bottom-driven cavity flow in 2D using an approximation of the Navier-Stokes equations. This model is run using `main.py` from the command line. This model has options for grid size, flow speed, flow viscosity, target Courant-Friedrich-Lewy number, model runtime, the temporal and spatial discretization methods, and gives the user the option to output plots for a given model time and NumPy arrays.

### Model options
- Grid size (`grid_size`): ranges from 5 to 150 (not tested above 150), this controls the timestep in conjunction with `cfl`
- Flow speed (`flow_speed`): ranges from 0 to 100 (not tested above 100), this controls the perturbation flow speed at the cavity edge (in m/s)
- Viscosity (`viscosity`): ranges from 0 to 1 (not tested above 1), this controls fluid viscosity (in m^2/s)
- CFL (`CFL`): ranges from 0 to 1 (not tested above 1), this controls the timestep in conjunction with `N`
- Model time (`model_time`): ranges from 0.05 to 3 (not tested above 3), this controls how long the model runs (in seconds)
- Timestep method (`timestep_method`): options are `explicit_euler`, `leapfrog`, and `projection`
- Spatial discretization method (`spatial_discretization`): options are `cdf2` and `QUICK`
- Print plots (`plots`): options are `True` or `False`, prints to the `figs` subdirectory
- Save Numpy data (`save_data`): options are `True` or `False`, outputs field data (velocities and pressure) to the `assets` subdirectory

### Instructions for `adroit`
1. Load the Python environment for `adroit`: `module load anaconda3/2024.2`
2. Determine the parameter ranges from the options set above. Default parameters are allowed if one wants to run a default case.
3. In the command line, enter the following block with custom argument values replacing the bracketed term:
`python main.py --grid_size={GRID_SIZE} --flow_speed={FLOW_SPEED} --viscosity={VISCOSITY} --CFL={CFL} --model_time={MODEL_TIME} --timestep_method={TIMESTEP_METHOD} --spatial_discretization={SPATIAL_DISCRETIZATION} --plots={PLOTS} --save_data={SAVE_DATA}`
4. Metadata will print to the console. If selected, plots and Numpy numeric data will save to their respective directories.