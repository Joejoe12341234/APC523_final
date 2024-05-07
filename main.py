##############################################################################
# Name:      Main run scripts
# Location:  ~/main.py
# Objective: Run the models for the APC 523 final project
##############################################################################

# Numerical packages
import numpy as np, xarray as xr
# Visualization packages
import matplotlib, matplotlib.pyplot as plt
import matplotlib.animation
# Utility packages
import time
import argparse

# Internal imports
from run import initialize, integrate
from utils import storage, helper, math, solver, visualization

def check_positive(value):
    # Credit: user "Yuushi" on StackOverflow at: https://stackoverflow.com/questions/14117415/how-can-i-constrain-a-value-parsed-with-argparse-for-example-restrict-an-integ
    ivalue = float(value)
    if ivalue <= 0:
        raise argparse.ArgumentTypeError("%s is an invalid positive int value" % value)
    return ivalue

def run_model(N=50, U=10, nu=0.1, cfl=0.25, runtime=0.1, timestep_method='projection', spatial_method='cdf2', visualize=False, save_data=False):

    ''' 1. Derive values needed to run initialization. '''

    # Print the Reynolds number to know the flow regime (inertial or viscous, assumes characteristic unit length)
    Re = U/nu
    print('Reynolds number: {0:.2f}'.format(Re))

    ''' 2. Initialize the model and domain. '''
    # Get initial data
    N, x, z, dx, dz, dt, time_max, data = initialize.init(N, U, rho_0=1, end_time=runtime, cfl=cfl)
    # Add time data variable (in contrast to the time index, t)
    data['time'] = (['t'], np.empty(len(data.t)))

    ''' 3. Run the model to solve for the given flow conditions. '''
    data, elapsed_time, iterations = integrate.run(data, time_max, timestep_method, spatial_method, dx, dz, dt, U, nu)

    print('###############################################################\nModel run with {0} time integration using {1} spatial discretization for an {2}x{2} grid ran in {3:.2f} s with {4} iterations'.format(timestep_method, spatial_method, N, elapsed_time, iterations))

    ''' 4. Visualize data. '''
    # Plot fields at the model run ending
    if visualize:
        visualization.field_plots(data, runtime-dt, dt, U, nu, Re=Re)
        
    # Plot fields at the model run ending
    if save_data:
        storage.dump(data, runtime, dt, Re, spatial_method, timestep_method)

''' Argument parsing. '''
# Set up intake of arguments from the command line
parser = argparse.ArgumentParser()
parser.add_argument('--grid_size', type=check_positive)
parser.add_argument('--flow_speed', type=check_positive)
parser.add_argument('--viscosity', type=check_positive)
parser.add_argument('--CFL', type=check_positive)
parser.add_argument('--model_time', type=check_positive)
parser.add_argument('--timestep_method', type=check_positive)
parser.add_argument('--spatial_discretization', type=check_positive)
parser.add_argument('--plots')
parser.add_argument('--save_data')

# Define variables from inputs
args = parser.parse_args()
N = args.grid_size if args.grid_size else 50
U = args.flow_speed if args.flow_speed else 10
nu = args.viscosity if args.viscosity else 0.1
cfl = args.CFL if args.CFL else 0.25
runtime = args.model_time if args.model_time else 0.75
timestep_method = args.timestep_method if args.timestep_method else 'projection'
spatial_method = args.spatial_discretization if args.spatial_discretization else 'cdf2'
visualize = args.plots if args.plots else True
save_data = args.save_data if args.save_data else True

# Run the model with user inputs
run_model(N, U, nu, cfl, runtime, timestep_method, spatial_method, visualize, save_data)
