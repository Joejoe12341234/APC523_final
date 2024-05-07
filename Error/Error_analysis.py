
"""

Description: code computes the L2 and maximum errors between the output and "true" datasets

Run using: python Error_analysis.py

Output: error_analysis.csv

"""


import numpy as np
import os

def compute_errors(sim_data, true_data):
    """Compute L2 and maximum errors between simulation and true data."""
    N_f = 51  # Define the number of points in the domain
    # Ensure data covers exactly N_f points. Adjust slicing or interpolation if necessary.
    
    print(sim_data.shape)
    print(true_data.shape)
    
    l2_error = np.sqrt(np.sum((sim_data - true_data) ** 2) / N_f)
    max_error = np.max(np.abs(sim_data - true_data))
    return l2_error, max_error

def main():

    reynolds_numbers = [10, 100]   
    timestep_methods = ['explicit_euler', 'projection']
    spatial_methods = ['QUICK', 'cdf2']
    fields = ['u', 'w', 'p']
    results = []

    # Loop through all combinations of settings
    for RE in reynolds_numbers:
        for timestep_method in timestep_methods:
            for spatial_method in spatial_methods:
                for field in fields:
                    
                    print(field, spatial_method, timestep_method, RE )
                    # True data filename pattern (modify path as needed)
                    if RE == 10:
                        true_filename = f'assets/{field}-0.75s-Re_{RE}.0-space_cdf2-time_projection.npy'
                    else:
                        true_filename = f'assets/{field}-3s-Re_{RE}.0-space_cdf2-time_projection.npy'
                    
                    if os.path.exists(true_filename):
                        true_data = np.load(true_filename)
                        print('Found', true_filename)

                        
                        sim_filename = f'output_error/{field}-0.75s-Re_{RE}-space_{spatial_method}-time_{timestep_method}.npy'
                        sim_data = np.load(sim_filename)
                        if sim_data.shape[0] > 50 or sim_data.shape[1] > 50:
                           sim_data = sim_data[:50, :50]  # Limit to 50x50
                        if true_data.shape[0] > 50 or sim_data.shape[1] > 50:
                            true_data = true_data[:50, :50]  # Limit to 50x50
                            
                        l2_error, max_error = compute_errors(sim_data, true_data)

                        result = {
                                'RE': RE,
                                'Timestep Method': timestep_method,
                                'Spatial Method': spatial_method,
                                'Field': field,
                                'L2 Error': l2_error,
                                'Maximum Error': max_error,
                                'Mean data': np.mean(sim_data)
                            }
                        results.append(result) 
                        
                        if os.path.exists(sim_filename):
                            print('Found', sim_filename)
                        else: 
                            print('Not found', {field}, {RE}, {spatial_method},{timestep_method} )
                            print('Not found', sim_filename)

    import pandas as pd

    # Create a DataFrame
    results_df = pd.DataFrame(results)

    # Save DataFrame to CSV file
    results_df.to_csv('output_error/error_analysis.csv', index=False)

    # Optionally, display the DataFrame
    print(results_df)

if __name__ == '__main__':
    main()


