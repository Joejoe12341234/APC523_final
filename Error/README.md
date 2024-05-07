# Error analyses

Step 1:
Description: code runs the code and outputs the last timestep to a 2D files for u,w,p for each time step and spatial scheme. 
The file also saves the time taken to run each scheme combination. 
Run using: python Output_data_for_error_analy.py
Output: simulation_results_timings.csv

Step 2:
Description: code computes the L2 and maximum errors between the output and "true" datasets
Run using: python Error_analysis.py
Output: error_analysis.csv

