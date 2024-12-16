import sys
import os
import numpy as np
import pandas as pd
import json
import time

from FunctionsNew import *



# run_index = int(sys.argv[1])
run_index = 500

degs = np.arange(0, 91, 30)                     # Angles from 0 to 90 degrees in steps of 30
size_multi_arr = np.sqrt(np.linspace(1, 20, 20))  # 20 points from 1 to sqrt(20)
num_multi_arr = np.arange(1, 101, 1)             # 100 points from 1 to 100

# Generate the 3D meshgrid
grid_col1, grid_col2, grid_col3 = np.meshgrid(size_multi_arr, num_multi_arr, degs, indexing='ij')

# Flatten the arrays and stack them into a 2D array
flattened_array = np.column_stack((grid_col1.ravel(), grid_col2.ravel(), grid_col3.ravel()))

# Access a specific row using run_index
size_multi, num_multi, deg = flattened_array[run_index, :]

belgium = pd.read_csv('BelgiumMonthlySunspotNum.csv')
spot_num_time_series = np.round(belgium['Monthly_Mean_Sunspot_Number'].values*num_multi)
obs_phi = np.array([deg])*np.pi/180
spot_ratio = np.array([3/5, 1])
latitude_method = ['solar butterfly']
radii_method = [('Nagovitsyn', True, size_multi), ('Baumann Group Max', False, size_multi)]


results = make_observations_parallel(n_rotations = 1200, 
                            num_spots = spot_num_time_series,
                            radii_method = radii_method,
                            radii_probs = np.array([0.15, 0.85]),
                            latitude_method = latitude_method,
                            obs_phi = obs_phi,
                            num_spots_type = 'Dist',
                            suppress_output = True,
                            return_full_data = False, 
                            spot_ratio = spot_ratio, n_processes = 1,
                            num_surf_pts = 450**2, return_Rper = True, n_observations = 100)

results['obs_phi'] = obs_phi.tolist() if isinstance(obs_phi, np.ndarray) else obs_phi
results['spot_ratio'] = spot_ratio.tolist() if isinstance(spot_ratio, np.ndarray) else spot_ratio
results['radii_method'] = [(method, flag, float(size)) for method, flag, size in radii_method]  # Converting each `size_multi` to float
results['latitude_method'] = latitude_method
results['num_multi'] = float(num_multi)  # Ensuring it's a JSON-serializable number
results['size_multi'] = float(size_multi)

name = f'robust_{run_index}_{int(deg)}_{int(num_multi)}_{np.round(size_multi,2)}_new.json'

with open(name, "w") as file:
    json.dump(results, file)

time.sleep(60)
print('Done!')
