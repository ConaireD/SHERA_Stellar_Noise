import sys
import os
import numpy as np
import pandas as pd
import json
import time

from Functions import *


degs = np.arange(0,91,1)

my_string = sys.argv[1]
run_index = extract_index(my_string)



size_multi_arr = np.linspace(1, np.sqrt(10), 25)  # 10 points from 1 to sqrt(10)
num_multi_arr = np.linspace(1, 5, 25)             # 5 points from 1 to 5

# Generate the meshgrid and flatten to 2D array
grid_col1, grid_col2 = np.meshgrid(size_multi_arr, num_multi_arr)
flattened_array = np.column_stack((grid_col1.ravel(), grid_col2.ravel()))

size_multi, num_multi = flattened_array[run_index,:]

belgium = pd.read_csv('BelgiumMonthlySunspotNum.csv')
spot_num_time_series = np.round(belgium['Monthly_Mean_Sunspot_Number'].values*num_multi)
obs_phi = np.array([75])*np.pi/180
spot_ratio = np.array([1])
latitude_method = ['butterfly']
radii_method = [('Baumann Group Max', True, size_multi)]


results = make_observations_parallel(n_rotations = 120, 
                            num_spots = spot_num_time_series,
                            radii_method = radii_method,
                            radii_probs = np.array([1]),
                            latitude_method = latitude_method,
                            obs_phi = obs_phi,
                            num_spots_type = 'Dist',
                            suppress_output = False,
                            return_full_data = False, 
                            spot_ratio = spot_ratio, n_processes = 4,
                            num_surf_pts = 450**2, return_Rper = True)

results['obs_phi'] = obs_phi
results['radii_method'] = radii_method
results['latitude_method'] = latitude_method
results['spot_ratio'] = spot_ratio
results['num_multi'] = num_multi
results['size_multi'] = size_multi

name = f'Rper_combination_{run_index}_75deg.json'

with open(name, "w") as file:
    json.dump(results, file)

time.sleep(60)
print('Done!')