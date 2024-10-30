import sys
import os
import numpy as np
import pandas as pd
import json

from Functions import *


degs = np.arange(0,91,1)

my_string = sys.argv[1]
run_index = extract_index(my_string)


belgium = pd.read_csv('BelgiumMonthlySunspotNum.csv')
spot_num_time_series = np.round(belgium['Monthly_Mean_Sunspot_Number'].values)

obs_phi = np.array([degs[run_index]])*np.pi/180
radii_method = ['solar']
spot_ratio = np.array([1])
latitude_method = ['uniform']

results = make_observations_parallel(n_rotations = 12*300, 
                            num_spots = solar_maxima_spot_num_dist,
                             radii_method = [('Nagovitsyn', True, 1), ('Baumann Group Max', False, 1)],
                             radii_probs = np.array([0.15, 0.85]),
                            latitude_method = latitude_method,
                            obs_phi = obs_phi,
                            num_spots_type = 'Dist',
                            suppress_output = True,
                            return_full_data = False, 
                            spot_ratio = spot_ratio, n_processes = 8,
                            num_surf_pts = 450**2)

results['obs_phi'] = obs_phi
results['radii_method'] = radii_method
results['latitude_method'] = latitude_method
results['spot_ratio'] = spot_ratio
results['radii_probs'] = np.array([0.15, 0.85])

name = f'all_gen_uniform_obs_phi_{degs[run_index]}.json'

with open(name, "w") as file:
    json.dump(results, file)

