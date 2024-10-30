#======================================================#
#                      Preamble                        #
#======================================================#

# This code was written by Conaire Deagan, a PhD
# candidate at the University of New South Wales (UNSW)
# Email: c.deagan@unsw.edu.au
# Repository: github.com/ConaireD/SHERA_Stellar_Noise
# Feel free to reach out for assistance with this, or
# with any feature requests.
# I make no guarentees that this code is any good
# this code has not been comphrensively tests.

#=======================================================#
# These are the custom functions needed for the         #
# SHERA_Stellar_Noise jupyter files                     #
#=======================================================#

#------------------------------ Notes -----------------------------#
# - In general, theta refers to the longitudanal angle, and phi
#   refers to the latitudanal angle.
# - Additionally, this document follows the (rare) convention of
#   spot_phi / inclination where spot_phi = 0 means equatorial view,
#   and spot_phi = np.pi/2 is pole on. 


#----------------------- Import Packages---------------------------#

################
# Mathematical #
################
import numpy as np

####################
# Speed increasing #
####################
from multiprocessing import Pool
import functools
from numba import jit
# Note, the first call of a function will take longer than subsequent calls
# The functions here are not completely optimised. Some functions would
# require rewrites to be compatible with numba

############
# Plotting #
############
import matplotlib.pyplot as plt

#################
# Nice to haves #
#################
import os
import warnings
warnings.filterwarnings('ignore')
import timeit
from tqdm.notebook import tqdm

#------------------------------ Functions -------------------------------#

@jit()
def spherical_to_cartesian(thetas, phis, obs_theta=0, obs_phi=0,
                           get_angles = False):
    '''
    ===========================================================================
    Converts spherical co-ordinates to cartesian co-ordinates.
    ---------------------------------------------------------------------------
    thetas     - in radians
    phis       - in radians
    obs_theta  - the (longitude) angle that the viewer observes the points from
    obs_phi    - the (latitude) angle that the viewer observes the points from
    get_angles - if True, returns an array containing angles that each point
                 is from the centre
    ---------------------------------------------------------------------------
    Returns xs, ys - cartesian positions
            vis  - a bool indicating if point is visible
            c    - (optional) The angle between the observation vector and the
                   normal vector from the spot, see: 
                   https://en.wikipedia.org/wiki/Orthographic_map_projection
    ===========================================================================
    '''
    xs  = np.cos(phis)*np.sin(thetas - obs_theta)
    ys  = np.cos(obs_phi)*np.sin(phis) -                                      \
          np.sin(obs_phi)*np.cos(phis)*np.cos(thetas - obs_theta)
    
    c   = np.arccos(np.sin(obs_phi)*np.sin(phis) +                            \
                  np.cos(obs_phi)*np.cos(phis)*np.cos(thetas-obs_theta))
    vis = np.abs(c) < np.pi/2
    
    if get_angles == False:
        return xs, ys, vis
    else:
        return xs, ys, vis, c
    
@jit()
def cartesian_to_spherical(x, y, obs_theta = 0, obs_phi = 0):
    '''
    ===========================================================================
    Converts cartesian co-ordinates to spherical co-ordinates
    ---------------------------------------------------------------------------
    x - position of points in cartesian space
    y - position of points in cartesian space
    obs_theta - the (longitude) angle that the viewer observes the points from
    obs_phi   - the (latitude) angle that the viewer observes the points from
    ---------------------------------------------------------------------------
    Returns theta (longitude)
            phi   (latitude)
    ===========================================================================
    '''
    rho = np.sqrt(x**2 +y**2)
    c   = np.arcsin(rho)
    
    phi = np.arcsin(np.cos(c)*np.sin(obs_phi) +                               \
            (y*np.sin(c)*np.cos(obs_phi))/(rho))
    
    theta = obs_theta +                                                       \
            np.arctan2(x*np.sin(c), rho*np.cos(c)*np.cos(obs_phi) -           \
                       y*np.sin(c)*np.sin(obs_phi))
    return theta, phi

@jit()
def limb_darkening(x,y, u1 = -0.47, u2 = -0.23):
    '''
    ===========================================================================
    Calculates the limb darkening for the star. The default 'u' values are
    set according to the standard Sun 'u' values at 550nm. See:
    https://en.wikipedia.org/wiki/Limb_darkening
    ---------------------------------------------------------------------------
    x - the x position of a point on a star
    y - the y position of a point on a star
    u1 - the first coefficient of the standard quadratic limb-darkening model
    u2 - the second coefficient of the standard quadratic limb-darkening model
    ---------------------------------------------------------------------------
    Returns limb_darkening_map - An array containing the limb darkening values
                                 for each point
    ===========================================================================
    '''
    phi_map = np.sqrt(x**2 + y**2)
    limb_darkening_map = 1 + u1*(1-np.cos(phi_map)) + u2*(1-np.cos(phi_map))**2
    
    return limb_darkening_map


def generate_surface_points(num_pts, number_observations):
    '''
    ===========================================================================
    Creates an array of points that are as equally spaced as possible on the 
    surface of a sphere using a Fibonnaci spiral.
    ---------------------------------------------------------------------------
    num_pts             -  the number of points on the sphere
    number_observations - number of observations for one rotation of the star
    ---------------------------------------------------------------------------
    Returns thetas, phis - the spherical co-ordinates of each point
    ===========================================================================
    '''
    # Create evenly spaced Fibonacci sequence
    indices = np.arange(num_pts) + 0.5
    phi     = np.arccos(1 - 2*indices/num_pts)
    theta   = np.pi * (1 + 5**0.5) * indices

    # Convert to Cartesian coordinates
    xs = np.cos(theta) * np.sin(phi)
    ys = np.sin(theta) * np.sin(phi)
    zs = np.cos(phi)

    thetas = np.arctan2(ys, xs)
    phis   = np.arcsin(zs)
    
    phis   = np.repeat(phis[:,np.newaxis], number_observations, axis = 1)
    thetas = np.repeat(thetas[:,np.newaxis], number_observations, axis = 1)
    
    return thetas, phis

def get_points_within_circ_on_sphere(centre_phi, centre_theta, radius, n=5e5,
                                    n_observations = 72):
    '''
    ===========================================================================
    Takes in the theta/phi location of a star-spot, generates n points on the
    surface of the star and returns only the thetas/phis of the points within
    a circle of the spot on the surface. This circle is a circle on the sphere, 
    and not in cartesian space. 
    ---------------------------------------------------------------------------
    centre_phi     - The latitude of the centre of the star-spot (radians)
    centre_theta   - The longitude of the centre of the star-spot (radians)
    radius         - The radius of the star-spot (radians)
    n              - Number of points to generate on the surface of the star
    n_observations - Number of observations during one rotation of the star
    ---------------------------------------------------------------------------
    Returns thetas, phis - A number of points around the original given point.
    ===========================================================================
    '''
    thetas, phis = generate_surface_points(n, n_observations)

    ang      = np.sin(centre_phi)*np.sin(phis) +                              \
               np.cos(centre_phi)*np.cos(phis)*np.cos(centre_theta - thetas)
    ang_dist = np.arccos(ang)
    
    valid_indices = np.where(ang_dist < radius)[0]
    
    return thetas[valid_indices], phis[valid_indices]

def get_analytic_paths(obs_phi, spot_phi, spot_theta, num_pts = 72, contrast = 1):
    '''
    ===========================================================================
    Takes in a vector of spot_phi and spot_thetas, each representing single 
    surface quanta, calculates the impact that this spot has on the COM as the 
    star makes one full rotation.
    NOTE: This assumes solid body rotation and no differential rotation. 
    NOTE: This function is only to be used for a single spot. If there are
    multiple spots, use multi_analytic_paths instead.
    NOTE: In almost all cases, even with a single spot, you probably should use
    multi_analytic_paths instead.
    ---------------------------------------------------------------------------
    obs_phi    - The observing angle from the equator in radians
    spot_phi   - The latitude of the spot (radians)
    spot_theta - The longitude of the spot (radians)
    num_pts    - The number of observations
    contrast   - contrast of spot against background. 1 = totally black. 
    ---------------------------------------------------------------------------
    Returns x_pos, y_pos - Position of COM/Photometric centroid
            delta        - The angle between the observation vector and the
                           normal vector from the spot
    ===========================================================================
    '''
    obs_phi = np.arcsin(obs_phi) ## 
    
    # Rotation of the star
    obs_theta = np.arange(0, 2*np.pi, 2*np.pi/num_pts)+spot_theta 
    spot_theta = 0                                           
    
    # Angle between centre of sphere and spot                                           
    delta = np.sin(obs_phi)*np.sin(spot_phi) +                                \
    np.cos(obs_phi)*np.cos(spot_phi)*np.cos(spot_theta - obs_theta - np.pi/2)
        #                                     needed for some reason  --^
        
    # Tests if spot is visible in a orthographic projection
    occultation = np.abs(np.arccos(delta)) < np.pi/2
    
    # 'Small' circles turn into ellipses in orthographic projections
    a = np.cos(spot_phi) # <--- Get radius of 'small' circle on sphere
    b = np.cos(spot_phi)*np.sin(obs_phi) # <--- Projection effect of
                                         #     observation angle
    
    # Get x,y positions of spot as star rotates
    x_pos = a*np.cos(obs_theta)
    y_pos = b*np.sin(obs_theta) + np.sin(spot_phi)*np.cos(obs_phi) # <--\ 
        # Account for fact that the centre of the small circle is 'above' the 
        # centre of the sphere
    
    # Calculate limb darkening
    dark = limb_darkening(x_pos, y_pos)
    
    # Apply projection effects
    x_pos *= delta
    y_pos *= delta
    
    # Apply limb darkening 
    x_pos *= dark
    y_pos *= dark
    
    # Apply contrast
    x_pos *= contrast
    y_pos *= contrast
    
    # Consider if spot is visible
    x_pos *= occultation
    y_pos *= occultation

    #  flip pos to get com motion rather than anti-com motion
    # uses a mean here so that number of surface points doesn't 
    # impact signal strength
    return -np.mean(x_pos, axis=0), -np.mean(y_pos, axis=0)

def multi_analytic_paths(radii, thetas, phis, contrast ,obs_phi,
                         n_observations = 72, n_surface = 250**2):
    '''
    ===========================================================================
    Takes in the centre location of the spots, generates a number of evenly 
    spaced points around this point, and then calculates the path of all those
    points and combines them to get a better approximation.
    NOTE: This does not calculate the impact of all points, only points within
          the radius of a spot. This allows for a massive speed up, as the 
          remaining points `cancel out`.
    ---------------------------------------------------------------------------
    radii   - The radii of each spot
    thetas  - The centre latitude of each spot
    phis    - The centre longitude of each spot
    obs_phi - The observers observation angle
    n_observations - Number of observations
    n_surface      - Number of points to generate over the entire surface,
                     before being cropped down into only the number of points
                     within the spot.
    ---------------------------------------------------------------------------
    Returns total_x_path, total_y_path - the x and y locations of the observed
                                         signal.
    ===========================================================================
    '''
    phase_shift = int(n_observations/4)-1
    
    total_x_path = np.zeros((n_observations,))
    total_y_path = np.zeros((n_observations,))
    
    # For each spot, calculate signal independantly 
    for i in range(len(radii)):
        radius = radii[i]
        theta  = thetas[i]
        phi    = phis[i]
        cont   = contrast[i]

        # get surface points covered by spot
        more_thetas, more_phis = get_points_within_circ_on_sphere(phi, theta,
                                        radius, n_surface, 1)
        x_j, y_j = get_analytic_paths(obs_phi, more_phis, more_thetas,
                                     n_observations, cont)

        if np.isnan(x_j).any() == True:
            continue
        
        x_j -= np.mean(x_j)
        y_j -= np.mean(y_j)
        
        x_j = np.roll(x_j, phase_shift)[::-1]
        y_j = np.roll(y_j, phase_shift)[::-1]

        x_j = x_j*(radius**2)/np.sqrt(3)
        y_j = y_j*(radius**2)/np.sqrt(3)
        
        total_x_path += x_j
        total_y_path += y_j


    return total_x_path, total_y_path

def add_noise(x,y,scale = 1):
    '''
    ===========================================================================
    Adds 'realistic' instrumentational noise to data
    NOTE: This noise is generated by uniformly sampling a "mean" value between
    0.025 and 0.035 (representing .25 - .35 muas instrument error), and then 
    uses this mean as the centre of a gaussian distribution to calculate the
    magnitude of noise. Then, a random angle is chosen, and the noise offsets
    in the direction of this angle. Changing 'scale' to 2 would have the effect 
    of making the instrumentational noise choose a mean from between 0.5-0.7 muas
    # TODO:
       - Make this function more flexible in how noise is added. 
       - Also include biases and correlated noise. 
    ---------------------------------------------------------------------------
    x     - the x positions of observed data
    y     - the y positions of observed data
    scale - Allows control over the amount of noise added. The default value of
            scale = 1, is realistic for noise from the TOLIMAN instrument 
            observing Alpha Centauri A/B. 
            
    ---------------------------------------------------------------------------
    Returns x_noise, y_noise - The noisy data
                      sigmas - The uncertainties on each data point
    ===========================================================================
    '''
    angle_ran   = np.random.uniform(0,2*np.pi, size = len(x))
    uncertainty = np.random.uniform(0.025,0.035, size = len(x)) * 2.3e-3 
        # units of solar radii (converted to realistic range) --^
    mag = np.random.normal(0, uncertainty, size = len(x))
    x_noise = mag*np.cos(angle_ran)*scale + x
    y_noise = mag*np.sin(angle_ran)*scale + y
    
    uncertainties = uncertainty*scale
    
    sigmas = np.concatenate((uncertainties/np.sqrt(2),
                         uncertainties/np.sqrt(2)))
    
    return x_noise, y_noise, sigmas 

@jit()
def simple_logL(observed_data, sigmas, model_data, s = None):
    '''
    ===========================================================================
    Calculates the log-likelihood of the data assuming gaussian errors, from
    one error source.
    ---------------------------------------------------------------------------
    observed_data - The true observed data
    sigmas        - The errors from the observed_data
    model_data    - data from whatever model is being used.
    ---------------------------------------------------------------------------
    Returns logL  - The log Likelihood
    '''
    if s == None:
        logL = - (1/2) * np.sum(((observed_data - model_data)**2)/sigmas**2)
    else:
        sigma_eff_squared = sigmas**2 + s**2
        logL = -0.5 * np.sum(np.log(2 * np.pi * sigma_eff_squared) + (observed_data - model_data)**2 / sigma_eff_squared)

    return logL

def get_data(radii, spot_thetas, spot_phis, contrast, obs_phi, n_observations, 
             num_pts = 250**2, verbose = True, s = None):
    '''
    ===========================================================================
    A function that takes in the location, size, and contrast of spots, along 
    with the viewing angle (i.e. inclination of the star) and returns the
    astrometric signal. Also takes in the number of observations per star rotation
    and the number of surface points to generate.
    ---------------------------------------------------------------------------
    radii          - Array containing the radii of each spot (in radians)
    spot_thetas    - Array containing the longitude of each spot (in radians)
    spot_phis      - Array containing the latitude of each spot (in radians)
    contrast       - The spot contrast. 0 is no contrast, 1 is completely dark
                     spot
    obs_phi        - The viewing angle. Equivalent to the inclination of the star.
                     Note, 0 = viewing from the equator, np.pi/2 = pole on.
                     (in radians)
    n_observations - The number of data points per stellar rotation
    num_pts        - The number of points to simulate the surface of a star
    verbose        - If True, will output a figure showing the signal, and the
                     surface of a star
    s              - An argument passed to simple_logL to account for underestimated
                     sigmas, for MCMC purposes
    '''
    true_x, true_y = multi_analytic_paths(radii, spot_thetas, spot_phis,
                                          contrast,obs_phi, n_surface = num_pts,
                                          n_observations = n_observations)
    ###########################
    true_x = np.nan_to_num(true_x)
    true_y = np.nan_to_num(true_y)
    ############################
    noisy_x, noisy_y, sigmas = add_noise(true_x, true_y, scale = 1)
    observed_data = np.concatenate([noisy_x, noisy_y])
    clean_data    = np.concatenate([true_x, true_y])
    
    if verbose == False:
        return (observed_data, clean_data, sigmas)
    else:
        ############################
        # Calculate log likelihood #
        ############################
        clean_data_LL         = -simple_logL(observed_data, sigmas, clean_data, s=s)
        print('Model Log Likelihood: {:.2f}'.format(clean_data_LL))

        
        ###########################
        # Create Star Surface Map #
        ###########################
        thetas, phis = generate_surface_points(180**2, number_observations = 1)
        dist = np.arccos(np.sin(spot_phis)*np.sin(phis) +                     \
                         np.cos(spot_phis)*np.cos(phis) *                     \
                         np.cos(thetas-spot_thetas))    <= radii
        in_circ = np.any(dist, axis = 1) 

        fig, axs = plt.subplot_mosaic([['A', 'B']], figsize=(18, 6),
                                      gridspec_kw={'width_ratios': [1, 2]})

        axs['A'].axis('equal')
        axs['A'].plot(clean_data[:n_observations] * 1000,
                      clean_data[n_observations:] * 1000, ls='-',
                      color='tab:blue', alpha=1, lw=2,
                      label='Fundamental True path')

        axs['A'].errorbar(observed_data[:n_observations] * 1000,
                          observed_data[n_observations:] * 1000,
                          xerr = sigmas[:n_observations] * 1000,
                          yerr = sigmas[n_observations:] * 1000,
                          fmt='.', color='gray', alpha=0.4,
                          label=r'Observed Data (1$\sigma$ errors)')

        axs['A'].legend(loc='upper right')
        axs['A'].set_title('Observed Data')
        axs['A'].set_xlabel(r'Equatorial Photometric Deflection $mR_*$')
        axs['A'].set_ylabel(r'Polar Photometric Deflection $mR_*$')

        axs['B'] = plt.subplot2grid((1, 2), (0, 1), projection='aitoff')
        axs['B'].set_title('Star spot map')
        axs['B'].scatter(thetas, phis, s=5, c=in_circ, alpha = 0.5,
                         cmap = 'coolwarm')
        axs['B'].set_xlabel('\n Longitude')
        axs['B'].set_ylabel('Latitude')
        axs['B'].grid(color = 'black')
        axs['B'].tick_params(axis='x', colors='white')
        plt.show()
        
        print('')
        print('      Latitude,  Longitude,  Radii')
        print('      (Degrees), (Degrees), (Radians)')
        for i in range(len(radii)):
            print('Spot {}: {:.1f}          {:.1f},    {:.2f}'.format(
                i + 1, 
                spot_phis[i] * 180 / np.pi, 
                spot_thetas[i] * 180 / np.pi, 
                radii[i]
            ))
        print('\nObservation angle: {:.1f} degrees'.format(obs_phi[0]*180/np.pi))

        
        return (observed_data, clean_data, sigmas)

@jit()
def MSH_to_input_radii(MSH): # shouldn't there be a factor of pi here?
    '''
    ===========================================================================
    A function that transforms spots measured in micro-solar hemispheres (MSH)
    to the units of input radii - the angle between the centre of a spot to the 
    edge in units of radians
    ---------------------------------------------------------------------------
    MSH - Micro Solar Hemispheres
    ===========================================================================
    '''
    return np.arcsin(np.sqrt(MSH/1e6))

@jit()
def input_radii_to_MSH(r):# shouldn't there be a factor of pi here?
    '''
    ===========================================================================
    A function that transforms spots measured in radians to units of 
    micro-Solar hemispheres
    ===========================================================================
    '''
    return 1e6*(np.sin(r)**2)

# Likely I don't need this, as the spots size dist might account for this
def spot_decay_logNormal(area, time, timestep = 30):
    """
    ===========================================================================
    Calculate the decay of spot areas over time using a log-normal distribution

    This function simulates the decay of solar spot areas based on a log-normal 
    decay model. The decay rate is randomly generated and constrained to a minimum 
    value. This function is based equation 4 from Baumann and Solanki 2005 
    "On the size distribution of sunspot groups in the Greenwich sunspot record
    1874-1976"

    ---------------------------------------------------------------------------
    area    - An array representing the initial area of solar spots in 
           micro-solar hemispheres (MSH). The shape of this array determines
           the number of spots being analysed.

    time     - A 1D array representing the time in months over which the decay is 
           calculated.
    timestep - The time increment for decay calculations in days.

    Returns:
    ---------------------------------------------------------------------------
        A 2D array of the same shape as 'area' representing the new spot 
        areas after decay, constrained to be non-negative.

    Notes:
    -----
    - The decay is computed based on a log-normal distribution with a mean 
      of 1.75 and a standard deviation derived from a variance of 2. 
      
    """

    decay = np.random.lognormal(mean = 1.75, sigma = np.sqrt(2),
                                size = np.shape(area.flatten()))
    # decay is a rate in MSH/day
    decay[decay<1] = 1
    new_area = (np.sqrt(area) - ((decay*timestep)/np.sqrt(area))*time[:,np.newaxis])
    new_area[new_area < 0] = 0
    new_area = new_area**2
    return new_area.T

@jit()
def get_spot_sizes(n, reject_small = True, method = 'Nagovitsyn', scale = 1,
                   mean = None, sigma = None):
    '''
    ===========================================================================
    This function returns sunspot sizes in radians (radii), based off of a
    log-normal distribution
    ---------------------------------------------------------------------------
    n            - number of spots to return
    reject_small - if True, spots below 0.02 radians will be removed.
    method       - selects spot size distribution, see notes for more detail
    scale        - modifies the RADIUS of the spot size
    mean         - if method is None (or invalid), this value will be used for
                   the mean of the log-normal size distribution
    sigma        - if method is None (or invalid), this value will be used for
                   the standard deviation of the log-normal size distribution
    ---------------------------------------------------------------------------
    NOTES: Nagovitsyn method is based off of the value reported in 
           Nagovitsyn and Petsov 2016 - "On the presence of two populations of
           sunspots". Here, it uses the log-normal distribution for the long
           lived (larger) spots. These spots make up about 40-60% of all spots,
           and if this method is use, n should be adjusted
           
           The Baumann values come from Table 1 in Baumann and Solanki 2005 -
           "On the size distribution of sunspot groups in the Greenwich sunspot
           record 1874-1976"
    ===========================================================================
    '''
    if method == 'Nagovitsyn':
        mean  = np.log(10**2.377) # ~ np.log(238)
        sigma = np.log(10**0.414) # ~ np.log(
        
    elif method == 'Baumann Single Max':
        mean  = np.log(45.5)
        sigma = np.log(2.11)
        
    elif method == 'Baumann Single Snapshot':
        mean  = np.log(30.2)
        sigma = np.log(2.14)
    
    elif method == 'Baumann Group Max':
        mean  = np.log(62.2)
        sigma = np.log(2.45)
        
    elif method == 'Baumann Group Snapshot':
        mean  = np.log(58.6)
        sigma = np.log(2.49)        
        
    else:
        if mean or sigma == None:
            print('Either choose a method, or specify a mean and sigma for a log-normal dist')
        
    areas = np.random.lognormal(mean = mean, sigma = sigma, size = n)
    radii_radians = MSH_to_input_radii(areas) * scale
    
    if reject_small == True:
        radii_radians = radii_radians[radii_radians >= 0.02] 
    
    return radii_radians
        
def spot_butterfly_distribution(size, mean = None, sigma = None):
    '''
    ===========================================================================
    Returns a series of latitudes for sun-spots, following a static butterfly
    diagram (i.e. a double gaussian). The default values come from Ivanov et al.
    2011 - "Form of the latitude distribution of sunspot activity". The values
    are the average values from the table on pg 915
    ---------------------------------------------------------------------------
    size  - the number of latitudes wanted
    mean  - mean of normal distrubtion
    sigma - the standard deviation of the normal distribution
    ---------------------------------------------------------------------------
    returns latitude in DEGREES
    NOTE: This function does not take into account the time varying latitudinal
          distribution. The paper listed above has good model based on activity.
    ===========================================================================
    '''
    
    if mean == None and sigma == None:
        mean  = 14.9
        sigma = 6.1 

    latitudes = np.random.normal(loc = mean, scale = sigma, size = size)
    sign = 2*np.random.randint(0,2,size=size)-1
    
    return sign*latitudes

def spot_uniform_distribution(size):
    '''
    ===========================================================================
    Returns latitudes of size (size) sampled uniformly in latitude, adjust
    so that each latitude probability is proportional to its relative size.
    I.e. latitudes near the equator are more likely than higher latitudes
    ===========================================================================
    '''
    a = np.arccos(np.random.uniform(low = -1, high = 1, size = size))

    return a

def spot_latitude_selection(size, method = 'butterfly', mean = None, sigma = None):
    '''
    ===========================================================================
    A wrapper function that selects from available spot latitude distributions
    ---------------------------------------------------------------------------
    method - must be either 'butterfly' or 'uniform', which will call
             'spot_butterfly_distribution' and 'spot_latitude_distribution' 
             functions respectively. If input is 'solar butterfly', it will
             return butterfly with some modified parameters that match jitter
             estimates. 
    ===========================================================================
    '''
    if method == 'butterfly':
        return spot_butterfly_distribution(size, mean, sigma)
    elif method == 'solar butterfly':
        return spot_butterfly_distribution(size, mean = 25, sigma = 6)
    elif method == 'uniform':
        if mean != None or sigma != None:
            print('Warning: Uniform method does not take a mean or sigma argument')
        return spot_uniform_distribution(size)
    else:
        print('Invalid selection')
        
def bootstrap(data, n_boots = 10000, c_level = 0.95, return_CI = False, suppress_output = True):
    '''
    ===========================================================================
    A function to bootstrap data to estimate uncertainties
    ---------------------------------------------------------------------------
    data      - A 1d array or list of the data to bootstrap
    n_boots   - The number of bootstrap samples to do, default = 10000
    c_level   - The confidence level to calculate, if return_CI = True
    return_CI - Bool. If true, returns confidence interval. Defaults False.
    suppress_output - Bool. Defaults to true. Disables tqdm output. Users not on
                  computing clusters should set this to be false.

    ---------------------------------------------------------------------------
    Returns - std_error if return_CI = False
            - std_error, (ci_lower, ci_upper), if return_CI = True
    '''
    data = np.array(data)
    n_samples = len(data)
    
    bootstrap_stds = np.zeros(n_boots)
    
    # Do bootstrapping
    for i in tqdm(range(n_boots), desc = 'Bootstrapping...', disable = suppress_output):
        bootstrap_sample = np.random.choice(data, size=n_samples, replace=True)
        bootstrap_stds[i] = np.std(bootstrap_sample)

    std_error = np.std(bootstrap_stds)

    if return_CI == True:
        ci_lower = np.percentile(bootstrap_stds, (1 - c_level) * 100 / 2)
        ci_upper = np.percentile(bootstrap_stds, (1 + c_level) * 100 / 2)
        return std_error, (ci_lower, ci_upper)
    else:
        return std_error

def custom_radii_method_builder(radii_method_ls, radii_probs, 
                                return_radii_method_flag = False):
    '''
    ===========================================================================
    A function that returns a custom get_spot_sizes function.
    ---------------------------------------------------------------------------
    radii_method_ls - A list of methods to use. If the method is a string,
                      then that string is passed to get_spot_sizes as a method.
                      Else, the method must be a tuple of length 3 or 4. If 
                      length three, must be of the form (method, reject_small, scale)
                      and if length 4, must be of the form
                      (reject_small, mean, std, scale).
    radii_probs     - An array of probabilities for each method. Must sum to 1. 
    return_radii_method_flag
                    - A bool, default False, that if True will make 
                      custom_radii_method return the index of the method chosen.
                      To be used in conjunction with a non-None spot_ratio in
                      make_observations
                      
    ---------------------------------------------------------------------------
    Note: Used in the make_observation function.
    ===========================================================================
    '''
    if not np.isclose(np.sum(radii_probs),  1):
        raise ValueError("Sum of probabilities must = 1")

    def custom_radii_method(n_spots):
        
        # create a cumulative prob distribution, then choose random number
        cum_probs = np.cumsum(radii_probs)
        die_roll = np.random.uniform()

        # Based on random number, choose method
        # if random number is p_r, then the method chosen will be
        # p_1 + ... + p_n < p_r < p_1 + ... + p_n + p_(n+1)
        # where p_n is in radii_probs
        for i, threshold in enumerate(cum_probs):
            # choose method
            if die_roll < threshold:
                temp_radii_method = radii_method_ls[i]
                # decode method input
                if type(temp_radii_method) == str:
                    radii = get_spot_sizes(n_spots, method = temp_radii_method)
                    return (radii, i) if return_radii_method_flag else radii

                elif (type(temp_radii_method) == tuple) &  (len(temp_radii_method) == 3):
                    method, reject_small, scale = temp_radii_method
                    radii = get_spot_sizes(n_spots, method = method,
                                           reject_small = reject_small,
                                           scale = scale)
                    return (radii, i) if return_radii_method_flag else radii

                elif (type(temp_radii_method) == tuple) &  (len(temp_radii_method) == 4):
                    reject_small, mean, std, scale = temp_radii_method
                    radii = get_spot_sizes(n_spots, reject_small = reject_small,
                                           mean = mean, sigma = std,
                                           scale = scale)
                    return (radii, i) if return_radii_method_flag else radii

                else:
                    raise ValueError('Error in radii_method construction')
    return custom_radii_method

def custom_latitude_dist_builder(latitude_method):
    '''
    ===========================================================================
    A function that returns a custom spot_latitude_selection function.
    ---------------------------------------------------------------------------
    radii_method_ls - A string or tuple of the form described in 
                     'make_observations'
    ---------------------------------------------------------------------------
    Note: This function is probably needlessly complicated currently, but should
    hopefully allow for better expansion of latitudes if need be.
    ===========================================================================
    '''
    

    if len(latitude_method) == 1:
        if type(latitude_method[0]) == str:
            if latitude_method[0] == 'butterfly':
                # user wants to use default butterfly
                def custom_lat_fn(n_spots):
                    return spot_latitude_selection(n_spots, method = 'butterfly')
                
            elif latitude_method[0] == 'uniform':
                # user wants to use uniform dist
                def custom_lat_fn(n_spots):

                    return spot_latitude_selection(n_spots, method = 'uniform')

            elif latitude_method[0] == 'solar butterfly':
                # user wants to use uniform dist
                def custom_lat_fn(n_spots):
                    return spot_latitude_selection(n_spots, method = 'solar butterfly')

            else:
                raise ValueError('''
                Unrecognised latitude method. Supported methods are 'butterfly'
                or 'uniform'
                ''')     

    elif type(latitude_method) == tuple:
        if len(latitude_method) != 2:
            raise  ValueError('''
            latitude_method must either be a string, or a tuple of the form
            (method, (**params))
            ''')
        else:
            method = latitude_method[0]
            params = latitude_method[1]
            if method == 'butterfly' or method == 'solar butterfly':
                try:
                    l_mean = params[0]
                    l_std  = params[1]
                    # User wants custom butterfly method
                    
                    def custom_lat_fn(n_spots):
                        return spot_latitude_selection(n_spots,
                                                       method = method,
                                                       mean = l_mean,
                                                       sigma= l_std)

                except:
                    raise  ValueError('''
                Incorrectly formatted parameters to latitude method
                ''')

            else:
                raise  ValueError('''
            Currently unsupported latitude_method. 
            ''')
    else:
        raise  ValueError('latitude method must be a string or tuple')
        
    return custom_lat_fn        

                     
def make_observations(n_rotations, num_spots, radii_method, radii_probs,
                      latitude_method, obs_phi, num_surf_pts = 350**2, do_bootstrap = True,
                      num_spots_type = 'Number', spot_contrasts = 0.7, 
                      longitude_type = 'Random', n_observations = 10, 
                      return_full_data = False, suppress_output = True,
                      spot_ratio = None):
    '''
    ===========================================================================
    A single large flexible function that wraps many other functions that 
    generates astrometric jitter data.
    ---------------------------------------------------------------------------
    n_rotations     - The number of stellar rotations to model. Must be an int,
                     or None. If None, then num_spots must be a time series of
                     spot numbers, and the number of stellar rotations will be
                     inferred from the length of that list/array.
    num_spots       - The number of spots to model each rotation. If an int,
                     this number will be used each time, and num_spots_type must
                     be 'Number'. If a tuple, this must be of the form
                     (mean, std) and num_spots_type must be 'Gauss Dist'. If
                     an array, num_spots_type must be either 'Dist' or 
                     'Time Series'. If 'Dist', the spots will be randomly sampled
                     from the given array. If 'Time Series', each rotation will 
                     choose a sequential spot number.
    radii_method    - Must be a string, or a list of strings and tuples. Strings
                     must match options given in 'get_spot_sizes'. tuples must be
                     of the form (reject_small, mean, std, scale) or 
                     (method, reject_small, scale) to be passed to a logNormal
                     distribution
    radii_probs     - An array of probabilties to be assigned to each radii method.
                     Must be the same size as radii_method
    latitude_method - The method by which spot latitudes are calculated. Must be
                     either a str, or a tuple. If a string, must be either
                     'butterfly' or 'uniform'. If a tuple, must be of the form
                     (method, (**params)).
    obs_phi         - The inclination of the star in radians 
    num_surf_pts    - The number of points on the surface of the star to simulate.
    do_bootstrap    - If true, returns uncertainty on jitter by calling the
                     bootstrap function.
    num_spots_type  - A string that specifies the input given to num_spots. Must
                     either be 'Number', 'Gauss Dist', 'Dist', or 'Time Series'
    spot_contrasts  - Contrasts of spots. Every spot has the same contrast.
    longitude_type  - Currently does nothing. 
    n_observations  - number of observations to make per rotation
    return_full_data - Bool. If true, return all the measurements, not just 
                       summary statistics. Default false. 
    suppress_output - Bool. Defaults to true. Disables tqdm output. Users not on
                      computing clusters should set this to be false.
    spot_ratio      - if not None, when given will modify num_spots on a per method
                      basis. See solar special case for details.
    ---------------------------------------------------------------------------
    Returns a dict containing the results
    ---------------------------------------------------------------------------
    # TODO:
         - Allow for contrast distributions
         - Allow for different longitude types (such as 'clumpy')
    ===========================================================================
    '''
    # this is here because of dumb heritage reasons. 
    obs_phi = np.cos(obs_phi)
    ################
    # Special Case #
    ################
    if radii_method == 'solar':
        if latitude_method != 'solar butterfly':
            print('''
            Warning! Latitude method was not "solar butterfly", method will
            be changed to "solar butterfly".
            ''')
            
        return make_observations(n_rotations, num_spots,
                                 radii_method = ['Nagovitsyn', 'Baumann Group Max'],
                                 radii_probs = np.array([0.15, 0.85]),
                                 latitude_method = ['solar butterfly'],
                                 obs_phi = obs_phi, num_surf_pts = num_surf_pts,
                                 do_bootstrap = do_bootstrap, 
                                 num_spots_type = num_spots_type,
                                 spot_contrasts = spot_contrasts,
                                 longitude_type = longitude_type,
                                 n_observations = n_observations,
                                 return_full_data = return_full_data, 
                                 suppress_output = suppress_output, 
                                 spot_ratio = np.array([3/5, 1]))

    if spot_ratio is not None:
        return_radii_method_flag = True
    else:
        return_radii_method_flag = False
    
    if n_observations == 1:
        raise ValueError("Known bug hits when n_observations = 1. Increase it to atleast 2.")

    
    if num_spots_type in ['Number', 'Dist', 'Time Series']:
        if not np.all(np.mod(num_spots, 1) == 0):
            raise ValueError("With the given num_spot_type, all values must be integers, or integer-like")

    
    # CHECK RADII INPUTS AND MAKE CUSTOM RADII FUNCTION
    if not np.isclose(np.sum(radii_probs),  1):
        raise ValueError("Sum of probabilities must = 1")
    elif len(radii_probs) != len(radii_method):
        raise ValueError("Radii method list and radii probs must be the same size")
    else:
        custom_radii_function = custom_radii_method_builder(radii_method,
                                                            radii_probs,
                                                            return_radii_method_flag)

    
    
    # CHECK LATITUDE INPUTS AND MAKE CUSTOM LATITUDE FUNCTION
    custom_latitude_fn = custom_latitude_dist_builder(latitude_method)

    # BUILD OBSERVATION PIPELINE
    # CHECK SPOT INPUT TYPE AND MAKE SPOT NUMBER FUNCTION
    xs = np.array([])
    ys = np.array([])
    
    if num_spots_type == 'Number':
        if n_rotations is None:
            raise ValueError("Specify number of rotations")

        for i in tqdm(range(n_rotations), desc = 'Simulating Star...', disable=suppress_output):
            n_spots     = num_spots # this is a dumb variable name...
            if n_spots < 0:
                print('n_spots requested is negative. Setting to zero instead...')
                n_spots = 0
            radii       = custom_radii_function(n_spots)
            if return_radii_method_flag == True:
                # if true, modulates the number of spots by method. 
                radii, idx = radii
                n_spots = int(spot_ratio[idx]*n_spots)
                radii = radii[:n_spots]
                
            spot_phis   = custom_latitude_fn(n_spots) * np.pi/180
            spot_thetas = np.random.uniform(low = 0, high = 2*np.pi, size = n_spots)
            contrasts   = spot_contrasts*np.ones(np.shape(radii))

            _, clean_data, _ = get_data(radii, spot_thetas, spot_phis, contrasts, obs_phi,
                               n_observations = n_observations , num_pts = num_surf_pts,
                               verbose = False)

            x = clean_data[:n_observations]
            y = clean_data[n_observations:]
            xs = np.concatenate((xs, x))
            ys = np.concatenate((ys, y))
            
    elif num_spots_type == 'Gauss Dist':

        if n_rotations is None:
            raise ValueError("Specify number of rotations")
            
        # Implies user has given a mean and std
        for i in tqdm(range(n_rotations), desc = 'Simulating Star...', disable=suppress_output):
            n_mean = num_spots[0]
            n_std  = num_spots[1]
            n_spots = int(np.random.normal(loc = n_mean, scale = n_std, size = 1))
            if n_spots < 0:
                print('n_spots requested is negative. Setting to zero instead...')
                n_spots = 0
            radii       = custom_radii_function(n_spots)

            if return_radii_method_flag == True:
                # if true, modulates the number of spots by method. 
                radii, idx = radii
                n_spots = int(spot_ratio[idx]*n_spots)
                radii = radii[:n_spots]
            spot_phis   = custom_latitude_fn(n_spots) * np.pi/180

            spot_thetas = np.random.uniform(low = 0, high = 2*np.pi, size = n_spots)
            contrasts   = spot_contrasts*np.ones(np.shape(radii))
            _, clean_data, _ = get_data(radii, spot_thetas, spot_phis, contrasts, obs_phi,
                               n_observations = n_observations , num_pts = num_surf_pts,
                               verbose = False)

            x = clean_data[:n_observations]
            y = clean_data[n_observations:]
            xs = np.concatenate((xs, x))
            ys = np.concatenate((ys, y))
            
    elif num_spots_type == 'Dist':
        if n_rotations is None:
            raise ValueError("Specify number of rotations")
            
        # Implies user has given a distribution to sample from
        for i in tqdm(range(n_rotations), desc = 'Simulating Star...', disable=suppress_output):
            if n_spots < 0:
                print('n_spots requested is negative. Setting to zero instead...')
                n_spots = 0

            n_spots = np.random.choice(num_spots)
            radii       = custom_radii_function(n_spots)
            if return_radii_method_flag == True:
                # if true, modulates the number of spots by method. 
                radii, idx = radii
                n_spots = int(spot_ratio[idx]*n_spots)
                radii = radii[:n_spots]
            spot_phis   = custom_latitude_fn(n_spots) * np.pi/180
            spot_thetas = np.random.uniform(low = 0, high = 2*np.pi, size = n_spots)
            contrasts   = spot_contrasts*np.ones(np.shape(radii))

            _, clean_data, _ = get_data(radii, spot_thetas, spot_phis, contrasts, obs_phi,
                               n_observations = n_observations , num_pts = num_surf_pts,
                               verbose = False)

            x = clean_data[:n_observations]
            y = clean_data[n_observations:]
            xs = np.concatenate((xs, x))
            ys = np.concatenate((ys, y))
    
    elif num_spots_type == 'Time Series':
        if n_rotations is not None:
            print("Specified rotation number will be ignored, as num_spots_type is 'Time Series'")
        # User has given a time series to follow
        for n_spots in tqdm(num_spots, desc = 'Simulating Star...', disable=suppress_output):
            if n_spots < 0:
                print('n_spots requested is negative. Setting to zero instead...')
                n_spots = 0

            radii       = custom_radii_function(n_spots)
            if return_radii_method_flag == True:
                # if true, modulates the number of spots by method. 
                radii, idx = radii
                n_spots = int(spot_ratio[idx]*n_spots)
                radii = radii[:n_spots]
            spot_phis   = custom_latitude_fn(n_spots) * np.pi/180
            spot_thetas = np.random.uniform(low = 0, high = 2*np.pi, size = n_spots)
            contrasts   = spot_contrasts*np.ones(np.shape(radii))

            _, clean_data, _ = get_data(radii, spot_thetas, spot_phis, contrasts, obs_phi,
                               n_observations = n_observations , num_pts = num_surf_pts,
                               verbose = False)

            x = clean_data[:n_observations]
            y = clean_data[n_observations:]
            xs = np.concatenate((xs, x))
            ys = np.concatenate((ys, y))
    else:
        raise ValueError('''
        num_spot_types must be either 'Number', 'Gauss Dist', 'Dist', or 
        'Time Series'. 
        ''')

    # DO BOOTSTRAPPING
    #                   v convert to mR_\odot
    xs_std = np.std(xs*1000)
    ys_std = np.std(ys*1000)

    result = {'xs_std' : xs_std,
              'ys_std' : ys_std}
  
    if do_bootstrap == True: 
        xs_std_err, xs_std_ci = bootstrap(xs*1000, return_CI = True, 
                                          suppress_output = suppress_output)
        ys_std_err, ys_std_ci = bootstrap(ys*1000, return_CI = True,
                                          suppress_output = suppress_output)

        result['xs_std_err'] = xs_std_err
        result['ys_std_err'] = ys_std_err

        result['xs_std_ci'] = xs_std_ci
        result['ys_std_ci'] = ys_std_ci

    if return_full_data == True:
        result['all_xs'] = xs
        result['all_ys'] = ys

    return result
            

def process_single_rotation(args):
    """Helper function to process a single rotation for parallel processing"""
    n_spots, radii_method, radii_probs, latitude_method, obs_phi, n_observations, num_surf_pts, spot_contrasts, spot_ratio = args

    # set seed
    np.random.seed(np.random.seed(os.getpid()))
    
    if spot_ratio is not None:
        return_radii_method_flag = True
    else:
        return_radii_method_flag = False

    # Build the custom functions inside the worker process
    custom_radii_function = custom_radii_method_builder(radii_method, radii_probs, return_radii_method_flag)
    custom_latitude_fn = custom_latitude_dist_builder(latitude_method)
    
    if n_spots < 0:
        print('n_spots requested is negative. Setting to zero instead...')
        n_spots = 0
    
    n_spots = int(n_spots)
    radii   = custom_radii_function(n_spots)
    
    if return_radii_method_flag == True:
        # if true, modulates the number of spots by method. 
        radii, idx = radii
        n_spots = int(spot_ratio[idx]*n_spots)
        radii = radii[:n_spots]
    spot_phis = custom_latitude_fn(n_spots) * np.pi/180
    spot_thetas = np.random.uniform(low=0, high=2*np.pi, size=n_spots)
    contrasts = spot_contrasts * np.ones(np.shape(radii))
    
    _, clean_data, _ = get_data(radii, spot_thetas, spot_phis, contrasts, obs_phi,
                               n_observations=n_observations, num_pts=num_surf_pts,
                               verbose=False)
    
    x = clean_data[:n_observations]
    y = clean_data[n_observations:]
    return x, y


def make_observations_parallel(n_rotations, num_spots, radii_method, radii_probs,
                             latitude_method, obs_phi, num_spots_type='Number',
                             spot_contrasts=0.7, num_surf_pts=350**2,
                             do_bootstrap=True, longitude_type='Random',
                             n_observations=10, return_full_data=False,
                             suppress_output=True, spot_ratio = None, n_processes=None):
    """
    ===========================================================================
    Parallelized version of make_observations function.
    Additional parameter:
    n_processes: Number of processes to use. If None, uses all available CPU cores.
    ---------------------------------------------------------------------------
    NOTE: This function was written with Claude.ai. There may be errors, 
    especially with the order of the results. However, this likely isn't relevant
    to most use cases (i.e np.std does not care about order).
    ===========================================================================
    """
    obs_phi = np.cos(obs_phi)

    ################
    # Special Case #
    ################
    if radii_method == 'solar':
        if latitude_method != 'solar butterfly':
            print('''
            Warning! Latitude method was not "solar butterfly", method will
            be changed to "solar butterfly".
            ''')
            
        return make_observations_parallel(n_rotations, num_spots,
                                 radii_method = ['Nagovitsyn', 'Baumann Group Max'],
                                 radii_probs = np.array([0.15, 0.85]),
                                 latitude_method = ['solar butterfly'],
                                 obs_phi = obs_phi, num_surf_pts = num_surf_pts,
                                 do_bootstrap = do_bootstrap, 
                                 num_spots_type = num_spots_type,
                                 spot_contrasts = spot_contrasts,
                                 longitude_type = longitude_type,
                                 n_observations = n_observations,
                                 return_full_data = return_full_data, 
                                 suppress_output = suppress_output, 
                                 spot_ratio = np.array([3/5, 1]),
                                 n_processes=n_processes)



    
    if n_observations == 1:
        raise ValueError("Known bug hits when n_observations = 1. Increase it to at least 2.")
    
    if num_spots_type in ['Number', 'Dist', 'Time Series']:
        if not np.all(np.mod(num_spots, 1) == 0):
            raise ValueError("With the given num_spot_type, all values must be integers, or integer-like")
    
    # Check radii inputs
    if not np.isclose(np.sum(radii_probs), 1):
        raise ValueError("Sum of probabilities must = 1")
    elif len(radii_probs) != len(radii_method):
        raise ValueError("Radii method list and radii probs must be the same size")
    
    # Prepare data for parallel processing
    if num_spots_type == 'Number':
        if n_rotations is None:
            raise ValueError("Specify number of rotations")
        spot_numbers = [num_spots] * n_rotations
    elif num_spots_type == 'Gauss Dist':
        if n_rotations is None:
            raise ValueError("Specify number of rotations")
        n_mean, n_std = num_spots
        spot_numbers = np.random.normal(loc=n_mean, scale=n_std, size=n_rotations)
    elif num_spots_type == 'Dist':
        if n_rotations is None:
            raise ValueError("Specify number of rotations")
        spot_numbers = np.random.choice(num_spots, size=n_rotations)
    elif num_spots_type == 'Time Series':
        spot_numbers = num_spots
    else:
        raise ValueError("num_spot_types must be either 'Number', 'Gauss Dist', 'Dist', or 'Time Series'")
    
    # Create arguments for parallel processing
    process_args = [(n_spots, radii_method, radii_probs, latitude_method, obs_phi,
                    n_observations, num_surf_pts, spot_contrasts, spot_ratio)
                   for n_spots in spot_numbers]
    
    # Process in parallel
    with Pool(processes=n_processes) as pool:
        if not suppress_output:
            results = list(tqdm(pool.imap(process_single_rotation, process_args),
                              total=len(process_args),
                              desc='Simulating Star...'))
        else:
            results = pool.map(process_single_rotation, process_args)
    
    # Combine results
    xs = np.concatenate([x for x, _ in results])
    ys = np.concatenate([y for _, y in results])
    
    # Calculate statistics
    xs_std = np.std(xs * 1000)
    ys_std = np.std(ys * 1000)
    
    result = {'xs_std': xs_std, 'ys_std': ys_std}
    
    if do_bootstrap:
        xs_std_err, xs_std_ci = bootstrap(xs * 1000, return_CI=True,
                                        suppress_output=suppress_output)
        ys_std_err, ys_std_ci = bootstrap(ys * 1000, return_CI=True,
                                        suppress_output=suppress_output)
        
        result['xs_std_err'] = xs_std_err
        result['ys_std_err'] = ys_std_err
        result['xs_std_ci'] = xs_std_ci
        result['ys_std_ci'] = ys_std_ci
    
    if return_full_data:
        result['all_xs'] = xs
        result['all_ys'] = ys
    
    return result














    
                      