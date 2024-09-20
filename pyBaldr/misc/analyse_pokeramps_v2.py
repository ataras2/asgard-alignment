import glob
import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits
from scipy.optimize import least_squares
from scipy.interpolate import interp1d
from scipy.signal import find_peaks
import pandas as pd 
import os
import json
import corner
from scipy.integrate import quad

"""
    _summary_
    -cos(X-a) = cos( X + a)  .. so we fit 
    fitting :  -amplitude * np.cos(2 * np.pi * frequency * x - phase) + offset
    this seems more stable with the fitting for some reason
    
    We have a series of actuators that we poke with a series of values x and measure responses y 
    at a point where we observe maximum sensitivity
    we store x and y in seperate 1D arrays (each of equal length)
    we have the system model:
    
    y = A^2 + B^2 + 2 * A * B * cos( 2 * np.pi * F  * x + mu)
    
    and we have a measurement of A^2 , y, x
    we therefore seek to fit B, F, mu , with least square fitting with tight tolerences 
    initial guess B^2 = np.mean( y - A^2 ) , initial guess of F and mu from fourier analysis of the signal y - np.mean(y) 
    
    for each poked actuator (labelled actuator_number) we store the measured values A, x,y and fitted values B. F. mu in a 
    dictionary with the following structure for fitted and measured parameters 
    
    fit_dict[actuator_number][<*parameter*>_fit] = value(s)
    fit_dict[actuator_number][<*parameter*>_measured] = value(s) 
    
    we then seek to write this dictionary to a json file 
    
    
    Raises:
        TypeError: _description_

    Returns:
        _type_: _description_
"""


# Define the model
def model(x, A, B, F, mu):
    return A**2 + B**2 + 2 * A * B * np.cos(2 * np.pi * F * x  + mu)

# Residuals function for least squares fitting
def residuals(params, x, y, A):
    B, F, mu = params
    return model(x, A, B, F, mu) - y

# Function to estimate initial frequency and phase using Fourier analysis
def estimate_initial_F_mu(x, y):
    dx = np.mean(np.diff(x))
    fft_result = np.fft.fft(y - np.mean(y))  # Subtract mean for Fourier analysis
    frequencies = np.fft.fftfreq(len(y), d=dx)
    magnitudes = np.abs(fft_result)
    
    # Find the peak frequency (non-zero)
    non_zero_indices = np.where(frequencies != 0)
    peak_index = non_zero_indices[0][np.argmax(magnitudes[non_zero_indices])]
    F_guess = abs(frequencies[peak_index])
    
    # Estimate initial phase as zero (can refine this)
    mu_guess = 0
    
    return F_guess, mu_guess


# def extend_period(x, y):
#     # typically we barely cover one period in the measured data
#     # to better estimate we frequencies etc we extend the period via mirroring around peak values 
    
#     # Find the minimum value in y (global minimum)
#     min_index = np.argmin(y)
    
#     # Find local peaks after the minimum
#     peaks, _ = find_peaks(y[min_index:])
    
#     if len(peaks) == 0:
#         raise ValueError("No local peaks found after the minimum. Check the data.")
    
#     # Get the first local peak after the minimum
#     first_peak_index = min_index + peaks[0]

#     # Crop the data from minimum to the next local peak
#     x_cropped = x[min_index:first_peak_index+1]
#     y_cropped = y[min_index:first_peak_index+1]

#     dx = np.mean(np.diff( x_cropped )) 
#     x_next = np.arange( x_cropped[-1] + dx, x_cropped[-1] + dx + dx * len(x_cropped) , dx  ) 
#     # Reverse the cropped data to simulate another period
#     x_extended = np.concatenate([x_cropped, x_next])
#     y_extended = np.concatenate([y_cropped, y_cropped[::-1]])

#     return x_extended, y_extended


def plot_phase(angle_array):
    """
    Plot the phase of the angles on a unit circle.
    
    Args:
        angle_array (numpy array): Array of angles in radians.
    """
    # Create a unit circle
    circle = plt.Circle((0, 0), 1, color='black', fill=False, linestyle='--')
    
    # Set up the plot
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.add_artist(circle)
    
    # Plot points corresponding to angles on the unit circle
    for angle in angle_array:
        x = np.cos(angle)
        y = np.sin(angle)
        ax.plot([0, x], [0, y], 'r-', lw=1)  # Line from center to the point
        ax.plot(x, y, 'ro')  # Plot the point
    
    # Set axis limits
    ax.set_xlim([-1.1, 1.1])
    ax.set_ylim([-1.1, 1.1])
    
    # Set axis to equal aspect ratio
    ax.set_aspect('equal')
    
    # Add grid and labels
    ax.grid(True)
    ax.set_xlabel('Re')
    ax.set_ylabel('Im')
    
    # Add horizontal and vertical lines for reference
    ax.axhline(0, color='black', lw=1)
    ax.axvline(0, color='black', lw=1)
    
    plt.title("Phase Plot")
    plt.show()



def extend_period(x, y):
    # Find the minimum value in y (global minimum)
    min_index = np.argmin(y)
    
    # Find local peaks after the minimum
    peaks, _ = find_peaks(y[min_index:])
    
    if len(peaks) == 0:
        raise ValueError("No local peaks found after the minimum. Check the data.")
    
    # Get the first local peak after the minimum
    first_peak_index = min_index + peaks[0]

    # Crop the data from minimum to the next local peak
    x_cropped = x[min_index:first_peak_index+1]
    y_cropped = y[min_index:first_peak_index+1]

    # Compute the average dx between x points
    dx = np.mean(np.diff(x_cropped)) 

    # Extend the x values to simulate another period
    #x_next = np.arange(x_cropped[-1] + dx, x_cropped[-1] + dx * (len(x_cropped) + 1), dx)
    #x_next = np.arange(x_cropped[-1] + dx, x_cropped[-1] + dx * len(y_cropped), dx)
    x_next = np.linspace(x_cropped[-1] + dx, x_cropped[-1] + dx * len(y_cropped), len(y_cropped))
    
    # Extend the y values by mirroring them
    x_extended = np.concatenate([x_cropped, x_next])
    y_extended = np.concatenate([y_cropped, y_cropped[::-1]])

    return x_extended, y_extended

# Function to fit B, F, mu for each actuator
def fit_actuator(A, x, y):
    # Initial guess for B^2
    B_guess = np.sqrt((np.max(y) - np.min(y)) / 2) #np.sqrt(np.mean(y - A**2))

    # Estimate initial guesses for F and mu from Fourier analysis
    F_guess, mu_guess = estimate_initial_F_mu(x, y - np.mean(y))

    # Perform least-squares fitting
    initial_guess = [B_guess, F_guess, mu_guess]
    result = least_squares(residuals, initial_guess, args=(x, y, A), bounds=([0, 0, -np.pi], [np.inf, np.inf, np.pi]), xtol=1e-12, ftol=1e-12, gtol=1e-12 ) #, xtol=1e-9, ftol=1e-9)

    # Extract the fitted parameters
    B_fit, F_fit, mu_fit = result.x
    
    return B_fit, F_fit, mu_fit

# Example of creating the fit dictionary and saving to JSON
def fit_and_store_results(actuators_data, extend_period=False):
    fit_dict = {}

    # Iterate through each actuator and perform fitting
    for actuator_number, data in actuators_data.items():
        A = data['A_measured']
        x = data['x_measured']
        y = data['y_measured']
        
        if extend_period:
            # Crop and extend the data to simulate additional periods
            x_extended, y_extended = extend_period(x, y)

            # Perform the fitting
            B_fit, F_fit, mu_fit = fit_actuator(A,  x_extended,  y_extended)
        else:
            B_fit, F_fit, mu_fit = fit_actuator(A,  x,  y)
        # Store measured and fitted values in dictionary
        fit_dict[actuator_number] = {
            'A_measured': A,
            'x_measured': x.tolist(),  # Convert to list for JSON compatibility
            'y_measured': y.tolist(),
            'B_fit': B_fit,
            'F_fit': F_fit,
            'mu_fit': mu_fit,
            'residual':  residuals((B_fit, F_fit, mu_fit), x, y, A)
        }

    # Save the fit dictionary to a JSON file
    #with open('fit_results.json', 'w') as json_file:
    #    json.dump(fit_dict, json_file, indent=4)

    return fit_dict


# Function to plot y vs fitted model and residuals
def plot_fits_and_residuals(fit_dict, actuator_number , savefig=None):
    # Extract data from the fit dictionary
    A_measured = fit_dict[actuator_number]['A_measured']
    x_measured = np.array(fit_dict[actuator_number]['x_measured'])
    y_measured = np.array(fit_dict[actuator_number]['y_measured'])
    B_fit = fit_dict[actuator_number]['B_fit']
    F_fit = fit_dict[actuator_number]['F_fit']
    mu_fit = fit_dict[actuator_number]['mu_fit']
    residual = fit_dict[actuator_number]['residual']
    
    # Generate the fitted model values
    y_fitted = model(x_measured, A_measured, B_fit, F_fit, mu_fit)
    
    # Create a figure with two subplots (top and bottom)
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 6), gridspec_kw={'height_ratios': [3, 1]}, sharex=True)
    
    # Plot x vs y_measured and x vs y_fitted in the top plot
    ax1.plot(x_measured, y_measured, 'b-', label='Measured Data')
    ax1.plot(x_measured, y_fitted, 'r--', label='Fitted Model')
    ax1.set_ylabel('Intensity [ADU]')
    ax1.set_title(f'Actuator {actuator_number}: Measured vs Fitted Model')
    ax1.legend()
    ax1.axvline( x=0, ymin=0, ymax=np.max(y_measured), ls = ':', color='k')
    ax1.grid(True)
    
    # Plot x vs residuals in the bottom plot
    ax2.plot(x_measured,  residual , 'k-', label='Residuals [%]')
    ax2.axvline( x=0, ymin=0, ymax=np.max(100 * residual / y_measured ), ls = ':', color='k')
    ax2.set_xlabel('Normalized poke amplitude')
    ax2.set_ylabel('Residuals')
    ax2.grid(True)
    
    # Adjust layout to ensure the subplots don't overlap
    plt.tight_layout()
    
    if savefig is not None:
        plt.savefig( savefig ,dpi = 300, bbox_inches = 'tight')
        
    #plt.show()


def get_DM_command_in_2D(cmd,Nx_act=12):
    # function so we can easily plot the DM shape (since DM grid is not perfectly square raw cmds can not be plotted in 2D immediately )
    #puts nan values in cmd positions that don't correspond to actuator on a square grid until cmd length is square number (12x12 for BMC multi-2.5 DM) so can be reshaped to 2D array to see what the command looks like on the DM.
    corner_indices = [0, Nx_act-1, Nx_act * (Nx_act-1), Nx_act*Nx_act]
    cmd_in_2D = list(cmd.copy())
    for i in corner_indices:
        cmd_in_2D.insert(i,np.nan)
    return( np.array(cmd_in_2D).reshape(Nx_act,Nx_act) )



def get_phasemask_phaseshift( wvl, depth, dot_material = 'N_1405' ):
    """
    wvl is wavelength in micrometers
    depth is the physical depth of the phasemask in micrometers
    dot material is the material of phaseshifting object

    it is assumed phasemask is in air (n=1).
    N_1405 is photoresist used for making phasedots in Sydney
    """
    print( 'reminder wvl input should be um!')
    if dot_material == 'N_1405':
        # wavelengths in csv file are in nanometers
        df = pd.read_csv(hardcoded_data_path + 'Exposed Ma-N 1405 optical constants.txt', sep='\s+', header=1)
        f = interp1d(df['Wavelength(nm)'], df['n'], kind='linear',fill_value=np.nan, bounds_error=False)
        n = f( wvl * 1e3 ) # convert input wavelength um - > nm
        phaseshift = 2 * np.pi/ wvl  * depth * (n -1)
        return( phaseshift )
    
    else:
        raise TypeError('No corresponding dot material for given input. Try N_1405.')

def remove_outliers(data, threshold=3.5):
    """
    Remove outliers from a dataset using the modified Z-score method.
    
    Args:
        data (np.ndarray): Array of data points (each row corresponds to a set of parameters for an actuator).
        threshold (float): The threshold for the modified Z-score (default is 3.5).
    
    Returns:
        np.ndarray: The dataset with outliers removed.
        np.ndarray: Boolean mask indicating which rows are considered inliers (True = inlier, False = outlier).
    """
    # Calculate the median and MAD (median absolute deviation)
    median = np.median(data, axis=0)
    mad = np.median(np.abs(data - median), axis=0)
    
    # Calculate the modified Z-score for each point
    modified_z_score = 0.6745 * (data - median) / mad
    
    # Find rows where all modified Z-scores are below the threshold (i.e., no outliers in any column)
    inliers = np.all(np.abs(modified_z_score) < threshold, axis=1)
    
    # Return the data inliers and outliers 
    return data[inliers], inliers

# Example of removing outliers from the fitted parameters in fit_results
def extract_parameters(fit_results):
    """Extract fitted parameters from the fit_results dictionary."""
    parameters = []
    for actuator_number, results in fit_results.items():
        B_fit = results['B_fit']
        F_fit = results['F_fit']
        mu_fit = results['mu_fit']
        parameters.append([B_fit, F_fit, mu_fit])
    return np.array(parameters)


def remove_outliers_from_fit_results(fit_results, threshold=5, plot_before_removal=False):
    """
    Remove outliers from fit_results based on fitted parameters using the modified Z-score,
    and prompt the user to decide if outliers should be removed after visual inspection.
    
    Args:
        fit_results (dict): The dictionary containing fit results for each actuator.
        threshold (float): The threshold for the modified Z-score (default is 3.5).
        
    Returns:
        dict: The cleaned fit_results dictionary with outliers removed based on user input.
    """
    # Extract the parameters (B_fit, F_fit, mu_fit)
    parameters = extract_parameters(fit_results)
    
    # Remove outliers using modified Z-score
    parameters_cleaned, inliers = remove_outliers(parameters, threshold=threshold)
    
    # New dictionary to store the cleaned results
    fit_results_cleaned = {}
    
    # Iterate through actuators and prompt user for each one
    for i, actuator in enumerate(fit_results):
        
        if inliers[i]:
            # we keep it
            fit_results_cleaned[actuator] = fit_results[actuator]
        else:
            
            if plot_before_removal:
                # Plot the fit and residuals for this actuator
                plot_fits_and_residuals(fit_results, actuator_number=actuator)
                
                #plt.show() 
                
                # Prompt the user to decide whether to keep or remove the fit
                user_input = input(f"Actuator {actuator}: Enter 1 to remove (outlier), 0 to keep: ")
            else:
                user_input = '1' # we don't check and get rid of the outlier
                
            # Take action based on user input
            if user_input == '0':
                # Keep this fit
                fit_results_cleaned[actuator] = fit_results[actuator]
                
            elif user_input == '1':
                # Skip this fit (i.e., remove it as an outlier)
                print(f"Actuator {actuator} removed.")
                
            else:
                print("Invalid input. Keeping this fit by default.")
                fit_results_cleaned[actuator] = fit_results[actuator]
    
    return fit_results_cleaned



def plot_fitted_parameters_on_DM(fit_results, Nx_act=12, savefig=None):
    """
    Plot B_fit, F_fit, and mu_fit parameters projected onto the deformable mirror (DM).
    
    Args:
        fit_results (dict): The dictionary containing fit results for each actuator.
        Nx_act (int): Number of actuators along one axis (default is 12 for a 12x12 DM).
    """
    # Create command arrays for each fitted parameter (B_fit, F_fit, mu_fit)
    B_cmd = np.zeros(140)
    F_cmd = np.zeros(140)
    mu_cmd = np.zeros(140)
    
    for a in fit_results:
        B_cmd[a] = fit_results[a]['B_fit']
        F_cmd[a] = fit_results[a]['F_fit']
        mu_cmd[a] = fit_results[a]['mu_fit']
    
    # Convert the 1D command arrays into 2D grid for plotting
    B_cmd_2D = get_DM_command_in_2D(B_cmd, Nx_act=Nx_act)
    F_cmd_2D = get_DM_command_in_2D( 2 * np.pi * np.array( F_cmd ) , Nx_act=Nx_act)
    mu_cmd_2D = get_DM_command_in_2D(np.arccos( np.cos(mu_cmd)), Nx_act=Nx_act)
    
    # Create a figure with 3 subplots
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # Titles for the subplots
    titles = [r'$|\psi_r|$', 'F [rad]', r'$\mu$ [rad]']
    
    # Data for the subplots
    commands = [B_cmd_2D, F_cmd_2D, mu_cmd_2D]
    
    for i, ax in enumerate(axes):
        im = ax.imshow(commands[i], cmap='viridis')
        ax.set_title(titles[i])
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)  # Add colorbar with padding
    
    plt.tight_layout()
    if savefig is not None:
        plt.savefig( savefig ,dpi = 300, bbox_inches = 'tight')
        
    ##plt.show()



def plot_corner(fit_results):
    # Extract fitted parameters (B_fit, F_fit, mu_fit) from fit_results
    parameters = []
    for actuator_number, results in fit_results.items():
        B_fit = results['B_fit']
        F_fit = results['F_fit']
        mu_fit = results['mu_fit']
        parameters.append([B_fit, F_fit, mu_fit])
    
    # Convert the list of parameters to a NumPy array
    parameters_array = np.array(parameters)
    
    # Create the corner plot
    figure = corner.corner(parameters_array, 
                           labels=["B_fit", "F_fit", "mu_fit"],  # Label the parameters
                           show_titles=True,  # Display titles on the 1D histograms
                           title_fmt=".3f",  # Format the numbers on the titles
                           quantiles=[0.16, 0.5, 0.84],  # Show quantiles
                           title_kwargs={"fontsize": 12})  # Title font size
    
    # Show the plot
    #plt.show()

def planck_law(wavelength, T):
    """Returns spectral radiance (Planck's law) at a given wavelength and temperature."""
    h = 6.62607015e-34
    c = 299792458.0
    k = 1.380649e-23
    return (2 * h * c**2) / (wavelength**5) / (np.exp(h * c / (wavelength * k * T)) - 1)


# Function to find the weighted average wavelength (central wavelength)
def find_central_wavelength(lambda_cut_on, lambda_cut_off, T):
    # Define integrands for energy and weighted wavelength
    def _integrand_energy(wavelength):
        return planck_law(wavelength, T)

    def _integrand_weighted(wavelength):
        return planck_law(wavelength, T) * wavelength

    # Integrate to find total energy and weighted energy
    total_energy, _ = quad(_integrand_energy, lambda_cut_on, lambda_cut_off)
    weighted_energy, _ = quad(_integrand_weighted, lambda_cut_on, lambda_cut_off)
    
    # Calculate the central wavelength as the weighted average wavelength
    central_wavelength = weighted_energy / total_energy
    return central_wavelength



plt.ion()

hardcoded_data_path = '/Users/bencb/Documents/baldr/data_sydney/hardcoded_data/'

with open(hardcoded_data_path + 'phasemask_parameters.json') as f:
    phasemask_parameters = json.load(f)
    

fig_path = '/Users/bencb/Documents/baldr/data_sydney/analysis_scripts/analysis_results/pokeramp_results/'

if not os.path.exists( fig_path ):
    os.makedirs( fig_path )
    
files = glob.glob('/Users/bencb/Documents/baldr/data_sydney/A_FINAL_SYD_DATA_18-09-2024/tmp/09-09-2024/poke_ramp_data/pokeramp*.fits')



# sydney Baldr using DMLP1180 Longpass Dichroic Mirrors/Beamsplitters: 1180 nm Cut-Off Wavelength (Baldr in reflection)
# C-RED 2 cut on at 900nm 
# black body source at 1900 K 
# therefore central wavelength
wvl0 = find_central_wavelength(lambda_cut_on=900e-9, lambda_cut_off=1180e-9, T=1900)



cleaned_fits = {} 
for f in files:  
    #f = files[0] 

    mask_label = f.split('_MASK_')[-1].split('_')[0]
    
    theta = get_phasemask_phaseshift( wvl = 1e6 * wvl0, depth = phasemask_parameters[mask_label]["depth"], dot_material = 'N_1405' )

    #mu_theory_raw =  np.arctan( np.sin(theta) / (np.cos(theta) - 1) ) 

    mu_theory =  np.arctan2( np.sin(theta) , (np.cos(theta) - 1) ) #np.angle((np.exp(1J*theta)-1) ) #np.arctan( np.sin(theta) / (np.cos(theta) - 1) ) 
    #(np.pi - np.arccos( np.cos(mu_theory_raw) ) ) / 2

    d = fits.open(f)
    No_ramps = int(d['SEQUENCE_IMGS'].header['#ramp steps'])
    max_ramp = float(d['SEQUENCE_IMGS'].header['in-poke max amp'])
    min_ramp = float(d['SEQUENCE_IMGS'].header['out-poke max amp'])
    ramp_values = np.linspace(min_ramp, max_ramp, No_ramps)

    Nmodes_poked = int(d[0].header['HIERARCH Nmodes_poked'])
    Nact = int(d[0].header['HIERARCH Nact'])
    N0 = d['FPM_OUT'].data
    I0 = d['FPM_IN'].data

    poke_imgs = d['SEQUENCE_IMGS'].data[1:].reshape(No_ramps, 140, I0.shape[0], I0.shape[1])

    # well registered actuators 
    registration_threshold = 30
    registration_threshold = registration_threshold * np.mean( np.std(abs(poke_imgs - I0), axis=(0,1)) )

    well_registered_actuators = []
    for actuator_number in range(140):
        a = 2
        delta_img = poke_imgs[len(ramp_values)//2 + a][actuator_number] - poke_imgs[len(ramp_values)//2 - a][actuator_number]
        peak_delta = np.max( abs(delta_img) ) 
        if peak_delta > registration_threshold:
            well_registered_actuators.append( actuator_number )


    actuators_data = {}
    for actuator_number in well_registered_actuators:

        a = 2
        
        delta_img = poke_imgs[len(ramp_values)//2 + a][actuator_number] - poke_imgs[len(ramp_values)//2 - a][actuator_number]

        i, j = np.unravel_index(np.argmax(abs(delta_img)), I0.shape)

        x, y = ramp_values, poke_imgs[:, actuator_number, i, j] 

        actuators_data[actuator_number] = {'A_measured': N0[i,j]**0.5, 'x_measured': x[5:], 'y_measured': y[5:] } 


    fit_results = fit_and_store_results(actuators_data)

    fit_results_cleaned = remove_outliers_from_fit_results(fit_results, threshold=5, plot_before_removal=False)

    cleaned_fits[mask_label] = fit_results_cleaned
    
    
    ### PLOTTING 
    actno = 65
    savefig = fig_path + f'{mask_label}_I_vs_cmd_act{actno}_DM_diameter-{phasemask_parameters[mask_label]["diameter"]}_depth-diameter={phasemask_parameters[mask_label]["depth"]}_wvl-{wvl0}.png'
    plot_fits_and_residuals(fit_results, actuator_number = actno, savefig=savefig)

    savefig = fig_path + f'{mask_label}_parameter_fits_on_DM_{actno}_DM_diameter-{phasemask_parameters[mask_label]["diameter"]}_depth-diameter={phasemask_parameters[mask_label]["depth"]}_wvl-{wvl0}.png'
    plot_fitted_parameters_on_DM(fit_results_cleaned , savefig=savefig)

    """B_fit, F_fit, mu_fit = extract_parameters( fit_results_cleaned ).T

    F_fit_all.append( F_fit )
    
    plt.figure(figsize=(8,5)) 
    #plt.hist( (np.pi - np.arccos( np.cos(mu_fit) ) ) / 2 , bins=10, label='measured',alpha=0.7)
    plt.hist( np.arccos( np.cos(mu_fit) ) , bins=10, label='measured',alpha=0.7)
    plt.axvline(  np.arccos( np.cos(  mu_theory))  , color='k', ls=':', label='theory' )
    plt.xlabel(r'$\mu$ [rad]', fontsize=15)
    plt.ylabel( 'counts', fontsize=15)
    plt.xlim([0, np.pi*2])
    plt.gca().tick_params(labelsize=15)
    plt.legend()
    #plt.show()
    plt.savefig( fig_path + f'{mask_label}_phase_histogram_diameter-{phasemask_parameters[mask_label]["diameter"]}_depth-diameter={phasemask_parameters[mask_label]["depth"]}_wvl-{wvl0}.png',dpi=300, bbox_inches='tight')


    plt.figure(figsize=(8,5)) 
    #plt.hist( (np.pi - np.arccos( np.cos(mu_fit) ) ) / 2 , bins=10, label='measured',alpha=0.7)
    plt.hist( np.rad2deg( 2 * ( np.arccos( np.cos(mu_fit) ) - np.pi/2) ) , bins=10, label='ZWFS measurement', alpha=0.7)
    plt.axvline(  np.rad2deg(theta), color='k', ls=':', label='atomic force microscopy measurement' )
    plt.xlabel(r'phaseshift [deg]', fontsize=15)
    plt.text( 50, 5, r'$\lambda$'+f'={round( wvl0 * 1e6,3)}um' )
    plt.ylabel( 'counts', fontsize=15)
    plt.xlim([0, 180])
    plt.gca().tick_params(labelsize=15)
    plt.legend()
    #plt.show()
    plt.savefig( fig_path + f'{mask_label}_theta_histogram_diameter-{phasemask_parameters[mask_label]["diameter"]}_depth-diameter={phasemask_parameters[mask_label]["depth"]}_wvl-{wvl0}.png',dpi=300, bbox_inches='tight')
        
    print( theta )
    
    theta_zwfs = 2 * ( np.arccos( np.cos(mu_fit) ) - np.pi/2)
    
    abs( B_fit / (np.exp(1j * theta_zwfs) - 1 ) )
    #plot_phase( [np.mean(np.arccos( np.cos(mu_fit) ))  , np.arccos( np.cos(  mu_theory))  ] )
    """


fig_mu, (axAmu,axBmu) = plt.subplots(1,2,figsize=(14,7),sharey=True, layout="compressed")
mu_J_all = []
mu_H_all = []
F_fit_all = []
for f in np.sort( files ):
    mask_label = f.split('_MASK_')[-1].split('_')[0]

    B_fit, F_fit, mu_fit = extract_parameters( cleaned_fits[mask_label] ).T
    
    F_fit_all.append( F_fit )
    
    if 'J' in mask_label:
        # using linear approx mu_fit = theta/2 + pi/2 ( good to within 1% over full range)
        mu_J_all.append( np.rad2deg( 2 * ( np.arccos( np.cos(mu_fit) ) - np.pi/2) ) )
        axAmu.hist( np.rad2deg( 2 * ( np.arccos( np.cos(mu_fit) ) - np.pi/2) ) , bins=10, label=f'ZWFS measurement {mask_label}', alpha=0.7, histtype='step')
    else:
        mu_H_all.append( np.rad2deg( 2 * ( np.arccos( np.cos(mu_fit) ) - np.pi/2) ) )
        axBmu.hist( np.rad2deg( 2 * ( np.arccos( np.cos(mu_fit) ) - np.pi/2) ) , bins=10, label=f'ZWFS measurement {mask_label}', alpha=0.7, histtype='step')

thetaJ = get_phasemask_phaseshift( wvl = 1e6 * wvl0, depth = phasemask_parameters["J1"]["depth"], dot_material = 'N_1405' )
thetaH = get_phasemask_phaseshift( wvl = 1e6 * wvl0, depth = phasemask_parameters["H1"]["depth"], dot_material = 'N_1405' )

axAmu.axvline(  np.rad2deg(thetaJ), color='k', ls=':', lw=2, label='atomic force microscopy measurement' )
#axAmu.axvline( np.mean([x for xs in mu_J_all for x in xs]  ), color='k', ls='-', lw=2, label='mean') 
axBmu.axvline(  np.rad2deg(thetaH), color='k', ls=':', lw=2, label='atomic force microscopy measurement' )
#axBmu.axvline( np.mean([x for xs in mu_H_all for x in xs]  ), color='k', ls='-', lw=2, label='mean') 
for axx in [axAmu, axBmu]:
    axx.set_xlim([0, 180])
    axx.set_xlabel(r'phaseshift [deg]', fontsize=15)
    axx.tick_params(labelsize=15)
    axx.text( 50, 5, r'$\lambda$'+f'={round( wvl0 * 1e6,3)}um' ,fontsize=15 )
    axx.legend()
axAmu.set_ylabel( 'counts', fontsize=15)
    
fig_mu.savefig( fig_path + f'theta_histogram_ALL_wvl-{wvl0}.png',dpi=300, bbox_inches='tight')


plt.figure(figsize=(8,5)) 
aa = 1e6 * wvl0 * np.array( [x for xs in F_fit_all for x in xs] )
plt.hist( aa , bins=10, label='measured',alpha=0.7) # rad = 2*pi*F 

#plt.hist( np.arccos( np.cos(mu_fit) )  / 2 , bins=10, label='measured',alpha=0.7)
plt.axvline(    np.mean( aa )  , color='k', ls=':', label='mean' )
plt.xlabel(r'OPD/$\Delta c$ [$\mu$m]', fontsize=15)
plt.ylabel( 'counts', fontsize=15)
plt.gca().tick_params(labelsize=15)
plt.legend()
#plt.show()
plt.savefig( fig_path + f'F_histogram_ALL_MASKS_wvl-{wvl0}.png',dpi=300, bbox_inches='tight')


plt.close('all')


