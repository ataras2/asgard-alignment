import numpy as np 
import toml
import argparse
import zmq
import time
import toml
import os 
import matplotlib.pyplot as plt
import glob
from scipy.interpolate import interp1d
from astropy.io import fits

import scipy.signal as signal

from scipy.signal import TransferFunction,welch, csd, dlti, dstep



def offset_ts( t, data, offset='10std', savefig='delme.jpeg',**kwargs):
    # data has shape (sample, location). plots a time series of 
    # data[:,i + offset]

    if 'std' in offset:
        global_std = np.std(data)
        offset_val = float(offset.split('std')[0]) * global_std
    else:
        offset_val = float( offset )


    fig, ax = plt.subplots(figsize=(12, 8))

    num_actuators = data.shape[1]

    for i in range(num_actuators):
        baseline = i * offset_val
        trace = data[:, i]
        
        # Plot the trace offset by its baseline
        ax.plot(t, trace + baseline, lw=0.8)
        
        # Plot a dashed horizontal line at the baseline (zero level for that actuator trace)
        ax.axhline(y=baseline, color='grey', linestyle='--', alpha=0.3)

    xlabel = kwargs.get("xlabel",None)
    ylabel = kwargs.get("ylabel",None)
    title = kwargs.get("title",None)
    fontsize = kwargs.get("fontsize", 15)
    labelsize = kwargs.get("labelsize", 15)

    ax.set_xlabel(xlabel,fontsize=fontsize)
    ax.set_ylabel(ylabel,fontsize=fontsize)
    ax.set_title(title,fontsize=fontsize)
    ax.tick_params(labelsize=labelsize)

    plt.tight_layout()
    if savefig is not None:
        plt.savefig(savefig,bbox_inches='tight',dpi=200)



def waterfall_ts(t, data, savefig=None, **kwargs):
    """
    Create a waterfall plot (heatmap) with time on the y-axis and locations (e.g., actuators) on the x-axis.

    Parameters:
        t (array-like): 1D array of time stamps (length n_time). Must be monotonically increasing.
        data (np.ndarray): 2D array of shape (n_time, n_locations) representing the data values.
        savefig (str or None): If provided, the filename to save the figure.
        **kwargs: Additional keyword arguments:
            - xlabel (str): Label for the x-axis. Default: "Location".
            - ylabel (str): Label for the y-axis. Default: "Time".
            - title (str): Plot title. Default: "Waterfall Plot".
            - fontsize (int): Font size for labels and title. Default: 15.
            - cmap (str): Colormap for the heatmap. Default: "viridis".
            - colorbar (bool): Whether to add a colorbar. Default: True.
            - aspect (str or float): Aspect ratio for imshow. Default: "auto".
    
    Returns:
        fig, ax: The matplotlib Figure and Axes objects.
    """
    xlabel = kwargs.get("xlabel", None)
    ylabel = kwargs.get("ylabel", None)
    title = kwargs.get("title", None)
    fontsize = kwargs.get("fontsize", 15)
    cmap = kwargs.get("cmap", "viridis")
    add_colorbar = kwargs.get("colorbar", True)
    aspect = kwargs.get("aspect", "auto")
    
    n_locations = data.shape[1]
    

    fig, ax = plt.subplots(figsize=(12, 8))
    
    im = ax.imshow(data, aspect=aspect, origin='lower', 
                   extent=[0, n_locations - 1, t[0], t[-1]],
                   cmap=cmap)
    
    if add_colorbar:
        plt.colorbar(im, ax=ax)
    
    ax.set_xlabel(xlabel, fontsize=fontsize)
    ax.set_ylabel(ylabel, fontsize=fontsize)
    ax.set_title(title, fontsize=fontsize)
    plt.tight_layout()
    
    if savefig is not None:
        plt.savefig(savefig, bbox_inches='tight', dpi=200)
    
    return fig, ax



def plot_psd( f_list, psd_list, savefig = None, **kwargs ):

    xlabel = kwargs.get("xlabel","Frequency [Hz]")
    ylabel = kwargs.get("ylabel", "Power Spectral Density")
    title = kwargs.get("title", None)
    fontsize = kwargs.get("fontsize", 15)
    labelsize = kwargs.get("labelsize", 15)
    plot_cumulative = kwargs.get("plot_cumulative",True)

    plt.figure( figsize=(8,5) )

    for f, psd in zip( f_list, psd_list) :
        df = np.mean( np.diff( f ) )
        plt.loglog( f, psd , color='k') 
        if plot_cumulative:
            plt.loglog(f, np.cumsum(psd[::-1] * df )[::-1], color='k', ls=':',linewidth=2, label=f"Reverse Cumulative")

    plt.gca().tick_params(labelsize=labelsize)
    plt.xlabel(xlabel,fontsize=fontsize)
    plt.ylabel(ylabel,fontsize=fontsize)
    plt.title( title )

    #plt.title("Pixel-wise Power Spectral Density (Welch)")
    plt.legend(fontsize=12)
    #plt.grid(True, which="both", linestyle="--", alpha=0.5)
    #plt.tight_layout()
    if savefig is not None:
        plt.savefig( savefig, dpi=200, bbox_inches = 'tight')
    plt.show()

    
def convert_12x12_to_140(arr):
    # Convert input to a NumPy array (if it isn't already)
    arr = np.asarray(arr)
    
    if arr.shape != (12, 12):
        raise ValueError("Input must be a 12x12 array.")
    
    # Flatten the array (row-major order)
    flat = arr.flatten()
    
    # The indices for the four corners in a 12x12 flattened array (row-major order):
    # Top-left: index 0
    # Top-right: index 11
    # Bottom-left: index 11*12 = 132
    # Bottom-right: index 143 (11*12 + 11)
    corner_indices = [0, 11, 132, 143]
    
    # Delete the corner elements from the flattened array
    vector = np.delete(flat, corner_indices)
    
    return vector

def process_dm_signals( sig ):
    # convert 12x12 DM shape to BMC input 140 on 2D or 3D array (1st axis is time)  
    if len( np.array(sig).shape )==2:

        flat_sig = np.array( convert_12x12_to_140(sig)  )

    elif len( np.array(sig).shape )==3:
        # first axis is sample
        flat_sig = np.array( [convert_12x12_to_140(s) for s in sig] )
    else:
        raise UserWarning("input should be 2D or 3D. ")
    
    return flat_sig
#------------------------------------------
# readin 
#------------------------------------------

## our analysis should generally try to include 2 files 
#   - one with all gains = 0 and the input disturb 
#   - one with the desired gain to analyse 

# turbulence should use the same random seed! 

# closed loop 
f_disturb_CL = "/home/asg/Videos/CL_kolmogorov_AO/disturb_telem.fits"
f_telem_CL = "/home/asg/Videos/CL_kolmogorov_AO/CL_beam2_maskH3_zonal_kolmogorov_r0-0.1.fits"

# open loop 
f_disturb_OL = "/home/asg/Videos/CL_kolmogorov_AO/disturb_telem.fits"
f_telem_OL = "/home/asg/Videos/CL_kolmogorov_AO/CL_beam2_maskH3_zonal_kolmogorov_r0-0.1.fits"

# what we want to extract from telemetry
extnames = ["e_HO",
            "u_HO", 
            "current_dm_ch1", 
            "current_dm_ch2", 
            "current_dm_ch3",
            "exterior_sig",
            "secondary_sig"]

interpolated_data = {} 

for (f_dist,f_telem), lab in zip([ ( f_disturb_CL, f_telem_CL ), (f_disturb_OL,f_telem_OL )], ["CL","OL"]) :
    # interpolate signals onto the same temporal grid from the control loop and the disturbance telemetry 
    d_disturb = fits.open( f_dist )
    d_telem = fits.open( f_telem )

    t_disturb = d_disturb["TIME"].data # np.array( [d_disturb["TIME"].data[i][0] for i in range(len(d_disturb["TIME"].data) ) ] ) #d_disturb["TIME"].data # DM disturbance 
    t_cam =  d_telem["time_cam"].data # control frame read
    t_dm =  d_telem["time_dm"].data # control DM time 

    t0 = np.min( [np.min(t_cam), np.min(t_dm)])
    t1 = np.max( [np.max(t_cam), np.max(t_dm)])

    dt = np.min( [np.mean( np.diff( t_disturb ) ) , np.mean( np.diff( t_cam ) ) ] )

    t_grid = np.arange( t0 , t1, dt )
    # t_grid = np.arange( np.min(t_cam ), np.max( t_cam), np.mean( np.diff( t_cam ) )/2)
    #------------------------------------------
    # set up interpolate function 
    #------------------------------------------
    disturb_interp = interp1d(t_disturb, d_disturb["DM_CMD"].data, axis=0, kind='linear',bounds_error=False, fill_value=np.nan) 

    interp_fn_dict = {}

    for extname in extnames:
        if "current_dm" in extname: #then we interpolate to the DM update timestamp
            if len(d_telem[extname].data.shape)>1:
                interp_fn_dict[extname] = interp1d(d_telem["time_dm"].data, d_telem[extname].data, axis=0, kind='linear',bounds_error=False, fill_value=np.nan) 
            else:
                interp_fn_dict[extname] = interp1d(  d_telem["time_dm"].data, d_telem[extname].data, kind='linear',bounds_error=False, fill_value=np.nan )
        else:#then we interpolate to the CAMERA update timestamp
            if len(d_telem[extname].data.shape)>1:
                interp_fn_dict[extname] = interp1d( d_telem["time_cam"].data, d_telem[extname].data, axis=0, kind='linear',bounds_error=False, fill_value=np.nan) 
            else:
                interp_fn_dict[extname] = interp1d(  d_telem["time_cam"].data, d_telem[extname].data, kind='linear',bounds_error=False, fill_value=np.nan)
        
    #------------------------------------------
    # Interpolate
    #------------------------------------------
    data_interp = {} 

    data_interp["t_grid"] = t_grid

    data_interp["disturbance"] = disturb_interp(t_grid) 
    for extname in extnames:
        data_interp[extname] = interp_fn_dict[extname](t_grid) 

    ###### FINAL PRODUCT 
    interpolated_data[lab] = data_interp

    d_telem.close()
    d_disturb.close( )

#------------------------------------------
# Analysis
#------------------------------------------



### Analysis 
mode = 65 
fs = 1/np.mean(np.diff(t_grid))
nperseg = 512


# only look where we have common non NAN data 
tFilt = np.isfinite(process_dm_signals( interpolated_data["CL"]["disturbance"])[:,mode] ) * np.isfinite( interpolated_data["CL"]["e_HO"][:, mode] )
# Power Spectral Densities 
# closed loop data 
f, S_dd = welch(process_dm_signals( interpolated_data["CL"]["disturbance"] )[tFilt , mode], fs=fs, nperseg=nperseg)
_, S_ee = welch(interpolated_data["CL"]["e_HO"][tFilt , mode], fs=fs, nperseg=nperseg)
_, S_uu = welch(interpolated_data["CL"]["u_HO"][tFilt , mode], fs=fs, nperseg=nperseg)

# open loop (ki=0) versions 
_, S_dd_0 = welch(process_dm_signals( interpolated_data["OL"]["disturbance"] )[tFilt , mode], fs=fs, nperseg=nperseg)
_, S_ee_0 = welch(interpolated_data["OL"]["e_HO"][tFilt , mode], fs=fs, nperseg=nperseg)  # Open loop

# Cross Spectral Densities (CL)
_, S_ed = csd(interpolated_data["CL"]["e_HO"][tFilt , mode], process_dm_signals( interpolated_data["CL"]["disturbance"] )[tFilt , mode] , fs=fs, nperseg=512)  # e/d
_, S_ud = csd(interpolated_data["CL"]["u_HO"][tFilt , mode], process_dm_signals( interpolated_data["CL"]["disturbance"] )[tFilt , mode], fs=fs, nperseg=512)  # u/d
_, S_ue = csd(interpolated_data["CL"]["u_HO"][tFilt , mode], interpolated_data["CL"]["e_HO"][tFilt , mode], fs=fs, nperseg=512)  # u/e




plot_psd( f_list = [f], psd_list=[S_dd], savefig = 'delme.png') 






############# SOME ANALYSIS 


# Known controller: Ki only (integrator)
Ki = 0.8
K_f = Ki / (1j * 2 * np.pi * f)  # Controller in freq domain: K(s) = Ki / s

# === Transfer Function Estimates ===

# Sensitivity Function (two methods)
S_f = S_ed / S_dd               # Method 1: e/d via CSD
S_f2 = S_ee / S_ee_0            # Method 2: PSD ratio e_CL / e_OL

# Complementary Sensitivity
T_f = S_ud / S_dd               # u/d via CSD

# Open Loop Transfer Function  Hjw = Ped / Pdd  
#L_f = T_f / S_f                 # S_ee_0 / S_dd_0   # # L = T / S S_ud/S_dd / S_ed / S_dd =  S_ud / S_ed
# OR 
L_f  = S_f / (1 - K_f * S_f)

# Control Sensitivity (F = C / (1 + PC))
F_f = S_ue / S_ee               # u/e

# Plant Estimate (G = L / C)
G_f = L_f / K_f                 # G = P(s) = L / C

# Optional: Inverse Plant (for checking model error)
G_inv_f = K_f / L_f             # Could also check phase lag, etc.

# Package into dict for optional return
tf_dict = {
    "f": f,
    "S_f": S_f,
    "S_f2": S_f2,
    "T_f": T_f,
    "L_f": L_f,
    "F_f": F_f,
    "K_f": K_f,
    "G_f": G_f
}

# Max stable gain = 1 / |G|
max_stable_gain = 1.0 / np.abs(G_f)

# Plotting
plt.figure(figsize=(12, 6))

plt.subplot(2, 2, 1)
plt.semilogx(f, 20 * np.log10(np.abs(S_f)))
#plt.semilogx(f, 20 * np.log10(np.abs(S_f2)))
plt.title("Sensitivity |S(f)| [dB]")
plt.xlabel("Frequency [Hz]")
plt.grid(True)

plt.subplot(2, 2, 2)
plt.semilogx(f, 20 * np.log10(np.abs(T_f)))
plt.title("Complementary Sensitivity |T(f)| [dB]")
plt.xlabel("Frequency [Hz]")
plt.grid(True)

plt.subplot(2, 2, 3)
plt.semilogx(f, 20 * np.log10(np.abs(L_f)))
plt.title("Open Loop |L(f)| [dB]")
plt.xlabel("Frequency [Hz]")
plt.grid(True)

plt.subplot(2, 2, 4)
plt.loglog(f, max_stable_gain)
plt.title("Max Stable Gain 1/|G(f)|")
plt.xlabel("Frequency [Hz]")
plt.ylabel("Gain")
plt.grid(True)

plt.tight_layout()
plt.show()

## PHASE 

plt.figure(figsize=(12, 6))

plt.subplot(2, 2, 1)
#plt.semilogx(f, 20 * np.log10(np.abs(S_f)))
plt.semilogx(f, np.rad2deg( np.angle(S_f)))
plt.title("Sensitivity |S(f)| ")
plt.xlabel("Frequency [Hz]")
plt.grid(True)

plt.subplot(2, 2, 2)
plt.semilogx(f, np.rad2deg( np.angle(T_f)))
plt.title("Complementary Sensitivity |T(f)| [dB]")
plt.xlabel("Frequency [Hz]")
plt.grid(True)

plt.subplot(2, 2, 3)
plt.semilogx(f, np.rad2deg( np.angle(L_f)) )
plt.title("Open Loop |L(f)| [dB]")
plt.xlabel("Frequency [Hz]")
plt.grid(True)

plt.subplot(2, 2, 4)
plt.loglog(f, max_stable_gain)
plt.title("Max Stable Gain 1/|G(f)|")
plt.xlabel("Frequency [Hz]")
plt.ylabel("Gain")
plt.grid(True)

plt.tight_layout()
plt.show()


### GETTING MAX GAINS 
# we could do this for all modes when we have better disturbance telemetry ! 

# Hjw = S_ed / S_dd  

# for ki in np.logspace(-3, 1, 50):
#     Cjw = ki / (1j * 2 * np.pi * f) # controller TF 
#     Gjw_test = Hjw / (1 - Hjw * Cjw) # CL TF 
    
#     if np.nanmax(np.abs(Gjw_test * Cjw)) < 1:
#         ki_max = ki
#     else:
#         break


Hjw = tf_dict["L_f"] # S_ed / S_dd  

phase_margin_limit = 30  # degrees

Hjw = S_ed / S_dd 
ki_max = None
for ki in np.logspace(-3, 5, 500):
    Cjw = ki / (1j * 2 * np.pi * f)         # Controller: Ki / s
    Gjw_test = Hjw / (1 - Hjw * Cjw)        # Estimate of plant G(jw)
    Ljw = Cjw[1:] * Gjw_test[1:]                    # Open-loop TF , index from 1 to avoid zero freqzz

    L_mag = np.abs(Ljw)
    L_phase = np.angle(Ljw, deg=True)      # Phase in degrees

    # Find gain crossover frequencies (where |L| ~ 1)
    idx = np.argmin(np.abs(L_mag - 1))

    if L_mag[idx] < 0.95 : #or L_mag[idx] > 1.05:
        # No valid gain crossover → system is stable
        ki_max = ki
        continue

    phase_margin = 180 + L_phase[idx]
    if phase_margin > phase_margin_limit:
        ki_max = ki
    else:
        break

print(f"Maximum stable Ki with > {phase_margin_limit}° phase margin: {ki_max:.4f}")

# Plot results
plt.figure(figsize=(8, 5))
plt.semilogy(range(N_modes), ki_max, 'o-', label='Max Stable $k_i$')
plt.xlabel("Mode Index")
plt.ylabel("Max Stable Integral Gain $k_i$")
plt.title("Estimated Maximum Stable Integral Gains (Synthetic System)")
plt.grid()
plt.legend()
plt.show()

print("Estimated max stable gains (synthetic system):", ki_max)



### Optimize 

def cost_fn(Kp, Ki, Kd, G_f, f, phase_margin_limit=30):
    s = 1j * 2 * np.pi * f
    C_f = Kp + Ki / s + Kd * s
    L_f = G_f * C_f
    S_f = 1 / (1 + L_f)

    # Check phase margin at gain crossover
    mag = np.abs(L_f)
    phase = np.angle(L_f, deg=True)
    idx = np.argmin(np.abs(mag - 1))
    phase_margin = 180 + phase[idx]

    if phase_margin < phase_margin_limit or mag[idx] < 0.8:
        return np.inf  # Reject unstable or low-margin configs

    return np.trapz(np.abs(S_f)**2, f)  # Integrated sensitivity

from scipy.optimize import minimize
def wrapped_cost(params):
    Kp, Ki, Kd = params
    # skip first index which is usually 0 Hz <- 1/0
    return cost_fn(Kp, Ki, Kd, G_f=tf_dict["G_f"][1:], f=tf_dict["f"][1:])

res = minimize(wrapped_cost, x0=[0.1, 0.1, 0.0], bounds=[(1e-6, 1000)]*3)
Kp_opt, Ki_opt, Kd_opt = res.x
print(f"Optimal PID gains: Kp={Kp_opt:.4f}, Ki={Ki_opt:.4f}, Kd={Kd_opt:.4f}")

print( cost_fn(Kp=Kp_opt, Ki= Ki_opt, Kd=Kd_opt, G_f=tf_dict["G_f"][1:], f=tf_dict["f"][1:]) )

NN = 10
kdgrid = np.logspace( -4, , NN)
kigrid = np.linspace(0,1,NN)
cost_map = np.zeros([NN,NN] )
for i,kd in enumerate( kdgrid ):
    for j, ki in enumerate( kigrid ):
        cost_map[i,j] = cost_fn(Kp=0, Ki=ki, Kd=kd, G_f=tf_dict["G_f"][1:], f=tf_dict["f"][1:])


plt.figure()
plt.imshow( np.log10( cost_map ) )
plt.colorbar()
plt.show() 

