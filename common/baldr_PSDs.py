

import numpy as np 
import zmq
import time
import toml
import os 
import argparse
import matplotlib.pyplot as plt
import argparse
import subprocess
import glob
from astropy.io import fits
from scipy.signal import TransferFunction,welch, csd, bode, dlti, dstep
from asgard_alignment.DM_shm_ctrl import dmclass
import common.DM_basis_functions as dmbases
import common.phasemask_centering_tool as pct
import pyBaldr.utilities as util 
import datetime
from xaosim.shmlib import shm
from asgard_alignment import FLI_Cameras as FLI


parser = argparse.ArgumentParser(description="Interaction and control matricies.")

default_toml = os.path.join("/usr/local/etc/baldr/", "baldr_config_#.toml") 

# Camera shared memory path
parser.add_argument(
    "--global_camera_shm",
    type=str,
    default="/dev/shm/cred1.im.shm",
    help="Camera shared memory path. Default: /dev/shm/cred1.im.shm"
)

# TOML file path; default is relative to the current file's directory.
parser.add_argument(
    "--toml_file",
    type=str,
    default=default_toml,
    help="TOML file pattern (replace # with beam_id) to write/edit. Default: ../config_files/baldr_config.toml (relative to script)"
)

# Beam ids: provided as a comma-separated string and converted to a list of ints.
parser.add_argument(
    "--beam_id",
    type=int,
    default=2,
    help="beam to look at"
)

parser.add_argument(
    "--phasemask",
    type=str,
    default="H3",
    help="Comma-separated beam IDs to apply. Default: 1,2,3,4"
)

parser.add_argument(
    "--DM_flat",
    type=str,
    default="baldr",
    help="What flat do we use on the DM during the calibration. either 'baldr' or 'factory'. Default: %(default)s"
)

parser.add_argument(
    '--cam_fps',
    type=int,
    default=1500,
    help="frames per second on camera. Default: %(default)s"
)


parser.add_argument(
    '--cam_gain',
    type=int,
    default=20,
    help="camera gain. Default: %(default)s"
)

parser.add_argument(
    '--number_of_samples',
    type=int,
    default=1000,
    help="camera gain. Default: %(default)s"
)

parser.add_argument(
    "--noll_inidices_2_look_at",
    type=int,
    nargs='+',  # one or more integers
    default=[1,2,3,4,5],
    help="Noll inidicies to consider tip is 1, tilt 2 etc. Default: %(default)s"
)


args=parser.parse_args()





def plot_ts( t_list, sig_list, savefig = None, **kwargs ):

    xlabel = kwargs.get("xlabel","Time [s]")
    ylabel = kwargs.get("ylabel", "Signal")
    title = kwargs.get("title", None)
    fontsize = kwargs.get("fontsize", 15)
    labelsize = kwargs.get("labelsize", 15)
    labels = kwargs.get("labels", [None for _ in range(len(t_list))])
    colors = kwargs.get("colors", ["k" for _ in range(len(t_list))])
    plt.figure( figsize=(8,5) )

    for i, (t, s) in enumerate( zip( t_list, sig_list) ) :
        
        plt.plot( t, s , color=colors[i], label = f"{labels[i]}") 
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

def plot_psd( f_list, psd_list, savefig = None, **kwargs ):

    xlabel = kwargs.get("xlabel","Frequency [Hz]")
    ylabel = kwargs.get("ylabel", "Power Spectral Density")
    title = kwargs.get("title", None)
    fontsize = kwargs.get("fontsize", 15)
    labelsize = kwargs.get("labelsize", 15)
    plot_cumulative = kwargs.get("plot_cumulative",True)
    labels = kwargs.get("labels", [None for _ in range(len(f_list))])
    colors = kwargs.get("colors", ["k" for _ in range(len(f_list))])
    plt.figure( figsize=(8,5) )

    for i, (f, psd) in enumerate( zip( f_list, psd_list) ) :
        df = np.mean( np.diff( f ) )
        plt.loglog( f, psd , color=colors[i] , label = f"{labels[i]}") 
        if plot_cumulative:
            plt.loglog(f, np.cumsum(psd[::-1] * df )[::-1], color=colors[i], alpha =0.5, ls=':', linewidth=2) #, label=f"{labels[i]} Reverse Cumulative")

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



### Set up 

# read in calibrated matricies and configurations 
with open(args.toml_file.replace('#',f'{args.beam_id}'), "r") as f:
    config_dict = toml.load(f)
    # Baldr pupils from global frame 
    baldr_pupils = config_dict['baldr_pupils']
    I2A = config_dict[f'beam{args.beam_id}']['I2A']
    
    pupil_mask = config_dict.get(f"beam{args.beam_id}", {}).get("pupil_mask", {}).get("mask", None)

    secondary_mask = np.array(config_dict.get(f"beam{args.beam_id}", {}).get("pupil_mask", {}).get("secondary", None) )

    exterior_mask = np.array(config_dict.get(f"beam{args.beam_id}", {}).get("pupil_mask", {}).get("exterior", None) )





#  camera object with server communication
c = FLI.fli(args.global_camera_shm, roi = [None,None,None,None]) #baldr_pupils[f"{args.beam_id}"]

# change to append master dark , bias , bad pixel mask 
c.send_fli_cmd(f"set gain {args.cam_gain}") 
time.sleep(1)
c.send_fli_cmd(f"set fps {args.cam_fps}")
time.sleep(1)

# make sure the camera cofig internal state is correct 
assert float(c.config['fps']) == float(args.cam_fps)
assert float(c.config['gain']) == float(args.cam_gain)

# check for recent calibration files in the current setting 
valid_cal_files = util.find_calibration_files(mode=c.config['mode'], gain=int(c.config['gain']), target_fps=float(c.config['fps']), base_dir="MASTER_DARK", time_diff_thresh=datetime.timedelta(2), fps_diff_thresh=10)

# if no valid ones than we make some
if not valid_cal_files: 
    print( "no valid calibration files within the last few days. Taking new ones! ")
    script_path = "/home/asg/Progs/repos/dcs/calibration_frames/gen_dark_bias_badpix.py"
    params = ["--gains", f"{int(c.config['gain'])}", 
              "--fps", f"{c.config['fps']}", 
              "--mode", f"{c.config['mode']}", #"--mode", f"{c_dict[args.beam_id[0]].config['mode']}", 
              "--method", "linear_fit" ]
    try:
        # Run the script and ensure it completes
        with subprocess.Popen(["python", script_path]+params, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True) as process:
            stdout, stderr = process.communicate()  # Wait for process to complete

            if process.returncode == 0:
                print("Script executed successfully!")
                print(stdout)  # Print standard output (optional)
            else:
                print(f"Script failed with error:\n{stderr}")

    except Exception as e:
        print(f"Error running script: {e}")


# get darks, bias and some raw darks to make bad pixels (we dont use the premade ones cause we adjust here the parameters)
bias_fits_files = util.find_calibration_files(mode=c.config['mode'], gain=int(c.config['gain']) , target_fps=float(c.config['fps']), base_dir="MASTER_BIAS", time_diff_thresh=datetime.timedelta(2), fps_diff_thresh=10) #glob.glob(f"/home/asg/Progs/repos/dcs/calibration_frames/products/{tstamp_rough}/MASTER_BIAS/*.fits") 
dark_fits_files = util.find_calibration_files(mode=c.config['mode'], gain=int(c.config['gain']) , target_fps=float(c.config['fps']), base_dir="MASTER_DARK", time_diff_thresh=datetime.timedelta(2), fps_diff_thresh=10) #glob.glob(f"/home/asg/Progs/repos/dcs/calibration_frames/products/{tstamp_rough}/MASTER_DARK/*.fits") 
raw_darks_files =  util.find_calibration_files(mode=c.config['mode'],gain=int(c.config['gain']) , target_fps=float(c.config['fps']), base_dir="RAW_DARKS", time_diff_thresh=datetime.timedelta(2), fps_diff_thresh=10) #glob.glob(f"/home/asg/Progs/repos/dcs/calibration_frames/products/{tstamp_rough}/RAW_DARKS/*.fits") 

filename_reduction_dict = {} # to hold the files used for reduction 

for lab, ff in zip(['bias','dark'], [bias_fits_files, dark_fits_files] ):
    # Assumes we just took one!!! would be quicker to check subdirectories for one that matches the mode and gain with nearest fps. 
    most_recent = max(ff, key=os.path.getmtime) 

    filename_reduction_dict[lab+'_file'] = most_recent

    with fits.open( most_recent ) as d:
        c.reduction_dict[lab].append(  d[0].data.astype(int) )       # for beam_id in args.beam_id:
        #     r1,r2,c1,c2 = baldr_pupils[f'{beam_id}']
        #     c_dict[beam_id].reduction_dict[lab].append(  d[0].data.astype(int)[r1:r2, c1:c2] )

# bad pixels 
most_recent = max(raw_darks_files , key=os.path.getmtime) 
filename_reduction_dict["raw_darks_file"] = most_recent

with fits.open( most_recent ) as d:

    bad_pixels, bad_pixel_mask = FLI.get_bad_pixels( d[0].data, std_threshold=4, mean_threshold=10)
    bad_pixel_mask[0][0] = False # the frame tag should not be masked! 
    #c_dict[beam_id].reduction_dict['bad_pixel_mask'].append(  (~bad_pixel_mask ).astype(int) )
    c.reduction_dict['bad_pixel_mask'].append(  (~bad_pixel_mask ).astype(int) )


# set up DM SHMs 
print( 'setting up DMs')
dm = dmclass( beam_id=args.beam_id )
if args.DM_flat.lower() == 'factory':
    # activate flat (does this on channel 1)
    dm.activate_flat()
elif args.DM_flat.lower() == 'baldr':
    # apply dm flat + calibrated offset (does this on channel 1)
    dm.activate_calibrated_flat()
    

# Move to phase mask
# for beam_id in args.beam_id:
#     message = f"fpm_movetomask phasemask{beam_id} {args.phasemask}"
#     res = send_and_get_response(message)
#     print(f"moved to phasemask {args.phasemask} with response: {res}")

# automatic alignment 
# fine-alignment.py

# project to zernike modes 
modal_basis = dmbases.zer_bank(1, np.max( args.noll_inidices_2_look_at ) + 2 ) 

# define the registered pupil on the DM coordinates 
dm_pup = I2A @ np.array(pupil_mask).astype(int).reshape(-1)
dm_pup_2D = util.get_DM_command_in_2D( dm_pup )
frames = c.get_some_frames(number_of_frames = 1000, apply_manual_reduction=True)

r1,r2,c1,c2 = baldr_pupils[f"{args.beam_id}"]

# individual frames
ii = frames[:,r1:r2,c1:c2]
ii_dm =  (I2A @ ii.reshape(ii.shape[0], -1 ).T).T

# average intensity
i = np.mean(frames,axis=0)[r1:r2,c1:c2]
i_dm = I2A @ i.reshape(-1)


# signal
ss = np.array( [ util.get_DM_command_in_2D( iii - i_dm ) for iii in ii_dm ])

# Now build IM to project onto modes 



IM = []
poke_amp = 0.03
for i,m in enumerate([modal_basis[ix] for ix in args.noll_inidices_2_look_at]):
    print(f'executing cmd {i}/{len(modal_basis)}')
    I_plus_list = []
    I_minus_list = []
    for sign in [(-1)**n for n in range(10)]: #[-1,1]:

        dm.set_data(  sign * poke_amp/2 * m ) 
        
        time.sleep(500/float(c.config["fps"])) # 200 because get data takes 200 frames

        imgtmp_global = c.get_data(apply_manual_reduction = True )

        if sign > 0:
            
            I_plus_list.append( list( np.mean( imgtmp_global[:,r1:r2,c1:c2], axis = 0)  ) )
            #I_plus *= 1/np.mean( I_plus )

        if sign < 0:
            
            I_minus_list.append( list( np.mean( imgtmp_global[:,r1:r2,c1:c2], axis = 0)  ) )
            #I_minus *= 1/np.mean( I_minus )


    I_plus = np.mean( I_plus_list, axis = 0).reshape(-1) #/ normalized_pupils[beam_id].reshape(-1)
    I_minus = np.mean( I_minus_list, axis = 0).reshape(-1) #/  normalized_pupils[beam_id].reshape(-1)
    
    errsig = I2A @ ( (I_plus - I_minus)  / poke_amp ) # 1 / DMcmd * (s * gain)  projected to DM space

    IM.append( list(  errsig.reshape(-1) ) ) 

# check the modes
#util.nice_heatmap_subplots([modal_basis[ix] for ix in args.noll_inidices_2_look_at], savefig='delme.png' )

# check the response
util.nice_heatmap_subplots( [util.get_DM_command_in_2D( imm ) for imm in IM], savefig='delme.png' )


I2M = np.linalg.pinv(IM ) # fine for small number of well defined modes

#util.nice_heatmap_subplots( [ util.get_DM_command_in_2D(  I2M.T @ IM[2] ) ],savefig='delme.png')
# util.nice_heatmap_subplots( [util.get_DM_command_in_2D( dm_pup), i, util.get_DM_command_in_2D( I2A @ i.reshape(-1) )], savefig='delme.png' )
# util.nice_heatmap_subplots( [m for m in modal_basis], savefig='delme.png' )



# signal
ss = np.array( [ iii - i_dm for iii in ii_dm ])

alpha = (I2M.T @ ss.T).T # samples, modes 

# alpha = [] #Zernike mode coefficients 
# for s in ss: 
#     alpha.append( [ np.nansum( dm_pup_2D * modal_basis[idx] * s ) / np.nansum( dm_pup_2D ) for idx in enumerate( args.noll_inidices_2_look_at )] )

# alpha=np.array(alpha)


f, S_zz = welch(alpha.T , fs=args.cam_fps, nperseg=2**7)

plt.figure();plt.loglog( f, S_zz[0]); plt.savefig('delme.png')

dm2opd = 7000 #nm/cmd unit 
plot_psd( f_list = [f for _ in S_zz], psd_list = [dm2opd**2 * zz for zz in S_zz], savefig = 'delme.png', ylabel=r"PSD [$nm^2$/Hz]" +'\n' +r"rev. cum. [$nm^2$]", labels = [f'Noll index {ix}' for ix in args.noll_inidices_2_look_at], colors=['r','g','b','orange','yellow'])




# #########################
# # sanity check 

frames = c.mySHM.get_latest_data()
plt.figure()
plt.hist( np.diff( [f[0][0] for f in frames] )  ) #, bins = np.linspace(0,200,200))
plt.xlim([0,200])
plt.xlabel("difference in frame counter (image tag) \nbetween samples of mySHM.get_latest_data() at FPS=1.5kHz")
plt.ylabel("frequency")
plt.savefig('delme.png', bbox_inches='tight')


nframes_grabbed = len(c.mySHM.get_latest_data()) # numebr of frames grabbed per call 
fraction_of_changed_pixels = []
sleep_grid = np.logspace(0, 4.5, 10)
for sleep_factor in sleep_grid:
    a = c.mySHM.get_latest_data()
    time.sleep(sleep_factor/args.cam_fps)
    b = c.mySHM.get_latest_data()

    fraction_of_changed_pixels.append( (a!=b).sum()/np.sum( np.isfinite(a) ) )

plt.figure()
plt.loglog( sleep_grid/args.cam_fps, fraction_of_changed_pixels)
plt.axvline( nframes_grabbed / args.cam_fps , label='#frames/FPS',color='k', ls= ":")
plt.legend()
plt.ylabel("fraction of changed pixels")
plt.xlabel("sleeptime between get_latest_data()\ncurrently grabs most recent 200 frames")
plt.savefig('delme.png', bbox_inches='tight')




# frame_counter_difference = []
# sleep_grid = np.logspace(0,4, 10)
# for sleep_factor in sleep_grid:
#     a = c.mySHM.get_latest_data_slice()
#     time.sleep(sleep_factor/args.cam_fps)
#     b = c.mySHM.get_latest_data_slice()

#     frame_counter_difference.append( b[0][0] - a[0][0] )

# plt.figure()
# plt.plot( sleep_grid, frame_counter_difference,'x',label="measured")
# plt.plot( sleep_grid, sleep_grid, label="1:1",color='r')
# #plt.axvline( nframes_grabbed / args.cam_fps , label='#frames/FPS',color='k', ls= ":")
# plt.legend()
# plt.ylabel("frame counter")
# plt.xlabel("n * FPS")
# plt.savefig('delme.png', bbox_inches='tight')


# plt.figure()
# plt.title( "SHM.get_latest_data_slice()" )
# plt.loglog( sleep_grid/args.cam_fps, frame_counter_difference,'-o',label="measured")
# plt.loglog( sleep_grid/args.cam_fps, sleep_grid, label="1:1",color='r')
# plt.axvline(1/ args.cam_fps , label='1/FPS',color='k', ls= ":")
# plt.axvline(nframes_grabbed/ args.cam_fps , label='#frames in buffer/FPS',color='grey', ls= ":")
# plt.legend()
# plt.ylabel("frame counter")
# plt.xlabel(r"$\Delta$ t [s]")
# plt.savefig('delme.png', bbox_inches='tight')


