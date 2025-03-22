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
from scipy.signal import TransferFunction, bode
from types import SimpleNamespace
import asgard_alignment.controllino as co # for turning on / off source \
from asgard_alignment.DM_shm_ctrl import dmclass
import common.DM_basis_functions as dmbases
import asgard_alignment.controllino as co
import common.phasemask_centering_tool as pct
import common.phasescreens as ps 
import pyBaldr.utilities as util 
import pyzelda.ztools as ztools
import datetime
from xaosim.shmlib import shm
from asgard_alignment import FLI_Cameras as FLI

MDS_port = 5555
MDS_host = 'localhost'
context = zmq.Context()
context.socket(zmq.REQ)
socket = context.socket(zmq.REQ)
socket.setsockopt(zmq.RCVTIMEO, 5000)
server_address = f"tcp://{MDS_host}:{MDS_port}"
socket.connect(server_address)
state_dict = {"message_history": [], "socket": socket}


def myplot(thing):
    if len(np.array( thing ).shape ) == 1:
        plt.figure()
        plt.plot( thing )
        plt.savefig('delme.png')
        plt.close()

    elif len(np.array( thing ).shape ) == 2:
        plt.figure()
        plt.imshow( thing )
        plt.colorbar()
        plt.savefig('delme.png')
        plt.close()
    else:
        raise UserWarning('input shape invalid')





parser = argparse.ArgumentParser(description="Interaction and control matricies.")

default_toml = os.path.join("config_files", "baldr_config_#.toml") 

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
    default=2, # 1, 2, 3, 4],
    help="beam to look at"
)

parser.add_argument(
    "--phasemask",
    type=str,
    default="H3",
    help="Comma-separated beam IDs to apply. Default: 1,2,3,4"
)

args=parser.parse_args()

# c, dms, darks_dict, I0_dict, N0_dict,  baldr_pupils, I2A = setup(args.beam_id,
#                               args.global_camera_shm, 
#                               args.toml_file) 



data_path = '/home/asg/Videos/latancy_tests/'
if not os.path.exists( data_path ):
    os.makedirs( data_path )

# read in TOML as dictionary for config 
with open(args.toml_file.replace('#',f'{args.beam_id}'), "r") as f:
    config_dict = toml.load(f)
    # Baldr pupils from global frame 
    baldr_pupils = config_dict['baldr_pupils']
    I2A = config_dict[f'beam{args.beam_id}']['I2A']
    pupil_mask = config_dict.get(f"beam{args.beam_id}", {}).get("pupil_mask", {}).get("mask", None)


c = FLI.fli( args.global_camera_shm, roi = baldr_pupils[f"{args.beam_id}"])

c.send_fli_cmd( "maxfps")

fps = 1000
gain = 5

c.send_fli_cmd( f"set fps {fps}" )

time.sleep(1)

c.send_fli_cmd( f"set gain {gain}" )

time.sleep( 3)

c.mySHM.catch_up_with_sem(c.semid)

c.build_manual_dark( no_frames = 100)

_ = c.get_bad_pixels( no_frames = 100, std_threshold = 20, mean_threshold=6 )

#c.save_fits( fname = '/home/asg/Videos/PSD_gain-{fain}_fps-{fps}.fits', number_of_frames = 2000, apply_manual_reduction=False )

#myplot( c.reduction_dict['bad_pixel_mask'][-1] )

# t0 = time.time()

# c.get_image(apply_manual_reduction=True)

# t1 = time.time()
# print(t1-t0)

# for _ in range(2):
#     a = c.mySHM.get_latest_data_slice( c.semid )
#     b = c.mySHM.get_latest_data_slice( c.semid )
#     print( a[0][0], b[0][0] )


# for fps in [50, 100, 500, 1000, 1500]:
#     c.send_fli_cmd (f"set fps {fps}")

#     time.sleep(5)

#     c.mySHM.catch_up_with_sem(c.semid)
#     a = c.mySHM.get_latest_data(c.semid)
#     #print median frames skipped per FPS 
#     print(fps, f"median frames skipped {np.median(  np.diff([aa[0][0] for aa in a])  )}" ) 


dm = dmclass( beam_id=args.beam_id )

# zero DMs
dm.zero_all()
dm.activate_flat()

no_runs = 10000
poke_every = 20
poke_act = 65
mode140 = np.zeros(140)
poke_amp = 0.1
timestamps = []
frames = []
commands = []

dm.set_data( dm.cmd_2_map2D( mode140 ) )

c.mySHM.catch_up_with_sem(c.semid)
jj = 0 # counting poke events 
for ii in range(no_runs):
    #print(ii)
    # t0 
    t0 = time.time()

    if np.mod( ii , poke_every )==0:
        # set the DM mode
        mode140[poke_act] = (-1)**(jj) * poke_amp
        # poke DM 
        dm.set_data( dm.cmd_2_map2D( mode140 ) )
        jj+=1

    # t1 
    t1 = time.time()

    # get frames 
    single_frame = c.get_image(apply_manual_reduction=False)

    #c.get_some_data( number_of_frames= 10,  apply_manual_reduction=False ) #

    # t2
    t2 = time.time()

    # append telemetry 
    timestamps.append( [t0,t1,t2] )
    frames.append( single_frame.copy() ) 
    commands.append( mode140.copy() )
    

t0s = np.array( [tt[0] for tt in timestamps] )
t1s = np.array( [tt[1] for tt in timestamps] )
t2s = np.array( [tt[2] for tt in timestamps] )

poke_samples = np.arange( poke_every , no_runs, poke_every )

bins = np.logspace( -5, -2, 50)
plt.figure() 
plt.hist( t1s[poke_samples] - t0s[poke_samples] ,bins = bins, label = r"DM update", alpha = 0.5)
plt.hist( t2s[poke_samples] - t1s[poke_samples] ,bins = bins, label = r"frame read", alpha = 0.5)
plt.hist( t2s[poke_samples] - t0s[poke_samples] ,bins = bins, label = r"full iteration", alpha = 0.5)
plt.legend( fontsize=15)
plt.gca().tick_params(labelsize=15)
plt.xscale("log")
plt.xlabel( "$\Delta$t [s]", fontsize=15)
plt.ylabel( "frequency", fontsize=15)
#plt.savefig('delme.png', bbox_inches = 'tight')
plt.savefig(data_path + f'timestamp_hist_beam{args.beam_id}_gain-{gain}_fps-{fps}.png',bbox_inches = 'tight')

# zero DMs
dm.zero_all()
dm.activate_flat()

# analysis
frames = np.array( frames )
cmd_state = np.array( [ cc[poke_act] for cc in commands])
# interpolated onto registered DM actuators
frames_on_dm = (np.array(I2A) @ frames.reshape( frames.shape[0], -1).T).T
# final .T to make shape = (samples, actuator pixels)


# some basic checks 
myplot( cmd_state )
myplot( frames_on_dm[: int(3*poke_every),poke_act] )

plt.figure()
NNN=100
plt.plot( frames_on_dm[:NNN,poke_act] / np.max(frames_on_dm[:,poke_act]) )
plt.plot( cmd_state[:NNN] / np.max( cmd_state), label='state')
plt.legend()
plt.savefig('delme.png')



mean_on_idx = np.where( cmd_state == poke_amp) 
mean_off_idx = np.where( cmd_state == -poke_amp) 

median_on = np.median( frames_on_dm[mean_on_idx[0],poke_act] )
median_off = np.median( frames_on_dm[mean_off_idx[0],poke_act] )
print( median_on, median_off )

# Number of batches
num_batches = 5
# Create subplots
fig, axes = plt.subplots(num_batches, 1, figsize=(8, 5 * num_batches), sharex=True)

for batch in range(num_batches):
    # each time we poke up (2 * poke_every) we get the intensity value 
    # at the registered poke actuators pixel at some number of samples (batch)
    # after the poke
    sampling_indices = np.arange(poke_every + batch, no_runs, 2 * poke_every)
    
    # Extract values for the current batch
    sampled_values = frames_on_dm[ sampling_indices, poke_act]
    
    # Compute mean
    mean_value = np.mean(sampled_values)
    
    # Plot histogram for the batch
    axes[batch].hist(sampled_values, bins=10, alpha=0.7)
    axes[batch].axvline(mean_value, color='red', linestyle='dashed', linewidth=2, label=f'Mean: {mean_value:.2f}')
    axes[batch].axvline(median_on, color='blue', linestyle='dashed', linewidth=2, label=f'Median state 0')
    axes[batch].axvline(median_off, color='k', linestyle='dashed', linewidth=2, label=f'Median state 1')
    axes[batch].set_ylabel('Frequency',fontsize=15)
    axes[batch].set_title(f'{batch} frames after DM poke (frame rate = {fps}Hz)')
    axes[batch].legend(fontsize=12)
    axes[batch].grid(True)
    axes[batch].tick_params(labelsize=15)

# Shared x-label
axes[-1].set_xlabel(f'Pixel {poke_act} Intensity',fontsize=15)
plt.tight_layout()
#plt.savefig('delme.png')
fname = f"latency_test_histogram_gain-{gain}_fps-{fps}.png"
plt.savefig(data_path + fname, bbox_inches = 'tight')



cam_config = c.get_camera_config()

hdul = fits.HDUList()

## CAMERA FRAMES
hdu = fits.ImageHDU( np.array( frames ) )
hdu.header['EXTNAME'] = 'FRAMES'
hdu.header['phasemask'] = args.phasemask
hdu.header['poke_act'] = poke_act
hdu.header['beam'] = args.beam_id
hdu.header['poke_amp'] = poke_amp
for k,v in cam_config.items():
    hdu.header[k] = v 

for k, v in zip( ['r1','r2','c1','c2'], baldr_pupils[f"{args.beam_id}"]):
    hdu.header[k] = v

hdul.append(hdu)

## DM COMMANDS 
hdu = fits.ImageHDU( np.array( commands ) )
hdu.header['EXTNAME'] = 'DM_CMDS'
hdu.header['phasemask'] = args.phasemask
hdu.header['poke_act'] = poke_act
hdu.header['beam'] = args.beam_id
hdu.header['poke_amp'] = poke_amp

hdul.append(hdu)

## TIMES 
hdu = fits.ImageHDU( np.array( timestamps ) )
hdu.header['EXTNAME'] = 'TIMESTAMPS'
hdu.header['phasemask'] = args.phasemask
hdu.header['poke_act'] = poke_act
hdu.header['beam'] = args.beam_id
hdu.header['poke_amp'] = poke_amp

hdul.append(hdu)

hdu = fits.ImageHDU( c.reduction_dict['dark'][-1] )
hdu.header['EXTNAME'] = 'DARKS'

hdu = fits.ImageHDU( c.reduction_dict['bad_pixel_mask'][-1] )
hdu.header['EXTNAME'] = 'BAD_PIXEL_MASK'


hdu = fits.ImageHDU(I2A)
hdu.header['EXTNAME'] = 'interpMatrix'
hdul.append(hdu)


fits_file = data_path + f'speedtest_beam{args.beam_id}_mask-{args.phasemask}_pokeamp_{poke_amp}_fps-{fps}_gain-{gain}.fits' #_{args.phasemask}.fits'
#f'IM_full_{Nmodes}ZERNIKE_beam{beam_id}_mask-H5_pokeamp_{poke_amp}.fits' #_{args.phasemask}.fits'
hdul.writeto(fits_file, overwrite=True)
print(f'wrote telemetry to \n{fits_file}')


c.close( erase_file = False )
dm.close( erase_file = False )




###         DONE 

#######################################
# # OLD RANDOM TESTS 







# # ## reset DMs 
# # dm.zero_all()
# # # apply dm flat + calibrated offset (does this on channel 1)
# # dm.activate_calibrated_flat()









# #---------- New Darks 
# # run a new set of darks 
# get_new_dark = False
# if get_new_dark:
#     script_path = "/home/asg/Progs/repos/asgard-alignment/calibration/gen_dark_bias_badpix.py"
#     try:
#         # Run the script and ensure it completes
#         with subprocess.Popen(["python", script_path], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True) as process:
#             stdout, stderr = process.communicate()  # Wait for process to complete

#             if process.returncode == 0:
#                 print("Script executed successfully!")
#                 print(stdout)  # Print standard output (optional)
#             else:
#                 print(f"Script failed with error:\n{stderr}")

#     except Exception as e:
#         print(f"Error running script: {e}")





# # get darks and bad pixels 
# dark_fits_files = glob.glob("/home/asg/Progs/repos/asgard-alignment/calibration/cal_data/darks/*.fits") 
# most_recent_dark = max(dark_fits_files, key=os.path.getmtime) 

# dark_fits = fits.open( most_recent_dark )

# bad_pixels, bad_pixel_mask = FLI.get_bad_pixels( dark_fits["DARK_FRAMES"].data, std_threshold=10, mean_threshold=10)
# bad_pixel_mask[0][0] = False # the frame tag should not be masked! 

# # SHM object to get frames 
# c = shm( args.global_camera_shm )


# #========================================================
# ###
# # Look at how many pixels in the 100 frames change vs time between 
# # acquiring them
# nn=[]
# tgrid = np.logspace(-3,1,20)
# for t in tgrid:
#     print(t)
#     a=c.get_data()
#     time.sleep(t)
#     b=c.get_data()
#     nn.append( np.sum(a!=b) / np.sum( a > -np.inf) )
#     time.sleep(1)

# plt.figure()
# plt.semilogx( tgrid, 100 * np.array(nn) )
# plt.xlabel(r"$\Delta t$ [s] befween frames$_a$ and frames$_b$")
# plt.ylabel( r"$\Sigma$ frames$_a$ != frames$_b$  [%]")
# plt.axvline( 100/fps , ls=":", color='k', label = "100 * Tint")
# plt.axvline( 1/fps , ls=":", color='k', label = "Tint")
# plt.legend()
# plt.savefig('delme.png')

# #========================================================
# # Capture 102 sets of 100 frames 
# nn = []
# frames = []
# timestamps = []

# for _ in range(102):  
#     t0 = time.time()  # Timestamp before acquisition
#     a = c.get_data()  # Get 100-frame batch
#     timestamps.append(t0)  # Store 1 timestamp per 100 frames
#     nn.append(a[:, 0, 0])  # Extract 100 frame counters and store them
#     frames.append( a[:,20:40,20:40] ) # just look within a small patch 

# # Convert to NumPy arrays
# timestamps = np.array(timestamps)
# nn = np.array(nn).ravel()
# frames = np.array( frames )

# #frame_change_fraction = np.sum(frames != frames[0], axis=(1,2,3)) / np.sum( frames[0] > -np.inf)

# frame_change_fraction = np.sum( frames[1:] - frames[:-1] ,axis=(1,2,3)) > 0

# # Generate x-axis indices for 100,000 frames
# frame_indices = np.arange(len(nn))

# Na = 4000
# # Select timestamps every 100 frames (since each timestamp applies to 100 frames)
# timestamp_positions = np.arange(0, len(nn[:Na]), 100)
# selected_timestamps = timestamps[:len(timestamp_positions)] - timestamps[0]  # Only take the first N timestamps

# # Create figure
# fig, ax = plt.subplots(2,figsize=(8, 5), sharex=True)

# ax1=ax[0]
# # Plot frame counter vs frame number
# ax1.plot(frame_indices[:Na], nn[:Na], label=f"Tint={1e3*1/fps}ms", color="blue")
# ax1.set_ylabel("Counter", color="blue")
# ax1.tick_params(axis="y", labelcolor="blue")
# ax1.legend()

# # Top X-Axis: Timestamps every 100 frames
# ax2 = ax1.twiny()
# ax2.set_xlim(ax1.get_xlim())  # Match the x-axis range
# ax2.set_xticks(timestamp_positions[:len(selected_timestamps)])
# ax2.set_xticklabels([f"{t*1e3:.2f}" for t in selected_timestamps], rotation=45)
# ax2.set_xlabel("Time [ms]")

# for aa in timestamp_positions[:len(selected_timestamps)]:
#     ax1.axvline( aa , color='k', ls=':')
#     ax[1].axvline( aa , color='k', ls=':')
# ax[1].axvline( aa , color='k', ls=':',label='get_data() (grabs 100 frames)')

# ax[1].plot( timestamp_positions[:len(selected_timestamps)], frame_change_fraction[:len(selected_timestamps)])
# ax[1].set_xlabel("Frame Number")
# ax[1].set_ylabel("difference in frames?")
# ax[1].legend()
# # Save and show the plot
# plt.tight_layout()
# plt.savefig("delme.png")
# plt.close('all')


# #========================================================
# ### LOOK AT COUNTER ON ONE BATCH 
# plt.figure()
# plt.plot( nn[0] )
# plt.xlabel("frame number")
# plt.ylabel( "counter")
# #plt.axvline( 100/fps , ls=":", color='k')
# plt.savefig('delme.png')



# #========================================================
# #========================================================
# # Try get end-to-end latency

# # we just read full frame and crop 
# r1,r2,c1,c2 = baldr_pupils[f'{args.beam_id}']

# dm = dmclass( beam_id=args.beam_id )

# #dm = shm(f"/dev/shm/dm2.im.shm")
# # the handles to the semaphores to post
# #sem1 = shm("/dev/shm/dm2.im.shm", nosem=False)

# #dm.set_data( np.zeros([12,12]) )


# # zero DMs
# dm.zero_all()
# dm.activate_flat()

# # poke mode (poke an actuator)
# mode140 = np.zeros(140)
# ii = 0 
# amp = 0.08
# poke_act = 65
# mode140[poke_act] =  amp
# # put to SHM format (BMC DM is 140 actuators but SHM regrids to 12x12 = 144 vector)
# mode144 =  np.nan_to_num(util.get_DM_command_in_2D( mode140 ),0)

# # dark master image 
# dark = dark_fits["MASTER DARK"].data[r1:r2,c1:c2]

# time.sleep(10)

# # reference image 
# refs =  c.get_data()[:,r1:r2,c1:c2]  - dark
# ref = np.mean( refs , axis=0 ) 
# #myplot( ref )

# ##
# frames_raw = []
# timestamps = []
# ## get some initial 100 reference images (before poke)
# frames_raw.append( c.get_data()[:,r1:r2,c1:c2] - dark )
# # poke DM 
# dm.set_data( mode144 )
# for i in range(10):  
#     time.sleep(200/fps)
#     #frames = c.get_data()[:,r1:r2,c1:c2] - dark
#     frames_raw.append( c.get_data()[:,r1:r2,c1:c2] - dark )
#     timestamps.append(time.time())
# frames = np.array(frames_raw)
# frames = frames.reshape(-1, frames.shape[2],frames.shape[3])


# otherthing = []
# for i in range(len(frames)):
#     idm = I2A @ (ref - frames[i] ).reshape(-1)
#     otherthing.append( idm )

# plt.figure(figsize=(4,10))
# plt.imshow( otherthing )
# plt.axhline( 100 ,color='r',ls=":",label="poke command sent")
# plt.colorbar(label="interpolated intensity on DM actuators")
# plt.xlabel( "DM actuator #")
# plt.ylabel( "Frame number")
# for a in range(len(otherthing)//100-2):
#     plt.axhline( (a+2)*100 ,color='c',ls=":")
# #plt.legend()
# plt.savefig('delme.png')


# #myplot( otherthing )

# myplot( np.array(otherthing)[:,poke_act] )

# np.where( np.array(otherthing)[:,poke_act] < -40 )

# # Extract every 100th row starting from index 5 (5, 105, 205, ...)
# indices = np.arange(24, np.array(otherthing).shape[0], 100)
# selected_values = np.array(otherthing)[indices[1:], 65]  # Extract column 65

# plt.figure()
# plt.plot( timestamps, selected_values[:] )
# plt.xlabel('time [s]')
# plt.ylabel('act 65 intensity')
# plt.savefig('delme.png')
# #myplot( np.array(otherthing)[:,poke_act] )



# ### Just look at the last one 
# ff = 2
# idm = I2A @ (ref.reshape(-1) - frames_raw[ff].reshape(100,-1)).T
# myplot(idm)


# # fig,ax = plt.subplots(5,1)

# # otherthing = []
# # for i in [0,1,2,-2,-1]:
# #     idm = I2A @ (ref - frames[i] ).reshape(-1)
# #     thing = dm.cmd_2_map2D( idm )

# #     # Plot the image difference
# #     ax[i].imshow(thing, extent=[0, thing.shape[1], 0, thing.shape[0]], cmap="viridis")
    
# #     # Find the maximum absolute difference
# #     #ii, jj = np.unravel_index(np.argmax(abs(thing)), thing.shape)

# #     # Scatter plot the point of maximum difference
# #     #ax[i].scatter(jj, ii, marker='x', color='red', s=5)

# #     # Labels and formatting
# #     ax[i].set_title(f"Frame {i+1}")

# # plt.savefig('delme.png') 





# # ib = np.argmax( abs(thing))

# # thing = np.array([I2A @ t.reshape(-1) for t in (ref - hundred_frames_1)])

# # myplot( thing[:,poke_act] )

# # ii+=1

# # mode140[poke_act] = (-1)**(ii) * amp

# # # poke DM 
# # dm.set_data( dm.cmd_2_map2D( mode140 ) )

# # # get frames 
# # hundred_frames_2 = c.get_data()[:,r1:r2,c1:c2] 

# # thing = (np.mean(hundred_frames_2,axis=0)- np.mean(hundred_frames_1,axis=0))
# # myplot( thing )

# # no_runs = 100
# # poke_act = 65
# # mode140 = np.zeros(140)
# # amp = 0.1
# # timestamps = []
# # frames = []
# # commands = []

# # for ii in range(no_runs):
# #     print( ii , mode140[poke_act])
# #     # set the DM mode
# #     mode140[poke_act] = (-1)**(ii) * amp
    
# #     # t0 
# #     t0 = time.time()

# #     # poke DM 
# #     dm.set_data( dm.cmd_2_map2D( mode140 ) )

# #     # t1 
# #     t1 = time.time()

# #     # get frames 
# #     hundred_frames = c.get_data() 

# #     # t2
# #     t2 = time.time()

# #     # append telemetry 
# #     timestamps.append( [t0,t1,t2] )
# #     frames.append( hundred_frames.copy() ) 
# #     commands.append( mode140.copy() )
    

# # ## reset DMs 
# # dm.zero_all()
# # # apply dm flat + calibrated offset (does this on channel 1)
# # dm.activate_calibrated_flat()


# # r1,r2,c1,c2 = baldr_pupils[f'{args.beam_id}']
# # cropped_frames = np.array( frames )[:,:,r1:r2,c1:c2]

# # # frames_dm = (np.array(I2A)[np.newaxis] @ cropped_frames.reshape( cropped_frames.shape[0], cropped_frames.shape[1], -1).T )

# # mean_states_dm = np.array(I2A) @ np.mean( cropped_frames, axis=1 ).reshape(cropped_frames.shape[0],-1).T

# # plt.figure()
# # plt.plot(  mean_states_dm[poke_act, :] )
# # plt.savefig('delme.png')



# # plt.figure()
# # plt.plot(  np.array(commands)[:, poke_act] )
# # plt.savefig('delme.png')

# # plt.figure()
# # plt.imshow( dm.cmd_2_map2D( np.array(commands)[0]) );plt.colorbar()
# # plt.savefig('delme.png')

# # # dms[beam_id].zero_all()
# # # time.sleep(1)
# # # dms[beam_id].activate_flat()


# # # save the IM to fits for later analysis 

# # cam_config = cam_commander.get_camera_config()


# # hdul = fits.HDUList()

# # ## CAMERA FRAMES
# # hdu = fits.ImageHDU( np.array( frames ) )
# # hdu.header['EXTNAME'] = 'FRAMES'
# # hdu.header['phasemask'] = args.phasemask
# # hdu.header['poke_act'] = poke_act
# # hdu.header['beam'] = args.beam_id
# # hdu.header['poke_amp'] = amp
# # for k,v in cam_config.items():
# #     hdu.header[k] = v 

# # for k, v in zip( ['r1','r2','c1','c2'], baldr_pupils[args.beam_id]):
# #     hdu.header[k] = v

# # hdul.append(hdu)

# # ## DM COMMANDS 
# # hdu = fits.ImageHDU( np.array( commands ) )
# # hdu.header['EXTNAME'] = 'DM_CMDS'
# # hdu.header['phasemask'] = args.phasemask
# # hdu.header['poke_act'] = poke_act
# # hdu.header['beam'] = args.beam_id
# # hdu.header['poke_amp'] = amp

# # hdul.append(hdu)

# # ## TIMES 
# # hdu = fits.ImageHDU( np.array( timestamps ) )
# # hdu.header['EXTNAME'] = 'DM_CMDS'
# # hdu.header['phasemask'] = args.phasemask
# # hdu.header['poke_act'] = poke_act
# # hdu.header['beam'] = args.beam_id
# # hdu.header['poke_amp'] = amp

# # hdul.append(hdu)

# # hdu = fits.ImageHDU( dark_fits["DARK_FRAMES"].data )
# # hdu.header['EXTNAME'] = 'DARKS'

# # hdu = fits.ImageHDU(I2A)
# # hdu.header['EXTNAME'] = 'interpMatrix'
# # hdul.append(hdu)


# # fits_file = '/home/asg/Videos/' + f'speedtest_beam{beam_id}_mask-{args.phasemask}_pokeamp_{poke_amp}.fits' #_{args.phasemask}.fits'
# # #f'IM_full_{Nmodes}ZERNIKE_beam{beam_id}_mask-H5_pokeamp_{poke_amp}.fits' #_{args.phasemask}.fits'
# # hdul.writeto(fits_file, overwrite=True)
# # print(f'wrote telemetry to \n{fits_file}')


# # #SCP AUTOMATICALLY TO MY MACHINE 
# # remote_file = fits_file   # The file you want to transfer
# # remote_user = "bencb"  # Your username on the target machine
# # remote_host = "10.106.106.34"  
# # # (base) bencb@cos-076835 Downloads % ifconfig | grep "inet " | grep -v 127.0.0.1
# # # 	inet 192.168.20.5 netmask 0xffffff00 broadcast 192.168.20.255
# # # 	inet 10.200.32.250 --> 10.200.32.250 netmask 0xffffffff
# # # 	inet 10.106.106.34 --> 10.106.106.33 netmask 0xfffffffc

# # remote_path = "/Users/bencb/Downloads/"  # Destination path on your computer

# # # Construct the SCP command
# # scp_command = f"scp {remote_file} {remote_user}@{remote_host}:{remote_path}"

# # # Execute the SCP command
# # try:
# #     subprocess.run(scp_command, shell=True, check=True)
# #     print(f"File {remote_file} successfully transferred to {remote_user}@{remote_host}:{remote_path}")
# # except subprocess.CalledProcessError as e:
# #     print(f"Error transferring file: {e}")

