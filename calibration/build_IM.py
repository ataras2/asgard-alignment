#!/usr/bin/env python
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


# By default HO in this construction of the IM will always contain zonal actuation of each DM actuator.
# Using LO we can also define our Lower order modes on a Zernike basis where LO 
# is the Noll index up to which modes to consider. These LO modes are probed first
# in the IM and then the HO (zonal) modes are probed  


MDS_port = 5555
MDS_host = "192.168.100.2" # simmode : "127.0.0.1" #'localhost'
context = zmq.Context()
context.socket(zmq.REQ)
socket = context.socket(zmq.REQ)
socket.setsockopt(zmq.RCVTIMEO, 5000)
server_address = f"tcp://{MDS_host}:{MDS_port}"
socket.connect(server_address)
state_dict = {"message_history": [], "socket": socket}




def send_and_get_response(message):
    # st.write(f"Sending message to server: {message}")
    state_dict["message_history"].append(
        f":blue[Sending message to server: ] {message}\n"
    )
    state_dict["socket"].send_string(message)
    response = state_dict["socket"].recv_string()
    if "NACK" in response or "not connected" in response:
        colour = "red"
    else:
        colour = "green"
    # st.markdown(f":{colour}[Received response from server: ] {response}")
    state_dict["message_history"].append(
        f":{colour}[Received response from server: ] {response}\n"
    )

    return response.strip()


def plot2d( thing ):
    plt.figure()
    plt.imshow(thing)
    plt.colorbar()
    plt.savefig('/home/asg/Progs/repos/asgard-alignment/delme.png')
    plt.close()

# split_mode 1 
#aa = shm("/dev/shm/baldr1.im.shm")
#util.nice_heatmap_subplots( [ aa.get_data() ],savefig='delme.png')

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
    type=lambda s: [int(item) for item in s.split(",")],
    default=[1,2,3,4], # 1, 2, 3, 4],
    help="Comma-separated beam IDs to apply. Default: 1,2,3,4"
)

parser.add_argument(
    "--phasemask",
    type=str,
    default="H3",
    help="Comma-separated beam IDs to apply. Default: 1,2,3,4"
)

parser.add_argument(
    "--LO",
    type=int,
    default=2,
    help="Up to what zernike order do we consider Low Order (LO). 2 is for tip/tilt, 3 would be tip,tilt,focus etc). Default: %(default)s"
)


# parser.add_argument(
#     "--basis_name",
#     type=str,
#     default="zonal",
#     help="basis used to build interaction matrix (IM). zonal, zernike, zonal"
# )

# parser.add_argument(
#     "--Nmodes",
#     type=int,
#     default=10,
#     help="number of modes to probe"
# )

parser.add_argument(
    "--poke_amp",
    type=float,
    default=0.05,
    help="amplitude to poke DM modes for building interaction matrix"
)

parser.add_argument(
    "--signal_space",
    type=str,
    default='dm',
    help="what space do we consider the signal on. either dm (uses I2A) or pixel"
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
    default=1000,
    help="frames per second on camera. Default: %(default)s"
)


parser.add_argument(
    '--cam_gain',
    type=int,
    default=10,
    help="camera gain. Default: %(default)s"
)

parser.add_argument("--fig_path", 
                    type=str, 
                    default='/home/asg/Progs/repos/asgard-alignment/calibration/reports/test/', 
                    help="path/to/output/image/ for the saved figures")



args=parser.parse_args()


# c, dms, darks_dict, I0_dict, N0_dict,  baldr_pupils, I2A = setup(args.beam_id,
#                               args.global_camera_shm, 
#                               args.toml_file) 

NNN= 10 # how many groups of 100 to take for reference images 

I2A_dict = {}
pupil_mask = {}
secondary_mask = {}
exterior_mask = {}
for beam_id in args.beam_id:

    # read in TOML as dictionary for config 
    with open(args.toml_file.replace('#',f'{beam_id}'), "r") as f:
        config_dict = toml.load(f)
        # Baldr pupils from global frame 
        baldr_pupils = config_dict['baldr_pupils']
        I2A_dict[beam_id] = config_dict[f'beam{beam_id}']['I2A']
        
        pupil_mask[beam_id] = config_dict.get(f"beam{beam_id}", {}).get("pupil_mask", {}).get("mask", None)

        secondary_mask[beam_id] = np.array(config_dict.get(f"beam{beam_id}", {}).get("pupil_mask", {}).get("secondary", None) )

        exterior_mask[beam_id] = np.array(config_dict.get(f"beam{beam_id}", {}).get("pupil_mask", {}).get("exterior", None) )







c = FLI.fli(args.global_camera_shm, roi = [None,None,None,None])
#print("taking Dark")## post TTonsky
# cant do this if this same subtraction isnt done in RTC

# could remove dark manually so it isnt subtracted (since already done in the cred1 server! )
#c.reduction_dict['dark'] = [] 
# but we keep the badpixels

#_ = input("press enter when ready to turn source off and check it is actually dark")
## post TTonsky
#c.build_manual_dark(no_frames = 200 , build_bad_pixel_mask=True, kwargs={'std_threshold':20, 'mean_threshold':6} )
#^ holds dark and bad pixel mask in c.reduction_dict['dark'] and c.reduction_dict['bad_pixel_mask'] , is nice to check time to time especially in the subfrmes 
# THis matters for the better normalization 
#print("Took dark and built bad pixel mask held in FLI camera object")

# read the data to get directly the number of reads without reset (this is what the buffer is typically set to in non-destructive read mode)
nrs = c.mySHM.get_data().shape[0] 

# # Dont do calibration in paranal since server is doing this now 
# # change to append master dark , bias , bad pixel mask 
# c.send_fli_cmd(f"set gain {args.cam_gain}") 
# time.sleep(1)
# c.send_fli_cmd(f"set fps {args.cam_fps}")
# time.sleep(1)

# # make sure the camera cofig internal state is correct 
# assert float(c.config['fps']) == float(args.cam_fps)
# assert float(c.config['gain']) == float(args.cam_gain)

# # check for recent calibration files in the current setting 
# ##CHANGED TO MAKE VALID TIME DIFFERENCE 0.1 DAYS (a few hours)
# valid_cal_files = util.find_calibration_files(mode=c.config['mode'], gain=int(c.config['gain']), target_fps=float(c.config['fps']), base_dir="MASTER_DARK", time_diff_thresh=datetime.timedelta(0.1), fps_diff_thresh=10)

# # if no valid ones than we make some
# if not valid_cal_files: 
#     print( "no valid calibration files within the last few days. Taking new ones! ")
#     script_path = "/home/asg/Progs/repos/dcs/calibration_frames/gen_dark_bias_badpix.py"
#     params = ["--gains", f"{int(c.config['gain'])}", 
#               "--fps", f"{c.config['fps']}", 
#               "--mode", f"{c.config['mode']}", #"--mode", f"{c_dict[args.beam_id[0]].config['mode']}", 
#               "--method", "linear_fit" ]
#     try:
#         # Run the script and ensure it completes
#         with subprocess.Popen(["python", script_path]+params, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True) as process:
#             stdout, stderr = process.communicate()  # Wait for process to complete

#             if process.returncode == 0:
#                 print("Script executed successfully!")
#                 print(stdout)  # Print standard output (optional)
#             else:
#                 print(f"Script failed with error:\n{stderr}")

#     except Exception as e:
#         print(f"Error running script: {e}")


# # get darks, bias and some raw darks to make bad pixels (we dont use the premade ones cause we adjust here the parameters)
# bias_fits_files = util.find_calibration_files(mode=c.config['mode'], gain=int(c.config['gain']) , target_fps=float(c.config['fps']), base_dir="MASTER_BIAS", time_diff_thresh=datetime.timedelta(2), fps_diff_thresh=10) #glob.glob(f"/home/asg/Progs/repos/dcs/calibration_frames/products/{tstamp_rough}/MASTER_BIAS/*.fits") 
# dark_fits_files = util.find_calibration_files(mode=c.config['mode'], gain=int(c.config['gain']) , target_fps=float(c.config['fps']), base_dir="MASTER_DARK", time_diff_thresh=datetime.timedelta(2), fps_diff_thresh=10) #glob.glob(f"/home/asg/Progs/repos/dcs/calibration_frames/products/{tstamp_rough}/MASTER_DARK/*.fits") 
# raw_darks_files =  util.find_calibration_files(mode=c.config['mode'],gain=int(c.config['gain']) , target_fps=float(c.config['fps']), base_dir="RAW_DARKS", time_diff_thresh=datetime.timedelta(2), fps_diff_thresh=10) #glob.glob(f"/home/asg/Progs/repos/dcs/calibration_frames/products/{tstamp_rough}/RAW_DARKS/*.fits") 

# filename_reduction_dict = {} # to hold the files used for reduction 

# for lab, ff in zip(['bias','dark'], [bias_fits_files, dark_fits_files] ):
#     # Assumes we just took one!!! would be quicker to check subdirectories for one that matches the mode and gain with nearest fps. 
#     most_recent = max(ff, key=os.path.getmtime) 

#     filename_reduction_dict[lab+'_file'] = most_recent

#     with fits.open( most_recent ) as d:
#         c.reduction_dict[lab].append(  d[0].data.astype(int) )       # for beam_id in args.beam_id:
#         #     r1,r2,c1,c2 = baldr_pupils[f'{beam_id}']
#         #     c_dict[beam_id].reduction_dict[lab].append(  d[0].data.astype(int)[r1:r2, c1:c2] )

# # bad pixels 
# most_recent = max(raw_darks_files , key=os.path.getmtime) 
# filename_reduction_dict["raw_darks_file"] = most_recent

# with fits.open( most_recent ) as d:

#     bad_pixels, bad_pixel_mask = FLI.get_bad_pixels( d[0].data, std_threshold=4, mean_threshold=10)
#     bad_pixel_mask[0][0] = False # the frame tag should not be masked! 
#     #c_dict[beam_id].reduction_dict['bad_pixel_mask'].append(  (~bad_pixel_mask ).astype(int) )
#     c.reduction_dict['bad_pixel_mask'].append(  (~bad_pixel_mask ).astype(int) )




# Set up global camera frame SHM 
#print('Setting up camera. You should manually set up camera settings before hand')

# I2A_dict = {}
# pupil_mask = {}
# for beam_id in [1,2,3,4]:

#     # read in TOML as dictionary for config 
#     with open(args.toml_file.replace('#',f'{beam_id}'), "r") as f:
#         config_dict = toml.load(f)
#         # Baldr pupils from global frame 
#         baldr_pupils = config_dict['baldr_pupils']
#         I2A_dict[beam_id] = config_dict[f'beam{beam_id}']['I2A']
        
#         pupil_mask[beam_id] = config_dict.get(f"beam{beam_id}", {}).get("pupil_mask", {}).get("mask", None)


# # #---------- New Darks 
# # run a new set of darks 
# get_new_dark = False
# if get_new_dark:
#     script_path = "/home/asg/Progs/repos/asgard-alignment/calibration/gen_dark_bias_badpix.py"
#     params = [--gains, f"{args.cam_gain}", --fps, f"{args.cam_fps}"]
#     try:
#         # Run the script and ensure it completes
#         with subprocess.Popen(["python", script_path]+params, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True) as process:
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


####################################################################################
####################################################################################
#### DELETE THIS LATER (30/3/25 - only due to ron on chns 32)
# bad_ron = np.ones( [256,320] ).astype(bool) 
# bad_ron[:, ::32 ] = False
# bad_ron[:, 1::32 ] = False
# bad_ron[:, 2::32 ] = False
# bad_ron[:, 3::32 ] = False
####################################################################################
####################################################################################
# c_dict = {}
# for beam_id in args.beam_id:
#     r1,r2,c1,c2 = baldr_pupils[f'{beam_id}']
#     c_dict[beam_id] = FLI.fli(args.global_camera_shm, roi = [r1,r2,c1,c2])
#     #c_dict[beam_id].reduction_dict['bad_pixel_mask'].append( (~bad_pixel_mask).astype(int)[r1:r2, c1:c2] )
#     #c_dict[beam_id].reduction_dict['dark'].append(  dark_fits["MASTER DARK"].data.astype(int)[r1:r2, c1:c2] )

#     # change to append master dark , bias , bad pixel mask 
#     c_dict[beam_id].send_fli_cmd(f"set gain {args.cam_gain}") 
#     time.sleep(1)
#     c_dict[beam_id].send_fli_cmd(f"set fps {args.cam_fps}")
#     time.sleep(1)

#     c_dict[beam_id].build_manual_bias(number_of_frames=500) # sets to fastest fps (keeping current gain) to calculate bias 

#     c_dict[beam_id].build_manual_dark(number_of_frames=500, 
#                                       apply_manual_reduction=True,
#                                       build_bad_pixel_mask=True, 
#                                       kwargs={'std_threshold':10, 'mean_threshold':10} )

#     ####################################################################################
#     ####################################################################################
#     #### DELETE THIS LATER (30/3/25 - only due to ron on chns 32)
#     # c_dict[beam_id].reduction_dict['bad_pixel_mask'][-1] *= bad_ron[r1:r2,c1:c2]
#     ####################################################################################
#     ####################################################################################





#c_dict[beam_id].build_dark( no_frames = 100)
#    c_dict[beam_id].reduction_dict['bad_pixel_mask'].append( (~bad_pixel_mask).astype(int)[r1:r2, c1:c2] )
#    c_dict[beam_id].reduction_dict['dark'].append(  dark_fits["MASTER DARK"].data.astype(int)[r1:r2, c1:c2] )


# fps = c_dict[beam_id].send_fli_cmd("fps")

# set up DM SHMs 
print( 'setting up DMs')
dm_shm_dict = {}
for beam_id in args.beam_id:
    dm_shm_dict[beam_id] = dmclass( beam_id=beam_id, main_chn=3 ) # we poke on ch3 so we can close TT on chn 2 with rtc when building IM 
    
    ###     UP TO USER TO PUT THE FLAT ON!!!
    # zero all channels
    # dm_shm_dict[beam_id].zero_all()
    
    # if args.DM_flat.lower() == 'factory':
    #     # activate flat (does this on channel 1)
    #     dm_shm_dict[beam_id].activate_flat()
    # elif args.DM_flat.lower() == 'baldr':
    #     # apply dm flat + calibrated offset (does this on channel 1)
    #     dm_shm_dict[beam_id].activate_calibrated_flat()
        
    # else:
    #     print( "Unknow flat option. Valid options are 'factory' or 'baldr'. Using baldr flat as default")
    #     args.DM_flat == 'baldr'
    #     dm_shm_dict[beam_id].activate_calibrated_flat()

# Move to phase mask
for beam_id in args.beam_id:
    message = f"fpm_movetomask phasemask{beam_id} {args.phasemask}"
    res = send_and_get_response(message)
    print(f"moved to phasemask {args.phasemask} with response: {res}")

time.sleep(1)

# Get reference pupils (later this can just be a SHM address)
zwfs_pupils = {}
clear_pupils = {}
normalized_pupils = {}
rel_offset = 200.0 #um phasemask offset for clear pupil
print( 'Moving FPM out to get clear pupils')
for beam_id in args.beam_id:
    message = f"moverel BMX{beam_id} {rel_offset}"
    res = send_and_get_response(message)
    print(res) 
    time.sleep( 1 )
    message = f"moverel BMY{beam_id} {rel_offset}"
    res = send_and_get_response(message)
    print(res) 


# wait 2 buffers 
# time.sleep(10)
# wait for a new buffer to fill before we read the buffer and average it.
t0 = c.mySHM.get_counter()
cnt = 0
while cnt < 2 * nrs : # wait at least 2 buffers before we average buffer 
    t1 = c.mySHM.get_counter()
    cnt = t1 - t0 
    time.sleep( 1/float(c.config['fps']) )
del cnt, t1, t0 # delete when finished


############## HERE  

#Clear Pupil
print( 'gettin clear pupils')
N0s = c.get_data( apply_manual_reduction=True  ) #get_some_frames( number_of_frames = 1000,  apply_manual_reduction=True ) 
inner_pupil_filt = {} # strictly inside (not on boundary)

for beam_id in args.beam_id:
    
    r1,r2,c1,c2 = baldr_pupils[f"{beam_id}"]
    
    clear_pupils[beam_id] = N0s[:,r1:r2,c1:c2]

    #bad_pix_mask_tmp = np.array( c.reduction_dict["bad_pixel_mask"][-1][r1:r2,c1:c2] ).astype(bool)

    # move back 
    print( 'Moving FPM back in beam.')
    message = f"moverel BMX{beam_id} {-rel_offset}"
    res = send_and_get_response(message)
    print(res) 
    time.sleep(1)
    message = f"moverel BMY{beam_id} {-rel_offset}"
    res = send_and_get_response(message)
    print(res) 
    time.sleep(10)

    inner_pupil_filt[beam_id] = util.remove_boundary(pupil_mask[beam_id])

    # set as clear pupils where we set exterior and bad pixels to mean interior clear pup signal

    # filter exterior pixels (that risk 1/0 error)
    pixel_filter = secondary_mask[beam_id].astype(bool)  | (~inner_pupil_filt[beam_id].astype(bool) ) #| (~bad_pix_mask_tmp )
    
    normalized_pupils[beam_id] = np.mean( clear_pupils[beam_id] , axis=0) 
    normalized_pupils[beam_id][ pixel_filter  ] = np.mean( np.mean(clear_pupils[beam_id],0)[~pixel_filter]  ) # set exterior and boundary pupils to interior mean

    # normalize by sum in the subframe ## post TTonsky 
    #normalized_pupils[beam_id] /= np.sum( N0s[:,r1:r2,c1:c2] )
    #^^ bug, it should sum over agregated frame 
    normalized_pupils[beam_id] /= np.sum( np.mean( N0s[:,r1:r2,c1:c2] ,axis=0 ) )

    #N0 for normalization ( set exterior pixels )
    #pupil_norm = np.mean( N0s ,axis=0)
    #pupil_norm[~np.array( pupil_mask[beam_id] ) ] = np.mean( pupil_norm[pupil_mask[beam_id]])

# tbbb = util.remove_boundary(pupil_mask[beam_id])
# pupil_norm = np.mean( N0s ,axis=0)
# pupil_norm[~tbbb ] = np.mean( pupil_norm[tbbb] )
# plt.figure(); plt.imshow( normalized_pupils[beam_id] );plt.savefig('delme.png')

# check 
#util.nice_heatmap_subplots( [ np.mean( N0s,axis=0),  ~pixel_filter , pixel_filter, normalized_pupils[beam_id]], savefig='delme.png')

############## HERE 


# check the alignment is still ok 
#input('ensure mask is realigned')
# print("running fine mask alignment")
# for beam_id in args.beam_id:
#     cmd = ["python", "calibration/fine_phasemask_alignment.py","--beam_id",f"{beam_id}","--method","brute_scan"]

#     with subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True) as process:
#         stdout, stderr = process.communicate()

#     print("STDOUT:", stdout)
#     print("STDERR:", stderr)


# beam = int( input( "\ndo you want to check the phasemasks for a beam. Enter beam number (1,2,3,4) or 0 to continue\n") )
# while beam :
#     save_tmp = 'delme.png'
#     print(f'open {save_tmp } to see generated images after each iteration')
    
#     move_relative_and_get_image(cam=c, beam=beam, baldr_pupils = baldr_pupils, phasemask=state_dict["socket"],  savefigName = save_tmp, use_multideviceserver=True )
    
#     beam = int( input( "\ndo you want to check the phasemasks for a beam. Enter beam number (1,2,3,4) or 0 to continue\n") )


# ZWFS Pupil
input("phasemasks aligned? ensure alignment then press enter")

# wait for a new buffer to fill before we read the buffer and average it.
t0 = c.mySHM.get_counter()
cnt = 0
while cnt < 2 * nrs : # wait at least 1 buffers before we average buffer 
    t1 = c.mySHM.get_counter()
    cnt = t1 - t0 
    time.sleep( 1/float(c.config['fps']) )
del cnt, t1, t0 # delete when finished

print( 'Getting ZWFS pupils')
I0s = c.get_data( apply_manual_reduction=True ) #get_some_frames( number_of_frames = 1000,  apply_manual_reduction=True ) 

for beam_id in args.beam_id:

    #I0s = []
    #for _ in range(NNN):
    #    I0s.append(  c_dict[beam_id].get_data( apply_manual_reduction=True ) )
    #I0s = np.array(  I0s ).reshape(-1,  I0s[0].shape[1],  I0s[0].shape[2])

    r1,r2,c1,c2 = baldr_pupils[f"{beam_id}"]
    #cropped_img = interpolate_bad_pixels(img[r1:r2, c1:c2], bad_pixel_mask[r1:r2, c1:c2])
    #cropped_img = [nn[r1:r2,c1:c2] for nn in I0s] #/np.mean(img[r1:r2, c1:c2][pupil_mask[bb]])
    
    #zwfs_pupils[beam_id] = I0s[:,r1:r2,c1:c2] / np.sum( I0s[:,r1:r2,c1:c2] ) ## post TTonsky#cropped_img
    #^^ bug! it should sum over a signal aggregated frame! 
    zwfs_pupils[beam_id] = I0s[:,r1:r2,c1:c2] / np.sum( np.mean( I0s[:,r1:r2,c1:c2],axis=0 ) )

# #dark = np.mean( darks_dict[beam_id],axis=0)

# I0 = {}
# N0 = {}
# I0_dm = {}
# N0_dm = {}
# dark_dm = {}
# dm_act_filt = {}
# dm_mask = {}
# for beam_id in args.beam_id:
#     N0[beam_id] = np.mean(clear_pupils[beam_id],axis=0)
#     I0[beam_id] = np.mean(zwfs_pupils[beam_id],axis=0)
#     #interpMatrix = I2A_dict[beam_id]
#     I0_dm[beam_id] = I2A_dict[beam_id] @ I0[beam_id].reshape(-1)
#     N0_dm[beam_id] = I2A_dict[beam_id] @ N0[beam_id].reshape(-1)

#     dark_dm[beam_id] = I2A_dict[beam_id] @ c_dict[beam_id].reduction_dict['dark'][-1].reshape(-1)

# dm_act_filt = {}
# dm_mask = {}
# for beam_id in args.beam_id:
#     ### IMPORTANT THIS IS WHERE WE FILTER ACTUATOR SPACE IN IM 
#     dm_mask[beam_id] = I2A_dict[beam_id] @  np.array(pupil_mask[beam_id] ).reshape(-1)
#     dm_act_filt[beam_id] = dm_mask[beam_id] > 0.95 # ignore actuators on the edge! 


# # checks 
# cbar_label_list = ['[adu]','[adu]', '[adu]']
# title_list = ['DARK','I0', 'N0']
# xlabel_list = ['','','']
# ylabel_list = ['','','']

# util.nice_heatmap_subplots( im_list = [ c_dict[beam_id].reduction_dict['dark'][-1], I0[beam_id] , N0  ], 
#                             cbar_label_list = cbar_label_list,
#                             title_list=title_list,
#                             xlabel_list=xlabel_list,
#                             ylabel_list=ylabel_list,
#                             savefig='delme.png' )


#basis_name = args.basis_name #"zonal" #"ZERNIKE"

LO_basis = dmbases.zer_bank(2, args.LO+1 )
zonal_basis = np.array([dm_shm_dict[beam_id].cmd_2_map2D(ii) for ii in np.eye(140)]) 

modal_basis = np.array( LO_basis.tolist() +  zonal_basis.tolist() ) 
# should be 144 x 140 (we deal with errors in 140 actuator space (columns), but SHM takes 144 vector as input (rows)) 
# this is why we do transpose 
M2C = modal_basis.copy().reshape(modal_basis.shape[0],-1).T # mode 2 command matrix 
# we do this in process script and project LO out of M2C_HO
# M2C_LO = LO_basis.reshape( LO_basis.shape[0],-1).T 
# M2C_HO = zonal_basis.reshape( zonal_basis.shape[0],-1).T 
# phase_cov = np.eye( 140 ) #np.array(IM).shape[0] )
# noise_cov = np.eye( Nmodes ) #np.array(IM).shape[1] )




# im_list = [m.reshape(12,12) for m in M2C.T[:4]]

# util.nice_heatmap_subplots( im_list = im_list, savefig='delme.png')


# we only need HO and require len 144x 140 (SHM input x number of actuatorss) which projects out the TT 
# _ , M2C_HO = util.project_matrix( np.nan_to_num( M2C[:,args.LO:], 0),  np.nan_to_num(M2C.T[:args.LO],0).reshape(-1,144) )

# m = np.zeros( M2C.shape[1])
# m[0] = 0.05
# m[65] = 0.2 
# M2C_HO @ m

# im_list = [(M2C @ m).reshape(12,12) , (M2C_LO @ m).reshape(12,12),(M2C_HO @ m).reshape(12,12)]
# title_list = ["M2C . m", "M2C_LO . m", "M2C_HO . m"]
# cbar_title_list = ["DM UNITS", "DM UNITS", "DM UNITS"]
# util.nice_heatmap_subplots( im_list = im_list ,title_list=title_list, cbar_label_list=  cbar_title_list, savefig='delme.png')



### Just doing 1 for now
#############
#beam_id = args.beam_id[0]
############


#r1,r2,c1,c2 = baldr_pupils[f'{beam_id}']



# # check they interpolate ok onto DM 
# cbar_label_list = ['[adu]','[adu]', '[adu]','[unitless]']
# title_list = ['DARK DM','I0 DM', 'N0 DM','(I-I0)/N0']
# xlabel_list = ['','','','']
# ylabel_list = ['','','','']
# # intensity from first measurement 
# idm = I2A_dict[beam_id] @  zwfs_pupils[beam_id][0].reshape(-1)

# im_list = [util.get_DM_command_in_2D(a) for a in [ dark_dm[beam_id], I0_dm[beam_id] , N0_dm[beam_id] ,(idm -I0_dm[beam_id]) /  N0_dm[beam_id] ]]
# util.nice_heatmap_subplots( im_list = im_list, 
#                             cbar_label_list = cbar_label_list,
#                             title_list=title_list,
#                             xlabel_list=xlabel_list,
#                             ylabel_list=ylabel_list,
#                             savefig='delme.png' )

# #dms[beam_id].zero_all()
# time.sleep(1)
# #dms[beam_id].activate_flat()

if args.signal_space.lower() not in ["dm", "pixel"] :
    raise UserWarning("signal space must either be 'dm' or 'pixel'")

#cam_config = c.config

IM = {beam_id:[] for beam_id in args.beam_id}
Iplus_all = {beam_id:[] for beam_id in args.beam_id}
Iminus_all = {beam_id:[] for beam_id in args.beam_id}




    # return img

#imgs_to_mean = 20 # for each poke we average this number of frames
# for now we use standard get_data mehtod which is 200 frames (april 2025)
for i,m in enumerate(modal_basis):
    print(f'executing cmd {i}/{len(modal_basis)}')
    #if i == args.LO:
    #    input("close Baldr TT and ensure stable. Then press enter.")
    I_plus_list = {beam_id:[] for beam_id in args.beam_id}
    I_minus_list = {beam_id:[] for beam_id in args.beam_id}
    for sign in [(-1)**n for n in range(4)]: #range(10)]: #[-1,1]:
        
        for beam_id in args.beam_id:
            dm_shm_dict[beam_id].set_data(  sign * args.poke_amp/2 * m ) 
        
        #print( "sleep", float(c.config["fps"]) )
        #time.sleep( nbreadworeset / 1 ) #float(c.config["fps"])) # 200 because get data takes 200 frames
        
        # wait for a new buffer to fill before we read the buffer and average it.
        t0 = c.mySHM.get_counter()
        cnt = 0
        while cnt < 2 * nrs : # wait at least 1 buffers before we average buffer 
            t1 = c.mySHM.get_counter()
            cnt = t1 - t0 
            time.sleep( 1/float(c.config['fps']) )
        del cnt, t1, t0 # delete when finished
        
        imgtmp_global = c.get_data(apply_manual_reduction = True )
        # quick version below just for testing . Use full ^ grab above for proper cal.
        #imgtmp_global = np.array([c.get_image(apply_manual_reduction = True ) ,c.get_image(apply_manual_reduction = True )] )#get_data(apply_manual_reduction = True ) # get_some_frames( number_of_frames = imgs_to_mean, apply_manual_reduction = True )

        for beam_id in args.beam_id:
            r1,r2,c1,c2 = baldr_pupils[f'{beam_id}']
            

            img_tmp = np.mean( imgtmp_global[:,r1:r2,c1:c2], axis = 0)

            

            img_tmp /= np.sum( img_tmp ) ## post TTonsky

            if sign > 0:

                I_plus_list[beam_id].append( list( img_tmp ) )

            if sign < 0:

                I_minus_list[beam_id].append( list( img_tmp ) )


    for beam_id in args.beam_id:
        I_plus = np.mean( I_plus_list[beam_id], axis = 0).reshape(-1) / normalized_pupils[beam_id].reshape(-1)
        I_minus = np.mean( I_minus_list[beam_id], axis = 0).reshape(-1) /  normalized_pupils[beam_id].reshape(-1)

        #errsig = dm_mask[beam_id] * ( I2A_dict[beam_id] @ ((I_plus - I_minus))  )  / args.poke_amp  # dont use pokeamp norm so I2M maps to naitive DM units (checked in /Users/bencb/Documents/ASGARD/Nice_March_tests/IM_zernike100/SVD_IM_analysis.py)
        #errsig = #( I2A_dict[beam_id] @ ((I_plus - I_minus))  )  / args.poke_amp  # dont use pokeamp norm so I2M maps to naitive DM units (checked in /Users/bencb/Documents/ASGARD/Nice_March_tests/IM_zernike100/SVD_IM_analysis.py)
        
        # Try minimize dependancies, if I2A not calibrated or DM mask then the above fails.. keep simple. We can deal with this in post processing
        
        #############
        #############
        
        # removing seoconary pixels 
        #(~secondary_mask[beam_id].astype(bool)).reshape(-1) 

        if args.signal_space.lower() == 'dm':
            ## post TTonsky
            errsig = I2A_dict[beam_id] @   (I_plus - I_minus)  / args.poke_amp  
            #errsig = I2A_dict[beam_id] @ ( float( c.config["gain"] ) / float( c.config["fps"] )  * (I_plus - I_minus)  / args.poke_amp ) # 1 / DMcmd * (s * gain)  projected to DM space
        elif args.signal_space.lower() == 'pixel':
            ## post TTonsky
            errsig =  (I_plus - I_minus)  / args.poke_amp  # 1 / DMcmd * (s * gain)  projected to Pixel space
            #errsig = ( float( c.config["gain"] ) / float( c.config["fps"] )  * (I_plus - I_minus)  / args.poke_amp ) # 1 / DMcmd * (s * gain)  projected to Pixel space
        
        #############
        #############

        # reenter pokeamp norm <- this is used for detailed analysis sometimes
        #Iplus_all[beam_id].append( I_plus_list )
        #Iminus_all[beam_id].append( I_minus_list )

        IM[beam_id].append( list(  errsig.reshape(-1) ) ) 

# # intensity to mode matrix 
# if args.inverse_method == 'pinv':
#     I2M = np.linalg.pinv( IM )

# elif args.inverse_method == 'MAP': # minimum variance of maximum posterior estimator 
#     I2M = (phase_cov @ IM @ np.linalg.inv(IM.T @ phase_cov @ IM + noise_cov) ).T #have to transpose to keep convention.. although should be other way round

# elif args.inverse_method == 'zonal':
    
#     dm_mask = I2A_dict[beam_id] @ np.array( pupil_mask[beam_id] ).reshape(-1)
#     # control matrix (zonal) - 
#     I2M = np.diag(  np.array( [dm_mask[i]/IM[i][i] if np.isfinite(1/IM[i][i]) else 0 for i in range(len(IM))]) )
    
# elif args.inverse_method == 'SVD truncation':

# else:
#     raise UserWarning('no inverse method provided')



# optical gain 
# slope per act 
# if args.basis_name == 'zonal':
#     b = []
#     for i in range(140):
#         # we have to square fps/gain to cancel gain/fps in IM and then make units 1/(dmunit * s * gain)
#         b.append( ( float( c.config["fps"] / float( c.config["gain"] )  ) )**2 * ( I2A_dict[beam_id] @ IM[i] ) [i] ) # get the slope at each actuator 

#     cbar_list = ["ADU/s/gain/DM command"]
#     util.nice_heatmap_subplots( [util.get_DM_command_in_2D(b)] , cbar_label_list=cbar_list ) ; plt.show() 



for beam_id in args.beam_id:
    U,S,Vt = np.linalg.svd( IM[beam_id], full_matrices=True)
    #singular values
    plt.figure()
    plt.semilogy(S) #/np.max(S))
    #plt.axvline( np.pi * (10/2)**2, color='k', ls=':', label='number of actuators in pupil')
    plt.legend()
    plt.xlabel('mode index')
    plt.ylabel('singular values')

    if not os.path.exists(args.fig_path):
        print(f"making directory {args.fig_path} for plotting some results.")
        os.makedirs( args.fig_path )
    
    plt.savefig(f'{args.fig_path}' + f'IM_singularvalues_beam{beam_id}.png', bbox_inches='tight', dpi=200)
    plt.close()


# n_row = round( np.sqrt( M2C.shape[0]) ) - 1

# fig,ax = plt.subplots(n_row, n_row, figsize=(15,15))
# plt.subplots_adjust(hspace=0.1,wspace=0.1)
# for i,axx in enumerate(ax.reshape(-1)):
#     axx.imshow( M2C.T @ U.T[i]  )
#     #axx.set_title(f'mode {i}, S={round(S[i]/np.max(S),3)}')
#     axx.text( 1,2,f'{i}',color='w',fontsize=6)
#     axx.text( 1,3,f'S={round(S[i]/np.max(S),3)}',color='w',fontsize=6)
#     axx.axis('off')
#     #plt.legend(ax=axplt.tight_layout()
# if save_path is not None:
#     plt.savefig(save_path +  f'cam_eignmodes_{pokeamp}.png',bbox_inches='tight',dpi=200)
# plt.show()



# fig,ax = plt.subplots(n_row, n_row, figsize=(15,15))
# plt.subplots_adjust(hspace=0.1,wspace=0.1)
# for i,axx in enumerate(ax.reshape(-1)):
#     axx.imshow( get_DM_command_in_2D( Vt[i] )  )
#     #axx.set_title(f'mode {i}, S={round(S[i]/np.max(S),3)}')
#     axx.text( 1,2,f'{i}',color='w',fontsize=6)
#     axx.text( 1,3,f'S={round(S[i]/np.max(S),3)}',color='w',fontsize=6)
#     axx.axis('off')
#     #plt.legend(ax=a

# if save_path is not None:
#     plt.savefig(save_path +  f'DM_eignmodes_{pokeamp}.png',bbox_inches='tight',dpi=200)
# plt.show()


## reset DMs 

#dm_shm_dict[beam_id].zero_all()
# apply dm flat + calibrated offset (does this on channel 1)

# dm_shm_dict[beam_id].activate_calibrated_flat()




#dict2write = {f"beam{beam_id}":{f"{args.phasemask}":{"ctrl_model":  {"camera_config" : {k:str(v) for k,v in c.config.items()}}}}}

######## WRITE TO TOML 
#  # we store all reference images as flattened array , boolean masks as ints
for beam_id in args.beam_id:
    r1,r2,c1,c2 = baldr_pupils[f'{beam_id}'] 

    dict2write = {f"beam{beam_id}":{f"{args.phasemask}":{"ctrl_model": {
                                                    "build_method":"double-sided-poke",
                                                    "DM_flat":args.DM_flat.lower(),
                                                    "signal_space":args.signal_space.lower(),
                                                    "crop_pixels": np.array( baldr_pupils[f"{beam_id}"] ).tolist(), # global corners (r1,r2,c1,c2) of sub pupil cropping region  (local frame)
                                                    "pupil_pixels" : np.where(  np.array( pupil_mask[beam_id] ).reshape(-1) )[0].tolist(),  # pupil pixels in local frame 
                                                    "interior_pixels" : np.where( np.array( inner_pupil_filt[beam_id].reshape(-1) )   )[0].tolist(), # strictly interior pupil pixels in local frame
                                                    "secondary_pixels" : np.where( np.array( secondary_mask[beam_id].reshape(-1) )  )[0].tolist(),   # pixels in secondary obstruction in local frame
                                                    "exterior_pixels" : np.where(  np.array( exterior_mask[beam_id].reshape(-1) )   )[0].tolist(),  # exterior pixels that maximise diffracted light from mask in local frame 
                                                    #"bad_pixels" : np.where( np.array( c.reduction_dict['bad_pixel_mask'][-1])[r1:r2,c1:c2].reshape(-1)   )[0].tolist(),
                                                    "IM": np.array( IM[beam_id] ).tolist(),
                                                    "poke_amp":args.poke_amp,
                                                    "LO":args.LO, ## THIS DEFINES WHAT INDEX IN IM WE HAVE LO VS HO MODES , DONE HERE NOW RATHER THAN build_baldr_control_matrix.py.
                                                    "M2C": np.nan_to_num( np.array(M2C), 0 ).tolist(),   # 
                                                    "I0": np.mean( zwfs_pupils[beam_id],axis=0).reshape(-1).tolist(), ## ## post TTonsky  #(float( c.config["fps"] ) / float( c.config["gain"] ) * np.mean( zwfs_pupils[beam_id],axis=0).reshape(-1) ).tolist(),  # ADU / s / gain (flattened)
                                                    "N0": np.mean( clear_pupils[beam_id],axis=0).reshape(-1).tolist(), ## ## post TTonsky #(float( c.config["fps"] ) / float( c.config["gain"] ) * np.mean( clear_pupils[beam_id],axis=0).reshape(-1) ).tolist(), # ADU / s / gain (flattened)
                                                    "norm_pupil": np.array( normalized_pupils[beam_id] ).reshape(-1).tolist(), ## post TTonsky #( float( c.config["fps"] ) / float( c.config["gain"] ) * np.array( normalized_pupils[beam_id] ).reshape(-1) ).tolist(),
                                                    "camera_config" : {k:str(v) for k,v in c.config.items()},
                                                    #"bias": np.array(c.reduction_dict["bias"][-1])[r1:r2,c1:c2].reshape(-1).tolist(),
                                                    #"dark": np.array(c.reduction_dict["dark"][-1])[r1:r2,c1:c2].reshape(-1).tolist(),
                                                    #"bad_pixel_mask": np.array(c.reduction_dict['bad_pixel_mask'][-1])[r1:r2,c1:c2].astype(int).reshape(-1).tolist(),
                                                    "pupil": np.array(pupil_mask[beam_id]).astype(int).reshape(-1).tolist(),
                                                    "secondary": np.array(secondary_mask[beam_id]).astype(int).reshape(-1).tolist(),
                                                    "exterior" : np.array(exterior_mask[beam_id]).astype(int).reshape(-1).tolist(),
                                                    "inner_pupil_filt": np.array(inner_pupil_filt[beam_id]).astype(int).reshape(-1).tolist(),
                                                    # !!!! Set these calibration things to zero since they should be dealt with by cred 1 server! 
                                                    "bias" : np.zeros([32,32]).reshape(-1).astype(int).tolist(),
                                                    "dark" : np.zeros([32,32]).reshape(-1).astype(int).tolist(),
                                                    "bad_pixel_mask" : np.ones([32,32]).reshape(-1).astype(int).tolist(),
                                                    "bad_pixels" : [], 
                                                }
                                                }
                                            }
                                        }

    # Check if file exists; if so, load and update.
    if os.path.exists(args.toml_file.replace('#',f'{beam_id}')):
        try:
            current_data = toml.load(args.toml_file.replace('#',f'{beam_id}'))
        except Exception as e:
            print(f"Error loading TOML file: {e}")
            current_data = {}
    else:
        current_data = {}


    current_data = util.recursive_update(current_data, dict2write)

    with open(args.toml_file.replace('#',f'{beam_id}'), "w") as f:
        toml.dump(current_data, f)

    print( f"updated configuration file {args.toml_file.replace('#',f'{beam_id}')}")



#closing stuff 
c.close(erase_file=False)

for beam_id in args.beam_id:
    #dm_shm_dict[beam_id].zero_all()
    time.sleep(0.1)
    #dm_shm_dict[beam_id].activate_calibrated_flat()
    time.sleep(0.1)
    dm_shm_dict[beam_id].close(erase_file=False)




# dms[beam_id].zero_all()
# time.sleep(1)
# dms[beam_id].activate_flat()


# ######################################
# # save the IM to fits for later analysis 



# hdul = fits.HDUList()

# hdu = fits.ImageHDU(IM)
# hdu.header['EXTNAME'] = 'IM'
# hdu.header['units'] = "sec.gain/DMunit"
# hdu.header['phasemask'] = args.phasemask
# hdu.header['beam'] = beam_id
# hdu.header['poke_amp'] = args.poke_amp
# for k,v in c.config.items():
#     hdu.header[k] = v 

# for ii , ll in zip([r1,r2,c1,c2],["r1","r2","c1","c2"]) :
#     hdu.header[ll] = ii
# hdu.header["frame_shape"] = f"{r2-r1}x{c2-c1}"

# hdul.append(hdu)


# # hdu = fits.ImageHDU( dark_fits["DARK_FRAMES"].data )
# # hdu.header['EXTNAME'] = 'DARKS'

# hdu = fits.ImageHDU(Iplus_all)
# hdu.header['EXTNAME'] = 'I+'
# hdul.append(hdu)


# hdu = fits.ImageHDU(clear_pupils[beam_id])
# hdu.header['EXTNAME'] = 'N0'
# hdul.append(hdu)

# hdu = fits.ImageHDU(zwfs_pupils[beam_id])
# hdu.header['EXTNAME'] = 'I0'
# hdul.append(hdu)

# hdu = fits.ImageHDU(normalized_pupils[beam_id])
# hdu.header['EXTNAME'] = 'normalized_pupil'
# hdul.append(hdu)


# hdu = fits.ImageHDU(Iplus_all)
# hdu.header['EXTNAME'] = 'I+'
# hdul.append(hdu)

# hdu = fits.ImageHDU(Iminus_all)
# hdu.header['EXTNAME'] = 'I-'
# hdul.append(hdu)

# hdu = fits.ImageHDU( np.array(pupil_mask[beam_id]).astype(int)) 
# hdu.header['EXTNAME'] = 'PUPIL_MASK'
# hdul.append(hdu)

# # hdu = fits.ImageHDU( dm_mask[beam_id] )
# # hdu.header['EXTNAME'] = 'PUPIL_MASK_DM'
# # hdul.append(hdu)

# hdu = fits.ImageHDU(modal_basis)
# hdu.header['EXTNAME'] = 'M2C'
# hdul.append(hdu)

# # hdu = fits.ImageHDU(I2M)
# # hdu.header['EXTNAME'] = 'I2M'
# # hdul.append(hdu)

# hdu = fits.ImageHDU(I2A_dict[beam_id])
# hdu.header['EXTNAME'] = 'interpMatrix'
# hdul.append(hdu)

# # hdu = fits.ImageHDU(zwfs_pupils[beam_id])
# # hdu.header['EXTNAME'] = 'I0'
# # hdul.append(hdu)

# # hdu = fits.ImageHDU(clear_pupils[beam_id])
# # hdu.header['EXTNAME'] = 'N0'
# # hdul.append(hdu)

# fits_file = '/home/asg/Videos/' + f'IM_full_{Nmodes}{basis_name}_beam{beam_id}_mask-{args.phasemask}_pokeamp_{args.poke_amp}_fps-{c.config["fps"]}_gain-{c.config["gain"]}.fits' #_{args.phasemask}.fits'
# #f'IM_full_{Nmodes}ZERNIKE_beam{beam_id}_mask-H5_pokeamp_{poke_amp}.fits' #_{args.phasemask}.fits'
# hdul.writeto(fits_file, overwrite=True)
# print(f'wrote telemetry to \n{fits_file}')








#SCP AUTOMATICALLY TO MY MACHINE 
# remote_file = fits_file   # The file you want to transfer
# remote_user = "bencb"  # Your username on the target machine
# remote_host = "10.106.106.34"  
# # (base) bencb@cos-076835 Downloads % ifconfig | grep "inet " | grep -v 127.0.0.1
# # 	inet 192.168.20.5 netmask 0xffffff00 broadcast 192.168.20.255
# # 	inet 10.200.32.250 --> 10.200.32.250 netmask 0xffffffff
# # 	inet 10.106.106.34 --> 10.106.106.33 netmask 0xfffffffc

# remote_path = "/Users/bencb/Downloads/"  # Destination path on your computer

# # Construct the SCP command
# scp_command = f"scp {remote_file} {remote_user}@{remote_host}:{remote_path}"

# # Execute the SCP command
# try:
#     subprocess.run(scp_command, shell=True, check=True)
#     print(f"File {remote_file} successfully transferred to {remote_user}@{remote_host}:{remote_path}")
# except subprocess.CalledProcessError as e:
#     print(f"Error transferring file: {e}")



######## TESTING 

# dm_mask = I2A_dict[beam_id] @ np.array( pupil_mask[beam_id] ).reshape(-1)
# # control matrix (zonal) - 
# I2M = np.diag(  np.array( [dm_mask[i]/IM[i][i] if np.isfinite(1/IM[i][i]) else 0 for i in range(len(IM))]) )

# pp = 0.04
# m = 65
# abb = pp * modal_basis[m] 
# dm_shm_dict[beam_id].set_data(  abb ) 


# dm_mask = I2A_dict[beam_id] @  np.array( pupil_mask[beam_id] ).reshape(-1)
# dm_act_filt = dm_mask > 0.95 # ignore actuators on the edge! 

# time.sleep(1)

# fps = float( c_dict[beam_id].config["fps"] )
# gain = float( c_dict[beam_id].config["gain"] )

# i = np.mean( c_dict[beam_id].get_data( apply_manual_reduction=True ), axis=0) 
# # plot2d( i )
# # (~secondary_mask[beam_id].astype(bool))
# s = (   (i - np.mean( zwfs_pupils[beam_id] , axis=0) ) / normalized_pupils[beam_id] ).reshape(-1)
# #plot2d( ((i - np.mean( zwfs_pupils[beam_id]) ) / normalized_pupils[beam_id] ) )

# sig = gain / fps  * ( I2A_dict[beam_id] @ s  )
# #plot2d( util.get_DM_command_in_2D( sig ) )

# err = I2M.T @ sig

# # err[9] = 0 # filter out spherical 
# reco =  (M2C.T @ err).T 

# res = abb - reco 

# rmse = np.sqrt( np.mean( (res)**2 ))

# #dmfilt_12x12 = util.get_DM_command_in_2D( dm_act_filt[beam_id] )
# #im_list = [i, util.get_DM_command_in_2D(sig), dm_act_filt[beam_id] * reco, dm_act_filt[beam_id] * res ]
# im_list = [abb ,  i.T , util.get_DM_command_in_2D(sig), util.get_DM_command_in_2D( dm_act_filt ) * reco,  util.get_DM_command_in_2D( dm_act_filt ) * res ]

# title_list = ["disturbance", "intensity", "signal", "reco.", "residual"]
# vlims = [[np.nanmin(thing), np.nanmax(thing)] for thing in im_list[:-1]] 
# vlims.append( vlims[-1] )
# cbar_label_list = ["DM UNITS", "ADU", "ADU", "DM UNITS", "DM UNITS"]
# util.nice_heatmap_subplots( im_list, title_list = title_list, cbar_label_list=cbar_label_list, vlims= vlims, savefig='delme.png')






# #### ZONAL 


# util.nice_heatmap_subplots( [util.get_DM_command_in_2D( IM[50] ) ] , savefig='delme.png' )

# fps = float( c_dict[beam_id].config["fps"] )
# gain = float( c_dict[beam_id].config["gain"] )


# dm_mask = I2A_dict[beam_id] @  np.array( pupil_mask[beam_id] ).reshape(-1)
# dm_act_filt = dm_mask > 0.95 # ignore actuators on the edge! 


# # control matrix 
# D = np.diag( gain / fps * np.array( [dm_mask[i]/IM[i][i] if np.isfinite(1/IM[i][i]) else 0 for i in range(len(IM))]) )


# util.nice_heatmap_subplots( [D ] , savefig='delme.png' )

# pp = 0.04
# m = 50
# abb = pp * modal_basis[m] 
# dm_shm_dict[beam_id].set_data(  abb ) 


# time.sleep(1)

# fps = float( c_dict[beam_id].config["fps"] )
# gain = float( c_dict[beam_id].config["gain"] )

# i = np.mean( c_dict[beam_id].get_data( apply_manual_reduction=True ), axis=0) 
# # plot2d( i )
# # (~secondary_mask[beam_id].astype(bool))
# s = (   (i - np.mean( zwfs_pupils[beam_id] , axis=0) ) / normalized_pupils[beam_id] ).reshape(-1)
# #plot2d( ((i - np.mean( zwfs_pupils[beam_id]) ) / normalized_pupils[beam_id] ) )

# sig =  ( I2A_dict[beam_id] @ s  )

# err = D @  sig

# M2Ctmp = np.eye(140) 

# reco =  util.get_DM_command_in_2D( (M2Ctmp @ err).T  )

# res = abb - reco 

# rmse = np.sqrt( np.mean( (res)**2 ))

# im_list = [abb ,  i.T , util.get_DM_command_in_2D(sig), util.get_DM_command_in_2D( dm_act_filt ) * reco,  util.get_DM_command_in_2D( dm_act_filt ) * res ]

# title_list = ["disturbance", "intensity", "signal", "reco.", "residual"]
# #vlims = [[np.nanmin(thing), np.nanmax(thing)] for thing in im_list[:-1]] 
# #vlims.append( vlims[-1] )
# cbar_label_list = ["DM UNITS", "ADU", "ADU", "DM UNITS", "DM UNITS"]
# #vlims= vlims,
# util.nice_heatmap_subplots( im_list, title_list = title_list, cbar_label_list=cbar_label_list,  savefig='delme.png')












# ###########################################################

# # def setup(beam_ids, global_camera_shm, toml_file) :

# #     NNN = 10 # number of time get_data() called / appended

# #     print( 'setting up controllino and MDS ZMQ communication')

# #     controllino_port = '172.16.8.200'

# #     myco = co.Controllino(controllino_port)


# #     print( 'Reading in configurations') 

# #     I2A_dict = {}
# #     for beam_id in beam_ids:

# #         # read in TOML as dictionary for config 
# #         with open(toml_file.replace('#',f'{beam_id}'), "r") as f:
# #             config_dict = toml.load(f)
# #             # Baldr pupils from global frame 
# #             baldr_pupils = config_dict['baldr_pupils']
# #             I2A_dict[beam_id] = config_dict[f'beam{beam_id}']['I2A']


# #     # Set up global camera frame SHM 
# #     print('Setting up camera. You should manually set up camera settings before hand')
# #     c = shm(global_camera_shm)

# #     # set up DM SHMs 
# #     print( 'setting up DMs')
# #     dm_shm_dict = {}
# #     for beam_id in beam_ids:
# #         dm_shm_dict[beam_id] = dmclass( beam_id=beam_id )
# #         # zero all channels
# #         dm_shm_dict[beam_id].zero_all()
# #         # activate flat (does this on channel 1)
# #         dm_shm_dict[beam_id].activate_flat()
# #         # apply dm flat offset (does this on channel 2)
# #         #dm_shm_dict[beam_id].set_data( np.array( dm_flat_offsets[beam_id] ) )
    


# #     # Get Darks
# #     print( 'getting Darks')
# #     myco.turn_off("SBB")
# #     time.sleep(15)
# #     darks = []
# #     for _ in range(NNN):
# #         darks.append(  c.get_data() )

# #     darks = np.array( darks ).reshape(-1, darks[0].shape[1], darks[0].shape[2])

# #     myco.turn_on("SBB")
# #     time.sleep(10)

# #     # crop for each beam
# #     dark_dict = {}
# #     for beam_id in beam_ids:
# #         r1,r2,c1,c2 = baldr_pupils[f"{beam_id}"]
# #         cropped_imgs = [nn[r1:r2,c1:c2] for nn in darks]
# #         dark_dict[beam_id] = cropped_imgs


# #     # Get reference pupils (later this can just be a SHM address)
# #     zwfs_pupils = {}
# #     clear_pupils = {}
# #     rel_offset = 200.0 #um phasemask offset for clear pupil
# #     print( 'Moving FPM out to get clear pupils')
# #     for beam_id in beam_ids:
# #         message = f"moverel BMX{beam_id} {rel_offset}"
# #         res = send_and_get_response(message)
# #         print(res) 
# #         time.sleep( 1 )
# #         message = f"moverel BMY{beam_id} {rel_offset}"
# #         res = send_and_get_response(message)
# #         print(res) 
# #         time.sleep(10)


# #     #Clear Pupil
# #     print( 'gettin clear pupils')
# #     N0s = []
# #     for _ in range(NNN):
# #          N0s.append(  c.get_data() )
# #     N0s = np.array(  N0s ).reshape(-1,  N0s[0].shape[1],  N0s[0].shape[2])


# #     for beam_id in beam_ids:
# #         r1,r2,c1,c2 = baldr_pupils[f"{beam_id}"]
# #         cropped_imgs = [nn[r1:r2,c1:c2] for nn in N0s]
# #         clear_pupils[beam_id] = cropped_imgs

# #         # move back 
# #         print( 'Moving FPM back in beam.')
# #         message = f"moverel BMX{beam_id} {-rel_offset}"
# #         res = send_and_get_response(message)
# #         print(res) 
# #         time.sleep(1)
# #         message = f"moverel BMY{beam_id} {-rel_offset}"
# #         res = send_and_get_response(message)
# #         print(res) 
# #         time.sleep(10)


# #     # check the alignment is still ok 
# #     beam = int( input( "\ndo you want to check the phasemasks for a beam. Enter beam number (1,2,3,4) or 0 to continue\n") )
# #     while beam :
# #         save_tmp = 'delme.png'
# #         print(f'open {save_tmp } to see generated images after each iteration')
        
# #         move_relative_and_get_image(cam=c, beam=beam, baldr_pupils = baldr_pupils, phasemask=state_dict["socket"],  savefigName = save_tmp, use_multideviceserver=True )
        
# #         beam = int( input( "\ndo you want to check the phasemasks for a beam. Enter beam number (1,2,3,4) or 0 to continue\n") )
    

# #     # ZWFS Pupil
# #     print( 'Getting ZWFS pupils')
# #     I0s = []
# #     for _ in range(NNN):
# #         I0s.append(  c.get_data() )
# #     I0s = np.array(  I0s ).reshape(-1,  I0s[0].shape[1],  I0s[0].shape[2])

# #     for beam_id in beam_ids:
# #         r1,r2,c1,c2 = baldr_pupils[f"{beam_id}"]
# #         #cropped_img = interpolate_bad_pixels(img[r1:r2, c1:c2], bad_pixel_mask[r1:r2, c1:c2])
# #         cropped_img = [nn[r1:r2,c1:c2] for nn in I0s] #/np.mean(img[r1:r2, c1:c2][pupil_mask[bb]])
# #         zwfs_pupils[beam_id] = cropped_img

# #     return c, dm_shm_dict, dark_dict, zwfs_pupils, clear_pupils, baldr_pupils, I2A_dict


# # def process_signal( i, I0, N0):
# #     # must be same as model cal. import from common module
# #     # i is intensity, I0 reference intensity (zwfs in), N0 clear pupil (zwfs out)
# #     return ( i - I0 ) / N0 






# # # PID and leaky integrator copied from /Users/bencb/Documents/asgard-alignment/playground/open_loop_tests_HO.py
# # class PIDController:
# #     def __init__(self, kp=None, ki=None, kd=None, upper_limit=None, lower_limit=None, setpoint=None):
# #         if kp is None:
# #             kp = np.zeros(1)
# #         if ki is None:
# #             ki = np.zeros(1)
# #         if kd is None:
# #             kd = np.zeros(1)
# #         if lower_limit is None:
# #             lower_limit = np.zeros(1)
# #         if upper_limit is None:
# #             upper_limit = np.ones(1)
# #         if setpoint is None:
# #             setpoint = np.zeros(1)

# #         self.kp = np.array(kp)
# #         self.ki = np.array(ki)
# #         self.kd = np.array(kd)
# #         self.lower_limit = np.array(lower_limit)
# #         self.upper_limit = np.array(upper_limit)
# #         self.setpoint = np.array(setpoint)
# #         self.ctrl_type = 'PID'
        
# #         size = len(self.kp)
# #         self.output = np.zeros(size)
# #         self.integrals = np.zeros(size)
# #         self.prev_errors = np.zeros(size)

# #     def process(self, measured):
# #         measured = np.array(measured)
# #         size = len(self.setpoint)

# #         if len(measured) != size:
# #             raise ValueError(f"Input vector size must match setpoint size: {size}")

# #         # Check all vectors have the same size
# #         error_message = []
# #         for attr_name in ['kp', 'ki', 'kd', 'lower_limit', 'upper_limit']:
# #             if len(getattr(self, attr_name)) != size:
# #                 error_message.append(attr_name)
        
# #         if error_message:
# #             raise ValueError(f"Input vectors of incorrect size: {' '.join(error_message)}")

# #         if len(self.integrals) != size:
# #             print("Reinitializing integrals, prev_errors, and output to zero with correct size.")
# #             self.integrals = np.zeros(size)
# #             self.prev_errors = np.zeros(size)
# #             self.output = np.zeros(size)

# #         for i in range(size):
# #             error = measured[i] - self.setpoint[i]  # same as rtc
            
# #             if self.ki[i] != 0: # ONLY INTEGRATE IF KI IS NONZERO!! 
# #                 self.integrals[i] += error
# #                 self.integrals[i] = np.clip(self.integrals[i], self.lower_limit[i], self.upper_limit[i])

# #             derivative = error - self.prev_errors[i]
# #             self.output[i] = (self.kp[i] * error +
# #                               self.ki[i] * self.integrals[i] +
# #                               self.kd[i] * derivative)
# #             self.prev_errors[i] = error

# #         return self.output

# #     def set_all_gains_to_zero(self):
# #         self.kp = np.zeros( len(self.kp ))
# #         self.ki = np.zeros( len(self.ki ))
# #         self.kd = np.zeros( len(self.kd ))
        
# #     def reset(self):
# #         self.integrals.fill(0.0)
# #         self.prev_errors.fill(0.0)
# #         self.output.fill(0.0)
        
# #     def get_transfer_function(self, mode_index=0):
# #         """
# #         Returns the transfer function for the specified mode index.

# #         Parameters:
# #         - mode_index: Index of the mode for which to get the transfer function (default is 0).
        
# #         Returns:
# #         - scipy.signal.TransferFunction: Transfer function object.
# #         """
# #         if mode_index >= len(self.kp):
# #             raise IndexError("Mode index out of range.")
        
# #         # Extract gains for the selected mode
# #         kp = self.kp[mode_index]
# #         ki = self.ki[mode_index]
# #         kd = self.kd[mode_index]
        
# #         # Numerator and denominator for the PID transfer function: G(s) = kp + ki/s + kd*s
# #         # Which can be expressed as G(s) = (kd*s^2 + kp*s + ki) / s
# #         num = [kd, kp, ki]  # coefficients of s^2, s, and constant term
# #         den = [1, 0]        # s term in the denominator for integral action
        
# #         return TransferFunction(num, den)

# #     def plot_bode(self, mode_index=0):
# #         """
# #         Plots the Bode plot for the transfer function of a specified mode.

# #         Parameters:
# #         - mode_index: Index of the mode for which to plot the Bode plot (default is 0).
# #         """
# #         # Get transfer function
# #         tf = self.get_transfer_function(mode_index)

# #         # Generate Bode plot data
# #         w, mag, phase = bode(tf)
        
# #         # Plot magnitude and phase
# #         fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 6))
        
# #         # Magnitude plot
# #         ax1.semilogx(w, mag)  # Bode magnitude plot
# #         ax1.set_title(f"Bode Plot for Mode {mode_index}")
# #         ax1.set_ylabel("Magnitude (dB)")
# #         ax1.grid(True, which="both", linestyle="--", linewidth=0.5)

# #         # Phase plot
# #         ax2.semilogx(w, phase)  # Bode phase plot
# #         ax2.set_xlabel("Frequency (rad/s)")
# #         ax2.set_ylabel("Phase (degrees)")
# #         ax2.grid(True, which="both", linestyle="--", linewidth=0.5)

# #         plt.tight_layout()
# #         plt.show()




# # def init_telem_dict(): 
# #     # i_list is intensity measured on the detector
# #     # i_dm_list is intensity interpolated onto DM actuators - it is used only in zonal_interp control methods 
# #     # s_list is processed intensity signal used in the control loop (e.g. I - I0)
# #     # e_* is control error signals 
# #     # u_* is control signals (e.g. after PID control)
# #     # c_* is DM command signals 
# #     telemetry_dict = {
# #         "time" : [],
# #         "i_list" : [],
# #         "i_dm_list":[], 
# #         "s_list" : [],
# #         "e_TT_list" : [],
# #         "u_TT_list" : [],
# #         "c_TT_list" : [], # the next TT cmd to send to ch2
# #         "e_HO_list" : [],
# #         "u_HO_list" : [], 
# #         "c_HO_list" : [], # the next H0 cmd to send to ch2 
# #         "current_dm_ch0" : [], # the current DM cmd on ch1
# #         "current_dm_ch1" : [], # the current DM cmd on ch2
# #         "current_dm_ch2" : [], # the current DM cmd on ch3
# #         "current_dm_ch3" : [], # the current DM cmd on ch4
# #         "current_dm":[], # the current DM cmd (sum of all channels)
# #         "modal_disturb_list":[],
# #         "dm_disturb_list" : []
# #         # "dm_disturb_list" : [],
# #         # "rmse_list" : [],
# #         # "flux_outside_pupil_list" : [],
# #         # "residual_list" : [],
# #         # "field_phase" : [],
# #         # "strehl": []
# #     }
# #     return telemetry_dict



# # ### using SHM camera structure
# # def move_relative_and_get_image(cam, beam, baldr_pupils, phasemask, savefigName=None, use_multideviceserver=True,roi=[None,None,None,None]):
# #     print(
# #         f"input savefigName = {savefigName} <- this is where output images will be saved.\nNo plots created if savefigName = None"
# #     )
# #     r1,r2,c1,c2 = baldr_pupils[f"{beam}"]
# #     exit = 0
# #     while not exit:
# #         input_str = input('enter "e" to exit, else input relative movement in um: x,y')
# #         if input_str == "e":
# #             exit = 1
# #         else:
# #             try:
# #                 xy = input_str.split(",")
# #                 x = float(xy[0])
# #                 y = float(xy[1])

# #                 if use_multideviceserver:
# #                     #message = f"fpm_moveabs phasemask{beam} {[x,y]}"
# #                     #phasemask.send_string(message)
# #                     message = f"moverel BMX{beam} {x}"
# #                     phasemask.send_string(message)
# #                     response = phasemask.recv_string()
# #                     print(response)

# #                     message = f"moverel BMY{beam} {y}"
# #                     phasemask.send_string(message)
# #                     response = phasemask.recv_string()
# #                     print(response)

# #                 else:
# #                     phasemask.move_relative([x, y])

# #                 time.sleep(0.5)
# #                 img = np.mean(
# #                     cam.get_data(),
# #                     axis=0,
# #                 )[r1:r2,c1:c2]
# #                 if savefigName != None:
# #                     plt.figure()
# #                     plt.imshow( np.log10( img[roi[0]:roi[1],roi[2]:roi[3]] ) )
# #                     plt.colorbar()
# #                     plt.savefig(savefigName)
# #             except:
# #                 print('incorrect input. Try input "1,1" as an example, or "e" to exit')

# #     plt.close()
