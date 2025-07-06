import os
import numpy as np 
import argparse
#import threading
#import zmq
#import time
import toml 
import matplotlib.pyplot as plt
import time 
from asgard_alignment import FLI_Cameras as FLI
from asgard_alignment.DM_shm_ctrl import dmclass
import pyBaldr.utilities as util 


default_toml = os.path.join("/usr/local/etc/baldr/", "baldr_config_#.toml")#"/home/asg/Progs/repos/asgard-alignment/config_files/baldr_config_#_stable.toml"

parser = argparse.ArgumentParser(description="closed")


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

parser.add_argument(
    "--number_of_iterations",
    type=int,
    default=1000,
    help="number of iterations to run"
)

# Beam ids: provided as a comma-separated string and converted to a list of ints.
parser.add_argument(
    "--beam_id",
    type=int,
    default=1,
    help="beam id (integrer)"
)


parser.add_argument(
    "--phasemask",
    type=str,
    default="H3",
    help="Comma-separated beam IDs to apply. Default: 1,2,3,4"
)

parser.add_argument("--fig_path", 
                    type=str, 
                    default=None, 
                    help="path/to/output/image/ for the saved figures")



args=parser.parse_args()


with open(args.toml_file.replace('#',f'{args.beam_id}'), "r") as f:
    print(f"using {f}")
    config_dict = toml.load(f)
    
    # Baldr pupils from global frame 
    baldr_pupils = config_dict['baldr_pupils']
    I2A = np.array( config_dict[f'beam{args.beam_id}']['I2A'] )
    
    # image pixel filters
    pupil_mask = np.array(config_dict.get(f"beam{args.beam_id}", {}).get("pupil_mask", {}).get("mask", None) ).astype(bool)   # matrix bool
    exter_mask = np.array(config_dict.get(f"beam{args.beam_id}", {}).get("pupil_mask", {}).get("exterior", None) ).astype(bool) # matrix bool
    secon_mask = np.array(config_dict.get(f"beam{args.beam_id}", {}).get("pupil_mask", {}).get("secondary", None) ).astype(bool) # matrix bool

    # ctrl model 
    IM = np.array(config_dict.get(f"beam{args.beam_id}", {}).get(f"{args.phasemask}", {}).get("ctrl_model", None).get("IM", None) ).astype(float)
    I2M_raw = np.array(config_dict.get(f"beam{args.beam_id}", {}).get(f"{args.phasemask}", {}).get("ctrl_model", None).get("I2M", None) ).astype(float)
    I2M_LO_raw = np.array(config_dict.get(f"beam{args.beam_id}", {}).get(f"{args.phasemask}", {}).get("ctrl_model", None).get("I2M_LO", None) ).astype(float)
    I2M_HO_raw = np.array(config_dict.get(f"beam{args.beam_id}", {}).get(f"{args.phasemask}", {}).get("ctrl_model", None).get("I2M_HO", None) ).astype(float)
    M2C = np.array(config_dict.get(f"beam{args.beam_id}", {}).get(f"{args.phasemask}", {}).get("ctrl_model", None).get("M2C", None) ).astype(float)
    M2C_LO = np.array(config_dict.get(f"beam{args.beam_id}", {}).get(f"{args.phasemask}", {}).get("ctrl_model", None).get("M2C_LO", None) ).astype(float)
    M2C_HO = np.array(config_dict.get(f"beam{args.beam_id}", {}).get(f"{args.phasemask}", {}).get("ctrl_model", None).get("M2C_HO", None) ).astype(float)
    I0 = np.array(config_dict.get(f"beam{args.beam_id}", {}).get(f"{args.phasemask}", {}).get("ctrl_model", None).get("I0", None) ).astype(float)
    N0 = np.array(config_dict.get(f"beam{args.beam_id}", {}).get(f"{args.phasemask}", {}).get("ctrl_model", None).get("N0", None) ).astype(float)
    N0i = np.array(config_dict.get(f"beam{args.beam_id}", {}).get(f"{args.phasemask}", {}).get("ctrl_model", None).get("norm_pupil", None) ).astype(float)
    
    # used to normalize exterior and bad pixels in N0 (calculation of N0i)
    inside_edge_filt = np.array(config_dict.get(f"beam{args.beam_id}", {}).get(f"{args.phasemask}", {}).get("ctrl_model", None).get("inner_pupil_filt", None) )#.astype(bool)
    
    # reduction products
    IM_cam_config = config_dict.get(f"beam{args.beam_id}", {}).get(f"{args.phasemask}", {}).get("ctrl_model", None).get("camera_config", None) # dictionary
    
    bad_pixel_mask = np.array( config_dict.get(f"beam{args.beam_id}", {}).get(f"{args.phasemask}", {}).get("ctrl_model", None).get("bad_pixel_mask", None) )#.astype(bool)
    bias = np.array( config_dict.get(f"beam{args.beam_id}", {}).get(f"{args.phasemask}", {}).get("ctrl_model", None).get("bias", None) ).astype(float)
    dark = np.array( config_dict.get(f"beam{args.beam_id}", {}).get(f"{args.phasemask}", {}).get("ctrl_model", None).get("dark", None) ).astype(float)

    dm_flat = np.array( config_dict.get(f"beam{args.beam_id}", {}).get(f"{args.phasemask}", {}).get("ctrl_model", None).get("DM_flat", None) )

    strehl_coe_sec = np.array( config_dict.get(f"beam{args.beam_id}", {}).get("strehl_model", None).get("secondary",None) ).astype(float) #coe, intercept (on central secondary pixel, ADU/s/gain)
    strehl_coe_ext = np.array( config_dict.get(f"beam{args.beam_id}", {}).get("strehl_model", None).get("exterior",None) ).astype(float)#coe, intercept (on median signal, ADU/s/gain)


no_roi = [None, None, None, None] #x1,x2,y1,y2

# Camera 
c = FLI.fli(args.global_camera_shm, 
            roi = no_roi,#baldr_pupils[f'{args.beam_id}'], 
            quick_startup=False) 




# we read the number of reads without reset from camera to define window size 
Ns_burst = int(FLI.extract_value( c.send_fli_cmd("nbreadworeset raw") ))
fs = float(FLI.extract_value( c.send_fli_cmd("fps raw") ))
T = Ns_burst / fs # burst period
t_bw = np.arange(0, T, 1/fs) # relative time sample bloc for burst

# initialize and array with the set window size
burst_window = np.zeros( Ns_burst )
time_window = np.zeros( Ns_burst )

idx_window =  np.zeros( Ns_burst )# dont need this apart form bug shooting 

# No samples to fit line to
# N = 15 
dt_window = 1e6 # how much of a time window we use for fit (nano seconds) 
for _ in range(1000):
    img = c.get_image()
    px_i = 100
    px_j = 100
    reset_cnt_down = img[0,2] # number of samples until the reset (countdown)

    idx = Ns_burst - reset_cnt_down - 1 # idx in our burst sequence 
    
    burst_window[idx] = img[px_i, px_j]
    now = time.process_time() #time.time_ns()
    time_window[idx] = now #ns

    idx_window[idx] = idx # dont need this apart form bug shooting 

    # # get the indicies to fit the line that wrap around the burst window boundary
    # # best to filter of a rolling time window (so discards skipped frames too!)
    time_filt = time_window > now - dt_window

    # fit line on this subwindow 
    m, b = np.polyfit(time_window[time_filt] , 
                burst_window[time_filt], 
                deg = 1)
    
    

i0,i1 = 0, 150    
fig,ax = plt.subplots(3,sharex=True)
ax[0].plot(burst_window[i0:i1] )
ax[1].plot(time_window[i0:i1] )
ax[2].plot(idx_window[i0:i1] )
plt.show()
