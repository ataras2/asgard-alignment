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


tstamp = datetime.datetime.now().strftime("%Y-%m-%dT%H-%M-%S")

parser = argparse.ArgumentParser(description="Calibrate new baldr reference intensity flat with lucky imaging.")

arg_default_toml = os.path.join("/usr/local/etc/baldr/", f"baldr_config_#.toml") 

parser.add_argument(
    "--toml_file",
    type=str,
    default=arg_default_toml,
    help="TOML file pattern (replace # with beam_id) to write/edit. Default: ../config_files/baldr_config.toml (relative to script)"
)
# Camera shared memory path
parser.add_argument(
    "--savepath",
    type=str,
    default="/home/asg/ben_bld_data/15-9-25night6/",
    help="where to save results . Default: /home/asg/ben_bld_data/15-9-25night6/"
)

# Beam ids: provided as a comma-separated string and converted to a list of ints.
parser.add_argument(
    "--beam_id",
    type=int, #lambda s: [int(item) for item in s.split(",")],
    default=1, # 1, 2, 3, 4],
    help="which beam to apply. Default: 1"
)

parser.add_argument(
    "--phasemask",
    type=str,
    default="H3",
    help="which phasemask to use "
)
parser.add_argument(
    "--user_input",
    type=int,
    default=0,
    help="do you want to check plots and input if to update the flat? Default 0 (false)"
)


args=parser.parse_args()

beam_id = args.beam_id 
default_toml =  args.toml_file
phasemask = args.phasemask
savepath = args.savepath


# Baldr RTC server addresses to update live RTC via zmq
SERVER_ADDR_DICT = {1:"tcp://127.0.0.1:6662",
                    2:"tcp://127.0.0.1:6663",
                    3:"tcp://127.0.0.1:6664",
                    4:"tcp://127.0.0.1:6665"}

# the defined shared memory address for the baldr subframes
global_camera_shm = f"/dev/shm/baldr{beam_id}.im.shm" 

# open the 
with open(default_toml.replace('#',f'{beam_id}'), "r") as f:

    config_dict = toml.load(f)

    baldr_pupils = config_dict.get("baldr_pupils", {})
    
    crop_mode_offset  = config_dict.get("crop_mode_offset", {})
    #  read in the current calibrated matricies 
    pupil_mask = np.array(config_dict.get(f"beam{beam_id}", {}).get("pupil_mask", {}).get("mask", None) ).astype(bool)   # matrix bool
    I2A = np.array( config_dict[f'beam{beam_id}']['I2A'] )
    IM = np.array(config_dict.get(f"beam{beam_id}", {}).get(f"{phasemask}", {}).get("ctrl_model", None).get("IM", None) ).astype(float)
    M2C = np.array(config_dict.get(f"beam{beam_id}", {}).get(f"{phasemask}", {}).get("ctrl_model", None).get("M2C", None) ).astype(float)

    pupil_mask = config_dict.get(f"beam{beam_id}", {}).get("pupil_mask", {}).get("mask", None)

    secondary_mask = np.array(config_dict.get(f"beam{beam_id}", {}).get("pupil_mask", {}).get("secondary", None) )

    exterior_mask = np.array(config_dict.get(f"beam{beam_id}", {}).get("pupil_mask", {}).get("exterior", None) )

    # # define our Tip/Tilt or lower order mode index on zernike DM basis 
    LO = config_dict.get(f"beam{beam_id}", {}).get(f"{phasemask}", {}).get("ctrl_model", None).get("LO", None)

    # tight (non-edge) pupil filter
    inside_edge_filt = np.array(config_dict.get(f"beam{beam_id}", {}).get(f"{phasemask}", {}).get("ctrl_model", None).get("inner_pupil_filt", None) )#.astype(bool)
    # clear pupil 
    I0 = np.array(config_dict.get(f"beam{beam_id}", {}).get(f"{phasemask}", {}).get("ctrl_model", None).get("I0", None) )
    N0 = np.array(config_dict.get(f"beam{beam_id}", {}).get(f"{phasemask}", {}).get("ctrl_model", None).get("N0", None) )#.astype(bool)
    # secondary filter
    sec = np.array(config_dict.get(f"beam{beam_id}" , {}).get(f"{phasemask}", {}).get("ctrl_model",None).get("secondary", None) )
    poke_amp = config_dict.get(f"beam{beam_id}", {}).get(f"{phasemask}", {}).get("ctrl_model", None).get("poke_amp", None)
    camera_config = config_dict.get(f"beam{beam_id}", {}).get(f"{phasemask}", {}).get("ctrl_model", None).get("camera_config", None)


# shared memory camera object 
c = shm(global_camera_shm, nosem=False)

# list to hold images
imgs = []

# how many do we capture 
Ncaptures = 10000

# go and capture
for _ in range(Ncaptures):
    time.sleep(0.01)

    imgs.append( c.get_data() )

# external signal (light scattered outside of pupil by phasemask which scales with strehl)
ext_signal = np.array( [np.mean( ii[exterior_mask.astype(bool)] ) for ii in imgs] )

# cut off quantile for lucky images
lucky_cutoff = np.quantile( ext_signal , 0.99) #10k samples => 99th perc. keeps 100 samples 

# also look at the unlucky ones for reference that this is working 
unlucky_cutoff = np.quantile( ext_signal , 0.10)

# filter the lucky external pupil signals  
lucky_ext_signals = np.array(ext_signal)[ext_signal > lucky_cutoff]

# if np.mean(lucky_ext_signals) < 0.1 * np.mean( I0[exterior_mask] ) : # if the lucky exterior signals are less than 10% of the internal reference exterior signal then we warn the user 
#     print("WARNING : lucky exterior signals are less than 10% of the internal reference exterior signal. May not be a good reference")

# filter the images 
lucky_imgs = np.array(imgs)[ext_signal > lucky_cutoff]

unlucky_imgs = np.array(imgs)[ext_signal < unlucky_cutoff]

# aggregate our lucky and unlucky ones 
I0_bad = np.mean( unlucky_imgs, axis=0)
I0_unlucky_ref = I0_bad / np.sum( I0_bad )

I0_new = np.mean( lucky_imgs, axis=0)
I0_ref = I0_new / np.sum( I0_new ) 

I0_dm_ref = I2A @ I0_ref.reshape(-1) 

# plot results 
img_list = [I0.reshape(32,32), I0_ref ,I0.reshape(32,32) - I0_ref, I0_unlucky_ref ]
title_list = ["original internal I0", "lucky onsky I0","delta", "unlucky onsky I0"]

util.nice_heatmap_subplots(im_list = img_list,
                           title_list = title_list, 
                           vlims=[[0,np.max(I0_ref)] for _ in range(len(img_list))])

img_fname = savepath + f"beam{beam_id}_I0_onsky_flat_{tstamp}.jpeg"
plt.savefig(img_fname, bbox_inches='tight')
if args.user_input:
    plt.show() # let the user review it 
else:
    plt.close()
# # On DM
# img_list = [util.get_DM_command_in_2D( I2A@I0 ), 
#             util.get_DM_command_in_2D(I2A @ (I0_ref.reshape(-1))) ,
#             util.get_DM_command_in_2D( I2A @(I0.reshape(32,32) - I0_ref).reshape(-1) )]
# title_list = ["original internal I0", "lucky onsky I0","delta"]

# util.nice_heatmap_subplots(im_list = img_list,
#                            title_list = title_list)

if args.user_input:
    update_flat = input("review the flat and enter 1 if we should update rtc")
else:
    update_flat = '1' # update automatically 
if update_flat=='1': 
    # connect to Baldr RTC socket and update I2A via ZMQ
    addr = SERVER_ADDR_DICT[beam_id] # "tcp://127.0.0.1:6662"  # this will change depending on if we are in simulation mode
    ctx = zmq.Context.instance()
    s = ctx.socket(zmq.REQ)
    s.RCVTIMEO = 5000  # ms
    s.SNDTIMEO = 5000  # ms
    s.connect(addr)

    # get the current config file 
    s.send_string('list_rtc_fields ""')
    rep = s.recv_json()

    # update the field (example!)
    #s.send_string('set_rtc_field "inj_signal.freq_hz",0.04')
    #rep = s.recv_json()


    # I0
    s.send_string(f'set_rtc_field "reference_pupils.I0",{I0_ref.reshape(-1).tolist()}')
    rep = s.recv_json()

    print( "sucess?", rep['ok'] )
    if not rep['ok'] :
        print( '  error: ', rep['error'] )

    #I0-dm
    s.send_string(f'set_rtc_field "reference_pupils.I0_dm",{I0_dm_ref.reshape(-1).tolist()}')
    rep = s.recv_json()

    print( "sucess?", rep['ok'] )
    if not rep['ok'] :
        print( '  error: ', rep['error'] )

    # I0-dm_runtime
    s.send_string(f'set_rtc_field "I0_dm_runtime",{I0_dm_ref.reshape(-1).tolist()}')
    rep = s.recv_json()

    print( "sucess?", rep['ok'] )
    if not rep['ok'] :
        print( '  error: ', rep['error'] )



    # close connection
    s.setsockopt(zmq.LINGER, 0)   # don't wait on unsent msgs
    s.close()  # closes the socket


# save the fits of the images 
from astropy.io import fits 
hdulist = fits.HDUList([])
hdu = fits.ImageHDU(imgs)
hdulist.append(hdu)
frame_fname = savepath + f"beam{beam_id}_I0_onsky_flat_{tstamp}.fits"
hdulist.writeto(frame_fname, overwrite=True)

print( f"saved the frames at {frame_fname}")

print("done")


