## Goal is to get long exposures of clear and zwfs pupils
# on sky to analyse to stability and NCPA to move towards
# onsky baldr DM flat and better statistics 


import matplotlib.pyplot as plt
import argparse
import numpy as np
from xaosim.shmlib import shm
from asgard_alignment.DM_shm_ctrl import dmclass
from asgard_alignment import FLI_Cameras as FLI
import os
import toml
import common.DM_basis_functions as dmbases

basis = dmbases.zer_bank(2, 20) # 12x12 format

global_camera_shm = "/dev/shm/cred1.im.shm"
default_toml = os.path.join("/usr/local/etc/baldr/", "baldr_config_#.toml") 

beam_list = [1,2,3,4]

# config 
I2A_dict = {}
pupil_mask = {}
secondary_mask = {}
exterior_mask = {}
for beam_id in beam_list:

    # read in TOML as dictionary for config 
    with open(default_toml.replace('#',f'{beam_id}'), "r") as f:
        config_dict = toml.load(f)
        # Baldr pupils from global frame 
        baldr_pupils = config_dict['baldr_pupils']
        I2A_dict[beam_id] = config_dict[f'beam{beam_id}']['I2A']
        
        pupil_mask[beam_id] = config_dict.get(f"beam{beam_id}", {}).get("pupil_mask", {}).get("mask", None)

        secondary_mask[beam_id] = np.array(config_dict.get(f"beam{beam_id}", {}).get("pupil_mask", {}).get("secondary", None) )

        exterior_mask[beam_id] = np.array(config_dict.get(f"beam{beam_id}", {}).get("pupil_mask", {}).get("exterior", None) )


## DMs
dm_shm_dict = {}
for beam in beam_list:
    dm_shm_dict[beam] = dmclass( beam_id=beam, main_chn=3 ) # we poke on ch3 so we can close TT on chn 2 with rtc when building IM 
    # zero all channels


## Camera 
c = FLI.fli(global_camera_shm, roi = [None,None,None,None])


### With the pupil out get a bunch of consecutive exposures , seperated by 5s 
# within 200 samples is the mean static ? 

### if I get longer (10k) telemetry from baldr what does the PSD look like?