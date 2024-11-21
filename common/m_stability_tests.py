import numpy as np
from astropy.io import fits
import os
import time
import matplotlib.pyplot as plt
import importlib
import datetime
import sys
from asgard_alignment import FLI_Cameras as FLI

"""
Nov 24 - we notice significant drifts likely coming from OAP 1 and solarstein down periscope. 
We have all four Baldr beams on the detector. This script is to run the camera and some DM 
commands to monitor 
(a) the drift of the beams on the detector
(b) the registration of the DM ono the detector 
(c) drift of the beams on the DM 
"""

#################
# Set up
#################

apply_manual_reduction = True

# light_mode = input('what mode are we running (dark, flat, etc). Input something meaningful')

tstamp = datetime.datetime.now().strftime("%d-%m-%YT%H.%M.%S")
data_path = f"/home/heimdallr/Downloads/{light_mode}_series_{tstamp}/"
if not os.path.exists(data_path):
    os.makedirs(data_path)


# set up camera object
c = FLI.fli(cameraIndex=0, roi=[None, None, None, None])
# configure with default configuration file
config_file_name = os.path.join(c.config_file_path, "default_cred1_config.json")
c.configure_camera(config_file_name)

c.send_fli_cmd("set mode globalresetcds")
time.sleep(1)
c.send_fli_cmd("set gain 1")
time.sleep(1)
c.send_fli_cmd("set fps 50")
