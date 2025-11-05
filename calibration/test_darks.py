
## TO DO  - make this script fit and generate a dark in units ADU / S / gain 

import zmq
from astropy.io import fits
import argparse
import time
import os
import datetime
import numpy as np
from asgard_alignment import FLI_Cameras as FLI # depends on xao shm

bp = fits.open( "calibration/cal_data/calibration_frames/22-03-2025/master_bad_pixel_map_fps-200.000_2025-03-22T05-17-27.fits")
d = fits.open( "calibration/cal_data/calibration_frames/22-03-2025/master_darks_adu_p_sec_fps-200.000_2025-03-22T05-17-27.fits")
#cc =  shm("/dev/shm/cred1.im.shm")  # testing 
c = FLI.fli()

conf = c.get_camera_config()


# get a dark and look at residuals from master dark 
context = zmq.Context()
context.socket(zmq.REQ)
mds_socket = context.socket(zmq.REQ)
mds_socket.setsockopt(zmq.RCVTIMEO, args.timeout)
mds_socket.connect( f"tcp://{args.host}:{args.port}")
