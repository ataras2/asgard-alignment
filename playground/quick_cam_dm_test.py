
import numpy as np
import time 
import zmq
import glob
import sys
import os 
import toml
import argparse
import matplotlib.pyplot as plt
from asgard_alignment.DM_shm_ctrl import dmclass
import common.DM_basis_functions as dmbases
from pyBaldr import utilities as util
from asgard_alignment import FLI_Cameras as FLI

parser = argparse.ArgumentParser(description="Baldr Pupil Fit Configuration.")


beam_id = 2
default_toml = os.path.join( "config_files", f"baldr_config_{beam_id}.toml") #os.path.dirname(os.path.abspath(__file__)), "..", "config_files", "baldr_config.toml")


# setting up socket to ZMQ communication to multi device server
parser.add_argument("--host", type=str, default="localhost", help="Server host")
parser.add_argument("--port", type=int, default=5555, help="Server port")
parser.add_argument(
    "--timeout", type=int, default=5000, help="Response timeout in milliseconds"
)

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
    help="TOML file to write/edit. Default: ../config_files/baldr_config.toml (relative to script)"
)

# Beam ids: provided as a comma-separated string and converted to a list of ints.
parser.add_argument(
    "--beam_id",
    type=lambda s: [int(item) for item in s.split(",")],
    default=[beam_id], #, 2, 3, 4],
    help="Comma-separated beam IDs to apply. Default: 1,2,3,4"
)



args=parser.parse_args()


# set up subpupils and pixel mask
with open(args.toml_file ) as file:
    pupildata = toml.load(file)
    # Extract the "baldr_pupils" section
    baldr_pupils = pupildata.get("baldr_pupils", {})


# global camera image shm 
roi = baldr_pupils[f"{beam_id}"]#[None for _ in range(4)]

c = FLI.fli(roi=roi) # #shm(args.global_camera_shm)

# DMs
dm_shm_dict = {}
#for beam_id in args.beam_id:
dm_shm_dict[beam_id] = dmclass( beam_id=beam_id )
# zero all channels
dm_shm_dict[beam_id].zero_all()
# activate flat 
dm_shm_dict[beam_id].activate_flat()



c.send_fli_cmd("set fps 100")
c.send_fli_cmd("set gain 5")


# get reference frame 
frames_a = c.get_some_frames( number_of_frames=10 , apply_manual_reduction=False)

# apply cross 
dm_shm_dict[beam_id].activate_cross()

# no immediately get some more frames. Do we see the cross 
frames_b = c.get_some_frames( number_of_frames=10 , apply_manual_reduction=False)

fig, ax = plt.subplots(1,3)

ax[0].imshow( frames_a[0] ) 
ax[1].imshow( frames_b[0] ) 
ax[2].imshow( np.mean( frames_b ,axis=0) - np.mean( frames_a ,axis=0) )
ax[0].set_title("before")
ax[1].set_title("after")
ax[2].set_title("difference")
plt.savefig('delme.png')

