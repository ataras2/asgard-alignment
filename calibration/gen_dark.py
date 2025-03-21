
# setting up socket to ZMQ communication to multi device server
import zmq
from astropy.io import fits
import argparse
import time
import numpy as np
from asgard_alignment import FLI_Cameras as FLI

from xaosim.shmlib import shm

parser = argparse.ArgumentParser(description="generate darks")

parser.add_argument(
    '--data_path',
    type=str,
    default=f"/home/asg/Progs/repos/asgard-alignment/calibration/cal_data/darks/darks_[replace_this_with_settings].fits",
    help="Path to the directory for storing dark data. Default: %(default)s"
)
parser.add_argument("--no_frames", type=int, default=1000, help="number of frames to take for dark")
parser.add_argument("--host", type=str, default="localhost", help="Server host")
parser.add_argument("--port", type=int, default=5555, help="Server port")
parser.add_argument(
    "--timeout", type=int, default=5000, help="Response timeout in milliseconds"
)



args = parser.parse_args()

context = zmq.Context()
context.socket(zmq.REQ)
mds_socket = context.socket(zmq.REQ)
mds_socket.setsockopt(zmq.RCVTIMEO, args.timeout)
mds_socket.connect( f"tcp://{args.host}:{args.port}")

sleeptime = 60 # seconds <--- required to get ride of persistance of pupils 

# start camera 
#cc =  shm("/dev/shm/cred1.im.shm")  # testing 
c = FLI.fli()


# try turn off source 
#my_controllino.turn_off("SBB")
message = "off SBB"
mds_socket.send_string(message)
response = mds_socket.recv_string()#.decode("ascii")
print( response )

print(f'turning off source and waiting {sleeptime}s')
time.sleep(sleeptime) # wait a bit to settle


print('...getting frames')
# testing 
dark_list = c.get_some_frames(number_of_frames = args.no_frames, apply_manual_reduction=False, timeout_limit = 20000 )

# try turn source back on 
#my_controllino.turn_on("SBB")
message = "on SBB"
mds_socket.send_string(message)
response = mds_socket.recv_string()#.decode("ascii")
print( response )
time.sleep(2)

#f"/home/asg/Progs/repos/asgard-alignment/calibration/cal_data/darks/dark_{}.fits"

# Create PrimaryHDU using FRAMES
primary_hdu = fits.PrimaryHDU(dark_list)
primary_hdu.header['EXTNAME'] = 'DARK_FRAMES'  # This is not strictly necessary for PrimaryHDU

# Append camera configuration to the primary header
config_tmp = c.get_camera_config()
for k, v in config_tmp.items():
    primary_hdu.header[k] = v

# Create HDUList and add the primary HDU
hdulist = fits.HDUList([primary_hdu])

hdu = fits.ImageHDU( np.mean( dark_list, axis = 0) )
hdu.header['EXTNAME'] = "MASTER DARK"
hdulist.append(hdu)

hdulist.writeto(args.data_path.replace("[replace_this_with_settings]",f"fps-{config_tmp['fps']}_gain-{config_tmp['gain']}"), overwrite=True)



print(f"DONE. wrote dark file to {args.data_path}")

