import sys
import os
import numpy as np 
import json
import matplotlib.pyplot as plt 
import pandas as pd 
from types import SimpleNamespace
import time
import argparse
import datetime
import zmq
from astropy.io import fits
sys.path.insert(1, "/Users/bencb/Documents/ASGARD/BaldrApp" )
#from common.baldr_core import PIDController, init_telem_dict
from common import DM_registration
from asgard_alignment import FLI_Cameras as FLI
from common import phasemask_centering_tool as pct

# import bmc and fli class 

# 
sys.path.insert(1, "/opt/Boston Micromachines/lib/Python3/site-packages/")
import bmc

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


tstamp = datetime.datetime.now().strftime("%d-%m-%YT%H.%M.%S")
tstamp_rough =  datetime.datetime.now().strftime("%d-%m-%Y")


# setting up socket to ZMQ communication to multi device server
parser = argparse.ArgumentParser(description="ZeroMQ Client and Mode setup")
parser.add_argument("--host", type=str, default="localhost", help="Server host")
parser.add_argument("--port", type=int, default=5555, help="Server port")
parser.add_argument(
    "--timeout", type=int, default=5000, help="Response timeout in milliseconds"
)
parser.add_argument(
    '--data_path',
    type=str,
    default=f"/home/heimdallr/Documents/asgard-alignment/calibration/reports/phasemask_aquisition/{tstamp_rough}/", #f"/home/heimdallr/data/phasemask_aquisition/{tstamp_rough}/",
    help="Path to the directory for storing pokeramp data. Default: %(default)s"
)
parser.add_argument(
    '--beam',
    type=str,
    default="2",
    help="what beam to look at?. Default: %(default)s"
)
parser.add_argument(
    '--phasemask_name',
    type=str,
    default="H3",
    help="what phasemask to look at (e.g. J1, J2..J5, H1, H2,...H5). Default: %(default)s - easy to find"
)
parser.add_argument(
    '--search_radius',
    type=float,
    default=100,
    help="search radius of spiral search in microns. Default: %(default)s"
)
parser.add_argument(
    '--step_size',
    type=float,
    default=20,
    help="set size in microns of square spiral search. Default: %(default)s"
)

parser.add_argument(
    '--cam_fps',
    type=int,
    default=50,
    help="frames per second on camera. Default: %(default)s"
)
parser.add_argument(
    '--cam_gain',
    type=int,
    default=5,
    help="camera gain. Default: %(default)s"
)


args = parser.parse_args()

context = zmq.Context()

context.socket(zmq.REQ)

socket = context.socket(zmq.REQ)

socket.setsockopt(zmq.RCVTIMEO, args.timeout)

server_address = f"tcp://{args.host}:{args.port}"

socket.connect(server_address)

state_dict = {"message_history": [], "socket": socket}


# paths 
DMshapes_path = "/home/heimdallr/Documents/asgard-alignment/DMShapes/"
dm_config_path = "/home/heimdallr/Documents/asgard-alignment/config_files/dm_serial_numbers.json"

#baldr_pupils_path = "/home/heimdallr/Documents/asgard-alignment/config_files/baldr_pupils_coords.json"
# with open(baldr_pupils_path, "r") as json_file:
#     baldr_pupils = json.load(json_file)


#for beam in [args.beam]:
with open(f'baldr_transform_dict_beam{args.beam}.json') as f:
    config = json.load(f)

# cropping regions for the beams pupil 
x_start, x_end, y_start, y_end = config['pupil_regions']


closed = True # 1=closed, 0 open
record_telemetry = True
telem = SimpleNamespace( **init_telem_dict() )


if not os.path.exists(args.data_path):
     print(f'made directory : {args.data_path}')
     os.makedirs(args.data_path)




# DM object 
with open(dm_config_path, "r") as f:
    dm_serial_numbers = json.load(f)

dm = {}
dm_err_flag = {}
#for beam, serial_number in dm_serial_numbers.items():
for beam in [args.beam]:
    serial_number = dm_serial_numbers[beam]
    dm[beam] = bmc.BmcDm()  # init DM object
    dm_err_flag[beam] = dm[beam].open_dm(serial_number)  # open DM
    if not dm_err_flag:
        print(f"Error initializing DM {beam}")


flatdm = {}
#for beam, serial_number in dm_serial_numbers.items():
for beam in [args.beam]:
    serial_number = dm_serial_numbers[beam]
    flatdm[beam] = pd.read_csv(
        DMshapes_path + f"{serial_number}_FLAT_MAP_COMMANDS.csv",
        header=None,
    )[0].values

# camera
roi = [None, None, None, None]
c = FLI.fli(cameraIndex=0, roi=roi)
# configure with default configuration file
config_file_name = os.path.join(c.config_file_path, "default_cred1_config.json")
c.configure_camera(config_file_name)

with open(config_file_name, "r") as file:
    camera_config = json.load(file)


# phasemask
message = f"!fpm_movetomask phasemask{args.beam} {args.phasemask_name}"
res = send_and_get_response(message)
print(res)

time.sleep(1)
img = np.mean( c.get_some_frames( number_of_frames=10, apply_manual_reduction=True ) , axis = 0 ) 
plt.figure(); plt.imshow( np.log10( img ) ) ; plt.colorbar(); plt.savefig('delme.png')


# manual centering 
#pct.move_relative_and_get_image(cam=c, beam=args.beam, phasemask=state_dict["socket"], savefigName='delme.png', use_multideviceserver=True)




##### CURRENTLY ONLY DEALING WITH ONE BEAM AT A TIME. 
disturbance = np.zeros( 140 )


# reference intensities interpolated onto registered DM actuators in pixel space
I0_dm = np.array( config['interpolated_I0'] )
N0_dm = np.array( config['interpolated_N0'] )

# Control model
slopes = np.array( [ config['control_model'][a]["slope_standard"] for a in range(140)] )
intercepts = np.array( [ config['control_model'][a]["intercept_standard"] for a in range(140)] )

# Controller
N = 140 
kp = 0. * np.ones( N)
ki = 0. * np.ones( N )
kd = 0. * np.ones( N )
setpoint = np.zeros( N )
lower_limit_pid = -100 * np.ones( N )
upper_limit_pid = 100 * np.ones( N )

ctrl_HO = PIDController(kp, ki, kd, upper_limit_pid, lower_limit_pid, setpoint)


# get an initial frame to initialize frame counter
full_img = c.get_image_in_another_region() 
current_frame_number = full_img[0][0]
previous_frame_number = current_frame_number.copy()
timeout_counter = 0
timeout_limit = 200000

apply_manual_reduction  = True

cnt = 0
while closed:
    time.sleep(0.0005) # sleep for 

    if timeout_counter > timeout_limit: 
        raise UserWarning(f'timeout! timeout_counter > {timeout_limit}, camera may be stuck')

    frame = c.get_image( apply_manual_reduction =apply_manual_reduction  )  
    current_frame_number = full_img[0][0] 
    
    if current_frame_number != previous_frame_number:
        new_frame=True
        previous_frame_number = current_frame_number.copy()
        timeout_counter = 0
    else:
        new_frame = False
        timeout_counter += 1


    if new_frame :

        # Crop the pupil region
        cropped_image = frame[y_start:y_end, x_start:x_end]

        # (2) interpolate intensities to DM 
        i_dm = DM_registration.interpolate_pixel_intensities(image = cropped_image , pixel_coords = config['dm_registration']['actuator_coord_list_pixel_space'])

        # (3) normalise 
        # current model has no normalization 
        sig = i_dm  - I0_dm
        
        # (4) apply linear model to get reconstructor 
        e_HO = slopes * sig + intercepts

        # PID 
        u_HO = ctrl_HO.process( e_HO )
        
        # forcefully remove piston 
        u_HO -= np.mean( u_HO )
        
        # send command (filter piston)
        #delta_cmd = np.zeros( len(zwfs_ns.dm.dm_flat ) )
        #delta_cmd zwfs_ns.reco.linear_zonal_model.act_filt_recommended ] = u_HO
        delta_cmd = u_HO

        if record_telemetry :
            telem.i_list.append( cropped_image )
            telem.i_dm_list.append( i_dm )
            telem.s_list.append( sig )
            telem.e_TT_list.append( np.zeros( len(e_HO) ) )
            telem.u_TT_list.append( np.zeros( len(e_HO) ) )
            telem.c_TT_list.append( np.zeros( len(delta_cmd) ) )

            telem.e_HO_list.append( e_HO )
            telem.u_HO_list.append( u_HO )
            telem.c_HO_list.append( delta_cmd )
            
        cmd = flatdm + disturbance - delta_cmd 

        if np.std( delta_cmd ) > 0.4:
            closed = False

        #send the command off 
        dm[beam].send_data( cmd )

        cnt+=1





# save telemetry

# Create a list of HDUs (Header Data Units)
hdul = fits.HDUList()

# Add each list to the HDU list as a new extension
for list_name, data_list in lists_dict.items():
    # Convert list to numpy array for FITS compatibility
    data_array = np.array(data_list, dtype=float)  # Ensure it is a float array or any appropriate type

    # Create a new ImageHDU with the data
    hdu = fits.ImageHDU(data_array)

    # Set the EXTNAME header to the variable name
    hdu.header['EXTNAME'] = list_name

    # Append the HDU to the HDU list
    hdul.append(hdu)

# Write the HDU list to a FITS file
hdul.writeto(current_path + f'{explabel}.fits', overwrite=True)


