import numpy as np
from astropy.io import fits
import os
import time
import matplotlib.pyplot as plt
import importlib
import json
import datetime
import sys
import toml
import pandas as pd
import argparse
import zmq
from scipy.optimize import leastsq
from scipy.ndimage import gaussian_filter, label, find_objects
import atexit 

from asgard_alignment import FLI_Cameras as FLI
import common.DM_basis_functions
import common.phasescreens as ps
from common import phasemask_centering_tool as pct
import pyBaldr.utilities as util

sys.path.insert(1, "/opt/Boston Micromachines/lib/Python3/site-packages/")
import bmc

import matplotlib 
matplotlib.use('Agg') # helps avoid freezing in remote sessions

"""
Apply Kolmogorov phasescreens across the DMs (4 by default) and records images on the CRED ONE
default mode globalresetcds with setup taken from default_cred1_config.json
user can change fps and gain as desired, the 
"""


def close_all_dms():
    try:
        for b in dm:
            dm[b].close_dm()
        print("All DMs have been closed.")
    except Exception as e:
        print(f"dm object doesn't seem to exist, probably already closed")
# Register the cleanup function to run at script exit
atexit.register(close_all_dms)


def get_motor_states_as_list_of_dicts( ): 

    motor_names = ["SDLA", "SDL12", "SDL34", "SSS", "BFO"]
    motor_names_no_beams = [
                "HFO",
                "HTPP",
                "HTPI",
                "HTTP",
                "HTTI",
                "BDS",
                "BTT",
                "BTP",
                "BMX",
                "BMY",
            ]


    for motor in motor_names_no_beams:
        for beam_number in range(1, 5):
            motor_names.append(f"{motor}{beam_number}")

    states = []
    for name in motor_names:
        message = f"read {name}"
        res = send_and_get_response(message)

        if "NACK" in res:
            is_connected = False
        else:
            is_connected = True

        state = {
            "name": name,
            "is_connected": is_connected,
        }
        if is_connected:
            state["position"] = float(res)

        states.append(state)

    return states


def save_motor_states_as_hdu(motor_states):
    """
    Create an HDU for motor states as a binary table.

    Parameters:
    - motor_states (list of dict): List of motor state dictionaries.

    Returns:
    - fits.BinTableHDU: The binary table HDU containing motor states.
    """
    # Prepare columns for the FITS binary table
    motor_names = [state["name"] for state in motor_states]
    is_connected = [state["is_connected"] for state in motor_states]
    positions = [state.get("position", np.nan) for state in motor_states]  # Use NaN for missing positions

    col1 = fits.Column(name="MotorName", format="20A", array=np.array(motor_names))  # ASCII strings
    col2 = fits.Column(name="IsConnected", format="L", array=np.array(is_connected))  # Logical (boolean)
    col3 = fits.Column(name="Position", format="E", array=np.array(positions, dtype=np.float32))  # Float32

    # Create the binary table HDU
    cols = fits.ColDefs([col1, col2, col3])
    return fits.BinTableHDU.from_columns(cols, name="MotorStates")



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



# paths and timestamps
tstamp = datetime.datetime.now().strftime("%d-%m-%YT%H.%M.%S")
tstamp_rough =  datetime.datetime.now().strftime("%d-%m-%Y")

# default data paths 
with open( "config_files/file_paths.json") as f:
    default_path_dict = json.load(f)

# positions to put thermal source on and take it out to empty position to get dark
source_positions = {"SSS": {"empty": 80.0, "SBB": 65.5}}


# setting up socket to ZMQ communication to multi device server
parser = argparse.ArgumentParser(description="ZeroMQ Client and Mode setup")
parser.add_argument("--host", type=str, default="172.16.8.6", help="Server host")
parser.add_argument("--port", type=int, default=5555, help="Server port")
parser.add_argument(
    "--timeout", type=int, default=5000, help="Response timeout in milliseconds"
)

parser.add_argument(
    '--dm_config_path',
    type=str,
    default="/home/asg/Progs/repos/asgard-alignment/config_files/dm_serial_numbers.json",
    help="Path to the DM configuration file. Default: %(default)s"
)
parser.add_argument(
    '--DMshapes_path',
    type=str,
    default="/home/asg/Progs/repos/asgard-alignment/DMShapes/",
    help="Path to the directory containing DM shapes. Default: %(default)s"
)
parser.add_argument(
    '--data_path',
    type=str,
    default=f"/home/heimdallr/data/baldr_calibration/{tstamp_rough}/",
    help="Path to the directory for storing phasescreen data. Default: %(default)s"
)

parser.add_argument(
    '--phasemask_name',
    type=str,
    default="H3",
    help="which phasemask? (J1-5 or H1-5). Default: %(default)s."
)

parser.add_argument(
    '--number_of_rolls',
    type=int,
    default=1000,
    help="number of phasescreen iterations (rolls) on the DM. Default: %(default)s."
)

parser.add_argument(
    '--scaling_factor',
    type=float,
    default=0.05,
    help="Scaling factor to the amplitude of the phasescreen applied to the DM. Default: %(default)s. Don't go too high!"
)


parser.add_argument(
    '--number_images_recorded_per_cmd',
    type=int,
    default=5,
    help="Number of images recorded per command (usually we take the average of these). Default: %(default)s."
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
    default=1,
    help="camera gain. Default: %(default)s"
)


parser.parse_args()

args = parser.parse_args()

context = zmq.Context()

context.socket(zmq.REQ)

socket = context.socket(zmq.REQ)

socket.setsockopt(zmq.RCVTIMEO, args.timeout)

server_address = f"tcp://{args.host}:{args.port}"

socket.connect(server_address)

state_dict = {"message_history": [], "socket": socket}




# Baldr pupils (for checking phasemask alignment before beginning)
# baldr_pupils_path = default_path_dict['baldr_pupil_crop'] #"/home/asg/Progs/repos/asgard-alignment/config_files/baldr_pupils_coords.json"
# with open(baldr_pupils_path, "r") as json_file:
#     baldr_pupils = json.load(json_file)

baldr_pupils_path = default_path_dict["pupil_crop_toml"] #"/home/asg/Progs/repos/asgard-alignment/config_files/baldr_pupils_coords.json"

# Load the TOML file
with open(baldr_pupils_path) as file:
    pupildata = toml.load(file)

# Extract the "baldr_pupils" section
baldr_pupils = pupildata.get("baldr_pupils", {})


#DMshapes_path = args.DMshapes_path #"/home/asg/Progs/repos/asgard-alignment/DMShapes/"
#dm_config_path = #"/home/asg/Progs/repos/asgard-alignment/config_files/dm_serial_numbers.json"
# data_path = f"/home/heimdallr/data/pokeramp/{tstamp_rough}/"
if not os.path.exists(args.data_path):
     os.makedirs(args.data_path)


########## ########## ##########
########## set up camera object
roi = [None, None, None, None]
c = FLI.fli(cameraIndex=0, roi=roi)
# configure with default configuration file
config_file_name = os.path.join(c.config_file_path, "default_cred1_config.json")
c.configure_camera(config_file_name)

with open(config_file_name, "r") as file:
    camera_config = json.load(file)

apply_manual_reduction = True

c.send_fli_cmd("set mode globalresetcds")
time.sleep(1)
c.send_fli_cmd(f"set gain {args.cam_gain}")
time.sleep(1)
c.send_fli_cmd(f"set fps {args.cam_fps}")

c.start_camera()




# phasemask
for beam in [1,2,3,4]:
    message = f"fpm_movetomask phasemask{beam} {args.phasemask_name}"
    res = send_and_get_response(message)
    print(res)
    time.sleep(2)

    
beam = int( input( "do you want to check the phasemasks for a beam. Enter beam number (1,2,3,4) or 0 to continue") )

while beam :
    print( 'we save images as delme.png in asgard-alignment project - open it!')
    img = np.sum( c.get_some_frames( number_of_frames=100, apply_manual_reduction=True ) , axis = 0 ) 
    r1,r2,c1,c2 = baldr_pupils[str(beam)]
    #print( r1,r2,c1,c2  )
    plt.figure(); plt.imshow( np.log10( img[r1:r2,c1:c2] ) ) ; plt.colorbar(); plt.savefig('delme.png')

    # time.sleep(5)

    # manual centering 
    pct.move_relative_and_get_image(cam=c, beam=beam, phasemask=state_dict["socket"], savefigName='delme.png', use_multideviceserver=True, roi=[r1,r2,c1,c2 ])

    beam = int( input( "do you want to check the phasemask alignment for a particular beam. Enter beam number (1,2,3,4) or 0 to continue") )



########## set up DMs
with open(args.dm_config_path, "r") as f:
    dm_serial_numbers = json.load(f)

dm = {}
dm_err_flag = {}
for beam, serial_number in dm_serial_numbers.items():
    dm[beam] = bmc.BmcDm()  # init DM object
    dm_err_flag[beam] = dm[beam].open_dm(serial_number)  # open DM
    if not dm_err_flag:
        print(f"Error initializing DM {beam}")


flatdm = {}
for beam, serial_number in dm_serial_numbers.items():
    flatdm[beam] = pd.read_csv(
        args.DMshapes_path + f"{serial_number}_FLAT_MAP_COMMANDS.csv",
        header=None,
    )[0].values



########## ########## ##########
# ACTION
# ======== Source out first for dark

state_dict["socket"].send_string(f"!moveabs SSS {source_positions['SSS']['empty']}")
res = socket.recv_string()
print(f"Response: {res}")

time.sleep(5)


DARK_list = []
DARK_list = c.get_some_frames(number_of_frames=100, apply_manual_reduction=True)

time.sleep(1)


state_dict["socket"].send_string(f"!moveabs SSS {source_positions['SSS']['SBB']}")
res = socket.recv_string()
print(f"Response: {res}")

time.sleep(5)

# ======== reference image with FPM OUT
# fourier tip to go off phase mask
fourier_basis = common.DM_basis_functions.construct_command_basis(
    basis="fourier",
    number_of_modes=40,
    Nx_act_DM=12,
    Nx_act_basis=12,
    act_offset=(0, 0),
    without_piston=True,
)

tip = fourier_basis[:, 0]
print("applying 2*tip cmd in Fourier basis to go off phase mask")
for b in dm_serial_numbers:
    dm[b].send_data(flatdm[b] + 1.8 * tip)

time.sleep(1)
N0_list = c.get_some_frames(
    number_of_frames=args.number_images_recorded_per_cmd, apply_manual_reduction=True
)
N0 = np.mean(N0_list, axis=0)

# ======== reference image with FPM IN
print("going back to DM flat to put beam ON phase mask")
for b in dm_serial_numbers:
    dm[b].send_data(flatdm[b])
time.sleep(2)
I0_list = c.get_some_frames(
    number_of_frames=args.number_images_recorded_per_cmd, apply_manual_reduction=True
)
I0 = np.mean(I0_list, axis=0)


# ====== make references fits files
I0_fits = fits.PrimaryHDU(I0_list)
N0_fits = fits.PrimaryHDU(N0_list)
DARK_fits = fits.PrimaryHDU(DARK_list)
I0_fits.header.set("EXTNAME", "FPM_IN")
N0_fits.header.set("EXTNAME", "FPM_OUT")
DARK_fits.header.set("EXTNAME", "DARK")

# flat_DM_fits = fits.PrimaryHDU( flat_dm_cmd )
# flat_DM_fits.header.set('EXTNAME','FLAT_DM_CMD')



########################################
## Now check Kolmogorov screen on DM

# maybe late we can make this user input 
D = 1.8
act_per_it = 0.5 # how many actuators does the screen pass per iteration 
V = 10 / act_per_it  / D #m/s (10 actuators across pupil on DM)
#scaling_factor = 0.05
I0_indicies = 10 # how many reference pupils do we get?

#scrn = aotools.infinitephasescreen.PhaseScreenVonKarman(nx_size= int( 12 / act_per_it ) , pixel_scale= zwfs_ns.grid.D / zwfs_ns.grid.N , r0=0.1, L0=12)
scrn = ps.PhaseScreenVonKarman(nx_size= int( 12 / act_per_it ) , pixel_scale= D / 12, r0=0.1, L0=12)
corner_indicies = [0, 11, 11 * 12, -1] # DM corner indidices


DM_command_sequence = [np.zeros(140) for _ in range(I0_indicies)]
for i in range(args.number_of_rolls):
    scrn.add_row()
    # bin phase screen onto DM space 
    dm_scrn = util.create_phase_screen_cmd_for_DM(scrn,  scaling_factor=args.scaling_factor, drop_indicies = [0, 11, 11 * 12, -1] , plot_cmd=False)
    # update DM command 
    #plt.figure(i)
    #plt.imshow(  util.get_DM_command_in_2D(dm_scrn) )
    #plt.colorbar()

    DM_command_sequence.append( dm_scrn )




# --- additional labels to append to fits file to keep information about the sequence applied
additional_header_labels = [
    ("cp_x1", roi[0]),
    ("cp_x2", roi[1]),
    ("cp_y1", roi[2]),
    ("cp_y2", roi[3]),
    ("number_of_rolls", args.number_of_rolls),
    ('I0_indicies','0-10'),
    ('act_per_it',act_per_it),
    ('D',D),
    ('V',V),
    ('scaling_factor', args.scaling_factor),
    ("Nact", 140)
]
# ("in-poke max amp", np.max(ramp_values)),
# ("out-poke max amp", np.min(ramp_values)),
# ("#ramp steps", number_amp_samples),
# ("seq0", "flatdm"),
# ("reshape", f"{number_amp_samples}-{modal_basis.shape[0]}-{modal_basis.shape[1]}"),

sleeptime_between_commands = 0.05
image_list = []
for cmd_indx, cmd in enumerate(DM_command_sequence):
    print(f"executing cmd_indx {cmd_indx} / {len(DM_command_sequence)}")
    # wait a sec
    time.sleep(sleeptime_between_commands)
    # ok, now apply command
    for b in dm:
        dm[b].send_data(flatdm[b] + cmd)

    # wait a sec
    time.sleep(sleeptime_between_commands)

    # get the image
    ims_tmp = [
        np.mean(
            c.get_some_frames(
                number_of_frames=args.number_images_recorded_per_cmd,
                apply_manual_reduction=True,
            ),
            axis=0,
        )
    ]  # [np.median([zwfs.get_image() for _ in range(args.number_images_recorded_per_cmd)] , axis=0)] #keep as list so it is the same type as when take_mean_of_images=False
    image_list.append(ims_tmp)


# init fits files if necessary
# should_we_record_images = True
take_mean_of_images = True
save_dm_cmds = True
save_fits = args.data_path + f"kolmogorov_calibration_{tstamp}.fits"
# save_file_name = data_path + f"stability_tests_{tstamp}.fits"
# if should_we_record_images:
# cmd2pix_registration
data = fits.HDUList([])  # init main fits file to append things to

# Camera data
cam_fits = fits.PrimaryHDU(image_list)

cam_fits.header.set("EXTNAME", "SEQUENCE_IMGS")

cam_config_dict = c.get_camera_config()
for k, v in cam_config_dict.items():
    cam_fits.header.set(k, v)

cam_fits.header.set("#images per DM command", args.number_images_recorded_per_cmd)
cam_fits.header.set("take_mean_of_images", take_mean_of_images)

# cam_fits.header.set('cropping_corners_r1', zwfs.pupil_crop_region[0] )
# cam_fits.header.set('cropping_corners_r2', zwfs.pupil_crop_region[1] )
# cam_fits.header.set('cropping_corners_c1', zwfs.pupil_crop_region[2] )
# cam_fits.header.set('cropping_corners_c2', zwfs.pupil_crop_region[3] )

# if user specifies additional headers using additional_header_labels
if additional_header_labels != None:
    if type(additional_header_labels) == list:
        for i, h in enumerate(additional_header_labels):
            cam_fits.header.set(h[0], h[1])
    else:
        cam_fits.header.set(additional_header_labels[0], additional_header_labels[1])


# if save_dm_cmds:
# put commands in fits format
dm_fits = fits.PrimaryHDU(DM_command_sequence)
# DM headers
dm_fits.header.set("timestamp", str(datetime.datetime.now()))
dm_fits.header.set("EXTNAME", "DM_CMD_SEQUENCE")
# dm_fits.header.set('DM', DM.... )
# dm_fits.header.set('#actuators', DM.... )


flat_DM_fits = fits.PrimaryHDU([flatdm[b] for b in dm])
flat_DM_fits.header.set("EXTNAME", "FLAT_DM_CMD")


# motor states 
motor_states = get_motor_states_as_list_of_dicts()
bintab_fits = save_motor_states_as_hdu( motor_states )

for b in dm:
    dm[b].send_data(flatdm[b])

# append to the data
data.append(cam_fits)
data.append(dm_fits)
data.append(flat_DM_fits)
data.append(I0_fits)
data.append(N0_fits)
data.append(DARK_fits)
data.append(bintab_fits )




if save_fits != None:
    if type(save_fits) == str:
        data.writeto(save_fits)
    else:
        raise TypeError(
            "save_images needs to be either None or a string indicating where to save file"
        )
    

data.close() 
for b in dm:
    dm[b].close_dm()

