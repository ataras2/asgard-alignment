import numpy as np
from astropy.io import fits
import os
import time
import matplotlib.pyplot as plt
import importlib
import json
import datetime
import sys
import pandas as pd
import argparse
import zmq
from scipy.optimize import leastsq
from scipy.ndimage import gaussian_filter, label, find_objects

from asgard_alignment import FLI_Cameras as FLI
import common.DM_basis_functions
import common.phasescreens as ps
import pyBaldr.utilities as util

sys.path.insert(1, "/opt/Boston Micromachines/lib/Python3/site-packages/")
import bmc



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
        message = f"!read {name}"
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


# positions to put thermal source on and take it out to empty position to get dark
source_positions = {"SSS": {"empty": 80.0, "SBB": 65.5}}


# setting up socket to ZMQ communication to multi device server
parser = argparse.ArgumentParser(description="ZeroMQ Client and Mode setup")
parser.add_argument("--host", type=str, default="localhost", help="Server host")
parser.add_argument("--port", type=int, default=5555, help="Server port")
parser.add_argument(
    "--timeout", type=int, default=5000, help="Response timeout in milliseconds"
)

parser.add_argument(
    '--dm_config_path',
    type=str,
    default="/home/heimdallr/Documents/asgard-alignment/config_files/dm_serial_numbers.json",
    help="Path to the DM configuration file. Default: %(default)s"
)
parser.add_argument(
    '--DMshapes_path',
    type=str,
    default="/home/heimdallr/Documents/asgard-alignment/DMShapes/",
    help="Path to the directory containing DM shapes. Default: %(default)s"
)

parser.add_argument(
    '--dm_map',
    type=str,
    default="flat",
    help="What pattern to put on DM (options are flat or cross). Default: %(default)s"
)

parser.add_argument(
    '--data_path',
    type=str,
    default=f"/home/heimdallr/data/stability_analysis/{tstamp_rough}/",
    help="Path to the directory for storing pokeramp data. Default: %(default)s"
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


args = parser.parse_args()

context = zmq.Context()

context.socket(zmq.REQ)

socket = context.socket(zmq.REQ)

socket.setsockopt(zmq.RCVTIMEO, args.timeout)

server_address = f"tcp://{args.host}:{args.port}"

socket.connect(server_address)

state_dict = {"message_history": [], "socket": socket}



#DMshapes_path = args.DMshapes_path #"/home/heimdallr/Documents/asgard-alignment/DMShapes/"
#dm_config_path = #"/home/heimdallr/Documents/asgard-alignment/config_files/dm_serial_numbers.json"
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



########## ########## ##########
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


crossdm = {}
for beam, serial_number in dm_serial_numbers.items():
    crossdm[beam] = pd.read_csv(
        args.DMshapes_path + f"Crosshair140.csv",
        header=None,
    )[0].values

# modal basis we might want to use in future
# modal_basis = common.DM_basis_functions.construct_command_basis(
#     basis=args.basis_name,
#     number_of_modes=args.number_of_modes,
#     Nx_act_DM=12,
#     Nx_act_basis=12,
#     act_offset=(0, 0),
#     without_piston=True,
# ).T # note transpose so modes are rows, cmds are columns

if args.dm_map == "flat":
    for b in dm_serial_numbers:
        dm[b].send_data(flatdm[f"{b}"])

elif args.dm_map == "cross":
    for b in dm_serial_numbers:
        dm[b].send_data(flatdm[f"{b}"] + 0.25 * crossdm[f"{b}"])

########## ########## ##########
########## set up darks
print('moving thermal source out to take dark..')
state_dict["socket"].send_string(f"!moveabs SSS {source_positions['SSS']['empty']}")
res = socket.recv_string()
print(f"Response: {res}")

time.sleep(5)

print('taking dark')
DARK_list = []
DARK_list = c.get_some_frames(number_of_frames=100, apply_manual_reduction=True)

time.sleep(5)

print('moving thermal source back in')
state_dict["socket"].send_string(f"!moveabs SSS {source_positions['SSS']['SBB']}")
res = socket.recv_string()
print(f"Response: {res}")

time.sleep(20)

# get motor states
motor_states = get_motor_states_as_list_of_dicts()
bintab_fits = save_motor_states_as_hdu(motor_states)

c.save_fits( args.data_path + f'heim_bald_pupils_fps-{args.cam_fps}_gain-{args.cam_gain}_dm-{args.dm_map}_{tstamp}.fits' ,  number_of_frames=200, apply_manual_reduction=True )

bintab_fits.writeto(args.data_path + f'heim_bald_motorstates_{tstamp}.fits' )

# close DMs
print('closing DMs')
for b in dm:
    dm[b].close_dm()

time.sleep(3)
print('Done')
##