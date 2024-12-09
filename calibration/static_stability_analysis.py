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
                "BTX",
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


# ########################
# # To plot the results 

# import os
# import re
# from astropy.io import fits
# import numpy as np
# import matplotlib.pyplot as plt
# from datetime import datetime
# import glob

# import matplotlib 
# matplotlib.use('Agg')

# def get_matching_files(base_directory, subdirectories, pattern):
#     """
#     Searches for files matching a pattern in the specified subdirectories.

#     Parameters:
#     - base_directory (str): Path to the base directory.
#     - subdirectories (list of str): List of subdirectory names to search in.
#     - pattern (str): Filename pattern to match (e.g., "heim_bald_motorstates_*.fits").

#     Returns:
#     - list of str: List of matching file paths.
#     """
#     matching_files = []
#     for subdir in subdirectories:
#         search_path = os.path.join(base_directory, subdir, pattern)
#         matching_files.extend(glob.glob(search_path))
#     return matching_files


# def extract_timestamp_from_filename(filename):
#     """
#     Extracts a timestamp from a filename in the format heim_bald_motorstates_08-12-2024T18.58.09.fits.
#     """
#     pattern = r'(\d{2}-\d{2}-\d{4}T\d{2}\.\d{2}\.\d{2})'
#     match = re.search(pattern, filename)
#     if match:
#         return datetime.strptime(match.group(1), '%d-%m-%YT%H.%M.%S')
#     return None


# def read_motor_states_fits(fits_path):
#     """
#     Reads motor states from a FITS file.

#     Parameters:
#     - fits_path (str): Path to the FITS file.

#     Returns:
#     - list of dict: Motor states as a list of dictionaries.
#     """
#     with fits.open(fits_path) as hdul:
#         data = hdul["MotorStates"].data
#         motor_states = []
#         for row in data:
#             motor_states.append({
#                 "name": row["MotorName"],  # No need to decode
#                 "is_connected": row["IsConnected"],
#                 "position": row["Position"],
#             })
#     return motor_states

# def plot_motor_states_vs_time(fits_files):
#     """
#     Reads multiple FITS files and plots motor states vs time.

#     Parameters:
#     - fits_files (list of str): List of paths to FITS files.
#     """
#     timestamps = []
#     motor_positions = {}

#     for fits_file in fits_files:
#         timestamp = extract_timestamp_from_filename(os.path.basename(fits_file))
#         if not timestamp:
#             continue
#         timestamps.append(timestamp)
        
#         motor_states = read_motor_states_fits(fits_file)
#         for motor in motor_states:
#             name = motor["name"]
#             position = motor.get("position", np.nan)
#             if name not in motor_positions:
#                 motor_positions[name] = []
#             motor_positions[name].append(position)

#     # Ensure timestamps are sorted
#     timestamps, motor_positions = zip(*sorted(zip(timestamps, motor_positions.items())))
#     timestamps = np.array(timestamps)

#     # Plot each motor's position vs time
#     plt.figure(figsize=(12, 8))
#     for motor, positions in motor_positions.items():
#         plt.plot(timestamps, positions, label=motor)

#     plt.xlabel("Time")
#     plt.ylabel("Position")
#     plt.title("Motor States vs Time")
#     plt.legend()
#     plt.grid()

#     plt.savefig('delme.png')
#     plt.close()


# def plot_motor_states_vs_time(fits_files):
#     """
#     Reads multiple FITS files and plots motor states vs time.

#     Parameters:
#     - fits_files (list of str): List of paths to FITS files.
#     """
#     timestamps = []
#     motor_positions = {}

#     for fits_file in fits_files:
#         timestamp = extract_timestamp_from_filename(os.path.basename(fits_file))
#         if not timestamp:
#             continue
#         timestamps.append(timestamp)

#         motor_states = read_motor_states_fits(fits_file)
#         for motor in motor_states:
#             name = motor["name"]
#             position = motor.get("position", np.nan)
#             if name not in motor_positions:
#                 motor_positions[name] = []
#             motor_positions[name].append(position)

#     # Sort timestamps and ensure positions align
#     sorted_indices = np.argsort(timestamps)
#     timestamps = np.array(timestamps)[sorted_indices]
#     motor_positions = {motor: np.array(positions)[sorted_indices] for motor, positions in motor_positions.items()}

#     # Plot each motor's position vs time
#     plt.figure(figsize=(12, 8))
#     for motor, positions in motor_positions.items():
#         plt.plot(timestamps, positions, label=motor)

#     plt.xlabel("Time")
#     plt.ylabel("Position")
#     plt.title("Motor States vs Time")
#     plt.legend()
#     plt.grid()
#     plt.savefig('delme.png')
#     plt.close()


#     #plt.show()


# def plot_motor_states_subplots(fits_files, motor_names, motor_names_no_beams):
#     """
#     Reads multiple FITS files and plots motor states in subplots.

#     Parameters:
#     - fits_files (list of str): List of paths to FITS files.
#     - motor_names (list of str): List of motors without beam assignments.
#     - motor_names_no_beams (list of str): List of motor groups with beam assignments (e.g., "BMX", "BMY").
#     """
#     timestamps = []
#     motor_positions = {motor: [] for motor in motor_names}
#     motor_positions_with_beams = {
#         motor: {f"{motor}{beam}": [] for beam in range(1, 5)} for motor in motor_names_no_beams
#     }

#     for fits_file in fits_files:
#         timestamp = extract_timestamp_from_filename(os.path.basename(fits_file))
#         if not timestamp:
#             continue
#         timestamps.append(timestamp)

#         motor_states = read_motor_states_fits(fits_file)
#         for motor in motor_states:
#             name = motor["name"]
#             position = motor.get("position", np.nan)

#             # Check if it's a motor with no beams
#             if name in motor_positions:
#                 motor_positions[name].append(position)

#             # Check if it's a motor with beams
#             for group in motor_names_no_beams:
#                 if name.startswith(group):
#                     motor_positions_with_beams[group][name].append(position)

#     # Sort timestamps and align positions
#     sorted_indices = np.argsort(timestamps)
#     timestamps = np.array(timestamps)[sorted_indices]

#     motor_positions = {
#         motor: np.array(positions)[sorted_indices] for motor, positions in motor_positions.items()
#     }
#     for group in motor_names_no_beams:
#         for motor, positions in motor_positions_with_beams[group].items():
#             motor_positions_with_beams[group][motor] = np.array(positions)[sorted_indices]

#     # Create subplots
#     n_motors = len(motor_names) + len(motor_names_no_beams)
#     fig, axes = plt.subplots(n_motors, 1, figsize=(10, 5 * n_motors), sharex=True)

#     # Plot motors without beams
#     for i, motor in enumerate(motor_names):
#         ax = axes[i]
#         ax.plot(timestamps, motor_positions[motor], label=motor)
#         ax.set_title(f"Motor: {motor}")
#         ax.set_ylabel("Position")
#         ax.legend()
#         ax.grid()

#     # Plot motors with beams
#     for i, group in enumerate(motor_names_no_beams, start=len(motor_names)):
#         ax = axes[i]
#         for motor, positions in motor_positions_with_beams[group].items():
#             ax.plot(timestamps, positions, label=motor)
#         ax.set_title(f"Motor Group: {group}")
#         ax.set_ylabel("Position")
#         ax.legend()
#         ax.grid()

#     axes[-1].set_xlabel("Time")
#     plt.tight_layout()
#     #plt.show()
#     plt.savefig('delme.png')
#     plt.close() 

# # Define the base directory and subdirectories
# base_directory = "/home/heimdallr/data/stability_analysis/"
# subdirectories = ["06-12-2024", "07-12-2024", "08-12-2024"]
# pattern = "heim_bald_motorstates_*.fits"

# # Get the list of matching files
# matching_files = get_matching_files(base_directory, subdirectories, pattern)

# #plot_motor_states_vs_time(matching_files)


# Define motor names
motor_names = ["SDLA", "SDL12", "SDL34", "SSS", "BFO"]
motor_names_no_beams = [
    "HFO", "HTPP", "HTPI", "HTTP", "HTTI", "BDS", "BTT", "BTP", "BMX", "BMY"
]

# Plot motor states
plot_motor_states_subplots(matching_files, motor_names, motor_names_no_beams)
