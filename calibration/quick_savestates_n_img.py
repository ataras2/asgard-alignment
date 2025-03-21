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
import atexit 
from asgard_alignment import FLI_Cameras as FLI
import common.DM_basis_functions
import common.phasescreens as ps
import pyBaldr.utilities as util


def get_motor_states_as_list_of_dicts( ): 

    motor_names = []

    motor_names += ["SDLA", "SDL12", "SDL34", "SSS", "SSF"]
    
    motor_names_all_beams = [
            "HFO",
            "HTPP",
            "HTPI",
            "HTTP",
            "HTTI",
        ]

    for motor in motor_names_all_beams:
        for beam_number in range(1, 5):
            motor_names.append(f"{motor}{beam_number}")
    
    motor_names += ["BFO"]

    motor_names_all_beams = [
            "BDS",
            "BTT",
            "BTP",
            "BMX",
            "BMY",
            "BLF",
        ]

    partially_common_motors = [
            "BOTT",
            "BOTP",
        ]

    for motor in partially_common_motors:
        for beam_number in range(2, 5):
            motor_names.append(f"{motor}{beam_number}")

    for motor in motor_names_all_beams:
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
        print(res, type(res), is_connected)
        if is_connected: 
            if res != 'None':
                state["position"] = float(res)
            print()
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
    default=f"/home/asg/data/stability_analysis/{tstamp_rough}/",
    help="Path to the directory for storing pokeramp data. Default: %(default)s"
)


# parser.add_argument(
#     '--cam_fps',
#     type=int,
#     default=50,
#     help="frames per second on camera. Default: %(default)s"
# )
# parser.add_argument(
#     '--cam_gain',
#     type=int,
#     default=1,
#     help="camera gain. Default: %(default)s"
# )


args = parser.parse_args()

context = zmq.Context()

context.socket(zmq.REQ)

socket = context.socket(zmq.REQ)

socket.setsockopt(zmq.RCVTIMEO, args.timeout)

server_address = f"tcp://{args.host}:{args.port}"

socket.connect(server_address)

state_dict = {"message_history": [], "socket": socket}



#DMshapes_path = args.DMshapes_path #"/home/asg/Progs/repos/asgard-alignment/DMShapes/"
#dm_config_path = #"/home/asg/Progs/repos/asgard-alignment/config_files/dm_serial_numbers.json"
# data_path = f"/home/heimdallr/data/pokeramp/{tstamp_rough}/"
if not os.path.exists(args.data_path):
     os.makedirs(args.data_path)




########## ########## ##########
########## set up camera object
roi = [None, None, None, None]
c = FLI.fli( roi=roi )

# we get frames in whatever state the user runs the camera in 

# get motor states
motor_states = get_motor_states_as_list_of_dicts()
bintab_fits = save_motor_states_as_hdu(motor_states)

frames = c.get_some_frames(number_of_frames=20, apply_manual_reduction=False)

primary_hdu = fits.PrimaryHDU(data=frames)
primary_hdu.header["EXTNAME"] = "FRAMES"

hdulist = fits.HDUList([primary_hdu, bintab_fits])
hdulist.writeto(args.data_path + f'imgs_n_all_motorstates_{tstamp}.fits', overwrite=True)

c.close()




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

# # Define motor names
# motor_names = ["SDLA", "SDL12", "SDL34", "SSS", "BFO"]
# motor_names_no_beams = [
#     "HFO", "HTPP", "HTPI", "HTTP", "HTTI", "BDS", "BTT", "BTP", "BMX", "BMY"
# ]

# # Plot motor states
# plot_motor_states_subplots(matching_files, motor_names, motor_names_no_beams)
