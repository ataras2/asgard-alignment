import numpy as np
from astropy.io import fits
import os
import time
import matplotlib.pyplot as plt
import importlib
import json
import toml
import datetime
import sys
import pandas as pd
import argparse
import glob
import zmq
from scipy.optimize import leastsq
from scipy.ndimage import gaussian_filter, label, find_objects
import atexit

#from asgard_alignment import FLI_Cameras as FLI
import common.DM_basis_functions
import common.phasescreens as ps
import pyBaldr.utilities as util
from common import phasemask_centering_tool as pct

from xaosim.shmlib import shm
from asgard_alignment.DM_shm_ctrl import dmclass

try:
    from asgard_alignment import controllino as co
    myco = co.Controllino('172.16.8.200')
    controllino_available = True
    print('controllino connected')
    
except:
    print('WARNING Controllino cannot connect. WILL NOT MOVE SOURCE OUT FOR DARK')
    controllino_available = False 



#sys.path.insert(1, "/opt/Boston Micromachines/lib/Python3/site-packages/")
#import bmc

#import matplotlib 
#matplotlib.use('Agg') # helps avoid freezing in remote sessions
"""
pokes each actuator on the DMs over a +/- range of values and records images on the CRED ONE
default mode globalresetcds with setup taken from default_cred1_config.json
user can change fps and gain as desired, also the modal basis to ramp over
"""

# def close_all_dms():
#     try:
#         for b in dm:
#             dm[b].close_dm()
#         print("All DMs have been closed.")
#     except Exception as e:
#         print(f"dm object doesn't seem to exist, probably already closed")
# # Register the cleanup function to run at script exit
# atexit.register(close_all_dms)


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


def percentile_based_detect_pupils(
    image, percentile=80, min_group_size=50, buffer=20, plot=True
):
    """
    Detects circular pupils by identifying regions with grouped pixels above a given percentile.

    Parameters:
        image (2D array): Full grayscale image containing multiple pupils.
        percentile (float): Percentile of pixel intensities to set the threshold (default 80th).
        min_group_size (int): Minimum number of adjacent pixels required to consider a region.
        buffer (int): Extra pixels to add around the detected region for cropping.
        plot (bool): If True, displays the detected regions and coordinates.

    Returns:
        list of tuples: Cropping coordinates [(x_start, x_end, y_start, y_end), ...].
    """
    # Normalize the image
    image = image / image.max()

    # Calculate the intensity threshold as the 80th percentile
    threshold = np.percentile(image, percentile)

    # Create a binary mask where pixels are above the threshold
    binary_image = image > threshold

    # Label connected regions in the binary mask
    labeled_image, num_features = label(binary_image)

    # Extract regions and filter by size
    regions = find_objects(labeled_image)
    pupil_regions = []
    for region in regions:
        y_slice, x_slice = region
        # Count the number of pixels in the region
        num_pixels = np.sum(labeled_image[y_slice, x_slice] > 0)
        if num_pixels >= min_group_size:
            # Add a buffer around the region for cropping
            y_start = max(0, y_slice.start - buffer)
            y_end = min(image.shape[0], y_slice.stop + buffer)
            x_start = max(0, x_slice.start - buffer)
            x_end = min(image.shape[1], x_slice.stop + buffer)
            pupil_regions.append((x_start, x_end, y_start, y_end))

    if plot:
        # Plot the original image with bounding boxes
        plt.figure(figsize=(10, 10))
        plt.imshow(image, cmap="gray", origin="upper")
        for x_start, x_end, y_start, y_end in pupil_regions:
            rect = plt.Rectangle(
                (x_start, y_start),
                x_end - x_start,
                y_end - y_start,
                edgecolor="red",
                facecolor="none",
                linewidth=2,
            )
            plt.gca().add_patch(rect)
        plt.title(f"Detected Pupils: {len(pupil_regions)}")
        plt.show()

    return pupil_regions


def crop_and_sort_pupils(image, pupil_regions):
    """
    Crops regions corresponding to pupils from the image and sorts them
    by their center column index.

    Parameters:
        image (2D array): Full grayscale image.
        pupil_regions (list of tuples): Cropping coordinates [(x_start, x_end, y_start, y_end), ...].

    Returns:
        list of 2D arrays: List of cropped pupil images sorted by center column index.
    """
    # List to store cropped pupil images with their center column index
    cropped_pupils = []

    for region in pupil_regions:
        x_start, x_end, y_start, y_end = region
        # Crop the pupil region
        cropped_image = image[y_start:y_end, x_start:x_end]
        # Calculate the center column index of the region
        center_col = (x_start + x_end) // 2
        cropped_pupils.append((center_col, cropped_image))

    # Sort the cropped pupils by their center column index
    cropped_pupils.sort(key=lambda x: x[0])

    # Extract the sorted cropped images
    sorted_pupil_images = [pupil[1] for pupil in cropped_pupils]

    return sorted_pupil_images


def detect_circle(image, sigma=2, threshold=0.5, plot=True):
    """
    Detects a circular pupil in a cropped image using edge detection and circle fitting.

    Parameters:
        image (2D array): Cropped grayscale image containing a single pupil.
        sigma (float): Standard deviation for Gaussian smoothing.
        threshold (float): Threshold for binarizing edges.
        plot (bool): If True, displays the image with the detected circle overlay.

    Returns:
        tuple: (center_x, center_y, radius) of the detected circle.
    """
    # Normalize the image
    image = image / image.max()

    # Smooth the image to suppress noise
    smoothed_image = gaussian_filter(image, sigma=sigma)

    # Calculate gradients (Sobel-like edge detection)
    grad_x = np.gradient(smoothed_image, axis=1)
    grad_y = np.gradient(smoothed_image, axis=0)
    edges = np.sqrt(grad_x**2 + grad_y**2)

    # Threshold edges to create a binary mask
    binary_edges = edges > (threshold * edges.max())

    # Get edge pixel coordinates
    y, x = np.nonzero(binary_edges)

    # Initial guess for circle parameters
    def initial_guess(x, y):
        center_x, center_y = np.mean(x), np.mean(y)
        radius = np.sqrt(((x - center_x) ** 2 + (y - center_y) ** 2).mean())
        return center_x, center_y, radius

    # Circle model for optimization
    def circle_model(params, x, y):
        center_x, center_y, radius = params
        return np.sqrt((x - center_x) ** 2 + (y - center_y) ** 2) - radius

    # Perform least-squares circle fitting
    guess = initial_guess(x, y)
    result, _ = leastsq(circle_model, guess, args=(x, y))
    center_x, center_y, radius = result

    if plot:
        # Create a circular overlay for visualization
        overlay = np.zeros_like(image)
        yy, xx = np.ogrid[: image.shape[0], : image.shape[1]]
        circle_mask = (xx - center_x) ** 2 + (yy - center_y) ** 2 <= radius**2
        overlay[circle_mask] = 1

        # Plot the image and detected circle
        plt.figure(figsize=(6, 6))
        plt.imshow(image, cmap="gray", origin="upper")
        plt.contour(binary_edges, colors="cyan", linewidths=1, label="Edges")
        plt.contour(overlay, colors="red", linewidths=1, label="Detected Circle")
        plt.scatter(center_x, center_y, color="blue", marker="+", label="Center")
        plt.title("Detected Pupil with Circle Overlay")
        plt.legend()
        plt.show()

    return center_x, center_y, radius


# Example usage (assuming `image` is your 2D numpy array):
# center_x, center_y, radius = detect_circle(image)


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


# #################
# # Set up
# #################


# paths and timestamps
tstamp = datetime.datetime.now().strftime("%d-%m-%YT%H.%M.%S")
tstamp_rough =  datetime.datetime.now().strftime("%d-%m-%Y")

# # default data paths 
# with open( "config_files/file_paths.json") as f:
#     default_path_dict = json.load(f)

# baldr_pupils_path = default_path_dict["pupil_crop_toml"]  # "/home/asg/Progs/repos/asgard-alignment/config_files/baldr_pupils_coords.json"

# # with open(baldr_pupils_path, "r") as json_file:
# #     baldr_pupils = json.load(json_file)

# # Load the TOML file
# with open(baldr_pupils_path) as file:
#     pupildata = toml.load(file)

# # Extract the "baldr_pupils" section
# baldr_pupils = pupildata.get("baldr_pupils", {})

default_toml = os.path.join( "config_files", "baldr_config_#.toml") #os.path.dirname(os.path.abspath(__file__)), "..", "config_files", "baldr_config.toml")

# just open 2 - they should be all the same
with open(default_toml.replace('#','2'), "r") as f:
    config_dict = toml.load(f)

    # Baldr pupils from global frame 
    baldr_pupils = config_dict['baldr_pupils']



# positions to put thermal source on and take it out to empty position to get dark
# source_positions = {"SSS": {"empty": 80.0, "SBB": 65.5}}


# setting up socket to ZMQ communication to multi device server
parser = argparse.ArgumentParser(description="ZeroMQ Client and Mode setup")
parser.add_argument("--host", type=str, default="localhost", help="Server host")
parser.add_argument("--port", type=int, default=5555, help="Server port")
parser.add_argument(
    "--timeout", type=int, default=5000, help="Response timeout in milliseconds"
)

# parser.add_argument(
#     '--dm_config_path',
#     type=str,
#     default="/home/asg/Progs/repos/asgard-alignment/config_files/dm_serial_numbers.json",
#     help="Path to the DM configuration file. Default: %(default)s"
# )
parser.add_argument(
    "--global_camera_shm",
    type=str,
    default="/dev/shm/cred1.im.shm",
    help="Camera shared memory path. Default: /dev/shm/cred1.im.shm"
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
    default=f"/home/asg/Videos/{tstamp_rough}/",
    help="Path to the directory for storing pokeramp data. Default: %(default)s"
)

parser.add_argument(
    '--phasemask_name',
    type=str,
    default="H3",
    help="which phasemask? (J1-5 or H1-5). Default: %(default)s."
)

parser.add_argument(
    '--number_images_recorded_per_cmd',
    type=int,
    default=5,
    help="Number of images recorded per command (usually we take the average of these). Default: %(default)s."
)

parser.add_argument(
    '--number_amp_samples',
    type=int,
    default=18,
    help="Number of samples to take between DM amplitude limits. Default: %(default)s."
)

parser.add_argument(
    '--amp_max',
    type=int,
    default=0.1,
    help="maximum DM amplitude to apply. Units are normalized between 0-1. We ramp between +/- of this value. Default: %(default)s."
)

parser.add_argument(
    '--basis_name',
    type=str,
    default='Zonal',
    help="Name of the basis to use for DM operations. Default: %(default)s. Options include Zonal, Zonal_pinned_edges, Hadamard, Zernike, Zernike_pinned_edges, fourier, fourier_pinned_edges,"
)

parser.add_argument(
    '--number_of_modes',
    type=int,
    default=140,
    help="Number of modes to use. Default: %(default)s"
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

if abs(args.amp_max) > 0.5:
    raise UserWarning("--amp_max is too high ({args.amp_max})!! try around 0.1")

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
# roi = [None, None, None, None]
# c = FLI.fli(cameraIndex=0, roi=roi)
# # configure with default configuration file
# config_file_name = os.path.join(c.config_file_path, "default_cred1_config.json")
# c.configure_camera(config_file_name)

# with open(config_file_name, "r") as file:
#     camera_config = json.load(file)

# apply_manual_reduction = True

# c.send_fli_cmd("set mode globalresetcds")
# time.sleep(1)
# c.send_fli_cmd(f"set gain {args.cam_gain}")
# time.sleep(1)
# c.send_fli_cmd(f"set fps {args.cam_fps}")

# c.start_camera()


# Set up global camera frame SHM 
c = shm(args.global_camera_shm)

# set up DM SHMs 
dm = {}
for beam_id in [1,2,3,4]:
    dm[beam_id] = dmclass( beam_id=beam_id )
    # zero all channels
    dm[beam_id].zero_all()
    # activate flat (does this on channel 1)
    dm[beam_id].activate_flat()
    # apply dm flat offset (does this on channel 2)
    #dm_shm_dict[beam_id].set_data( np.array( dm_flat_offsets[beam_id] ) )



#bad_pixels = c.get_bad_pixel_indicies(  no_frames = 200, std_threshold = 10 , flatten=False)
#set_bad_pixels_to = int( c.send_fli_cmd("aduoffset") )
#c.build_bad_pixel_mask(  bad_pixels , set_bad_pixels_to = set_bad_pixels_to)

# frames = c.get_some_frames( number_of_frames=200, apply_manual_reduction=True )
# plt.figure() ; plt.imshow( np.std( frames, axis=0)) ;plt.colorbar() ; plt.savefig('delme.png')

# mean_frame = np.mean(frames, axis=0)
# std_frame = np.std(frames, axis=0)

# # Identify bad pixels
# global_mean = np.mean(mean_frame)
# global_std = np.std(mean_frame)
# bad_pixel_map = (np.abs(mean_frame - global_mean) > 5 * global_std) | (std_frame > 5 * np.median(std_frame))


# ########## set up DMs
# with open(args.dm_config_path, "r") as f:
#     dm_serial_numbers = json.load(f)

# dm = {}
# dm_err_flag = {}
# for beam, serial_number in dm_serial_numbers.items():
#     dm[beam] = bmc.BmcDm()  # init DM object
#     dm_err_flag[beam] = dm[beam].open_dm(serial_number)  # open DM
#     if not dm_err_flag:
#         print(f"Error initializing DM {beam}")


# flatdm = {}
# for beam, serial_number in dm_serial_numbers.items():
#     flatdm[beam] = pd.read_csv(
#         args.DMshapes_path + f"{serial_number}_FLAT_MAP_COMMANDS.csv",
#         header=None,
#     )[0].values

#modal_basis = np.eye(140)
modal_basis = common.DM_basis_functions.construct_command_basis(
    basis=args.basis_name,
    number_of_modes=args.number_of_modes,
    Nx_act_DM=12,
    Nx_act_basis=12,
    act_offset=(0, 0),
    without_piston=True,
).T # note transpose so modes are rows, cmds are columns

# for b in dm_serial_numbers:
#     dm[b].send_data(flatdm[f"{b}"])

# number_images_recorded_per_cmd = 5
# number_amp_samples = 18
# amp_max = 0.1
ramp_values = np.linspace(-args.amp_max, args.amp_max, args.number_amp_samples)


# phasemask



for beam in [1,2,3,4]:



    # ensuring using most recent file (sometimes MDS not up to date if not reset )
    # # get all available files 
    valid_reference_position_files = glob.glob(
        f"/home/asg/Progs/repos/asgard-alignment/config_files/phasemask_positions/beam{beam}/*json"
        )

    # read in the most recent and make initial posiition the most recent one for given mask 
    with open(max(valid_reference_position_files, key=os.path.getmtime)
    , "r") as file:
        start_position_dict = json.load(file)

        Xpos0 = start_position_dict[args.phasemask_name][0]
        Ypos0 = start_position_dict[args.phasemask_name][1]

    #message = f"fpm_movetomask phasemask{beam} {args.phasemask_name}"
    #res = send_and_get_response(message)
    #print(res)
    
        # check and manually move to best 
    message = f"moveabs BMX{beam} {Xpos0}"
    send_and_get_response(message)
    time.sleep(2)
    message = f"moveabs BMY{beam} {Ypos0}"
    send_and_get_response(message)
    time.sleep(2)



# beam = int( input( "do you want to check the phasemasks for a beam. Enter beam number (1,2,3,4) or 0 to continue") )

# while beam :
#     print( 'we save images as delme.png in asgard-alignment project - open it!')
#     img = np.sum( c.get_some_frames( number_of_frames=100, apply_manual_reduction=True ) , axis = 0 ) 
#     r1,r2,c1,c2 = baldr_pupils[str(beam)]
#     #print( r1,r2,c1,c2  )
#     plt.figure(); plt.imshow( np.log10( img[r1:r2,c1:c2] ) ) ; plt.colorbar(); plt.savefig('delme.png')

#     # time.sleep(5)

#     # manual centering 
#     pct.move_relative_and_get_image(cam=c, beam=beam, phasemask=state_dict["socket"], savefigName='delme.png', use_multideviceserver=True, roi=[r1,r2,c1,c2 ])

#     beam = int( input( "do you want to check the phasemask alignment for a particular beam. Enter beam number (1,2,3,4) or 0 to continue") )





########## ########## ##########
# ACTION
# ======== Source out first for dark

# state_dict["socket"].send_string(f"moveabs SSS {source_positions['SSS']['empty']}")
# res = socket.recv_string()
# print(f"Response: {res}")

# time.sleep(5)


# DARK_list = []
# DARK_list = c.get_data() # c.get_some_frames(number_of_frames=100, apply_manual_reduction=True)

# time.sleep(1)


# state_dict["socket"].send_string(f"moveabs SSS {source_positions['SSS']['SBB']}")
# res = socket.recv_string()
# print(f"Response: {res}")

# time.sleep(5)


# Get Darks 
if controllino_available:
    
    myco.turn_off("SBB")
    time.sleep(10)
    
    DARK_list = c.get_data()

    myco.turn_on("SBB")
    time.sleep(10)

    #bad_pixel_mask = get_bad_pixel_indicies( dark_raw, std_threshold = 20, mean_threshold=6)
else:
    DARK_list = c.get_data()

    #bad_pixel_mask = get_bad_pixel_indicies( dark_raw, std_threshold = 20, mean_threshold=6)



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
# for b in dm_serial_numbers:
#     dm[b].send_data(flatdm[b] + 1.8 * tip)

time.sleep(1)
N0_list = c.get_data()
# c.get_some_frames(
#     number_of_frames=args.number_images_recorded_per_cmd, apply_manual_reduction=True
# )
N0 = np.mean(N0_list, axis=0)

# ======== reference image with FPM IN
print("going back to DM flat to put beam ON phase mask")
for b in dm:
    dm[b].activate_flat() #.send_data(flatdm[b])
time.sleep(2)
I0_list = c.get_data()
#c.get_some_frames(
#    number_of_frames=args.number_images_recorded_per_cmd, apply_manual_reduction=True
#)
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


# ======== RAMPING ACTUATORS

# --- creating sequence of dm commands (we add the flat onto these dependingg on DM at the moment of sending command)
_DM_command_sequence = [list(amp * modal_basis) for amp in ramp_values]
# add in flat dm command at beginning of sequence and reshape so that cmd sequence is
# [0, a0*b0,.. aN*b0, a0*b1,...,aN*b1, ..., a0*bM,...,aN*bM]
DM_command_sequence = [np.zeros(140)] + list(
    np.array(_DM_command_sequence).reshape(
        args.number_amp_samples * modal_basis.shape[0], modal_basis.shape[1]
    )
)

# --- additional labels to append to fits file to keep information about the sequence applied
# ("cp_x1", roi[0]),
# ("cp_x2", roi[1]),
# ("cp_y1", roi[2]),
# ("cp_y2", roi[3]),
additional_header_labels = [
    ("in-poke max amp", np.max(ramp_values)),
    ("out-poke max amp", np.min(ramp_values)),
    ("#ramp steps", args.number_amp_samples),
    ("seq0", "flatdm"),
    ("reshape", f"{args.number_amp_samples}-{modal_basis.shape[0]}-{modal_basis.shape[1]}"),
    ("Nmodes_poked", len(modal_basis)),
    ("Nact", 140),
    ("basis_name",args.basis_name),
    ("number_of_modes", args.number_of_modes)
]

sleeptime_between_commands = 0.05
image_list = []
for cmd_indx, cmd in enumerate(DM_command_sequence):
    print(f"executing cmd_indx {cmd_indx} / {len(DM_command_sequence)}")
    # wait a sec
    time.sleep(sleeptime_between_commands)
    # ok, now apply command
    for b in dm:
        dm[b].set_data( dm[b].cmd_2_map2D(cmd ) )#send_data(flatdm[b] + cmd)

    # wait a sec
    time.sleep(sleeptime_between_commands)

    # c.get_some_frames(
    #     number_of_frames=args.number_images_recorded_per_cmd,
    #     apply_manual_reduction=True,
    # )
    # get the image
    ims_tmp = [
        np.mean(
            c.get_data(),
            axis=0,
        )
    ]  # [np.median([zwfs.get_image() for _ in range(args.number_images_recorded_per_cmd)] , axis=0)] #keep as list so it is the same type as when take_mean_of_images=False
    image_list.append(ims_tmp)


# init fits files if necessary
# should_we_record_images = True
take_mean_of_images = True
save_dm_cmds = True
save_fits = args.data_path + f"calibration_{args.basis_name}_{tstamp}.fits"
# save_file_name = data_path + f"stability_tests_{tstamp}.fits"
# if should_we_record_images:
# cmd2pix_registration
data = fits.HDUList([])  # init main fits file to append things to

# Camera data
cam_fits = fits.PrimaryHDU(image_list)

cam_fits.header.set("EXTNAME", "SEQUENCE_IMGS")

# cam_config_dict = c.get_camera_config()
# for k, v in cam_config_dict.items():
#     cam_fits.header.set(k, v)

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



for b in dm:
    dm[b].zero_all()
    dm[b].activate_flat() #send_data(flatdm[b])

flat_DM_fits = fits.PrimaryHDU([dm[b].shm0.get_data() for b in dm])
flat_DM_fits.header.set("EXTNAME", "FLAT_DM_CMD")

# motor states 
motor_states = get_motor_states_as_list_of_dicts()
bintab_fits = save_motor_states_as_hdu( motor_states )


# append to the data
data.append(cam_fits)
data.append(dm_fits)
data.append(flat_DM_fits)
data.append(I0_fits)
data.append(N0_fits)
data.append(DARK_fits)
data.append(bintab_fits)



if save_fits != None:
    if type(save_fits) == str:
        data.writeto(save_fits)
    else:
        raise TypeError(
            "save_images needs to be either None or a string indicating where to save file"
        )

##########
# close the data and DMs
data.close() 

# for b in dm:
#     dm[b].close_dm()

