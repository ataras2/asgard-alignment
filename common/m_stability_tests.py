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

from asgard_alignment import FLI_Cameras as FLI
import common.DM_basis_functions
import common.phasescreens as ps
import pyBaldr.utilities as util

sys.path.insert(1, "/opt/Boston Micromachines/lib/Python3/site-packages/")
import bmc

"""

REDUNDANT - THIS SCRIPT IS NOW SPLIT TO SEPERATE SCRIPTS (02/12/24)
/home/asg/Progs/repos/asgard-alignment/calibration/pokeramps.py
/home/asg/Progs/repos/asgard-alignment/calibration/kolmogorov_phasescreen_on_dm.py


Nov 24 - we notice significant drifts likely coming from OAP 1 and solarstein down periscope. 
We have all four Baldr beams on the detector. This script is to run the camera and some DM 
commands to monitor 
(a) the drift of the beams on the detector
(b) the registration of the DM ono the detector 
(c) drift of the beams on the DM 
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import leastsq
from scipy.ndimage import gaussian_filter
from scipy.signal import find_peaks

import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter, label, find_objects

from scipy.ndimage import label, find_objects


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

DMshapes_path = "/home/asg/Progs/repos/asgard-alignment/DMShapes/"

data_path = f"/home/heimdallr/data/stability_tests/"
if not os.path.exists(data_path):
    os.makedirs(data_path)


# positions to put thermal source on and take it out to empty position to get dark
source_positions = {"SSS": {"empty": 80.0, "SBB": 65.5}}

# setting up socket to ZMQ communication to multi device server
parser = argparse.ArgumentParser(description="ZeroMQ Client")
parser.add_argument("--host", type=str, default="localhost", help="Server host")
parser.add_argument("--port", type=int, default=5555, help="Server port")
parser.add_argument(
    "--timeout", type=int, default=5000, help="Response timeout in milliseconds"
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
c.send_fli_cmd("set gain 5")
time.sleep(1)
c.send_fli_cmd("set fps 50")

c.start_camera()


########## set up DMs
with open("config_files/dm_serial_numbers.json", "r") as f:
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
        DMshapes_path + f"{serial_number}_FLAT_MAP_COMMANDS.csv",
        header=None,
    )[0].values

modal_basis = np.eye(140)

for b in dm_serial_numbers:
    dm[b].send_data(flatdm[f"{b}"])

number_images_recorded_per_cmd = 5
number_amp_samples = 18
amp_max = 0.1
ramp_values = np.linspace(-amp_max, amp_max, number_amp_samples)


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
    number_of_frames=number_images_recorded_per_cmd, apply_manual_reduction=True
)
N0 = np.mean(N0_list, axis=0)

# ======== reference image with FPM IN
print("going back to DM flat to put beam ON phase mask")
for b in dm_serial_numbers:
    dm[b].send_data(flatdm[b])
time.sleep(2)
I0_list = c.get_some_frames(
    number_of_frames=number_images_recorded_per_cmd, apply_manual_reduction=True
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


# ======== RAMPING ACTUATORS

# --- creating sequence of dm commands (we add the flat onto these dependingg on DM at the moment of sending command)
_DM_command_sequence = [list(amp * modal_basis) for amp in ramp_values]
# add in flat dm command at beginning of sequence and reshape so that cmd sequence is
# [0, a0*b0,.. aN*b0, a0*b1,...,aN*b1, ..., a0*bM,...,aN*bM]
DM_command_sequence = [np.zeros(140)] + list(
    np.array(_DM_command_sequence).reshape(
        number_amp_samples * modal_basis.shape[0], modal_basis.shape[1]
    )
)

# --- additional labels to append to fits file to keep information about the sequence applied
additional_header_labels = [
    ("cp_x1", roi[0]),
    ("cp_x2", roi[1]),
    ("cp_y1", roi[2]),
    ("cp_y2", roi[3]),
    ("in-poke max amp", np.max(ramp_values)),
    ("out-poke max amp", np.min(ramp_values)),
    ("#ramp steps", number_amp_samples),
    ("seq0", "flatdm"),
    ("reshape", f"{number_amp_samples}-{modal_basis.shape[0]}-{modal_basis.shape[1]}"),
    ("Nmodes_poked", len(modal_basis)),
    ("Nact", 140),
]

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
                number_of_frames=number_images_recorded_per_cmd,
                apply_manual_reduction=True,
            ),
            axis=0,
        )
    ]  # [np.median([zwfs.get_image() for _ in range(number_images_recorded_per_cmd)] , axis=0)] #keep as list so it is the same type as when take_mean_of_images=False
    image_list.append(ims_tmp)


# init fits files if necessary
# should_we_record_images = True
take_mean_of_images = True
save_dm_cmds = True
save_fits = data_path + f"calibration_{tstamp}.fits"
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

cam_fits.header.set("#images per DM command", number_images_recorded_per_cmd)
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


for b in dm:
    dm[b].send_data(flatdm[b])

# append to the data
data.append(cam_fits)
data.append(dm_fits)
data.append(flat_DM_fits)
data.append(I0_fits)
data.append(N0_fits)
data.append(DARK_fits)


if save_fits != None:
    if type(save_fits) == str:
        data.writeto(save_fits)
    else:
        raise TypeError(
            "save_images needs to be either None or a string indicating where to save file"
        )


##########
# close the data 
data.close() 




########################################
## Now check Kolmogorov screen on DM

D = 1.8
act_per_it = 0.5 # how many actuators does the screen pass per iteration 
V = 10 / act_per_it  / D #m/s (10 actuators across pupil on DM)
scaling_factor = 0.05
I0_indicies = 10 # how many reference pupils do we get?

#scrn = aotools.infinitephasescreen.PhaseScreenVonKarman(nx_size= int( 12 / act_per_it ) , pixel_scale= zwfs_ns.grid.D / zwfs_ns.grid.N , r0=0.1, L0=12)
scrn = ps.PhaseScreenVonKarman(nx_size= int( 12 / act_per_it ) , pixel_scale= D / 12, r0=0.1, L0=12)
corner_indicies = [0, 11, 11 * 12, -1] # DM corner indidices


DM_command_sequence = [np.zeros(140) for _ in range(I0_indicies)]
for i in range(1000):
    scrn.add_row()
    # bin phase screen onto DM space 
    dm_scrn = util.create_phase_screen_cmd_for_DM(scrn,  scaling_factor=scaling_factor, drop_indicies = [0, 11, 11 * 12, -1] , plot_cmd=False)
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
    ('I0_indicies','0-10'),
    ('act_per_it',act_per_it),
    ('D',D),
    ('V',V),
    ('scaling_factor', scaling_factor),
    ("Nmodes_poked", len(modal_basis)),
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
                number_of_frames=number_images_recorded_per_cmd,
                apply_manual_reduction=True,
            ),
            axis=0,
        )
    ]  # [np.median([zwfs.get_image() for _ in range(number_images_recorded_per_cmd)] , axis=0)] #keep as list so it is the same type as when take_mean_of_images=False
    image_list.append(ims_tmp)


# init fits files if necessary
# should_we_record_images = True
take_mean_of_images = True
save_dm_cmds = True
save_fits = data_path + f"kolmogorov_calibration_{tstamp}.fits"
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

cam_fits.header.set("#images per DM command", number_images_recorded_per_cmd)
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


for b in dm:
    dm[b].send_data(flatdm[b])

# append to the data
data.append(cam_fits)
data.append(dm_fits)
data.append(flat_DM_fits)
data.append(I0_fits)
data.append(N0_fits)
data.append(DARK_fits)


if save_fits != None:
    if type(save_fits) == str:
        data.writeto(save_fits)
    else:
        raise TypeError(
            "save_images needs to be either None or a string indicating where to save file"
        )
    

for b in dm:
    dm[b].close_dm()

