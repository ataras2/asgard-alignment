
import zmq
import numpy as np
import toml  # Make sure to install via `pip install toml` if needed
import argparse
import os
import json
import time
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
from scipy.ndimage import median_filter

from xaosim.shmlib import shm
import asgard_alignment.controllino as co # for turning on / off source 
from asgard_alignment.DM_shm_ctrl import dmclass
import common.DM_registration as DM_registration

try:
    from asgard_alignment import controllino as co
    myco = co.Controllino('172.16.8.200')
    controllino_available = True
    print('controllino connected')
    
except:
    print('WARNING Controllino cannot connect. WILL NOT MOVE SOURCE OUT FOR DARK')
    controllino_available = False 
"""
idea it to be able to align phasemask position 
in a mode independent way with significant focus offsets
using image symmetry across registered pupil as objective 

TO DO : tweak zero point with clear ppupil quadrants . 
# check error sign 
"""


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




def split_into_quadrants(image, pupil_mask):
    """
    Split the image into four quadrants using the active pupil mask.

    Parameters:
        image (ndarray): Input image.
        pupil_mask (ndarray): Boolean array representing the active pupil.

    Returns:
        dict: Dictionary of quadrants (top-left, top-right, bottom-left, bottom-right).
    """
    y, x = np.indices(image.shape)
    cx, cy = np.mean(np.where(pupil_mask), axis=1).astype(int)

    # Create boolean masks for each quadrant
    top_left_mask = (y < cy) & (x < cx) & pupil_mask
    top_right_mask = (y < cy) & (x >= cx) & pupil_mask
    bottom_left_mask = (y >= cy) & (x < cx) & pupil_mask
    bottom_right_mask = (y >= cy) & (x >= cx) & pupil_mask

    quadrants = {
        "top_left": image[top_left_mask],
        "top_right": image[top_right_mask],
        "bottom_left": image[bottom_left_mask],
        "bottom_right": image[bottom_right_mask],
    }

    return quadrants

def weighted_photometric_difference(quadrants):
    """
    Calculate the weighted photometric difference between quadrants.

    Parameters:
        quadrants (dict): Dictionary of quadrants.

    Returns:
        tuple: (x_error, y_error) error vectors.
    """
    top = np.mean(quadrants["top_left"]) + np.sum(quadrants["top_right"])
    bottom = np.mean(quadrants["bottom_left"]) + np.sum(quadrants["bottom_right"])

    left = np.mean(quadrants["top_left"]) + np.sum(quadrants["bottom_left"])
    right = np.mean(quadrants["top_right"]) + np.sum(quadrants["bottom_right"])

    y_error = top - bottom
    x_error = left - right

    return x_error, y_error


def get_bad_pixel_indicies( imgs, std_threshold = 20, mean_threshold=6):
    # To get bad pixels we just take a bunch of images and look at pixel variance and mean

    ## Identify bad pixels
    mean_frame = np.mean(imgs, axis=0)
    std_frame = np.std(imgs, axis=0)

    global_mean = np.mean(mean_frame)
    global_std = np.std(mean_frame)
    bad_pixel_map = (np.abs(mean_frame - global_mean) > mean_threshold * global_std) | (std_frame > std_threshold * np.median(std_frame))

    return bad_pixel_map


def interpolate_bad_pixels(img, bad_pixel_map):
    filtered_image = img.copy()
    filtered_image[bad_pixel_map] = median_filter(img, size=3)[bad_pixel_map]
    return filtered_image



# def plot_telemetry(telemetry, savepath=None):
#     """
#     Plots the phasemask centering telemetry for each beam.
    
#     Parameters:
#         telemetry (dict): A dictionary where keys are beam IDs and values are dictionaries
#                           with keys:
#                               "phasmask_Xpos" - list of X positions,
#                               "phasmask_Ypos" - list of Y positions,
#                               "phasmask_Xerr" - list of X errors,
#                               "phasmask_Yerr" - list of Y errors.
#     """
#     for beam_id, data in telemetry.items():
#         # Determine the number of iterations
#         num_iterations = len(data["phasmask_Xpos"])
#         iterations = np.arange(1, num_iterations + 1)
        
#         # Create a figure with two subplots: one for positions and one for errors
#         fig, axs = plt.subplots(1, 2, figsize=(12, 5))
#         fig.suptitle(f"Telemetry for Beam {beam_id}", fontsize=14)
        
#         # Plot phasemask positions
#         axs[0].plot(iterations, data["phasmask_Xpos"], marker='o', label="X Position")
#         axs[0].plot(iterations, data["phasmask_Ypos"], marker='s', label="Y Position")
#         axs[0].set_xlabel("Iteration")
#         axs[0].set_ylabel("Position (um)")
#         axs[0].set_title("Phasemask Positions")
#         axs[0].legend()
#         axs[0].grid(True)
        
#         # Plot phasemask errors
#         axs[1].plot(iterations, data["phasmask_Xerr"], marker='o', label="X Error")
#         axs[1].plot(iterations, data["phasmask_Yerr"], marker='s', label="Y Error")
#         axs[1].set_xlabel("Iteration")
#         axs[1].set_ylabel("Error (um)")
#         axs[1].set_title("Phasemask Errors")
#         axs[1].legend()
#         axs[1].grid(True)
        
#         plt.tight_layout(rect=[0, 0, 1, 0.95])
#         if savepath is not None:
#             plt.savefig(savepath)
#         plt.show()


def plot_telemetry(telemetry,savepath='delme.png'):
    """
    For each beam, produce scatter plots of:
      - Phasemask positions: X vs. Y
      - Phasemask errors: X error vs. Y error

    Parameters:
        telemetry (dict): Dictionary with beam IDs as keys. Each beam's value is a
                          dictionary with keys:
                              "phasmask_Xpos" : list of X positions,
                              "phasmask_Ypos" : list of Y positions,
                              "phasmask_Xerr" : list of X errors,
                              "phasmask_Yerr" : list of Y errors.
    """
    for beam_id, data in telemetry.items():
        # Create a figure with two subplots side-by-side.
        fig, axs = plt.subplots(1, 2, figsize=(12, 5))
        fig.suptitle(f"Telemetry Scatter Plots for Beam {beam_id}", fontsize=14)
        
        # Scatter plot for phasemask positions.
        axs[0].scatter(data["phasmask_Xpos"], data["phasmask_Ypos"], 
                       color='blue', marker='o', s=50)
        axs[0].set_xlabel("Phasemask X Position")
        axs[0].set_ylabel("Phasemask Y Position")
        axs[0].set_title("Positions")
        axs[0].grid(True)
        
        # Scatter plot for phasemask errors.
        axs[1].scatter(data["phasmask_Xerr"], data["phasmask_Yerr"],
                       color='red', marker='s', s=50)
        axs[1].set_xlabel("Phasemask X Error")
        axs[1].set_ylabel("Phasemask Y Error")
        axs[1].set_title("Errors")
        axs[1].grid(True)
        
        plt.tight_layout(rect=[0, 0, 1, 0.92])
        if savepath is not None:
            plt.savefig(savepath)
        plt.show()


def image_slideshow(telemetry, beam_id):

    # Interactive plot with slider
    positions = [(x,y) for x,y in zip(telemetry[beam_id]["phasmask_Xpos"],telemetry[beam_id]["phasmask_Ypos"])]
    images = telemetry[beam_id]["img"]
    fig, ax = plt.subplots(1, 2, figsize=(12, 6))
    plt.subplots_adjust(bottom=0.2)

    # Initialize plots
    image_plot = ax[0].imshow(images[0], cmap='hot')
    ax[0].set_title("Image")
    position_plot, = ax[1].plot(positions[:, 0], positions[:, 1], marker='o', linestyle='-', color='blue', alpha=0.5)
    current_position, = ax[1].plot(positions[0, 0], positions[0, 1], marker='o', color='red')
    ax[1].set_xlim(positions[:, 0].min() - 5, positions[:, 0].max() + 5)
    ax[1].set_ylim(positions[:, 1].min() - 5, positions[:, 1].max() + 5)
    ax[1].set_title("Phasemask Center History")
    ax[1].set_xlabel("x position")
    ax[1].set_ylabel("y position")
    ax[1].grid()

    # Slider setup
    ax_slider = plt.axes([0.2, 0.05, 0.65, 0.03])
    slider = Slider(ax_slider, "Iteration", 0, len(images) - 1, valinit=0, valstep=1)

    # Update function for slider
    def update(val):
        idx = int(slider.val)
        image_plot.set_data(images[idx])
        current_position.set_data([positions[idx, 0]], [positions[idx, 1]])
        fig.canvas.draw_idle()

    slider.on_changed(update)
    plt.show()


parser = argparse.ArgumentParser(description="Baldr phase mask fine x-y adjustment")

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
default_toml = os.path.join("config_files", "baldr_config.toml") #os.path.dirname(os.path.abspath(__file__)), "..", "config_files", "baldr_config.toml")
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
    default=[2], #1, 2, 3, 4],
    help="Comma-separated list of beam IDs. Default: 1,2,3,4"
)

parser.add_argument(
    "--max_iterations",
    type=int,
    default=10,
    help="maximum number of iterations allowed in centering. Default = 10"
)

parser.add_argument(
    "--gain",
    type=int,
    default=0.1,
    help="gain to be applied for centering beam. Default = 0.1 "
)

parser.add_argument(
    "--tol",
    type=int,
    default=0.1,
    help="tolerence for convergence of centering algorithm. Default = 0.1 "
)

# Plot: default is True, with an option to disable.
parser.add_argument(
    "--plot", 
    dest="plot",
    action="store_true",
    default=True,
    help="Enable plotting (default: True)"
)


args = parser.parse_args()

# set up commands to move motors phasemask
context = zmq.Context()
context.socket(zmq.REQ)
socket = context.socket(zmq.REQ)
socket.setsockopt(zmq.RCVTIMEO, args.timeout)
server_address = f"tcp://{args.host}:{args.port}"
socket.connect(server_address)
state_dict = {"message_history": [], "socket": socket}

# phasemask specific commands
# message = f"fpm_movetomask phasemask{args.beam} {args.phasemask_name}"
# res = send_and_get_response(message)
# print(res)
phasemask_center = {}
for beam_id in args.beam_id:
    message = f"read BMX{beam_id}"
    Xpos = float( send_and_get_response(message) )

    message = f"read BMY{beam_id}"
    Ypos = float( send_and_get_response(message) )

    print(f'starting from current positiom X={Xpos}, Y={Ypos}um on beam {beam_id}')
    phasemask_center[beam_id] = [Xpos, Ypos]

# beam 2 initial pos
# In [56]: Xpos, Ypos
# Out[56]: (6054.994874999997, 3589.9963124999986)

# #example to move x-y of each beam's phasemask 
# for beam_id in args.beam_id:
#     message = f"moveabs BMX{beam_id} {Xpos}"
#     res = send_and_get_response(message)
#     print(res) 

#     message = f"moveabs BMY{beam_id} {Ypos}"
#     res = send_and_get_response(message)
#     print(res) 

# to manually adjust
# yi=20.0
# message = f"moverel BMY{beam_id} {yi}"
# res = send_and_get_response(message)
# print(res) 
# time.sleep(0.5)
# img = np.mean( c.get_data() ,axis=0) 
# for beam_id in args.beam_id:
#     r1,r2,c1,c2 = baldr_pupils[f"{beam_id}"]
#     cropped_img = interpolate_bad_pixels(img[r1:r2, c1:c2], bad_pixel_mask[r1:r2, c1:c2])
#     cropped_img *= 1/np.mean(cropped_img[pupil_masks[beam_id]])
#     clear_pupils[beam_id] = cropped_img
# plt.figure();plt.imshow(cropped_img);plt.savefig('delme1.png')

# set up commands to move DM 
assert hasattr(args.beam_id , "__len__")
assert len(args.beam_id) <= 4
assert max(args.beam_id) <= 4
assert min(args.beam_id) >= 1 

dm_shm_dict = {}
for beam_id in args.beam_id:
    dm_shm_dict[beam_id] = dmclass( beam_id=beam_id )
    # zero all channels
    dm_shm_dict[beam_id].zero_all()
    # activate flat 
    dm_shm_dict[beam_id].activate_flat()

# set up camera 
c = shm(args.global_camera_shm)

# set up subpupils and pixel mask
with open(args.toml_file ) as file:
    pupildata = toml.load(file)
    # Extract the "baldr_pupils" section
    baldr_pupils = pupildata.get("baldr_pupils", {})

    # the registered pupil mask for each beam (in the local frame)
    pupil_masks={}
    for beam_id in args.beam_id:
        pupil_masks[beam_id] = pupildata.get(f"beam{beam_id}", {}).get("pupil_mask", {}).get("mask", None) #pupildata.get(f"beam{beam_id}.pupil_mask.mask")
        if pupil_masks[beam_id] is None:
            raise UserWarning(f"pupil mask returned none in toml file. check for beam{beam_id}.pupil_mask.mask in the file:{args.toml_file}")


# dark and badpixel mask on global frame
if controllino_available:

    myco.turn_off("SBB")
    time.sleep(2)
    
    dark_raw = c.get_data()

    myco.turn_on("SBB")
    time.sleep(2)

    bad_pixel_mask = get_bad_pixel_indicies( dark_raw, std_threshold = 20, mean_threshold=6)
else:
    dark_raw = c.get_data()

    bad_pixel_mask = get_bad_pixel_indicies( dark_raw, std_threshold = 20, mean_threshold=6)


# get initial clear pupil
rel_offset = 500.0
clear_pupils = {}

message = f"moverel BMX{beam_id} {rel_offset}"
res = send_and_get_response(message)
print(res) 

message = f"moverel BMY{beam_id} {rel_offset}"
res = send_and_get_response(message)
print(res) 

time.sleep(1)

img = np.mean( c.get_data() ,axis=0) 
for beam_id in args.beam_id:
    r1,r2,c1,c2 = baldr_pupils[f"{beam_id}"]
    cropped_img = interpolate_bad_pixels(img[r1:r2, c1:c2], bad_pixel_mask[r1:r2, c1:c2])
    cropped_img *= 1/np.mean(cropped_img[pupil_masks[beam_id]])
    clear_pupils[beam_id] = cropped_img
plt.figure();plt.imshow(cropped_img);plt.savefig('delme1.png')

message = f"moverel BMX{beam_id} {-rel_offset}"
res = send_and_get_response(message)
print(res) 

message = f"moverel BMY{beam_id} {-rel_offset}"
res = send_and_get_response(message)
print(res) 

time.sleep(1)

# get initial image
img = np.mean( c.get_data() ,axis=0) #  full image 
initial_images = {}
for beam_id in args.beam_id:
    r1,r2,c1,c2 = baldr_pupils[f"{beam_id}"]
    cropped_img = interpolate_bad_pixels(img[r1:r2, c1:c2], bad_pixel_mask[r1:r2, c1:c2])
    initial_images[beam_id] = cropped_img
plt.figure();plt.imshow(initial_images[beam_id]);plt.savefig('delme1.png')


# begin centering algorithm, tracking telemetry
telemetry={b:{"img":[],"phasmask_Xpos":[],"phasmask_Ypos":[],"phasmask_Xerr":[], "phasmask_Yerr":[]} for b in args.beam_id }

complete_flag={b:False for b in args.beam_id}

for iteration in range(args.max_iterations):
    time.sleep(0.5)
    # get image 
    img = np.mean(c.get_data(), axis=0) # full image 

    for beam_id in args.beam_id:
        if not complete_flag[beam_id]:
            r1,r2,c1,c2 = baldr_pupils[f"{beam_id}"]
            cropped_img = interpolate_bad_pixels(img[r1:r2, c1:c2], bad_pixel_mask[r1:r2, c1:c2])
            # normalize by the mean within defined pupil mask
            cropped_img *= 1/np.mean(cropped_img[pupil_masks[beam_id]])

            # normed 
            normed_img = cropped_img / clear_pupils[beam_id] 
            # will need some clip and filtering 
        
            quadrants = split_into_quadrants(normed_img , pupil_masks[beam_id]) #cropped_img, pupil_masks[beam_id])
            #print( [len(v) for _,v in quadrants] )
            x_error, y_error = weighted_photometric_difference(quadrants)

            # Update phasemask center
            phasemask_center[beam_id][0] -= args.gain * x_error / np.sum(pupil_masks[beam_id])
            phasemask_center[beam_id][1] -= args.gain * y_error / np.sum(pupil_masks[beam_id])

            telemetry[beam_id]["img"].append( cropped_img )
            telemetry[beam_id]["phasmask_Xpos"].append( phasemask_center[beam_id][0] )
            telemetry[beam_id]["phasmask_Ypos"].append( phasemask_center[beam_id][1] )
            telemetry[beam_id]["phasmask_Xerr"].append( x_error )
            telemetry[beam_id]["phasmask_Yerr"].append( y_error )

            # Move 
            message = f"moveabs BMX{beam_id} {phasemask_center[beam_id][0]}"
            ok =  send_and_get_response(message) 
            print(ok)
            message = f"moveabs BMY{beam_id} {phasemask_center[beam_id][1]}"
            ok = send_and_get_response(message) 
            print(ok)
            
            # Check for convergence
            metric = np.sqrt(x_error**2 + y_error**2)
            if metric < args.tol:
                print(f"Beam {beam_id} converged in {iteration + 1} iterations.")
                complete_flag[beam_id] = True

            

            # taking slow for initial testing
            # if iteration > 0:
            #     #plot_telemetry(telemetry, savepath='delme.png')
            #     plt.figure();plt.imshow(cropped_img);plt.savefig('delme2.png')
            #     print("saving telemetry plot in rproject root delme.png to review.")
            #     input('continue?')

# some diagnostic plots  
plot_telemetry(telemetry, savepath='delme.png')

# get final image after convergence 
img = np.mean( c.get_data() ,axis=0) #  full image 
final_images = {}
for beam_id in args.beam_id:
    r1,r2,c1,c2 = baldr_pupils[f"{beam_id}"]
    cropped_img = interpolate_bad_pixels(img[r1:r2, c1:c2], bad_pixel_mask[r1:r2, c1:c2])
    final_images[beam_id] = cropped_img
plt.figure();plt.imshow(cropped_img);plt.savefig('delme2.png')


ii=0
plt.figure();plt.imshow( telemetry[beam_id]["img"][ii]);plt.savefig('delme.png')

# slideshow of images for a beam
#image_slideshow(telemetry, beam_id)