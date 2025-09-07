
"""
script to scan generally a set of two tip/tilt mirrors that define 
an image or pupil plane, keeping the other one fixed.
e.g. move a pupil without moving the respective image plane
this is important for alignment through the coldstop on the CRED1! 
"""
import zmq
import phasemask_centering_tool as pct
import time
import toml
import argparse
import os 
import datetime
import json 
import numpy as np 
from scipy.interpolate import griddata
import matplotlib.pyplot as plt 
from scipy.optimize import leastsq
from scipy.ndimage import label, find_objects
from xaosim.shmlib import shm

from pyBaldr import utilities as util
from asgard_alignment import FLI_Cameras as FLI
import m_process_scan 

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


def expand_roi( roi0, expand_by_percent=50 ):
    X = roi0[1] - roi0[0]
    extra =expand_by_percent/100 * X 
    roi = []
    for i,rr in enumerate( roi0 ):
        if np.mod(i,2)==0:
            roi.append( int( rr - extra ) )
        else:
            roi.append( int(rr + extra ) )
    return roi 

# paths and timestamps
tstamp = datetime.datetime.now().strftime("%d-%m-%YT%H.%M.%S")
tstamp_rough =  datetime.datetime.now().strftime("%d-%m-%Y")

# setting up socket to ZMQ communication to multi device server
parser = argparse.ArgumentParser(description="ZeroMQ Client and Mode setup")
parser.add_argument("--host", type=str, default="192.168.100.2", help="Server host")
parser.add_argument("--port", type=int, default=5555, help="Server port")
parser.add_argument(
    "--timeout", type=int, default=5000, help="Response timeout in milliseconds"
)

parser.add_argument('--non_verbose',
                action='store_false', 
                help="disable verbose mode (default). This asks before doing things and allows fine adjustments")

parser.add_argument(
    '--data_path',
    type=str,
    default=f"/home/asg/Progs/repos/asgard-alignment/calibration/reports/{tstamp_rough}/", #f"/home/heimdallr/data/phasemask_aquisition/{tstamp_rough}/",
    help="Path to the directory for storing saved data. Default: %(default)s"
)

parser.add_argument(
    '--system',
    type=str,
    default="baldr_cred1",
    help="heim_cred1, heim_intermediate_focus, baldr_cred1. Default: %(default)s. Options: baldr, heimdallr"
)

parser.add_argument(
    '--move_plane',
    type=str,
    default="pupil",
    help="what plane to move in the system (pupil or image). Default: %(default)s"
)

parser.add_argument(
    '--beam',
    type=str,
    default="3",
    help="what beam to look at?. Default: %(default)s"
)

parser.add_argument(
    "--global_camera_shm",
    type=str,
    default="/dev/shm/cred1.im.shm",
    help="Camera shared memory path. Default: /dev/shm/cred1.im.shm"
)


parser.add_argument(
    '--initial_pos',
    type=str,
    default="current",
    help="x,y initial position of search or 'current' to use most recent calibration file. Default: %(default)s "
)

parser.add_argument(
    '--search_radius',
    type=float,
    default=10,
    help="search radius of spiral search in microns. Default: %(default)s"
)
parser.add_argument(
    '--dx',
    type=float,
    default=2,
    help="step size in motor units during scan. Default: %(default)s"
)

parser.add_argument(
    '--roi',
    type=str,
    default=None, #"[188, 252, 141, 205]",#
    help="region to crop in camera (row1, row2, col1, col2). Default:%(default)s"
)

parser.add_argument(
    "--scantype",
    type=str,
    default="square_spiral",
    help="waht type of scan to do? Default:%(default)s"
)

parser.add_argument(
    "--sleeptime",
    type=float,
    default=0.4,
    help="sleep time between movements? Default:%(default)s"
)

parser.add_argument(
    "--record_images",
    type=int,
    default=1,
    help="Do we want to include images? 1=>yes, 0=>no Default:%(default)s"
)


args = parser.parse_args()

# create save path if doesnt exist 
if not os.path.exists(args.data_path):
     print(f'made directory : {args.data_path}')
     os.makedirs(args.data_path)


# setup zmq to tal to motor
context = zmq.Context()
context.socket(zmq.REQ)
socket = context.socket(zmq.REQ)
socket.setsockopt(zmq.RCVTIMEO, args.timeout)
server_address = f"tcp://{args.host}:{args.port}"
socket.connect(server_address)
state_dict = {"message_history": [], "socket": socket}

if int(args.record_images):
    if args.roi is None:
        default_toml = os.path.join("/usr/local/etc/baldr/", "baldr_config_#.toml") 

        with open(default_toml.replace('#',f'{args.beam}'), "r") as f:
            config_dict = toml.load(f)
            # Baldr pupils from global frame 
            baldr_pupils = config_dict['baldr_pupils']

        roi0 = baldr_pupils[f'{args.beam}']

        roi = expand_roi( roi0, expand_by_percent=50 )

        c = FLI.fli(roi = roi)
    else:
        # set up camera 
        c = FLI.fli( roi=eval(args.roi) ) #shm(args.global_camera_shm)


# try get a dark and build bad pixel map 
try:
    print("comment out dark - this should now be done by server!")
    #c.build_manual_dark()
except Exception as e:
    print('failed to take dark with exception {e}')



# generate the scan points 
# starting point is always relative to current position so hard coded to 0,0
if args.scantype == "square_spiral":
    scan_pattern = pct.square_spiral_scan(starting_point=[0, 0], step_size= float(args.dx), search_radius = float(args.search_radius))
elif args.scantype == "raster":
    scan_pattern = pct.raster_scan_with_orientation(starting_point=[0, 0], dx=float(args.dx), dy=float(args.dx), width=float(args.search_radius), height=float(args.search_radius), orientation=0)
elif args.scantype == "cross": # tested in software not hardware (20/6/25)
    scan_pattern = pct.cross_scan(starting_point=[0, 0], dx=float(args.dx), dy=float(args.dx), width=float(2 * args.search_radius), height=float(2* args.search_radius), angle=0)
else:
    raise UserWarning("invalud scan. Try square spiral,raster")   

# zip them
x_points, y_points = zip(*scan_pattern)



# # we should have predifed json file for these..
# if args.motor == 'BTX':
#     safety_limits = {"xmin":-0.6,"xmax":0.6, "ymin":-0.6,"ymax":0.6}
# else:
#     safety_limits = {"xmin":-np.inf,"xmax":np.inf, "ymin":-np.inf,"ymax":np.inf}


# convert to relative offsets
rel_x_points = np.array(list(x_points)[1:]) - np.array(
    list(x_points)[:-1]
)
rel_y_points = np.array(list(y_points)[1:]) - np.array(
    list(y_points)[:-1]
)


if "image" in args.move_plane:
    print( "reminder: image plane units are typically mm.")
elif "pupil" in args.move_plane:
    print( "reminder: pupil plane units are typically pixels of the respective camera.")

# we chose more user friendly names for the system,
# but need convert the names to MDS convention : 
["c_red_one_focus", "intermediate_focus", "baldr"]
if args.system == "heim_cred1":
    config = "c_red_one_focus"
elif args.system == "heim_intermediate_focus":
    config = "intermediate_focus"
elif args.system == "baldr_cred1":
    config = "baldr"


# No Botx on beam one 

if (config == "baldr") and (int(args.beam) == 1):
    raise UserWarning(
        "warning no BOTX motor on beam 1 - so move pupil / image for baldr on beam 1 is invalid"
    )


# get original positions before scan 
if "baldr" in config:
    axes = [f"BTP{args.beam}", f"BTT{args.beam}", f"BOTP{args.beam}", f"BOTT{args.beam}"]
else:
    axes = [f"HTPP{args.beam}", f"HTTP{args.beam}", f"HTPI{args.beam}", f"HTTI{args.beam}"]

pos_dict_original = {}
for axis in axes:
    pos = send_and_get_response(f"read {axis}")
    pos_dict_original[axis] = pos

# now start 
#############

# init dicitionaries 
if args.record_images:
    img_dict = {}
    motor_pos_dict = {}

# try get a dark
# try:
#     c.build_manual_dark()
# except Exception as e:
#     st.write(f"failed to take dark with exception {e}")



for it, (delx, dely) in enumerate(zip(rel_x_points, rel_y_points)):
    #progress_bar.progress(it / len(rel_x_points))

    if "image" in args.move_plane:
        # asgard_alignment.Engineering.move_image(
        #     beam, delx, dely, send_and_get_response, config
        # )
        cmd = f"mv_img {config} {args.beam} {delx} {dely}"
        send_and_get_response(cmd)

    elif "pupil" in args.move_plane:
        # asgard_alignment.Engineering.move_pupil(
        #     beam, delx, dely, send_and_get_response, config
        # )
        cmd = f"mv_pup {config} {args.beam} {delx} {dely}"
        send_and_get_response(cmd)

    
    time.sleep(args.sleeptime)

    # get all the motor positions
    pos_dict = {}
    for axis in axes:
        pos = send_and_get_response(f"read {axis}")
        # st.write(f"{axis}: {pos}")
        pos_dict[axis] = pos

    if args.record_images:
        motor_pos_dict[str((x_points[it], y_points[it]))] = pos_dict

        # get the images
        # index dictionary by absolute position of the scan
        imgtmp = np.mean(c.get_data(apply_manual_reduction=True), axis=0)
        img_dict[str((x_points[it], y_points[it]))] = imgtmp

# move back to original position
print(f"moving back to original position: {pos_dict_original}")

for axis, pos in pos_dict_original.items():
    msg = f"moveabs {axis} {pos}"
    send_and_get_response(msg)
    print(f"Moving {axis} back to {pos}")

if args.record_images:
    # save
    img_json_file_path = args.data_path + f"img_dict_beam{args.beam}-{args.move_plane}.json"
    with open(img_json_file_path, "w") as json_file:
        json.dump(util.convert_to_serializable(img_dict), json_file)

    print(f"wrote {img_json_file_path}")


    motorpos_json_file_path = (
        args.data_path + f"motorpos_dict_beam{args.beam}-{args.move_plane}.json"
    )
    with open(motorpos_json_file_path, "w") as json_file:
        json.dump(util.convert_to_serializable(motor_pos_dict), json_file)

    print(f"wrote {motorpos_json_file_path}")


### read it back in 
# look at pct aggrate functions 
kwargs = {}
processed_imgs = m_process_scan.process_scan( scan_data=img_dict , method='frame_aggregate', kwargs = kwargs)

means = np.array( list( v["mean"] for v in processed_imgs.values() ) )



best_pos = list( motor_pos_dict.values() )[ np.argmax( means )  ]
print(f"best position at {best_pos}")

for axis, pos in best_pos.items():
    msg = f"moveabs {axis} {pos}"
    send_and_get_response(msg)
    print(f"Moving {axis} to best found pos: {pos}")
    

# ##=======================================
# # update phasemask 
# print("now updating phasemask positions based on BOTX offsets")
# # matrix to update phasemask positions relative to BOTX offsets 
# phasemask_matrix = asgard_alignment.Engineering.phasemask_botx_matricies

# #best_pos.items() # e.g. {"BOTX1":0.2}
# #pos_dict_original.items()

# # get difference from new (best) positions and the original starting position
# delta_BOTP = float(best_pos[f"BOTP{args.beam}"]) - float(pos_dict_original[f"BOTP{args.beam}"])
# delta_BOTT = float(best_pos[f"BOTT{args.beam}"]) - float(pos_dict_original[f"BOTT{args.beam}"])

# # convert to offsets in phasemask BMX and BMY
# delta_BMX, delta_BMY = phasemask_matrix[int(beam)] @ [
#         delta_BOTP,
#         delta_BOTT,
#     ]

# # move phasemasks

# # Y
# msg = f"moverel BMY{beam} {delta_BMY}"
# resp = send_and_get_response(msg)
# print( f"offset BMY {delta_BMY}: {resp}" )
# time.sleep(0.1)
# # X
# msg = f"moverel BMX{args.beam} {delta_BMX}"
# resp = send_and_get_response(msg)
# print( f"offset BMX {delta_BMX}: {resp}" )
# time.sleep(0.1)


# # update all phasemask positions 
# msg = f"fpm_offsetallmaskpositions phasemask{args.beam} {delta_BMX} {delta_BMY}"
# resp = send_and_get_response(msg)
# print( f"updating all local phasemask positions based on offset : {resp}" )

# # write file 
# msg = f"fpm_writemaskpos phasemask{beam}"
# resp = send_and_get_response(msg)
# print( f"saving updated phasemask position file for beam {args.beam} : {resp}" )

# print('done')


# # f"/home/asg/Progs/repos/asgard-alignment/config_files/phasemask_positions/beam{beam}/*json"
#             valid_reference_position_files = glob.glob(
#                 f"/home/asg/Progs/repos/asgard-alignment/config_files/phasemask_positions/beam{beam}/*json"
#             )  # save_path + f"/beam{beam}/*json")

#             # Sort by modification time (most recent first)
#             valid_ref_files_sorted = sorted(
#                 valid_reference_position_files, key=os.path.getmtime, reverse=True
#             )

#             # Create display names (just filenames)
#             display_names = [os.path.basename(f) for f in valid_ref_files_sorted]

#             # Create a mapping from display name to full path
#             file_map = dict(zip(display_names, valid_ref_files_sorted))

#             # Show selectbox with display names
#             selected_display_name = st.selectbox(
#                 "Select Reference Position File to Calculate Relative Separations Between Masks",
#                 display_names,
#                 key="selected_reference_position_file",
#             )

#             # Retrieve the full file path based on the selected display name
#             selected_reference_file = file_map[selected_display_name]

#             # # Sort files by modification time (most recent first)
#             # valid_ref_files_sorted = sorted(valid_reference_position_files, key=os.path.getmtime, reverse=True)

#             # selected_reference_file = st.selectbox(
#             #     "Select Reference Position File to Calculate Relative Seperations Between Masks",
#             #     display_names,
#             #     key="selected_file",
#             # )
#             submit_reference_file = st.form_submit_button(
#                 f"Update All Mask Positions Relative to Current registered {st.session_state['selected_mask']} Position (local - not saved)"
#             )

#             if submit_reference_file:
#                 if "unknown" not in st.session_state["selected_mask"][0].lower():

#                     message = f"fpm_updateallmaskpos {targets[0]} {st.session_state['selected_mask'][0]} {selected_reference_file}"

#                     res = send_and_get_response(message)

#                     if "NACK" in res:
#                         st.error(f"Failed to update registered mask: {res}")
#                     else:
#                         st.success(
#                             f"Successfully updated registered mask {st.session_state['selected_mask']}"
#                         )

#                 else:
#                     st.error(f"Cannot update mask position with 'Unknown' mask.")

#     with save_col:

#         st.write("Default save path is: 'config_files/phasemask_positions/'")

#         if st.button(
#             f"Save All Registered Mask Positions in json (update first if changed)"
#         ):

#             if "unknown" not in st.session_state["selected_mask"][0].lower():

#                 # save_path = send_and_get_response(f"fpm_getsavepath {targets[0]}")

#                 # Save the updated positions to file
#                 save_message = f"fpm_writemaskpos {targets[0]}"
#                 save_res = send_and_get_response(save_message)

#                 if "NACK" in save_res:
#                     st.error(
#                         f"Failed to save updated positions"
#                     )  # to file: {save_res}")
#                 else:
#                     st.success(
#                         "Updated positions successfully saved to file"  # at: " + save_path
#                     )
#             else:
#                 st.error(f"Cannot update mask position with 'Unknown' mask.")

#     # message = "fpm_updateallmaskpos {} {} {}"








# import numpy as np 
# import zmq
# import time
# import toml
# import os 
# import argparse
# import matplotlib.pyplot as plt
# import argparse
# import subprocess


# parser = argparse.ArgumentParser(description="scan pupil/image planes in baldr to optimize pos in cold stop.")

# #input beam id 
# parser.add_argument(
#     '--beam',
#     type=str,
#     default="3",
#     help="what beam to look at?. Default: %(default)s"
# )


# parser.add_argument(
#     '--move_plane',
#     type=str,
#     default="pupil",
#     help="what plane to move in the system (pupil or image). Default: %(default)s"
# )


# parser.add_argument(
#     '--search_radius',
#     type=float,
#     default=9,
#     help="search radius of spiral search in microns. Default: %(default)s"
# )
# parser.add_argument(
#     '--dx',
#     type=float,
#     default=3,
#     help="step size in motor units during scan. Default: %(default)s"
# )
# #open json and get regions , exapand by twice size 

# args=parser.parse_args()

# def expand_roi( roi0, expand_by_percent=50 ):
#     X = roi0[1] - roi0[0]
#     extra =expand_by_percent/100 * X 
#     roi = []
#     for i,rr in enumerate( roi0 ):
#         if np.mod(i,2)==0:
#             roi.append( int( rr - extra ) )
#         else:
#             roi.append( int(rr + extra ) )
#     return roi 

# data_path = f"/home/asg/Progs/repos/asgard-alignment/calibration/reports/scan_{args.move_plane}/"
# default_toml = os.path.join("/usr/local/etc/baldr/", "baldr_config_#.toml") 

# with open(default_toml.replace('#',f'{args.beam}'), "r") as f:
#     config_dict = toml.load(f)
#     # Baldr pupils from global frame 
#     baldr_pupils = config_dict['baldr_pupils']

# roi0 = baldr_pupils[f'{args.beam}']

# roi = expand_roi( roi0, expand_by_percent=50 )

# scantype = 'spiral'

# command = [
#     "python",
#     "common/m_scan_multiple_mirrors.py",
#     "--beam",
#     f"{args.beam}",
#     '--system',
#     "baldr",
#     "--move_plane",
#     args.move_plane ,
#     "--global_camera_shm",
#     "/dev/shm/cred1.im.shm",
#     "--initial_pos",
#     "current",
#     "--search_radius",
#     f"{args.search_radius}",
#     "--dx",
#     f"{args.dx}",
#     "--roi",
#     str(roi),
#     "--scantype",
#     scantype,
#     "--data_path",
#     data_path,
#     ]

# process = subprocess.run(
#     command, capture_output=True, text=True
# )

# print("Script Output", process.stdout)