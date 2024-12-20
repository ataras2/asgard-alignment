import numpy as np
from astropy.io import fits
import os
import time
import matplotlib.pyplot as plt
import importlib
import json
import datetime
import sys
import glob
import pandas as pd
import argparse
import zmq

from asgard_alignment import FLI_Cameras as FLI
from common import phasemask_centering_tool as pct
import atexit

# if server is stuck 
# sudo lsof -i :5555 then kill the PID 

# to use plotting when remote sometimes X11 forwarding is bogus.. so use this: 
import matplotlib 
matplotlib.use('Agg')


import numpy as np
from astropy.io import fits
import os
import time
import matplotlib.pyplot as plt
import importlib
import json
import datetime
import sys
import glob
import pandas as pd
import argparse
import zmq

from asgard_alignment import FLI_Cameras as FLI
from common import phasemask_centering_tool as pct
import atexit

sys.path.insert(1, "/opt/Boston Micromachines/lib/Python3/site-packages/")
import bmc

# if server is stuck 
# sudo lsof -i :5555 then kill the PID 

# to use plotting when remote sometimes X11 forwarding is bogus.. so use this: 
import matplotlib 
matplotlib.use('Agg')


def close_all_dms():
    try:
        for b in dm:
            dm[b].close_dm()
        print("All DMs have been closed.")
    except Exception as e:
        print(f"dm object doesn't seem to exist, probably already closed")
# Register the cleanup function to run at script exit
atexit.register(close_all_dms)

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


def convert_to_serializable(obj):
    """
    Recursively converts NumPy arrays and other non-serializable objects to serializable forms.
    Also converts dictionary keys to standard types (str, int, float).
    """
    if isinstance(obj, np.ndarray):
        return obj.tolist()  # Convert NumPy arrays to lists
    elif isinstance(obj, np.integer):
        return int(obj)  # Convert NumPy integers to Python int
    elif isinstance(obj, np.floating):
        return float(obj)  # Convert NumPy floats to Python float
    elif isinstance(obj, dict):
        return {str(key): convert_to_serializable(value) for key, value in obj.items()}  # Ensure keys are strings
    elif isinstance(obj, list):
        return [convert_to_serializable(item) for item in obj]
    else:
        return obj  # Base case: return the object itself if it doesn't need conversion


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
parser.add_argument("--host", type=str, default="localhost", help="Server host")
parser.add_argument("--port", type=int, default=5555, help="Server port")
parser.add_argument(
    "--timeout", type=int, default=5000, help="Response timeout in milliseconds"
)

parser.add_argument(
    '--data_path',
    type=str,
    default=f"/home/heimdallr/Documents/asgard-alignment/calibration/reports/reference_pupils/{tstamp_rough}/", #f"/home/heimdallr/data/phasemask_aquisition/{tstamp_rough}/",
    help="Path to the directory for storing pokeramp data. Default: %(default)s"
)
parser.add_argument(
    '--beam',
    type=str,
    default="1",
    help="what beam to look at?. Default: %(default)s"
)
parser.add_argument(
    '--phasemask_name',
    type=str,
    default="H3",
    help="what phasemask to look at (e.g. J1, J2..J5, H1, H2,...H5). Default: %(default)s - easy to find"
)

parser.add_argument(
    '--no_frames',
    type=int,
    default=100,
    help="how many frames to take in each setting. Default: %(default)s - easy to find"
)


# parser.add_argument(
#     '--search_radius',
#     type=float,
#     default=100,
#     help="search radius of spiral search in microns. Default: %(default)s"
# )
# parser.add_argument(
#     '--step_size',
#     type=float,
#     default=20,
#     help="set size in microns of square spiral search. Default: %(default)s"
# )

# parser.add_argument(
#     '--cam_fps',
#     type=int,
#     default=50,
#     help="frames per second on camera. Default: %(default)s"
# )
# parser.add_argument(
#     '--cam_gain',
#     type=int,
#     default=5,
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

if not os.path.exists(args.data_path):
     print(f'made directory : {args.data_path}')
     os.makedirs(args.data_path)




######################################
# define pupil crop regions 
######################################
baldr_pupils_path = default_path_dict['baldr_pupil_crop'] #"/home/heimdallr/Documents/asgard-alignment/config_files/baldr_pupils_coords.json"

with open(baldr_pupils_path, "r") as json_file:
    baldr_pupils = json.load(json_file)

######################################
# Setup camera
######################################
# init camera - we set up to crop on the Baldr pupil
roi = baldr_pupils[str(args.beam)] #[None, None, None, None] # 
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

time.sleep(5)


######################################
# Setup DMs
######################################
DMshapes_path = default_path_dict["DMshapes_path"]#"/home/heimdallr/Documents/asgard-alignment/DMShapes/"
dm_config_path = default_path_dict["DM_config_path"] #"/home/heimdallr/Documents/asgard-alignment/config_files/dm_serial_numbers.json"

########## ########## ##########
########## set up DMs
with open(dm_config_path, "r") as f:
    dm_serial_numbers = json.load(f)

dm = {}
dm_err_flag = {}
for beam, serial_number in dm_serial_numbers.items():
    print(f'beam {beam}====\n\n')
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


# apply flats to DM 
for beam, _ in dm_serial_numbers.items():
    dm[beam].send_data( flatdm[beam] )
    
######################################
# check the cropped pupil regions are correct
######################################
full_im = c.get_image_in_another_region( )

# Plot the image
plt.figure(figsize=(8, 8))
plt.imshow(np.log10(full_im), cmap='gray',origin='upper' ) #, origin='upper') #extent=[0, full_im.shape[1], 0, full_im.shape[0]]
plt.colorbar(label='Intensity')

# Overlay red boxes for each cropping region
for beam_tmp, (row1, row2, column1, column2) in  baldr_pupils.items():
    plt.plot([column1, column2, column2, column1, column1],
             [row1, row1, row2, row2, row1],
             color='red', linewidth=2, label=f'Beam {beam_tmp}' if beam_tmp == 1 else "")
    plt.text((column1 + column2) / 2, row1 , f'Beam {beam_tmp}', 
             color='red', fontsize=15, ha='center', va='bottom')


# Add labels and legend
plt.title('Image with Baldr Cropping Regions')
plt.xlabel('Columns')
plt.ylabel('Rows')
plt.legend(loc='upper right')
plt.savefig('delme.png')
plt.show()
plt.close() 





######################################
# Move to the requested phasemask 
######################################
message = f"!fpm_movetomask phasemask{args.beam} {args.phasemask_name}"
res = send_and_get_response(message)
print(res)

######################################
# Manual fine alignment check
######################################
savefig_tmp = 'delme.png'
print( f'\n\n=======\nopen {savefig_tmp} to see the images!')
pct.move_relative_and_get_image(cam=c, 
                                beam=args.beam, 
                                phasemask=state_dict["socket"], 
                                savefigName=savefig_tmp, 
                                use_multideviceserver=True)



######################################
# Start Data Acquisition 
######################################
fps_grid = np.array( [20, 50, 100, 200, 500, 1000, 2000])
N0_dict = {} 
I0_dict = {} 
mask_offset = 200.0 #um <- offset applied to BMX to get a clear pupil (take phase mask out)

# A better (maybe to-do) sequence to avoid moving the source and mask in/out each iteration we 
# take 
# 1) darks 
# 2) phasemask in frames 
# 3) phasemask out frames 
# only issue is built darks and pixel masks are saved in fits and so won't be synchroinised if taken in different order
for fps in fps_grid:

    fname_base = f'beam{args.beam}_reference_pupils_fps-{round(fps)}_gain-{args.cam_gain}_'

    print( fps )
    
    c.send_fli_cmd(f"set fps {fps}")
    time.sleep( 1 )

    ######################################
    # Move source out to get raw darks 
    ######################################
    #---->MOVE SOURCE OUT 
    state_dict["socket"].send_string(f"!moveabs SSS {source_positions['SSS']['empty']}")
    res = socket.recv_string()
    print(f"Response: {res}")

    time.sleep(5)

    # we save the raw data 
    c.save_fits( fname = args.data_path + fname_base + f'DARKS_{tstamp}.fits' , number_of_frames=args.no_frames, apply_manual_reduction=False)

    # also build darks / bad pixel map to include in the following saved fits files
    c.build_manual_dark( no_frames = 100 )

    bad_pixels = c.get_bad_pixel_indicies( no_frames = 300, std_threshold = 20, mean_threshold=6 , flatten=False)

    c.build_bad_pixel_mask( bad_pixels , set_bad_pixels_to = 0)

    time.sleep(1)

    #<----MOVE SOURCE IN
    state_dict["socket"].send_string(f"!moveabs SSS {source_positions['SSS']['SBB']}")
    res = socket.recv_string()
    print(f"Response: {res}")

    time.sleep(5)

    c.save_fits( fname = args.data_path + fname_base + f'I0_{tstamp}.fits' , number_of_frames=args.no_frames, apply_manual_reduction=False)

    #---->MOVE PHASEMASK OUT 
    message = f"!moverel BMX{args.beam} {mask_offset}"
    res = send_and_get_response(message)
    print(res) 

    time.sleep( 1 )

    c.save_fits( fname = args.data_path + fname_base + f'N0_{tstamp}.fits' , number_of_frames=args.no_frames, apply_manual_reduction=False)


    #<----MOVE PHASEMASK IN
    message = f"!moverel BMX{args.beam} {-mask_offset}"
    res = send_and_get_response(message)
    print(res) 

    time.sleep(5)



# # old (outdated) example of analysis from ben-branch asgard-alignment/playground/get_series_of_reference_pupils.py
# # will need to edit this to read things in differently 
# # analysis 

# a = hdulist 

# plt.figure(1)
# pup_filt =  a[0].data[0] > np.mean( a[0].data[0] ) + 0.5 * np.std( a[0].data[0] )
# pup_filt[:,:25] = False # some bad pixels here 

# fig,ax = plt.subplots(1,2 )
# ax[0].imshow( a[0].data[0] )
# ax[1].imshow( pup_filt )
# plt.savefig( data_path + f'pupil_filter_{tstamp}.png')    


# import matplotlib.colors as colors

# # look at the actual variance on the pupil 

# snr_list = [];  t_list = []
# for i in range(len(a)//2):
#     plt.close()
#     tint = round( 1e3 * ( float(a[i].header['EXTNAME'].split('tint')[-1] ) ),2)
#     t_list.append( tint )
#     plt.figure(4)
#     SNR = np.mean(  a[i].data ,axis=0 )  / np.std( a[i].data ,axis=0 )
#     im = plt.imshow( SNR ) 
#     plt.colorbar(im, label='SNR' ) #, norm=colors.LogNorm() )
#     plt.title( f"DIT = {tint}ms")

#     snr_list.append( SNR[pup_filt] )
    
#     plt.savefig(data_path + f'pupil_N0_SNR_DIT-{tint}ms_{tstamp}.png')


    
# plt.figure(i+1); 

# plt.xlabel('integration time [ms]',fontsize=15)
# plt.ylabel('SNR',fontsize=15)
# plt.gca().tick_params(labelsize=15)
# plt.semilogx( t_list, np.array(snr_list).mean(axis=1), '.')
# plt.savefig( data_path + f'SNR_vs_tint_logscale_{tstamp}.png')    
    

# # look at I0 (phasemask in)

# snr_list = [];  t_list = []
# for i in range(len(a)//2, len(a)):
#     plt.close()
#     tint = round( 1e3 * ( float(a[i].header['EXTNAME'].split('tint')[-1] ) ),2)
#     t_list.append( tint )
#     plt.figure(4)
#     SNR = np.mean(  a[i].data ,axis=0 )  / np.std( a[i].data ,axis=0 )
#     im = plt.imshow( SNR ) 
#     plt.colorbar(im, label='SNR' ) #, norm=colors.LogNorm() )
#     plt.title( f"DIT = {tint}ms")

#     snr_list.append( SNR[pup_filt] )
    
#     plt.savefig(data_path + f'pupil_I0_SNR_DIT-{tint}ms_{tstamp}.png')


