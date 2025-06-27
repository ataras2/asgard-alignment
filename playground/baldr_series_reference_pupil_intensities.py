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
import toml

from asgard_alignment import FLI_Cameras as FLI
from common import phasemask_centering_tool as pct
from pyBaldr import utilities as util
import atexit


# if server is stuck 
# sudo lsof -i :5555 then kill the PID 

# to use plotting when remote sometimes X11 forwarding is bogus.. so use this: 

# def close_all_dms():
#     try:
#         for b in dm:
#             dm[b].close_dm()
#         print("All DMs have been closed.")
#     except Exception as e:
#         print(f"dm object doesn't seem to exist, probably already closed")
# # Register the cleanup function to run at script exit
# atexit.register(close_all_dms)

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


# # default data paths 
# with open( "config_files/file_paths.json") as f:
#     default_path_dict = json.load(f)

# # positions to put thermal source on and take it out to empty position to get dark
# source_positions = {"SSS": {"empty": 80.0, "SBB": 65.5}}

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
    default=f"/home/asg/Progs/repos/asgard-alignment/calibration/reports/reference_pupils/{tstamp_rough}/", #f"/home/heimdallr/data/phasemask_aquisition/{tstamp_rough}/",
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
    '--no_frames',
    type=int,
    default=1000,
    help="how many frames to take in each setting. Default: %(default)s - easy to find"
)

# parser.add_argument(
#     '--cam_mode',
#     type=str,
#     default='globalresetcds',
#     help="camera mode. Default: %(default)s"
# )


parser.add_argument(
    '--cam_gain',
    type=int,
    default=1,
    help="camera gain. Default: %(default)s"
)

# parser.add_argument(
#     '--cam_fps',
#     type=int,
#     default=50,
#     help="frames per second on camera. Default: %(default)s"
# )





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
# Start Data Acquisition 
######################################

toml_file = os.path.join( "config_files", "baldr_config_#.toml")

with open(toml_file.replace('#',f'{args.beam}') ) as file:
    configdata = toml.load(file)
    # Extract the "baldr_pupils" section
    baldr_pupils = configdata.get("baldr_pupils", {})
    #heim_pupils = configdata.get("heimdallr_pupils", {})
            
            
roi = baldr_pupils[args.beam]

c = FLI.fli(roi=roi)

c.send_fli_cmd(f"set gain {args.cam_gain}")


def get_quick_image() :
    im = c.get_image(apply_manual_reduction=True) 

    plt.figure();plt.imshow( im) ;plt.savefig( "delme.png" )


get_quick_image()

fps_grid = np.array( [ 50, 100, 500, 1000,1500, 1720]) #np.array( [ 1500, 1720]) #
N0_dict = {} 
I0_dict = {} 


for fps in fps_grid:
    print( fps )

    fname_base = f'beam{args.beam}_reference_pupils_fps-{round(fps)}_gain-{args.cam_gain}_'
    
    c.send_fli_cmd(f"set fps {fps}")
    time.sleep(1)

    # get a dark 
    c.build_manual_bias(number_of_frames=args.no_frames)
    c.build_manual_dark(number_of_frames=args.no_frames, 
                                      apply_manual_reduction=True,
                                      build_bad_pixel_mask=True, 
                                      kwargs={'std_threshold':10, 'mean_threshold':10} )
    time.sleep(5)

    c.save_fits( fname = args.data_path + fname_base + f'N0_{tstamp}.fits' ,
                 number_of_frames=args.no_frames,
                 apply_manual_reduction=True)






# # Analysis
# import numpy as np
# import matplotlib.pyplot as plt
# import glob
# from astropy.io import fits


# Get all FITS files
files = glob.glob(args.data_path + "*fits")

def extract_fps(filename):
    import re
    """
    Extracts the fps value from the filename using a regular expression.
    Assumes the filename contains 'fps-<number>'.
    Returns the fps as an integer. If not found, returns a large number to sort it to the end.
    """
    m = re.search(r'fps-(\d+)', filename)
    if m:
        return int(m.group(1))
    return float('inf')

# Sort the files based on the extracted fps value.
sorted_files = sorted(files, key=extract_fps)


# Store results
fps_list = []
snr_list = []
snr_unc_list = []
mean_list = []
mean_unc_list = []
# Process each FITS file
for file in sorted_files:
    with fits.open(file) as hdul:
        # Extract FPS from header
        fps = float(hdul[0].header.get("FPS", -1))  # Use -1 if FPS is missing
        print(f"file {file} with fps {fps}")
        if fps == -1:
            print(f"Warning: FPS not found in {file}. Skipping.")
            continue

        # Read frames (shape: (1000, 53, 54))
        frames = hdul['FRAMES'].data  # Shape: (1000, 53, 54)

        if frames is None:
            print(f"Warning: No frame data in {file}. Skipping.")
            continue

        # Sum over time axis (0) to get the pupil image
        summed_image = np.sum(frames, axis=0)  # Shape: (53, 54)

        # Detect pupil in summed image
        center_x, center_y, radius = util.detect_circle(summed_image, plot=False)

        # Define pupil mask
        y, x = np.ogrid[:frames.shape[1], :frames.shape[2]]
        pupil_mask = (x - center_x)**2 + (y - center_y)**2 <= radius**2

        # Extract pupil pixels over all 1000 frames
        pupil_pixels = frames[:, pupil_mask]  # Shape: (1000, pupil_pixels)

        # Compute mean signal per frame in pupil region
        mean_signal_per_frame = np.mean(pupil_pixels, axis=1)  # Mean over pupil pixels

        # Compute standard deviation across frames
        std_signal_over_time = np.std(mean_signal_per_frame)  # STD over 1000 frames

        mean_signal_over_time = np.mean(mean_signal_per_frame) 

        # Compute SNR ratio
        snr = mean_signal_over_time / std_signal_over_time
        
        delta_m = std_signal_over_time / np.sqrt(len(mean_signal_per_frame))
        delta_sigma = std_signal_over_time / np.sqrt(2 * len(mean_signal_per_frame))

        # Error propagation:
        delta_snr = np.sqrt((delta_m / std_signal_over_time)**2 + ((mean_signal_over_time * delta_sigma) / std_signal_over_time**2)**2)

        # Store results
        fps_list.append(fps)
        snr_list.append(snr)
        snr_unc_list.append(delta_snr)
        mean_list.append(  mean_signal_over_time )
        mean_unc_list.append( delta_m  )
        print(f"Processed {file}: FPS={fps}, SNR={snr:.2f}")

# Sort FPS in **ascending** order before plotting
#fps_list, snr_list = zip(*sorted(zip(fps_list, snr_list), key=lambda x: x[0]))


# snr ~ a * (tint - bias)^(0.5) + b
# snr ~ a * tint^(b) 
m1,b1 = np.polyfit( np.sqrt( 1/ np.array(fps_list) ) , snr_list , deg = 1)

fit_label1 = f"Model: SNR = {m1:.2e} sqrt(T) + {b1:.2e}"

m2,b2 = np.polyfit(  np.log10( 1/ np.array(fps_list) )  , np.log10( snr_list ) , deg = 1)

# fit_label2 = f"Model: SNR = {10**(b2):.2e} T ^ {m2:.2e}"

# from scipy.optimize import curve_fit
# # Define the model function: snr = a * sqrt(tint - bias) + b.
# # Note: tint - bias must be positive.
# def model_func(t, a, bias, b):
#     return a * np.sqrt(t - bias) + b

# # Initial guesses for parameters a, bias, and b.
# initial_guess = [1.0, 0.0, 0.0]

# # # Fit the model to the data.
# # # curve_fit will try to adjust a, bias, and b to best match snr_list.
# # params, covariance = curve_fit(model_func,  1/ np.array(fps_list), snr_list, p0=initial_guess)
# # a_fit, bias_fit, b_fit = params

# # # Compute the fitted SNR values.
# # snr_fit = model_func(tint, a_fit, bias_fit, b_fit)

# # # Create a label string with the fitted coefficients in scientific notation (2 decimals).
# # fit_label = f"Model: snr = {a_fit:.2e} √(tint - {bias_fit:.2e}) + {b_fit:.2e}"
# # snr_fit = model_func(1/ np.array(fps_list), a_fit, bias_fit, b_fit)


kwargs = {"fontsize":15}

fit_label1_norm_gain1 = f"Model: SNR = {m1/args.cam_gain:.2e} sqrt(T) + {b1/args.cam_gain:.2e}"

# Plot results
plt.figure(figsize=(8, 5))
#plt.plot( 1/ np.array(fps_list) , (m1* np.sqrt( 1/ np.array(fps_list) ) + b1)/ args.cam_gain,  ":", color='k', label= fit_label1_norm_gain1  )
plt.plot( 1/ np.array(fps_list) , (m1* np.sqrt( 1/ np.array(fps_list) ) + b1),  ":", color='k', label= fit_label1 )
#plt.plot( 1/ np.array(fps_list) , (10**b2) * ( 1/ np.array(fps_list) )**(m2) , ":", color='k',label= fit_label2 )
#plt.plot(1/ np.array(fps_list), 100 * (1/ np.array(fps_list) )**0.5 , label=r"$\sqrt{t}$", color='r')
plt.errorbar(1/ np.array(fps_list), np.array(snr_list) , yerr=snr_unc_list, fmt='.', color='b', ecolor='b', capsize=0, label="Measured")
plt.xscale('log')
plt.yscale('log')
plt.ylim( [10, 1000])
#plt.loglog( 1/ np.array(fps_list) , snr_list, marker='o', linestyle='-', color='b')
plt.xlabel("Integration Time [s]", kwargs)
plt.ylabel(r"Clear Pupil Pixel SNR [$\mu/\sigma$]", kwargs)
plt.text( 5e-3, 50, f"C-RED One\nmode = CDS\ngain = {args.cam_gain}" )
plt.gca().tick_params(labelsize=15)
plt.legend()
# plt.title("SNR vs Integration Time")
plt.grid(True)
'clear_pupil_SNR_vs_tint.jpeg'
"clear_pupil_SNR_vs_tint.jpeg"
'/home/asg/Progs/repos/asgard-alignment/delme.png'
plt.savefig(args.data_path + "clear_pupil_SNR_vs_tint.jpeg" ,bbox_inches = 'tight', dpi=200)
#plt.show()


kwargs = {"fontsize":15}
# Plot results
plt.figure(figsize=(8, 5))
#plt.loglog( 1/ np.array(fps_list) , mean_list, marker='o', linestyle='-', color='b')
plt.errorbar(x=1/ np.array(fps_list), y=mean_list, yerr=mean_unc_list,  color='b', ecolor='b', capsize=0)
plt.xscale('log')
plt.yscale('log')
plt.xlabel("Integration Time [s]", kwargs)
plt.ylabel(r"Clear Pupil Pixel Mean [ADU]", kwargs)
plt.gca().tick_params(labelsize=15)
# plt.title("SNR vs Integration Time")
plt.grid(True)
plt.savefig('/home/asg/Progs/repos/asgard-alignment/clear_pupil_adu_vs_tint.jpeg',bbox_inches = 'tight', dpi=200)
#plt.show()














### 

# ######################################
# # define pupil crop regions 
# ######################################
# baldr_pupils_path = default_path_dict['baldr_pupil_crop'] #"/home/asg/Progs/repos/asgard-alignment/config_files/baldr_pupils_coords.json"

# with open(baldr_pupils_path, "r") as json_file:
#     baldr_pupils = json.load(json_file)

# ######################################
# # Setup camera
# ######################################
# # init camera - we set up to crop on the Baldr pupil
# #roi = baldr_pupils[str(args.beam)] #[None, None, None, None] # 
# #c = FLI.fli(cameraIndex=0, roi=roi)
# # configure with default configuration file
# #config_file_name = os.path.join(c.config_file_path, "default_cred1_config.json")
# #c.configure_camera(config_file_name)

# with open(config_file_name, "r") as file:
#     camera_config = json.load(file)

# apply_manual_reduction = True

# c.send_fli_cmd(f"set mode {args.cam_mode}")
# time.sleep(1)
# c.send_fli_cmd(f"set gain {args.cam_gain}")
# time.sleep(1)

# # time.sleep(1)
# # c.send_fli_cmd(f"set fps {args.cam_fps}")

# c.start_camera()

# time.sleep(5)


# ######################################
# # Setup DMs
# ######################################
# DMshapes_path = default_path_dict["DMshapes_path"]#"/home/asg/Progs/repos/asgard-alignment/DMShapes/"
# dm_config_path = default_path_dict["DM_config_path"] #"/home/asg/Progs/repos/asgard-alignment/config_files/dm_serial_numbers.json"

# ########## ########## ##########
# ########## set up DMs
# with open(dm_config_path, "r") as f:
#     dm_serial_numbers = json.load(f)

# dm = {}
# dm_err_flag = {}
# for beam, serial_number in dm_serial_numbers.items():
#     print(f'beam {beam}====\n\n')
#     dm[beam] = bmc.BmcDm()  # init DM object
#     dm_err_flag[beam] = dm[beam].open_dm(serial_number)  # open DM
#     if not dm_err_flag:
#         print(f"Error initializing DM {beam}")


# flatdm = {}
# for beam, serial_number in dm_serial_numbers.items():
#     flatdm[beam] = pd.read_csv(
#         DMshapes_path + f"{serial_number}_FLAT_MAP_COMMANDS.csv",
#         header=None,
#     )[0].values


# # apply flats to DM 
# for beam, _ in dm_serial_numbers.items():
#     dm[beam].send_data( flatdm[beam] )
    
# ######################################
# # check the cropped pupil regions are correct
# ######################################
# full_im = c.get_image_in_another_region( )

# # Plot the image
# plt.figure(figsize=(8, 8))
# plt.imshow(np.log10(full_im), cmap='gray',origin='upper' ) #, origin='upper') #extent=[0, full_im.shape[1], 0, full_im.shape[0]]
# plt.colorbar(label='Intensity')

# # Overlay red boxes for each cropping region
# for beam_tmp, (row1, row2, column1, column2) in  baldr_pupils.items():
#     plt.plot([column1, column2, column2, column1, column1],
#              [row1, row1, row2, row2, row1],
#              color='red', linewidth=2, label=f'Beam {beam_tmp}' if beam_tmp == 1 else "")
#     plt.text((column1 + column2) / 2, row1 , f'Beam {beam_tmp}', 
#              color='red', fontsize=15, ha='center', va='bottom')


# # Add labels and legend
# plt.title('Image with Baldr Cropping Regions')
# plt.xlabel('Columns')
# plt.ylabel('Rows')
# plt.legend(loc='upper right')
# plt.savefig('delme.png')
# plt.show()
# plt.close() 





# ######################################
# # Move to the requested phasemask 
# ######################################
# message = f"!fpm_movetomask phasemask{args.beam} {args.phasemask_name}"
# res = send_and_get_response(message)
# print(res)

# ######################################
# # Manual fine alignment check
# ######################################
# savefig_tmp = 'delme.png'
# print( f'\n\n=======\nopen {savefig_tmp} to see the images!')
# pct.move_relative_and_get_image(cam=c, 
#                                 beam=args.beam, 
#                                 phasemask=state_dict["socket"], 
#                                 savefigName=savefig_tmp, 
#                                 use_multideviceserver=True)



# ######################################
# # Start Data Acquisition 
# ######################################
# fps_grid = np.array( [20, 50, 100, 200, 500, 1000, 2000])
# N0_dict = {} 
# I0_dict = {} 
# mask_offset = 200.0 #um <- offset applied to BMX to get a clear pupil (take phase mask out)

# # A better (maybe to-do) sequence to avoid moving the source and mask in/out each iteration we 
# # take 
# # 1) darks 
# # 2) phasemask in frames 
# # 3) phasemask out frames 
# # only issue is built darks and pixel masks are saved in fits and so won't be synchroinised if taken in different order
# for fps in fps_grid:

#     fname_base = f'beam{args.beam}_reference_pupils_fps-{round(fps)}_gain-{args.cam_gain}_'

#     print( fps )
#     c.stop_camera()
#     time.sleep( 1 )
#     c.send_fli_cmd(f"set fps {fps}")
#     time.sleep( 1 )
#     c.start_camera()
#     time.sleep( 1 )
#     ######################################
#     # Move source out to get raw darks 
#     ######################################
#     #---->MOVE SOURCE OUT 
#     state_dict["socket"].send_string(f"!moveabs SSS {source_positions['SSS']['empty']}")
#     res = socket.recv_string()
#     print(f"Response: {res}")

#     time.sleep(5)

#     # we save the raw data 
#     c.save_fits( fname = args.data_path + fname_base + f'DARKS_{tstamp}.fits' , number_of_frames=args.no_frames, apply_manual_reduction=False)

#     # also build darks / bad pixel map to include in the following saved fits files
#     c.build_manual_dark( no_frames = 100 )

#     bad_pixels = c.get_bad_pixel_indicies( no_frames = 300, std_threshold = 20, mean_threshold=6 , flatten=False)

#     c.build_bad_pixel_mask( bad_pixels , set_bad_pixels_to = 0)

#     time.sleep(1)

#     #<----MOVE SOURCE IN
#     state_dict["socket"].send_string(f"!moveabs SSS {source_positions['SSS']['SBB']}")
#     res = socket.recv_string()
#     print(f"Response: {res}")

#     time.sleep(5)

#     c.save_fits( fname = args.data_path + fname_base + f'I0_{tstamp}.fits' , number_of_frames=args.no_frames, apply_manual_reduction=False)

#     #---->MOVE PHASEMASK OUT 
#     message = f"!moverel BMX{args.beam} {mask_offset}"
#     res = send_and_get_response(message)
#     print(res) 

#     time.sleep( 1 )

#     c.save_fits( fname = args.data_path + fname_base + f'N0_{tstamp}.fits' , number_of_frames=args.no_frames, apply_manual_reduction=False)


#     #<----MOVE PHASEMASK IN
#     message = f"!moverel BMX{args.beam} {-mask_offset}"
#     res = send_and_get_response(message)
#     print(res) 

#     time.sleep(5)





#__________________________________________

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


