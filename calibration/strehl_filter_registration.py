#!/usr/bin/env python
import zmq
import numpy as np
import toml  # Make sure to install via `pip install toml` if needed
import argparse
import os
import json
import time
import datetime
import subprocess 
import glob 
from astropy.io import fits
import matplotlib.pyplot as plt
from scipy.interpolate import RegularGridInterpolator

import matplotlib.gridspec as gridspec

from pyBaldr import utilities as util
from asgard_alignment import FLI_Cameras as FLI
from asgard_alignment.DM_shm_ctrl import dmclass


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




def plot_strehl_pixel_registration(data , exterior_filter, secondary_filter, savefig = None):

    label = "I0-N0"
    fs = 18
    if np.sum( exterior_filter ):
        # Exterior filter boundaries (red)
        ext_x_min, ext_x_max = 0.5 + np.min(np.where(np.abs(np.diff(exterior_filter, axis=1)) > 0)[1]), \
                            0.5 + np.max(np.where(np.abs(np.diff(exterior_filter, axis=1)) > 0)[1])
        ext_y_min, ext_y_max = 0.5+ np.min(np.where(np.abs(np.diff(exterior_filter, axis=0)) > 0)[0]), \
                            0.5 + np.max(np.where(np.abs(np.diff(exterior_filter, axis=0)) > 0)[0])
    
    if np.sum( secondary_filter ):   
        # Secondary filter boundaries (blue)
        sec_x_min, sec_x_max =  0.5 + np.min( np.where( abs(np.diff( secondary_filter, axis=1  )) > 0)[1] ), \
                                0.5 + np.max( np.where( abs(np.diff( secondary_filter, axis=1  )) > 0)[1] )
        sec_y_min, sec_y_max =  0.5 + np.min( np.where( abs(np.diff( secondary_filter, axis=0   )) > 0)[0] ), \
                                0.5 + np.max( np.where( abs(np.diff( secondary_filter, axis=0  )) > 0)[0] )

    # Create figure and gridspec for joint plot
    fig = plt.figure(figsize=(10, 8))
    gs = gridspec.GridSpec(2, 2, width_ratios=[4, 1], height_ratios=[1, 4],
                        wspace=0.05, hspace=0.05)

    # Axes: Main heatmap, top x histogram, right y histogram
    ax_main = plt.subplot(gs[1, 0])
    ax_xhist = plt.subplot(gs[0, 0], sharex=ax_main)
    ax_yhist = plt.subplot(gs[1, 1], sharey=ax_main)

    
    # Plot the imag (main axis)

    im = ax_main.imshow(data, aspect='auto', origin='lower', interpolation='nearest')
    ax_main.text(0,0, f"{label}",fontsize=25, color='white')
    ax_main.set_xlabel('X (pixels)',fontsize=fs)
    ax_main.set_ylabel('Y (pixels)',fontsize=fs)

  
    # marginal histograms (for counts)
    # --------------------------
    x_counts = np.sum(data, axis=0)
    y_counts = np.sum(data, axis=1)

    ax_xhist.bar(np.arange(len(x_counts)), x_counts, color='gray', edgecolor='black')
    ax_yhist.barh(np.arange(len(y_counts)), y_counts, color='gray', edgecolor='black')
    ax_yhist.set_xlabel("ADU", fontsize=fs)
    ax_xhist.set_ylabel("ADU", fontsize=fs)
    # Remove tick labels on marginal plots
    plt.setp(ax_xhist.get_xticklabels(), visible=False)
    plt.setp(ax_yhist.get_yticklabels(), visible=False)

    # Ensure the histogram axes align with the main heatmap axes:
    ax_xhist.set_xlim(ax_main.get_xlim())
    ax_yhist.set_ylim(ax_main.get_ylim())

    # Draw contours for the filter regions 
    # --------------------------
    # Convert boolean filters to float so that contour finds a level at 0.5.
    if np.sum( exterior_filter ):
        ax_main.contour(exterior_filter.astype(float), levels=[0.5], extent=[-0.5, data.shape[1]-0.5, -0.5, data.shape[0]-0.5],
                        colors='red', linestyles='-', linewidths=2, origin='lower')

        ex_coords = np.argwhere(exterior_filter)      # shape (N, 2)

        # Plot a cross at each True pixel - to be ABSOLUTTTELY SURE
        # Note: row = y, col = x. So when calling scatter or plot, pass x=col, y=row.
        ax_main.scatter(ex_coords[:,1], ex_coords[:,0],
                        marker='x', color='red', alpha =0.4, label='Exterior Filter')

    if np.sum( secondary_filter ):    
        ax_main.contour(secondary_filter.astype(float), levels=[0.5], extent=[-0.5, data.shape[1]-0.5, -0.5, data.shape[0]-0.5],
                        colors='blue', linestyles='-', linewidths=2, origin='lower')

        sec_coords = np.argwhere(secondary_filter)    # shape (M, 2)

        ax_main.scatter(sec_coords[:,1], sec_coords[:,0],
                        marker='x', color='blue',alpha =0.4, label='Secondary Filter')

    ax_main.legend(fontsize=fs)
    # --------------------------
    # Draw vertical lines on the x-axis (top histogram and main heatmap)

    # Exterior filter (red)
    if np.sum( exterior_filter ):
        ax_xhist.axvline(ext_x_min, color='red', linestyle='--', linewidth=2, label='Exterior Boundary')
        ax_xhist.axvline(ext_x_max, color='red', linestyle='--', linewidth=2)
        ax_main.axvline(ext_x_min, color='red', linestyle='--', linewidth=2)
        ax_main.axvline(ext_x_max, color='red', linestyle='--', linewidth=2)

    # Secondary filter (blue)
    if np.sum( secondary_filter ):  
        ax_xhist.axvline(sec_x_min, color='blue', linestyle='--', linewidth=2, label='Secondary Boundary')
        ax_xhist.axvline(sec_x_max, color='blue', linestyle='--', linewidth=2)
        ax_main.axvline(sec_x_min, color='blue', linestyle='--', linewidth=2)
        ax_main.axvline(sec_x_max, color='blue', linestyle='--', linewidth=2)

    #ax_xhist.legend(loc='upper right', fontsize=fs)

    # Draw horizontal lines on the y-axis (right histogram and main heatmap)
    # --------------------------
    # Exterior filter (red)
    if np.sum( exterior_filter ):
        ax_yhist.axhline(ext_y_min, color='red', linestyle='--', linewidth=2)
        ax_yhist.axhline(ext_y_max, color='red', linestyle='--', linewidth=2)
        ax_main.axhline(ext_y_min, color='red', linestyle='--', linewidth=2)
        ax_main.axhline(ext_y_max, color='red', linestyle='--', linewidth=2)

    # Secondary filter (blue)
    if np.sum( secondary_filter ):  
        ax_yhist.axhline(sec_y_min, color='blue', linestyle='--', linewidth=2)
        ax_yhist.axhline(sec_y_max, color='blue', linestyle='--', linewidth=2)
        ax_main.axhline(sec_y_min, color='blue', linestyle='--', linewidth=2)
        ax_main.axhline(sec_y_max, color='blue', linestyle='--', linewidth=2)


    ax_xhist.tick_params(labelsize=15)
    ax_yhist.tick_params(labelsize=15)
    ax_main.tick_params(labelsize=15)

    plt.tight_layout()
    if savefig is not None:
        plt.savefig( savepath, bbox_inches='tight', dpi=200)
        print( f"saving image {savepath}")
    #plt.show()
    plt.close()



parser = argparse.ArgumentParser(description="Interaction and control matricies for fine phasemask alignment")


######## HARD CODED 
hc_fps = 200 
hc_gain = 1 
default_toml = os.path.join("/usr/local/etc/baldr/", "baldr_config_#.toml") 

# Camera shared memory path
parser.add_argument(
    "--global_camera_shm",
    type=str,
    default="/dev/shm/cred1.im.shm",
    help="Camera shared memory path. Default: /dev/shm/cred1.im.shm"
)

# TOML file path; default is relative to the current file's directory.
parser.add_argument(
    "--toml_file",
    type=str,
    default=default_toml,
    help="TOML file pattern (replace # with beam_id) to write/edit. Default: ../config_files/baldr_config.toml (relative to script)"
)

# Beam ids: provided as a comma-separated string and converted to a list of ints.
parser.add_argument(
    "--beam_id",
    type=lambda s: [int(item) for item in s.split(",")],
    default=[1,2,3,4], # 1, 2, 3, 4],
    help="Comma-separated beam IDs to apply. Default: 1,2,3,4"
)

parser.add_argument(
    "--phasemask",
    type=str,
    default="H3",
    help="phasemask to move to. Try use a reasonable size one like H3 (default)"
)


parser.add_argument("--lobe_threshold",
                    type=float, 
                    default=0.03, 
                    help="threshold for pupil side lobes to define a Strehl proxy pixels. \
                        These are generally where |I0 - N0| > lobe_threshold * <N0[pupil]>,\
                            in addition to some other radii criteria.  Default: Default: %(default)s")        

parser.add_argument("--fig_path", 
                    type=str, 
                    default=None, 
                    help="path/to/output/image/ for the saved figures")

parser.add_argument("--host", type=str, default="172.16.8.6", help="Server host")
parser.add_argument("--port", type=int, default=5555, help="Server port")
parser.add_argument(
    "--timeout", type=int, default=5000, help="Response timeout in milliseconds"
)


args=parser.parse_args()

tstamp_rough = datetime.datetime.now().strftime("%d-%m-%Y")

# set up commands to move motors phasemask
context = zmq.Context()
context.socket(zmq.REQ)
socket = context.socket(zmq.REQ)
socket.setsockopt(zmq.RCVTIMEO, args.timeout)
server_address = f"tcp://{args.host}:{args.port}"
socket.connect(server_address)
state_dict = {"message_history": [], "socket": socket}


#########################


pupil_masks = {}
for beam_id in args.beam_id:

    # read in TOML as dictionary for config 
    with open(args.toml_file.replace('#',f'{beam_id}'), "r") as f:
        config_dict = toml.load(f)
        # Baldr pupils from global frame 
        baldr_pupils = config_dict['baldr_pupils'] 


#c_dict = {}
# just open one camera
c = FLI.fli(args.global_camera_shm, roi = [None,None,None,None])
# for beam_id in args.beam_id:
#     r1,r2,c1,c2 = baldr_pupils[f'{beam_id}']
#     c_dict[beam_id] = FLI.fli(args.global_camera_shm, roi = [r1,r2,c1,c2])

#####################
#Hard coded frame rate and gain 
#####################


## get initial gain and fps
# fps0 = FLI.extract_value( c_dict[args.beam_id[0]].send_fli_cmd( "fps raw" ) ) 
# gain0 = FLI.extract_value( c_dict[args.beam_id[0]].send_fli_cmd( "gain raw" ) ) 
fps0 = FLI.extract_value( c.send_fli_cmd( "fps raw" ) ) 
gain0 = FLI.extract_value( c.send_fli_cmd( "gain raw" ) ) 



## Set to standard hard coded frame rate and gain for this script
# c_dict[args.beam_id[0]].send_fli_cmd(f"set fps {hc_fps}")
# time.sleep(1)
# c_dict[args.beam_id[0]].send_fli_cmd(f"set gain {hc_gain}")
# time.sleep(1)
c.send_fli_cmd(f"set fps {hc_fps}")
time.sleep(1)
c.send_fli_cmd(f"set gain {hc_gain}")
time.sleep(1)

# get new darks, bias bad pixel map 
# #---------- New Darks 
# run a new set of darks 
# we should later just find a recent one that fits settings and then go with that 


valid_cal_files = util.find_calibration_files(mode=c.config['mode'], gain=int(hc_gain) , target_fps=float(hc_fps), base_dir="MASTER_DARK", time_diff_thresh=datetime.timedelta(2), fps_diff_thresh=10)



# if no valid ones than we make some
if not valid_cal_files: 
    print( "no valid calibration files within the last few days. Taking new ones! ")
    script_path = "/home/asg/Progs/repos/dcs/calibration_frames/gen_dark_bias_badpix.py"
    params = ["--gains", f"{int(c.config['gain'])}", 
              "--fps", f"{c.config['fps']}", 
              "--mode", f"{c.config['mode']}", #"--mode", f"{c_dict[args.beam_id[0]].config['mode']}", 
              "--method", "linear_fit" ]
    try:
        # Run the script and ensure it completes
        with subprocess.Popen(["python", script_path]+params, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True) as process:
            stdout, stderr = process.communicate()  # Wait for process to complete

            if process.returncode == 0:
                print("Script executed successfully!")
                print(stdout)  # Print standard output (optional)
            else:
                print(f"Script failed with error:\n{stderr}")

    except Exception as e:
        print(f"Error running script: {e}")

# get darks and bad pixels 
bias_fits_files = util.find_calibration_files(mode=c.config['mode'], gain=int(hc_gain) , target_fps=float(hc_fps), base_dir="MASTER_BIAS", time_diff_thresh=datetime.timedelta(2), fps_diff_thresh=10) #glob.glob(f"/home/asg/Progs/repos/dcs/calibration_frames/products/{tstamp_rough}/MASTER_BIAS/*.fits") 
dark_fits_files = util.find_calibration_files(mode=c.config['mode'], gain=int(hc_gain) , target_fps=float(hc_fps), base_dir="MASTER_DARK", time_diff_thresh=datetime.timedelta(2), fps_diff_thresh=10) #glob.glob(f"/home/asg/Progs/repos/dcs/calibration_frames/products/{tstamp_rough}/MASTER_DARK/*.fits") 
raw_darks_files =  util.find_calibration_files(mode=c.config['mode'],gain=int(hc_gain) , target_fps=float(hc_fps), base_dir="RAW_DARKS", time_diff_thresh=datetime.timedelta(2), fps_diff_thresh=10) #glob.glob(f"/home/asg/Progs/repos/dcs/calibration_frames/products/{tstamp_rough}/RAW_DARKS/*.fits") 

for lab, ff in zip(['bias','dark'], [bias_fits_files, dark_fits_files] ):
    # Assumes we just took one!!! would be quicker to check subdirectories for one that matches the mode and gain with nearest fps. 
    most_recent = max(ff, key=os.path.getmtime) 
    with fits.open( most_recent ) as d:
        c.reduction_dict[lab].append(  d[0].data.astype(int) )       # for beam_id in args.beam_id:
        #     r1,r2,c1,c2 = baldr_pupils[f'{beam_id}']
        #     c_dict[beam_id].reduction_dict[lab].append(  d[0].data.astype(int)[r1:r2, c1:c2] )

# bad pixels 
most_recent = max(raw_darks_files , key=os.path.getmtime) 
with fits.open( most_recent ) as d:

    bad_pixels, bad_pixel_mask = FLI.get_bad_pixels( d[0].data, std_threshold=3, mean_threshold=3)
    bad_pixel_mask[0][0] = False # the frame tag should not be masked! 
    #c_dict[beam_id].reduction_dict['bad_pixel_mask'].append(  (~bad_pixel_mask ).astype(int) )
    c.reduction_dict['bad_pixel_mask'].append(  (~bad_pixel_mask ).astype(int) )

    # for beam_id in args.beam_id:
    #     r1,r2,c1,c2 = baldr_pupils[f'{beam_id}']
    #     # we reverse so its true on pixels we want to keep 
    #     c_dict[beam_id].reduction_dict['bad_pixel_mask'].append(  (~bad_pixel_mask ).astype(int)[r1:r2, c1:c2] )


#c_dict[beam_id].reduction_dict["dark"].append( dark )


# for beam_id in args.beam_id:
#     c_dict[beam_id].build_manual_bias(number_of_frames=500)
#     c_dict[beam_id].build_manual_dark(number_of_frames=500, 
#                                       apply_manual_reduction=True,
#                                       build_bad_pixel_mask=True, 
#                                       sleeptime = 10,
#                                       kwargs={'std_threshold':10, 'mean_threshold':6} )
  


# img = np.mean( c_dict[beam_id].get_some_frames(number_of_frames=100, apply_manual_reduction=True) , axis=0)
# title_list = ['bias','dark','img']
# im_list = [c_dict[beam_id].reduction_dict['bias'][-1], c_dict[beam_id].reduction_dict['dark'][-1], img]
# util.nice_heatmap_subplots( im_list, savefig='delme.png')


# set up DM SHMs 
print( 'setting up DMs')
dm_shm_dict = {}
for beam_id in args.beam_id:
    dm_shm_dict[beam_id] = dmclass( beam_id=beam_id )
    # zero all channels
    dm_shm_dict[beam_id].zero_all()
    
    # activate flat (does this on channel 1)
    #dm_shm_dict[beam_id].activate_flat()

    # apply dm flat + calibrated offset (does this on channel 1)
    dm_shm_dict[beam_id].activate_calibrated_flat()
    


# # move to phasemask 
# for beam_id in args.beam_id:
#     message = f"fpm_movetomask phasemask{beam_id} {args.phasemask}"
#     res = send_and_get_response(message)
#     print(f"moved to phasemask {args.phasemask} on beam {beam_id} with response: {res}")


# # optimize alignment 
# for beam_id in args.beam_id:
#     cmd = ["python", "calibration/fine_phasemask_alignment.py","--beam_id",f"{beam_id}","--method","gradient_descent"] # brute_scan

#     with subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True) as process:
#         stdout, stderr = process.communicate()

#     print("STDOUT:", stdout)
#     print("STDERR:", stderr)


## Get ZWFS and CLEAR reference pupils 

########____ ASSUME THAT WE HAAVE THINGS ALIGNED WHEN CALLING THIS SCRIPT 
zwfs_pupils = {}
for beam_id in args.beam_id:
    r1,r2,c1,c2 = baldr_pupils[f'{beam_id}']
    # zwfs_pupils[beam_id] = float(c_dict[beam_id].config['fps']) * np.mean( 
    #     c_dict[beam_id].get_data( 
    #         apply_manual_reduction = True ),
    #         axis = 0) # ADU/s !    so we multiply by FPS

    img = float(c.config['fps']) * np.mean( 
        c.get_data( 
            apply_manual_reduction = True ),
            axis = 0)[r1:r2,c1:c2] # ADU/s !    so we multiply by FPS

    # on top of the bad pixel mask 
    #img[img > 0.9e5] = 0
    #img[img < -1e2] = 0
    zwfs_pupils[beam_id] = img


util.nice_heatmap_subplots( [zz for zz in zwfs_pupils.values()], savefig='delme.png')

# Get reference pupils (later this can just be a SHM address)
clear_pupils = {}
secondary_filter_dict = {}
exterior_filter_dict = {}
#initial_pos = {}
rel_offset = 200.0 #um phasemask offset for clear pupil
print( 'Moving FPM out to get clear pupils')
for beam_id in args.beam_id:

    r1,r2,c1,c2 = baldr_pupils[f'{beam_id}']

    message = f"moverel BMX{beam_id} {rel_offset}"
    res = send_and_get_response(message)
    print(res) 
    time.sleep(2)
    print( 'gettin clear pupils')
    N0s = c.get_data( apply_manual_reduction=True)
    #N0s = c_dict[beam_id].get_data( apply_manual_reduction=True) #get_some_frames(number_of_frames = 1000, apply_manual_reduction=True) 

    # move back (so we have time buffer while calculating b)
    print( 'Moving FPM back in beam.')
    message = f"moverel BMX{beam_id} {-rel_offset}"
    res = send_and_get_response(message)
    print(res) 
    time.sleep(2)

    # Now procees/fit the pupil  (ADU/S)!!
    #clear_pupils[beam_id] = float(c_dict[beam_id].config['fps']) *  np.mean( N0s , axis=0)
    img = float(c.config['fps']) *  np.mean( N0s , axis=0)[r1:r2,c1:c2]

    #img[img > 0.9e5] = 0
    #img[img < -1e2] = 0
    clear_pupils[beam_id] = img 

    ### DETECT A PUPIL MASK FROM CLEAR MASK 
    center_x, center_y, a, b, theta, pupil_mask = util.detect_pupil(clear_pupils[beam_id], sigma=2, threshold=0.5, plot=False, savepath=None)


    # for beam_id in args.beam_id:
    #     message = f"read BMX{beam_id}"
    #     initial_Xpos = float(send_and_get_response(message))

    #     message = f"read BMY{beam_id}"
    #     initial_Ypos = float(send_and_get_response(message))
        
    #     # definition is [X,Y]
    #     initial_pos[beam_id] = [initial_Xpos, initial_Ypos]


    secondary_filter = util.get_secondary_mask(pupil_mask, (center_x, center_y))

    # filter edge of pupil and out radii limit for the strehl mask 
    pupil_edge_filter = util.filter_exterior_annulus(pupil_mask, inner_radius=7, outer_radius=100) # to limit pupil edge pixels
    pupil_limit_filter = ~util.filter_exterior_annulus(pupil_mask, inner_radius=11, outer_radius=100) # to limit far out pixel

    #lobe_threshold = 0.1 # percentage of mean clear pupil interior. Absolute values above this in the exterior pixels are candidates for Strehl pixels 
    #exterior_filter =  ( abs( I0  - N0 )  > lobe_threshold * np.mean( N0[pupil_mask] )  ) * (~pupil_mask) * pupil_edge_filter 
    
    # to be more aggressive we can remove ~pupil_mask in filter
    exterior_filter =  ( abs( zwfs_pupils[beam_id]  - clear_pupils[beam_id] ) > args.lobe_threshold  * np.mean( clear_pupils[beam_id][pupil_mask] ) ) * (~pupil_mask) * pupil_edge_filter * pupil_limit_filter

    exterior_filter_dict[beam_id] = exterior_filter 
    secondary_filter_dict[beam_id] = secondary_filter
    # write to toml 
    ## Eventually this exterior filter should be phasemask dependant (maybe).. lets see how operates! 
    # Note we also define this roughly in pupil_registration script 
    # We do not make these pixels phasemask specific!!!
    new_data = {
            f"beam{beam_id}": {
                "pupil_mask": {
                    "exterior": exterior_filter.astype(int).tolist(),
                    "secondary": secondary_filter.astype(int).tolist(), 
                }
            }
        }

    # Check if file exists; if so, load and update.
    if os.path.exists(args.toml_file.replace('#',f'{beam_id}')):
        try:
            current_data = toml.load(args.toml_file.replace('#',f'{beam_id}'))
        except Exception as e:
            print(f"Error loading TOML file: {e}")
            current_data = {}
    else:
        current_data = {}

    # Update current data with new_data (beam specific)
    #current_data.update(new_data)
    current_data = util.recursive_update(current_data, new_data)

    # Write the updated data back to the TOML file.
    with open(args.toml_file.replace('#',f'{beam_id}'), "w") as f:
        toml.dump(current_data, f)


print("returning back to prior camera settings")
# c_dict[args.beam_id[0]].send_fli_cmd(f"set fps {fps0}")
# time.sleep(1)
# c_dict[args.beam_id[0]].send_fli_cmd(f"set gain {gain0}")
# time.sleep(1)
c.send_fli_cmd(f"set fps {fps0}")
time.sleep(1)
c.send_fli_cmd(f"set gain {gain0}")
time.sleep(1)

print('saving output figure')

try:
    for beam_id in args.beam_id:
        if args.fig_path is None:
            savepath=f"delme{beam_id}.png"
        else: # we save with default name at fig path 
            savepath=args.fig_path + f'strehl_pixel_filter{beam_id}.png'

        print(f"saving figure at : {savepath}")
        
        plot_strehl_pixel_registration( data = np.array( zwfs_pupils[beam_id] ) - np.array( clear_pupils[beam_id] ),  
                                       exterior_filter=exterior_filter_dict[beam_id], 
                                       secondary_filter=secondary_filter_dict[beam_id], 
                                       savefig = savepath )

        plt.close("all")

except Exception as e:
    print(f"failed to produce plots : {e}")


print("closing camera and DM SHM objects")

c.close(erase_file=False)
for beam_id in args.beam_id:
    #c_dict[beam_id].close(erase_file=False)
    
    dm_shm_dict[beam_id].close(erase_file=False)

