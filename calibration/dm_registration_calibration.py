
# script_dir = os.path.dirname(os.path.abspath(__file__))
# sys.path.append(script_dir)
import numpy as np
import time 
import zmq
import glob
import sys
import os 
import toml
import datetime
import json
import argparse
from astropy.io import fits
import subprocess
import matplotlib.pyplot as plt
from scipy.ndimage import median_filter
from xaosim.shmlib import shm
import asgard_alignment.controllino as co # for turning on / off source \
from asgard_alignment.DM_shm_ctrl import dmclass
import common.DM_registration as DM_registration
import common.DM_basis_functions as dmbases
import common.phasemask_centering_tool as pct
from asgard_alignment import FLI_Cameras as FLI
from pyBaldr import utilities as util

# try:
#     from asgard_alignment import controllino as co
#     myco = co.Controllino('172.16.8.200')
#     controllino_available = True
#     print('controllino connected')
    
# except:
#     print('WARNING Controllino cannot connect. WILL NOT MOVE SOURCE OUT FOR DARK')
#     controllino_available = False 

#################################
# This script is to calibrate the DM actuator registration in pixel space. 
# It generates a bilinear interpolation matrix for each beam to project 
# intensities on rach DM actuator. 
# calibration is done by applying push pull commands on DM corner actuators,
# fitting an interpolated gaussian to the mean region of peak influence in the 
# image for each actuator, and then finding the intersection between the 
# imterpolated image peaks and solving the affine transform matrix.
# By temporal modulation this method can also be used on sky.



def recursive_update(orig, new):
    """
    Recursively update dictionary 'orig' with 'new' without overwriting sub-dictionaries.
    """
    for key, value in new.items():
        if (key in orig and isinstance(orig[key], dict) 
            and isinstance(value, dict)):
            recursive_update(orig[key], value)
        else:
            orig[key] = value
    return orig


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


parser = argparse.ArgumentParser(description="Baldr Pupil Fit Configuration.")

default_toml = os.path.join( "config_files", "baldr_config_#.toml") #os.path.dirname(os.path.abspath(__file__)), "..", "config_files", "baldr_config.toml")

# setting up socket to ZMQ communication to multi device server
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
    default=[4],
    help="Comma-separated beam IDs to apply. Default: 1,2,3,4"
)
parser.add_argument(
    "--use_baldr_flat", 
    action="store_false",
    default=True,
    help="calibrate the Baldr flat starting with the current baldr flat. If False we beging with the BMC factory flat"
)
# Plot: default is True, with an option to disable.
parser.add_argument(
    "--plot", 
    dest="plot",
    action="store_true",
    default=True,
    help="Enable plotting (default: True)"
)


parser.add_argument("--fig_path", 
                    type=str, 
                    default='', 
                    help="path/to/output/image/ for where the saved figures are (DM_registration_in_pixel_space.png)")


args=parser.parse_args()






# inputs 
number_of_pokes = 8 
poke_amplitude = 0.05
sleeptime = 0.5 #10 is very safe
dm_4_corners = DM_registration.get_inner_square_indices(outer_size=12, inner_offset=4) # flattened index of the DM actuator 
dm_turbulence = False # roll phasescreen on DM?
#all_dm_shms_list = [args.dm1_shm, args.dm2_shm, args.dm3_shm, args.dm4_shm]

assert hasattr(args.beam_id , "__len__")
assert len(args.beam_id) <= 4
assert max(args.beam_id) <= 4
assert min(args.beam_id) >= 1 

pupil_mask = {}
for beam_id in args.beam_id:
    with open(args.toml_file.replace('#',f'{beam_id}') ) as file:
        config_dict = toml.load(file)

        # Extract the "baldr_pupils" section
        baldr_pupils = config_dict.get("baldr_pupils", {})

        # get the pupil mask (we only consider pixels within here for the DM calibration)
        pupil_mask[beam_id] = config_dict.get(f"beam{beam_id}", {}).get("pupil_mask", {}).get("mask", None)

# global camera image shm - we do this to make it easier to run for multiple beams interms of changing settings etc
c = FLI.fli(args.global_camera_shm, roi=[None,None,None,None]) #shm(args.global_camera_shm)
# We read them in now! 
# c.build_manual_bias(number_of_frames=500)
# c.build_manual_dark(no_frames = 500 , build_bad_pixel_mask=True, kwargs={'std_threshold':10, 'mean_threshold':6} )

# we get the current settings to return to prior
fps0 = FLI.extract_value( c.send_fli_cmd( "fps raw" ) ) 
gain0 = FLI.extract_value( c.send_fli_cmd( "gain raw" ) ) 

### HARD CODED GAIN AND FPS FOR THIS CALIBRATION 
hc_fps = 200 # Hz 
hc_gain = 1 # 

c.send_fli_cmd(f"set fps {hc_fps}")
time.sleep(1)
c.send_fli_cmd(f"set gain {hc_gain}")
time.sleep(1)

# after changing ensure that everything is ok (updated fine in the camera local config)
assert float( c.config['gain'] ) == float(hc_gain )
assert float( c.config['fps']) == float( hc_fps )


# r1,r2,c1,c2 = baldr_pupils[f"{beam_id}"]
# util.nice_heatmap_subplots( [c.reduction_dict['dark'][-1][r1:r2,c1:c2], c.reduction_dict['bias'][-1][r1:r2,c1:c2], c.reduction_dict['bad_pixel_mask'][-1][r1:r2,c1:c2]], savefig = 'delme.png')


valid_cal_files = util.find_calibration_files(mode=c.config['mode'], gain=int(c.config['gain']), target_fps=float(c.config['fps']), base_dir="MASTER_DARK", time_diff_thresh=datetime.timedelta(2), fps_diff_thresh=10)


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


# get darks, bias and some raw darks to make bad pixels (we dont use the premade ones cause we adjust here the parameters)
bias_fits_files = util.find_calibration_files(mode=c.config['mode'], gain=int(c.config['gain']) , target_fps=float(c.config['fps']), base_dir="MASTER_BIAS", time_diff_thresh=datetime.timedelta(2), fps_diff_thresh=10) #glob.glob(f"/home/asg/Progs/repos/dcs/calibration_frames/products/{tstamp_rough}/MASTER_BIAS/*.fits") 
dark_fits_files = util.find_calibration_files(mode=c.config['mode'], gain=int(c.config['gain']) , target_fps=float(c.config['fps']), base_dir="MASTER_DARK", time_diff_thresh=datetime.timedelta(2), fps_diff_thresh=10) #glob.glob(f"/home/asg/Progs/repos/dcs/calibration_frames/products/{tstamp_rough}/MASTER_DARK/*.fits") 
raw_darks_files =  util.find_calibration_files(mode=c.config['mode'],gain=int(c.config['gain']) , target_fps=float(c.config['fps']), base_dir="RAW_DARKS", time_diff_thresh=datetime.timedelta(2), fps_diff_thresh=10) #glob.glob(f"/home/asg/Progs/repos/dcs/calibration_frames/products/{tstamp_rough}/RAW_DARKS/*.fits") 

reduction_dict = {} # to hold the files used for reduction 

for lab, ff in zip(['bias','dark'], [bias_fits_files, dark_fits_files] ):
    # Assumes we just took one!!! would be quicker to check subdirectories for one that matches the mode and gain with nearest fps. 
    most_recent = max(ff, key=os.path.getmtime) 
    reduction_dict[lab] = most_recent
    with fits.open( most_recent ) as d:
        c.reduction_dict[lab].append(  d[0].data.astype(int) )       # for beam_id in args.beam_id:
        #     r1,r2,c1,c2 = baldr_pupils[f'{beam_id}']
        #     c_dict[beam_id].reduction_dict[lab].append(  d[0].data.astype(int)[r1:r2, c1:c2] )

# bad pixels 
most_recent = max(raw_darks_files , key=os.path.getmtime) 
reduction_dict["raw_darks"] = most_recent

with fits.open( most_recent ) as d:

    bad_pixels, bad_pixel_mask = FLI.get_bad_pixels( d[0].data, std_threshold=4, mean_threshold=10)
    bad_pixel_mask[0][0] = False # the frame tag should not be masked! 
    #c_dict[beam_id].reduction_dict['bad_pixel_mask'].append(  (~bad_pixel_mask ).astype(int) )
    c.reduction_dict['bad_pixel_mask'].append(  (~bad_pixel_mask ).astype(int) )

# r1,r2,c1,c2 = baldr_pupils[f"{beam_id}"]
# util.nice_heatmap_subplots( [c.reduction_dict['dark'][-1][r1:r2,c1:c2], c.reduction_dict['bias'][-1][r1:r2,c1:c2], c.reduction_dict['bad_pixel_mask'][-1][r1:r2,c1:c2]], savefig = 'delme.png')







####################################################################################
####################################################################################
#### DELETE THIS LATER (30/3/25 - only due to ron on chns 32)
# bad_ron = np.ones_like( c.get_image() ).astype(bool)
# bad_ron[:, ::32 ] = False
# bad_ron[:, 1::32 ] = False
# bad_ron[:, 2::32 ] = False
# bad_ron[:, 3::32 ] = False
# plt.figure();plt.imshow( bad_ron) ;plt.colorbar() ; plt.savefig('delme.png')

#c.reduction_dict['bad_pixel_mask'][-1] *= bad_ron
####################################################################################
####################################################################################

# DMs
dm_shm_dict = {}
for beam_id in args.beam_id:
    dm_shm_dict[beam_id] = dmclass( beam_id=beam_id )
    # zero all channels
    dm_shm_dict[beam_id].zero_all()
    # activate flat 
    #dm_shm_dict[beam_id].activate_flat()
    # apply DM flat offset 
    if not args.use_baldr_flat:
        dm_shm_dict[beam_id].activate_flat()
    else:
        dm_shm_dict[beam_id].activate_calibrated_flat()


# # try get dark and build bad pixel mask 
# if controllino_available:
    
#     myco.turn_off("SBB")
#     time.sleep(2)
    
#     dark_raw = c.get_data()

#     myco.turn_on("SBB")
#     time.sleep(2)

#     bad_pixel_mask = get_bad_pixel_indicies( dark_raw, std_threshold = 20, mean_threshold=6)
# else:
#     dark_raw = c.get_data()

#     bad_pixel_mask = get_bad_pixel_indicies( dark_raw, std_threshold = 20, mean_threshold=6)


# setup
#assert len( dm_shm_dict ) == len( camera_shm_list )
#disturbance = np.zeros(144) #[np.zeros(140) for _ in DM_list]

# incase we want to test this with dynamic dm cmds (e.g phasescreen)
current_cmd_list = [ np.zeros(144)  for _ in args.beam_id]
img_4_corners  = [[] for _ in args.beam_id] 
transform_dicts = []
bilin_interp_matricies = []



##################
# some sanity checks that we can see things
###################
# poke_vector = np.zeros(140)
# act=65 #dm_4_corners[0]
# poke_vector[act]=0.2
# ii=2

# dm_shm_dict[ii].set_data( current_cmd_list[ii] )
# time.sleep(2)
# img = np.mean( c.get_data() , axis=0)
# r1,r2,c1,c2 = baldr_pupils[f"{ii}"]
# cropped_img1 = interpolate_bad_pixels(img[r1:r2, c1:c2], bad_pixel_mask[r1:r2, c1:c2])

# dm_shm_dict[ii].set_data( current_cmd_list[ii] + dm_shm_dict[ii].cmd_2_map2D(poke_vector, fill=0).reshape(-1) )
# time.sleep(2)
# img = np.mean( c.get_data() , axis=0)
# r1,r2,c1,c2 = baldr_pupils[f"{ii}"]
# cropped_img0 = interpolate_bad_pixels(img[r1:r2, c1:c2], bad_pixel_mask[r1:r2, c1:c2])


# fig,ax = plt.subplots(2,1)
# ax[0].imshow(cropped_img0-cropped_img1)
# ax[1].imshow( dm_shm_dict[ii].shms[2].get_data() ) #dm_shm_dict[ii].shm0.get_data())
# #dm_shm_dict[ii].cmd_2_map2D(poke_vector, fill=0)) #
# plt.savefig('delme.png')

# dm_shm_dict[ii].set_data( current_cmd_list[ii] )


# # check whats written on all channels
# fig, ax = plt.subplots( len(dm_shm_dict[ii].shms) +1 , 1 )

# #dm_shm_dict[ii].shms[2].set_data(dm_shm_dict[ii].cmd_2_map2D(poke_vector, fill=0).reshape(-1))
# #dm_shm_dict[ii].set_data(dm_shm_dict[ii].cmd_2_map2D(poke_vector, fill=0).reshape(-1))
# dm_shm_dict[ii].set_data( current_cmd_list[ii] + dm_shm_dict[ii].cmd_2_map2D(poke_vector, fill=0).reshape(-1) )
# for i, axx in enumerate( ax.reshape(-1)[:-1]):
#     axx.imshow( dm_shm_dict[ii].shms[i].get_data())
#     axx.set_title( dm_shm_dict[ii].shmfs[i] )

# ax[-1].imshow( dm_shm_dict[ii].shm0.get_data())
# plt.savefig('delme.png')




print(f'GOING VERY SLOW ({sleeptime}s delays) DUE TO SHM DELAY DM')
for act in dm_4_corners: # 4 corner indicies are in 140 length vector (not 144 2D map)
    print(f"actuator {act}")
    img_list_push = [[] for _ in args.beam_id]
    img_list_pull = [[] for _ in args.beam_id]
    poke_vector = np.zeros(140) # 4 corner indicies are in 140 length vector (not 144 2D map)
    for nn in range(number_of_pokes):
        print( f'poke {nn}')
        poke_vector[act] = (-1)**nn * poke_amplitude
        # send DM commands 
        for ii, beam_id in enumerate( args.beam_id):
            dm_shm_dict[beam_id].set_data( dm_shm_dict[beam_id].cmd_2_map2D(poke_vector, fill=0) ) 
            ## Try without #DM_flat_offset[beam_id]  )
        time.sleep(1)
        # get the images 
        img = np.mean( c.get_data( apply_manual_reduction=True) , axis=0)

        ####################################################################################
        ####################################################################################
        #### DELETE THIS LATER (30/3/25 - only due to ron on chns 32)
        #img[~bad_ron] = 0 
        ####################################################################################
        ####################################################################################
        ####################################################################################

        for ii, bb in enumerate( args.beam_id ):
            r1,r2,c1,c2 = baldr_pupils[f"{bb}"]
            cropped_img = img[r1:r2, c1:c2] #interpolate_bad_pixels(img[r1:r2, c1:c2], bad_pixel_mask[r1:r2, c1:c2])

            if np.mod(nn,2):
                img_list_push[ii].append(  cropped_img  )
            else:
                img_list_pull[ii].append( cropped_img )

        if dm_turbulence: 
            # current_cmd_list
            # roll dm screen 
            print('to do for on sky test')

    for ii, beam_id in enumerate( args.beam_id):
        delta_img = abs( np.mean(img_list_push[ii],axis=0) - np.mean(img_list_pull[ii],axis=0) )
        # the mean difference in images from push/pulls on the current actuator
        img_4_corners[ii].append( np.array( pupil_mask[beam_id] ).astype(float) * delta_img ) #  We multiply by the pupil mask to ignore all external pixels! These can be troublesome with hot pixels etc 



for beam_id in args.beam_id:
    # zero all channels
    dm_shm_dict[beam_id].zero_all()
    # activate flat 
    #dm_shm_dict[beam_id].activate_calibrated_flat() #activate_flat()
    if not args.use_baldr_flat:
        dm_shm_dict[beam_id].activate_flat()
    else:
        dm_shm_dict[beam_id].activate_calibrated_flat()



## lets see the registration 
#plt.figure()

#titles = ['pull','push','diff']
#imgs = [np.mean(img_list_pull[ii],axis=0), np.mean(img_list_push[ii],axis=0),np.mean(img_list_push[ii],axis=0) - np.mean(img_list_pull[ii],axis=0)  ]
# titles = [f'corner{i}' for i in [1,2,3,4]]
# imgs = [ii for ii in img_4_corners[ii]]
# util.nice_heatmap_subplots( im_list=imgs, title_list=titles,savefig='delme.png')
# #plt.imshow( np.sum(img_4_corners[0],axis=0))
# # should see four corner dm pokes in the image 
# #plt.savefig('delme.png')

## we do beam specific directory from fig_path
if not os.path.exists( args.fig_path + f"beam{beam_id}/"):
    os.makedirs(  args.fig_path + f"beam{beam_id}/" )

# Calibrating coordinate transforms 
dict2write={}
for ii, beam_id in enumerate( args.beam_id ):

    #calibraate the affine transform between DM and camera pixel coordinate systems
    if args.fig_path is not None:
        savefig = args.fig_path #+ 'DM_registration_in_pixel_space.png'
    else:
        savefig = os.path.expanduser('~/Downloads')

    if not os.path.exists( savefig + f"beam{beam_id}/"):
        os.mkdir( savefig + f"beam{beam_id}/" )

    plt.close() # close any open figiures
    transform_dicts.append( DM_registration.calibrate_transform_between_DM_and_image( dm_4_corners, img_4_corners[ii] , debug=True, fig_path = savefig + f"beam{beam_id}/"  ) )
    plt.close() # close any open figiures

    # From affine transform construct bilinear interpolation matrix on registered DM actuator positions
    #(image -> actuator transform)
    img = img_4_corners[ii][0].copy()
    x_target = np.array( [x for x,_ in transform_dicts[ii]['actuator_coord_list_pixel_space']] )
    y_target = np.array( [y for _,y in transform_dicts[ii]['actuator_coord_list_pixel_space']] )
    x_grid = np.arange(img.shape[0])
    y_grid = np.arange(img.shape[1])
    M = DM_registration.construct_bilinear_interpolation_matrix(image_shape=img.shape, 
                                            x_grid=x_grid, 
                                            y_grid=y_grid, 
                                            x_target=x_target,
                                            y_target=y_target)

    try:
        M @ img.reshape(-1)
    except:
        raise UserWarning("matrix dimensions don't match! ")
    bilin_interp_matricies.append( M )

    # update I2A instead of I2M
    dict2write[f"beam{beam_id}"] = {"I2A":M.tolist()}

    # Check if file exists; if so, load and update.
    if os.path.exists(args.toml_file.replace('#',f'{beam_id}')):
        try:
            current_data = toml.load(args.toml_file.replace('#',f'{beam_id}'))
        except Exception as e:
            print(f"Error loading TOML file: {e}")
            current_data = {}
    else:
        current_data = {}


    current_data = recursive_update(current_data, dict2write)

    with open(args.toml_file.replace('#',f'{beam_id}'), "w") as f:
        toml.dump(current_data, f)




# ## write the json file to keep record of stability 
for ii, beam_id in enumerate( args.beam_id ):

    tstamp = datetime.datetime.now().strftime("%Y-%m-%dT%H-%M-%S")
    path_tmp = f"/home/asg/Progs/repos/asgard-alignment/calibration/cal_data/dm_registration/beam{beam_id}/"
    if not os.path.exists(path_tmp):
        os.makedirs( path_tmp )

    file_tmp = f"dm_reg_beam{beam_id}_{tstamp}.json"
    with open(path_tmp + file_tmp, "w") as json_file:
        json.dump(util.convert_to_serializable(transform_dicts[ii]), json_file)
    print( f"saved dm registration json : {path_tmp + file_tmp}")



print('returning to original camera gain and fps settings')
c.send_fli_cmd(f"set fps {fps0}")
time.sleep(1)
c.send_fli_cmd(f"set gain {gain0}")
time.sleep(1)

# Close everything 
c.close(erase_file=False)

for beam_id in args.beam_id:
    dm_shm_dict[beam_id].close(erase_file=False)

## lets see the registration 
# plt.figure()
# plt.imshow( np.sum(img_4_corners[0],axis=0))
# plt.savefig('delme.png')

# # example to overlay the registered actuators in pixel space with the pupil (FPM out of beam)
# fig = plt.figure()
# img = np.mean( c.get_data() , axis=0)[r1:r2,c1:c2]
# im = plt.imshow( img )
# cbar = fig.colorbar(im, ax=plt.gca(), pad=0.01)
# cbar.set_label(r'Intensity', fontsize=15, labelpad=10)

# plt.scatter(transform_dicts[0]['actuator_coord_list_pixel_space'][:, 0],\
#     transform_dicts[0]['actuator_coord_list_pixel_space'][:, 1], \
#         color='blue', marker='.', label = 'DM actuators')

# plt.legend() 
# savefig = f'pupil_on_DM_in_pixel_space_{beam_id}.png'
# fig.savefig(savefig, dpi=300, bbox_inches = 'tight' )
# plt.show()    




    # stacked_corner_img = []
    # img_corners = []
    # corner_fits = {}
    # for actuator_number, delta_img in zip(dm_4_corners, img_4_corners[ii] ):  # <<< added actuator number 
    #     stacked_corner_img.append( delta_img )

    #     peak_pixels_raw = tuple( np.array( list(np.where( abs(delta_img) == np.max( abs(delta_img) )  ) ) ).ravel() )
                
    #     # fit actuator position in pixel space after interpolation and Gaussian fit 
    #     corner_fit_dict = DM_registration.interpolate_and_fit_gaussian(coord=peak_pixels_raw, radius=5, pixel_values= abs(delta_img), factor=5)
    #     corner_fits[actuator_number] = corner_fit_dict
    #     #plot_fit_results( corner_fit_dict )
        
    #     img_corners.append( ( corner_fit_dict['x0'],  corner_fit_dict['y0'] ) )
    #     # #Check individual registration          
    #     # plt.figure(actuator_number)
    #     # plt.imshow( delta_img )
    #     # plt.plot( corner_fit_dict['x0'],  corner_fit_dict['y0'], 'x', color='red', lw=4, label='registered position') 
    #     # plt.legend()
    #     # plt.colorbar(label='Intensity')
    #     # plt.show() 
    
        
    # #[top_left, top_right, bottom_right, bottom_left]
    # intersection = DM_registration.find_quadrilateral_diagonal_intersection( img_corners ) 

    # fig = plt.figure()
    # im = plt.imshow( np.sum( stacked_corner_img , axis=0 ))
    # cbar = fig.colorbar(im, ax=plt.gca(), pad=0.01)
    # cbar.set_label(r'$\Delta$ Intensity', fontsize=15, labelpad=10)

    # for i,c in  enumerate( img_corners ):
    #     if i==0:
    #         plt.plot( c[0],  c[1], 'x', color='red', lw=4, label='registered position') 
    #     else:
    #         plt.plot( c[0],  c[1], 'x', color='red', lw=4 )


    # top_left, top_right, bottom_right, bottom_left = img_corners

    # # Correct cross diagonal plotting
    # x_cross = [top_left[0], bottom_right[0], None, top_right[0], bottom_left[0]]
    # y_cross = [top_left[1], bottom_right[1], None, top_right[1], bottom_left[1]]

    # plt.plot(x_cross, y_cross, 'ro-', markersize=3, label="Cross Lines")  # Red lines with circle markers

    # #plt.plot(  [bottom_left[0], top_right[0]], [bottom_right[1], top_right[1]] , 'r', lw=1)
    # #plt.plot(  [top_left[0], bottom_right[0]],   [top_left[1], bottom_right[1]] , 'r', lw=1)
    # plt.plot( intersection[0], intersection[1], 'x' ,color='white', lw = 5 )
    # plt.legend()
    # savefig =  'DM_center_in_pixel_space.png'
    # plt.savefig(savefig, dpi=300, bbox_inches = 'tight') 
    # plt.show()





# img = c.get_data() #  full image
# img2 = np.mean( img, axis=0)[r1:r2,c1:c2]

# idm = np.array(M) @ img2.reshape(-1)
# plt.imshow( util.get_DM_command_in_2D(idm) ); plt.savefig('delme.png')


"""
ome rand test 

amp=0.03
imlist = [] 
amps = np.linspace( -0.04, 0.04, 25)
for amp in amps:
    print(amp)
    zbasis = dmbases.zer_bank(1, 10 )
    bb=2
    dm_shm_dict[bb].set_data( amp * zbasis[3] )
    #dm_shm_dict[bb].shms[2].set_data(amp * zbasis[3])
    time.sleep(1)
    img = np.mean( c.get_data() ,axis = 0 ) 
    r1,r2,c1,c2 = baldr_pupils[f"{bb}"]
    imlist.append( img[r1:r2, c1:c2] )


fig,ax = plt.subplots( 5,5, figsize=(15,15))
for i, a, axx in zip( imlist, amps, ax.reshape(-1)):
    axx.imshow(i ) 
    axx.set_title( f'amp={round(a,3)}')
plt.savefig('delme1.png') 

cropped_img = interpolate_bad_pixels(img[r1:r2, c1:c2], bad_pixel_mask[r1:r2, c1:c2])
#plt.figure();plt.imshow(cropped_img);plt.savefig('delme1.png')
plt.figure();plt.imshow(img[r1:r2, c1:c2]);plt.savefig('delme1.png')

plt.figure();plt.imshow(dm_shm_dict[bb].shm0.get_data());plt.savefig('delme1.png')

for bb in [1,2,3,4]:
    dm_shm_dict[bb].zero_all()  
    dm_shm_dict[bb].activate_flat()
"""





# # # get all available files 
# valid_reference_position_files = glob.glob(
#     f"/home/asg/Progs/repos/asgard-alignment/config_files/phasemask_positions/beam{args.beam_id}/*json"
#     )


# # read in the most recent and make initial posiition the most recent one for given mask 
# with open(max(valid_reference_position_files, key=os.path.getmtime)
# , "r") as file:
#     start_position_dict = json.load(file)

# Xpos0 = start_position_dict[phasemask_name][0]
# Ypos0 = start_position_dict[phasemask_name][1]


# search_dict = spiral_square_search_and_save_images(
#     cam=c,
#     beam=2,
#     baldr_pupils=baldr_pupils,
#     phasemask=state_dict["socket"],
#     starting_point='recent',
#     step_size=20,
#     search_radius=200,
#     sleep_time=0.5,
#     use_multideviceserver=True,
# )

# move_relative_and_get_image(cam=c, beam=2,baldr_pupils=baldr_pupils, phasemask=state_dict["socket"], savefigName='delme.png', use_multideviceserver=True)
