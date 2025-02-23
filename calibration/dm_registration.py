
# script_dir = os.path.dirname(os.path.abspath(__file__))
# sys.path.append(script_dir)
import numpy as np
import time 
import glob
import sys
import os 
import toml
import json
import argparse
import matplotlib.pyplot as plt
from scipy.ndimage import median_filter
from xaosim.shmlib import shm
import asgard_alignment.controllino as co # for turning on / off source \
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

#################################
# This script is to calibrate the DM actuator registration in pixel space. 
# It generates a bilinear interpolation matrix for each beam to project 
# intensities on rach DM actuator. 
# calibration is done by applying push pull commands on DM corner actuators,
# fitting an interpolated gaussian to the mean region of peak influence in the 
# image for each actuator, and then finding the intersection between the 
# imterpolated image peaks and solving the affine transform matrix.
# By temporal modulation this method can also be used on sky.




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

default_toml = os.path.join( "config_files", "baldr_config.toml") #os.path.dirname(os.path.abspath(__file__)), "..", "config_files", "baldr_config.toml")
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
    default=[1, 2, 3, 4],
    help="Comma-separated beam IDs to apply. Default: 1,2,3,4"
)

# Plot: default is True, with an option to disable.
parser.add_argument(
    "--plot", 
    dest="plot",
    action="store_true",
    default=True,
    help="Enable plotting (default: True)"
)


args=parser.parse_args()






# inputs 
number_of_pokes = 100 
poke_amplitude = 0.02
dm_4_corners = DM_registration.get_inner_square_indices(outer_size=12, inner_offset=4) # flattened index of the DM actuator 
dm_turbulence = False # roll phasescreen on DM?
#all_dm_shms_list = [args.dm1_shm, args.dm2_shm, args.dm3_shm, args.dm4_shm]

assert hasattr(args.beam_id , "__len__")
assert len(args.beam_id) <= 4
assert max(args.beam_id) <= 4
assert min(args.beam_id) >= 1 

with open(args.toml_file ) as file:
    pupildata = toml.load(file)

    # Extract the "baldr_pupils" section
    baldr_pupils = pupildata.get("baldr_pupils", {})


# global camera image shm 
c = shm(args.global_camera_shm)

# DMs
dm_shm_dict = {}
for beam_id in args.beam_id:
    dm_shm_dict[beam_id] = dmclass( beam_id=beam_id )
    # zero all channels
    dm_shm_dict[beam_id].zero_all()
    # activate flat 
    dm_shm_dict[beam_id].activate_flat()


# try get dark and build bad pixel mask 
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


# setup
#assert len( dm_shm_dict ) == len( camera_shm_list )
disturbance = np.zeros(144) #[np.zeros(140) for _ in DM_list]

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



# poking DM and getting images 
for act in dm_4_corners: # 4 corner indicies are in 140 length vector (not 144 2D map)
    print(f"actuator {act}")
    img_list_push = [[] for _ in args.beam_id]
    img_list_pull = [[] for _ in args.beam_id]
    poke_vector = np.zeros(140) # 4 corner indicies are in 140 length vector (not 144 2D map)
    for nn in range(number_of_pokes):
        poke_vector[act] = (-1)**nn * poke_amplitude
        # send DM commands 
        for ii, dm in enumerate( dm_shm_dict.values() ):
            dm.set_data( current_cmd_list[ii] + dm.cmd_2_map2D(poke_vector, fill=0) )
        time.sleep(0.05)
        # get the images 
        img = np.mean( c.get_data() , axis=0)
        for ii, bb in enumerate( args.beam_id ):
            r1,r2,c1,c2 = baldr_pupils[f"{bb}"]
            cropped_img = interpolate_bad_pixels(img[r1:r2, c1:c2], bad_pixel_mask[r1:r2, c1:c2])

            if np.mod(nn,2):
                img_list_push[ii].append(  cropped_img  )
            else:
                img_list_pull[ii].append( cropped_img )

        if dm_turbulence: 
            # current_cmd_list
            # roll dm screen 
            print('to do')

    for ii, _ in enumerate( args.beam_id):
        delta_img = abs( np.mean(img_list_push[ii],axis=0) - np.mean(img_list_pull[ii],axis=0) )
        # the mean difference in images from push/pulls on the current actuator
        img_4_corners[ii].append( delta_img ) 


# Calibrating coordinate transforms 
dict2write={}
for ii, bb in enumerate( dm_shm_dict ):

    #calibraate the affine transform between DM and camera pixel coordinate systems
    transform_dicts.append( DM_registration.calibrate_transform_between_DM_and_image( dm_4_corners, img_4_corners[ii] , debug=True, fig_path = None  ) )

    # From affine transform construct bilinear interpolation matrix on registered DM actuator positions
    #(image -> actuator transform)
    img = img_4_corners[ii].copy()
    x_target = np.array( [x for x,_ in transform_dicts[ii]['actuator_coord_list_pixel_space']] )
    y_target = np.array( [y for _,y in transform_dicts[ii]['actuator_coord_list_pixel_space']] )
    x_grid = np.arange(img.shape[0])
    y_grid = np.arange(img.shape[1])
    M = DM_registration.construct_bilinear_interpolation_matrix(image_shape=img.shape, 
                                            x_grid=x_grid, 
                                            y_grid=y_grid, 
                                            x_target=x_target,
                                            y_target=y_target)

    bilin_interp_matricies.append( M )

    dict2write[f"beam{bb}"] = {"I2M":M}

# Check if file exists; if so, load and update.
if os.path.exists(args.toml_file):
    try:
        current_data = toml.load(args.toml_file)
    except Exception as e:
        print(f"Error loading TOML file: {e}")
        current_data = {}
else:
    current_data = {}

current_data.update(dict2write)

with open(args.toml_file, "w") as f:
    toml.dump(current_data, f)
