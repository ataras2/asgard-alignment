
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
from scipy.ndimage import median_filter
from xaosim.shmlib import shm
import asgard_alignment.controllino as co # for turning on / off source 
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


class dmclass():
    def __init__(self, beam_id, shape_wdir=''):
        
        beam_id = int(beam_id)

        assert beam_id in [1,2,3,4]
        # beam number 
        self.beam_id = beam_id
        # where DM shapes are kept
        self.shape_wdir = shape_wdir 
        # sub channels shared memory 
        self.shmfs = np.sort(glob.glob(f"/dev/shm/dm{beam_id}disp*.im.shm"))
        #combined channels 
        self.shmf0 = f"/dev/shm/dm{beam_id}.im.shm"
        # number of sub channels
        self.nch = len(self.shmfs)
        # actual shared memory objects 
        self.shms = []
        for ii in range(self.nch):
            self.shms.append(shm(self.shmfs[ii]))
            print(f"added: {self.shmfs[ii]}") 
        #actual combined shared memory 
        if self.nch != 0:
            self.shm0 = shm(self.shmf0)
        else:
            print("Shared memory structures unavailable. DM server started?")
            
            

    def select_flat_cmd(self,  wdir='DMShapes'):
        '''Matches a DM flat command file to a DM id #.

        Returns the name of the file in the work directory.
        '''
        flat_cmd_files = {
                        "1":"17DW019#113_FLAT_MAP_COMMANDS.txt",
                        "2":"17DW019#053_FLAT_MAP_COMMANDS.txt",
                        "3":"17DW019#093_FLAT_MAP_COMMANDS.txt",
                        "4":"17DW019#122_FLAT_MAP_COMMANDS.txt"
                        }
        
        return wdir + '/' + flat_cmd_files[f"{self.beam_id}"]

    def cmd_2_map2D(self, cmd, fill=np.nan):
        '''Convert a 140 cmd into a 2D DM map for display.

        shm set_data method requires 2D 144 array?

        Just need to add the four corners (0 or nan) and reshape
        Parameters:
        - cmd  : 1D numpy array of 139 components
        - fill : filling values for corners (default = np.nan)
        '''
        return np.insert(cmd, [0, 10, 130, 140], fill).reshape((12, 12))

    def activate_flat(self):
        """
        convention to apply flat command on channel 0!
        """
        if self.nch == 0:
            return
        wdir = "/home/asg/Progs/repos/asgard-alignment/DMShapes/" #os.path.dirname(__file__)
        flat_cmd = np.loadtxt(self.select_flat_cmd( wdir))
        self.shms[0].set_data(self.cmd_2_map2D(flat_cmd, fill=0.0))


    def activate_cross(self, amp=0.1):
        """
        convention to apply calibration shapes on channel 1!
        """
        dms=12
        ii0 = dms // 2 - 1 
        cross_cmd = np.zeros((dms,dms))
        cross_cmd[ii0:ii0+2, :] = amp
        cross_cmd[:, ii0:ii0+2] = amp
        self.shms[1].set_data(cross_cmd)

    def apply_modes(self, amplitude_list, basis_list):
        """
        convention to apply DM modes on channel 2!
        amplitude_list is list of amplitudes to be applied to each mode in basis_list
        amplitude_list must be same lengthh as basis_list
        applies the amplitude weighted sum of modes to DM on shm channel 2 
        """        
        cmd = np.sum( [ aa * MM for aa, MM in zip(amplitude_list, basis_list)])
        self.shms[2].set_data(cmd)


    def set_data(self, cmd):
        """
        convention to apply any user specific commands on channel 2!
        """
        self.shms[2].set_data(cmd)

    def zero_all(self):
        cmd = np.zeros(144)
        for ii, ss in enumerate(self.shms):
            ss.set_data(cmd)
            print(f"zero'd {self.shmfs[ii]}")

    def closeEvent(self, event):
        # freeing all shared memory structures
        for ii in range(self.nch):
            self.shms[ii].close(erase_file=False)
        for ii in range(self.nch):
            self.shms.pop(0)
        print("end of program")

        sys.exit()




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


# poking DM and getting images 
for act in dm_4_corners:
    print(f"actuator {act}")
    img_list_push = [[] for _ in args.beam_id]
    img_list_pull = [[] for _ in args.beam_id]
    poke_vector = np.zeros(144)
    for nn in range(number_of_pokes):
        poke_vector[act] = (-1)**nn * poke_amplitude
        # send DM commands 
        for ii, dm in enumerate( dm_shm_dict.values() ):
            dm.set_data( current_cmd_list[ii] + poke_vector )
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
