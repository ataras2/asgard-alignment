import numpy as np
import glob 
from astropy.io import fits
import time
import os 
import matplotlib.pyplot as plt 
import importlib
#import rtc
import sys
import datetime
sys.path.append('pyBaldr/' )  
sys.path.insert(1, '/opt/FirstLightImaging/FliSdk/Python/demo/')
sys.path.insert(1,'/opt/Boston Micromachines/lib/Python3/site-packages/')

from pyBaldr import utilities as util
from pyBaldr import ZWFS
from pyBaldr import phase_control
from pyBaldr import pupil_control

from playground import phasemask_centering_tool 

import bmc
import FliSdk_V2
from zaber_motion.ascii import Connection
from asgard_alignment.ZaberMotor import BaldrPhaseMask, LAC10AT4A,  BifrostDichroic, SourceSelection

trouble_shooting_dict = {
    #format:
    'short error key' :
    {
        'error string' : 'longer string descibing error',
        'fix': 'how to fix it '
    },
    'SerialPortBusyException' : 
    {
        'error string':"SerialPortBusyException: SerialPortBusyException: Cannot open serial port: Port is likely already opened by another application.",
        'fix':"You can check if any processes are using the serial port with the following command: lsof /dev/*name* (e.g. name=ttyUSB0).\nIf you found a process using the port from the previous step, you can terminate it with: sudo kill -9 <PID> "
    }

}


def print_current_state(full_report=False):
    if full_report:
        print(f'source motor: \n   {source_selection.device}')
        print(f'    -available sources: {source_selection.sources}')
        print(f'    -current position: {source_selection.current_position}')
        for d in dichroics:
            print(f'dichroic motor:\n   {d.device}')
            print(f'    -available dichroic positions: {d.dichroics}' )
            print(f'    -current position: {d.current_dichroic}')
        print('availabel phasemask positions: ', )
        print(f' phasemask motors: \n   {phasemask.motors}')
        print(f'    -available positions:')
        for l, p in phasemask.phase_positions.items():
            print(f'   {l, p}')
        print(f'    -current position: {phasemask.get_position()}um')
        print(f'focus motor:\n   {focus_motor}')
        print(f'    -current position: {focus_motor.get_position()}um')

    else:
        print(f'source motor {source_selection.device} current position: {source_selection.current_position}')
        for d in dichroics:
            print(f'dichroic motor {d.device} current position: {d.current_dichroic}')
        print(f'phasemask name: {phasemask_name}')
        print(f'phasemask motors {phasemask.motors["x"].axis} current position: {phasemask.get_position()[0]}um')
        print(f'phasemask motors {phasemask.motors["y"].axis} current position: {phasemask.get_position()[1]}um')
        print(f'focus motor {focus_motor.axis} current position: {focus_motor.get_position()}um')



def exit_all():
    # close things 
    try:
        con.close() #"192.168.1.111"
    except:
        print('no "con" to close')
    try:
        connection.close() # "/dev/ttyUSB0"
    except:
        print('no "connection" to close')
    try:
        zwfs.exit_dm() # DM 
    except:
        print('no DM to close')
    try:
        zwfs.exit_camera() #camera
    except:
        print('no camera to close')

        
# timestamp
tstamp = datetime.datetime.now().strftime("%d-%m-%YT%H.%M.%S")


fig_path = f'tmp/{tstamp.split("T")[0]}/' #'/home/baldr/Documents/baldr/ANU_demo_scripts/BALDR/figures/' 
data_path = f'tmp/{tstamp.split("T")[0]}/' #'/home/baldr/Documents/baldr/ANU_demo_scripts/BALDR/data/' 


if not os.path.exists(fig_path):
   os.makedirs(fig_path)


# =====================
#   SETUP
# =====================
# ====== hardware variables
beam = 3
phasemask_name = 'J3'
phasemask_OUT_offset = [1000,1000]  # relative offset (um) to take phasemask out of beam
BFO_pos = 4000 # um (absolute position of detector imgaging lens) 
dichroic_name = "J"
source_name = 'SBB'
DM_serial_number = '17DW019#122' # Syd = '17DW019#122', ANU = '17DW019#053'


# ======  set up source 

# start with source out !

# ======  set up dichroic 

# do manually (COM3 communication issue)

#  ConnectionFailedException: ConnectionFailedException: Cannot open serial port: no such file or directory

connection =  Connection.open_serial_port("/dev/ttyUSB0")
connection.enable_alerts()

device_list = connection.detect_devices()
print("Found {} devices".format(len(device_list)))

dichroics = []
source_selection = None
for dev in device_list:
    if dev.name == "X-LSM150A-SE03":
        dichroics.append(BifrostDichroic(dev))
    elif dev.name == "X-LHM100A-SE03":
        source_selection = SourceSelection(dev)
print(f"Found {len(dichroics)} dichroics")
if source_selection is not None:
    print("Found source selection")

for dichroic in dichroics:
    dichroic.set_dichroic("J")

while dichroics[0].get_dichroic() != "J":
    pass

# ====== set up phasemask
con = Connection.open_tcp("192.168.1.111")
print("Found {} devices".format(len(con.detect_devices())))
x_axis = con.get_device(1).get_axis(1)
y_axis = con.get_device(1).get_axis(3)

# get most recent positions file
maskpos_files = glob.glob( f"phase_positions_beam_{beam}*.json")
latest_maskpos_file = max(maskpos_files, key=os.path.getctime)
phasemask = BaldrPhaseMask(
    LAC10AT4A(x_axis), LAC10AT4A(y_axis), latest_maskpos_file 
)
""" 
# e.g: to update position and write to file 
phasemask.move_absolute( [3346, 1205])
phasemask.update_mask_position( 'J3' )
phasemask.write_current_mask_positions() 
"""

# ====== set up focus 
focus_axis = con.get_device(1).get_axis(2)
focus_motor = LAC10AT4A(focus_axis)
focus_motor.move_absolute( BFO_pos )

# print out motors we have 

print_current_state()
# ====== Set up and calibrate 

debug = True # plot some intermediate results 

# take out source to calibate 
source_selection.set_source(  'none' )
time.sleep(1)
focus_motor.move_absolute( BFO_pos )
time.sleep(1)
phasemask.move_to_mask(phasemask_name) 
time.sleep(1)
dichroic.set_dichroic("J")
time.sleep(1)


pupil_crop_region = [204,268,125, 187] #[None, None, None, None] #[204 -50 ,268+50,125-50, 187+50] 

#init our ZWFS (object that interacts with camera and DM) (old path = home/baldr/Documents/baldr/ANU_demo_scripts/BALDR/)
zwfs = ZWFS.ZWFS(DM_serial_number=DM_serial_number, cameraIndex=0, DMshapes_path = 'DMShapes/', pupil_crop_region=pupil_crop_region ) 

# the sydney BMC multi-3.5 calibrated flat seems shit! Try with just a 

zwfs.set_camera_dit( 0.001 );time.sleep(0.2)
zwfs.set_camera_fps( 200 );time.sleep(0.2)
zwfs.set_sensitivity('high');time.sleep(0.2)
zwfs.enable_frame_tag(tag = True);time.sleep(0.2)
zwfs.bias_off();time.sleep(0.2)
zwfs.flat_off();time.sleep(0.2)

# trying different DM flat 
#zwfs.dm_shapes['flat_dm'] = 0.5 * np.ones(140)

zwfs.start_camera()

## ------- Calibrate detector (dark, badpixels)
# Source should be out
# at sydney move 01 X-LSM150A-SE03 to 133.07mm
zwfs.build_manual_dark()

# get our bad pixels 
bad_pixels = zwfs.get_bad_pixel_indicies( no_frames = 1000, std_threshold = 50 , flatten=False)

# update zwfs bad pixel mask and flattened pixel values 
zwfs.build_bad_pixel_mask( bad_pixels , set_bad_pixels_to = 0 )

## ------- move source back in 
source_selection.set_source(  source_name )
time.sleep(2)

# quick check that dark subtraction works and we have signal
I0 = zwfs.get_image( apply_manual_reduction  = True)
plt.figure(); plt.title('test image \nwith dark subtraction \nand bad pixel mask'); plt.imshow( I0 ); plt.colorbar()
plt.savefig( fig_path + 'delme.png')
plt.close()

print_current_state()


## ------- Checking centering 

# ensure flat DM 
zwfs.dm.send_data(zwfs.dm_shapes['flat_dm'])

# == manual search 

phasemask_centering_tool.move_relative_and_get_image(zwfs, phasemask, savefigName=fig_path + 'delme.png')
# if you like the improvement 
# phasemask.update_mask_position( phasemask_name )
# phasemask.write_current_mask_positions()

# == automatic search 
"""
phasemask_diameter = 50 # um <- have to ensure grid is at this resolution 
search_radius = 100  # search radius for spiral search (um)
dtheta = np.pi / 20  # angular increment (rad) 
iterations_per_circle = 2*np.pi / dtheta
dr = phasemask_diameter / iterations_per_circle # cover 1 phasemask diameter per circle

# move off phase mask, its good to make sure zwfs object has dark, bad pixel map etc first to see better
phasemask.move_to_mask( phasemask_name )
phasemask.move_relative( [1000,1000] )  # 1mm in each axis
time.sleep(1.2)
reference_img =  np.mean(zwfs.get_some_frames(number_of_frames = 10, apply_manual_reduction = True ) , axis=0 ) # Capture reference image when misaligned
phasemask.move_to_mask( phasemask_name )  # move back to initial_position 
time.sleep(1.2)

# look at savefigName png image while running to see updates ()
phasemask_centering_tool.spiral_search_and_center(zwfs, phasemask, phasemask_name, search_radius, dr, dtheta, \
    reference_img, fine_tune_threshold=2, savefigName='tmp/delme.png', usr_input=True)

"""

#init our pupil controller (object that processes ZWFS images and outputs VCM commands)
pupil_ctrl = pupil_control.pupil_controller_1(config_file = None)

#analyse pupil and decide if it is ok. This must be done before reconstructor
pupil_report = pupil_control.analyse_pupil_openloop( zwfs, debug = True, return_report = True)

if pupil_report['pupil_quality_flag'] == 1: 
    zwfs.update_reference_regions_in_img( pupil_report ) # 


# x,y in compass referenced to DM right (+x), up (+y)
I0, N0 = util.get_reference_images(zwfs, phasemask, theta_degrees=11.8, number_of_frames=256, \
compass = True, compass_origin=None, savefig=fig_path + f'FPM-in-out_{phasemask_name}.png' )



# =====================
#   OPTIMIZING I0 - INTENSITY REFERENCE WITH PHASEMASK INSERTED IN BEAM
# =====================



import numpy as np
from scipy.optimize import minimize

# Define your objective function
def objective_function(delta_cmd, zwfs, pupil_mask):
    # Send the current command to the deformable mirror
    zwfs.dm.send_data(zwfs.dm.send_data(zwfs.dm_shapes['flat_dm']) + delta_cmd)
    
    # Get the new image and average it
    image_raw = np.mean(zwfs.get_some_images(), axis=0)
    # normalize the image before filtering for the pupil
    image = ( image_raw - np.mean(image_raw) )/ np.std(image_raw )

    # Mask the pupil region
    pupil_pixels = image[pupil_mask]
    
    # Calculate the sum and variance of the intensity within the pupil
    sum_intensity = np.sum(pupil_pixels)
    variance_intensity = np.var(pupil_pixels)
    
    # Define the cost function: we want to maximize sum_intensity and minimize variance_intensity
    # Since minimize is the objective, we can minimize -sum_intensity (to maximize it) and variance_intensity
    cost = -sum_intensity + alpha * variance_intensity
    
    return cost

# Initial guess for the DM command (can start with zeros, random, or previous good result)
initial_cmd = np.random.uniform(0, 1, 140)

# Define the bounds for the DM command (between 0 and 1 for each element)
bounds = [(0, 1) for _ in range(140)]

# Optimization settings
alpha = 1.0  # Weighting factor for the variance term in the cost function

# Define a mask to select only the pupil pixels
# Assuming you already have a pupil mask defined as a boolean array of the same shape as the image
pupil_mask = ...  # Your code to define this mask

# Perform the optimization
result = minimize(objective_function, initial_cmd, args=(zwfs,pupil_mask,), bounds=bounds, method='L-BFGS-B')

# The result contains the optimized DM command
optimal_cmd = result.x

# Send the optimal command to the DM
zwfs.dm.send_data(optimal_cmd)
