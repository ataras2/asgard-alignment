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


def print_current_state():
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

fig_path = 'tmp/' #'/home/baldr/Documents/baldr/ANU_demo_scripts/BALDR/figures/' 
data_path = 'tmp/' #'/home/baldr/Documents/baldr/ANU_demo_scripts/BALDR/data/' 

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


pupil_crop_region = [204,268,125, 187] #[None, None, None, None] #[0, 192, 0, 192] 

#init our ZWFS (object that interacts with camera and DM) (old path = home/baldr/Documents/baldr/ANU_demo_scripts/BALDR/)
zwfs = ZWFS.ZWFS(DM_serial_number=DM_serial_number, cameraIndex=0, DMshapes_path = 'DMShapes/', pupil_crop_region=pupil_crop_region ) 

# the sydney BMC multi-3.5 calibrated flat seems shit! Try with just a 

zwfs.set_camera_dit( 0.001 );time.sleep(0.2)
zwfs.set_camera_fps( 200 );time.sleep(0.2)
zwfs.set_sensitivity('high');time.sleep(0.2)
zwfs.enable_frame_tag(tag = True);time.sleep(0.2)
zwfs.bias_off();time.sleep(0.2)
zwfs.flat_off();time.sleep(0.2)

zwfs.dm_shapes['flat_dm'] = 0.5 * np.ones(140)

zwfs.start_camera()

# !!!! TAKE OUT SOURCE !!!! 
# at sydney move 01 X-LSM150A-SE03 to 133.07mm
zwfs.build_manual_dark()

# get our bad pixels 
bad_pixels = zwfs.get_bad_pixel_indicies( no_frames = 1000, std_threshold = 50 , flatten=False)

# update zwfs bad pixel mask and flattened pixel values 
zwfs.build_bad_pixel_mask( bad_pixels , set_bad_pixels_to = 0 )

# move source back in 
source_selection.set_source(  source_name )
time.sleep(2)

# quick check that dark subtraction works and we have signal
I0 = zwfs.get_image( apply_manual_reduction  = True)
plt.figure(); plt.title('test image \nwith dark subtraction \nand bad pixel mask'); plt.imshow( I0 ); plt.colorbar()
plt.savefig( fig_path + 'delme.png')
plt.close()

print_current_state()

from playground import phasemask_centering_tool as pct


initial_pos= phasemask.phase_positions[phasemask_name] # starting position of phase mask
phasemask_diameter = 50 # um <- have to ensure grid is at this resolution 
search_radius = 100  # search radius for spiral search (um)
dtheta = np.pi / 20  # angular increment (rad) 
iterations_per_circle = 2*np.pi / dtheta
dr = phasemask_diameter / iterations_per_circle # cover 1 phasemask diameter per circle

# move off phase mask, its good to make sure zwfs object has dark, bad pixel map etc first to see better
phasemask.move_absolute( initial_pos ) 
phasemask.move_relative( [1000,1000] )  # 1mm in each axis
time.sleep(1.2)
reference_img =  np.mean(zwfs.get_some_frames(number_of_frames = 10, apply_manual_reduction = True ) , axis=0 ) # Capture reference image when misaligned
phasemask.move_absolute( initial_pos )  # move back to initial_position 
time.sleep(1.2)

# Start spiral search and fine centering
fine_tune_threshold=3
savefigName = fig_path + 'delme.png'

centered_position = pct.spiral_search_and_center(
    zwfs, phasemask, phasemask_name, search_radius, dr, dtheta, reference_img, fine_tune_threshold=fine_tune_threshold, savefigName=savefigName
)

def compute_image_difference(img1, img2):
    # normalize both images first
    img1 = img1.copy() /np.sum(img1)
    img2 = img2.copy() /np.sum(img2)
    return np.sum(np.abs(img1 - img2))

def calculate_movement_directions(image):
    """
    Calculate the direction to move the phase mask to improve symmetry.
    
    Parameters:
    - image: 2D numpy array representing the image.
    
    Returns:
    - Tuple of (dx, dy) indicating the direction to move the phase mask.
    """
    y_center, x_center = np.array(image.shape) // 2

    # Extract the four quadrants
    q1 = image[:y_center, :x_center]  # Top-left
    q2 = np.flip(image[y_center:, :x_center], axis=0)  # Bottom-left (flipped)
    q3 = np.flip(image[:y_center, x_center:], axis=1)  # Top-right (flipped)
    q4 = np.flip(image[y_center:, x_center:], axis=(0, 1))  # Bottom-right (flipped)

    # Calculate the differences
    diff1 = np.sum(np.abs(q1 - q4))
    diff2 = np.sum(np.abs(q2 - q3))

    # Determine movement directions based on differences
    dx = (np.sum(np.abs(q3 - q1)) - np.sum(np.abs(q2 - q4))) / (np.sum(np.abs(q3 + q1)) + np.sum(np.abs(q2 + q4)))
    dy = (np.sum(np.abs(q2 - q1)) - np.sum(np.abs(q3 - q4))) / (np.sum(np.abs(q2 + q1)) + np.sum(np.abs(q3 + q4)))

    # Normalize to unit length
    magnitude = np.sqrt(dx**2 + dy**2)
    if magnitude > 0:
        dx /= magnitude
        dy /= magnitude

    return dx, dy

def is_symmetric(image, threshold=0.1):
    """
    Check if the image is symmetric and calculate the direction to move for better symmetry.
    
    Parameters:
    - image: 2D numpy array representing the image.
    - threshold: float, maximum allowable difference for symmetry to be considered acceptable.
    
    Returns:
    - Tuple of (is_symmetric, (dx, dy)) indicating whether the image is symmetric and the direction to move.
    """
    y_center, x_center = np.array(image.shape) // 2

    # Extract the four quadrants
    q1 = image[:y_center, :x_center]  # Top-left
    q2 = np.flip(image[y_center:, :x_center], axis=0)  # Bottom-left (flipped)
    q3 = np.flip(image[:y_center, x_center:], axis=1)  # Top-right (flipped)
    q4 = np.flip(image[y_center:, x_center:], axis=(0, 1))  # Bottom-right (flipped)

    # Calculate the differences
    diff1 = np.sum(np.abs(q1 - q4))
    diff2 = np.sum(np.abs(q2 - q3))
    
    # Determine if the image is symmetric
    symmetric = diff1 <= threshold and diff2 <= threshold

    # Calculate the direction to move if not symmetric
    if not symmetric:
        dx, dy = calculate_movement_directions(image)
    else:
        dx, dy = 0, 0

    return symmetric, (dx, dy)

if 1:
    x, y = initial_pos
    angle = 0
    radius = 0
    plot_cnt = 0 # so we don't plot every iteration 
    
    diff_list = [] # to track our metrics 
    x_pos_list = [] 
    y_pos_list = []
    sleep_time = 0.7 #s
    while radius < search_radius:
        x_pos = x + radius * np.cos(angle)
        y_pos = y + radius * np.sin(angle)

        phasemask.move_absolute([x_pos, y_pos])
        time.sleep( sleep_time)  # wait for the phase mask to move and settle
        img = zwfs.get_image()

        diff = compute_image_difference(img, reference_img)
        diff_list.append( diff )
        x_pos_list.append( x_pos )
        y_pos_list.append( y_pos )
        print(f'img diff = {diff}, fine_tune_threshold={fine_tune_threshold}')

        # Update for next spiral step
        angle += dtheta
        radius += dr




        #print( radius )
        #_ = input('next')
        if savefigName != None: 
            if np.mod( plot_cnt , 5) == 0:

                norm = plt.Normalize(0 , fine_tune_threshold)

                fig,ax = plt.subplots( 1,3 ,figsize=(10,6))
                ax[0].set_title( 'image' )
                ax[1].set_title( f'search positions\nx:{phasemask.motors["x"]}\ny:{phasemask.motors["y"]}' )
                ax[2].set_title( 'search metric' )

                ax[0].imshow( img )
                ax[1].plot( [x_pos,y_pos] , 'x', color='r', label='current pos')
                ax[1].plot( [initial_pos[0],initial_pos[1]] , 'o', color='k', label='current pos')
                tmp_diff_list = np.array(diff_list)
                tmp_diff_list[tmp_diff_list < 1e-5 ] = 0.1 # very small values got to finite value (errors whern 0!)
                # s= np.exp( 400 * np.array(tmp_diff_list) / fine_tune_threshold )
                ax[1].scatter( x_pos_list, y_pos_list , s = 10   ,\
                 marker='o', c=diff_list, cmap='viridis', norm=norm)
                ax[1].set_xlim( [initial_pos[0] - search_radius,  initial_pos[0] + search_radius] )
                ax[1].set_ylim( [initial_pos[1] - search_radius,  initial_pos[1] + search_radius] )
                ax[1].legend() 
                ax[2].plot( diff_list )
                ax[2].set_xlim( [0, search_radius/dr] )

                ax[0].axis('off')
                ax[1].set_ylabel( 'y pos' )
                ax[1].set_xlabel( 'x pos' )
                ax[2].set_ylabel( r'$\Sigma|img - img_off|$' )
                ax[2].set_xlabel( 'iteration' )
                plt.savefig( savefigName)
                plt.close()
            plot_cnt += 1

    best_pos = [ x_pos_list[np.argmax( diff_list )], y_pos_list[np.argmax( diff_list )] ]

    move2best = input( f'move to recommended best position = {best_pos}? enter 1/0')
    if move2best :
        phasemask.move_absolute( best_pos )
    else :
        print('moving back to initial position')
        phasemask.move_absolute( initial_pos )

    #phasemask.move_absolute( phasemask.phase_positions[phasemask_name]  )
    time.sleep(0.5)
    img = zwfs.get_image()
    plt.figure();plt.imshow( img ) ;plt.savefig( savefigName )

    do_fine_adjustment = input('ready for fine adjustment') 
    if do_fine_adjustment:
        # do fine adjustments 
        fine_adj_imgs = []
        for i in range(5):
            img = zwfs.get_image() 
            fine_adj_imgs.append( img )
            dr = dr/2 # half movements each time  
            dx, dy = calculate_movement_directions(img) # dx, dy are normalized to radius 1
            phasemask.move_relative( [dr * dx, dr * dy] ) 

        fig,ax = plt.subplots( len(fine_adj_imgs))
        for img,axx in zip(fine_adj_imgs,ax.reshape(-1)):
            axx.imshow( img )
        plt.savefig( savefigName )

        


def calculate_movement_directions(image):
    """
    Calculate the direction to move the phase mask to improve symmetry.
    
    Parameters:
    - image: 2D numpy array representing the image.
    
    Returns:
    - Tuple of (dx, dy) indicating the direction to move the phase mask.
    """
    y_center, x_center = np.array(image.shape) // 2

    # Extract the four quadrants
    q1 = image[:y_center, :x_center]  # Top-left
    q2 = np.flip(image[y_center:, :x_center], axis=0)  # Bottom-left (flipped)
    q3 = np.flip(image[:y_center, x_center:], axis=1)  # Top-right (flipped)
    q4 = np.flip(image[y_center:, x_center:], axis=(0, 1))  # Bottom-right (flipped)

    # Calculate the differences
    diff1 = np.sum(np.abs(q1 - q4))
    diff2 = np.sum(np.abs(q2 - q3))

    # Determine movement directions based on differences
    dx = (np.sum(np.abs(q3 - q1)) - np.sum(np.abs(q2 - q4))) / (np.sum(np.abs(q3 + q1)) + np.sum(np.abs(q2 + q4)))
    dy = (np.sum(np.abs(q2 - q1)) - np.sum(np.abs(q3 - q4))) / (np.sum(np.abs(q2 + q1)) + np.sum(np.abs(q3 + q4)))

    # Normalize to unit length
    magnitude = np.sqrt(dx**2 + dy**2)
    if magnitude > 0:
        dx /= magnitude
        dy /= magnitude

    return dx, dy

def is_symmetric(image, threshold=0.1):
    """
    Check if the image is symmetric and calculate the direction to move for better symmetry.
    
    Parameters:
    - image: 2D numpy array representing the image.
    - threshold: float, maximum allowable difference for symmetry to be considered acceptable.
    
    Returns:
    - Tuple of (is_symmetric, (dx, dy)) indicating whether the image is symmetric and the direction to move.
    """
    y_center, x_center = np.array(image.shape) // 2

    # Extract the four quadrants
    q1 = image[:y_center, :x_center]  # Top-left
    q2 = np.flip(image[y_center:, :x_center], axis=0)  # Bottom-left (flipped)
    q3 = np.flip(image[:y_center, x_center:], axis=1)  # Top-right (flipped)
    q4 = np.flip(image[y_center:, x_center:], axis=(0, 1))  # Bottom-right (flipped)

    # Calculate the differences
    diff1 = np.sum(np.abs(q1 - q4))
    diff2 = np.sum(np.abs(q2 - q3))
    
    # Determine if the image is symmetric
    symmetric = diff1 <= threshold and diff2 <= threshold

    # Calculate the direction to move if not symmetric
    if not symmetric:
        dx, dy = calculate_movement_directions(image)
    else:
        dx, dy = 0, 0

    return symmetric, (dx, dy)






import numpy as np
import scipy.interpolate as interp
import matplotlib.pyplot as plt

# Assuming you have these arrays:
# positions: list of [x, y] positions
# diffs: corresponding diff values for each [x, y] position

def convert_to_polar(x, y, center=(0, 0)):
    """ Convert (x, y) to polar coordinates (radius, angle) relative to a center. """
    x_centered, y_centered = x - center[0], y - center[1]
    radius = np.sqrt(x_centered**2 + y_centered**2)
    angle = np.arctan2(y_centered, x_centered)
    return radius, angle

def bin_by_angle(radius, angle, diff_list, nbins):
    """ Bin diffs by angle into nbins. """
    bins = np.linspace(-np.pi, np.pi, nbins + 1)
    digitized = np.digitize(angle, bins)

    # Adjust the binning so that values in the last bin are included
    digitized = np.clip(digitized, 1, nbins)
    
    binned_data = {i: [] for i in range(1, nbins + 1)}

    for r, a, d, bin_id in zip(radius, angle, diff_list, digitized):
        binned_data[bin_id].append((r, d))
    
    return binned_data
import numpy as np
import scipy.interpolate as interp

def interpolate_max_in_bins(binned_data):
    """ Interpolate diff vs radius and find max in each angle bin. """
    max_radii = []
    max_angles = []
    interpolated_data = {}

    for bin_id, data in binned_data.items():
        if len(data) < 2:  # Ensure at least 2 points for interpolation
            continue

        data = sorted(data, key=lambda x: x[0])  # Sort by radius
        radii, diffs = zip(*data)
        
        if len(radii) > 3:  # Default spline degree is 3
            spline = interp.UnivariateSpline(radii, diffs, s=0)
            fine_radii = np.linspace(min(radii), max(radii), 1000)
            fine_diffs = spline(fine_radii)
            max_idx = np.argmax(fine_diffs)
            max_radii.append(fine_radii[max_idx])
            max_angles.append(np.mean([2 * np.pi / len(binned_data) * (bin_id - 1)]))
            interpolated_data[bin_id] = (fine_radii, fine_diffs)
        else:
            max_radii.append(max(radii, key=lambda r: diffs[radii.index(r)]))
            max_angles.append(np.mean([2 * np.pi / len(binned_data) * (bin_id - 1)]))
            interpolated_data[bin_id] = (radii, diffs)  # No fine interpolation for sparse data

    return max_radii, max_angles, interpolated_data


def polar_to_cartesian(radius, angle, center=(0, 0)):
    """ Convert polar coordinates (radius, angle) to cartesian (x, y). """
    x = center[0] + radius * np.cos(angle)
    y = center[1] + radius * np.sin(angle)
    return x, y

# Assume initial center is at [0, 0], adjust as needed
center = initial_pos

# Convert positions to polar coordinates
radii, angles = zip(*[convert_to_polar(x, y, center) for x, y in zip(x_pos_list, y_pos_list)])

# Bin diffs by angle with 4 * dangle bins
nbins = int(4 * (2 * np.pi) / dtheta)
binned_data = bin_by_angle(radii, angles, diff_list, nbins)

# Interpolate and find the max diff in each angle bin
max_radii, max_angles, interpolated_data  = interpolate_max_in_bins(binned_data)



def plot_polar_heatmap(interpolated_data):
    import matplotlib.colors as mcolors
    # Define the number of bins and number of points
    nbins = len(interpolated_data)
    n_points = 1000  # Number of points for fine_radii

    # Create arrays to hold the data
    radii_list = []
    angles_list = []
    diffs_list = []

    # Collect data
    for bin_id, (fine_radii, fine_diffs) in interpolated_data.items():
        angles = np.full_like(fine_radii, np.mean([2 * np.pi / nbins * (bin_id - 1)]))
        radii_list.extend(fine_radii)
        angles_list.extend(angles)
        diffs_list.extend(fine_diffs)

    # Convert lists to numpy arrays
    radii_array = np.array(radii_list)
    angles_array = np.array(angles_list)
    diffs_array = np.array(diffs_list)

    # Create polar plot
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, polar=True)
    
    # Normalize the diff values for colormap
    norm = mcolors.Normalize(vmin=np.min(diffs_array), vmax=np.max(diffs_array))
    cmap = plt.get_cmap('viridis')
    
    # Scatter plot for heatmap
    sc = ax.scatter(angles_array, radii_array, c=diffs_array, cmap=cmap, norm=norm, s=10, edgecolors='none')
    
    # Add colorbar
    cbar = plt.colorbar(sc, ax=ax, orientation='vertical')
    cbar.set_label('Diff')
    
    # Set labels and title
    ax.set_xlabel('Angle (radians)')
    ax.set_ylabel('Radius')
    ax.set_title('Polar Heatmap of Radius vs Diff')
    plt.savefig( savefigName )






plt.figure(figsize=(12, 8))

for bin_id, (fine_radii, fine_diffs) in interpolated_data.items():
    plt.plot(fine_radii, fine_diffs, label=f'Bin {bin_id}')

plt.xlabel('Radius')
plt.ylabel('Diff')
plt.title('Radius vs Diff for Each Angle Bin')
plt.legend()
plt.grid(True)
plt.savefig( savefigName )

# Find the absolute max
max_diff_idx = np.argmax(max_radii)
best_radius = max_radii[max_diff_idx]
best_angle = max_angles[max_diff_idx]

# Convert best radius and angle to x, y position
best_x, best_y = polar_to_cartesian(best_radius, best_angle, center)

# Move the phase mask to the optimal position
phasemask.move_absolute([best_x, best_y])

#update phasemask position 

print(f"Phase mask moved to optimal position: ({best_x}, {best_y})")

















plt.figure() ; plt.title('dark'); plt.imshow( zwfs.reduction_dict['dark'][0] ); plt.colorbar() ; plt.savefig(fig_path + 'delme.png' )

plt.figure() ;  plt.title('bad pixels'); plt.imshow( zwfs.bad_pixel_filter.reshape(zwfs.reduction_dict['dark'][0].shape) ); plt.savefig(fig_path + 'delme.png' )


# !!!! PUT IN SOURCE !!!! 

# quick check that dark subtraction works
I0 = zwfs.get_image( apply_manual_reduction  = True)
plt.figure(); plt.title('test image \nwith dark subtraction \nand bad pixel mask'); plt.imshow( I0 ); plt.colorbar()
plt.savefig( fig_path + 'delme.png')


zwfs.dm.send_data(zwfs.dm_shapes['flat_dm'])


# ====== testing reconstruction 
#init our phase controller (object that processes ZWFS images and outputs DM commands)
phase_ctrl = phase_control.phase_controller_1(config_file = None) 
# to change basis : 
#phase_ctrl.change_control_basis_parameters( controller_label = ctrl_method_label, number_of_controlled_modes=phase_ctrl.config['number_of_controlled_modes'], basis_name='Zonal' , dm_control_diameter=None, dm_control_center=None)

#init our pupil controller (object that processes ZWFS images and outputs VCM commands)
pupil_ctrl = pupil_control.pupil_controller_1(config_file = None)

#analyse pupil and decide if it is ok. This must be done before reconstructor
pupil_report = pupil_control.analyse_pupil_openloop( zwfs, debug = True, return_report = True)

if pupil_report['pupil_quality_flag'] == 1: 
    zwfs.update_reference_regions_in_img( pupil_report ) # 




# --- linear ramps 
# use baldr.
recon_data = util.GET_BDR_RECON_DATA_INTERNAL(zwfs, number_amp_samples = 20, amp_max = 0.2,\
number_images_recorded_per_cmd = 4, save_fits = data_path+f'pokeramp_data_MASK_J3_sydney_{tstamp}.fits') 
#recon_data = fits.open( data_path+'recon_data_LARGE_SECONDARY_19-04-2024T12.19.22.fits' )



zwfs.pupil_pixel_filter = ~zwfs.bad_pixel_filter
zwfs.pupil_pixels = np.where( ~zwfs.bad_pixel_filter )[0]

ctrl_method_label = 'ctrl_1'
zwfs.dm.send_data( zwfs.dm_shapes['flat_dm'] )
time.sleep( 0.1 )
# TRY model_2 WITH  method='single_side_poke', or 'double_sided_poke'
#phase_ctrl.change_control_basis_parameters(  number_of_controlled_modes=140, basis_name ='Zonal', dm_control_diameter=None, dm_control_center=None,controller_label=None)
#phase_ctrl.build_control_model_2(zwfs, poke_amp = -0.1, label='ctrl_1', poke_method='double_sided_poke', inverse_method='MAP',  debug = True)
phase_ctrl.build_control_model_2(zwfs, poke_amp = -0.3, label='ctrl_1', poke_method='double_sided_poke', inverse_method='MAP',  debug = True)
#phase_ctrl.build_control_model( zwfs , poke_amp = -0.15, label='ctrl_1', debug = True)  

#phase_ctrl.plot_SVD_modes( zwfs, 'ctrl_1', save_path=fig_path)


# write fits to input into RTC
zwfs.write_reco_fits( phase_ctrl, 'ctrl_1', save_path=data_path)





# can we reconstruct in open loop 
mode_basis = phase_ctrl.config['M2C']  # readability 
I2M = phase_ctrl.ctrl_parameters[ctrl_method_label]['I2M']
IM = phase_ctrl.ctrl_parameters[ctrl_method_label]['IM'] # readability 
# unfiltered CM
CM = phase_ctrl.ctrl_parameters[ctrl_method_label]['CM'] # readability 
R_TT = phase_ctrl.ctrl_parameters[ctrl_method_label]['R_TT'] # readability 
R_HO = phase_ctrl.ctrl_parameters[ctrl_method_label]['R_HO'] # readability 

M2C = phase_ctrl.ctrl_parameters[ctrl_method_label]['M2C_4reco'] # readability  # phase_ctrl.ctrl_parameters[ctrl_method_label]['M2C_4reco']#
I0 = phase_ctrl.ctrl_parameters[ctrl_method_label]['ref_pupil_FPM_in']

poke_amp = phase_ctrl.ctrl_parameters[ctrl_method_label]['poke_amp']
for mode_indx in range( len(M2C)-1 ) :  

    mode_aberration = mode_basis.T[mode_indx]#   M2C.T[mode_indx]
    #plt.imshow( util.get_DM_command_in_2D(amp*mode_aberration));plt.colorbar();plt.show()
    
    dm_cmd_aber = zwfs.dm_shapes['flat_dm'] + amp * mode_aberration 

    zwfs.dm.send_data( dm_cmd_aber )
    time.sleep(0.1)
    raw_img_list = []
    for i in range( 10 ) :
        raw_img_list.append( zwfs.get_image() ) # @D, remember for control_phase method this needs to be flattened and filtered for pupil region
    raw_img = np.median( raw_img_list, axis = 0) 
    # plt.figure() ; plt.imshow( raw_img ) ; plt.savefig( fig_path + f'delme.png') # <- signal?
    
    err_img = phase_ctrl.get_img_err( 1/np.mean(raw_img) * raw_img.reshape(-1)[zwfs.pupil_pixel_filter]  ) 
    # plt.figure() ; plt.hist( err_img, label='meas', alpha=0.3 ) ; plt.hist( IM[mode_indx] , label='from IM', alpha=0.3); plt.legend() ; plt.savefig( fig_path + f'delme.png') # <- should be around zeros

    #mode_res_test : inject err_img from interaction matrix to I2M .. should result in perfect reconstruction  
    #plt.figure(); plt.plot( I2M.T @ IM[2] ); plt.savefig( fig_path + f'delme.png')
    #plt.figure(); plt.plot( I2M.T @ IM[mode_indx]  ,label='reconstructed amplitude'); plt.axvline(mode_indx  , ls=':', color='k', label='mode applied') ; plt.xlabel('mode index'); plt.ylabel('mode amplitude'); plt.legend(); plt.savefig( fig_path + f'delme.png')
    mode_res =  I2M.T @ err_img 


    plt.figure(); plt.plot( mode_res ); plt.axvline(mode_indx  , ls=':', color='k') ; plt.savefig( fig_path + f'delme.png')
    plt.figure(figsize=(8,5));
    plt.plot( mode_res  ,label='reconstructed amplitude');
    app_amp = np.zeros( len( mode_res ) ) 

    app_amp[mode_indx] = amp / poke_amp

    plt.plot( app_amp ,'x', label='applied amplitude');
    plt.axvline(mode_indx  , ls=':', color='k', label='mode applied') ; plt.xlabel('mode index',fontsize=15); 
    plt.ylabel('mode amplitude',fontsize=15); plt.gca().tick_params(labelsize=15) ; plt.legend();
    plt.savefig( fig_path + f'delme.png')

    _ = input('press when ready to see mode reconstruction')
    
    cmd_res = 1/poke_amp * M2C @ mode_res
    
    # WITH RESIDUALS 
    
    im_list = [util.get_DM_command_in_2D( mode_aberration ),1/np.mean(raw_img) * raw_img - I0/np.mean(I0),  util.get_DM_command_in_2D( cmd_res ) ,util.get_DM_command_in_2D( mode_aberration - cmd_res ) ]
    xlabel_list = [None, None, None, None]
    ylabel_list = [None, None, None, None]
    title_list = ['Aberration on DM', 'ZWFS signal', 'reconstructed DM cmd', 'residual']
    cbar_label_list = ['DM command', 'ADU (Normalized)', 'DM command' , 'DM command' ] 
    savefig = fig_path + 'delme.png' #f'mode_reconstruction_images/phase_reconstruction_example_mode-{mode_indx}_basis-{phase_ctrl.config["basis"]}_ctrl_modes-{phase_ctrl.config["number_of_controlled_modes"]}ctrl_act_diam-{phase_ctrl.config["dm_control_diameter"]}_readout_mode-12x12.png'

    util.nice_heatmap_subplots( im_list , xlabel_list, ylabel_list, title_list, cbar_label_list, fontsize=15, axis_off=True, cbar_orientation = 'bottom', savefig=savefig)
    
    _ = input('press when ready to go to next moce ')




a = fits.open( 'tmp/RECONSTRUCTORS_DIT-0.001_gain_high_22-08-2024T21.26.09.fits' ) 





U,S,Vt = np.linalg.svd( IM @ IM.T )
plt.figure(); plt.semilogy( S ); plt.savefig( fig_path + "delme.png")

plt.figure(); plt.imshow( util.get_DM_command_in_2D(U.T[0]) ); plt.savefig( fig_path + "delme.png")



# test we can get on/off phase mask with DM 
fourier_basis = util.construct_command_basis( basis='fourier', number_of_modes = 40, Nx_act_DM = 12, Nx_act_basis = 12, act_offset=(0,0), without_piston=True)
tip = fourier_basis[:,0]
tilt = fourier_basis[:,5]

fig,ax = plt.subplots(1,2)
ax[0].imshow( util.get_DM_command_in_2D( tip ) ); ax[0].set_title('')
ax[1].imshow( util.get_DM_command_in_2D( tilt ) ); ax[1].set_title('')
plt.savefig( fig_path + "delme.png")

zwfs.dm.send_data( zwfs.dm_shapes['flat_dm'] )
time.sleep(0.1)
I0 =  zwfs.get_image( apply_manual_reduction = True )
zwfs.dm.send_data( 0.5 + 2* tip )
time.sleep(0.1)
N0 = zwfs.get_image( apply_manual_reduction = True )
zwfs.dm.send_data( zwfs.dm_shapes['flat_dm'] )

fig,ax = plt.subplots(1,2)
ax[0].imshow( I0 ); ax[0].set_title('FPM ON')
ax[1].imshow( N0 ); ax[1].set_title('FPM OFF')
plt.savefig( fig_path + "delme.png")



M2C = phase_ctrl.config['M2C'] # readability 

I2M = phase_ctrl.ctrl_parameters[ctrl_method_label]['I2M']

IM = phase_ctrl.ctrl_parameters[ctrl_method_label]['IM'] # readability 
# unfiltered CM
CM = phase_ctrl.ctrl_parameters[ctrl_method_label]['CM'] # readability 

U,S,Vt = np.linalg.svd( IM @ IM.T )

plt.figure(); plt.semilogy( S ); plt.savefig( fig_path + "delme.png")

plt.figure(); plt.imshow( phase_ctrl.ctrl_parameters[ctrl_method_label]['ref_pupil_FPM_in'] ); plt.savefig( fig_path + "delme.png")

plt.figure(); plt.plot( I2M.T @ IM[65] ); plt.savefig( fig_path + "delme.png")


# do one poke and check we actually get something is
zwfs.dm.send_data( zwfs.dm_shapes['flat_dm'] )
act = 65
damp = 0.1

time.sleep(0.1)
I0 =  zwfs.get_image( apply_manual_reduction = True )
cmd = zwfs.dm_shapes['flat_dm'].copy() 
cmd[act] += damp 
zwfs.dm.send_data( cmd )
time.sleep(0.1)
Ip = zwfs.get_image( apply_manual_reduction = True )

cmd = zwfs.dm_shapes['flat_dm'].copy() 
cmd[act] -= damp 
zwfs.dm.send_data( cmd )
time.sleep(0.1)
Im = zwfs.get_image( apply_manual_reduction = True )

fig,ax = plt.subplots(1,2)
ax[0].imshow( Ip-I0 ); ax[0].set_title('I(a65+=0.1) - I0')
ax[1].imshow( Im-I0 ); ax[1].set_title('I(a65-=0.1) - I0')
plt.savefig( fig_path + "delme.png")

#plt.figure(); plt.imshow( Ip-I0 ); plt.savefig( fig_path + "delme.png")

