

import zmq
import common.phasemask_centering_tool as pct
import time
import toml
import argparse
import os 
import datetime
import numpy as np 
from scipy.interpolate import griddata
import matplotlib.pyplot as plt 
from scipy.optimize import leastsq
from scipy.ndimage import label, find_objects
from xaosim.shmlib import shm


def percentile_based_detect_pupils(
    image, percentile=80, min_group_size=50, buffer=20, plot=True
):
    """
    Detects circular pupils by identifying regions with grouped pixels above a given percentile.

    Parameters:
        image (2D array): Full grayscale image containing multiple pupils.
        percentile (float): Percentile of pixel intensities to set the threshold (default 80th).
        min_group_size (int): Minimum number of adjacent pixels required to consider a region.
        buffer (int): Extra pixels to add around the detected region for cropping.
        plot (bool): If True, displays the detected regions and coordinates.

    Returns:
        list of tuples: Cropping coordinates [(x_start, x_end, y_start, y_end), ...].
    """
    # Normalize the image
    image = image / image.max()

    # Calculate the intensity threshold as the 80th percentile
    threshold = np.percentile(image, percentile)

    # Create a binary mask where pixels are above the threshold
    binary_image = image > threshold

    # Label connected regions in the binary mask
    labeled_image, num_features = label(binary_image)

    # Extract regions and filter by size
    regions = find_objects(labeled_image)
    pupil_regions = []
    for region in regions:
        y_slice, x_slice = region
        # Count the number of pixels in the region
        num_pixels = np.sum(labeled_image[y_slice, x_slice] > 0)
        if num_pixels >= min_group_size:
            # Add a buffer around the region for cropping
            y_start = max(0, y_slice.start - buffer)
            y_end = min(image.shape[0], y_slice.stop + buffer)
            x_start = max(0, x_slice.start - buffer)
            x_end = min(image.shape[1], x_slice.stop + buffer)
            pupil_regions.append((x_start, x_end, y_start, y_end))

    if plot:
        # Plot the original image with bounding boxes
        plt.figure(figsize=(10, 10))
        plt.imshow(image, cmap="gray", origin="upper")
        for x_start, x_end, y_start, y_end in pupil_regions:
            rect = plt.Rectangle(
                (x_start, y_start),
                x_end - x_start,
                y_end - y_start,
                edgecolor="red",
                facecolor="none",
                linewidth=2,
            )
            plt.gca().add_patch(rect)
        plt.title(f"Detected Pupils: {len(pupil_regions)}")
        plt.savefig('delme.png')
        plt.show()
        

    return pupil_regions

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



# paths and timestamps
tstamp = datetime.datetime.now().strftime("%d-%m-%YT%H.%M.%S")
tstamp_rough =  datetime.datetime.now().strftime("%d-%m-%Y")



# setting up socket to ZMQ communication to multi device server
parser = argparse.ArgumentParser(description="ZeroMQ Client and Mode setup")
parser.add_argument("--host", type=str, default="localhost", help="Server host")
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
    default="baldr",
    help="do we look on Baldr or Heimdallr side of camera?. Default: %(default)s. Options: baldr, heimdallr"
)

parser.add_argument(
    '--motor',
    type=str,
    default="BTX",
    help="what motor to scan. Default: %(default)s"
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
    default=0.3,
    help="search radius of spiral search in microns. Default: %(default)s"
)
parser.add_argument(
    '--dx',
    type=float,
    default=0.1,
    help="step size in motor units during scan. Default: %(default)s"
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

# set up camera 
c = shm(args.global_camera_shm)

# copied from Engineering GUI 
if args.motor in ["HTXP", "HTXI", "BTX", "BOTX"]:
    # replace the X in target with P
    target = f"{args.motor}{args.beam}"
    targets = [target.replace("X", "P"), target.replace("X", "T")]
    
if args.motor in ["c_red_one_focus", "intermediate_focus", "baldr"]
    config = args.motor
    
    # Update original_positions
    if config == 'baldr':
        axes = [f"BTP{args.beam}", f"BTT{args.beam}", f"BOTP{args.beam}", f"BOTT{args.beam}"]
    else:
        axes = [f"HTPP{beam}", f"HTTP{args.beam}", f"HTPI{args.beam}", f"HTTI{args.beam}"]



pos_dict = {}
for axis in axes:
    pos = send_and_get_response(f"read {axis}")
    pos_dict[axis] = pos

# try read the positions first as a check
try:
    
    message = f"read {targets[0]}"
    initial_Xpos = float(send_and_get_response(message))

    message = f"read {targets[1]}"
    initial_Ypos = float(send_and_get_response(message))

    print( f"{args.motor} position before starting = {initial_Xpos}, {initial_Ypos}"  )

except:
    raise UserWarning( "failed 'read {args.motor}X{args.beam}' or  'read {args.motor}Y{args.beam}'")



# Remove all other beams besides what we want to look at 
take_out_beams = list(filter(lambda x: x != int( args.beam ), [1,2,3,4] ))

for b in take_out_beams:
    message = f"moveabs SSF{b} 0.0"
    send_and_get_response(message)
    print(f"moved out beam {b}")
    time.sleep( 0.5 )

# initial image to find suitable cropponmg regions to zoom on feature
img_raw = c.get_data()

## Identify bad pixels (this can throw it off!!)
mean_frame = np.mean(img_raw, axis=0)
std_frame = np.std(img_raw, axis=0)

global_mean = np.mean(mean_frame)
global_std = np.std(mean_frame)
bad_pixel_map = (np.abs(mean_frame - global_mean) > 20 * global_std) | (std_frame > 100 * np.median(std_frame))

# plt.figure()
# plt.imshow( bad_pixel_map ) #[ crop_pupil_coords[i][2]:crop_pupil_coords[i][3],crop_pupil_coords[i][0]:crop_pupil_coords[i][1]])
# plt.colorbar()
# plt.savefig( "delme.png")

img = np.mean( img_raw , axis=0)

img[bad_pixel_map] = 0

# mask baldr or heimdallr side of camera?
baldr_mask = np.zeros_like(img).astype(bool)
baldr_mask[img.shape[0]//2 : img.shape[0] , : ] = True # baldr occupies top half (pixels)
heim_mask = ~baldr_mask # heimdallr occupies bottom half

if args.system.lower() == 'baldr':
    mask = baldr_mask
    print('looking onn Baldr side')
    
elif args.system.lower() == 'heimdallr':
    mask = heim_mask
    print('looking on Heimdallr side')
else:
    print('no valid system input. Looking at entire image.. Could misclassify')
    mask = np.ones_like(img).astype(bool)



### A SMARTER WAY WOULD BE TO JUST MOVE MY MOTOR OF INTEREST AND DIFFERENCE IMAGES!!!

crop_pupil_coords = np.array( percentile_based_detect_pupils(
        img * mask, percentile = 99, min_group_size=100, buffer=60, plot=True
    ) )
#plt.savefig('delme.png')





# if multiple things detected just look at the one with heightest mean signal 
if len( crop_pupil_coords ) == 0:
    raise UserWarning('light source off? we cant detect anythin')

elif len( crop_pupil_coords ) > 1:
    sigtmp = []
    for roi in crop_pupil_coords:
        c1,c2,r1,r2 = roi
    
        # look at mean in tight region around the center
        cx, cy = (r1+r2)//2 , (c1+c2)//2
        cr = 15
        meamI = np.nanmean( (img * mask)[cx-cr:cx+cr, cy-cr:cy+cr] )

        sigtmp.append( meamI ) #np.sum( (img * mask)[r1:r2, c1:c2]  > meamI) )

    high_sig_idx = np.argmax( sigtmp ) 
    c1,c2,r1,r2 = crop_pupil_coords[high_sig_idx]

else :
    c1,c2,r1,r2 = crop_pupil_coords

# plt.figure()
# plt.imshow( img[r1:r2,c1:c2] ) #[ crop_pupil_coords[i][2]:crop_pupil_coords[i][3],crop_pupil_coords[i][0]:crop_pupil_coords[i][1]])
# plt.colorbar()
# plt.title('cropped image')
# plt.savefig( "delme.png")
# plt.show()


# Get our starting position based on user input 
if args.initial_pos == 'current':
    starting_point = [initial_Xpos, initial_Ypos]
else:
    tmp_pos = args.initial_pos.split( ',')

    Xpos = float( tmp_pos[0] )  # start_position_dict[args.phasemask_name][0]
    Ypos = float( tmp_pos[1] ) # st
    
    starting_point = [initial_Xpos, initial_Ypos]
    
# generate the scan points 
spiral_pattern = pct.square_spiral_scan(starting_point, args.dx, args.search_radius)

x_points, y_points = zip(*spiral_pattern)

img_dict = {}

sleep_time = 1

# we should have predifed json file for these..
if args.motor == 'BTX':
    safety_limits = {"xmin":-0.6,"xmax":0.6, "ymin":-0.6,"ymax":0.6}
else:
    safety_limits = {"xmin":-np.inf,"xmax":np.inf, "ymin":-np.inf,"ymax":np.inf}


## Start 
for i, (x_pos, y_pos) in enumerate(zip(x_points, y_points)):
    print("at ", x_pos, y_pos)
    print(f"{100 * i/len(x_points)}% complete")

    # motor limit safety checks!
    if x_pos <= safety_limits['xmin']:
        print(f'x_pos < safemin. set x_pos = {safety_limits["xmin"]}')
        x_pos = safety_limits['xmin']
    if x_pos >= safety_limits['xmax']:
        print(f'x_pos > safemax. set x_pos = {safety_limits["xmax"]}')
        x_pos = safety_limits['xmax']
    if y_pos <= safety_limits['ymin']:
        print(f'y_pos < safemin. set y_pos = {safety_limits["ymin"]}')
        y_pos = safety_limits['ymin']
    if y_pos >= safety_limits['ymax']:
        print(f'y_pos > 10000. set y_pos = {safety_limits["ymax"]}')
        y_pos = safety_limits['ymax']


    #message = f"fpm_moveabs phasemask{beam} {[x_pos, y_pos]}"
    #message = f"moveabs {targets[0]} {x_pos}"
    
    #cmd = f"move_pupil {config} {beam} {delx} {dely}"

    response = send_and_get_response(message)
    print(response)

    #message = f"moveabs {targets[1]} {y_pos}"
    #cmd = f"move_pupil {config} {beam} {delx} {dely}"
    response = send_and_get_response(message)
    print(response)

    time.sleep(sleep_time)  # wait for the phase mask to move and settle

    img_raw = np.mean(
        c.get_data(),
        axis=0,
    )
    
    img_raw[bad_pixel_map] = 0

    img = img_raw[r1:r2,c1:c2]
    
    img_dict[(x_pos, y_pos)] = img

