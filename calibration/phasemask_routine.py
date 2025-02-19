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
import atexit
# to use plotting when remote sometimes X11 forwarding is bogus.. so use this: 
import matplotlib 
matplotlib.use('Agg')

# custom
from asgard_alignment import FLI_Cameras as FLI
from common import phasemask_centering_tool as pct

# for making movies of scan (optional dependancy)
import matplotlib.animation as animation
from scipy.ndimage import median_filter
import ast

# if server is stuck 
# sudo lsof -i :5555 then kill the PID 




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


# for making movies of scan (optional dependancy)
def create_scatter_image_movie(data_dict, save_path="scatter_image_movie.mp4", fps=5):
    """
    Creates a movie showing:
    - A scatter plot of x, y positions up to the current index.
    - An image corresponding to the current index.

    Parameters:
    - data_dict: Dictionary where keys are x, y positions (string tuples) and
                 values are 2D arrays (images).
    - save_path: Path to save the movie file (e.g., "output.mp4").
    - fps: Frames per second for the output movie.
    !!!!!!!!!
    designed to use img_dict returned from spiral_square_search_and_save_images()
    as input 
    !!!!!!!!!
    """
    # Extract data from the dictionary
    #positions = [eval(key) for key in data_dict.keys()]
    positions = []
    for key in data_dict.keys():
        if isinstance(key, str):  # If key is a string
            try:
                # Attempt to safely interpret the string as a tuple
                positions.append(ast.literal_eval(key))
            except (ValueError, SyntaxError):  # If it's not a valid tuple, keep it as is
                positions.append(key)
        elif isinstance(key, tuple):  # If key is already a tuple
            positions.append(key)
        else:
            raise TypeError(f"Unsupported key type: {type(key)}")
        
    images = list(data_dict.values())
    x_positions, y_positions = zip(*positions)

    num_frames = len(positions)

    # Create the figure with two subplots
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    scatter_ax, image_ax = axes

    # Initialize the scatter plot
    scatter = scatter_ax.scatter([], [], c='b', label='Positions')
    scatter_ax.set_xlim(min(x_positions) - 1, max(x_positions) + 1)
    scatter_ax.set_ylim(min(y_positions) - 1, max(y_positions) + 1)
    scatter_ax.set_xlabel("X Position")
    scatter_ax.set_ylabel("Y Position")
    scatter_ax.set_title("Scatter Plot of Positions")
    scatter_ax.legend()

    # Initialize the image plot
    img_display = image_ax.imshow(images[0], cmap='viridis')
    cbar = fig.colorbar(img_display, ax=image_ax)
    cbar.set_label("Intensity")
    image_ax.set_title("Image at Current Position {}")

    # Function to update the plots for each frame
    def update_frame(frame_idx):
        # Update scatter plot
        scatter.set_offsets(np.c_[x_positions[:frame_idx + 1], y_positions[:frame_idx + 1]])

        current_x = round(x_positions[frame_idx])
        current_y = round( y_positions[frame_idx] )
        scatter_ax.set_title(f"Scatter Plot of Positions (Current: x={current_x}, y={current_y})")
        
        # Update image plot
        img_display.set_data(images[frame_idx])
        image_ax.set_title(f"Image at Position x={current_x}, y={current_y}")


        return scatter, img_display

    # Create the animation
    ani = animation.FuncAnimation(fig, update_frame, frames=num_frames, blit=False, repeat=False)

    # Save the animation as a movie file
    ani.save(save_path, fps=fps, writer='ffmpeg')

    plt.close(fig)  # Close the figure to avoid displaying it unnecessarily



def interpolate_bad_pixels(image, bad_pixel_map):
    filtered_image = image.copy()
    filtered_image[bad_pixel_map] = median_filter(image, size=3)[bad_pixel_map]
    return filtered_image


def pixelmask_image_dict(data, bad_pixel_map):
    """
    Apply bad pixel interpolation to all frames and pokes.
    """
    #imgs = np.array( list(  data.values() ) )
    #keys = np.array( list(  data.keys() ) )

    filtered_images = {}
    for c, i in data.items():
        filtered_images[c] = interpolate_bad_pixels(np.array( i ), bad_pixel_map)
    return filtered_images


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

parser.add_argument('--non_verbose',
                action='store_false', 
                help="disable verbose mode (default). This asks before doing things and allows fine adjustments")

parser.add_argument('--open_dms_here',
                action='store_true', 
                help="greedily open the DMs locally here (This will fail if they are opened in a server somewhere else!!! so think first)")

parser.add_argument(
    '--data_path',
    type=str,
    default=f"/home/asg/Progs/repos/asgard-alignment/calibration/reports/phasemask_aquisition/{tstamp_rough}/", #f"/home/heimdallr/data/phasemask_aquisition/{tstamp_rough}/",
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
    '--initial_pos',
    type=str,
    default="recent",
    help="x,y initial position of search or 'recent' to use most recent calibration file. Default: %(default)s "
)

parser.add_argument(
    '--search_radius',
    type=float,
    default=100,
    help="search radius of spiral search in microns. Default: %(default)s"
)
parser.add_argument(
    '--step_size',
    type=float,
    default=20,
    help="set size in microns of square spiral search. Default: %(default)s"
)

parser.add_argument(
    '--cam_fps',
    type=int,
    default=50,
    help="frames per second on camera. Default: %(default)s"
)
parser.add_argument(
    '--cam_gain',
    type=int,
    default=5,
    help="camera gain. Default: %(default)s"
)



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



# home mask.. 
# message = f"init BMX2"
# res = send_and_get_response(message)
# print(res)
# message = f"init BMY2"
# res = send_and_get_response(message)
# print(res)

baldr_pupils_path = default_path_dict['baldr_pupil_crop'] #"/home/asg/Progs/repos/asgard-alignment/config_files/baldr_pupils_coords.json"

with open(baldr_pupils_path, "r") as json_file:
    baldr_pupils = json.load(json_file)



# init camera 
roi = baldr_pupils[str(args.beam)] #[None, None, None, None] # 
c = FLI.fli( roi=roi)

### can uncomment when commands are in 
# configure with default configuration file
# config_file_name = os.path.join(c.config_file_path, "default_cred1_config.json")
# c.configure_camera(config_file_name)

# with open(config_file_name, "r") as file:
#     camera_config = json.load(file)

# apply_manual_reduction = True

# c.send_fli_cmd("set mode globalresetcds")
# time.sleep(1)
# c.send_fli_cmd(f"set gain {args.cam_gain}")
# time.sleep(1)
# c.send_fli_cmd(f"set fps {args.cam_fps}")

# c.start_camera()

# time.sleep(5)

# check the cropped pupil regions are correct: 
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


#### Check we are on the right beam !!!! 
if args.open_dms_here:
    sys.path.insert(1, "/opt/Boston Micromachines/lib/Python3/site-packages/")
    import bmc

    DMshapes_path = "/home/asg/Progs/repos/asgard-alignment/DMShapes/"
    dm_config_path = "/home/asg/Progs/repos/asgard-alignment/config_files/dm_serial_numbers.json"
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


    crossdm = {}
    for beam, serial_number in dm_serial_numbers.items():
        crossdm[beam] = pd.read_csv(
            DMshapes_path + f"Crosshair140.csv",
            header=None,
        )[0].values


    #bbb = args.beam #"3" 
    for bbb in ['1','2','3','4']:
        dm[bbb].send_data( flatdm[bbb]  )

    ## for beam, serial_number in dm_serial_numbers.items():
    ##     dm[beam].close_dm()

    # ## Checking pupil crop region beam labels are correct
    # fig,ax = plt.subplots( 1,4 )
    # for bbb,axx in zip( ['1','2','3','4'], ax.reshape(-1)):

    #     dm[bbb].send_data( flatdm[bbb]  + 0.2 * crossdm[bbb])
    #     time.sleep(2)
    #     img = np.mean( c.get_some_frames( number_of_frames=10, apply_manual_reduction=True ) , axis = 0 ) 

    #     x_start, x_end , y_start, y_end = baldr_pupils[ bbb ]# str(args.beam)]
    #     #plt.figure(); plt.imshow( np.log10( img) ) ; plt.colorbar(); plt.savefig('delme.png')

    #     #plt.figure(); 
    #     iii = axx.imshow( np.log10( img[ x_start:x_end, y_start:y_end ] ) ) 
    #     axx.set_title(f'beam{bbb}')
    #     #plt.colorbar(); 
    

    #     dm[bbb].send_data( flatdm[bbb]  )

    #     #input('check inmage in "delme.png". press any key to go to next beam')

    # plt.savefig('delme.png')


# ======== Source out first for dark and bad pixel map 
# state_dict["socket"].send_string(f"moveabs SSS {source_positions['SSS']['empty']}")
# res = socket.recv_string()
# print(f"Response: {res}")

# time.sleep(5)

# c.build_manual_dark( no_frames = 100 )

# bad_pixels = c.get_bad_pixel_indicies( no_frames = 300, std_threshold = 20, mean_threshold=6 , flatten=False)

# c.build_bad_pixel_mask( bad_pixels , set_bad_pixels_to = 0)

# time.sleep(1)


# state_dict["socket"].send_string(f"moveabs SSS {source_positions['SSS']['SBB']}")
# res = socket.recv_string()
# print(f"Response: {res}")



message = f"fpm_movetomask phasemask{args.beam} {args.phasemask_name}"
res = send_and_get_response(message)
print(res)

message = f"read BMX{args.beam}"
Xpos = float( send_and_get_response(message) )

message = f"read BMY{args.beam}"
Ypos = float( send_and_get_response(message) )

"""
When the MDS is started it looks for the most recent available 
positions file.. However if this is not reset after a new 
calibration we may not have the most recent file

Therefore we manually get the most recent file
and start here

"""

# # get all available files 
valid_reference_position_files = glob.glob(
    f"/home/asg/Progs/repos/asgard-alignment/config_files/phasemask_positions/beam{args.beam}/*json"
    )


if 'recent' in args.initial_pos:

    # read in the most recent and make initial posiition the most recent one for given mask 
    with open(max(valid_reference_position_files, key=os.path.getmtime)
    , "r") as file:
        start_position_dict = json.load(file)

    Xpos = start_position_dict[args.phasemask_name][0]
    Ypos = start_position_dict[args.phasemask_name][1]

else: 
    try: 
        tmp_pos = args.initial_pos.split( ',')

        Xpos = int( tmp_pos[0] )  # start_position_dict[args.phasemask_name][0]
        Ypos = int( tmp_pos[1] ) # start_position_dict[args.phasemask_name][1]
        print(f'using user input initial position x,y ={Xpos}, {Ypos}')
    except: 
        print( 'invalid user input {args.initial_pos} for initial position x,y. Try (for example) --initial_pos 5000,5000  (i.e they are comma-separated)' )
        recent_file = max(valid_reference_position_files, key=os.path.getmtime)
        with open(recent_file, "r") as file:
            start_position_dict = json.load(file)

        print( '\n\n--using the most recent calibration file \n  {recent_file}')

        Xpos = start_position_dict[args.phasemask_name][0]
        Ypos = start_position_dict[args.phasemask_name][1]

#initial_pos_input = input("input initial x,y position for scan (in um, seperated by a comman ',' ).enter 'e' to use the position from the most recent calibration file ")
# if initial_pos_input == "e":

#     # read in the most recent and make initial posiition the most recent one for given mask 
#     with open(max(valid_reference_position_files, key=os.path.getmtime)
#     , "r") as file:
#         start_position_dict = json.load(file)

#     Xpos = start_position_dict[args.phasemask_name][0]
#     Ypos = start_position_dict[args.phasemask_name][1]

# else: 
#     while initial_pos_input != 'e':
#         tmp_pos = initial_pos_input.split( ',')
#         try:
#             Xpos = int( tmp_pos[0] )  # start_position_dict[args.phasemask_name][0]
#             Ypos = int( tmp_pos[1] ) # start_position_dict[args.phasemask_name][1]
#             initial_pos_input = 'e' # to exit loop 
#         except:
#             print( "\ninvalid input. Enter integers for x,y positions seperated by comman or 'e' to use previous calibrated positions\n Try again\n")

#             initial_pos_input = input("input initial x,y position for scan (in um, seperated by a comman ',' ).\nenter 'e' to use the position from the most recent calibration file ")

#message = f"fpm_moveabs phasemask{args.beam} {final_coord}"
message = f"moveabs BMX{args.beam} {Xpos}"
res = send_and_get_response(message)
print(res) 

message = f"moveabs BMY{args.beam} {Ypos}"
res = send_and_get_response(message)
print(res) 

### checkoing pupil regioons for Baldr on the current calibrated position
img = np.mean( c.get_some_frames( number_of_frames=10, apply_manual_reduction=True ) , axis = 0 ) 
plt.figure(); plt.imshow( img  ) ; plt.colorbar(); plt.savefig('delme.png')


# if using multidevice server phasemask is the MDS server socket
print( f'doing square spiral search for beam {args.beam}')
img_dict = pct.spiral_square_search_and_save_images(
    cam=c,
    beam=args.beam,
    phasemask=state_dict["socket"],
    starting_point=[Xpos, Ypos],
    step_size=args.step_size,
    search_radius=args.search_radius,
    sleep_time=1,
    use_multideviceserver=True,
    )


final_coord = pct.analyse_search_results(img_dict, savepath="delme.png", plot_logscale=True)


if not args.non_verbose:
    save_search_dict = int(input('save the search images (input 1 or 0) - only save if you want to inspect it later'))
else:
    save_search_dict = 1

if save_search_dict:

    with open(args.data_path+f'search_dictionary_beam{args.beam}.json', "w") as json_file:
        json.dump(convert_to_serializable(img_dict), json_file, indent=4)

if not args.non_verbose:
    make_movie = int( input("make and save a movie of the search scan (takes a few minutes)?  (input 1 or 0)") )
else:
    make_movie = 1

if make_movie:
    print('making move.. please wait a minute')
    imgs = np.array( list(  img_dict.values() ) )
    ## Identify bad pixels
    mean_frame = np.mean(imgs, axis=0)
    std_frame = np.std(imgs, axis=0)

    global_mean = np.mean(mean_frame)
    global_std = np.std(mean_frame)
    bad_pixel_map = (np.abs(mean_frame - global_mean) > 6 * global_std) | (std_frame > 20 * np.median(std_frame))

    # remove bad pixels
    filtered_images = pixelmask_image_dict(img_dict, bad_pixel_map)

    create_scatter_image_movie(filtered_images, save_path=args.data_path+f"phasemask_search_movie_beam{args.beam}.mp4", fps=15)

    print( f"COMPLETE. MOVIE SAVED --> ", args.data_path+f"phasemask_search_movie_beam{args.beam}.mp4",)



### Move to the aquired position 

#message = f"fpm_moveabs phasemask{args.beam} {final_coord}"
message = f"moveabs BMX{args.beam} {final_coord[0]}"
res = send_and_get_response(message)
print(res) 

message = f"moveabs BMY{args.beam} {final_coord[1]}"
res = send_and_get_response(message)
print(res) 

# # check image and do fine adjustment 
img = np.mean( c.get_some_frames( number_of_frames=10, apply_manual_reduction=True ) , axis = 0 ) 
plt.figure(); plt.imshow( img  ) ; plt.colorbar(); plt.savefig('delme.png')

###### MANUAL FINE ADJUSTMENT 
if not args.non_verbose:
    pct.move_relative_and_get_image(cam=c, beam=args.beam, phasemask=state_dict["socket"], savefigName='delme.png', use_multideviceserver=True)

message = f"read BMX{args.beam}"
Xpos = float( send_and_get_response(message) )

message = f"read BMY{args.beam}"
Ypos = float( send_and_get_response(message) )



# # Update the current mask position
message = f"fpm_updatemaskpos phasemask{args.beam} {args.phasemask_name}"
res = send_and_get_response(message)


# # Find the most recently modified file
most_recent_file = max(valid_reference_position_files, key=os.path.getmtime)
reference_mask_pos_file = most_recent_file  # this could also be a user input 

print( f'using {reference_mask_pos_file } as the reference file to calculate offsets for the other phase masks relative to the current acquired mask')

# check the reference file 
# Read the JSON file into a dictionary
with open(reference_mask_pos_file, "r") as file:
    mask_positions = json.load(file)

# Separate data for scatter plot
x_positions = []
y_positions = []
labels = []

# Extract data from the dictionary
for mask, position in mask_positions.items():
    x, y = position  # Unpack x, y coordinates
    x_positions.append(x)
    y_positions.append(y)
    labels.append(mask)

# Create scatter plot
plt.figure(figsize=(8, 6))
plt.scatter(x_positions, y_positions, color='blue', s=50, label="Mask Positions")

# Add labels to each point
for label, x, y in zip(labels, x_positions, y_positions):
    plt.text(x, y, label, fontsize=9, ha='right', va='bottom')

# Add plot details
plt.xlabel("X Position")
plt.ylabel("Y Position")
plt.title("Reference Mask Positions \n(used to update all mask positions relative to current)")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.savefig( args.data_path + f'reference_mask_positions_beam{args.beam}.png')




# # Update all other mask positions relative to the current (aquired!) mask
message = f"fpm_updateallmaskpos phasemask{args.beam} {args.phasemask_name} {reference_mask_pos_file}"
res = send_and_get_response(message)


# Initialize the figure
fig, axes = plt.subplots(5, 2, figsize=(10, 15))  # 5 rows, 2 columns

# Loop over J masks (column 1)
for i, mask in enumerate([f"J{x}" for x in range(1, 6)]):

    # Send move command
    message = f"fpm_movetomask phasemask{args.beam} {mask}"
    res = send_and_get_response(message)
    if "ACK" in res:
        print(f'Successfully moved to mask {mask}')
    else:
        print(f'Failed to move to mask {mask}')
        #continue

    time.sleep(3)

    # Capture and process the image
    img = np.mean(c.get_some_frames(number_of_frames=10, apply_manual_reduction=True), axis=0)

    # Plot the image in the grid (column 1, row i)
    ax = axes[i, 0]
    im = ax.imshow(np.log10(img), cmap='viridis')  # Log scale and colormap
    ax.set_title(f"Mask {mask}")
    ax.axis('off')  # Turn off axes for cleaner appearance

    # Add a colorbar with label "ADU"
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("log ADU")

# Loop over H masks (column 2)
for i, mask in enumerate([f"H{x}" for x in range(1, 6)]):
    
    # Send move command
    message = f"fpm_movetomask phasemask{args.beam} {mask}"
    res = send_and_get_response(message)
    if "ACK" in res:
        print(f'Successfully moved to mask {mask}')
    else:
        print(f'Failed to move to mask {mask}')
        #continue

    time.sleep(3)

    # Capture and process the image
    img = np.mean(c.get_some_frames(number_of_frames=10, apply_manual_reduction=True), axis=0)

    # Plot the image in the grid (column 2, row i)
    ax = axes[i, 1]
    im = ax.imshow(np.log10(img), cmap='viridis')  # Log scale and colormap
    ax.set_title(f"Mask {mask}")
    ax.axis('off')  # Turn off axes for cleaner appearance

    # Add a colorbar with label "ADU"
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("log ADU")

# Adjust layout for readability
plt.tight_layout()
plt.savefig('delme.png')
plt.savefig(args.data_path + f'calibrated_phasemasks_beam{args.beam}_{tstamp}.png')


# # write it! 
write2file = int( input('write to file? enter 1 for yes, 0 for no') )

if write2file:
    message = f"fpm_writemaskpos phasemask{args.beam}"
    res = send_and_get_response(message)

    if "ACK" in res:
        print(f'Successfully saved new calibrated phasemask positions for beam {args.beam} to \n config_files/phasemask_positions/beam{args.beam}/')
    else:
        print(f'Failed to save new calibrated phasemask positions')





