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


# python -i calibration/phasemask_raster.py --beam 2 --initial_pos 10,10 --dx 500 --dy 3000 --width 9000 --height 9800 --orientation 0 

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



def plot_cluster_heatmap(x_positions, y_positions, clusters, show_grid=True, grid_color="white", grid_linewidth=0.5):
    """
    Creates a 2D heatmap of cluster numbers vs x, y positions, with an optional grid overlay.

    Parameters:
        x_positions (list or array): List of x positions.
        y_positions (list or array): List of y positions.
        clusters (list or array): Cluster numbers corresponding to the x, y positions.
        show_grid (bool): If True, overlays a grid on the heatmap.
        grid_color (str): Color of the grid lines (default is 'white').
        grid_linewidth (float): Linewidth of the grid lines (default is 0.5).

    Returns:
        None
    """
    # Convert inputs to NumPy arrays
    x_positions = np.array(x_positions)
    y_positions = np.array(y_positions)
    clusters = np.array(clusters)

    # Ensure inputs have the same length
    if len(x_positions) != len(y_positions) or len(x_positions) != len(clusters):
        raise ValueError("x_positions, y_positions, and clusters must have the same length.")

    # Get unique x and y positions to define the grid
    unique_x = np.unique(x_positions)
    unique_y = np.unique(y_positions)

    # Create an empty grid to store cluster numbers
    heatmap = np.full((len(unique_y), len(unique_x)), np.nan)  # Use NaN for empty cells

    # Map each (x, y) to grid indices
    x_indices = np.searchsorted(unique_x, x_positions)
    y_indices = np.searchsorted(unique_y, y_positions)

    # Fill the heatmap with cluster values
    for x_idx, y_idx, cluster in zip(x_indices, y_indices, clusters):
        heatmap[y_idx, x_idx] = cluster

    # Plot the heatmap
    fig, ax = plt.subplots(figsize=(8, 6))
    cmap = plt.cm.get_cmap('viridis', len(np.unique(clusters)))  # Colormap with distinct colors
    cax = ax.imshow(heatmap, origin='lower', cmap=cmap, extent=[unique_x.min(), unique_x.max(), unique_y.min(), unique_y.max()])

    # Add colorbar
    cbar = fig.colorbar(cax, ax=ax, ticks=np.unique(clusters))
    cbar.set_label('Cluster Number', fontsize=12)

    # Label the axes
    ax.set_xlabel('X Position', fontsize=12)
    ax.set_ylabel('Y Position', fontsize=12)
    ax.set_title('Cluster Heatmap', fontsize=14)

    # Add grid overlay if requested
    if show_grid:
        ax.set_xticks(unique_x, minor=True)
        ax.set_yticks(unique_y, minor=True)
        ax.grid(which="minor", color=grid_color, linestyle="-", linewidth=grid_linewidth)
        ax.tick_params(which="minor", length=0)  # Hide minor tick marks

    plt.tight_layout()
    plt.show()




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

parser.add_argument(
    '--data_path',
    type=str,
    default=f"/home/heimdallr/Documents/asgard-alignment/calibration/reports/phasemask_aquisition/{tstamp_rough}/", #f"/home/heimdallr/data/phasemask_aquisition/{tstamp_rough}/",
    help="Path to the directory for storing pokeramp data. Default: %(default)s"
)
parser.add_argument(
    '--beam',
    type=str,
    default="1",
    help="what beam to look at?. Default: %(default)s"
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
    '--dx',
    type=float,
    default=20,
    help="set size in microns of x increments in raster. Default: %(default)s"
)

parser.add_argument(
    '--dy',
    type=float,
    default=1000,
    help="set size in microns of y increments in raster. Default: %(default)s"
)

parser.add_argument(
    '--width',
    type=float,
    default=1000,
    help="set width (x-axis in local frame) in microns of raster search. Default: %(default)s"
)

parser.add_argument(
    '--height',
    type=float,
    default=1000,
    help="set height (y-axis in local frame) in microns of raster search. Default: %(default)s"
)

parser.add_argument(
    '--orientation',
    type=float,
    default=0,
    help="rotate frame in degrees of local axes of raster search. Default: %(default)s"
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



message = f"!read BMX{args.beam}"
initial_Ypos = float(send_and_get_response(message))

message = f"!read BMX{args.beam}"
initial_Xpos = float(send_and_get_response(message))

# home mask.. 
# message = f"!init BMX2"
# res = send_and_get_response(message)
# print(res)
# message = f"!init BMY2"
# res = send_and_get_response(message)
# print(res)

baldr_pupils_path = default_path_dict['baldr_pupil_crop'] #"/home/heimdallr/Documents/asgard-alignment/config_files/baldr_pupils_coords.json"

with open(baldr_pupils_path, "r") as json_file:
    baldr_pupils = json.load(json_file)



# init camera 
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
plt.savefig('delme.png')



# goiing to initial position 
try: 
    tmp_pos = args.initial_pos.split( ',')

    Xpos = int( tmp_pos[0] )  # start_position_dict[args.phasemask_name][0]
    Ypos = int( tmp_pos[1] ) # start_position_dict[args.phasemask_name][1]
    print(f'using user input initial position x,y ={Xpos}, {Ypos}')
except: 
    raise UserWarning( 'invalid user input {args.initial_pos} for initial position x,y. Try (for example) --initial_pos 5000,5000  (i.e they are comma-separated)' )

message = f"!moveabs BMX{args.beam} {Xpos}"
res = send_and_get_response(message)
print(res) 

message = f"!moveabs BMY{args.beam} {Ypos}"
res = send_and_get_response(message)
print(res) 



# if using multidevice server phasemask is the MDS server socket
print( f'doing raster search for beam {args.beam}')

img_dict = pct.raster_square_search_and_save_images(
    cam=c,
    beam=args.beam,
    phasemask=state_dict["socket"],
    starting_point=[Xpos, Ypos],
    dx=args.dx, 
    dy=args.dy, 
    width=args.width, 
    height=args.height, 
    orientation=args.orientation,
    sleep_time=1,
    use_multideviceserver=True,
    plot_grid_before_scan=args.non_verbose
)
    
#final_coord = pct.analyse_search_results(img_dict, savepath="delme.png", plot_logscale=True)


if not args.non_verbose:
    save_search_dict = int(input('save the search images (input 1 or 0) - only save if you want to inspect it later'))
else:
    save_search_dict = 1

if save_search_dict:
    with open(args.data_path+f'raster_search_dictionary_beam{args.beam}.json', "w") as json_file:
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
    filtered_images = pct.pixelmask_image_dict(img_dict, bad_pixel_map)

    create_scatter_image_movie(filtered_images, save_path=args.data_path+f"phasemask_search_movie_beam{args.beam}.mp4", fps=15)

    print( f"COMPLETE. MOVIE SAVED --> ", args.data_path+f"raster_search_movie_beam{args.beam}.mp4")



### Move back to the initial position 

message = f"!moveabs BMX{args.beam} {initial_Xpos}"
res = send_and_get_response(message)
print(res) 

message = f"!moveabs BMY{args.beam} {initial_Ypos}"
res = send_and_get_response(message)
print(res) 


### cluster analysis on the search results
bad_pixel_map = pct.create_bad_pixel_mask( img_dict, mean_thresh=6, std_thresh=20 )
masked_search_dict = pct.pixelmask_image_dict(img_dict, bad_pixel_map)
# cluster analysis on the fitted pupil center and radius 
image_list = np.array( list( masked_search_dict.values() ) ) 
res = pct.cluster_analysis_on_searched_images(images= image_list,
                                          detect_circle_function=pct.detect_circle, 
                                          n_clusters=6, 
                                          plot_clusters=False)



positions = [eval(str(key)) for key in masked_search_dict.keys()]
x_positions, y_positions = zip(*positions)

# extent  = (min(x_positions), max(x_positions), min(y_positions), max(y_positions))



plot_cluster_heatmap( x_positions,  y_positions ,  res['clusters'] ) 
plt.savefig('delme.png')

pct.plot_aggregate_cluster_images(images = image_list, clusters = res['clusters'], operation="std")
plt.savefig('delme2.png')


plt.close('all')

print('Finished')