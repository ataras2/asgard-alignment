
#!/usr/bin/env python
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import leastsq
from scipy.signal import find_peaks
from scipy.ndimage import gaussian_filter, label, find_objects
from astropy.io import fits
import os
import time
import importlib
import json
import toml
import datetime
import sys
import argparse

from asgard_alignment import FLI_Cameras as FLI

#from common import phasemask_centering_tool as pct

# to use plotting when remote sometimes X11 forwarding is bogus.. so use this: 
# import matplotlib 
# matplotlib.use('Agg')

"""
TO DO: maybe later cut the region of interest in 
half to only get the baldr beams

# if server is stuck 
# sudo lsof -i :5555 then kill the PID 
"""

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


def percentile_based_detect_pupils(
    image,
    percentile=80,
    min_group_size=50,
    buffer=20,
    square_region=True,
    fixed_square_size=32,
    plot=True,
):
    """
    Detects circular pupils by identifying regions with grouped pixels above a given percentile.

    Parameters:
        image (2D array): Full grayscale image containing multiple pupils.
        percentile (float): Percentile of pixel intensities to set the threshold (default 80th).
        min_group_size (int): Minimum number of adjacent pixels required to consider a region.
        buffer (int): Extra pixels to add around the detected region for cropping.
        square_region (bool): If True, return square bounding boxes.
        fixed_square_size (int or None): If not None and square_region is True,
                                         crops fixed-size square regions centered on each pupil.
        plot (bool): If True, displays the detected regions and coordinates.

    Returns:
        list of tuples: Cropping coordinates [(x_start, x_end, y_start, y_end), ...].
    """
    # Normalize the image
    image = image / image.max()

    # Calculate the intensity threshold as the Nth percentile
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
        num_pixels = np.sum(labeled_image[y_slice, x_slice] > 0)
        if num_pixels >= min_group_size:
            y_start = max(0, y_slice.start - buffer)
            y_end = min(image.shape[0], y_slice.stop + buffer)
            x_start = max(0, x_slice.start - buffer)
            x_end = min(image.shape[1], x_slice.stop + buffer)

            if square_region:
                if fixed_square_size is not None:
                    # Center the fixed square on the pupil center
                    x_center = (x_start + x_end) // 2
                    y_center = (y_start + y_end) // 2
                    half = fixed_square_size // 2

                    x_start_sq = max(0, x_center - half)
                    x_end_sq = min(image.shape[1], x_center + half)
                    y_start_sq = max(0, y_center - half)
                    y_end_sq = min(image.shape[0], y_center + half)

                    pupil_regions.append((x_start_sq, x_end_sq, y_start_sq, y_end_sq))
                else:
                    # Auto-size square region based on bounding box
                    max_delta = np.max([x_end - x_start, y_end - y_start])
                    pupil_regions.append(
                        (x_start, x_start + max_delta, y_start, y_start + max_delta)
                    )
            else:
                pupil_regions.append((x_start, x_end, y_start, y_end))

    if plot:
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
        plt.savefig("delme.png")
        plt.show()

    return pupil_regions

# def percentile_based_detect_pupils(
#     image, percentile=80, min_group_size=50, buffer=20, square_region=True, plot=True
# ):
#     """
#     Detects circular pupils by identifying regions with grouped pixels above a given percentile.

#     Parameters:
#         image (2D array): Full grayscale image containing multiple pupils.
#         percentile (float): Percentile of pixel intensities to set the threshold (default 80th).
#         min_group_size (int): Minimum number of adjacent pixels required to consider a region.
#         buffer (int): Extra pixels to add around the detected region for cropping.
#         plot (bool): If True, displays the detected regions and coordinates.

#     Returns:
#         list of tuples: Cropping coordinates [(x_start, x_end, y_start, y_end), ...].
#     """
#     # Normalize the image
#     image = image / image.max()

#     # Calculate the intensity threshold as the 80th percentile
#     threshold = np.percentile(image, percentile)

#     # Create a binary mask where pixels are above the threshold
#     binary_image = image > threshold

#     # Label connected regions in the binary mask
#     labeled_image, num_features = label(binary_image)

#     # Extract regions and filter by size
#     regions = find_objects(labeled_image)
#     pupil_regions = []
#     for region in regions:
#         y_slice, x_slice = region
#         # Count the number of pixels in the region
#         num_pixels = np.sum(labeled_image[y_slice, x_slice] > 0)
#         if num_pixels >= min_group_size:
#             # Add a buffer around the region for cropping
#             y_start = max(0, y_slice.start - buffer)
#             y_end = min(image.shape[0], y_slice.stop + buffer)
#             x_start = max(0, x_slice.start - buffer)
#             x_end = min(image.shape[1], x_slice.stop + buffer)

#             if square_region:
#                 max_delta = np.max( [x_end - x_start,y_end - y_start] )                
#                 pupil_regions.append((x_start, x_start+max_delta, y_start, y_start+max_delta))
#             else:
#                 pupil_regions.append((x_start, x_end, y_start, y_end))
            
#     if plot:
#         # Plot the original image with bounding boxes
#         plt.figure(figsize=(10, 10))
#         plt.imshow(image, cmap="gray", origin="upper")
#         for x_start, x_end, y_start, y_end in pupil_regions:
#             rect = plt.Rectangle(
#                 (x_start, y_start),
#                 x_end - x_start,
#                 y_end - y_start,
#                 edgecolor="red",
#                 facecolor="none",
#                 linewidth=2,
#             )
#             plt.gca().add_patch(rect)
#         plt.title(f"Detected Pupils: {len(pupil_regions)}")
#         plt.savefig('delme.png')
#         plt.show()
        

#     return pupil_regions


def crop_and_sort_pupils(image, pupil_regions):
    """
    Crops regions corresponding to pupils from the image and sorts them
    by their center column index.

    Parameters:
        image (2D array): Full grayscale image.
        pupil_regions (list of tuples): Cropping coordinates [(x_start, x_end, y_start, y_end), ...].

    Returns:
        list of 2D arrays: List of cropped pupil images sorted by center column index.
    """
    # List to store cropped pupil images with their center column index
    cropped_pupils = []

    for region in pupil_regions:
        x_start, x_end, y_start, y_end = region
        # Crop the pupil region
        cropped_image = image[y_start:y_end, x_start:x_end]
        # Calculate the center column index of the region
        center_col = (x_start + x_end) // 2
        cropped_pupils.append((center_col, cropped_image))

    # Sort the cropped pupils by their center column index
    cropped_pupils.sort(key=lambda x: x[0])

    # Extract the sorted cropped images
    sorted_pupil_images = [pupil[1] for pupil in cropped_pupils]

    return sorted_pupil_images


def detect_circle(image, sigma=2, threshold=0.5, plot=True):
    """
    Detects a circular pupil in a cropped image using edge detection and circle fitting.

    Parameters:
        image (2D array): Cropped grayscale image containing a single pupil.
        sigma (float): Standard deviation for Gaussian smoothing.
        threshold (float): Threshold for binarizing edges.
        plot (bool): If True, displays the image with the detected circle overlay.

    Returns:
        tuple: (center_x, center_y, radius) of the detected circle.
    """
    # Normalize the image
    image = image / image.max()

    # Smooth the image to suppress noise
    smoothed_image = gaussian_filter(image, sigma=sigma)

    # Calculate gradients (Sobel-like edge detection)
    grad_x = np.gradient(smoothed_image, axis=1)
    grad_y = np.gradient(smoothed_image, axis=0)
    edges = np.sqrt(grad_x**2 + grad_y**2)

    # Threshold edges to create a binary mask
    binary_edges = edges > (threshold * edges.max())

    # Get edge pixel coordinates
    y, x = np.nonzero(binary_edges)

    # Initial guess for circle parameters
    def initial_guess(x, y):
        center_x, center_y = np.mean(x), np.mean(y)
        radius = np.sqrt(((x - center_x) ** 2 + (y - center_y) ** 2).mean())
        return center_x, center_y, radius

    # Circle model for optimization
    def circle_model(params, x, y):
        center_x, center_y, radius = params
        return np.sqrt((x - center_x) ** 2 + (y - center_y) ** 2) - radius

    # Perform least-squares circle fitting
    guess = initial_guess(x, y)
    result, _ = leastsq(circle_model, guess, args=(x, y))
    center_x, center_y, radius = result

    if plot:
        # Create a circular overlay for visualization
        overlay = np.zeros_like(image)
        yy, xx = np.ogrid[: image.shape[0], : image.shape[1]]
        circle_mask = (xx - center_x) ** 2 + (yy - center_y) ** 2 <= radius**2
        overlay[circle_mask] = 1

        # Plot the image and detected circle
        plt.figure(figsize=(6, 6))
        plt.imshow(image, cmap="gray", origin="upper")
        plt.contour(binary_edges, colors="cyan", linewidths=1, label="Edges")
        plt.contour(overlay, colors="red", linewidths=1, label="Detected Circle")
        plt.scatter(center_x, center_y, color="blue", marker="+", label="Center")
        plt.title("Detected Pupil with Circle Overlay")
        plt.legend()
        plt.show()

    return center_x, center_y, radius




def find_optimal_connected_region(image, connectivity=4, initial_percentile_threshold=95):
    """
    Robust way to detect the pupil which is immune to outliers, noise and uneven illumination. 
    
    Finds the connected region that maximizes the product between the normalized number of pixels
    in the region and the percentile of the lowest pixel value in the group.

    Parameters:
        image (2D array): The input image.
        connectivity (int): Pixel connectivity. Options:
            - 4: Only cardinal neighbors (up, down, left, right).
            - 8: Diagonal neighbors are also considered.
        initial_percentile_threshold (float): Initial percentile threshold to identify connected regions.
        NOTE: The initial percentile threshold should be in the range [0, 100] 
        - results are very sensitive to this parameter. Best to keep it high around 95 

    Returns:
        2D boolean array: A mask where True represents the selected connected region.
    """
    # Ensure the input image is a numpy array
    image = np.asarray(image)

    # Threshold to get initial candidates for connected regions
    mask = image > np.percentile( image, initial_percentile_threshold ) 

    # Define neighbor offsets based on connectivity
    if connectivity == 4:
        neighbors = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    elif connectivity == 8:
        neighbors = [(-1, 0), (1, 0), (0, -1), (0, 1), (-1, -1), (-1, 1), (1, -1), (1, 1)]
    else:
        raise ValueError("Connectivity must be 4 or 8")

    def flood_fill(start):
        """Flood fill algorithm to find connected components."""
        stack = [start]
        region = []
        while stack:
            x, y = stack.pop()
            if (0 <= x < image.shape[0]) and (0 <= y < image.shape[1]) and mask[x, y]:
                region.append((x, y))
                mask[x, y] = False  # Mark as visited
                for dx, dy in neighbors:
                    stack.append((x + dx, y + dy))
        return region

    # Find all connected components
    regions = []
    for x in range(image.shape[0]):
        for y in range(image.shape[1]):
            if mask[x, y]:
                regions.append(flood_fill((x, y)))

    # Initialize variables to track the best region
    best_score = -np.inf
    best_region = None
    total_pixels = image.size  # Total number of pixels in the image

    # Evaluate each region
    for region in regions:
        # Number of pixels in the region (normalized)
        num_pixels = len(region)
        normalized_num_pixels = (num_pixels / total_pixels) * 100

        # Percentile of the lowest pixel value in the region
        region_pixels = np.array([image[x, y] for x, y in region])
        lowest_value = np.min(region_pixels)
        lowest_value_percentile = (np.sum(image <= lowest_value) / image.size) * 100

        # Calculate the score
        score = normalized_num_pixels * lowest_value_percentile

        # Update the best region if the score is higher
        if score > best_score:
            best_score = score
            best_region = region

    # Create a mask for the best region
    final_mask = np.zeros_like(image, dtype=bool)
    for x, y in best_region:
        final_mask[x, y] = True

    return final_mask




# paths and timestamps
tstamp = datetime.datetime.now().strftime("%d-%m-%YT%H.%M.%S")
tstamp_rough =  datetime.datetime.now().strftime("%d-%m-%Y")

# default data paths 
# with open( "config_files/file_paths.json") as f:
#     default_path_dict = json.load(f)
    
# setting up socket to ZMQ communication to multi device server
parser = argparse.ArgumentParser(description="Mode setup")
parser.add_argument(
    '--data_path',
    type=str,
    default="/home/asg/Progs/repos/asgard-alignment/config_files/",
    help="Path to the directory for storing pokeramp data. Default: %(default)s"
)

default_toml = os.path.join("/usr/local/etc/baldr/", "baldr_config_#.toml") #os.path.dirname(os.path.abspath(__file__)), "..", "config_files", "baldr_config.toml")

# TOML file path; default is relative to the current file's directory.
parser.add_argument(
    "--toml_file",
    type=str,
    default=default_toml,
    help="TOML file to write/edit. Default: ../config_files/baldr_config.toml (relative to script)"
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
    default=1,
    help="camera gain. Default: %(default)s"
)
parser.add_argument(
    '--saveformat',
    type=str,
    default='toml',
    help="file type to save pupil coordinates. Default: %(default)s. Options: json, toml"
)

parser.add_argument("--fig_path", type=str, default='', help="path/to/output/image/ for the saved figures")

args = parser.parse_args()

if not os.path.exists(args.data_path):
     print(f'made directory : {args.data_path}')
     os.makedirs(args.data_path)




# baldr_pupils_path = default_path_dict['baldr_pupil_crop'] #"/home/asg/Progs/repos/asgard-alignment/config_files/baldr_pupils_coords.json"

# with open(baldr_pupils_path, "r") as json_file:
#     baldr_pupils = json.load(json_file)

# init camera 
roi = [None, None, None, None]
c = FLI.fli( roi=roi)

#c.send_fli_cmd( "set fps 50")
#c.send_fli_cmd( "set gain 5")

##########
# configure with default configuration file
##########
#config_file_name = os.path.join(c.config_file_path, "default_cred1_config.json")
#c.configure_camera(config_file_name)

# with open(config_file_name, "r") as file:
#     camera_config = json.load(file)

apply_manual_reduction = True

# c.send_fli_cmd("set mode globalresetcds")
# time.sleep(1)
# c.send_fli_cmd(f"set gain {args.cam_gain}")
# time.sleep(1)
# c.send_fli_cmd(f"set fps {args.cam_fps}")

# c.start_camera()


### getting pupil regioons for Baldr 
img =  np.mean( c.get_data(),axis=0)  #c.get_some_frames( number_of_frames=100, apply_manual_reduction=True ) , axis = 0 ) 
#plt.figure(); plt.imshow( np.log10( img ) ) ; plt.savefig('delme.png')

######## ISSUE WHEN WE HAD BAD READNOISE ON LINES 32n
# img[:, ::32] = np.median( img )
# img[:, ::33] = np.median( img )


baldr_mask = np.zeros_like(img).astype(bool)
baldr_mask[img.shape[0]//2 : img.shape[0] , : ] = True # baldr occupies top half (pixels)
heim_mask = ~baldr_mask # heimdallr occupies bottom half

mask = baldr_mask

dict2write = {}

regiom_labels  = ["baldr"] #, "heimdallr"]
mask_list = [baldr_mask, heim_mask]
for mask, lab in zip( mask_list, regiom_labels):
    print(f"looking at {lab}")
    if lab == 'baldr':

        crop_pupil_coords = np.array( percentile_based_detect_pupils(
            img * mask, percentile = 99.3, min_group_size=100, buffer=9, square_region=True,fixed_square_size=32, plot=True
        ) )
    elif lab == 'heimdallr':
        crop_pupil_coords = np.array( percentile_based_detect_pupils(
            img * mask, percentile = 90, min_group_size=50, buffer=20,  square_region=True, plot=True
        ) )
    #cropped_pupils = crop_and_sort_pupils(img, crop_pupil_coords)

    # sorts by rows (indicies are r1,r2,c1,c2)
    sorted_crop_pupil_coords = crop_pupil_coords[crop_pupil_coords[:, 0].argsort()]


    #Swap the coordinates in the dictionary (since CRED one has funny convention with rows and cols)
    if lab == 'baldr':

        if len(sorted_crop_pupil_coords) == 4:
            # Convert to dictionary with keys 4,3,2,1 (order of baldr beams)
            sorted_dict = {str(i): row.tolist() for i, row in zip(['4','3','2','1'],sorted_crop_pupil_coords)}

            # flipped coordinates
            swapped_dict = {
                key: [coords[2], coords[3], coords[0], coords[1]] for key, coords in sorted_dict.items()
            }
            beam_id_2write = ['4','3','2','1']
        else:
            ui = input("4 pupils not detected in Baldr. enter which beams we have from left to right, comma seperated, 0 to exit")
            #ui = '1'
            if '0' in ui:
                raise UserWarning("4 pupils not detected in Baldr.")
            else:
                try:
                    beam_id_2write = [str(int(x.strip())) for x in ui.split(",")]
                except:
                    raise UserWarning("could not convert input string to list of integers. make sure they are comma seperated!")
                #raise UserWarning("4 pupils not detected in Baldr.")
                sorted_dict = {str(i): row.tolist() for i, row in zip(beam_id_2write,sorted_crop_pupil_coords)}

                # flipped coordinates
                swapped_dict = {
                    key: [coords[2], coords[3], coords[0], coords[1]] for key, coords in sorted_dict.items()
                }

            # if ui != '1':
            #     #raise UserWarning("4 pupils not detected in Baldr.")
            #     sorted_dict = {str(i): row.tolist() for i, row in zip(beam_ids,sorted_crop_pupil_coords)}

            #     # flipped coordinates
            #     swapped_dict = {
            #         key: [coords[2], coords[3], coords[0], coords[1]] for key, coords in sorted_dict.items()
            #     }
    elif lab == 'heimdallr':
        if len(sorted_crop_pupil_coords) == 2: 
            # K1 (bright) on the left (low pixels), K2 on right (high pixels)
            sorted_dict = {str(k):v.tolist() for k, v in zip(['K1','K2'], sorted_crop_pupil_coords)}
            # flipped coordinates
            swapped_dict = {
                key: [coords[2], coords[3], coords[0], coords[1]] for key, coords in sorted_dict.items()
            }
        else:
            #ui = input("2 pupils not detected in Heimdallr. enter 1 to contiune, anything else to exit")
            #ui = "1"
            #if ui != '1':
            #    raise UserWarning("2 pupils not detected in Heimdallr.")
            print("WARNING : Clear regions not found for Heimdallr. is K2 aligned?")
            print("We will use hard coded values. Check the image that these are ok!")
            swapped_dict = {"K1" : [ 1, 76, 0, 66], "K2" : [ 9, 67, 255, 310]}

    else:
        raise UserWarning("no valid label.")


    dict2write[lab+"_pupils"] = swapped_dict 

    #print( f'wrote (detected) baldr pupil cropping coordinates to : {args.data_path + "baldr_pupils_coords.json"}')



if args.saveformat=='json':
    json_file_path = os.path.join(args.data_path,f'pupils_coords.json')
    # Save to a JSON file
    with open(json_file_path, "w") as json_file:
        json.dump(dict2write, json_file, indent=4)

    print(f'wrote {json_file_path}')

elif args.saveformat=='toml':

    for beam_id in beam_id_2write:#[1,2,3,4]: 
        
        toml_file_path = args.toml_file.replace('#',f'{beam_id}') #os.path.join(args.data_path, f"baldr_config_{beam_id}.toml")  #")#f'pupils_coords.toml')

        # Check if file exists; if so, load and update.
        if os.path.exists(toml_file_path):
            try:
                current_data = toml.load(toml_file_path)
            except Exception as e:
                # ui = input(f"Error loading TOML file: {e}, do you want to write a new one? (enter 1 if yes. Anybutton if no)")
                # if ui == '1':
                #     current_data = {}
                # else: 
                raise UserWarning(f'exception caught: {e}' ) #user does not want to overwrite')
        else:
            current_data = {}

        #current_data.update(dict2write)

        current_data = recursive_update(current_data, dict2write)
        
        # Write the dictionary to the TOML file
        with open(toml_file_path, "w") as toml_file:
            toml.dump(current_data, toml_file)

        print(f'wrote {toml_file_path}')


### WRITING TO THE CRED 1 SPLIT CONFIG FILE FOR THE SERVER
cred_server_split_file = "/home/asg/.config/cred1_split.json"
# read 

ui = input("enter 1 if we should update cred split json (make sure Cred server is not in split mode before updating) any button otherwise.")

if "1" in ui:
    with open( cred_server_split_file ) as f:
        split_dict = json.load(f)
    
        # "/home/asg/.config/cred1_split.json"
        # "/home/asg/Progs/repos/dcs/asgard-cred1-server/cred1_split.json"

        for beam_id in dict2write["baldr_pupils"]:
            r1,r2,c1,c2 = dict2write["baldr_pupils"][ beam_id ]
            if f"baldr{beam_id}" not in split_dict:
                split_dict[f"baldr{beam_id}"] = {}

            split_dict[f"baldr{beam_id}"].update({
                "x0": c1,
                "y0": r1,
                "xsz": c2-c1,
                "ysz": r2-r1,
            })

    print("writing now")
    #split_dict[f"baldr{beam_id}"]['x0'] = c1
    #split_dict[f"baldr{beam_id}"]['y0'] = r1
    #split_dict[f"baldr{beam_id}"]['xsz'] = c2-c1
    #split_dict[f"baldr{beam_id}"]['ysz'] = r2-r1

    with open(cred_server_split_file, "w") as json_file:
        json.dump(split_dict, json_file, indent=4)
        print(f"wrote split pupil coords to {cred_server_split_file}")


## img[y0: y0+dy, x0:x0+dx] #191:191+40,271:271+40
## util.nice_heatmap_subplots( [ img[r1:r2,c1:c2]],savefig='delme.png')
## convert_index_convention( corners ):


### Plot final results for check
plt.figure(figsize=(8, 8))
plt.imshow(np.log10(img), cmap='gray',origin='upper' ) #, origin='upper') #extent=[0, full_im.shape[1], 0, full_im.shape[0]]
plt.colorbar(label='Intensity')

# Overlay red boxes for each cropping region
for lab in regiom_labels:
    for beam_tmp, (row1, row2, column1, column2) in  dict2write[f"{lab}_pupils"].items():
        plt.plot([column1, column2, column2, column1, column1],
                [row1, row1, row2, row2, row1],
                color='red', linewidth=2, label=f'Beam {beam_tmp}' if beam_tmp == 1 else "")

        plt.text((column1 + column2) / 2, row1 , f'{lab} {beam_tmp}', 
                color='red', fontsize=15, ha='center', va='bottom')

#plt.title('Image with Baldr Cropping Regions')
plt.xlabel('Columns')
plt.ylabel('Rows')
plt.legend(loc='upper right')
print(f"saving images here : {args.fig_path}")
plt.savefig(f'{args.fig_path}' + 'detected_pupils.png')
#plt.show()
plt.close() 

#c.send_fli_cmd( "set gain 1")
#c.send_fli_cmd( "set fps 100")

c.close(erase_file = False) 

print("DONE.")
