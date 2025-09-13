

########################
# To plot the results 

import os
import re
from astropy.io import fits
import numpy as np
import matplotlib.pyplot as plt
import datetime
import glob
from scipy.stats import pearsonr
import itertools
from scipy.optimize import leastsq
from scipy.ndimage import gaussian_filter, label, find_objects
from fpdf import FPDF
from PIL import Image
import argparse
import json 
import matplotlib
from matplotlib.animation import FuncAnimation
from matplotlib.colors import LogNorm
matplotlib.use('Agg')

origin_time_string = "01-11-2024T00.00.00"


class PDFReport(FPDF):
    def header(self):
        # Add a title for the PDF
        self.set_font('Arial', 'B', 12)
        self.cell(0, 10, 'Stability Analysis Report', align='C', ln=1)

    def footer(self):
        # Add a footer with the page number
        self.set_y(-15)
        self.set_font('Arial', 'I', 8)
        self.cell(0, 10, f'Page {self.page_no()}', align='C')

    def add_introduction(self):
        # Add an introduction section
        self.add_page()
        self.set_font('Arial', 'B', 14)
        self.cell(0, 10, 'Introduction', ln=1, align='L')
        self.ln(10)  # Add spacing

        self.set_font('Arial', '', 12)
        intro_text = (
            "This report summarizes the stability analysis of Baldr/Heimdallr, "
            "focusing on the correlation between motor states and pupil parameters over time. "
            "data used in this report is produced from /home/asg/Progs/repos/asgard-alignment/calibration/static_stability_analysis.py"
            "The report is automated from /home/asg/Progs/repos/asgard-alignment/calibration/static_stability_analysis_PLOTS.py"
            "The analysis involves two primary steps:\n\n"
            "1. **Data Acquisition and Preparation**:\n"
            "   - Motor positions are read and logged via a ZeroMQ server connected to the multi-device controller.\n"
            "   - Pupil images are acquired using a C-RED camera system, configured for the required gain and frame rate.\n"
            "   - Motor states and pupil data are saved in FITS format for synchronization and post-processing.\n\n"
            "2. **Pupil Detection**:\n"
            "   - Pupil coordinates (X, Y) and radius are identified using image processing techniques:\n"
            "      - The average of several frames is computed to minimize noise.\n"
            "      - Bright regions corresponding to pupils are segmented based on a user-defined intensity percentile.\n"
            "      - Circular regions are fitted to the segmented pupils using a least-squares optimization method, "
            "detecting the center (X, Y) and radius of each pupil.\n"
            "      - A common cropping region is applied across all images to ensure consistent coordinate comparisons.\n\n"
            "The following pages provide plots from the stability analysis correlating pupil position with motor states. "
            )
        self.multi_cell(0, 10, intro_text)



def process_fits_files(base_directory, output_json):
    """
    Iterates through subdirectories, processes FITS files, and writes the average data to a JSON file.

    Parameters:
        base_directory (str): The base directory containing subdirectories with FITS files.
        output_json (str): The output JSON file to save the results.

    Returns:
        None
    """
    # Initialize the dictionary to store averaged images
    averaged_data = {}

    # Walk through the directory tree
    for root, _, files in os.walk(base_directory):
        for file in files:
            # Process only FITS files
            if file.endswith(".fits"):
                file_path = os.path.join(root, file)

                try:
                    # Open the FITS file
                    with fits.open(file_path) as hdul:
                        # Extract and average the 'FRAMES' extension
                        if 'FRAMES' in hdul:
                            img = np.mean(hdul['FRAMES'].data, axis=0)

                            # Extract the timestamp from the file name
                            timestamp = file.split('_')[-1].replace('.fits', '')

                            # Store the averaged image in the dictionary
                            averaged_data[timestamp] = img.tolist()  # Convert NumPy array to list for JSON serialization
                        else:
                            print(f"'FRAMES' extension not found in {file_path}")
                except Exception as e:
                    print(f"Failed to process {file_path}: {e}")

    # Write the dictionary to a JSON file
    with open(output_json, 'w') as json_file:
        json.dump(averaged_data, json_file, indent=4)
    print(f"Processed data saved to {output_json}")


# base_directory = "/home/heimdallr/data/stability_analysis/24-12-2024/pupils"
# output_json = base_directory + "/averaged_fits_data.json"
# process_fits_files(base_directory, output_json)

def create_movie_from_json(json_file, output_file, crop_region=None, fps=5):
    """
    Creates a movie animation from images stored in a JSON file.

    Parameters:
        json_file (str): Path to the JSON file containing image data.
        output_file (str): Path to save the generated movie (e.g., .mp4 file).
        crop_region (tuple): Optional subregion to crop the images as (xmin, xmax, ymin, ymax).
        fps (int): Frames per second for the movie.

    Returns:
        None
    """
    # Load data from JSON file
    with open(json_file, 'r') as file:
        data = json.load(file)

    # Extract timestamps and images
    timestamps = sorted(data.keys())
    images = [np.array(data[timestamp]) for timestamp in timestamps]

    # Apply cropping if specified
    if crop_region:
        xmin, xmax, ymin, ymax = crop_region
        images = [img[ xmin:xmax, ymin:ymax] for img in images]

    # Create a figure for the animation
    fig, ax = plt.subplots(figsize=(6, 6))
    im = ax.imshow(images[0], cmap='gray', origin='lower', norm=LogNorm(vmin=np.min(images[0]), vmax=np.max(images[0])))
    ax.set_title(timestamps[0], fontsize=10)
    plt.colorbar(im, ax=ax, orientation='vertical', fraction=0.046, pad=0.04)

    # Update function for the animation
    def update(frame):
        im.set_data(images[frame])
        ax.set_title(timestamps[frame], fontsize=10)
        return [im]

    # Create the animation
    anim = FuncAnimation(fig, update, frames=len(images), interval=1000 // fps, blit=True)

    # Save the animation
    if not os.path.exists(os.path.dirname(output_file)):
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
    anim.save(output_file, writer='ffmpeg', fps=fps)
    print(f"Animation saved to {output_file}")

    

def resize_image_to_fit(image_path, max_width, max_height):
    """
    Resize an image to fit within the given dimensions while maintaining the aspect ratio.

    Parameters:
    - image_path (str): Path to the image file.
    - max_width (int): Maximum width in mm for the image in the PDF.
    - max_height (int): Maximum height in mm for the image in the PDF.

    Returns:
    - (width, height): Resized dimensions in mm for the PDF.
    """
    with Image.open(image_path) as img:
        img_width, img_height = img.size
        aspect_ratio = img_width / img_height

        # Calculate dimensions to fit within max_width and max_height
        if (img_width / max_width) > (img_height / max_height):
            width = max_width
            height = max_width / aspect_ratio
        else:
            height = max_height
            width = max_height * aspect_ratio

    return width, height


def get_matching_files(base_directory, subdirectories, pattern):
    """
    Searches for files matching a pattern in the specified subdirectories.

    Parameters:
    - base_directory (str): Path to the base directory.
    - subdirectories (list of str): List of subdirectory names to search in.
    - pattern (str): Filename pattern to match (e.g., "heim_bald_motorstates_*.fits").

    Returns:
    - list of str: List of matching file paths.
    """
    matching_files = []
    for subdir in subdirectories:
        search_path = os.path.join(base_directory, subdir, pattern)
        matching_files.extend(glob.glob(search_path))
    return matching_files


def extract_timestamp_from_filename(filename):
    """
    Extracts a timestamp from a filename in the format heim_bald_motorstates_08-12-2024T18.58.09.fits.
    """
    pattern = r'(\d{2}-\d{2}-\d{4}T\d{2}\.\d{2}\.\d{2})'
    match = re.search(pattern, filename)
    if match:
        return datetime.datetime.strptime(match.group(1), '%d-%m-%YT%H.%M.%S')
    return None


def read_motor_states_fits(fits_path):
    """
    Reads motor states from a FITS file.

    Parameters:
    - fits_path (str): Path to the FITS file.

    Returns:
    - list of dict: Motor states as a list of dictionaries.
    """
    with fits.open(fits_path) as hdul:
        data = hdul["MotorStates"].data
        motor_states = []
        for row in data:
            motor_states.append({
                "name": row["MotorName"],  # No need to decode
                "is_connected": row["IsConnected"],
                "position": row["Position"],
            })
    return motor_states

def plot_motor_states_vs_time(fits_files):
    """
    Reads multiple FITS files and plots motor states vs time.

    Parameters:
    - fits_files (list of str): List of paths to FITS files.
    """
    timestamps = []
    motor_positions = {}

    for fits_file in fits_files:
        timestamp = extract_timestamp_from_filename(os.path.basename(fits_file))
        if not timestamp:
            continue
        timestamps.append(timestamp)
        
        motor_states = read_motor_states_fits(fits_file)
        for motor in motor_states:
            name = motor["name"]
            position = motor.get("position", np.nan)
            if name not in motor_positions:
                motor_positions[name] = []
            motor_positions[name].append(position)

    # Ensure timestamps are sorted
    timestamps, motor_positions = zip(*sorted(zip(timestamps, motor_positions.items())))
    timestamps = np.array(timestamps)

    # Plot each motor's position vs time
    plt.figure(figsize=(12, 8))
    for motor, positions in motor_positions.items():
        plt.plot(timestamps, positions, label=motor)

    plt.xlabel("Time")
    plt.ylabel("Position")
    plt.title("Motor States vs Time")
    plt.legend()
    plt.grid()

    plt.savefig('delme.png')
    plt.close()




def plot_motor_states_subplots(fits_files, motor_names, motor_names_no_beams, savefig=None):
    """
    Reads multiple FITS files and plots motor states in subplots.

    Parameters:
    - fits_files (list of str): List of paths to FITS files.
    - motor_names (list of str): List of motors without beam assignments.
    - motor_names_no_beams (list of str): List of motor groups with beam assignments (e.g., "BMX", "BMY").
    """
    timestamps = []
    motor_positions = {motor: [] for motor in motor_names}
    motor_positions_with_beams = {
        motor: {f"{motor}{beam}": [] for beam in range(1, 5)} for motor in motor_names_no_beams
    }

    for fits_file in fits_files:
        timestamp = extract_timestamp_from_filename(os.path.basename(fits_file))
        if not timestamp:
            continue
        timestamps.append(timestamp)

        motor_states = read_motor_states_fits(fits_file)
        for motor in motor_states:
            name = motor["name"]
            position = motor.get("position", np.nan)

            # Check if it's a motor with no beams
            if name in motor_positions:
                motor_positions[name].append(position)

            # Check if it's a motor with beams
            for group in motor_names_no_beams:
                if name.startswith(group):
                    motor_positions_with_beams[group][name].append(position)

    # Sort timestamps and align positions
    sorted_indices = np.argsort(timestamps)
    timestamps = np.array(timestamps)[sorted_indices]

    motor_positions = {
        motor: np.array(positions)[sorted_indices] for motor, positions in motor_positions.items()
    }
    for group in motor_names_no_beams:
        for motor, positions in motor_positions_with_beams[group].items():
            motor_positions_with_beams[group][motor] = np.array(positions)[sorted_indices]

    # Create subplots
    n_motors = len(motor_names) + len(motor_names_no_beams)
    fig, axes = plt.subplots(n_motors, 1, figsize=(10, 5 * n_motors), sharex=True)

    # Plot motors without beams
    for i, motor in enumerate(motor_names):
        ax = axes[i]
        ax.plot(timestamps, motor_positions[motor], label=motor)
        ax.set_title(f"Motor: {motor}")
        ax.set_ylabel("Position")
        ax.legend()
        ax.grid()

    # Plot motors with beams
    for i, group in enumerate(motor_names_no_beams, start=len(motor_names)):
        ax = axes[i]
        for motor, positions in motor_positions_with_beams[group].items():
            ax.plot(timestamps, positions, label=motor)
        ax.set_title(f"Motor Group: {group}")
        ax.set_ylabel("Position")
        ax.legend()
        ax.grid()

    axes[-1].set_xlabel("Time")
    plt.tight_layout()
    #plt.show()
    if savefig is not None:
        plt.savefig(savefig)
    plt.close() 






def percentile_based_detect_pupils(
    image, percentile=80, min_group_size=50, buffer=20, plot=True, savefig=None,
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
        plt.show()
        if savefig is not None:
            plt.savefig( savefig )
        plt.close()
    return pupil_regions


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


def detect_circle(image, sigma=2, threshold=0.5, plot=True, savefig=None):
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
        if savefig is not None:
            plt.savefig( savefig )

    return center_x, center_y, radius


def extract_motor_positions(fits_files, motor_names, motor_names_no_beams):
    """
    Extracts motor positions from FITS files.

    Parameters:
    - fits_files (list of str): List of paths to FITS files.
    - motor_names (list of str): List of motors without beam assignments.
    - motor_names_no_beams (list of str): List of motor groups with beam assignments (e.g., "BMX", "BMY").

    Returns:
    - dict: Dictionary of motor positions, where keys are motor names and values are lists of positions.
    """
    motor_positions = {motor: [] for motor in motor_names}
    motor_positions_with_beams = {
        f"{motor}{beam}": [] for motor in motor_names_no_beams for beam in range(1, 5)
    }

    for fits_file in fits_files:
        motor_states = read_motor_states_fits(fits_file)
        for motor in motor_states:
            name = motor["name"]
            position = motor.get("position", np.nan)

            # Check if it's a motor with no beams
            if name in motor_positions:
                motor_positions[name].append(position)

            # Check if it's a motor with beams
            if name in motor_positions_with_beams:
                motor_positions_with_beams[name].append(position)

    # Combine the dictionaries
    motor_positions.update(motor_positions_with_beams)
    return motor_positions


# Normalize motor positions for consistent scales
def normalize_motor_positions(motor_positions):
    """
    Normalize motor positions so all have the same scale.

    Parameters:
    - motor_positions (dict): Dictionary of motor positions {motor_name: [positions]}.

    Returns:
    - dict: Dictionary of normalized motor positions {motor_name: [normalized_positions]}.
    """
    min_val = min(min(positions) for positions in motor_positions.values() if len(positions) > 0)
    max_val = max(max(positions) for positions in motor_positions.values() if len(positions) > 0)

    return {
        motor: [(pos - min_val) / (max_val - min_val) for pos in positions]
        for motor, positions in motor_positions.items()
    }

def normalize_and_offset_motor_positions(motor_positions, offset=1.1):
    """
    Normalize motor positions between 0 and 1 and add an offset between motors.

    Parameters:
    - motor_positions (dict): Dictionary of motor positions {motor_name: [positions]}.
    - offset (float): Offset to add between each motor's normalized positions (default is 1.1).

    Returns:
    - dict: Dictionary of normalized and offset motor positions {motor_name: [normalized_positions]}.
    """
    # Find the global min and max for normalization
    min_val = min(min(positions) for positions in motor_positions.values() if positions)
    max_val = max(max(positions) for positions in motor_positions.values() if positions)

    # Normalize and offset positions
    normalized_positions = {}
    for idx, (motor, positions) in enumerate(motor_positions.items()):
        normalized_positions[motor] = [
            (pos - min_val) / (max_val - min_val) + idx * offset for pos in positions
        ]

    return normalized_positions


def correlate_pupil_motor(matching_files_pupils, matching_files_motors):
    """
    Correlates pupil positions (X, Y, Radius) with motor positions based on matching timestamps.

    Parameters:
    - matching_files_pupils (list of str): List of pupil FITS file paths.
    - matching_files_motors (list of str): List of motor FITS file paths.

    Returns:
    - list: Sorted list of correlations [(beam, pupil_param, motor_name, correlation)].
    """
    # Extract pupil positions and timestamps
    coords = get_pupil_positions_from_fits(matching_files_pupils)
    hrs_pupil = get_timestamp_in_hrs(matching_files_pupils)

    # Extract motor positions and timestamps
    motor_positions = get_motor_positions_from_fits(matching_files_motors)
    hrs_motor = get_timestamp_in_hrs(matching_files_motors)

    # Find matching timestamps between pupils and motors
    matched_indices_pupil = []
    matched_indices_motor = []

    for i, t_pupil in enumerate(hrs_pupil):
        for j, t_motor in enumerate(hrs_motor):
            if abs(t_pupil - t_motor) < 1e-3:  # Match within a small tolerance (e.g., milliseconds)
                matched_indices_pupil.append(i)
                matched_indices_motor.append(j)

    # Subset pupil and motor data based on matching timestamps
    matched_coords = {beam: [coords[beam][i] for i in matched_indices_pupil] for beam in coords}
    matched_motor_positions = {
        motor: [positions[j] for j in matched_indices_motor]
        for motor, positions in motor_positions.items()
    }

    # Compute correlations
    correlations = []
    for beam, pupil_data in matched_coords.items():
        for param_idx, param_name in enumerate(["X", "Y", "Radius"]):
            pupil_param = np.array([coord[param_idx] for coord in pupil_data])
            for motor, positions in matched_motor_positions.items():
                if len(pupil_param) >= 2 and len(positions) >= 2:  # Ensure lengths are valid for correlation
                    corr, _ = pearsonr(pupil_param, positions)
                    correlations.append((beam, param_name, motor, corr))

    # Sort by absolute correlation strength
    correlations.sort(key=lambda x: abs(x[3]), reverse=True)
    return correlations








def get_motor_positions_from_fits(fits_files):
    """
    Extracts motor positions from a list of FITS files.

    Parameters:
    - fits_files (list of str): List of paths to FITS files.

    Returns:
    - dict: Dictionary of motor positions {motor_name: [positions]}.
    """
    motor_positions = {}

    for fits_file in fits_files:
        motor_states = read_motor_states_fits(fits_file)  # Use the provided function
        for motor in motor_states:
            name = motor["name"]
            position = motor.get("position", np.nan)
            if name not in motor_positions:
                motor_positions[name] = []
            motor_positions[name].append(position)

    return motor_positions

def get_pupil_positions_from_fits(matching_files_pupils): 
    coords = {"1": [], "2": [], "3": [], "4": []}
    timestamps = []

    for i, f in enumerate(matching_files_pupils):
        stamp = f.split("_")[-1].split(".fits")[0]
        timestamps.append(datetime.datetime.strptime(stamp, "%d-%m-%YT%H.%M.%S"))
        with fits.open(f) as d:
            img = np.mean(d["FRAMES"].data[1:, 1:], axis=0) # get rid of image tags 

            # FILTER OUT HEIMDALLR !!!
            img[0:150] = np.median( img )
            ##########################

            if (
                i == 0
            ):  # we define a common cropping for all files!!! very important to compare relative pixel shifts
                crop_pupil_coords = percentile_based_detect_pupils(
                    img, percentile=99, min_group_size=100, buffer=20, plot=True
                )
            cropped_pupils = crop_and_sort_pupils(img, crop_pupil_coords)

            for beam, p in zip(["1", "2", "3", "4"], cropped_pupils):
                coords[beam].append(detect_circle(p, sigma=2, threshold=0.5, plot=True))


    # hrs = np.array(
    #     [tt.total_seconds() / 60 / 60 for tt in np.array(timestamps) - timestamps[0]]
    # )
    return coords  #, hrs )


def get_timestamp_in_hrs( fits_files ):
    timestamps = []

    zero_timestamp = datetime.datetime.strptime(origin_time_string, "%d-%m-%YT%H.%M.%S")
    for i, f in enumerate(fits_files):
        stamp = f.split("_")[-1].split(".fits")[0]
        timestamps.append(datetime.datetime.strptime(stamp, "%d-%m-%YT%H.%M.%S"))

    hrs = np.array(
        [tt.total_seconds() / 60 / 60 for tt in np.array(timestamps) - zero_timestamp ]
    )


    return hrs 



def plot_all_beam_and_motor_positions(motor_positions, coords, hrs_pupil, hrs_motor, offset= 0.1 , savefig = None):


    # Create a color mapping for motors
    motor_names = set(motor[:-1] for motor in motor_positions if motor[-1].isdigit())  # Strip beam numbers
    color_cycle = itertools.cycle(plt.cm.tab10.colors)  # Use a colormap (e.g., tab10)
    motor_color_map = {motor: next(color_cycle) for motor in motor_names}

    # Start plotting
    fig, ax = plt.subplots(4, 4, figsize=(12, 12))

    # Set labels for the first three rows
    ax[0, 0].set_ylabel("center X")
    ax[1, 0].set_ylabel("center Y")
    ax[2, 0].set_ylabel("radius")

    # Normalize motor positions with offsets
    offset_motor_positions = normalize_and_offset_motor_positions(motor_positions, offset= offset)

    # Loop through beams
    handles = {}  # For shared legend
    for b in coords:
        # Extract pupil parameters
        x = [c[0] for c in coords[b]]
        y = [c[1] for c in coords[b]]
        r = [c[2] for c in coords[b]]

        beam_index = int(b) - 1  # Convert beam to zero-based index

        # Plot pupil parameters
        ax[0, beam_index].set_title(f"Beam {b}")
        ax[0, beam_index].plot(hrs_pupil - hrs_pupil[0], x, ".", label="X")
        ax[1, beam_index].plot(hrs_pupil - hrs_pupil[0], y, ".", label="Y")
        ax[2, beam_index].plot(hrs_pupil - hrs_pupil[0], r, ".", label="Radius")

        # Set x-label for the third row
        ax[2, beam_index].set_xlabel("Time [hours]")

        # Plot normalized motor positions for the beam
        for motor, positions in offset_motor_positions.items():
            motor_base = motor[:-1]  # Remove beam number
            if motor.endswith(b):  # Check if motor belongs to the current beam
                line, = ax[3, beam_index].plot(
                    hrs_motor - hrs_motor[0], positions, ".", label=motor_base,
                    color=motor_color_map[motor_base]
                )
                # Only add to legend if it's the first occurrence of the motor
                if motor_base not in handles:
                    handles[motor_base] = line

        # Label the fourth row
        ax[3, beam_index].set_xlabel("Time [hours]")
        ax[3, beam_index].set_ylabel("Normalized Position")

    # Add a shared legend below the plot
    plt.subplots_adjust(bottom=0.2)  # Add space at the bottom for the legend
    fig.legend(handles.values(), handles.keys(), loc="lower center", fontsize=20, ncol=3)

    # Save and show the plot
    plt.tight_layout(rect=[0, 0.25, 1, 1])  # Adjust rect to avoid overlap with the legend
    timestamp = datetime.datetime.now().strftime("%d-%m-%YT%H.%M.%S")
    if savefig is not None:
        plt.savefig(savefig)
    plt.show()


def save_pupil_plots(matching_files_pupils, coords, fig_path):
    """
    Save 1x4 pupil plots for each timestamp.

    Parameters:
    - matching_files_pupils (list of str): List of pupil FITS file paths.
    - coords (dict): Dictionary of pupil coordinates for beams.
    - fig_path (str): Path to save pupil plots.
    """
    pupil_plots_path = os.path.join(fig_path, 'pupil_plots')
    if not os.path.exists(pupil_plots_path):
        os.makedirs(pupil_plots_path)

    for i, file in enumerate(matching_files_pupils):
        # Open the FITS file and extract the image
        with fits.open(file) as d:
            img = np.mean(d["FRAMES"].data[1:, 1:], axis=0)

            # FILTER OUT HEIMDALLR !!!
            img[0:150] = np.median( img )
            ##########################

            if (
                i == 0
            ):  # we define a common cropping for all files!!! very important to compare relative pixel shifts
                crop_pupil_coords = percentile_based_detect_pupils(
                    img, percentile=99, min_group_size=100, buffer=20, plot=False
                )
        
        cropped_pupils = crop_and_sort_pupils(img, crop_pupil_coords)

        # Plot the 1x4 pupil images
        fig, ax = plt.subplots(1, 4, figsize=(16, 4))
        for j, pupil in enumerate(cropped_pupils):
            ax[j].imshow(pupil, cmap='gray')
            ax[j].set_title(f'Pupil {j + 1}')
            ax[j].axis('off')

        # Save the plot with the timestamp
        timestamp = extract_timestamp_from_filename(file)
        hr = (timestamp - datetime.datetime.strptime(origin_time_string, "%d-%m-%YT%H.%M.%S")).total_seconds() / 3600
        plot_path = os.path.join(pupil_plots_path, f'pupil_{hr:.2f}.png')
        plt.savefig(plot_path)
        plt.close()


def add_pupil_plots_to_pdf(pdf, fig_path):
    """
    Add pupil plots to the PDF report.

    Parameters:
    - pdf (PDFReport): PDF instance to add the plots to.
    - fig_path (str): Path where pupil plots are saved.
    """
    pupil_plots_path = os.path.join(fig_path, 'pupil_plots')
    pupil_files = sorted(os.listdir(pupil_plots_path))  # Ensure order by timestamp

    for i in range(0, len(pupil_files), 5):  # 5 plots per page
        pdf.add_page()
        pdf.set_font('Arial', 'B', 16)
        pdf.cell(0, 10, 'Pupil Images', ln=1, align='C')
        pdf.ln(10)

        for j, pupil_file in enumerate(pupil_files[i:i + 5]):
            # Resize and add each pupil plot
            plot_path = os.path.join(pupil_plots_path, pupil_file)
            x = 10
            y = 30 + (50 * j)  # Stack 5 plots vertically per page
            width, height = resize_image_to_fit(plot_path, max_width=190, max_height=45)
            pdf.image(plot_path, x=x, y=y, w=width, h=height)


###################################################################
##### Go

if __name__ == "__main__":
    # Argument parser
    parser = argparse.ArgumentParser(description="Generate stability analysis report.")
    parser.add_argument(
        "--base_directory",
        type=str,
        default="/home/heimdallr/data/stability_analysis/",
        help="Base directory for data storage (default: %(default)s)."
    )
    parser.add_argument(
        "--subdirectories",
        type=str,
        nargs="+",
        default=["06-12-2024", "07-12-2024", "08-12-2024"],
        help="Subdirectories to include (default: %(default)s)."
    )
    parser.add_argument(
        "--pupil_pattern",
        type=str,
        default="heim_bald_pupils_*.fits",
        help="Filename pattern for pupil FITS files (default: %(default)s)."
    )
    parser.add_argument(
        "--motor_pattern",
        type=str,
        default="heim_bald_motorstates_*.fits",
        help="Filename pattern for motor states FITS files (default: %(default)s)."
    )
    args = parser.parse_args()

    # Output directory for plots and report
    fig_path = '/home/asg/Progs/repos/asgard-alignment/calibration/reports/stability/'
    if not os.path.exists(fig_path):
        os.makedirs(fig_path)
        print(f"Created directory: {fig_path}")
    else:
        print(f"Directory already exists: {fig_path}")

    # Get the list of matching files
    matching_files_pupils = get_matching_files(args.base_directory, args.subdirectories, args.pupil_pattern)
    matching_files_motors = get_matching_files(args.base_directory, args.subdirectories, args.motor_pattern)

    coords = get_pupil_positions_from_fits(matching_files_pupils)
    hrs_pupil = get_timestamp_in_hrs(matching_files_pupils)
    motor_positions = get_motor_positions_from_fits(matching_files_motors)
    hrs_motor = get_timestamp_in_hrs(matching_files_motors)

    # Define motor names
    motor_names = ["SDLA", "SDL12", "SDL34", "SSS", "BFO"]
    motor_names_no_beams = [
        "HFO", "HTPP", "HTPI", "HTTP", "HTTI", "BDS", "BTT", "BTP", "BMX", "BMY"
    ]

    save_pupil_plots(matching_files_pupils, coords, fig_path)

    # Plot example of pupil detection for report
    f = matching_files_pupils[0]
    with fits.open(f) as d:
        img = np.mean(d["FRAMES"].data[1:, 1:], axis=0)
        img[0:150] = np.median(img)  # FILTER OUT HEIMDALLR !!!

        crop_pupil_coords = percentile_based_detect_pupils(
            img, percentile=99, min_group_size=100, buffer=20, plot=True, savefig=fig_path + 'baldr_pup_detection.png'
        )

    # Plot motor states
    plot_motor_states_subplots(matching_files_motors, motor_names, motor_names_no_beams, savefig=fig_path + 'motor_states.png')

    # Plot normalized motor states along with registered pupil positions
    plot_all_beam_and_motor_positions(motor_positions, coords, hrs_pupil, hrs_motor, offset=0.1, savefig=fig_path + 'motor_states_w_pupil.png')

    # Generate PDF report
    timestamp = datetime.datetime.now().strftime("%d-%m-%YT%H.%M.%S")
    pdf_report_path = os.path.join(fig_path, f'stability_analysis_report_{timestamp}.pdf')

    pdf = PDFReport()
    pdf.set_auto_page_break(auto=True, margin=15)

    pdf.add_introduction()

    # Add BALDR pupil detection image
    pdf.add_page()
    pdf.set_font('Arial', 'B', 16)
    pdf.cell(0, 10, 'BALDR PUPIL DETECTION', ln=1, align='C')
    pdf.ln(10)
    pup_detection_path = fig_path + 'baldr_pup_detection.png'
    width, height = resize_image_to_fit(pup_detection_path, max_width=190, max_height=250)
    pdf.image(pup_detection_path, x=10, y=30, w=width, h=height)

    # Add Motor States plot
    pdf.add_page()
    pdf.set_font('Arial', 'B', 16)
    pdf.cell(0, 10, 'Motor States Over Time', ln=1, align='C')
    pdf.ln(10)
    motor_states_path = fig_path + 'motor_states.png'
    width, height = resize_image_to_fit(motor_states_path, max_width=190, max_height=250)
    pdf.image(motor_states_path, x=10, y=30, w=width, h=height)

    # Add Motor States and Pupil Parameters plot
    pdf.add_page()
    pdf.set_font('Arial', 'B', 16)
    pdf.cell(0, 10, 'Motor States and Pupil Parameters', ln=1, align='C')
    pdf.ln(10)
    motor_states_w_pupil_path = fig_path + 'motor_states_w_pupil.png'
    width, height = resize_image_to_fit(motor_states_w_pupil_path, max_width=190, max_height=250)
    pdf.image(motor_states_w_pupil_path, x=10, y=30, w=width, h=height)

    # add the pupil images 
    add_pupil_plots_to_pdf(pdf, fig_path)

    # Save the PDF
    pdf.output(pdf_report_path)
    print(f"PDF report saved to: {pdf_report_path}")

# if __name__=="__main__":

#     fig_path = '/home/asg/Progs/repos/asgard-alignment/calibration/reports/stability/'
#     if not os.path.exists(fig_path):
#         # Create the directory (and any necessary parent directories)
#         os.makedirs(fig_path)
#         print(f"Created directory: {fig_path}")
#     else:
#         print(f"Directory already exists: {fig_path}")


#     ####
#     # PUPIL IN CRED 1 
#     ####
#     base_directory = "/home/heimdallr/data/stability_analysis/"
#     subdirectories = ["06-12-2024", "07-12-2024", "08-12-2024"]
   
#     pupil_pattern = "heim_bald_pupils_*.fits"

#     motor_pattern = "heim_bald_motorstates_*.fits"

#     # Get the list of matching files
#     matching_files_pupils = get_matching_files(base_directory, subdirectories, pupil_pattern)

#     matching_files_motors = get_matching_files(base_directory, subdirectories, motor_pattern)


#     coords = get_pupil_positions_from_fits(matching_files_pupils)
#     hrs_pupil = get_timestamp_in_hrs( matching_files_pupils ) 

#     motor_positions = get_motor_positions_from_fits( matching_files_motors )
#     hrs_motor = get_timestamp_in_hrs( matching_files_motors ) 

#     # Define motor names
#     motor_names = ["SDLA", "SDL12", "SDL34", "SSS", "BFO"]
#     motor_names_no_beams = [
#         "HFO", "HTPP", "HTPI", "HTTP", "HTTI", "BDS", "BTT", "BTP", "BMX", "BMY"
#     ]


#     # plot example of pupil detection for report
#     f = matching_files_pupils[0]

#     with fits.open(f) as d:
#         img = np.mean(d["FRAMES"].data[1:, 1:], axis=0)
        
#         # FILTER OUT HEIMDALLR !!!
#         img[0:150] = np.median( img )


#         crop_pupil_coords = percentile_based_detect_pupils(
#             img, percentile=99, min_group_size=100, buffer=20, plot=True, savefig=fig_path +'baldr_pup_detection.png')


#     # Plot motor states
#     plot_motor_states_subplots(matching_files_motors, motor_names, motor_names_no_beams, savefig=fig_path +'motor_states.png')

#     # plot normalized motor states along with registered pupil positions
#     plot_all_beam_and_motor_positions(motor_positions, coords, hrs_pupil, hrs_motor, offset= 0.1 , savefig = fig_path + 'motor_states_w_pupil.png')


#     # get correlations (TO DO - Plot these )
#     #correlations = correlate_pupil_motor(coords, motor_positions, motor_names_no_beams)

#     # Path to save the PDF
#     timestamp = datetime.datetime.now().strftime("%d-%m-%YT%H.%M.%S")
#     pdf_report_path = os.path.join(fig_path, f'stability_analysis_report_{timestamp}.pdf')

#     # Create a PDF instance
#     pdf = PDFReport()

#     pdf.set_auto_page_break(auto=True, margin=15)

#     pdf.add_introduction()

#     # baldr pupil detection
#     pdf.add_page()
#     pdf.set_font('Arial', 'B', 16)
#     pdf.cell(0, 10, 'BALDR PUPIL DECTION', ln=1, align='C')
#     pdf.ln(10)  # Add vertical spacing

#     pup_detection_path =fig_path +'baldr_pup_detection.png'
#     width, height = resize_image_to_fit(pup_detection_path, max_width=190, max_height=250)
#     pdf.image(pup_detection_path, x=10, y=30, w=width, h=height)



#     # Add the first page with motor states plot
#     pdf.add_page()
#     pdf.set_font('Arial', 'B', 16)
#     pdf.cell(0, 10, 'Motor States Over Time', ln=1, align='C')
#     pdf.ln(10)  # Add vertical spacing

#     # Resize and add the motor states image
#     motor_states_path = fig_path + 'motor_states.png'
#     width, height = resize_image_to_fit(motor_states_path, max_width=190, max_height=250)  # Adjust max dimensions as needed
#     pdf.image(motor_states_path, x=10, y=30, w=width, h=height)

#     # Add the second page with motor states and pupil data plot
#     pdf.add_page()
#     pdf.set_font('Arial', 'B', 16)
#     pdf.cell(0, 10, 'Motor States and Pupil Parameters', ln=1, align='C')
#     pdf.ln(10)  # Add vertical spacing

#     # Add the second plot image
#     motor_states_w_pupil_path = fig_path + 'motor_states_w_pupil.png'
#     width, height = resize_image_to_fit(motor_states_w_pupil_path, max_width=190, max_height=250)
#     pdf.image(motor_states_w_pupil_path, x=10, y=30, w=width, h=height)

#     # Save the PDF
#     pdf.output(pdf_report_path)

#     print(f"PDF report saved to: {pdf_report_path}")



# fig, ax = plt.subplots(3, 4, figsize=(10, 10))

# ax[0, 0].set_ylabel(f"center X")
# ax[1, 0].set_ylabel(f"center Y")
# ax[2, 0].set_ylabel(f"radius")
# for b in coords:
#     # for obs in coords[b]:
#     x = [c[0] for c in coords[b]]
#     y = [c[1] for c in coords[b]]
#     r = [c[2] for c in coords[b]]

#     ax[0, int(b) - 1].set_title(f"Beam {b}")
#     ax[0, int(b) - 1].plot(hrs_pupil, x, ".")
#     ax[1, int(b) - 1].plot(hrs_pupil, y, ".")
#     ax[2, int(b) - 1].plot(hrs_pupil, r, ".")

#     ax[2, int(b) - 1].set_xlabel(f"time [hours]")

# timestamp = datetime.datetime.now().strftime("%d-%m-%YT%H.%M.%S")
# plt.tight_layout()
# plt.savefig(data_path + f"stability_test_{timestamp}.png")
# plt.show()






# # Combined plot of s
# fig, ax = plt.subplots(4, 4, figsize=(12, 12))

# # Set labels for the first three rows
# ax[0, 0].set_ylabel("center X")
# ax[1, 0].set_ylabel("center Y")
# ax[2, 0].set_ylabel("radius")

# # Normalize motor positions
# normalized_motor_positions = normalize_and_offset_motor_positions(motor_positions, offset=.1) #normalize_motor_positions(motor_positions)


# # Loop through beams
# include_legend = False
# for b in coords:
#     # Extract pupil parameters
#     x = [c[0] for c in coords[b]]
#     y = [c[1] for c in coords[b]]
#     r = [c[2] for c in coords[b]]

#     beam_index = int(b) - 1  # Convert beam to zero-based index

#     # Plot pupil parameters
#     ax[0, beam_index].set_title(f"Beam {b}")
#     ax[0, beam_index].plot(hrs_pupil-hrs_pupil[0], x, ".", label="X")
#     ax[1, beam_index].plot(hrs_pupil-hrs_pupil[0], y, ".", label="Y")
#     ax[2, beam_index].plot(hrs_pupil-hrs_pupil[0], r, ".", label="Radius")

#     # Set x-label for the third row
#     ax[2, beam_index].set_xlabel("Time [hours]")

#     # Plot normalized motor positions for the beam
#     for motor, positions in normalized_motor_positions.items():
#         if motor.endswith(b):  # Check if motor belongs to the current beam
#             ax[3, beam_index].plot(hrs_motor-hrs_motor[0], positions, ".", label=motor)

#     # Label the fourth row
#     ax[3, beam_index].set_xlabel("Time [hours]")
#     ax[3, beam_index].set_ylabel("Normalized Position")
#     if include_legend:
#         ax[3, beam_index].legend(fontsize=8, loc="upper right")

# # Adjust layout and save
# plt.tight_layout()
# timestamp = datetime.datetime.now().strftime("%d-%m-%YT%H.%M.%S")
# plt.savefig( 'delme.png' )#data_path + f"stability_test_{timestamp}.png")
# plt.show()
