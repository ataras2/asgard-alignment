import numpy as np
from astropy.io import fits
import os
import time
import matplotlib.pyplot as plt
import importlib
import json
import datetime
import sys
import pandas as pd
import glob
from asgard_alignment import FLI_Cameras as FLI
import common.DM_basis_functions

"""
Nov 24 - we notice significant drifts likely coming from OAP 1 and solarstein down periscope. 
We have all four Baldr beams on the detector. This script is to run the camera and some DM 
commands to monitor 
(a) the drift of the beams on the detector
(b) the registration of the DM ono the detector 
(c) drift of the beams on the DM 
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import leastsq
from scipy.ndimage import gaussian_filter
from scipy.signal import find_peaks

import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter, label, find_objects

from scipy.ndimage import label, find_objects


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
        plt.show()

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


###################################################################
##### Go

data_path = f"/home/heimdallr/data/stability_tests/"
cross_files = glob.glob(data_path + "*cross*fits")
flat_files = glob.glob(data_path + "stability_tests*fits")

coords = {"1": [], "2": [], "3": [], "4": []}
timestamps = []

for i, f in enumerate(flat_files):
    stamp = f.split("_")[-1].split(".fits")[0]
    timestamps.append(datetime.datetime.strptime(stamp, "%d-%m-%YT%H.%M.%S"))
    with fits.open(f) as d:
        img = np.mean(d["FRAMES"].data[1:, 1:], axis=0)
        if (
            i == 0
        ):  # we define a common cropping for all files!!! very important to compare relative pixel shifts
            crop_pupil_coords = percentile_based_detect_pupils(
                img, percentile=99, min_group_size=100, buffer=20, plot=True
            )
        cropped_pupils = crop_and_sort_pupils(img, crop_pupil_coords)

        for beam, p in zip(["1", "2", "3", "4"], cropped_pupils):
            coords[beam].append(detect_circle(p, sigma=2, threshold=0.5, plot=True))


hrs = np.array(
    [tt.total_seconds() / 60 / 60 for tt in np.array(timestamps) - timestamps[0]]
)

fig, ax = plt.subplots(3, 4, figsize=(10, 10))

ax[0, 0].set_ylabel(f"center X")
ax[1, 0].set_ylabel(f"center Y")
ax[2, 0].set_ylabel(f"radius")
for b in coords:
    # for obs in coords[b]:
    x = [c[0] for c in coords[b]]
    y = [c[1] for c in coords[b]]
    r = [c[2] for c in coords[b]]

    ax[0, int(b) - 1].set_title(f"Beam {b}")
    ax[0, int(b) - 1].plot(hrs, x, ".")
    ax[1, int(b) - 1].plot(hrs, y, ".")
    ax[2, int(b) - 1].plot(hrs, r, ".")

    ax[2, int(b) - 1].set_xlabel(f"time [hours]")

timestamp = datetime.datetime.now().strftime("%d-%m-%YT%H.%M.%S")
plt.tight_layout()
plt.savefig(data_path + f"stability_test_{timestamp}.png")
plt.show()
