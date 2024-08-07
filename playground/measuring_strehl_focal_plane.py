#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug  2 15:11:39 2024

@author: bencb

measuring Strehl with 635nm laser on point grey camera at phase mask
focal plane (after OAP). 

method
-select region. 
-generate the measured and theoretical PSF in region
-subtract bias etc in measured image. 
-normalize intensities in each image by sum of pixels within region
-take ratio of peak intensity 


"""

import numpy as np
import pandas as pd
import sys
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import scipy
from mpl_toolkits.axes_grid1 import make_axes_locatable
from scipy.interpolate import griddata

# from scipy import special
# from scipy import interpolate


# ========== PLOTTING STANDARDS
def nice_heatmap_subplots(
    im_list,
    xlabel_list,
    ylabel_list,
    title_list,
    cbar_label_list,
    fontsize=15,
    cbar_orientation="bottom",
    axis_off=True,
    vlims=None,
    savefig=None,
):

    n = len(im_list)
    fs = fontsize
    fig = plt.figure(figsize=(5 * n, 5))

    for a in range(n):
        ax1 = fig.add_subplot(int(f"1{n}{a+1}"))
        ax1.set_title(title_list[a], fontsize=fs)

        if vlims != None:
            im1 = ax1.imshow(im_list[a], vmin=vlims[a][0], vmax=vlims[a][1])
        else:
            im1 = ax1.imshow(im_list[a])
        ax1.set_title(title_list[a], fontsize=fs)
        ax1.set_xlabel(xlabel_list[a], fontsize=fs)
        ax1.set_ylabel(ylabel_list[a], fontsize=fs)
        ax1.tick_params(labelsize=fs)

        if axis_off:
            ax1.axis("off")
        divider = make_axes_locatable(ax1)
        if cbar_orientation == "bottom":
            cax = divider.append_axes("bottom", size="5%", pad=0.05)
            cbar = fig.colorbar(im1, cax=cax, orientation="horizontal")

        elif cbar_orientation == "top":
            cax = divider.append_axes("top", size="5%", pad=0.05)
            cbar = fig.colorbar(im1, cax=cax, orientation="horizontal")

        else:  # we put it on the right
            cax = divider.append_axes("right", size="5%", pad=0.05)
            cbar = fig.colorbar(im1, cax=cax, orientation="vertical")

        cbar.set_label(cbar_label_list[a], rotation=0, fontsize=fs)
        cbar.ax.tick_params(labelsize=fs)
    if savefig != None:
        plt.savefig(savefig, bbox_inches="tight", dpi=300)

    plt.show()


def interpolate_saturated_pixels(image, threshold):
    """
    Interpolates the saturated pixels in a 2D image.

    Parameters:
    image (np.ndarray): 2D array representing the image.
    threshold (float): The saturation threshold. Pixels with values above this threshold will be interpolated.

    Returns:
    np.ndarray: The image with interpolated values for saturated pixels.
    """
    # Get the coordinates of the pixels
    x, y = np.indices(image.shape)

    # Mask for valid (non-saturated) pixels
    mask = image <= threshold

    # Coordinates of valid pixels
    valid_coords = np.array((x[mask], y[mask])).T

    # Values of valid pixels
    valid_values = image[mask]

    # Coordinates of saturated pixels
    saturated_coords = np.array((x[~mask], y[~mask])).T

    # Perform interpolation
    interpolated_values = griddata(
        valid_coords, valid_values, saturated_coords, method="cubic"
    )

    # Create a copy of the original image
    interpolated_image = np.copy(image)

    # Replace the values of the saturated pixels with the interpolated values
    interpolated_image[~mask] = interpolated_values

    return interpolated_image


# img = mpimg.imread('/Users/bencb/Downloads/psf_beam3_at_fpm_dm_flat.png')

img = mpimg.imread("data/lab_imgs/beam_3_f400_laser_top_level_nd3.png")

# qe = 0.2
wvl = 0.635e-6  # laser wavelength
D = 12e-3  # mm
# f = 254e-3 #mm
f = 400e-3  # mm
# N = f/D #
pixel_scale = 3.45e-6  # on point grey

# img_psf_region = img[800:900,50:150]
# img_bkg = img[:,200:]

xc, yc = np.unravel_index(np.argmax(img), img.shape)

ext = 35
img_psf_region = img[xc - ext : xc + ext, yc - ext : yc + ext]
img_bkg = img[:, 300:]  # just crop out the psf region

pointgrey_grid_x_um = np.linspace(
    -pixel_scale * img_psf_region.shape[0] / 2,
    pixel_scale * img_psf_region.shape[0] / 2,
    img_psf_region.shape[0],
)
pointgrey_grid_y_um = np.linspace(
    -pixel_scale * img_psf_region.shape[0] / 2,
    pixel_scale * img_psf_region.shape[1] / 2,
    img_psf_region.shape[1],
)

x, y = np.meshgrid(pointgrey_grid_x_um, pointgrey_grid_y_um)

# position to anglue ( pos = wvl * f / D ) at given wvl
theta_x = x / f  # np.linspace(-3*wvl/D,3*wvl/D,1000)
theta_y = y / f  # np.linspace(-3*wvl/D,3*wvl/D,1000)

theta_r = (theta_x**2 + theta_y**2) ** 0.5

airy_2D = (
    2
    * scipy.special.jv(1, 2 * np.pi / wvl * (D / 2) * np.sin(theta_r))
    / (2 * np.pi / wvl * (D / 2) * np.sin(theta_r))
) ** 2
airy_2D *= 1 / np.sum(airy_2D)  #

# in this case pixels are saturated so ignore them and do cubic interpolation here (not ideal)
reduced_psf_meas = interpolate_saturated_pixels(
    img_psf_region - np.mean(img_bkg), threshold=0.8
)

# reduced_psf_meas =  img_psf_region_interp

# im_list = [airy_2D, reduced_psf_meas / np.sum(reduced_psf_meas)]
# xlabel_list = ["", ""]
# ylabel_list = ["", ""]
# title_list = ["diffraction limit", "measured"]
# cbar_label_list = ["intensity", "intensity"]
# nice_heatmap_subplots(
#     im_list,
#     xlabel_list,
#     ylabel_list,
#     title_list,
#     cbar_label_list,
#     fontsize=15,
#     cbar_orientation="bottom",
#     axis_off=True,
# )


print(
    "Strehl = ", np.max(reduced_psf_meas / np.sum(reduced_psf_meas)) / np.max(airy_2D)
)
