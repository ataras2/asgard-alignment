
# Set of tools for registering a DM in the pupil image plane on a detector with a Zernike Wavefront Sensor
# initialiatally designed for the BMC Multi3.5 DM. 
# Diverged from courtney-barrer's BaldrApp.

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy.interpolate import RectBivariateSpline
import glob 
from astropy.io import fits 
import os 
import json
import datetime 
from scipy.interpolate import griddata
from scipy.ndimage import map_coordinates
from scipy.sparse import lil_matrix


def construct_bilinear_interpolation_matrix(image_shape, x_grid, y_grid, x_target, y_target):
    """
    Constructs a bilinear interpolation matrix that maps image pixels to target coordinates.
    (e.g. map from ZWFS image to DM actuators)
    Parameters:
        image_shape (tuple): Shape of the image (height, width).
        x_grid (ndarray): 1D array of x-coordinates corresponding to the image grid.
        y_grid (ndarray): 1D array of y-coordinates corresponding to the image grid.
        x_target (ndarray): 1D array of target x-coordinates where we want to interpolate.
        y_target (ndarray): 1D array of target y-coordinates where we want to interpolate.

    Returns:
        interpolation_matrix (2D array)
    """
    height, width = image_shape
    num_targets = len(x_target)

    # Create an empty sparse matrix (rows = target points, columns = original pixels)
    interpolation_matrix = lil_matrix((num_targets, height * width))

    for i in range(num_targets):
        # Find the surrounding grid indices
        x1_idx = np.searchsorted(x_grid, x_target[i]) - 1
        x2_idx = x1_idx + 1
        y1_idx = np.searchsorted(y_grid, y_target[i]) - 1
        y2_idx = y1_idx + 1

        # Ensure indices are within bounds
        x1_idx = max(0, min(x1_idx, width - 2))
        x2_idx = x1_idx + 1
        y1_idx = max(0, min(y1_idx, height - 2))
        y2_idx = y1_idx + 1

        # Get the actual x, y coordinates of the grid points
        x1, x2 = x_grid[x1_idx], x_grid[x2_idx]
        y1, y2 = y_grid[y1_idx], y_grid[y2_idx]

        # Compute bilinear interpolation weights
        w11 = ((x2 - x_target[i]) * (y2 - y_target[i])) / ((x2 - x1) * (y2 - y1))
        w21 = ((x_target[i] - x1) * (y2 - y_target[i])) / ((x2 - x1) * (y2 - y1))
        w12 = ((x2 - x_target[i]) * (y_target[i] - y1)) / ((x2 - x1) * (y2 - y1))
        w22 = ((x_target[i] - x1) * (y_target[i] - y1)) / ((x2 - x1) * (y2 - y1))

        # Flattened indices in the image
        idx11 = y1_idx * width + x1_idx
        idx21 = y1_idx * width + x2_idx
        idx12 = y2_idx * width + x1_idx
        idx22 = y2_idx * width + x2_idx

        # Fill the interpolation matrix
        interpolation_matrix[i, idx11] = w11
        interpolation_matrix[i, idx21] = w21
        interpolation_matrix[i, idx12] = w12
        interpolation_matrix[i, idx22] = w22

    return interpolation_matrix #.tocsr().toarray()  # Convert to compressed sparse row format for efficiency



# Function to get indices for the inner square on DM, accounting for missing corners
def get_inner_square_indices(outer_size, inner_offset, without_outer_corners=True):
    
    if without_outer_corners: # assumes outer corners of DM are missing 
        
        # Create a 12x12 grid with missing corners marked by NaN
        grid = np.arange(outer_size**2).reshape(outer_size, outer_size).astype(float)
        
        # Mark the missing corners
        grid[0, 0] = grid[0, -1] = grid[-1, 0] = grid[-1, -1] = np.nan

        # Define the inner square boundaries
        top_row = inner_offset
        bottom_row = outer_size - inner_offset - 1
        left_col = inner_offset
        right_col = outer_size - inner_offset - 1

        # Get the indices of the four corners of the inner square
        inner_corners = [
            grid[top_row, left_col],
            grid[top_row, right_col],
            grid[bottom_row, left_col],
            grid[bottom_row, right_col]
        ]

        # Remove the NaNs from the grid (we only want valid actuator indices)
        valid_indices = np.isfinite(grid.flatten())
        valid_actuators = np.where(valid_indices)[0]

        # Convert the inner square corner indices to valid actuator indices
        inner_square_indices = [np.where(valid_actuators == np.where(grid.flatten() == corner)[0][0])[0][0] for corner in inner_corners]
    else:  # outer corners not missing - this case is important with pinned edge actuators since the modal space then does not have missing corners! 
        grid = np.arange(outer_size**2).reshape(outer_size, outer_size)

        # Define the inner square boundaries
        top_row = inner_offset
        bottom_row = outer_size - inner_offset - 1
        left_col = inner_offset
        right_col = outer_size - inner_offset - 1

        # Get the indices of the four corners of the inner square
        inner_corners = [
            grid[top_row, left_col],
            grid[top_row, right_col],
            grid[bottom_row, left_col],
            grid[bottom_row, right_col]
        ]

        # Convert the inner square corner indices to valid actuator indices
        inner_square_indices = [np.where(grid.flatten() == corner)[0][0] for corner in inner_corners]

    return inner_square_indices



# Define a 2D Gaussian function
def gaussian_2d(xy, amp, x0, y0, sigma_x, sigma_y, theta, offset):
    x, y = xy
    x0, y0 = float(x0), float(y0)
    a = (np.cos(theta)**2)/(2*sigma_x**2) + (np.sin(theta)**2)/(2*sigma_y**2)
    b = -(np.sin(2*theta))/(4*sigma_x**2) + (np.sin(2*theta))/(4*sigma_y**2)
    c = (np.sin(theta)**2)/(2*sigma_x**2) + (np.cos(theta)**2)/(2*sigma_y**2)
    
    return offset + amp * np.exp(-(a*(x-x0)**2 + 2*b*(x-x0)*(y-y0) + c*(y-y0)**2))


# Function to get adjacent pixels within a given radius
def get_adjacent_pixels(coord, radius, pixel_values):
    """
    I, J: Coordinates of the central pixel
    radius: Radius (in pixels) for the region of interest
    pixel_values: 2D array of the image pixel values
    
    Returns:
    - Adjacent pixel indices and values within the radius
    """
    n,m = coord
    y, x = np.ogrid[-radius:radius+1, -radius:radius+1]
    mask = x**2 + y**2 <= radius**2
    indices = np.argwhere(mask)
    indices[:, 0] += n - radius
    indices[:, 1] += m - radius
    valid_indices = [(i, j) for i, j in indices if i >= 0 and j >= 0 and i < pixel_values.shape[0] and j < pixel_values.shape[1]]
    pixel_values_adj = np.array([pixel_values[i, j] for i, j in valid_indices])
    return np.array(valid_indices), pixel_values_adj


# Spline interpolation and Gaussian fitting
def interpolate_and_fit_gaussian(coord, radius, pixel_values, factor=5):
    """
    coord = tuple of coordinates of the central pixel (row, column)
    radius: Radius of the neighborhood
    pixel_values: 2D array of pixel values (image)
    factor: Upsampling factor for spline interpolation (default 5x finer grid)
    
    Returns:
    A dictionary with fitted Gaussian parameters and residuals.
    """
    i, j = coord
    # Get the adjacent pixels within the radius
    indices, pixel_values_adj = get_adjacent_pixels((i, j), radius, pixel_values)
    x = indices[:, 1]
    y = indices[:, 0]
    
    # Create a coarse grid
    x_grid = np.unique(x)
    y_grid = np.unique(y)
    X, Y = np.meshgrid(x_grid, y_grid)
    
    # Get pixel values on the coarse grid
    Z = pixel_values[np.ix_(y_grid, x_grid)]
    
    # Spline interpolate onto a finer grid
    spline = RectBivariateSpline(y_grid, x_grid, Z)
    
    # Create a finer grid
    x_fine = np.linspace(x_grid.min(), x_grid.max(), factor * len(x_grid))
    y_fine = np.linspace(y_grid.min(), y_grid.max(), factor * len(y_grid))
    X_fine, Y_fine = np.meshgrid(x_fine, y_fine)
    Z_fine = spline(y_fine, x_fine)
    
    # Initial guess for Gaussian fitting
    initial_guess = (Z_fine.max(), np.mean(x_fine), np.mean(y_fine), 1, 1, 0, np.median(Z_fine))
    
    # Perform Gaussian fitting
    try:
        popt, pcov = curve_fit(gaussian_2d, (X_fine.ravel(), Y_fine.ravel()), Z_fine.ravel(), p0=initial_guess)
    except RuntimeError:
        return {"error": "Fit did not converge"}
    
    # Calculate residuals
    Z_fit = gaussian_2d((X_fine, Y_fine), *popt).reshape(X_fine.shape)
    residuals = Z_fine - Z_fit
    
    # Extract fitted parameters
    amp_fit, x0_fit, y0_fit, sigma_x_fit, sigma_y_fit, theta_fit, offset_fit = popt
    
    # Convert the fine-grid coordinates of the peak to the global pixel frame
    x0_global = x0_fit
    y0_global = y0_fit
    
    # Create a dictionary to store the results
    fit_dict = {
        "amplitude": amp_fit,
        "x0": x0_global,
        "y0": y0_global,
        "sigma_x": sigma_x_fit,
        "sigma_y": sigma_y_fit,
        "theta": theta_fit,
        "offset": offset_fit,
        "X":X,
        "Y":Y,
        "I_raw":Z,
        "X_interp":X_fine,
        "Y_interp":Y_fine,
        "I_interp":Z_fine,
        "I_fit": Z_fit,
        "residuals": residuals,
         
    }
    
    return fit_dict



def plot_fit_results(fit_dict, savefig = None):
    """
    Plots the original, interpolated, fitted, and residual images from the fit_dict, with X and Y axes.
    
    Args:
        fit_dict: Dictionary returned by the interpolate_and_fit_gaussian function.
    """
    # Extract data from the fit_dict
    I = fit_dict["I_raw"]  # Original image
    I_interp = fit_dict["I_interp"]  # Interpolated image
    I_fit = fit_dict["I_fit"]  # Fitted Gaussian model
    residuals = fit_dict["residuals"]  # Residuals

    # Extract the grid coordinates
    X = fit_dict["X"]
    Y = fit_dict["Y"]
    X_interp = fit_dict["X_interp"]
    Y_interp = fit_dict["Y_interp"]

    # Create a figure with 4 subplots (in a single row)
    fig, axes = plt.subplots(1, 4, figsize=(16, 4))

    # Plot Original Image (I) with X, Y axis
    im1 = axes[0].imshow(I, origin='lower', extent=[X.min(), X.max(), Y.min(), Y.max()], cmap='viridis')
    axes[0].set_title('Original Image')
    #axes[0].set_xlabel('X')
    #axes[0].set_ylabel('Y')
    cbar1 = plt.colorbar(im1, ax=axes[0], orientation='horizontal', pad=0.1)
    cbar1.set_label('Intensity')

    # Plot Interpolated Image (I_interp) with X_interp, Y_interp axis
    im2 = axes[1].imshow(I_interp, origin='lower', extent=[X_interp.min(), X_interp.max(), Y_interp.min(), Y_interp.max()], cmap='viridis')
    axes[1].set_title('Interpolated Image')
    #axes[1].set_xlabel('X_interp')
    #axes[1].set_ylabel('Y_interp')
    cbar2 = plt.colorbar(im2, ax=axes[1], orientation='horizontal', pad=0.1)
    cbar2.set_label('Interpolated Intensity')

    # Plot Fitted Gaussian Model (I_fit) with X_interp, Y_interp axis
    im3 = axes[2].imshow(I_fit, origin='lower', extent=[X_interp.min(), X_interp.max(), Y_interp.min(), Y_interp.max()], cmap='viridis')
    axes[2].set_title('Fitted Gaussian')
    #axes[2].set_xlabel('X_interp')
    #axes[2].set_ylabel('Y_interp')
    cbar3 = plt.colorbar(im3, ax=axes[2], orientation='horizontal', pad=0.1)
    cbar3.set_label('Fitted Intensity')

    # Plot Residuals with X_interp, Y_interp axis
    im4 = axes[3].imshow(residuals, origin='lower', extent=[X_interp.min(), X_interp.max(), Y_interp.min(), Y_interp.max()], cmap='coolwarm')
    axes[3].set_title('Residuals')
    #axes[3].set_xlabel('X_interp')
    #axes[3].set_ylabel('Y_interp')
    cbar4 = plt.colorbar(im4, ax=axes[3], orientation='horizontal', pad=0.1)
    cbar4.set_label('Residual Intensity')

    # Display the plots
    plt.tight_layout()
    if savefig :
        plt.savefig(savefig) 
    plt.show()


# Function to calculate the intersection of two lines
def find_intersection(x1, y1, x2, y2, x3, y3, x4, y4):
    """
    Find the intersection point of two lines:
    Line 1: passing through (x1, y1) and (x2, y2)
    Line 2: passing through (x3, y3) and (x4, y4)
    """

    # Line 1 coefficients (a1*x + b1*y = c1)
    a1 = y2 - y1
    b1 = x1 - x2
    c1 = a1 * x1 + b1 * y1

    # Line 2 coefficients (a2*x + b2*y = c2)
    a2 = y4 - y3
    b2 = x3 - x4
    c2 = a2 * x3 + b2 * y3

    # Calculate the determinant of the system
    det = a1 * b2 - a2 * b1

    if det == 0:
        raise ValueError("The lines are parallel and do not intersect.")

    # Calculate the intersection point
    x_intersect = (b2 * c1 - b1 * c2) / det
    y_intersect = (a1 * c2 - a2 * c1) / det

    return x_intersect, y_intersect


# def sort_corners(corners):
#     # THIS FUNCTION FAILS WHEN CORNERS ARE HORIZONTALLY OR VERTICALLY ALIGNED 
#     sorted_corners = sorted(corners, key=lambda p: (p[0], p[1]))

#     # Separate them into logical quadrilateral corners
#     top_left = sorted_corners[0]
#     top_right = sorted_corners[1] if sorted_corners[1][1] > sorted_corners[0][1] else sorted_corners[2]
#     bottom_left = sorted_corners[2] if sorted_corners[1][1] > sorted_corners[0][1] else sorted_corners[1]
#     bottom_right = sorted_corners[3]
    
#     return(top_left, top_right,  bottom_left , bottom_right )

def sort_corners(corners):
    # Convert the corners to a numpy array
    corners = np.array(corners)

    # Calculate the centroid (geometric center) of the four points
    centroid = np.mean(corners, axis=0)

    # Sort the points based on their angle with respect to the centroid
    def angle_from_centroid(point):
        return np.arctan2(point[1] - centroid[1], point[0] - centroid[0])

    sorted_corners = sorted(corners, key=angle_from_centroid)

    # Assign corners: the sorting will order the points in a consistent counter-clockwise order
    # Now label them accordingly
    top_left = sorted_corners[3]
    top_right = sorted_corners[2]
    bottom_right = sorted_corners[1]
    bottom_left = sorted_corners[0]

    return top_left, top_right, bottom_left, bottom_right

# Main function to compute the intersection of diagonals of a quadrilateral
def find_quadrilateral_diagonal_intersection(corners ):
    """
    Given four corners of a quadrilateral, find the intersection point of its diagonals.
    
    Args:
    - corners: A list of 4 tuples, each representing the (x, y) coordinates of a corner.
    
    Output:
    - Intersection point (x, y) of the diagonals
    """
    # Sort the corners by x value first, then by y value for easier identification of top-left, top-right, etc.
    #sorted_corners = sorted(corners, key=lambda p: (p[0], p[1]))

    # Separate them into logical quadrilateral corners
    #top_left = sorted_corners[0]
    #top_right = sorted_corners[1] if sorted_corners[1][1] > sorted_corners[0][1] else sorted_corners[2]
    #bottom_left = sorted_corners[2] if sorted_corners[1][1] > sorted_corners[0][1] else sorted_corners[1]
    #bottom_right = sorted_corners[3]

    top_left, top_right,  bottom_left , bottom_right  = sort_corners(corners)
    # Diagonal 1: passing through top-left and bottom-right
    # Diagonal 2: passing through top-right and bottom-left
    intersection = find_intersection(top_left[0], top_left[1], bottom_right[0], bottom_right[1], 
                                     top_right[0], top_right[1], bottom_left[0], bottom_left[1])
    
    return intersection #, [top_left, top_right, bottom_right, bottom_left]


# Plotting function to visualize the quadrilateral and diagonals
def plot_quadrilateral_with_diagonals(corners, intersection):
    # Unpack the corners
    #sorted_corners = sorted(corners, key=lambda p: (p[0], p[1]))

    # Separate them into logical quadrilateral corners
    #top_left = sorted_corners[0]
    #top_right = sorted_corners[1] if sorted_corners[1][1] > sorted_corners[0][1] else sorted_corners[2]
    #bottom_left = sorted_corners[2] if sorted_corners[1][1] > sorted_corners[0][1] else sorted_corners[1]
    #bottom_right = sorted_corners[3]

    top_left, top_right,  bottom_left , bottom_right  = sort_corners(corners) 
    # Plot the quadrilateral
    plt.figure(figsize=(6, 6))
    
    # Plot the quadrilateral edges
    x_vals = [top_left[0], top_right[0], bottom_right[0], bottom_left[0], top_left[0]]
    y_vals = [top_left[1], top_right[1], bottom_right[1], bottom_left[1], top_left[1]]
    plt.plot(x_vals, y_vals, 'b-', label='Quadrilateral')

    # Plot the diagonals
    plt.plot([top_left[0], bottom_right[0]], [top_left[1], bottom_right[1]], 'r--', label='Diagonal 1')
    plt.plot([top_right[0], bottom_left[0]], [top_right[1], bottom_left[1]], 'g--', label='Diagonal 2')

    # Plot the intersection point
    plt.plot(intersection[0], intersection[1], 'ko', label='Intersection', markersize=10)

    # Label the points
    plt.text(top_left[0], top_left[1], 'Top Left', fontsize=12, verticalalignment='bottom')
    plt.text(top_right[0], top_right[1], 'Top Right', fontsize=12, verticalalignment='bottom')
    plt.text(bottom_left[0], bottom_left[1], 'Bottom Left', fontsize=12, verticalalignment='top')
    plt.text(bottom_right[0], bottom_right[1], 'Bottom Right', fontsize=12, verticalalignment='top')

    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title('Quadrilateral with Diagonals and Intersection')
    plt.grid(True)
    plt.legend()
    plt.show()



def get_DM_command_in_2D(cmd,Nx_act=12):
    # function so we can easily plot the DM shape (since DM grid is not perfectly square raw cmds can not be plotted in 2D immediately )
    #puts nan values in cmd positions that don't correspond to actuator on a square grid until cmd length is square number (12x12 for BMC multi-2.5 DM) so can be reshaped to 2D array to see what the command looks like on the DM.
    corner_indices = [0, Nx_act-1, Nx_act * (Nx_act-1), Nx_act*Nx_act]
    cmd_in_2D = list(cmd.copy())
    for i in corner_indices:
        cmd_in_2D.insert(i,np.nan)
    return( np.array(cmd_in_2D).reshape(Nx_act,Nx_act) )



def generate_dm_coordinates(Nx=12, Ny=12, spacing=1.0):
    """
    Generates the x, y coordinates of the actuators in a 12x12 grid DM with missing corners.
    
    Args:
        Nx, Ny: Number of actuators in the x and y directions (12x12 grid).
        spacing: The spacing between actuators (default is 1 unit).
    
    Returns:
        - coords: A list of tuples (x, y) representing the coordinates of the actuators.
        - flattened_indices: A dictionary that maps actuator indices (0 to 139) to (x, y) coordinates.
        - coord_to_index: A dictionary mapping (x, y) coordinates to actuator indices.
    """
    coords = []
    coord_to_index = {}
    flattened_indices = {}
    
    center_x = (Nx - 1) / 2  # Center of the grid in x
    center_y = (Ny - 1) / 2  # Center of the grid in y
    
    actuator_index = 0
    for i in range(Ny):
        for j in range(Nx):
            # Skip the missing corners
            if (i == 0 and j == 0) or (i == 0 and j == Nx - 1) or (i == Ny - 1 and j == 0) or (i == Ny - 1 and j == Nx - 1):
                continue

            # Calculate x and y coordinates relative to the center
            x = (j - center_x) * spacing
            y = (i - center_y) * spacing
            
            coords.append((x, y))
            coord_to_index[(x, y)] = actuator_index
            flattened_indices[actuator_index] = (x, y)
            actuator_index += 1

    return coords, flattened_indices, coord_to_index


def get_nearest_actuator(x, y, flattened_indices):
    """
    Finds the nearest actuator index for a given (x, y) coordinate.
    
    Args:
        x, y: The (x, y) coordinates to match to the nearest actuator.
        flattened_indices: A dictionary mapping actuator indices to (x, y) coordinates.
    
    Returns:
        Nearest actuator index.
    """
    distances = {index: np.sqrt((x - coord[0])**2 + (y - coord[1])**2) for index, coord in flattened_indices.items()}
    return min(distances, key=distances.get)


def actuator_to_xy(actuator_index, flattened_indices):
    """
    Given an actuator index, return the corresponding (x, y) coordinates.
    
    Args:
        actuator_index: The actuator number in the flattened 140-length array.
        flattened_indices: A dictionary mapping actuator indices to (x, y) coordinates.
    
    Returns:
        (x, y) coordinates of the actuator.
    """
    return flattened_indices.get(actuator_index)


def fit_affine_transformation_with_center(corners_dm, corners_img, intersection_img):
    """
    Fit an affine transformation from DM space to image space, using the DM center as the origin (0,0).
    
    Args:
        corners_dm: List of (x, y) coordinates of DM corners in DM space (relative to the DM center).
        corners_img: List of (x, y) coordinates of the corresponding points in image space.
        intersection_img: The (x, y) coordinates of the DM center in image space.
    
    Returns:
        - transform_matrix: A 2x3 matrix that transforms DM coordinates to pixel coordinates.
    """
    # Create arrays for the corners
    dm = np.array(corners_dm)
    img = np.array(corners_img)

    # Subtract the DM center (intersection) from the image coordinates to compute translation
    tx, ty = intersection_img
    
    # Now we need to solve for the linear transformation matrix (a, b, c, d)
    # We have the relationship: [x_img, y_img] = A * [x_dm, y_dm] + [tx, ty]
    # where A is the 2x2 matrix with components [a, b; c, d]
    
    # Create the matrix for DM space (without the translation part)
    dm_coords = np.vstack([dm.T, np.ones(len(dm))]).T
    
    # Subtract translation from image coordinates (image coordinates relative to DM center)
    img_coords = img - np.array([tx, ty])

    # Solve the linear system A * dm_coords = img_coords for A (a, b, c, d)
    # Solve the two systems independently for x and y
    A_x = np.linalg.lstsq(dm_coords[:, :2], img_coords[:, 0], rcond=None)[0]
    A_y = np.linalg.lstsq(dm_coords[:, :2], img_coords[:, 1], rcond=None)[0]
    
    # Construct the 2x3 affine transformation matrix
    transform_matrix = np.array([
        [A_x[0], A_x[1], tx],  # [a, b, tx]
        [A_y[0], A_y[1], ty]   # [c, d, ty]
    ])
    
    return transform_matrix

def pixel_to_dm(pixel_coord, transform_matrix):
    """
    Converts pixel coordinates to DM coordinates using the inverse of the affine transformation.
    
    Args:
        pixel_coord: A tuple (x, y) in pixel space.
        transform_matrix: The affine transformation matrix from DM space to pixel space.
    
    Returns:
        Tuple (x_dm, y_dm) in DM coordinates.
    """
    A = transform_matrix[:, :2]  # 2x2 matrix part
    t = transform_matrix[:, 2]   # translation part
    
    # Inverse transformation
    A_inv = np.linalg.inv(A)
    pixel_coord = np.array(pixel_coord)
    dm_coord = np.dot(A_inv, pixel_coord - t)
    return tuple(dm_coord)

def dm_to_pixel(dm_coord, transform_matrix):
    """
    Converts DM coordinates to pixel coordinates using the affine transformation.
    
    Args:
        dm_coord: A tuple (x, y) in DM space.
        transform_matrix: The affine transformation matrix from DM space to pixel space.
    
    Returns:
        Tuple (x_pixel, y_pixel) in pixel coordinates.
    """
    dm_coord = np.array(dm_coord)
    pixel_coord = np.dot(transform_matrix[:, :2], dm_coord) + transform_matrix[:, 2]
    return tuple(pixel_coord)



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


# def interpolate_pixel_intensities(image, pixel_coords):
#     """
#     Interpolates pixel intensities from an image onto the specified actuator pixel coordinates.
    
#     Args:
#         image: 2D array of pixel intensities (image).
#         pixel_coords: 2D array of actuator coordinates in pixel space (from transform_dict['actuator_coord_list_pixel_space']).
        
#     Returns:
#         Interpolated intensities at the given actuator pixel coordinates.
#     """
#     # Create a grid of original pixel coordinates
#     y, x = np.mgrid[0:image.shape[0], 0:image.shape[1]]
    
#     # Flatten the image and grid for interpolation
#     points = np.vstack((x.ravel(), y.ravel())).T  # Original pixel coordinates
#     values = image.ravel()  # Corresponding pixel values
    
#     # Interpolate the pixel values at the actuator pixel coordinates
#     interpolated_intensities = griddata(points, values, pixel_coords, method='cubic')
    
#     return interpolated_intensities


# below is faster
def interpolate_pixel_intensities(image, pixel_coords):
    """
    Fast interpolation using scipy's map_coordinates, which is highly optimized for 2D grids.

    Args:
        image: 2D array of pixel intensities (image).
        pixel_coords: 2D array of actuator coordinates in pixel space.
        
    Returns:
        Interpolated intensities at the given actuator pixel coordinates.
    """
    # pixel_coords needs to be in the format [y_coords, x_coords] for map_coordinates
    pixel_coords = np.array(pixel_coords).T  # Transpose to match map_coordinates format

    # Perform the interpolation using map_coordinates (linear interpolation by default)
    interpolated_intensities = map_coordinates(image.T, pixel_coords, order=1, mode='nearest')
    
    return interpolated_intensities


def calibrate_transform_between_DM_and_image( dm_4_corners, img_4_corners , debug = False, fig_path= None ):

    stacked_corner_img = []
    img_corners = []
    corner_fits = {}
    for actuator_number, delta_img in zip(dm_4_corners, img_4_corners):  # <<< added actuator number 
        stacked_corner_img.append( delta_img )

        peak_pixels_raw = tuple( np.array( list(np.where( abs(delta_img) == np.max( abs(delta_img) )  ) ) ).ravel() )
                
        # fit actuator position in pixel space after interpolation and Gaussian fit 
        corner_fit_dict = interpolate_and_fit_gaussian(coord=peak_pixels_raw, radius=5, pixel_values= abs(delta_img), factor=5)
        corner_fits[actuator_number] = corner_fit_dict
        #plot_fit_results( corner_fit_dict )
        
        img_corners.append( ( corner_fit_dict['x0'],  corner_fit_dict['y0'] ) )
        # #Check individual registration          
        # plt.figure(actuator_number)
        # plt.imshow( delta_img )
        # plt.plot( corner_fit_dict['x0'],  corner_fit_dict['y0'], 'x', color='red', lw=4, label='registered position') 
        # plt.legend()
        # plt.colorbar(label='Intensity')
        # plt.show() 
    
        
    #[top_left, top_right, bottom_right, bottom_left]
    intersection = find_quadrilateral_diagonal_intersection( img_corners ) 
    
    # fig = plt.figure()
    # im = plt.imshow( np.sum( stacked_corner_img , axis=0 ))
    # cbar = fig.colorbar(im, ax=plt.gca(), pad=0.01)
    # cbar.set_label(r'$\Delta$ Intensity', fontsize=15, labelpad=10)

    # for i,c in  enumerate( img_corners ):
    #     if i==0:
    #         plt.plot( c[0],  c[1], 'x', color='red', lw=4, label='registered position') 
    #     else:
    #         plt.plot( c[0],  c[1], 'x', color='red', lw=4 )
    
    # top_left, top_right, bottom_right, bottom_left = img_corners
    # plt.plot(  [top_left[0], bottom_right[0]], [top_left[1], bottom_right[1]] , 'r', lw=1)
    # plt.plot(  [bottom_left[0], top_right[0]],  [bottom_left[1], top_right[1]] , 'r', lw=1)
    # plt.plot( intersection[0], intersection[1], 'x' ,color='white', lw = 5 )
    # plt.legend()
    # savefig = fig_path + 'DM_center_in_pixel_space.png'
    # plt.savefig(savefig, dpi=300, bbox_inches = 'tight') 
    # plt.show()
    


    # Generate the DM coordinates for a 12x12 grid with missing corners with dictionaries
    # to convert between actuator number and coordinate frame 
    coords, dm_actuator_to_coord, dm_coord_to_actuator = generate_dm_coordinates()
        
    # get the coordinates of the DM corners used to probe the image space 
    dm_corners = np.array( [dm_actuator_to_coord[ c ] for c in dm_4_corners] )
    
    # ====== Fit the affine transformation to convert DM coordinates to Pixel space and visa versa
    # Important that indexing of corners in each space correspond!!!
    # ---!!!! --- THIS IS CRITICAL ---!!!----
    # ensure this by filling img_corners in the same order as dm_corners (as above).
    transform_matrix = fit_affine_transformation_with_center(dm_corners, img_corners, intersection)

    #FINALLY - projecting each actuator onto pixel space
    # get coordinates of each actuator 
    act_coord_list = np.array( [dm_actuator_to_coord[a] for a in range(140)] )
    ## check
    #plt.figure()
    #plt.scatter(act_coord_list[:, 0], act_coord_list[:, 1], color='blue', marker='o')
    #plt.show() 
    
    # put them in pixel space 
    pixel_coord_list = np.array( [dm_to_pixel(c, transform_matrix) for c in act_coord_list] )
    ## check 
    # plt.figure()
    # plt.scatter(pixel_coord_list[:, 0], pixel_coord_list[:, 1], color='blue', marker='o')
    # plt.show() 
    
    if debug:
        # Show the probing DM geometry 
        fig = plt.figure(1) 
        tmp =np.zeros(140)
        tmp[np.array(dm_4_corners)] = 1
        plt.imshow( get_DM_command_in_2D( tmp ) )
        if fig_path is not None:
            savefig = fig_path + 'DM_corner_poke_in_DM_space.png'
            fig.savefig(savefig, dpi=300, bbox_inches = 'tight' )


        # Show the actuator registration in Pixel space 
        fig = plt.figure(2)
        im = plt.imshow( np.sum( stacked_corner_img , axis=0 ))
        cbar = fig.colorbar(im, ax=plt.gca(), pad=0.01)
        cbar.set_label(r'$\Delta$ Intensity', fontsize=15, labelpad=10)
        
        plt.scatter(pixel_coord_list[:, 0], pixel_coord_list[:, 1], color='blue', marker='.', label = 'DM actuators')
        
        for i,c in  enumerate( img_corners ):
            if i==0:
                plt.plot( c[0],  c[1], 'x', color='red', lw=4, label='registered corners') 
            else:
                plt.plot( c[0],  c[1], 'x', color='red', lw=4 )
        
        top_left, top_right,  bottom_left , bottom_right = sort_corners( img_corners ) 
        plt.plot(  [top_left[0],bottom_right[0]], [top_left[1],bottom_right[1]] , 'r', lw=1)
        plt.plot(  [bottom_left[0],top_right[0]],  [bottom_left[1],top_right[1]] , 'r', lw=1)
        plt.plot( intersection[0], intersection[1], 'x' ,color='white', lw = 5 )
        plt.legend()
        if fig_path is not None:
            savefig = fig_path + 'DM_registration_in_pixel_space.png'
            plt.savefig(savefig, dpi=300, bbox_inches = 'tight' )
        plt.show()

        

    ## write to file 
    write_dict = {
     "dm_corners" : dm_corners, 
     "img_corners" : img_corners, 
     "corner_fit_results" : corner_fits,
     "DM_center_pixel_space" : intersection,
     "actuator_to_pixel_matrix" : transform_matrix,
     "dm_actuator_to_coord" : dm_actuator_to_coord ,
     "actuator_coord_list_dm_space" : act_coord_list,
     "actuator_coord_list_pixel_space" : pixel_coord_list
    }
    
    return write_dict


    


if __name__== "__main__":
    
    # anyone else running this will need to change paths to their own. 
    
    example = 1 
    
    if example == 1:
        """
        doing DM registration from the interaction matrix 
        - example of how to deal with the case of different zonal basis (pinned edge actuators vs no pinning)
        """  
        
        # Define the parent directory where the search should start
        parent_dir = '/Users/bencb/Documents/baldr/data_sydney/A_FINAL_SYD_DATA_18-09-2024/tmp/'

        # Define the pattern to match
        pattern = os.path.join(parent_dir, '**/zonal_reconstructor/RECONSTRUCTORS_zonal*.fits')

        # Recursively search for files that match the pattern
        file_paths = glob.glob(pattern, recursive=True)

        dates = np.array( [datetime.datetime.strptime(f.split('.fits')[0].split('_')[-1],'%d-%m-%YT%H.%M.%S') for f in file_paths] )

        dm_center_in_detector = []
        for f in file_paths:
            
            with fits.open(f) as d:

                # ===================================
                # Look at DM center in pixel space 
                    
                # get inner corners for estiamting DM center in pixel space (have to deal seperately with pinned actuator basis)
                if d['I2M'].data.shape[1] == 100: # outer actuators are pinned, 
                    corner_indicies = get_inner_square_indices(outer_size=10, inner_offset=3, without_outer_corners=False)
                    
                elif d['I2M'].data.shape[1] == 140: # outer acrtuators are free 
                    print(140)
                    corner_indicies = get_inner_square_indices(outer_size=12, inner_offset=4, without_outer_corners=True)
                else:
                    print("CASE NOT MATCHED  d['I2M'].data.shape = { d['I2M'].data.shape}")
                    
                img_4_corners = []
                dm_4_corners = []
                for i in corner_indicies:
                    dm_4_corners.append( np.where( d['M2C'].data[:,i] )[0][0] )
                    #dm2px.get_DM_command_in_2D( d['M2C'].data[:,i]  # if you want to plot it 

                    tmp = np.zeros( d['I0'].data.shape )
                    tmp.reshape(-1)[d['pupil_pixels'].data] = d['IM'].data[i] 

                    #plt.imshow( tmp ); plt.show()
                    img_4_corners.append( abs(tmp ) )

                #plt.imshow( np.sum( tosee, axis=0 ) ); plt.show()

                # dm_4_corners should be an array of length 4 corresponding to the actuator index in the (flattened) DM command space
                # img_4_corners should be an array of length 4xNxM where NxM are the image dimensions.
                # !!! It is very important that img_4_corners are registered in the same order as dm_4_corners !!!
                transform_dict = calibrate_transform_between_DM_and_image( dm_4_corners, img_4_corners , debug=False, fig_path = None)

                dm_center_in_detector.append( transform_dict['DM_center_pixel_space'] )
                
    
    
    elif example == 2: 
        """
        #Example using poke ramp data to calibrate affine transform between DM and pixel space
        # change paths to whatever you want       
        """
        fig_path = '/Users/bencb/Documents/baldr/data_sydney/analysis_scripts/analysis_results/DM_registration_results/'
        files = glob.glob('/Users/bencb/Documents/baldr/data_sydney/A_FINAL_SYD_DATA_18-09-2024/tmp/09-09-2024/poke_ramp_data/pokeramp*.fits')

        if not os.path.exists( fig_path ):
            os.makedirs( fig_path )
            
        f = files[0]
        d = fits.open(f)
        No_ramps = int(d['SEQUENCE_IMGS'].header['#ramp steps'])
        max_ramp = float(d['SEQUENCE_IMGS'].header['in-poke max amp'])
        min_ramp = float(d['SEQUENCE_IMGS'].header['out-poke max amp'])
        ramp_values = np.linspace(min_ramp, max_ramp, No_ramps)

        #Nmodes_poked = int(d[0].header['HIERARCH Nmodes_poked'])
        Nact = int(d[0].header['HIERARCH Nact'])
        N0 = d['FPM_OUT'].data
        I0 = d['FPM_IN'].data

        poke_imgs = d['SEQUENCE_IMGS'].data[1:].reshape(No_ramps, Nact, I0.shape[0], I0.shape[1])

        dm_4_corners = get_inner_square_indices(outer_size=12, inner_offset=3) # flattened index of the DM actuator 
        ## look at what actuators we are considering
        # tmp =np.zeros(140)
        # tmp[np.array(dm_4_corners)] = 1
        # plt.figure()
        # plt.imshow( get_DM_command_in_2D( tmp ) )
        # plt.show() 
        
        img_4_corners = []
        for actuator_number in dm_4_corners:
            # for each actuator poke we get the corresponding (differenced) image
            # and append it to img_4_corners
            
            a = 4 # amplitude index 
            delta_img = poke_imgs[len(ramp_values)//2 + a][actuator_number] - I0 #poke_imgs[len(ramp_values)//2 - a][actuator_number]
            img_4_corners.append( delta_img  ) 
            
        # from the 4 images corresponding to the 4 (corner) actuator pokes we calibrate our transform
        
        # dm_4_corners should be an array of length 4 corresponding to the actuator index in the (flattened) DM command space
        # img_4_corners should be an array of length 4xNxM where NxM are the image dimensions.
        # !!! It is very important that img_4_corners are registered in the same order as dm_4_corners !!!
        transform_dict = calibrate_transform_between_DM_and_image( dm_4_corners, img_4_corners , debug=True, fig_path = fig_path)

        # example to overlay the registered actuators in pixel space with the pupil (FPM out of beam)
        fig = plt.figure(3)
        
        im = plt.imshow( N0 )
        cbar = fig.colorbar(im, ax=plt.gca(), pad=0.01)
        cbar.set_label(r'Intensity', fontsize=15, labelpad=10)
        
        plt.scatter(transform_dict['actuator_coord_list_pixel_space'][:, 0],\
            transform_dict['actuator_coord_list_pixel_space'][:, 1], \
                color='blue', marker='.', label = 'DM actuators')
        
        plt.legend() 
        savefig = fig_path + 'pupil_on_DM_in_pixel_space.png'
        fig.savefig(savefig, dpi=300, bbox_inches = 'tight' )
        plt.show()    

        # write the transform dict to a json file 
        tstamp = datetime.datetime.now().strftime("%d-%m-%YT%H.%M.%S")
        serializable_dict = convert_to_serializable(transform_dict)

        with open(fig_path + f'DM2img_{tstamp}.json', 'w') as f:
            json.dump(serializable_dict, f)
            
            
        # example to interpolate the measured intensities onto registered actuators in pixel space 
        interpolated_intensities_I0 = interpolate_pixel_intensities(image = I0, pixel_coords = transform_dict['actuator_coord_list_pixel_space'])
        interpolated_intensities_N0 = interpolate_pixel_intensities(image = N0, pixel_coords = transform_dict['actuator_coord_list_pixel_space'])

        plt.figure()
        plt.title('interpolating I0 onto \nregistered DM actuators')
        plt.imshow( get_DM_command_in_2D( interpolated_intensities_N0  ) )
        plt.show() 
        
        # example to register "well sensed" actuators 
        plt.figure()
        plt.title('interpolating I0 onto \nregistered DM actuators')
        plt.imshow( get_DM_command_in_2D( interpolated_intensities_N0  ) > 40 )
        plt.show() 

