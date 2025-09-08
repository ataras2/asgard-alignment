import numpy as np
import time
import datetime
from scipy.ndimage import center_of_mass
import matplotlib.pyplot as plt
import scipy.interpolate as interp
from matplotlib import animation
from scipy.cluster.vq import kmeans, vq
from scipy.ndimage import gaussian_filter, median_filter
from scipy.optimize import leastsq

# "!fpm_movetomask {} {}": fpm_move_to_phasemask_msg,
# "!fpm_moverel {} {}": fpm_move_relative_msg,
# "!fpm_moveabs {} {}": fpm_move_absolute_msg,
# "!fpm_readpos {}": fpm_read_position_msg,
# "!fpm_updatemaskpos {} {}": fpm_update_mask_position_msg,
# "!fpm_updatemaskpos {}": fpm_write_mask_positions_msg,
# "!fpm_updateallmaskpos {} {} {}": fpm_update_all_mask_positions_relative_to_current_msg,



def complete_collinear_points(known_points, separation, tolerance=20):
    """
    Completes the dictionary of collinear points given known positions and separation,
    used to find phase mask positions when only some of the positions are known. 

    Parameters:
    known_points (dict): A dictionary where keys are integers (1-5) representing the order of points,
                         and values are tuples (x, y) representing the known coordinates.
    separation (float): The separation between consecutive points.
    tolerance (float): Allowed deviation for the separation constraint.

    Returns:
    dict: A dictionary with all points (1-5) and their computed positions, ordered by keys.
    """
    # Validate input
    if not known_points:
        raise ValueError("known_points dictionary cannot be empty.")
    if separation <= 0:
        raise ValueError("Separation must be a positive value.")
    if any(key < 1 or key > 5 for key in known_points):
        raise ValueError("Keys in known_points must be integers between 1 and 5.")
    
    # Extract known keys and positions
    keys = np.array(sorted(known_points.keys()))
    positions = np.array([known_points[k] for k in keys])  # Shape: (n_points, 2)

    # Validate separation constraints for known points
    for i in range(len(keys) - 1):
        actual_separation = np.sqrt(np.sum((positions[i + 1] - positions[i])**2))
        expected_separation = (keys[i + 1] - keys[i]) * separation
        if not (np.abs(actual_separation - expected_separation) <= tolerance):
            raise ValueError(
                f"Separation constraint violated between points {keys[i]} and {keys[i+1]}: "
                f"actual={actual_separation}, expected={expected_separation}"
            )

    # Fit a line through the known points
    t = keys - keys.min()  # Normalize to start from 0
    x, y = positions[:, 0], positions[:, 1]
    px = np.polyfit(t, x, 1)  # Fit x(t)
    py = np.polyfit(t, y, 1)  # Fit y(t)

    # Calculate the unit direction vector of the line
    dx, dy = px[0], py[0]  # Gradients dx/dt and dy/dt
    direction = np.array([dx, dy])
    direction /= np.linalg.norm(direction)  # Normalize

    # Reconstruct all positions along the line
    result = {}
    min_key = keys.min()

    for i in range(1, 6):
        if i in known_points:
            result[i] = known_points[i]
        else:
            offset = (i - min_key) * separation  # Offset distance along the line
            reference_point = known_points[min_key]
            position = np.array(reference_point) + offset * direction
            result[i] = tuple(position)

    # Print the calculated separation for verification
    positions_array = np.array([result[i] for i in sorted(result.keys())])
    separations = np.sqrt(np.sum(np.diff(positions_array, axis=0)**2, axis=1))
    print("Calculated separations between consecutive points:", separations)

    # Order the result by keys
    ordered_result = {key: result[key] for key in sorted(result.keys())}
    return ordered_result



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
    #plt.show()
    return fig, ax 
    




def compute_image_difference(img1, img2):
    # normalize both images first
    img1 = img1.copy() / np.sum(img1)
    img2 = img2.copy() / np.sum(img2)
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
    dx = (np.sum(np.abs(q3 - q1)) - np.sum(np.abs(q2 - q4))) / (
        np.sum(np.abs(q3 + q1)) + np.sum(np.abs(q2 + q4))
    )
    dy = (np.sum(np.abs(q2 - q1)) - np.sum(np.abs(q3 - q4))) / (
        np.sum(np.abs(q2 + q1)) + np.sum(np.abs(q3 + q4))
    )

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


def square_spiral_scan(starting_point, step_size, search_radius):
    """
    Generates a square spiral scan pattern starting from the initial point within a given search radius and step size.

    Parameters:
    starting_point (tuple): The initial (x, y) point to start the spiral.
    step_size (float): The size of each step in the grid.
    search_radius (float): The maximum radius to scan in both x and y directions.

    Returns:
    list: A list of tuples where each tuple contains (x_amp, y_amp), the left/right and up/down amplitudes for the scan.
    """
    x, y = starting_point  # Start at the given initial point
    dx, dy = step_size, 0  # Initial movement to the right
    scan_points = [(x, y)]
    steps_taken = 0  # Counter for steps taken in the current direction
    step_limit = 1  # Initial number of steps in each direction

    while max(abs(x - starting_point[0]), abs(y - starting_point[1])) <= search_radius:
        for _ in range(
            2
        ):  # Repeat twice: once for horizontal, once for vertical movement
            for _ in range(step_limit):
                x, y = x + dx, y + dy
                if (
                    max(abs(x - starting_point[0]), abs(y - starting_point[1]))
                    > search_radius
                ):
                    return scan_points
                scan_points.append((x, y))

            # Rotate direction (right -> up -> left -> down)
            dx, dy = -dy, dx

        # Increase step limit after a complete cycle (right, up, left, down)
        step_limit += 1

    return scan_points


def raster_scan_with_orientation(starting_point, dx, dy, width, height, orientation=0):
    """
    Generates a raster scan pattern within a defined rectangular area and rotates it by a given orientation.

    Parameters:
    starting_point (tuple): The initial (x, y) point to start the raster scan.
    dx (float): Step size in the x-direction.
    dy (float): Step size in the y-direction.
    width (float): Total width of the scan area.
    height (float): Total height of the scan area.
    orientation (float): Orientation angle in degrees (rotation counterclockwise).

    Returns:
    list: A list of tuples where each tuple contains (x, y) positions for the scan.
    """
    x_start, y_start = starting_point
    scan_points = []

    # Define the bounds of the scan area
    x_min = 0
    x_max = width
    y_min = 0
    y_max = height

    # Initialize y and direction for x movement
    y = y_min
    direction = 1  # 1 for left-to-right, -1 for right-to-left

    while y <= y_max:
        # Generate a row of points
        row_points = []
        if direction == 1:  # Left-to-right
            x = x_min
            while x <= x_max:
                row_points.append((x, y))
                x += dx
        else:  # Right-to-left
            x = x_max
            while x >= x_min:
                row_points.append((x, y))
                x -= dx

        # Add the row to the scan points
        scan_points.extend(row_points)

        # Move to the next row and flip direction
        y += dy
        direction *= -1

    # Rotate points based on the orientation angle
    angle_rad = np.radians(orientation)
    cos_theta, sin_theta = np.cos(angle_rad), np.sin(angle_rad)

    # Apply rotation and translate back to the starting point
    rotated_points = []
    for x, y in scan_points:
        x_rot = cos_theta * x - sin_theta * y
        y_rot = sin_theta * x + cos_theta * y
        rotated_points.append((x_rot + x_start, y_rot + y_start))

    return rotated_points


def cross_scan(starting_point, dx, dy, width, height, angle):
    """
    Generates a cross scan pattern with a given angle of rotation. This function
    generates two lines crossing at the origin and rotates them based on the given angle.

    Parameters:
    starting_point (tuple): The center of the cross scan (origin of cross).
    dx (float): Step size in the x-direction (spacing between points).
    dy (float): Step size in the y-direction (spacing between points).
    X_amp (float): Amplitude of the cross in the x-direction (half-length).
    height (float): Amplitude of the cross in the y-direction (half-length).
    angle (float): Rotation angle in degrees (counterclockwise).

    Returns:
    list: A list of tuples where each tuple contains (x, y) positions for the scan.
    """
    # Define the lines along x and y axes before rotation (horizontal and vertical)
    line_1 = [( i, 0) for i in np.arange(-width/2, width/2+dx, dx)]  # Horizontal line (X-axis)
    line_2 = [(0,  i) for i in np.arange(-height/2, height/2+dy, dy)]  # Vertical line (Y-axis)

    # Rotate the lines based on the angle
    angle_rad = np.radians(angle - 90)  # Adjust the angle so 0 degrees is aligned with X,Y axes
    cos_theta, sin_theta = np.cos(angle_rad), np.sin(angle_rad)

    # Apply rotation to both lines
    rotated_line_1 = [(cos_theta * x - sin_theta * y, sin_theta * x + cos_theta * y) for x, y in line_1]
    rotated_line_2 = [(cos_theta * x - sin_theta * y, sin_theta * x + cos_theta * y) for x, y in line_2]

    # Shift back to the starting point
    rotated_line_1 = [(x + starting_point[0], y + starting_point[1]) for x, y in rotated_line_1]
    rotated_line_2 = [(x + starting_point[0], y + starting_point[1]) for x, y in rotated_line_2]

    # Combine both lines into one list of tuples (x, y)
    cross_points = rotated_line_1 + rotated_line_2

    return cross_points


    
    
    

def raster_square_search_and_save_images(
    cam,
    beam,
    phasemask,
    starting_point,
    dx, 
    dy, 
    width, 
    height, 
    orientation=0,
    sleep_time=1,
    use_multideviceserver=True,
    plot_grid_before_scan=True
):
    """
    Perform a raster search pattern to map the phase mask.
    if use_multideviceserver is True, the function will use ZMQ protocol to communicate with the
    MultiDeviceServer to move the phase mask. 
    !!! In this case phasemask should be the socket for the ZMQ protocol.
    !!! Otherwise, it will move the phase mask directly and phasemask shold be the BaldrPhaseMask object.
    """

    spiral_pattern = raster_scan_with_orientation(starting_point, dx, dy, width, height, orientation)

    x_points, y_points = zip(*spiral_pattern)
    img_dict = {}

    if plot_grid_before_scan:
        # Plot the scan points
        plt.figure(figsize=(6, 6))
        plt.scatter(x_points, y_points, color="blue", label="Scan Points")
        plt.plot(x_points, y_points, linestyle="--", color="gray", alpha=0.7)
        plt.title(f"Raster Scan Pattern with {orientation}° Rotation")
        plt.xlabel("X")
        plt.ylabel("Y")
        plt.legend()
        plt.grid()
        plt.axis("equal")  # Ensure equal scaling
        plt.savefig( 'delme.png')
        plt.show()
        plt.close()


    for i, (x_pos, y_pos) in enumerate(zip(x_points, y_points)):
        print("at ", x_pos, y_pos)
        print(f"{100 * i/len(x_points)}% complete")

        # motor limit safety checks!
        if x_pos <= 0:
            print('x_pos < 0. set x_pos = 1')
            x_pos = 1
        if x_pos >= 10000:
            print('x_pos > 10000. set x_pos = 9999')
            x_pos = 9999
        if y_pos <= 0:
            print('y_pos < 0. set y_pos = 1')
            y_pos = 1
        if y_pos >= 10000:
            print('y_pos > 10000. set y_pos = 9999')
            y_pos = 9999

        if use_multideviceserver:
            #message = f"fpm_moveabs phasemask{beam} {[x_pos, y_pos]}"
            message = f"moveabs BMX{beam} {x_pos}"
            phasemask.send_string(message)
            response = phasemask.recv_string()
            print(response)

            message = f"moveabs BMY{beam} {y_pos}"
            phasemask.send_string(message)
            response = phasemask.recv_string()
            print(response)
        else:
            phasemask.move_absolute([x_pos, y_pos])

        time.sleep(sleep_time)  # wait for the phase mask to move and settle
        img = np.mean(
            cam.get_data(), #get_some_frames( number_of_frames=10, apply_manual_reduction=True),
            axis=0,
        )

        img_dict[(x_pos, y_pos)] = img

    return img_dict


def spiral_square_search_and_save_images(
    cam,
    beam,
    phasemask,
    starting_point,
    step_size,
    search_radius,
    sleep_time=2,
    use_multideviceserver=True,
):
    """
    Perform a spiral square search pattern to find the best position for the phase mask.
    if use_multideviceserver is True, the function will use ZMQ protocol to communicate with the
    MultiDeviceServer to move the phase mask. In this case phasemask should be the socket for the ZMQ protocol.
    Otherwise, it will move the phase mask directly and phasemask shold be the BaldrPhaseMask object.
    """

    spiral_pattern = square_spiral_scan(starting_point, step_size, search_radius)

    x_points, y_points = zip(*spiral_pattern)
    img_dict = {}

    for i, (x_pos, y_pos) in enumerate(zip(x_points, y_points)):
        print("at ", x_pos, y_pos)
        print(f"{100 * i/len(x_points)}% complete")

        # motor limit safety checks!
        if x_pos <= 0:
            print('x_pos < 0. set x_pos = 1')
            x_pos = 1
        if x_pos >= 10000:
            print('x_pos > 10000. set x_pos = 9999')
            x_pos = 9999
        if y_pos <= 0:
            print('y_pos < 0. set y_pos = 1')
            y_pos = 1
        if y_pos >= 10000:
            print('y_pos > 10000. set y_pos = 9999')
            y_pos = 9999

        if use_multideviceserver:
            #message = f"fpm_moveabs phasemask{beam} {[x_pos, y_pos]}"
            message = f"moveabs BMX{beam} {x_pos}"
            phasemask.send_string(message)
            response = phasemask.recv_string()
            print(response)

            message = f"moveabs BMY{beam} {y_pos}"
            phasemask.send_string(message)
            response = phasemask.recv_string()
            print(response)
        else:
            phasemask.move_absolute([x_pos, y_pos])

        time.sleep(sleep_time)  # wait for the phase mask to move and settle
        
        #number_of_frames=10, apply_manual_reduction=True),
        img = np.mean(
            cam.get_data()[-10:], 
            axis=0,
        )

        img_dict[(x_pos, y_pos)] = img

    return img_dict


def analyse_search_results(search, savepath="delme.png", plot_logscale=True):
    """
    analyse results from spiral_square_search_and_save_images
    search = img_dict output from spiral_square_search_and_save_images function.
    """
    coord = np.array([k for k, v in search.items()])
    img_list = np.array([v for k, v in search.items()])

    # make the ref image the median 
    reftmp = np.median( img_list ,axis = 0)

    dif_imgs = np.mean((img_list[0] - img_list) ** 2, axis=(1, 2))

    # dont do log in the case of dark subtraction (negative adu!)
    #dif_imgs = np.mean((np.log10( reftmp ) - np.log10( img_list) ) ** 2, axis=(1, 2))

    plt.figure()
    plt.plot(dif_imgs)
    plt.savefig(savepath)
    # order indicies from the best (highest) according to metric dif_imgs
    candidate_indx = sorted(
        range(len(dif_imgs)), key=lambda i: dif_imgs[i], reverse=True
    )

    i = 0
    metric_candidate = dif_imgs[candidate_indx[i]]
    img_candidate = search[tuple(coord[candidate_indx[i]])]

    prompt = 1
    i = 0
    while prompt != "e":

        metric_candidate = dif_imgs[candidate_indx[i]]
        img_candidate = search[tuple(coord[candidate_indx[i]])]

        plt.figure()
        if plot_logscale:
            im = plt.imshow(np.log10(img_candidate))
            plt.colorbar(im, label='log ADU')
        else: 
            im = plt.imshow(img_candidate)
            plt.colorbar(im, label = "ADU")
        plt.title(
            f"position = { coord[candidate_indx[i]]}, metric = {metric_candidate}"
        )
        plt.savefig(savepath)
        prompt = input("1 to go to next, 0 to go back, e to exit")

        if prompt == "1":
            i += 1
        elif prompt == "0":
            i -= 1
        elif prompt == "e":
            print(
                f"stoped at index {candidate_indx[i]}, coordinates {coord[candidate_indx[i]]}"
            )
            stop_coord = coord[candidate_indx[i]]
        else:
            print("invalid input. 1 to go to next, 0 to go back, e to exit")

    # phasemask.move_absolute( stop_coord )

    # phasemask_centering_tool.move_relative_and_get_image(cam, phasemask, savefigName=fig_path + 'delme.png')

    # # TO update positions and write file if needed
    # phasemask.update_mask_position( phasemask_name )
    # phasemask.update_all_mask_positions_relative_to_current( phasemask_name, 'phase_positions_beam_3 original_DONT_DELETE.json')
    # phasemask.write_current_mask_positions()

    return stop_coord


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
    positions = [eval(key) for key in data_dict.keys()]
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
    image_ax.set_title("Image at Current Position")

    # Function to update the plots for each frame
    def update_frame(frame_idx):
        # Update scatter plot
        scatter.set_offsets(np.c_[x_positions[:frame_idx + 1], y_positions[:frame_idx + 1]])

        # Update image plot
        img_display.set_data(images[frame_idx])

        return scatter, img_display

    # Create the animation
    ani = animation.FuncAnimation(fig, update_frame, frames=num_frames, blit=False, repeat=False)

    # Save the animation as a movie file
    ani.save(save_path, fps=fps, writer='ffmpeg')

    plt.close(fig)  # Close the figure to avoid displaying it unnecessarily


def spiral_search_and_center(
    cam,
    phasemask,
    phasemask_name,
    beam,
    starting_point,
    search_radius,
    dr,
    dtheta,
    reference_img,
    fine_tune_threshold=3,
    savefigName=None,
    usr_input=True,
    use_multideviceserver=True
):

    if use_multideviceserver:
        #message = f"fpm_moveabs phasemask{beam} {[x_pos, y_pos]}"
        message = f"moveabs BMX{beam} {x_pos}"
        phasemask.send_string(message)
        response = phasemask.recv_string()
        print(response)

        message = f"moveabs BMY{beam} {y_pos}"
        phasemask.send_string(message)
        response = phasemask.recv_string()
        print(response)
    else:
        phasemask.move_absolute([x_pos, y_pos])

    x, y = starting_point
    angle = 0
    radius = 0
    plot_cnt = 0  # so we don't plot every iteration

    diff_list = []  # to track our metrics
    x_pos_list = []
    y_pos_list = []
    sleep_time = 0.7  # s
    while radius < search_radius:
        x_pos = x + radius * np.cos(angle)
        y_pos = y + radius * np.sin(angle)

        phasemask.move_absolute([x_pos, y_pos])
        time.sleep(sleep_time)  # wait for the phase mask to move and settle
        img = np.mean(
            cam.get_some_frames(number_of_frames=10, apply_manual_reduction=True),
            axis=0,
        )

        initial_img = img.copy()  # take a copy of original image

        diff = compute_image_difference(img, reference_img)
        diff_list.append(diff)
        x_pos_list.append(x_pos)
        y_pos_list.append(y_pos)
        print(f"img diff = {diff}, fine_tune_threshold={fine_tune_threshold}")

        # Update for next spiral step
        angle += dtheta
        radius += dr

        # print( radius )
        # _ = input('next')
        if savefigName != None:
            if np.mod(plot_cnt, 5) == 0:

                norm = plt.Normalize(0, fine_tune_threshold)

                fig, ax = plt.subplots(1, 3, figsize=(20, 7))
                ax[0].set_title(f"image\nphasemask={phasemask_name}")
                ax[1].set_title(
                    f'search positions\nx:{phasemask.motors["x"]}\ny:{phasemask.motors["y"]}'
                )
                ax[2].set_title("search metric")

                ax[0].imshow(img)
                ax[1].plot([x_pos, y_pos], "x", color="r", label="current pos")
                ax[1].plot(
                    [starting_point[0], starting_point[1]],
                    "o",
                    color="k",
                    label="current pos",
                )
                tmp_diff_list = np.array(diff_list)
                tmp_diff_list[tmp_diff_list < 1e-5] = (
                    0.1  # very small values got to finite value (errors whern 0!)
                )
                # s= np.exp( 400 * np.array(tmp_diff_list) / fine_tune_threshold )
                ax[1].scatter(
                    x_pos_list,
                    y_pos_list,
                    s=10,
                    marker="o",
                    c=diff_list,
                    cmap="viridis",
                    norm=norm,
                )
                ax[1].set_xlim(
                    [starting_point[0] - search_radius, starting_point[0] + search_radius]
                )
                ax[1].set_ylim(
                    [starting_point[1] - search_radius, starting_point[1] + search_radius]
                )
                ax[1].legend()
                ax[2].plot(diff_list)
                ax[2].set_xlim([0, search_radius / dr])

                ax[0].axis("off")
                ax[1].set_ylabel("y pos (um)")
                ax[1].set_xlabel("x pos (um)")
                ax[2].set_ylabel(r"$\Sigma|img - img_{off}|$")
                ax[2].set_xlabel("iteration")
                plt.savefig(savefigName)
                plt.close()
            plot_cnt += 1

    best_pos = [x_pos_list[np.argmax(diff_list)], y_pos_list[np.argmax(diff_list)]]

    if usr_input:
        move2best = int(
            input(
                f"Spiral search complete. move to recommended best position = {best_pos}? enter 1 or 0"
            )
        )
    else:
        move2best = True

    if move2best:

        if use_multideviceserver:
            message = f"fpm_moveabs phasemask{beam} {best_pos}"
            phasemask.send_string(message)

        else:
            phasemask.move_absolute(best_pos)

    else:
        print("moving back to initial position")
        if use_multideviceserver:
            message = f"fpm_moveabs phasemask{beam} {starting_point}"
            phasemask.send_string(message)
        else:
            phasemask.move_absolute(starting_point)

    # phasemask.move_absolute( phasemask.phase_positions[phasemask_name]  )
    time.sleep(0.5)
    if savefigName != None:
        img = np.mean(
            cam.get_some_frames(number_of_frames=10, apply_manual_reduction=True),
            axis=0,
        )
        plt.figure()
        plt.imshow(img)
        plt.savefig(savefigName)
        plt.close()
    if usr_input:
        do_fine_adjustment = int(input("ready for fine adjustment? enter 1 or 0"))
    else:
        do_fine_adjustment = False

    if do_fine_adjustment:
        # do fine adjustments
        fine_adj_imgs = []
        for i in range(5):
            img = np.mean(
                cam.get_some_frames(number_of_frames=20, apply_manual_reduction=True),
                axis=0,
            )
            fine_adj_imgs.append(img)
            # dr = dr/2 # half movements each time
            dx, dy = calculate_movement_directions(
                img
            )  # dx, dy are normalized to radius 1
            phasemask.move_relative([dr * dx, dr * dy])

            if savefigName != 0:
                fig, ax = plt.subplots(1, 2, figsize=(14, 7))
                ax[0].imshow(fine_adj_imgs[0])
                ax[1].imshow(fine_adj_imgs[-1])
                ax[0].set_title("origin of fine adjustment")
                ax[1].set_title("current image of fine adjustment")
                plt.savefig(savefigName)
                plt.close()

            # fig,ax = plt.subplots( len(fine_adj_imgs))
            # for img,axx in zip(fine_adj_imgs,ax.reshape(-1)):
            #     axx.imshow( img )
            # plt.savefig( savefigName )

    if usr_input:
        manual_alignment = int(input("enter manual alignment mode? enter 1 or 0"))
        if manual_alignment:

            move_relative_and_get_image(cam, beam, phasemask, savefigName=savefigName)

    if not usr_input:  # we by default save the final image
        tstamp = datetime.datetime.now().strftime("%d-%m-%YT%H.%M.%S")
        final_img = np.mean(
            cam.get_some_frames(number_of_frames=10, apply_manual_reduction=True),
            axis=0,
        )
        fig, ax = plt.subplots(1, 2, figsize=(5, 10))
        ax[0].imshow(initial_img)
        ax[1].imshow(final_img)
        ax[0].set_title(f"initial ({phasemask_name})")
        ax[1].set_title(f"final ({phasemask_name})")
        plt.savefig(
            f"tmp/phasemask_alignment_SYD_{phasemask_name}_{tstamp}.png", dpi=150
        )
        plt.close()

    if usr_input:
        save_pos = int(input("save position? enter 1 or 0"))
    else:
        save_pos = True

    if save_pos:

        if use_multideviceserver:
            message = f"fpm_updatemaskpos phasemask{beam} {phasemask_name}"
            phasemask.send_string(message)

        else:
            phasemask.update_mask_position(phasemask_name)

    if use_multideviceserver:
        message = f"fpm_pos phasemask{beam}"
        phasemask.send_string(message)
        message = f"fpm_readpos phasemask{beam}"
        phasemask.send_string(message)
        pos = phasemask.recv_string()
    else:
        pos = phasemask.get_position()

    return pos





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

def move_relative_and_get_image(cam, beam, phasemask, savefigName=None, use_multideviceserver=True,roi=[None,None,None,None]):
    print(
        f"input savefigName = {savefigName} <- this is where output images will be saved.\nNo plots created if savefigName = None"
    )
    exit = 0
    while not exit:
        input_str = input('enter "e" to exit, else input relative movement in um: x,y')
        if input_str == "e":
            exit = 1
        else:
            try:
                xy = input_str.split(",")
                x = float(xy[0])
                y = float(xy[1])

                if use_multideviceserver:
                    #message = f"fpm_moveabs phasemask{beam} {[x,y]}"
                    #phasemask.send_string(message)
                    message = f"moverel BMX{beam} {x}"
                    phasemask.send_string(message)
                    response = phasemask.recv_string()
                    print(response)

                    message = f"moverel BMY{beam} {y}"
                    phasemask.send_string(message)
                    response = phasemask.recv_string()
                    print(response)

                else:
                    phasemask.move_relative([x, y])

                time.sleep(0.5)
                img = np.mean(
                    cam.get_some_frames(
                        number_of_frames=10, apply_manual_reduction=True
                    ),
                    axis=0,
                )
                if savefigName != None:
                    plt.figure()
                    plt.imshow( np.log10( img[roi[0]:roi[1],roi[2]:roi[3]] ) )
                    plt.colorbar()
                    plt.savefig(savefigName)
            except:
                print('incorrect input. Try input "1,1" as an example, or "e" to exit')



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

def create_bad_pixel_mask( search_dict, mean_thresh=6, std_thresh=20 ):
    # search_dict is a dictionary keyed by x,y coordinates with 
    # images (2D array like) as values.
    # create a bad pixel mask from search results (e.g. a dictionary returned from spiral_square_search_and_save_images() function)
    imgs = np.array( list(  search_dict.values() ) )

    positions = [eval(str(key)) for key in search_dict.keys()] # keys are sometimes strings, sometimes tuple ints.. so force to string so eval should always work
    x_positions, y_positions = zip(*positions)

    mean_frame = np.mean(imgs, axis=0)
    std_frame = np.std(imgs, axis=0)

    global_mean = np.mean(mean_frame)
    global_std = np.std(mean_frame)

    # thresh_grid =  np.linspace( 1, 50, 50)
    # no_bp=[]
    # for thr in thresh_grid:
    #     bad_pixel_map = (np.abs(mean_frame - global_mean) > 5 * global_std) | (std_frame > thr * np.median(std_frame))
    #     no_bp.append(  np.sum(bad_pixel_map) )
    # plt.semilogy(thresh_grid, no_bp); plt.show()

    bad_pixel_map = (np.abs(mean_frame - global_mean) > mean_thresh * global_std) | (std_frame > std_thresh * np.median(std_frame))

    return bad_pixel_map 



def find_optimal_clusters(images, detect_circle_function, max_clusters=10, plot_elbow=False):
    """
    Determines the optimal number of clusters for pupil centers using the Elbow Method.

    Parameters:
        images (list of 2D arrays): List of cropped grayscale images containing single pupils.
        detect_circle_function (function): Function to detect circular pupils (e.g., detect_circle).
        max_clusters (int): Maximum number of clusters to evaluate.
        plot_elbow (bool): If True, plots the Elbow Method graph.

    Returns:
        int: Optimal number of clusters.
    """
    # Detect pupils in all images
    centers = []
    for idx, image in enumerate(images):
        try:
            center_x, center_y, radius = detect_circle_function(image, plot=False)
            centers.append((center_x, center_y, radius))
        except Exception as e:
            print(f"Warning: Failed to detect circle in image {idx}. Error: {e}")
    
    # Convert to NumPy array and filter NaN values
    centers_array = np.array([c for c in centers if not np.isnan(c).any()])
    
    if len(centers_array) < 2:
        raise ValueError("Not enough valid pupil centers detected for clustering.")

    # Compute Within-Cluster Sum of Squares (WCSS) for different k values
    wcss = []
    cluster_range = range(1, min(max_clusters, len(centers_array)) + 1)
    
    for k in cluster_range:
        centroids, _ = kmeans(centers_array, k)
        labels, _ = vq(centers_array, centroids)
        wcss.append(sum(np.linalg.norm(centers_array - centroids[labels], axis=1)**2))

    # Find the "elbow" in WCSS curve
    diffs = np.diff(wcss)  # First derivative
    second_diffs = np.diff(diffs)  # Second derivative
    optimal_k = np.argmax(second_diffs) + 2  # Elbow is at max second derivative (+2 because of diff index shift)

    # Plot elbow curve
    if plot_elbow:
        plt.figure(figsize=(6, 4))
        plt.plot(cluster_range, wcss, marker='o', linestyle='-')
        plt.xlabel("Number of Clusters")
        plt.ylabel("WCSS (Within-Cluster Sum of Squares)")
        plt.title("Elbow Method for Optimal Clusters")
        plt.axvline(optimal_k, color='red', linestyle="--", label=f"Optimal k={optimal_k}")
        plt.legend()
        plt.show()

    return optimal_k


def cluster_analysis_on_searched_images(images, detect_circle_function, n_clusters=3, plot_clusters=False):
    """
    Detects circular pupils in a list of images, performs clustering on their positions and radii
    using scipy's k-means, and returns the cluster assignments for each image.

    Parameters:
        images (list of 2D arrays): List of cropped grayscale images containing single pupils.
        detect_circle_function (function): Function to detect circular pupils (e.g., your detect_circle function).
        n_clusters (int): Number of clusters to use for k-means clustering.
        plot_clusters (bool): If True, displays the clustering results.

    Returns:
        dict: A dictionary with keys:
            - "centers" (list): List of tuples (x, y, radius) for each detected pupil.
            - "clusters" (list): Cluster labels for each image.
            - "centroids" (ndarray): Centroids of the clusters.
    """
    # Step 1: Detect circles in all images
    centers = []
    for idx, image in enumerate(images):
        try:
            center_x, center_y, radius = detect_circle_function(image, plot=False)
            centers.append((center_x, center_y, radius))
        except Exception as e:
            print(f"Warning: Failed to detect circle in image {idx}. Error: {e}")
            centers.append((np.nan, np.nan, np.nan))  # Handle failure gracefully

    # Convert to a numpy array for clustering
    centers_array = np.array([center for center in centers if not np.isnan(center).any()])

    if len(centers_array) < n_clusters:
        raise ValueError("Number of valid centers is less than the number of clusters.")

    # Perform k-means clustering using scipy
    centroids, _ = kmeans(centers_array, n_clusters)
    cluster_labels, _ = vq(centers_array, centroids)

    #  Assign cluster labels back to all images (use NaN for failed detections)
    cluster_assignments = []
    idx_center = 0
    for center in centers:
        if np.isnan(center).any():
            cluster_assignments.append(np.nan)
        else:
            cluster_assignments.append(cluster_labels[idx_center])
            idx_center += 1

    # Plot clustering results (optional)
    if plot_clusters:
        plt.figure(figsize=(8, 6))
        for cluster_id in range(n_clusters):
            cluster_points = centers_array[cluster_labels == cluster_id]
            plt.scatter(cluster_points[:, 0], cluster_points[:, 1], label=f"Cluster {cluster_id}")
        plt.scatter(centroids[:, 0], centroids[:, 1], 
                    color="red", marker="x", s=100, label="Cluster Centers")
        plt.xlabel("X Coordinate")
        plt.ylabel("Y Coordinate")
        plt.legend()
        plt.title("Clustering of Detected Pupil Centers")
        plt.show()

    return {
        "centers": centers,
        "clusters": cluster_assignments,
        "centroids": centroids
    }



def plot_aggregate_cluster_images(images, clusters, operation="median"):
    """
    Computes and plots the aggregate (median, mean, or std) image for each cluster.

    Parameters:
        images (list of 2D arrays): List of images corresponding to the data points.
        clusters (list or array): Cluster labels corresponding to each image.
        operation (str): Statistical operation to apply ('median', 'mean', 'std').

    Returns:
        None
    """
    # Validate operation
    valid_operations = {"median", "mean", "std"}
    if operation not in valid_operations:
        raise ValueError(f"Invalid operation. Choose from {valid_operations}.")

    # Convert images to a NumPy array
    images_array = np.array(images)

    # Get unique clusters (exclude NaN)
    unique_clusters = [cluster for cluster in np.unique(clusters) if not np.isnan(cluster)]

    # Prepare the plot
    num_clusters = len(unique_clusters)
    fig, axes = plt.subplots(1, num_clusters, figsize=(6 * num_clusters, 6))
    if num_clusters == 1:
        axes = [axes]  # Ensure axes is iterable for a single cluster

    # Process and plot images for each cluster
    for ax, cluster in zip(axes, unique_clusters):
        # Get indices of images in the current cluster
        cluster_indices = np.where(np.array(clusters) == cluster)[0]

        # Stack the images for the current cluster
        cluster_images = images_array[cluster_indices]

        # Compute the aggregate image
        if operation == "median":
            aggregate_image = np.median(cluster_images, axis=0)
        elif operation == "mean":
            aggregate_image = np.mean(cluster_images, axis=0)
        elif operation == "std":
            aggregate_image = np.std(cluster_images, axis=0)

        # Plot the aggregate image
        im = ax.imshow(aggregate_image, cmap="viridis", origin="lower")
        ax.set_title(f"Cluster {int(cluster)} - {operation.capitalize()} Image")
        fig.colorbar(im, ax=ax, orientation="vertical")

    #plt.tight_layout()
    #plt.show()
    return fig, ax 



def plot_mask_positions( positions_dict):
    # positions_dict = {"H1":[x,y],"H2",[x,y]...}
    x = [xx[0] for xx in positions_dict.values()]
    y = [yy[1] for yy in positions_dict.values()]
    plt.figure()
    plt.plot(x,y,'x')
    plt.show()


def plot_image_grid(image_dict, savepath='delme.png'):
    """
    Plots images on a grid where positions correspond to their (x, y) keys,
    with dynamically computed extents for maximum size without overlap.
    
    works best for up to 6x6 scan ! (36 points total)

    Parameters:
        image_dict (dict): Dictionary with (x, y) tuple keys and image arrays as values.
    """
    # Extract unique x and y coordinates
    x_positions = sorted(set(pos[0] for pos in image_dict.keys()))
    y_positions = sorted(set(pos[1] for pos in image_dict.keys()))

    # Compute minimum spacing between points (avoiding overlap)
    dx = min(np.diff(x_positions)) if len(x_positions) > 1 else 1
    dy = min(np.diff(y_positions)) if len(y_positions) > 1 else 1

    # Scale factor to prevent touching (adjust if needed)
    scale_factor = 0.9  # Slightly less than full spacing to leave small gaps

    fig, ax = plt.subplots(figsize=(8, 8))

    for (x, y), img in image_dict.items():
        # Compute extent using calculated dx, dy
        extent = [
            x - (dx * scale_factor) / 2,  # Left boundary
            x + (dx * scale_factor) / 2,  # Right boundary
            y - (dy * scale_factor) / 2,  # Bottom boundary
            y + (dy * scale_factor) / 2   # Top boundary
        ]

        # Display the image
        ax.imshow(img, extent=extent, origin='upper', cmap='gray')

        # Add text label at the top of each image
        ax.text(x, y + (dy * 0.4), f"({round(x)}, {round(y)})", ha='center', va='bottom', fontsize=8, color='white',
                bbox=dict(facecolor='black', alpha=0.5, edgecolor='none', boxstyle='round,pad=0.3'))

    # Set limits based on x, y ranges
    ax.set_xlim(min(x_positions) - dx, max(x_positions) + dx)
    ax.set_ylim(min(y_positions) - dy, max(y_positions) + dy)

    # Set aspect ratio to equal
    ax.set_aspect('equal')

    # Remove axis ticks
    ax.set_xticks([])
    ax.set_yticks([])

    if savepath is not None:
        plt.savefig( savepath )
    plt.show()
    plt.close()


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


if __name__ == "__main__":

    from asgard_alignment.FLI_Cameras import fli
    import argparse
    import zmq

    ## ensure MultiDeviceServer is running first!! If not run the following command in terminal from asgard_alignment directory:
    ## asgard_alignment/MultiDeviceServer.py -c motor_info_full_system_with_DMs.json

    cam = fli()
    parser = argparse.ArgumentParser(description="ZeroMQ Client")
    parser.add_argument("--host", type=str, default="localhost", help="Server host")
    parser.add_argument("--port", type=int, default=5555, help="Server port")
    parser.add_argument(
        "--timeout", type=int, default=5000, help="Response timeout in milliseconds"
    )

    parser.parse_args()

    args = parser.parse_args()

    context = zmq.Context()

    context.socket(zmq.REQ)

    socket = context.socket(zmq.REQ)

    socket.setsockopt(zmq.RCVTIMEO, args.timeout)

    server_address = f"tcp://{args.host}:{args.port}"

    socket.connect(server_address)

    state_dict = {"message_history": [], "socket": socket}

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

    # socket.send_string(f"movetomask phasemask1 J1")
    # res = socket.recv_string()
    # print(f"Response: {res}")
