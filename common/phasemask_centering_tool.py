import numpy as np
import time
import datetime
from scipy.ndimage import center_of_mass
import matplotlib.pyplot as plt
import scipy.interpolate as interp
from matplotlib import animation

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


def spiral_square_search_and_save_images(
    cam,
    beam,
    phasemask,
    starting_point,
    step_size,
    search_radius,
    sleep_time=1,
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
            y_pox = 1
        if y_pos >= 10000:
            print('y_pos > 10000. set y_pos = 9999')
            y_pos = 9999

        if use_multideviceserver:
            #message = f"!fpm_moveabs phasemask{beam} {[x_pos, y_pos]}"
            message = f"!moveabs BMX{beam} {x_pos}"
            phasemask.send_string(message)
            response = phasemask.recv_string()
            print(response)

            message = f"!moveabs BMY{beam} {y_pos}"
            phasemask.send_string(message)
            response = phasemask.recv_string()
            print(response)
        else:
            phasemask.move_absolute([x_pos, y_pos])

        time.sleep(sleep_time)  # wait for the phase mask to move and settle
        img = np.mean(
            cam.get_some_frames(number_of_frames=10, apply_manual_reduction=True),
            axis=0,
        )

        img_dict[(x_pos, y_pos)] = img

    return img_dict


def analyse_search_results(search, savepath="delme.png"):
    """
    analyse results from spiral_square_search_and_save_images
    search = img_dict output from spiral_square_search_and_save_images function.
    """
    coord = np.array([k for k, v in search.items()])
    img_list = np.array([v for k, v in search.items()])

    # make the ref image the median 
    reftmp = np.median( img_list ,axis = 0)

    dif_imgs = np.mean((img_list[0] - img_list) ** 2, axis=(1, 2))

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
        plt.imshow(np.log10(img_candidate))
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

def display_scatter_and_image(data_dict):
    """
    
    Displays an interactive plot with:
    - A scatter plot of x, y positions up to the current index.
    - An image corresponding to the current index.


    Parameters:
    - data_dict: Dictionary where keys are x, y positions (string tuples) and
                 values are 2D arrays (images).
                 !!!!!!!!!
                 designed to use img_dict returned from spiral_square_search_and_save_images()
                 as input 
                 !!!!!!!!!
    """
    # Extract and process data
    positions = [eval(key) for key in data_dict.keys()]
    images = list(data_dict.values())
    x_positions, y_positions = zip(*positions)

    num_frames = len(positions)

    # Create figure with two subplots
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    scatter_ax, image_ax = axes

    # Initial scatter plot
    scatter = scatter_ax.scatter(x_positions[:1], y_positions[:1], c='b', label='Positions')
    scatter_ax.set_xlim(min(x_positions) - 1, max(x_positions) + 1)
    scatter_ax.set_ylim(min(y_positions) - 1, max(y_positions) + 1)
    scatter_ax.set_xlabel("X Position")
    scatter_ax.set_ylabel("Y Position")
    scatter_ax.set_title("Scatter Plot of Positions")
    scatter_ax.legend()

    # Initial image plot
    img_display = image_ax.imshow(images[0], cmap='viridis')
    cbar = fig.colorbar(img_display, ax=image_ax)
    cbar.set_label("Intensity")
    image_ax.set_title("Image at Current Position")

    # Slider setup
    ax_slider = plt.axes([0.2, 0.02, 0.65, 0.03], facecolor='lightgoldenrodyellow')
    frame_slider = Slider(ax_slider, 'Frame', 0, num_frames - 1, valinit=0, valstep=1)

    # Update function for the slider
    def update(val):
        index = int(frame_slider.val)

        # Update scatter plot
        scatter.set_offsets(np.c_[x_positions[:index + 1], y_positions[:index + 1]])

        # Update image plot
        img_display.set_data(images[index])

        fig.canvas.draw_idle()

    # Connect the slider to the update function
    frame_slider.on_changed(update)

    plt.tight_layout()
    plt.show()



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
    search_radius,
    dr,
    dtheta,
    reference_img,
    fine_tune_threshold=3,
    savefigName=None,
    usr_input=True,
):

    if use_multideviceserver:
        message = f"!fpm_movetomask phasemask{beam} {phasemask_name}"
        phasemask.send_string(message)

        message = f"!fpm_readpos phasemask{beam}"
        phasemask.send_string(message)
        initial_pos = phasemask.recv_string()

    else:
        phasemask.move_to_mask(phasemask_name)  # move to phasemask
        initial_pos = phasemask.phase_positions[phasemask_name]  # set initial position

    x, y = initial_pos
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
                    [initial_pos[0], initial_pos[1]],
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
                    [initial_pos[0] - search_radius, initial_pos[0] + search_radius]
                )
                ax[1].set_ylim(
                    [initial_pos[1] - search_radius, initial_pos[1] + search_radius]
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
            message = f"!fpm_moveabs phasemask{beam} {best_pos}"
            phasemask.send_string(message)

        else:
            phasemask.move_absolute(best_pos)

    else:
        print("moving back to initial position")
        if use_multideviceserver:
            message = f"!fpm_moveabs phasemask{beam} {initial_pos}"
            phasemask.send_string(message)
        else:
            phasemask.move_absolute(initial_pos)

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
            message = f"!fpm_updatemaskpos phasemask{beam} {phasemask_name}"
            phasemask.send_string(message)

        else:
            phasemask.update_mask_position(phasemask_name)

    if use_multideviceserver:
        message = f"!fpm_pos phasemask{beam}"
        phasemask.send_string(message)
        message = f"!fpm_readpos phasemask{beam}"
        phasemask.send_string(message)
        pos = phasemask.recv_string()
    else:
        pos = phasemask.get_position()

    return pos


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
                    #message = f"!fpm_moveabs phasemask{beam} {[x,y]}"
                    #phasemask.send_string(message)
                    message = f"!moverel BMX{beam} {x}"
                    phasemask.send_string(message)
                    response = phasemask.recv_string()
                    print(response)

                    message = f"!moverel BMY{beam} {y}"
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


if __name__ == "__main__":

    print(
        " THIS TAKES SEVERAL MINUTES TO RUN. WHAT WE ARE DOING IS: \n\
     - connect to motors, DM and camera. Set up detector with darks, bad pixel mask etc.\n \
     - iterate through all phase masks on beam 3 (in Sydney) and update phasemask positions and save them.\n \
     - This should only serve as example. must be called from asgard_alignment folder\n \
    DEVELOPED IN SYDNEY WITH MOTORS ONLY ON BEAM 3 - UPDATE ACCORDINGLY \n \
    DOING AUTOMATED SEARCH OVER LIMITED RADIUS, IF RESULTS ARE POOR - ADJUST SEARCH RADIUS / GRID.\
    "
    )
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

    # socket.send_string(f"!movetomask phasemask1 J1")
    # res = socket.recv_string()
    # print(f"Response: {res}")
