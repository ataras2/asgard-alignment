import numpy as np
import time
from scipy.ndimage import center_of_mass
import matplotlib.pyplot as plt
import scipy.interpolate as interp


# "!fpm_movetomask {} {}": fpm_move_to_phasemask_msg,
# "!fpm_moverel {} {}": fpm_move_relative_msg,
# "!fpm_moveabs {} {}": fpm_move_absolute_msg,
# "!fpm_readpos {}": fpm_read_position_msg,
# "!fpm_updatemaskpos {} {}": fpm_update_mask_position_msg,
# "!fpm_updatemaskpos {}": fpm_write_mask_positions_msg,
# "!fpm_updateallmaskpos {} {} {}": fpm_update_all_mask_positions_relative_to_current_msg,


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


def move_relative_and_get_image(cam, beam, phasemask, savefigName=None, use_multideviceserver=True):
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
                    plt.imshow( np.log10( img) )
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
