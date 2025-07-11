"""
A script with functions for engineering
Includes functions for:
- move_image : move an image to a new location
- move_pupil : move the pupil (on N1) to a new location
"""

import numpy as np
import time


def get_matricies(config):
    """
    Get the matricies for moving the pupil and image, depending on wthe configuration

    Parameters
    ----------
    config : str
        The configuration to use - either "c_red_one_focus" or "intermediate_focus" or "baldr"

    Returns
    -------
    pupil_move_matricies : dict
        A dictionary of matricies for moving the pupil
    image_move_matricies : dict
        A dictionary of matricies for moving the image
    """
    if config == "c_red_one_focus":
        # this is the N1, Cred one matrix:
        pup_img_mat = {
            1: np.array([[0.50252, -0.00057], [-0.21, 0.00072]]),
            2: np.array([[0.56627, -0.00064], [-0.27377, 0.00079]]),
            3: np.array([[0.63447, -0.00072], [-0.34199, 0.00087]]),
            4: np.array([[0.72672, -0.00083], [-0.43427, 0.00097]]),
        }
    elif config == "intermediate_focus":
        # v2 using more decimals!
        pup_img_mat = {
            1: np.array([[-0.07467856, 0.00017185], [0.03166371, -0.00002345]]),
            2: np.array([[-0.08426023, 0.0001939], [0.04124538, -0.00004549]]),
            3: np.array([[-0.09453813, 0.00021755], [0.05152328, -0.00006915]]),
            4: np.array([[-0.10848638, 0.00024964], [0.06547153, -0.00010124]]),
        }
    elif config == "baldr":
        # Baldr matrix from 5/3/25 calculation
        pup_img_mat = {
            1: np.array([[0.06809, -0.04768], [-0.00804, 0.02254]]),
            2: np.array([[0.01589, -0.01113], [-0.00188, 0.01823]]),
            3: np.array([[0.00929, -0.0065], [-0.0011, 0.01768]]),
            4: np.array([[0.00649, -0.00455], [-0.00077, 0.01745]]),
        }

    else:
        raise ValueError("Invalid configuration")

    # amount in degrees needed to move a mirror by 1mm on N1
    pupil_move_matricies = {
        1: pup_img_mat[1][:, 0],
        2: pup_img_mat[2][:, 0],
        3: pup_img_mat[3][:, 0],
        4: pup_img_mat[4][:, 0],
    }

    # amount in degrees needed to move by 1 pixel at the focus
    image_move_matricies = {
        1: pup_img_mat[1][:, 1],
        2: pup_img_mat[2][:, 1],
        3: pup_img_mat[3][:, 1],
        4: pup_img_mat[4][:, 1],
    }

    return pupil_move_matricies, image_move_matricies


# matrix to calculate phasemask relative offsets (BMX,BMY)
# based on relative OAP mirror tip/tilt (BOTX) offsets  ()
# calculate / calibrated 13/3/25 baldr logbook
# convention is [BMX, BMY] = Matrix . [BOTP , BOTT]
# on eng GUI BOTP is U, BOTT is V
# convention delta_U = after - before
# NO BOTX ON BEAM 1 !
phasemask_botx_matricies = {
    1: np.array([[0.0, 0.0], [0.0, 0.0]]),
    2: np.array([[0.0, -80 / 0.01], [80 / 0.01, 0.0]]),
    3: np.array([[0.0, -80 / 0.01], [80 / 0.01, 0.0]]),
    4: np.array([[0.0, 80 / 0.01], [80 / 0.01, 0.0]]),
}

# matricies for the M100D, that solve [beamx, beamy] = matrix @ [u, v]
RH_motor = np.array(
    [
        [0.0, 1.0],
        [-1.0, 0.0],
    ]
)

LH_motor = np.array(
    [
        [-1.0, 0.0],
        [0.0, -1.0],
    ]
)

angled_motor = (
    1
    / (np.sqrt(2))
    * np.array(
        [
            [-1.0, 1.0],
            [-1.0, -1.0],
        ]
    )
)

# describes the mapping from horizontal movements to u,v directions
spherical_orientation_matricies = {
    1: RH_motor,
    2: RH_motor,
    3: LH_motor,
    4: RH_motor,
}

knife_edge_orientation_matricies = {
    1: RH_motor,
    2: angled_motor,
    3: RH_motor,
    4: RH_motor,
}


def move_img_calc(config, beam_number, desired_deviation):
    _, image_move_matricies = get_matricies(config)

    M_I = image_move_matricies[beam_number]
    M_I_pupil = M_I[0]
    M_I_image = M_I[1]

    changes_to_deviations = np.array(
        [
            [M_I_pupil, 0.0],
            [0.0, M_I_pupil],
            [M_I_image, 0.0],
            [0.0, M_I_image],
        ]
    )

    # used for heimdallr
    ke_matrix = knife_edge_orientation_matricies
    so_matrix = spherical_orientation_matricies

    if config == "baldr":
        pupil_motor = RH_motor
        image_motor = LH_motor
    else:
        pupil_motor = np.linalg.inv(ke_matrix[beam_number])
        image_motor = np.linalg.inv(so_matrix[beam_number])

    deviations_to_uv = np.block(
        [
            [pupil_motor, np.zeros((2, 2))],
            [np.zeros((2, 2)), image_motor],
        ]
    )

    beam_deviations = changes_to_deviations @ desired_deviation

    uv_commands = deviations_to_uv @ beam_deviations
    return uv_commands


def move_pup_calc(config, beam_number, desired_deviation):

    pupil_move_matricies, _ = get_matricies(config)

    M_P = pupil_move_matricies[beam_number]
    M_P_pupil = M_P[0]
    M_P_image = M_P[1]

    changes_to_deviations = np.array(
        [
            [M_P_pupil, 0.0],
            [0.0, M_P_pupil],
            [M_P_image, 0.0],
            [0.0, M_P_image],
        ]
    )

    ke_matrix = knife_edge_orientation_matricies
    so_matrix = spherical_orientation_matricies
    # used for baldr
    # LH_motor = LH_motor
    # RH_motor = RH_motor

    if config == "baldr":
        # Baldr has a different orientation. This will be correct
        # up to a sign in front of one of the motors.
        pupil_motor = RH_motor
        image_motor = LH_motor
    else:
        pupil_motor = np.linalg.inv(ke_matrix[beam_number])
        image_motor = np.linalg.inv(so_matrix[beam_number])

    deviations_to_uv = np.block(
        [
            [pupil_motor, np.zeros((2, 2))],
            [np.zeros((2, 2)), image_motor],
        ]
    )

    beam_deviations = changes_to_deviations @ desired_deviation

    print(f"beam deviations: {beam_deviations}")

    uv_commands = deviations_to_uv @ beam_deviations
    return uv_commands


## Move image and pupil functions belong only as method to instrument class
# To test with this commented out, eventually to delete once stable.
# def move_image(beam_number, x, y, send_command, config):
#     """
#     Move image to a new location (relative motion)
#     x,y are in pixels

#     Parameters
#     ----------
#     beam_number : int
#         The beam number to move
#     x : float
#         The x coordinate to move to, in pixels
#     y : float
#         The y coordinate to move to, in pixels
#     send_command : function
#         A function to send commands to the motors
#     config : str
#         The configuration to use - either "c_red_one_focus" or "intermediate_focus" or "baldr"

#     Returns
#     -------
#     uv_commands : np.array
#         The u,v commands sent to the motors
#     axis_list : list
#         The list of axis names
#     """
#     desired_deviation = np.array([[x], [y]])

#     _, image_move_matricies = get_matricies(config)

#     M_I = image_move_matricies[beam_number]
#     M_I_pupil = M_I[0]
#     M_I_image = M_I[1]

#     changes_to_deviations = np.array(
#         [
#             [M_I_pupil, 0.0],
#             [0.0, M_I_pupil],
#             [M_I_image, 0.0],
#             [0.0, M_I_image],
#         ]
#     )

#     if config == "baldr":
#         # Baldr has a different orientation. This will be correct
#         # up to a sign in front of one of the motors.
#         pupil_motor = RH_motor
#         image_motor = LH_motor
#     else:
#         pupil_motor = np.linalg.inv(knife_edge_orientation_matricies[beam_number])
#         image_motor = np.linalg.inv(spherical_orientation_matricies[beam_number])

#     deviations_to_uv = np.block(
#         [
#             [pupil_motor, np.zeros((2, 2))],
#             [np.zeros((2, 2)), image_motor],
#         ]
#     )

#     beam_deviations = changes_to_deviations @ desired_deviation

#     print(f"beam deviations: {beam_deviations}")

#     uv_commands = deviations_to_uv @ beam_deviations
#     if config == "baldr":
#         axis_list = ["BTP", "BTT", "BOTP", "BOTT"]
#     else:
#         axis_list = ["HTPP", "HTTP", "HTPI", "HTTI"]

#     commands = [
#         f"moverel {axis}{beam_number} {command[0]}"
#         for axis, command in zip(axis_list, uv_commands)
#     ]

#     # shuffle to parallelise
#     send_command(commands[0])
#     send_command(commands[2])
#     time.sleep(0.5)
#     send_command(commands[1])
#     send_command(commands[3])
#     time.sleep(0.5)

#     return uv_commands, axis_list


# def move_pupil(beam_number, x, y, send_command, config):
#     """
#     Move the pupil to a new location
#     x,y are in mm

#     Parameters
#     ----------
#     beam_number : int
#         The beam number to move
#     x : float
#         The x coordinate to move to
#     y : float
#         The y coordinate to move to
#     send_command : function
#         A function to send commands to the motors
#     config : str
#         The configuration to use - either "c_red_one_focus" or "intermediate_focus" of "baldr"


#     Returns
#     -------
#     uv_commands : np.array
#         The u,v commands sent to the motors
#     axis_list : list
#         The list of axis names
#     """
#     desired_deviation = np.array([[x], [y]])

#     pupil_move_matricies, _ = get_matricies(config)

#     M_P = pupil_move_matricies[beam_number]
#     M_P_pupil = M_P[0]
#     M_P_image = M_P[1]

#     changes_to_deviations = np.array(
#         [
#             [M_P_pupil, 0.0],
#             [0.0, M_P_pupil],
#             [M_P_image, 0.0],
#             [0.0, M_P_image],
#         ]
#     )

#     if config == "baldr":
#         # Baldr has a different orientation. This will be correct
#         # up to a sign in front of one of the motors.
#         pupil_motor = RH_motor
#         image_motor = LH_motor
#     else:
#         pupil_motor = np.linalg.inv(knife_edge_orientation_matricies[beam_number])
#         image_motor = np.linalg.inv(spherical_orientation_matricies[beam_number])


#     deviations_to_uv = np.block(
#         [
#             [pupil_motor, np.zeros((2, 2))],
#             [np.zeros((2, 2)), image_motor],
#         ]
#     )

#     beam_deviations = changes_to_deviations @ desired_deviation

#     print(f"beam deviations: {beam_deviations}")

#     uv_commands = deviations_to_uv @ beam_deviations
#     if config == "baldr":
#         axis_list = ["BTP", "BTT", "BOTP", "BOTT"]
#     else:
#         axis_list = ["HTPP", "HTTP", "HTPI", "HTTI"]

#     commands = [
#         f"moverel {axis}{beam_number} {command[0]}"
#         for axis, command in zip(axis_list, uv_commands)
#     ]

#     send_command(commands[0])
#     send_command(commands[2])
#     time.sleep(0.5)
#     send_command(commands[1])
#     send_command(commands[3])
#     time.sleep(0.5)

#     return uv_commands, axis_list


if __name__ == "__main__":
    # print(move_image(1, 50, 50, print))
    # print(move_pupil(1, 1, 1, print))
    config = "intermediate_focus"
    print(move_image(1, 50, 0, print, config))
    print(move_image(1, 0, 50, print, config))

    print(move_image(4, 50, 0, print, config))
    print(move_image(4, 0, 50, print, config))
    # print(move_pupil(1, 1, 1, print))
