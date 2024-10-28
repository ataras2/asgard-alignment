"""
A script with functions for engineering
Includes functions for:
- move_image : move an image to a new location
- move_pupil : move the pupil (on N1) to a new location
"""

import numpy as np
import time

# this is the N0, Cred one matrix:
# the first row is the pupil mirror, the second is the image mirror
pup_img_mat = {
    1: np.array([[0.44157, -0.002], [-0.18453, 0.00132]]),
    2: np.array([[0.49721, -0.00225], [-0.24038, 0.00157]]),
    3: np.array([[0.55664, -0.00252], [-0.30004, 0.00184]]),
    4: np.array([[0.63688, -0.00289], [-0.38058, 0.00221]]),
}

# this is the intermediate focus, pointgrey camera matrix:
# the first row is the pupil mirror, the second is the image mirror
pup_img_mat = {
    1: np.array([[-0.07586, 0.00018], [0.03167, -0.00002]]),
    2: np.array([[-0.08547, 0.0002], [0.04128, -0.00005]]),
    3: np.array([[-0.09575, 0.00022], [0.05156, -0.00007]]),
    4: np.array([[-0.10965, 0.00026], [0.06546, -0.0001]]),
}

# v2 using more decimals!
pup_img_mat = {
    1: np.array([[-0.07467856, 0.00017185], [0.03166371, -0.00002345]]),
    2: np.array([[-0.08426023, 0.0001939], [0.04124538, -0.00004549]]),
    3: np.array([[-0.09453813, 0.00021755], [0.05152328, -0.00006915]]),
    4: np.array([[-0.10848638, 0.00024964], [0.06547153, -0.00010124]]),
}


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
    2: LH_motor,
    3: LH_motor,
    4: RH_motor,
}

knife_edge_orientation_matricies = {
    1: RH_motor,
    2: angled_motor,
    3: RH_motor,
    4: RH_motor,
}


def move_image(beam_number, x, y, send_command):
    """
    Move an image to a new location
    x,y are in pixels
    """
    desired_deviation = np.array([[x], [y]])

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

    pupil_motor = np.linalg.inv(knife_edge_orientation_matricies[beam_number])
    image_motor = np.linalg.inv(spherical_orientation_matricies[beam_number])

    deviations_to_uv = np.block(
        [
            [pupil_motor, np.zeros((2, 2))],
            [np.zeros((2, 2)), image_motor],
        ]
    )

    beam_deviations = changes_to_deviations @ desired_deviation

    print(f"beam deviations: {beam_deviations}")

    uv_commands = deviations_to_uv @ beam_deviations
    axis_list = ["HTPP", "HTTP", "HTPI", "HTTI"]

    commands = [
        f"!moverel {axis}{beam_number} {command[0]}"
        for axis, command in zip(axis_list, uv_commands)
    ]

    # shuffle to parallelise
    send_command(commands[0])
    send_command(commands[2])
    time.sleep(0.5)
    send_command(commands[1])
    send_command(commands[3])

    return uv_commands, axis_list


def move_pupil(beam_number, x, y, send_command):
    """
    Move the pupil to a new location
    x,y are in mm
    """
    desired_deviation = np.array([[x], [y]])

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

    pupil_motor = np.linalg.inv(knife_edge_orientation_matricies[beam_number])
    image_motor = np.linalg.inv(spherical_orientation_matricies[beam_number])

    deviations_to_uv = np.block(
        [
            [pupil_motor, np.zeros((2, 2))],
            [np.zeros((2, 2)), image_motor],
        ]
    )

    beam_deviations = changes_to_deviations @ desired_deviation

    print(f"beam deviations: {beam_deviations}")

    uv_commands = deviations_to_uv @ beam_deviations
    axis_list = ["HTPP", "HTTP", "HTPI", "HTTI"]

    commands = [
        f"!moverel {axis}{beam_number} {command[0]}"
        for axis, command in zip(axis_list, uv_commands)
    ]

    send_command(commands[0])
    send_command(commands[2])
    time.sleep(0.5)
    send_command(commands[1])
    send_command(commands[3])

    return uv_commands, axis_list


if __name__ == "__main__":
    # print(move_image(1, 50, 50, print))
    # print(move_pupil(1, 1, 1, print))
    print(move_image(1, 50, 0, print))
    print(move_image(1, 0, 50, print))

    print(move_image(4, 50, 0, print))
    print(move_image(4, 0, 50, print))
    # print(move_pupil(1, 1, 1, print))
