"""
A script with functions for engineering
Includes functions for:
- move_image : move an image to a new location
- move_pupil : move the pupil (on N1) to a new location
"""

import numpy as np

pup_img_mat = {
    1: np.array([[0.44157, -0.002], [-0.18453, 0.00132]]),
    2: np.array([[0.49721, -0.00225], [-0.24038, 0.00157]]),
    3: np.array([[0.55664, -0.00252], [-0.30004, 0.00184]]),
    4: np.array([[0.63688, -0.00289], [-0.38058, 0.00221]]),
}

motor_orientation_mat = {
    1: np.array([[1, 0], [0, 1]]),
    2: np.array([[1, 1], [-1, 1]]) / np.sqrt(2),
    3: np.array([[1, 0], [0, 1]]),
    4: np.array([[1, 0], [0, 1]]),
}

"""
For example, to get a 1mm motion of the pupil for beam 1, we move the pupil mirror 
0.44 degrees and the image mirror 0.18 degrees.
"""


def move_image(beam_number, amount, write_message_fn):
    """
    Move the image to a new location

    Parameters:
        beam_number (int) : the beam number
        amount (float) : the amount to move the image by, in pixels on the detector
    """
    components = (motor_orientation_mat[beam_number] @ pup_img_mat[beam_number])[:, 1]

    scaled_components = components * amount

    # echo the move_rel commands
    write_message_fn(f"move_rel HTXP{beam_number} {scaled_components[0]}")
    write_message_fn(f"move_rel HTXI{beam_number} {scaled_components[1]}")


def move_pupil(beam_number, amount, write_message_fn):
    """
    Move the pupil to a new location

    Parameters:
        beam_number (int) : the beam number
        amount (float) : the amount to move the pupil by, in mm on N1
    """
    components = (motor_orientation_mat[beam_number] @ pup_img_mat[beam_number])[:, 0]

    scaled_components = components * amount

    write_message_fn(f"move_rel HTXP{beam_number} {scaled_components[0]}")
    write_message_fn(f"move_rel HTXI{beam_number} {scaled_components[1]}")


if __name__ == "__main__":
    move_image(1, 1, print)
    move_pupil(1, 1, print)
