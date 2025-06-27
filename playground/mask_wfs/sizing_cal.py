import numpy as np
import astropy.units as u


def rotm(angle):
    return np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])


scale = 1.36 * u.mm

hole_coords = (
    np.vstack(
        [
            np.array([0.0, 0.0]),
            np.array([0.0, -3.0]),
            np.array([2.0, -3.0]),
            np.array([0.0, -3.0]) @ rotm(2 * np.pi / 3),
            np.array([2.0, -3.0]) @ rotm(2 * np.pi / 3),
            np.array([0.0, -3.0]) @ rotm(4 * np.pi / 3),
            np.array([2.0, -3.0]) @ rotm(4 * np.pi / 3),
        ]
    )
    * scale
)

print(hole_coords)

import itertools

baseline_lengths = (
    np.array(
        [
            np.linalg.norm(h1 - h2).value
            for h1, h2 in itertools.combinations(hole_coords, 2)
        ]
    )
    * u.mm
)


# pixel_size = 6.0 * u.micron
pixel_size = 3.45 * u.micron
# beam_diam = 12.0 * u.mm
longest_baseline = np.max(baseline_lengths)
print(f"Longest Baseline: {longest_baseline}")
focal_length = 125.0 * u.mm
wavelength = 0.635 * u.micron

# calulate the plate scale
plate_scale = np.arctan(pixel_size / focal_length)

# calulate the angular fringe spacing
fringe_spacing = np.arctan(wavelength / (2 * longest_baseline))

print(f"Plate Scale: {plate_scale.to(u.arcsec)}")
print(f"Angular Fringe Spacing: {fringe_spacing.to(u.arcsec)}")

print(f"ratio: {fringe_spacing/plate_scale}")
