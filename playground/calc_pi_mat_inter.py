import numpy as np

np.set_printoptions(precision=8, suppress=True)

# using c red one pixel size, 20um

px_size = 3.45e-3  # pixel size in mm

# distances from the knife edges to the intermediate focus, in mm
dps = np.array([835, 966, 1077, 1194]) + 13

# dm = 341.3 + 307  # mask to fold + fold to intermediate focus
dm = 346 + 320  # mask to fold + fold to intermediate focus

#
ii = (
    2000 * np.pi / 180 * 1 / px_size
)  # pixels per degree of beam motion (NB not mirror)
ip = (2000 - dm) * np.pi / 180  # mm per degree of beam motion

for i, dp in enumerate(dps):
    pp = (dp - dm) * np.pi / 180  # how much the "pupil" motor moves the N1 beam (pupil)
    pi = dp * np.pi / 180 * 1 / px_size  # how much the "pupil" motor moves the image
    T = (
        np.array([[pp, ip], [pi, ii]]) * 2
    )  # *2 converting to beam angle from mirror angle
    print(f"Pupil and Image Matrix for beam {i+1}")
    print(np.linalg.inv(T))
