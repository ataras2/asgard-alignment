import numpy as np

np.set_printoptions(precision=5, suppress=True)

# using c red one pixel size, 20um

# distances from the knife edges to the intermediate focus, in mm
dps = [835, 966, 1077, 1194]

# 
ii = 1043  # pixels per degree of beam motion (NB not mirror)
ip = 4.73  # mm per degree of beam motion

for i, dp in enumerate(dps):
    pp = 0.0014 * dp + 1.94  # how much the "pupil" motor moves the N1 beam (pupil)
    pi = 0.522 * dp  # how much the "pupil" motor moves the image
    T = (
        np.array([[pp, ip], [pi, ii]]) * 2
    )  # *2 converting to beam angle from mirror angle
    print(f"Pupil and Image Matrix for beam {i+1}")
    print(np.linalg.inv(T))
