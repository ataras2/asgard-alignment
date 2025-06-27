"""
Here, the image plane motion per degree of mirror tilt is easy to calculate.

There is a demagnification of M=0.717, so the image-plane distance per degree 
of motion is simply:
np.pi/180 * (distance_to_intermediate_image) * M

This gives 1043 pixels per degree for the image mirrors, and less for the pupil mirrors
(in proportion to the distance "dp", which his the distance from the knife-edge
to the intermediate image plane)

The pupil plane motion is a little more complex. I started by considering the 
effect of the thin lens, which has an output angle:
theta_l = theta_in - Delta y/f, 
with Delta y = 2239 i, or  (239 + dp) p
This gave, with angles in radians:
Delta N1 = 271 i 
Delta N1 = (111 + 0.08 dp) p

The numbers below are just these multiplied by pi/180. But the key is that for beam 4 
the worse beam, pi/ii = 0.60, and pp/ip = 0.76. i.e. both mirrors seem to move the pupil 
and image by similar amounts.

We can double check by considering the relative motion of the re-imaged N1 plane.

N1 was calculated to be ~92mm from the reimaging lens, so it is re-imaged to a position 
1150mm behind the  re-imaging lens. This is 2000 + 1150mm=3150mm from the spherical 
mirror, and 1194+1150mm = 2344mm from the knife-edge. Unlike my intuition, these are not 
nearly equal, as they would hve been if N1 was 100mm from the re-imaging lens. 

Double-checking... OAP2 forms a virtual pupil image 234mm behind itself. Then the 
spherical mirror, 1200mm further on, forms another virtual pupil image 5m behind itself.
This means that the "92mm" probably should be 101.5mm.

Also, measuring on a screen shot gives different answers to the workbook, and more like 
103mm.

Choosing a round number of 100mm, we get:

Delta N = f theta, which is simple!

This seemed to work more or less. More details...



Measurements:

Move image by 10 pixels moves: 13 pixels... irrespective of N1_dist!
        For B4, there are 6 x 10 pixel moves before vignetting at N1_dist=104
        
Move pupil by 0.1mm.

***
xk = np.array([1092.2,1102.5,1114.7,1129.6])
zk = np.array([243.1,409.0,562.6,721.9])
xl = np.array([1025.4,868.6,729.1,590.7])
zl = np.array([233.7,406.9,606.3,820.9])
"""
plane = "CRed"
plane = "Baldr"
import numpy as np

np.set_printoptions(precision=5, suppress=True)

# using c red one pixel size, 24um

# distances from the knife edges to the intermediate focus, in mm

N1_dist = 101.5 

if (plane == "CRed"):
    dps = [835, 966, 1077, 1194]
    # 
    ii = 1043  # pixels per degree of beam motion (NB not mirror)
    #ip = 4.73  # mm per degree of beam motion
    #ip = 1.75  # Simplistic N1_dist=100
    ip = np.pi/180*(2239 - 21.39*N1_dist)
elif (plane == "Baldr"):
    dps = [67.5,233.9,388.1,547.9]
    ii = 29.56
    ip = 20.7
else:
    raise UserWarning


for i, dp in enumerate(dps):
    if (plane == "CRed"):
        #pp = 0.0014 * dp + 1.94  # how much the "pupil" motor moves the N1 beam (pupil)
        #pp = 1.75
        pp = np.pi/180*(dp + 239 + N1_dist*(1 - (dp + 239)/100))
        pi = 0.522 * dp  # how much the "pupil" motor moves the image
    elif (plane == "Baldr"):
        pi = 3.49
        pp = 0.145*dp
    
    T = (
        np.array([[pp, ip], [pi, ii]]) * 2
    )  # *2 converting to beam angle from mirror angle
    print(f"Pupil and Image Matrix for beam {i+1}")
    print(np.linalg.inv(T))
