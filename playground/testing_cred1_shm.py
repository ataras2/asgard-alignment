import numpy as np
import time
import datetime
import sys
from pathlib import Path
import re
import os 
from astropy.io import fits
import json
import numpy as np
import matplotlib.pyplot as plt
from xaosim.shmlib import shm

"""
Frantz added new features to SHM to get most recent frames in 
SHM and semaphores. Email with report on 18/3/25
Here we test its functionality 
"""

c = shm("/dev/shm/cred1.im.shm", nosem=False)

# Imagine that the camera has been running for a while before you decide to do anything with these images,
# in your wavefront sensing/control application. By now, the image has probably been updated hundreds of
# thousands of times - and the semaphores attached to this image have been posted (ie. incremented) an equal
# number of times. Luckily, the C libImageStreamIO library on which all of this relies, ensures that the
# value of the semaphore saturates at 10: further image updates still occur, but the value of the semaphores
# are not further incremented.
# It’s therefore likely that when you want to start your loop again, you’re no longer going to be in sync with
# the acquisitions. Right before getting into your loop, you’ll have to run a single function call to catch-up:

semid = 0 # this is the index of the semaphore your application uses
c.catch_up_with_sem(semid)


# From now on, the idea is that each iteration will wait for a new image, by waiting for the camera server to
# post (ie increment) your semaphore, which it does when it dumps a camera frame to the shared memory.
# You have two options, depending on whether you’re reading from the global "full image" circular buffer
# shared memory (the current scenario at the time of this writing), or whether you’re reading from a single
# "live" subarray 2D image (the likely final scenario for your application).
# If you’re reading from the circular buffer, you should proceed like this:

# get some images 
for _ in range(10): # you're in your sensing/control loop
    img = c.get_latest_data_slice(semid)
    # then run your code



# (1) test that the frame counter increments correctly when we poll as fast as possible
cnt = []
c.catch_up_with_sem(semid)
for _ in range(1000): # you're in your sensing/control loop
    cnt.append( c.get_latest_data_slice(semid)[0][0] )
    # then run your code

if np.max(np.diff(cnt)) == 1 :
    print(f"Perfect!")

if np.max(np.diff(cnt)) > 1 :
    print(f"skipped {np.max(np.diff(cnt))} frames")

else:
    print("something strange is happending ")

plt.figure() ; plt.plot(cnt); plt.savefig('delme.png')

#############################################
# (2) testing with wrapper class that adds additional functionality 
# of directly communicating with camera and MDS via ZMQ and saving 
# data with camera configuration headers, also grabbing arbitary number of frames 
import matplotlib.pyplot as plt
import numpy as np
from astropy.io import fits
from asgard_alignment import FLI_Cameras as FLI

# region of interest 
roi = [ 188, 241, 194, 247]
c2 = FLI.fli(roi = roi)
# change camera settings via ZMQ to camera server 
c2.send_fli_cmd("set fps 100")
c2.send_fli_cmd("set gain 1")

# to get a list of all available commands
c2.print_camera_commands()

# build a dark (uses ZMQ to talk to MDS to turn off SSB source)
c2.build_manual_dark( no_frames = 200)

# this appends the dark to the self.reduction_dict
print( c2.reduction_dict.keys() )

# to look at the most recent dark here
plt.figure(); plt.imshow( c2.reduction_dict['dark'][-1]); plt.savefig('delme.png')

# to get any number of the most recent frames without reducing them 
# get some frames also looks at the frametag counter and prints warnings if
# frames are skipped (which they shouldn't with the semaphore!)
raw_frames = c2.get_some_frames(number_of_frames=103, apply_manual_reduction=False)

# if you want to reduce them using the dark we previously took 
red_frames = c2.get_some_frames(number_of_frames=103, apply_manual_reduction=True)

fig,ax = plt.subplots(1,2)
ax[0].imshow( np.mean( raw_frames, axis=0 ) )
ax[1].imshow( np.mean( red_frames, axis=0 ) )
ax[0].set_title('raw frame')
ax[1].set_title('dark subtracted frame')
plt.savefig('delme.png')

# to just get the most recent frame, reduce it, specifying which index
# in the reduction list to use 
new_frame = c2.get_image(apply_manual_reduction  = True, which_index = -1 )

# Finally to save a fits file with the full camera configuration 
# and any reduction products (dark, bias etc)
c2.save_fits( fname = 'delme.fits' ,  number_of_frames=101, apply_manual_reduction=False)

# to open and check structure
d = fits.open( "delme.fits" )
d.info()

# Filename: delme.fits
# No.    Name      Ver    Type      Cards   Dimensions   Format
#   0  FRAMES        1 PrimaryHDU      26   (53, 53, 101)   float64   
#   1  bias          1 ImageHDU         7   (0,)      
#   2  dark          1 ImageHDU         8   (53, 53)   int64   
#   3  flat          1 ImageHDU         7   (0,)      
#   4  bad_pixel_mask    1 ImageHDU         7   (0,)   