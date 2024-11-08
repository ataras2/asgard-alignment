
import numpy as np
from astropy.io import fits
import time
import matplotlib.pyplot as plt 
import importlib
import datetime
import sys
sys.path.append('pyBaldr/' )  
from pyBaldr import ZWFS

#################
# Set up
#################

tstamp = datetime.datetime.now().strftime("%d-%m-%YT%H.%M.%S")
data_path = '~/Downloads/'
pupil_crop_region = [None, None, None, None] # [204,268,125, 187] #[None, None, None, None] #[204 -50 ,268+50,125-50, 187+50] 
DM_serial_number = None # DM will go to simulation mode 
#init our ZWFS object (interacts with camera and/or DMs)
zwfs = ZWFS.ZWFS(DM_serial_number=DM_serial_number, cameraIndex=0, DMshapes_path = 'DMShapes/', pupil_crop_region=pupil_crop_region ) 

zwfs.deactive_cropping() # zwfs.set_camera_cropping(r1, r2, c1, c2 ) #<- use this for latency tests , set back after with zwfs.set_camera_cropping(0, 639, 0, 511 ) 
zwfs.set_camera_dit( 0.001 );time.sleep(0.2)
zwfs.set_camera_fps( 100 );time.sleep(0.2)
### !!! TEST THIS 
zwfs.set_sensitivity(1);time.sleep(0.2)  # CRED 1  !
#####
# zwfs.set_sensitivity('high');time.sleep(0.2) # CRED 2 or 3
zwfs.enable_frame_tag(tag = True);time.sleep(0.2)
zwfs.bias_off();time.sleep(0.2)
zwfs.flat_off();time.sleep(0.2)
# ".flf" is a FirstLight format. It is a very simple proprietary format. A fixed 2048 bytes header is added at the beginning of the file. In this header, the camera configuration is written in ascii such a way it can be easily read. The fields are separated by semicolon.
# FliCred_saveCameraSettings_V2()

#################
# get series of darks
#################
dark_dict = {}
gain_grid = np.linspace(1, 20, 5)
tint_grid = 1e-3 * np.logspace(-1, 1, 10) 
number_of_frames = 500
for gain in gain_grid:
            
    zwfs.set_sensitivity( gain )
    time.sleep(0.2)

    dark_dict[gain] = {}

    for tint in tint_grid:
    
        zwfs.set_set_camera_fps( 1  / ( 1.05 * tint ) )
        mindit, maxdit = zwfs.get_dit_limits()
        zwfs.set_camera_dit( maxdit )
        time.sleep(0.2)

        dark_dict[gain][maxdit] =  zwfs.get_some_frames(number_of_frames=number_of_frames,\
                                                       apply_manual_reduction=False, timeout_limit = 20000)  


#################
# Save
#################
hdulist = fits.HDUList([])
for gain, nested_dict in dark_dict.items():
    
    for tint, frames in nested_dict:
        # Convert list to numpy array for FITS compatibility
        data_array = np.array(data_list, dtype=float)  # Ensure it is a float array or any appropriate type

        # Create a new ImageHDU with the data
        hdu = fits.ImageHDU( np.array(frames) )

        # Set the EXTNAME header to the variable name
        hdu.header['EXTNAME'] = f'DIT-{round(1e3*tint,1)}_GAIN-{round(gain,1)}'
        hdu.header['tint'] = tint
        hdu.header['gain'] = gain
        # Append the HDU to the HDU list
        hdulist.append(hdu)

hdulist.writeto(data_path + f'dark_series_{tstamp}.fits', overwrite=True)
    



