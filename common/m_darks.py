
import numpy as np
from astropy.io import fits
import os
import time
import matplotlib.pyplot as plt 
import importlib
import datetime
import sys
from asgard_alignment import FLI_Cameras as FLI

"""
Script to run a series of images in different modes, frame rates and gain settings.
Available modes for the C-RED ONE are:
    globalreset: Set global reset mode (legacy compatibility)
    globalresetsingle: Set global reset mode (single frame)
    globalresetcds: Set global reset correlated double sampling
    globalresetbursts: Set global reset multiple non-destructive readout mode
    rollingresetsingle: Set rolling reset (single frame)
    rollingresetcds: Set rolling reset correlated double sampling (compatibility)
    rollingresetnro: Set rolling reset multiple non-destructive readout

default 
"""

def not_ok_response(ok):
    print('how do we deal with not oks?')

# def safety_check(  adu_threshold = 50000, no_pixels_above_threshold = 10):
#     frames = c.get_some_frames(  number_of_frames=10, apply_manual_reduction=False, timeout_limit = 200000 )
#     frames_med = np.median( frames, axis = 0)
#     if np.sum( frames_med  > adu_threshold ) > no_pixels_above_threshold:
#         plt.figure()
#         plt.imshow( frames_med ); plt.colorbar()
#         plt.savefig('delme.png')
#         plt.show()
#         ok = int( input( 'look at the median frames, should we continue?\n(input 0 for no, 1 for yes).\
#                         \nImage is saved as delme.png in the current directory') )
#         plt.close('all')
        
#     else:
#         ok = 1   

#     return ok


#################
# Set up
#################

apply_manual_reduction = False

light_mode = input('what mode are we running (dark, flat, etc). Input something meaningful')  

tstamp = datetime.datetime.now().strftime("%d-%m-%YT%H.%M.%S")
data_path = f'/home/heimdallr/Downloads/{light_mode}_series_{tstamp}/'
if not os.path.exists( data_path ):
    os.makedirs( data_path )


# set up camera object 
c = FLI.fli(cameraIndex=0, roi=[None, None, None, None])
# configure with default configuration file 
config_file_name = os.path.join( c.config_file_path , "default_cred1_config.json")
c.configure_camera( config_file_name )



# define our data grid 
mode_grid = ['globalresetsingle', 'globalresetcds']#, 'globalresetbursts','rollingresetsingle','rollingresetcds','rollingresetnro']
fps_grid = [25, 50, 100, 1000, 3000]
gain_grid = [1,5,10,20] #,40]

#################
# Run 
#################
ok = c.start_camera()

for gain in gain_grid :

    print(f'gain = {gain}')

    ok = c.send_fli_cmd( f"set gain {gain}" )
    time.sleep(1)
    print( f"applied {gain}, registered {c.send_fli_cmd( 'gain' )}" )

    if not ok:
        not_ok_response( ok )
    else:
        pass
    for mode in mode_grid:

        ok = c.send_fli_cmd( f"set mode {mode}" )
        time.sleep(10)

        print( f"applied {mode}, registered {c.send_fli_cmd( 'mode' )}" )

        if not ok[0]:
            not_ok_response( ok )


        for fps in fps_grid:

            ok = c.send_fli_cmd( f"set fps {fps}" )
            time.sleep(1)

            print( f"applied {fps}, registered {c.send_fli_cmd( 'fps' )}" )

            if not ok[0]:
                not_ok_response( ok )

            # safety check for high gain modes 
            if gain > 1:

                #ok = safety_check(  ) 
                adu_threshold = 40000
                no_pixels_above_threshold = 20
                frames = c.get_image( apply_manual_reduction  = True, which_index = -1 ) #c.get_some_frames(  number_of_frames=10, apply_manual_reduction=False, timeout_limit = 20000 )
                #frames_med = np.median( frames, axis = 0)
                if np.sum( frames  > adu_threshold ) > no_pixels_above_threshold:
                    plt.figure()
                    plt.imshow( frames ); plt.colorbar()
                    plt.savefig('delme.png')
                    plt.show()
                    safe = int( input( 'look at the median frames, should we continue?\n(input 0 for no, 1 for yes).\
                                    \nImage is saved as delme.png in the current directory') )
                    plt.close('all')
                    
                else:
                    safe = 1   

            else:
                safe = 1

            if safe: 

                fname = f'{light_mode}_mode-{mode}_gain-{gain}_fps-{fps}.fits'
                #frames = c.get_some_frames( number_of_frames=100, apply_manual_reduction=False )
                c.save_fits( data_path + fname  ,  number_of_frames=100, apply_manual_reduction=apply_manual_reduction )

            else:
                ok = c.send_fli_cmd( "set gain 1" )

                raise UserWarning('deemed not safe illumination, set gain to unity and existing program')
            

"""

a = fits.open()

plt.show()

fig, ax = plt.subplots(10,10 )

for axx, d in zip( ax.reshape(-1) , [dd for dd in a['FRAMES'].data]):
    axx.imshow(d - a['FRAMES'].data[0])
plt.show()


"""