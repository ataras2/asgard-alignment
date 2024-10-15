import sys
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import argparse
import os
#import bmc
sys.path.insert(1,'/opt/Boston Micromachines/lib/Python3/site-packages/')
import bmc
import atexit
# Dynamically add the path to pyBaldr based on the location of this script
# script_dir = os.path.dirname(os.path.realpath(__file__))
# pyBaldr_path = os.path.join(script_dir, '../pyBaldr/')
# sys.path.append(pyBaldr_path)
# from pyBaldr import utilities as util

# Load predefined shapes and DM serial numbers
## >>>> ADAM - YOU WILL NEED TO UPDATE THIS DICTIONARY WITH THE CORRECT SERIAL NUMBERS FOR YOUR DMs <<<<<< 
DM_serial_number_dict = {'1':'17DW019#122', '2': '17DW019#122', '3': '17DW019#122', '4':'17DW019#122'}  
DMshapes_path = 'DMShapes/'

crosshair = pd.read_csv(DMshapes_path + 'Crosshair140.csv', header=None)[0].values
fourTorres = pd.read_csv(DMshapes_path + 'four_torres.csv', header=None)[0].values


def close_dm():
    try:
        dm.close_dm()
    except:
        print( 'Failed to close DM or DM object does not exist' )
atexit.register(close_dm)
   
def get_DM_command_in_2D(cmd,Nx_act=12):
    # function so we can easily plot the DM shape (since DM grid is not perfectly square raw cmds can not be plotted in 2D immediately )
    #puts nan values in cmd positions that don't correspond to actuator on a square grid until cmd length is square number (12x12 for BMC multi-2.5 DM) so can be reshaped to 2D array to see what the command looks like on the DM.
    corner_indices = [0, Nx_act-1, Nx_act * (Nx_act-1), Nx_act*Nx_act]
    cmd_in_2D = list(cmd.copy())
    for i in corner_indices:
        cmd_in_2D.insert(i,np.nan)
    return( np.array(cmd_in_2D).reshape(Nx_act,Nx_act) )


def main(beam, shape, strength, plot_shape = False):
    
    SIMULATION = False
    # Initialize deformable mirror
    if SIMULATION:
        dm = {}
        dm_err_flag = 0
    else:
        dm = bmc.BmcDm()
        dm_err_flag = dm.open_dm(DM_serial_number_dict[beam])
        
    flatdm = pd.read_csv(DMshapes_path + '{}_FLAT_MAP_COMMANDS.csv'.format(DM_serial_number_dict[beam]), header=None)[0].values
    
    # Define available shapes    
    available_shapes = {
    'flat': flatdm,
    'crosshair': crosshair,
    'four_torres': fourTorres,
    }

    if dm_err_flag != 0:
        print(f"Error opening DM: {dm_err_flag}")
        sys.exit(1)

    # Get the shape from the command-line argument
    if shape not in available_shapes:
        print(f"Shape '{shape}' not recognized. Available shapes: {list(available_shapes.keys())}")
        sys.exit(1)

    # Apply shape with specified strength
    selected_shape = available_shapes[shape]
    flat_shape = available_shapes['flat']
    if shape == 'flat':
        dm_command = flat_shape 
    else:
        dm_command = flat_shape + strength * selected_shape
     
    if not SIMULATION:
        dm.send_data(dm_command)

    print(f"Applied {shape} shape with strength {strength} on beam {beam}")
    
    if plot_shape:
        print('plotting...')
        plt.figure(1)
        plt.imshow( get_DM_command_in_2D(dm_command) )
        plt.colorbar( label='DM command' ) 
        plt.show( )
        
if __name__ == "__main__":
    # Set up argument parser
    parser = argparse.ArgumentParser(description="Apply a shape to the DM")
    parser.add_argument("-beam", type=str, required=True, help="Beam identifier")
    parser.add_argument("-shape", type=str, required=True, choices=['crosshair', 'flat', 'four_torres'], help="Shape to apply")
    parser.add_argument("-strength", type=float, required=True, help="Strength of the shape to apply")
    parser.add_argument("-plot_shape", type=bool, required=False, help="Do you want to plot the shape you are applying?")
    
    # Parse arguments
    args = parser.parse_args()

    # Run main function
    main(args.beam, args.shape, args.strength, args.plot_shape)
