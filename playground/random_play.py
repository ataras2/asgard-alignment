import numpy as np
import glob 
from astropy.io import fits
import time
import os 
import matplotlib.pyplot as plt 
import importlib
#import rtc
import sys
import datetime
sys.path.append('pyBaldr/' )  
sys.path.insert(1, '/opt/FirstLightImaging/FliSdk/Python/demo/')
sys.path.insert(1,'/opt/Boston Micromachines/lib/Python3/site-packages/')

from pyBaldr import utilities as util
from pyBaldr import ZWFS
from pyBaldr import phase_control
from pyBaldr import pupil_control

from playground import phasemask_centering_tool 

import bmc
import FliSdk_V2
from zaber_motion.ascii import Connection
from asgard_alignment.ZaberMotor import BaldrPhaseMask, LAC10AT4A,  BifrostDichroic, SourceSelection



# ========================

# testing basis development 

# ========================
# timestamp
tstamp = datetime.datetime.now().strftime("%d-%m-%YT%H.%M.%S")


fig_path = f'tmp/basis_tests_{tstamp.split("T")[0]}/' #'/home/baldr/Documents/baldr/ANU_demo_scripts/BALDR/figures/' 
data_path = f'tmp/basis_tests_{tstamp.split("T")[0]}/' #'/home/baldr/Documents/baldr/ANU_demo_scripts/BALDR/data/' 


if not os.path.exists(fig_path):
   os.makedirs(fig_path)


# ZONAL PINNING 
b = np.eye(100) #util.construct_command_basis( basis='Zernike', number_of_modes = 20, Nx_act_DM = 12, Nx_act_basis = 12, act_offset=(0,0), without_piston=False)

bpin = np.array( [util.pin_outer_actuators_to_inner_diameter(bb) for bb in b.T] )
fig_path_tmp = fig_path + 'zonal_pinned/'
if not os.path.exists(fig_path_tmp):
   os.makedirs(fig_path_tmp)

for i in range(len(bpin)):

    util.nice_DM_plot( bpin[i] , savefig = fig_path_tmp + f'act{i}.png')
    #plt.figure(); plt.imshow( util.get_DM_command_in_2D( bpin[i] )); plt.savefig(fig_path + 'delme.png')
    _=input("next")


# FOURIER PINNING 
b = util.construct_command_basis( basis='fourier', number_of_modes = 20, Nx_act_DM = 10, Nx_act_basis = 10, act_offset=(0,0), without_piston=False)

number_of_modes = 20
Nx_act_DM = 10 
n = round( number_of_modes**0.5 ) + 1 # number of modes = (n-1)*(m-1) , n=m => (n-1)**2 
control_basis_dict  = util.develop_Fourier_basis( n, n ,P = 2 * Nx_act_DM, Nx = Nx_act_DM, Ny = Nx_act_DM )
        
# create raw basis as ordered list from our dictionary
raw_basis = []
for i in range( n-1 ):
    for j in np.arange( i , n-1 ):
        if i==j:
            raw_basis.append( control_basis_dict[i,i] )
        else:
            raw_basis.append( control_basis_dict[i,j] ) # get either side of diagonal 
            raw_basis.append( control_basis_dict[j,i] )
            

bpin = np.array( [util.pin_outer_actuators_to_inner_diameter(bb.reshape(-1)) for bb in np.array( raw_basis)] )


fig_path_tmp = fig_path + 'fourier_pinned/'
if not os.path.exists(fig_path_tmp):
   os.makedirs(fig_path_tmp)

for i in range(len(bpin)):

    util.nice_DM_plot( bpin[i] , savefig = fig_path_tmp + f'mode{i}.png' )#'delme.png') # f'act{i}.png')
    #plt.figure(); plt.imshow( util.get_DM_command_in_2D( bpin[i] )); plt.savefig(fig_path + 'delme.png')
    _=input("next")



# ZERNIKE PINNING 

nact_len = 12
b0 = util.construct_command_basis( basis='Zernike', number_of_modes = 20, Nx_act_DM = nact_len, Nx_act_basis = nact_len, act_offset=(0,0), without_piston=False)

#nanmask_filter = b0.T[0] == 0
#nanmask_filter[~nanmask_filter] = np.nan

# put values outside pupil to nan 
btmp = np.array( [util.get_DM_command_in_2D( bb ) for bb in b0.T])

# interpolate
nan_mask = btmp[0] #util.get_DM_command_in_2D( b0.T[0] != 0 )
nan_mask[nan_mask==0] = np.nan

#plt.figure(); plt.imshow(  nan_mask ); plt.savefig(fig_path + 'delme.png')

#plt.figure(); plt.imshow( util.get_DM_command_in_2D( nan_mask )); plt.savefig(fig_path + 'delme.png')

#nan_mask = np.isnan(nan_mask)
nearest_index = distance_transform_edt(np.isnan(nan_mask), return_distances=False, return_indices=True)

# Use the indices to replace NaNs with the nearest non-NaN values
with_corners = np.array( [ (nan_mask * bb)[tuple(nearest_index)] for bb in btmp[1:]] ).T
#filled_data = util.get_DM_command_in_2D( new_flat )[tuple(nearest_index)]


# Define the indices of the corners to be removed
corners = [(0, 0), (0, nact_len-1), (nact_len-1, 0), (nact_len-1, nact_len-1)]
# Convert 2D corner indices to 1D
corner_indices = [i * 12 + j for i, j in corners]

# Flatten the array
M2C = []
for w in with_corners.T:
    flattened_array = w.flatten()
    filtered_array = np.delete(flattened_array, corner_indices)

    M2C.append( filtered_array )

M2C = np.array( M2C ).T



fig_path_tmp = fig_path + 'zernike_pinned/'
if not os.path.exists(fig_path_tmp):
   os.makedirs(fig_path_tmp)

for i in range(len(M2C.T)):

    util.nice_DM_plot( M2C.T[i] , savefig = fig_path_tmp + f'mode{i}.png') # f'act{i}.png')
    #plt.figure(); plt.imshow( util.get_DM_command_in_2D( bpin[i] )); plt.savefig(fig_path + 'delme.png')
    _=input("next")




# Ok test in util 
zon_b = util.construct_command_basis( basis='Zonal_pinned_edges', number_of_modes = 20, Nx_act_DM = 12, Nx_act_basis = 12, act_offset=(0,0), without_piston=True)
zer_b = util.construct_command_basis( basis='Zernike_pinned_edges', number_of_modes = 20, Nx_act_DM = 12, Nx_act_basis = 12, act_offset=(0,0), without_piston=False)
fou_b = util.construct_command_basis( basis='fourier_pinned_edges', number_of_modes = 20, Nx_act_DM = 12, Nx_act_basis = 12, act_offset=(0,0), without_piston=True)

ref_b = util.construct_command_basis( basis='Zernike', number_of_modes = 20, Nx_act_DM = 12, Nx_act_basis = 12, act_offset=(0,0), without_piston=False)

# check shapes
print( ref_b.shape, fou_b.shape, zer_b.shape, zon_b.shape)
# check normalization 
print( np.sum( zon_b.T[0]**2 ), np.sum( zer_b.T[0]**2 ) , np.sum( fou_b.T[0]**2 ))
# plot them 
for i in range(len(M2C.T)):

    util.nice_DM_plot( zer_b.T[i] , savefig = fig_path + f'delme.png') # f'act{i}.png')
    #plt.figure(); plt.imshow( util.get_DM_command_in_2D( bpin[i] )); plt.savefig(fig_path + 'delme.png')
    _=input("next")


