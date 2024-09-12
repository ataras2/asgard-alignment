
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep  9 13:39:11 2024

@author: bencb
"""








import numpy as np
import glob 
from astropy.io import fits
import aotools
import time
import os 
import matplotlib.pyplot as plt 
import importlib
#import rtc
import sys
import datetime
import pandas as pd
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

class PIDController:
    def __init__(self, kp=None, ki=None, kd=None, upper_limit=None, lower_limit=None, setpoint=None):
        if kp is None:
            kp = np.zeros(1)
        if ki is None:
            ki = np.zeros(1)
        if kd is None:
            kd = np.zeros(1)
        if lower_limit is None:
            lower_limit = np.zeros(1)
        if upper_limit is None:
            upper_limit = np.ones(1)
        if setpoint is None:
            setpoint = np.zeros(1)

        self.kp = np.array(kp)
        self.ki = np.array(ki)
        self.kd = np.array(kd)
        self.lower_limit = np.array(lower_limit)
        self.upper_limit = np.array(upper_limit)
        self.setpoint = np.array(setpoint)

        size = len(self.kp)
        self.output = np.zeros(size)
        self.integrals = np.zeros(size)
        self.prev_errors = np.zeros(size)

    def process(self, measured):
        measured = np.array(measured)
        size = len(self.setpoint)

        if len(measured) != size:
            raise ValueError(f"Input vector size must match setpoint size: {size}")

        # Check all vectors have the same size
        error_message = []
        for attr_name in ['kp', 'ki', 'kd', 'lower_limit', 'upper_limit']:
            if len(getattr(self, attr_name)) != size:
                error_message.append(attr_name)
        
        if error_message:
            raise ValueError(f"Input vectors of incorrect size: {' '.join(error_message)}")

        if len(self.integrals) != size:
            print("Reinitializing integrals, prev_errors, and output to zero with correct size.")
            self.integrals = np.zeros(size)
            self.prev_errors = np.zeros(size)
            self.output = np.zeros(size)

        for i in range(size):
            error = measured[i] - self.setpoint[i]  # same as rtc
            self.integrals[i] += error
            self.integrals[i] = np.clip(self.integrals[i], self.lower_limit[i], self.upper_limit[i])

            derivative = error - self.prev_errors[i]
            self.output[i] = (self.kp[i] * error +
                              self.ki[i] * self.integrals[i] +
                              self.kd[i] * derivative)
            self.prev_errors[i] = error

        return self.output

    def reset(self):
        self.integrals.fill(0.0)
        self.prev_errors.fill(0.0)
        
        

class LeakyIntegrator:
    def __init__(self, rho=None, lower_limit=None, upper_limit=None, kp=None):
        # If no arguments are passed, initialize with default values
        if rho is None:
            self.rho = []
            self.lower_limit = []
            self.upper_limit = []
            self.kp = []
        else:
            if len(rho) == 0:
                raise ValueError("Rho vector cannot be empty.")
            if len(lower_limit) != len(rho) or len(upper_limit) != len(rho):
                raise ValueError("Lower and upper limit vectors must match rho vector size.")
            if kp is None or len(kp) != len(rho):
                raise ValueError("kp vector must be the same size as rho vector.")

            self.rho = np.array(rho)
            self.output = np.zeros(len(rho))
            self.lower_limit = np.array(lower_limit)
            self.upper_limit = np.array(upper_limit)
            self.kp = np.array(kp)  # kp is a vector now

    def process(self, input_vector):
        input_vector = np.array(input_vector)

        # Error checks
        if len(input_vector) != len(self.rho):
            raise ValueError("Input vector size must match rho vector size.")

        size = len(self.rho)
        error_message = ""

        if len(self.rho) != size:
            error_message += "rho "
        if len(self.lower_limit) != size:
            error_message += "lower_limit "
        if len(self.upper_limit) != size:
            error_message += "upper_limit "
        if len(self.kp) != size:
            error_message += "kp "

        if error_message:
            raise ValueError("Input vectors of incorrect size: " + error_message)

        if len(self.output) != size:
            print(f"output.size() != size.. reinitializing output to zero with correct size")
            self.output = np.zeros(size)

        # Process with the kp vector
        self.output = self.rho * self.output + self.kp * input_vector
        self.output = np.clip(self.output, self.lower_limit, self.upper_limit)

        return self.output

    def reset(self):
        self.output = np.zeros(len(self.rho))

        


trouble_shooting_dict = {
    #format:
    'short error key' :
    {
        'error string' : 'longer string descibing error',
        'fix': 'how to fix it '
    },
    'SerialPortBusyException' : 
    {
        'error string':"SerialPortBusyException: SerialPortBusyException: Cannot open serial port: Port is likely already opened by another application.",
        'fix':"You can check if any processes are using the serial port with the following command: lsof /dev/*name* (e.g. name=ttyUSB0).\nIf you found a process using the port from the previous step, you can terminate it with: sudo kill -9 <PID> "
    }

}


def print_current_state(full_report=False):
    if full_report:
        print(f'source motor: \n   {source_selection.device}')
        print(f'    -available sources: {source_selection.sources}')
        print(f'    -current position: {source_selection.current_position}')
        for d in dichroics:
            print(f'dichroic motor:\n   {d.device}')
            print(f'    -available dichroic positions: {d.dichroics}' )
            print(f'    -current position: {d.current_dichroic}')
        print('availabel phasemask positions: ', )
        print(f' phasemask motors: \n   {phasemask.motors}')
        print(f'    -available positions:')
        for l, p in phasemask.phase_positions.items():
            print(f'   {l, p}')
        print(f'    -current position: {phasemask.get_position()}um')
        print(f'focus motor:\n   {focus_motor}')
        print(f'    -current position: {focus_motor.get_position()}um')

    else:
        print(f'source motor {source_selection.device} current position: {source_selection.current_position}')
        for d in dichroics:
            print(f'dichroic motor {d.device} current position: {d.current_dichroic}')
        print(f'phasemask name: {phasemask_name}')
        print(f'phasemask motors {phasemask.motors["x"].axis} current position: {phasemask.get_position()[0]}um')
        print(f'phasemask motors {phasemask.motors["y"].axis} current position: {phasemask.get_position()[1]}um')
        print(f'focus motor {focus_motor.axis} current position: {focus_motor.get_position()}um')



def exit_all():
    # close things 
    try:
        con.close() #"192.168.1.111"
    except:
        print('no "con" to close')
    try:
        connection.close() # "/dev/ttyUSB0"
    except:
        print('no "connection" to close')
    try:
        zwfs.exit_dm() # DM 
    except:
        print('no DM to close')
    try:
        zwfs.exit_camera() #camera
    except:
        print('no camera to close')

        
# timestamp
tstamp = datetime.datetime.now().strftime("%d-%m-%YT%H.%M.%S")


# 
#iter1 : 

fig_path = f'tmp/{tstamp.split("T")[0]}/'

exper_path = f'build_IM_onsky_with_new_dmflat_{tstamp}/'

# setup paths 
if not os.path.exists(fig_path + exper_path ):
   os.makedirs(fig_path + exper_path )



"""
exper_path = 'reconstructors/'
if not os.path.exists(fig_path + exper_path):
   os.makedirs(fig_path+ exper_path)"""

# ====== hardware variables
beam = 3
phasemask_name = 'J3'
phasemask_OUT_offset = [1000, 1000] # relative offset (um) to take phasemask out of beam
BFO_pos = 3000 # um (absolute position of detector imgaging lens) 
dichroic_name = "J"
source_name = 'SBB'
DM_serial_number = '17DW019#122' # Syd = '17DW019#122', ANU = '17DW019#053'


# ======  set up source 

# start with source out !

# ======  set up dichroic 

# do manually (COM3 communication issue)

#  ConnectionFailedException: ConnectionFailedException: Cannot open serial port: no such file or directory

# run python playground/x_usb.py 
connection =  Connection.open_serial_port("/dev/ttyUSB0")
connection.enable_alerts()

device_list = connection.detect_devices()
print("Found {} devices".format(len(device_list)))

dichroics = []
source_selection = None
for dev in device_list:
    if dev.name == "X-LSM150A-SE03":
        dichroics.append(BifrostDichroic(dev))
    elif dev.name == "X-LHM100A-SE03":
        source_selection = SourceSelection(dev)
print(f"Found {len(dichroics)} dichroics")
if source_selection is not None:
    print("Found source selection")

for dichroic in dichroics:
    dichroic.set_dichroic("J")

while dichroics[0].get_dichroic() != "J":
    pass

# ====== set up phasemask
con = Connection.open_tcp("192.168.1.111")
print("Found {} devices".format(len(con.detect_devices())))
x_axis = con.get_device(1).get_axis(1)
y_axis = con.get_device(1).get_axis(3)

# get most recent positions file
maskpos_files = glob.glob( f"phase_positions_beam_{beam}*.json")
latest_maskpos_file = max(maskpos_files, key=os.path.getctime)
phasemask = BaldrPhaseMask(
    LAC10AT4A(x_axis), LAC10AT4A(y_axis), latest_maskpos_file 
)
""" 
# e.g: to update position and write to file 
phasemask.move_absolute( [3346, 1205])
phasemask.update_mask_position( 'J3' )
phasemask.write_current_mask_positions() 
"""

# ====== set up focus 
focus_axis = con.get_device(1).get_axis(2)
focus_motor = LAC10AT4A(focus_axis)
focus_motor.move_absolute( BFO_pos )

# print out motors we have 

print_current_state()
# ====== Set up and calibrate 

debug = True # plot some intermediate results 

# take out source to calibate 
source_selection.set_source(  'none' )
time.sleep(1)
focus_motor.move_absolute( BFO_pos )
time.sleep(1)
phasemask.move_to_mask(phasemask_name) 
time.sleep(1)
dichroic.set_dichroic("J")
time.sleep(1)


pupil_crop_region = [160,220, 110,185] # [204,268,125, 187] #[None, None, None, None] #[204 -50 ,268+50,125-50, 187+50] 

#init our ZWFS (object that interacts with camera and DM) (old path = home/baldr/Documents/baldr/ANU_demo_scripts/BALDR/)
zwfs = ZWFS.ZWFS(DM_serial_number=DM_serial_number, cameraIndex=0, DMshapes_path = 'DMShapes/', pupil_crop_region=pupil_crop_region ) 

# the sydney BMC multi-3.5 calibrated flat seems shit! Try with just a 

# calibrated new one. Set here 
new_flat_cmd = pd.read_csv( '/home/heimdallr/Documents/asgard-alignment/tmp/11-09-2024/optimize_ref_int_method_4/newflat_test_using_measN0_in_theory_3/calibrated_flat_phasemas-J3.csv' ) #'/home/heimdallr/Documents/asgard-alignment/tmp/08-09-2024/optimize_ref_int_method_3/newflat_test_3/calibrated_flat_phasemas-J3.csv').values[:,1]

#zwfs.dm_shapes['flat_dm_original'] = zwfs.dm_shapes['flat_dm'].copy() # keep the original BMC calibrated flat 
#zwfs.dm_shapes['flat_dm'] = new_flat_cmd  # update to ours! 

zwfs.deactive_cropping( ) # zwfs.set_camera_cropping(r1, r2, c1, c2 ) #<- use this for latency tests , set back after with zwfs.set_camera_cropping(0, 639, 0, 511 ) 
zwfs.set_camera_dit( 0.001 );time.sleep(0.2)
zwfs.set_camera_fps( 100 );time.sleep(0.2)
zwfs.set_sensitivity('high');time.sleep(0.2)
zwfs.enable_frame_tag(tag = True);time.sleep(0.2)
zwfs.bias_off();time.sleep(0.2)
zwfs.flat_off();time.sleep(0.2)

# trying different DM flat 
#zwfs.dm_shapes['flat_dm'] = 0.5 * np.ones(140)

zwfs.start_camera()

## ------- Calibrate detector (dark, badpixels)
# Source should be out
# at sydney move 01 X-LSM150A-SE03 to 133.07mm
zwfs.build_manual_dark()

# get our bad pixels 
bad_pixels = zwfs.get_bad_pixel_indicies( no_frames = 200, std_threshold = 25 , flatten=False) # std_threshold = 50

# update zwfs bad pixel mask and flattened pixel values 
zwfs.build_bad_pixel_mask( bad_pixels , set_bad_pixels_to = 0 )

## ------- move source back in 
source_selection.set_source(  source_name )
time.sleep(2)
"""
# quick check that dark subtraction works and we have signal
I0 = zwfs.get_image( apply_manual_reduction  = True)
plt.figure(); plt.title('test image \nwith dark subtraction \nand bad pixel mask'); plt.imshow( I0 ); plt.colorbar()
plt.savefig( fig_path +  'delme.png')
plt.show()
#plt.close()
"""
print_current_state()




zwfs.dm.send_data(zwfs.dm_shapes['flat_dm'] )
phasemask_centering_tool.move_relative_and_get_image(zwfs, phasemask, savefigName=fig_path + exper_path + 'delme.png')

# == init pupil region classification  
#init our pupil controller (object that processes ZWFS images and outputs VCM commands)
pupil_ctrl = pupil_control.pupil_controller_1(config_file = None)

#analyse pupil and decide if it is ok. This must be done before reconstructor
pupil_report = pupil_control.analyse_pupil_openloop( zwfs, debug = False, return_report = True, symmetric_pupil=False, std_below_med_threshold=1. )

if pupil_report['pupil_quality_flag'] == 1: 
    zwfs.update_reference_regions_in_img( pupil_report ) # 


# last minute check 
# zwfs.dm.send_data(zwfs.dm_shapes['flat_dm'] + 0.1 * zwfs.dm_shapes['four_torres'])
zwfs.dm.send_data(zwfs.dm_shapes['flat_dm'] )
phasemask_centering_tool.move_relative_and_get_image(zwfs, phasemask, savefigName=fig_path + 'delme.png')



# BUILD THE RECONSTRUCTOR HERE 
I0, N0 = util.get_reference_images(zwfs, phasemask, theta_degrees=11.8, number_of_frames=256, \
compass = True, compass_origin=None, savefig= fig_path + exper_path + f'FPM-in-out_{phasemask_name}.png' )




"""
# Build onsky IM 
"""

# get a series of off mask images 
phasemask.move_relative( [ 200, 0 ] )

N0_list = zwfs.get_some_frames(number_of_frames=1000, apply_manual_reduction=True)

phasemask.move_relative( [ -200, 0 ] )

basis_name = 'Zonal_pinned_edges'
modal_basis = util.construct_command_basis( basis=basis_name).T

flat_dm_cmd = zwfs.dm_shapes['flat_dm'].copy()
# TO POKE CMDS WHILE ROLLING PHASE SCREEN 
j=0 # to count when to poke next actuator
modal_basis = modal_basis  #np.eye(140) # actuator basis 
scrn_scaling_factor =  0.06 #1.2*( np.random.rand() - 0.5 )

kolmogorov_random = []


#random realizations of Kolmogorov screen 
# --- create infinite phasescreen from aotools module 
Nx_act = 12
corner_indicies = [0, Nx_act-1, Nx_act * (Nx_act-1), -1] 
screen_pixels = Nx_act*2**2 # some multiple of numer of actuators across DM 
D = 1.8 #m effective diameter of the telescope

scrn = aotools.infinitephasescreen.PhaseScreenVonKarman(nx_size=screen_pixels, pixel_scale=D/screen_pixels,r0=0.1,L0=12)

#print( modal_basis[140] )
j=0
number_of_I0 = 1000
swap_act_every = 100
total_iterations = 15000
poke_amp = 0.04
jump_per_it = 2
for i in range(total_iterations): # 1 per second, ~17 minutes for 1000...
    for _ in range(jump_per_it): # roll it more slowly 
        scrn.add_row()

    if j<len( modal_basis ):
        poke_cmd = (-1)**i * poke_amp * modal_basis[j]
    else:
        poke_cmd = np.zeros(140)
 
    if np.mod(i+1, swap_act_every)==0:                
        j+=1

    kol_cmd = util.create_phase_screen_cmd_for_DM(scrn,  scaling_factor=scrn_scaling_factor   , drop_indicies = corner_indicies, plot_cmd=False)
    # added flat_dm_cmd in !
    kolmogorov_random.append( flat_dm_cmd + kol_cmd + poke_cmd  )
    


descr_mode = f'LONG_rolling_kolmogorov_scaling-{round(scrn_scaling_factor ,3)}_pokeamp-{poke_amp}' #f'rolling_kolmogorov_scaling-{round(scrn_scaling_factor ,3)}' #'random_normal' #'zonal_pokes', 'zernike_pokes'

DM_command_sequence = [flat_dm_cmd for _ in range(number_of_I0)] + kolmogorov_random #list(modal_basis_1)

# to check
plt.figure();
plt.imshow( util.get_DM_command_in_2D( DM_command_sequence[number_of_I0 + 1] ));plt.colorbar();plt.savefig( fig_path + exper_path + 'delme.png'); i+=1

# --- additional labels to append to fits file to keep information about the sequence applied 
additional_labels = [('D',D),('jump_per_it',jump_per_it),('poke_amp',poke_amp),('no_flats',number_of_I0),("swap_act_every",swap_act_every), \
                     ('no_modes',len(modal_basis)),('basis_name',basis_name),('seq0','flatdm'), ('seq1',f'active_{swap_act_every}x{len(modal_basis)}'), \
                        ('seq2',f'passive_{len(DM_command_sequence)-swap_act_every*len(modal_basis) - number_of_I0}')] 

# --- poke DM in and out and record data. Extension 0 corresponds to images, extension 1 corresponds to DM commands
#raw_recon_data = apply_sequence_to_DM_and_record_images(zwfs, DM_command_sequence, number_images_recorded_per_cmd = number_images_recorded_per_cmd, take_median_of_images=True, save_dm_cmds = True, calibration_dict=None, additional_header_labels = additional_labels,sleeptime_between_commands=0.03, cropping_corners=None,  save_fits = None ) # None

skyIM_1 = util.apply_sequence_to_DM_and_record_images(zwfs, DM_command_sequence, number_images_recorded_per_cmd = 3, \
                                                take_mean_of_images=True, save_dm_cmds = True, calibration_dict=None,\
                                                      additional_header_labels=additional_labels, sleeptime_between_commands=0.02, cropping_corners=additional_labels, save_fits = fig_path + exper_path +f'open_loop_ONSKY_IM_data_{descr_mode}_{phasemask_name}_{tstamp}.fits')

# ========== ANALYSIS to append to fits 

#data_path = fig_path + exper_path +f'open_loop_ONSKY_IM_data_{descr_mode}_{phasemask_name}_{tstamp}.fits' #'tmp/10-09-2024/onSky_IM/' #'/Users/bencb/Documents/baldr/lab_data/OPEN_LOOP_RUNS/'

#files = glob.glob( data_path+'*.fits')
file_name = fig_path + exper_path +f'open_loop_ONSKY_IM_data_{descr_mode}_{phasemask_name}_{tstamp}.fits' 

#skyIM_1 = fits.open( file_name ) #'/Users/bencb/Documents/baldr/lab_data/OPEN_LOOP_RUNS/open_loop_data_rolling_kolmogorov_scaling-0.17_16-07-2024T20.56.28.fits' )

        
cmd2opd  = 2800        
basis = skyIM_1[0].header['basis_name']
modal_basis = util.construct_command_basis( basis=basis_name).T

poke_amp = skyIM_1[0].header['poke_amp']
no_flats = skyIM_1[0].header['no_flats']
act_no = skyIM_1[0].header['no_modes']
iter_per_act = skyIM_1[0].header['swap_act_every']
flat_dm = skyIM_1['DM_CMD_SEQUENCE'].data[0]

I0_list = skyIM_1['SEQUENCE_IMGS'].data[:no_flats]

dm_push = []
dm_pull = [] 
img_push = []
img_pull = []

for i in range(no_flats, iter_per_act*act_no + no_flats):
    if np.mod(i,2)==0:
        dm_push.append( skyIM_1['DM_CMD_SEQUENCE'].data[i] )
        img_push.append(  skyIM_1['SEQUENCE_IMGS'].data[i] )
    else :
        dm_pull.append( skyIM_1['DM_CMD_SEQUENCE'].data[i] )
        img_pull.append(  skyIM_1['SEQUENCE_IMGS'].data[i] )


# visual check at beggining and end to make sure mapped correctly 
print(f'lengths should equal {iter_per_act * act_no / 2}. They are = ' ,len( dm_push ), len( dm_pull ) ) 

plt.figure(1) ; plt.imshow( util.get_DM_command_in_2D( dm_pull[0] - dm_push[0] )); plt.savefig( fig_path + exper_path +'delme.png')

plt.figure(2) ; plt.imshow( util.get_DM_command_in_2D( dm_pull[-1] - dm_push[-1] )); plt.savefig( fig_path + exper_path +'delme.png')



dm_push_avg = []
dm_pull_avg = []
img_push_avg =[]
img_pull_avg = []
for act in range( act_no ): 
    dm_push_avg.append(  np.mean( dm_push[iter_per_act * act // 2 :  iter_per_act * (act+1) // 2] , axis=0 )  )
    dm_pull_avg.append( np.mean( dm_pull[iter_per_act * act // 2 :  iter_per_act * (act+1) // 2] , axis=0 )  ) 

    img_push_avg.append( np.mean( img_push[iter_per_act * act // 2 :  iter_per_act * (act+1) // 2] , axis=(0,1) ) )
    img_pull_avg.append( np.mean( img_pull[iter_per_act * act // 2 :  iter_per_act * (act+1) // 2] , axis=(0,1) ) )
    
plt.figure() ; plt.imshow( util.get_DM_command_in_2D( dm_pull_avg[-1] - dm_push_avg[-1] ) ); plt.savefig( fig_path + exper_path +'delme.png')


# building IM 
#i=64
#plt.imshow( img_push_avg[i] - img_pull_avg[i] );plt.colorbar(); i=i+1  # check it works!!! 

IM = []
for act in range( act_no ): 
    IM.append( (img_push_avg[act].reshape(-1)/np.mean(img_push_avg[act] ) - img_pull_avg[act].reshape(-1)/np.mean(img_push_avg[act] ) ) /2 )

IM=np.array(IM) 

# Create a list of HDUs (Header Data Units)
#hdul = fits.HDUList()

lists_dict = {
    "basis":modal_basis,
    "IM": IM,
    "I0":I0_list,
    "N0": N0,
    "N0_list":N0_list,
    "pupil_pixels":zwfs.pupil_pixels,
    "dm_push":dm_push,
    "dm_pull":dm_pull,
    "img_push": img_push,
    "img_pull": img_push,
    "poke_amp":[poke_amp]
}


# Add each list to the HDU list as a new extension
for list_name, data_list in lists_dict.items():
    # Convert list to numpy array for FITS compatibility
    data_array = np.array(data_list, dtype=float)  # Ensure it is a float array or any appropriate type

    # Create a new ImageHDU with the data
    hdu = fits.ImageHDU(data_array)

    # Set the EXTNAME header to the variable name
    hdu.header['EXTNAME'] = list_name

    # Append the HDU to the HDU list
    skyIM_1.append(hdu)

# Write the HDU list to a FITS file
name = file_name.split( '/')[-1].split('.fits')[0] 
current_path = file_name.split( name )[0]
skyIM_1.writeto(current_path + f'onsky_IM_processed_{name}.fits', overwrite=True)



full_img_avg = []
cropped_img_avg = [] 

for _ in range(1000):
    time.sleep(0.1)
    full_img_avg.append( zwfs.get_image_in_another_region() )
    cropped_img_avg.append( zwfs.get_image(apply_manual_reduction=False).mean() )












#%%% 

## creating gifs 

"""
## WITHOUT ACTIVE PUSHES 
#fig_path =  '/Users/bencb/Documents/baldr/data_sydney/10-09-2024/onSky_IM/DM_IMG_GIF_PASSIVE/' #'/Users/bencb/Downloads/'

if not os.path.exists(fig_path ):
   os.makedirs(fig_path )

# plot camera data to check DM flushing 
act_no = 140
i0 = 1 + iter_per_act * act_no
i1 = 1 + iter_per_act * act_no + 120
I0 = skyIM_1['SEQUENCE_IMGS'].data[0,0,:,:] 
current_act = act_no

cmd2opd  = 2800
for i in range( i0 , i1 ): # looking at images 

    if i > 1 + iter_per_act * (act_no + 1) :
        current_act += 1
    
    plt.figure()
    sky_im = np.flipud( skyIM_1['SEQUENCE_IMGS'].data[i,0,:,:]  - I0   )[13:-13,20:-20]
    
    dm_img = cmd2opd * skyIM_1['DM_CMD_SEQUENCE'].data[i]
    #dm_img[current_act] = np.nan # dm_img[current_act-1] # just nearest interp 
    
    im_list = [util.get_DM_command_in_2D( dm_img ), sky_im ]
    xlabel_list = [None, None, None]
    ylabel_list = [None, None, None]
    vlims = [[-160,160],[-200,200]]
    title_list = ['input phase', 'ZWFS signal']
    cbar_label_list = [r'OPD [nm]',r'normalized intensity [ADU]'] 
    savefig = fig_path + f'build_IM_on_sky_img_{i}.png' #f'mode_reconstruction_images/phase_reconstruction_example_mode-{mode_indx}_basis-{phase_ctrl.config["basis"]}_ctrl_modes-{phase_ctrl.config["number_of_controlled_modes"]}ctrl_act_diam-{phase_ctrl.config["dm_control_diameter"]}_readout_mode-12x12.png'
    
    util.nice_heatmap_subplots( im_list , xlabel_list, ylabel_list, title_list, cbar_label_list, vlims=vlims, \
                               fontsize=15, axis_off=True, cbar_orientation = 'bottom', savefig=savefig)
        
    
        
        
        
    
## WITH ACTIVE PUSHES 
fig_path =  '/Users/bencb/Documents/baldr/data_sydney/10-09-2024/onSky_IM/DM_IMG_GIF_ACTIVE/' #'/Users/bencb/Downloads/'

if not os.path.exists(fig_path ):
    os.makedirs(fig_path )

# plot camera data to check DM flushing 
act_no = 52
i0 = 1 + iter_per_act * act_no + 40
i1 = i0 + 120
I0 = skyIM_1['SEQUENCE_IMGS'].data[0,0,:,:] 
current_act = act_no

cmd2opd  = 2800
for i in range( i0 , i1 ): # looking at images 

    if np.mod( i , 1 + iter_per_act * (act_no + 1) ) == 0 :
        current_act += 1
    
    
    plt.figure()
    sky_im = np.flipud( skyIM_1['SEQUENCE_IMGS'].data[i,0,:,:]  - I0   )[13:-13,20:-20]
    
    dm_img = cmd2opd * skyIM_1['DM_CMD_SEQUENCE'].data[i]
    dm_img[current_act] = np.nan #dm_img[current_act-1] # just nearest interp 
    
    im_list = [util.get_DM_command_in_2D( dm_img ), sky_im ]
    xlabel_list = [None, None, None]
    ylabel_list = [None, None, None]
    vlims = [[-160,160],[-200,200]]
    title_list = ['input phase', 'ZWFS signal']
    cbar_label_list = [r'OPD [nm]',r'normalized intensity [ADU]'] 
    savefig = fig_path + f'build_IM_on_sky_img_{i}.png' #f'mode_reconstruction_images/phase_reconstruction_example_mode-{mode_indx}_basis-{phase_ctrl.config["basis"]}_ctrl_modes-{phase_ctrl.config["number_of_controlled_modes"]}ctrl_act_diam-{phase_ctrl.config["dm_control_diameter"]}_readout_mode-12x12.png'
    
    util.nice_heatmap_subplots( im_list , xlabel_list, ylabel_list, title_list, cbar_label_list, vlims=vlims, \
                               fontsize=15, axis_off=True, cbar_orientation = 'bottom', savefig=savefig)
        
"""