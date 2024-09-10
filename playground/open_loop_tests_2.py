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


fig_path = f'tmp/{tstamp.split("T")[0]}/' #'/home/baldr/Documents/baldr/ANU_demo_scripts/BALDR/figures/' 
data_path = f'tmp/{tstamp.split("T")[0]}/' #'/home/baldr/Documents/baldr/ANU_demo_scripts/BALDR/data/' 


if not os.path.exists(fig_path):
   os.makedirs(fig_path)
"""
exper_path = 'reconstructors/'
if not os.path.exists(fig_path + exper_path):
   os.makedirs(fig_path+ exper_path)"""

# ====== hardware variables
beam = 3
phasemask_name = 'J1'
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
new_flat_cmd = pd.read_csv( '/home/heimdallr/Documents/asgard-alignment/tmp/08-09-2024/optimize_ref_int_method_3/newflat_test_3/calibrated_flat_phasemas-J3.csv').values[:,1]

#zwfs.dm_shapes['flat_dm_original'] = zwfs.dm_shapes['flat_dm'].copy() # keep the original BMC calibrated flat 
#zwfs.dm_shapes['flat_dm'] = new_flat_cmd  # update to ours! 

zwfs.deactive_cropping() # zwfs.set_camera_cropping(r1, r2, c1, c2 ) #<- use this for latency tests , set back after with zwfs.set_camera_cropping(0, 639, 0, 511 ) 
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








# 
#itera = 2 # first try here J1 
#itera = 3 #$ \\ moving to J3 and using MAP 
#itera = 4 # by part 2 IM was clearly out for iter 3  - retaking 
#itera = 5 # still shit, trying single sided IM 
#itera = 6 #playing with signs 
#itera = 7 # kinda working on Kolmogorov + closed loop 
itera = 8 # retaking IM to go more closed loop dynamic 

# timestamp
tstamp = datetime.datetime.now().strftime("%d-%m-%YT%H.%M.%S")

fig_path = f'tmp/{tstamp.split("T")[0]}/'

exper_path = f'open_loop_{itera}/'

# setup paths 
if not os.path.exists(fig_path + exper_path ):
   os.makedirs(fig_path + exper_path )

"""if not os.path.exists(f'tmp/{tstamp.split("T")[0]}/'):
   os.makedirs(f'tmp/{tstamp.split("T")[0]}/')


if not os.path.exists(fig_path + exper_path):
   os.makedirs(fig_path + exper_path)
"""

"""# reco file is with cal dm flat, reco file 2 with bmc cal 
reco_file = 'tmp/09-09-2024/iter_4_J3/zonal_reconstructor/RECONSTRUCTORS_zonal_0.07pokeamp_in-out_pokes_map_DIT-0.001_gain_high_09-09-2024T08.28.05.fits'
#"/home/heimdallr/Documents/asgard-alignment/tmp/09-09-2024/iter_3_J3/fourier_90modes_pinv_reconstructor/RECONSTRUCTORS_fourier90_0.2pokeamp_in-out_pokes_pinv_DIT-0.001_gain_high_09-09-2024T07.38.23.fits"
#"/home/heimdallr/Documents/asgard-alignment/tmp/09-09-2024/iter_4_J3/fourier_90modes_pinv_reconstructor/RECONSTRUCTORS_fourier90_0.2pokeamp_in-out_pokes_pinv_DIT-0.001_gain_high_09-09-2024T08.30.27.fits"
#"/home/heimdallr/Documents/asgard-alignment/tmp/09-09-2024/iter_3_J3/fourier_90modes_pinv_reconstructor/RECONSTRUCTORS_fourier90_0.2pokeamp_in-out_pokes_pinv_DIT-0.001_gain_high_09-09-2024T07.38.23.fits"
#"/home/heimdallr/Documents/asgard-alignment/tmp/09-09-2024/iter_3_J3/zonal_reconstructor/RECONSTRUCTORS_zonal_0.07pokeamp_in-out_pokes_map_DIT-0.001_gain_high_09-09-2024T07.35.58.fits" #'/home/heimdallr/Documents/asgard-alignment/tmp/08-09-2024/iter_1_J3/zonal_reconstructor/RECONSTRUCTORS_zonal_0.07pokeamp_in-out_pokes_map_DIT-0.001_gain_high_08-09-2024T18.17.34.fits'
#reco_file2 = '/home/heimdallr/Documents/asgard-alignment/tmp/08-09-2024/iter_3_J3/zonal_reconstructor/RECONSTRUCTORS_zonal_0.07pokeamp_in-out_pokes_map_DIT-0.001_gain_high_08-09-2024T20.54.08.fits'
ff = fits.open(reco_file) 
#ff2 = fits.open( reco_file2)

# _,S1, _ = np.linalg.svd( ff['IM'].data)
# _,S2, _ = np.linalg.svd( ff2['IM'].data)

# plt.figure(); plt.plot( S1 ); plt.plot( S2, label='S2'); plt.savefig(fig_path + 'delme.png')

# poke_amp = ff['INFO'].header['poke_amplitude']    

"""

zwfs.dm.send_data(zwfs.dm_shapes['flat_dm'] )
phasemask_centering_tool.move_relative_and_get_image(zwfs, phasemask, savefigName=fig_path + 'delme.png')

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


modal_basis = util.construct_command_basis('Zonal_pinned_edges')


####
# NOTE HERE WE BUILD WITH REAL REGISTRATION 
###
poke_amp = 0.06
IM_list = []
method='double_sided'

if method=='double_sided':

    for i,m in enumerate(modal_basis.T):
        print(f'executing cmd {i}/{len(modal_basis)}')
        I_plus_list = []
        I_minus_list = []
        imgs_to_mean = 10
        for sign in [(-1)**n for n in range(10)]: #[-1,1]:
            zwfs.dm.send_data( list( zwfs.dm_shapes['flat_dm'] + sign * poke_amp/2 * m )  )
            time.sleep(0.02)
            if sign > 0:
                I_plus_list += zwfs.get_some_frames(number_of_frames = imgs_to_mean, apply_manual_reduction = True )
                #I_plus *= 1/np.mean( I_plus )
            if sign < 0:
                I_minus_list += zwfs.get_some_frames(number_of_frames = imgs_to_mean, apply_manual_reduction = True )
                #I_minus *= 1/np.mean( I_minus )

        I_plus = np.mean( I_plus_list, axis = 0).reshape(-1)  # flatten so can filter with pupil_pixels
        I_plus *= 1/np.mean( I_plus )

        I_minus = np.mean( I_minus_list, axis = 0).reshape(-1)  # flatten so can filter with pupil_pixels
        I_minus *= 1/np.mean( I_minus )

        errsig = (I_plus - I_minus)[np.array( zwfs.pupil_pixels )]
        IM_list.append( list(  errsig.reshape(-1) ) ) #toook out 1/poke_amp *
elif method=='single_sided': 
    for i,m in enumerate(modal_basis.T):
        print(f'executing cmd {i}/{len(modal_basis)}')
        I_plus_list = []
        I_minus_list = []
        imgs_to_mean = 10
        for sign in [(-1)**n > 0 for n in range(10)]: #[-1,1]:
            # go between poke and flat DM 
            zwfs.dm.send_data( list( zwfs.dm_shapes['flat_dm'] + sign * poke_amp * m )  )
            time.sleep(0.02)
            if sign == False: # measured I0
                I_plus_list += zwfs.get_some_frames(number_of_frames = imgs_to_mean, apply_manual_reduction = True )
                #I_plus *= 1/np.mean( I_plus )
            if sign == True:
                I_minus_list += zwfs.get_some_frames(number_of_frames = imgs_to_mean, apply_manual_reduction = True )
                #I_minus *= 1/np.mean( I_minus )

        I_plus = np.mean( I_plus_list, axis = 0).reshape(-1)  # flatten so can filter with pupil_pixels
        I_plus *= 1/np.mean( I_plus )

        I_minus = np.mean( I_minus_list, axis = 0).reshape(-1)  # flatten so can filter with pupil_pixels
        I_minus *= 1/np.mean( I_minus )

        errsig = (I_plus - I_minus)[np.array( zwfs.pupil_pixels )]
        IM_list.append( list(  errsig.reshape(-1) ) ) #toook out 1/poke_amp *


IM= np.array( IM_list ).T # 1/poke_amp * 

zwfs.dm.send_data(zwfs.dm_shapes['flat_dm'] )

M2C_0 = modal_basis.T


bb = util.construct_command_basis('Zernike' ,without_piston=False)[:,0]

dm_pupil_filt =  bb > 0  # np.std((M2C.T @ IM.T) ,axis=1) > 4

def plot_DM( cmd , save_path = fig_path + 'delme.png'):

    plt.figure(); plt.imshow( util.get_DM_command_in_2D( cmd  ) ); plt.colorbar() ; plt.savefig(save_path)

#plot_DM( dm_pupil_filt ) 


## WE CAN DO ALL THIS WITH DIFFERENT PUPIL REGISTRATIONS!!! 
pupil_pixel_shift=0
pupil_pixel_filter = np.roll(zwfs.pupil_pixel_filter, shift=pupil_pixel_shift, axis=0)
pupil_pixels = np.where( pupil_pixel_filter )[0]  #pupil_pixels.copy() 

# 16 is measured from data , 1100 rough central wavelength of measurement 
cmd2opd = 16 * 1100 / (2*np.pi) #3200 # to go to cmd space to nm OPD 


# ----- 0) reconstruction of flat field (statistics of reconstructor)


current_path = fig_path + exper_path +  "0_reco_on_reference/" 
if not os.path.exists(current_path ):
   os.makedirs(current_path)


zwfs.dm.send_data(zwfs.dm_shapes['flat_dm'] )


disturb_list = [ np.zeros(140) ]

Smax_grid = [2, 10, 30, 50, 70, 80, 130] 

residual_list = {s:[] for s in Smax_grid}
rmse_list = {s:[] for s in Smax_grid}
c_list = {s:[] for s in Smax_grid}
sig_list = {s:[] for s in Smax_grid}
R_list = {s:[] for s in Smax_grid}

for Smax in Smax_grid: 
    print('running for Smax = ', Smax)
    
    U, S, Vt = np.linalg.svd( IM, full_matrices=False)

    R  = (Vt.T * [1/ss if i < Smax else 0 for i,ss in enumerate(S)])  @ U.T

    for _ in range(500):
        i = np.mean( zwfs.get_some_frames(number_of_frames=2, apply_manual_reduction=True),axis=0) #z.detection_chain( test_field, FPM_on=True, include_shotnoise=True, ph_per_s_per_m2_per_nm=True, grids_aligned=True, replace_nan_with=0 )
        #o = #z.detection_chain( test_field, FPM_on=False, include_shotnoise=True, ph_per_s_per_m2_per_nm=True, grids_aligned=True, replace_nan_with=0 )
        
        sig = i / np.mean( i ) -  I0 / np.mean( I0 ) # I0_theory/ np.mean(I0_theory) #

        c = poke_amp * M2C_0.T  @ R @ sig.reshape(-1)[pupil_pixels]

        residual =  (disturbance_cmd - c)[dm_pupil_filt]

        rmse = np.nanstd( residual )    
        rmse_list[Smax].append( rmse ) 
        
        residual_list[Smax].append( disturbance_cmd - c)
        
        c_list[Smax].append( c )
        
        sig_list[Smax].append( sig )

    R_list[Smax].append( R )
    

    plt.figure(figsize=(8,5))
    plt.plot( cmd2opd * np.array( c_list[Smax] ).T ,color='grey', alpha=0.3)
    plt.plot( cmd2opd * np.array( np.mean(c_list[Smax],axis=0)), color='g',label='mean')
    plt.plot( cmd2opd * np.array( np.mean(c_list[Smax],axis=0) + np.std( np.mean(c_list[Smax],axis=0) )), color='r',label=r'$\sigma$')
    plt.plot( cmd2opd * np.array( np.mean(c_list[Smax],axis=0) - np.std( np.mean(c_list[Smax],axis=0) )), color='r')
    plt.gca().tick_params(labelsize=15)
    plt.xlabel('DM actuator',fontsize=15)
    plt.ylabel('reconstructed amplitude [nm]',fontsize=15)
    plt.savefig( current_path + f'reconstructed_cmd_stat_on_reference.png',bbox_inches='tight',dpi=200)
    

    im_list = [ util.get_DM_command_in_2D( cmd2opd * np.array( np.mean(c_list[Smax], axis=0)) ), \
                util.get_DM_command_in_2D( cmd2opd * np.array( np.std(c_list[Smax], axis=0)) )]
                
    xlabel_list = ['' for _ in range(len(im_list))]
    ylabel_list = ['' for _ in range(len(im_list))]
    vlims = [[np.nanmin(iii), np.nanmax(iii)] for iii in im_list]
    title_list = ['bias per actuator\nin reconstructor', 'noise per actuator\nin reconstructor']
    cbar_label_list = [ r'$\mu$ [nm]', r'$\sigma$ [nm]']
    util.nice_heatmap_subplots(im_list , xlabel_list, ylabel_list, title_list,cbar_label_list, fontsize=15, vlims=vlims, \
                            cbar_orientation = 'bottom', axis_off=True, savefig= current_path + f'reconstructed_cmd_stat_on_DM_svdTruc-{Smax}_mask{phasemask_name}.png')



    write_dict = {
    "disturbance" : disturb_list,
    "signal" : sig_list[Smax],
    "svd_trucation_index":[Smax],
    "reconstructor" :R_list[Smax] ,
    "reco_cmd":c_list[Smax] ,
    "cmd_rmse":rmse_list[Smax] ,
    "cmd_residual":residual_list[Smax] ,
    "IM":IM   
    }

    # Create a list of HDUs (Header Data Units)
    hdul = fits.HDUList()

    # Add each list to the HDU list as a new extension
    for list_name, data_list in write_dict.items():
        # Convert list to numpy array for FITS compatibility
        data_array = np.array(data_list, dtype=float)  # Ensure it is a float array or any appropriate type

        # Create a new ImageHDU with the data
        hdu = fits.ImageHDU(data_array)

        # Set the EXTNAME header to the variable name
        hdu.header['EXTNAME'] = list_name

        # Append the HDU to the HDU list
        hdul.append(hdu)

    # Write the HDU list to a FITS file
    hdul.writeto( current_path + f'pt0-reco_I0_signal_telemetry_svd_truc{Smax}_{phasemask_name}_{tstamp}_round2.fits', overwrite=True)


plt.figure(figsize=(8,5))
plt.plot( Smax_grid, 2*np.pi/1050 * cmd2opd * np.array( [np.mean(rmse_list[s]) for s in rmse_list] )  )
#plt.axhline( np.var(test_field.phase[z.wvls[0]][z.pup>0]) , label='initial')
plt.legend()
plt.xlabel('Singular value truncation index',fontsize=15)
plt.ylabel('Reconstructed RMSE [rad]',fontsize=15) 
plt.gca().tick_params(labelsize=15)   
plt.savefig(current_path  +f'rmse_vs_svd_trunction_measured_onReference_I0_{phasemask_name}.png', bbox_inches='tight', dpi=200)



# ---- 0.1  reconstruction of flat field (statistics of reconstructor) with MAP 

current_path = fig_path + exper_path +  "0.1_reco_on_reference_with_MAP/" 
if not os.path.exists(current_path ):
   os.makedirs(current_path)


zwfs.dm.send_data(zwfs.dm_shapes['flat_dm'] )


disturb_list = [ np.zeros(140) ]

Smax_grid = [2, 10, 30, 50, 70, 80, 130] 

residual_list = {s:[] for s in Smax_grid}
rmse_list = {s:[] for s in Smax_grid}
c_list = {s:[] for s in Smax_grid}
sig_list = {s:[] for s in Smax_grid}
R_list = {s:[] for s in Smax_grid}

# detector noise covariance matrix
noise_cov = zwfs.estimate_noise_covariance( number_of_frames = 1000, where = 'pupil' )

im_list = [ noise_cov ]            
xlabel_list = ['registered\npupil pixels' for _ in range(len(im_list))]
ylabel_list = ['registered \npupil pixels' for _ in range(len(im_list))]
vlims = [[np.nanmin(iii), np.nanmax(iii)] for iii in im_list]
title_list = ['reference pupil covariance']
cbar_label_list = [ r'$\sigma^2$' ]
util.nice_heatmap_subplots(im_list , xlabel_list, ylabel_list, title_list,cbar_label_list, fontsize=15, vlims=vlims, \
                        cbar_orientation = 'right', axis_off=False, savefig= current_path + f'covariance_of_registered_pixels_mask{phasemask_name}.png')

# no input phase aberrations 
phase_cov = np.eye( np.array(IM).shape[1] )

for Smax in Smax_grid: 
    print('running for Smax = ', Smax)
    

    R_map = (phase_cov @ IM.T @ np.linalg.pinv(IM @ phase_cov.T @ IM.T + noise_cov) )

    U, S, Vt = np.linalg.svd( R_map, full_matrices=False)

    R = (U * np.array([ss if i < Smax else 0 for i,ss in enumerate(S)]))  @ Vt

    for _ in range(500):
        i = np.mean( zwfs.get_some_frames(number_of_frames=2, apply_manual_reduction=True),axis=0) #z.detection_chain( test_field, FPM_on=True, include_shotnoise=True, ph_per_s_per_m2_per_nm=True, grids_aligned=True, replace_nan_with=0 )
        #o = #z.detection_chain( test_field, FPM_on=False, include_shotnoise=True, ph_per_s_per_m2_per_nm=True, grids_aligned=True, replace_nan_with=0 )
        
        sig = i / np.mean( i ) -  I0 / np.mean( I0 ) # I0_theory/ np.mean(I0_theory) #

        c = poke_amp * M2C_0.T  @ R @ sig.reshape(-1)[pupil_pixels]

        residual =  (disturbance_cmd - c)[dm_pupil_filt]

        rmse = np.nanstd( residual )    
        rmse_list[Smax].append( rmse ) 
        
        residual_list[Smax].append( disturbance_cmd - c)
        
        c_list[Smax].append( c )
        
        sig_list[Smax].append( sig )

    R_list[Smax].append( R )
    

    plt.figure(figsize=(8,5))
    plt.plot( cmd2opd * np.array( c_list[Smax] ).T ,color='grey', alpha=0.3)
    plt.plot( cmd2opd * np.array( np.mean(c_list[Smax],axis=0)), color='g',label='mean')
    plt.plot( cmd2opd * np.array( np.mean(c_list[Smax],axis=0) + np.std( np.mean(c_list[Smax],axis=0) )), color='r',label=r'$\sigma$')
    plt.plot( cmd2opd * np.array( np.mean(c_list[Smax],axis=0) - np.std( np.mean(c_list[Smax],axis=0) )), color='r')
    plt.gca().tick_params(labelsize=15)
    plt.xlabel('DM actuator',fontsize=15)
    plt.ylabel('reconstructed amplitude [nm]',fontsize=15)
    plt.savefig( current_path + f'MAP_reconstructed_cmd_stat_on_reference.png',bbox_inches='tight',dpi=200)
    

    im_list = [ util.get_DM_command_in_2D( cmd2opd * np.array( np.mean(c_list[Smax], axis=0)) ), \
                util.get_DM_command_in_2D( cmd2opd * np.array( np.std(c_list[Smax], axis=0)) )]
                
    xlabel_list = ['' for _ in range(len(im_list))]
    ylabel_list = ['' for _ in range(len(im_list))]
    vlims = [[np.nanmin(iii), np.nanmax(iii)] for iii in im_list]
    title_list = ['bias per actuator\nin reconstructor', 'noise per actuator\nin reconstructor']
    cbar_label_list = [ r'$\mu$ [nm]', r'$\sigma$ [nm]']
    util.nice_heatmap_subplots(im_list , xlabel_list, ylabel_list, title_list,cbar_label_list, fontsize=15, vlims=vlims, \
                            cbar_orientation = 'bottom', axis_off=True, savefig= current_path + f'reconstructedMAP_cmd_stat_on_DM_svdTruc-{Smax}_mask{phasemask_name}.png')



    write_dict = {
    "disturbance" : disturb_list,
    "signal" : sig_list[Smax],
    "svd_trucation_index":[Smax],
    "reconstructor" :R_list[Smax] ,
    "reco_cmd":c_list[Smax] ,
    "cmd_rmse":rmse_list[Smax] ,
    "cmd_residual":residual_list[Smax]  ,
    "IM":IM
    }

    # Create a list of HDUs (Header Data Units)
    hdul = fits.HDUList()

    # Add each list to the HDU list as a new extension
    for list_name, data_list in write_dict.items():
        # Convert list to numpy array for FITS compatibility
        data_array = np.array(data_list, dtype=float)  # Ensure it is a float array or any appropriate type

        # Create a new ImageHDU with the data
        hdu = fits.ImageHDU(data_array)

        # Set the EXTNAME header to the variable name
        hdu.header['EXTNAME'] = list_name

        # Append the HDU to the HDU list
        hdul.append(hdu)

    # Write the HDU list to a FITS file
    hdul.writeto( current_path + f'pt0-MAPreco_I0_signal_telemetry_svd_truc{Smax}_{phasemask_name}_{tstamp}_round2.fits', overwrite=True)


plt.figure(figsize=(8,5))
plt.plot( Smax_grid, 2*np.pi/1050 * cmd2opd * np.array( [np.mean(rmse_list[s]) for s in rmse_list] )  )
#plt.axhline( np.var(test_field.phase[z.wvls[0]][z.pup>0]) , label='initial')
plt.legend()
plt.xlabel('Singular value truncation index',fontsize=15)
plt.ylabel('Reconstructed RMSE [rad]',fontsize=15) 
plt.gca().tick_params(labelsize=15)   
plt.savefig(current_path  +f'rmse_vs_MAP_svd_trunction_measured_onReference_I0_{phasemask_name}.png', bbox_inches='tight', dpi=200)




# ---- 1) reconstruct previous IM signal 
# use MAP reconstructor 

current_path = fig_path + exper_path +  "1_reco_IM_signal/" 
if not os.path.exists(current_path ):
   os.makedirs(current_path)

zwfs.dm.send_data(zwfs.dm_shapes['flat_dm'] )

Smax_grid = np.arange( 2,100,3)

#m=65 # mode to put on
for m in [43,55,65,75]:
    # put on IM signal with a bit of noise (1/2 std )
    noise_sigma = 0.5 * np.std( IM )

    bias = np.mean( IM )

    sig =  IM.T[m] + noise_sigma * np.random.rand( IM.T[m].shape[0] )  + bias
    tmp = np.zeros( I0.shape )
    tmp.reshape(-1)[pupil_pixels] = sig
    sig_list = [tmp] 

    disturbance_cmd  =  poke_amp * M2C_0[m]    
    disturb_list = [disturbance_cmd]

    rmse_list =[]
    c_list =[]
    R_list = []
    residual_list = [] 
    for Smax in Smax_grid: 
        
        #U, S, Vt = np.linalg.svd( IM, full_matrices=False)

        #R  = (Vt.T * [1/ss if i < Smax else 0 for i,ss in enumerate(S)])  @ U.T
        
        R_map = (phase_cov @ IM.T @ np.linalg.pinv(IM @ phase_cov.T @ IM.T + noise_cov) )

        U, S, Vt = np.linalg.svd( R_map, full_matrices=False)

        R = (U * np.array([ss if i < Smax else 0 for i,ss in enumerate(S)]))  @ Vt

        c = poke_amp * M2C_0.T @ R @ sig.reshape(-1) #dont need pupil filter as its inbuilt in IM 
        
        residual =  (disturbance_cmd - c)

        rmse = np.nanstd( residual[dm_pupil_filt] )
        
        residual_list.append( residual )    
        rmse_list.append( rmse ) 
        c_list.append( c )
        R_list.append( R )
        
    best_i = np.argmin( rmse_list )

    im_list = [ np.flipud(sig_list[0]), cmd2opd * util.get_DM_command_in_2D( disturbance_cmd ) , cmd2opd * util.get_DM_command_in_2D( c_list[best_i] ) , \
                cmd2opd * util.get_DM_command_in_2D(residual_list[best_i])  ]
    xlabel_list = ['' for _ in range(len(im_list))]
    ylabel_list = ['' for _ in range(len(im_list))]

    dm_mode = cmd2opd *  util.get_DM_command_in_2D( disturbance_cmd)

    vlims=[[np.min(sig_list[0]),np.max(sig_list[0])]] + [[np.nanmin(dm_mode), np.nanmax(dm_mode)] for _ in im_list[1:]]
    title_list = ['ZWFS signal', 'DM aberration' , 'DM reconstruction', 'residuals']
    cbar_label_list = [ 'normalized signal [adu]', 'OPD [nm]', 'OPD [nm]','OPD [nm]']
    util.nice_heatmap_subplots(im_list , xlabel_list, ylabel_list, title_list,cbar_label_list, fontsize=15, vlims=vlims, \
                            cbar_orientation = 'bottom', axis_off=True, savefig= current_path + f'reco_IM_signal_{m}_mask{phasemask_name}.png')

        
    #basis_name = 'fourier' #'zonal'
    plt.figure(figsize=(8,5))
    plt.plot( Smax_grid, 2*np.pi/1050 * cmd2opd * np.array( rmse_list)  )
    #plt.axhline( np.var(test_field.phase[z.wvls[0]][z.pup>0]) , label='initial')
    #plt.legend()
    plt.xlabel('Singular value truncation index',fontsize=15)
    plt.ylabel('Reconstructed RMSE [rad]',fontsize=15) 
    plt.gca().tick_params(labelsize=15)   
    plt.savefig(current_path + f'rmse_vs_svd_trunction_measured_IM_signal_{m}_{phasemask_name}.png', bbox_inches='tight', dpi=200)


    write_dict = {
    "disturbance" : disturb_list,
    "signal" : sig_list,
    "svd_trucation_grid":Smax_grid,
    "reconstructor" :R_list ,
    "reco_cmd":c_list ,
    "cmd_rmse":rmse_list ,
    "cmd_residual":residual_list  ,
    "IM": IM
    }



    # Create a list of HDUs (Header Data Units)
    hdul = fits.HDUList()

    # Add each list to the HDU list as a new extension
    for list_name, data_list in write_dict.items():
        # Convert list to numpy array for FITS compatibility
        data_array = np.array(data_list, dtype=float)  # Ensure it is a float array or any appropriate type

        # Create a new ImageHDU with the data
        hdu = fits.ImageHDU(data_array)

        # Set the EXTNAME header to the variable name
        hdu.header['EXTNAME'] = list_name

        # Append the HDU to the HDU list
        hdul.append(hdu)

    # Write the HDU list to a FITS file
    hdul.writeto( current_path + f'pt1-reco_IM_signal_telemetry_act{m}_{phasemask_name}_{tstamp}.fits', overwrite=True)



# ---- 2) poke mode on reconstructor basis and reconstruct 

# check centering 
zwfs.dm.send_data(zwfs.dm_shapes['flat_dm'] )
phasemask_centering_tool.move_relative_and_get_image(zwfs, phasemask, savefigName=fig_path + 'delme.png')

# get new reference 
I0, N0 = util.get_reference_images(zwfs, phasemask, theta_degrees=11.8, number_of_frames=256, \
compass = True, compass_origin=None, savefig= None )

zwfs.dm.send_data(zwfs.dm_shapes['flat_dm'] )

current_path = fig_path + exper_path +  "2_reco_on_basis_signal/" 
if not os.path.exists(current_path ):
   os.makedirs(current_path)

# detector noise covariance matrix
noise_cov = zwfs.estimate_noise_covariance( number_of_frames = 1000, where = 'pupil' )

im_list = [ noise_cov ]            
xlabel_list = ['registered\npupil pixels' for _ in range(len(im_list))]
ylabel_list = ['registered \npupil pixels' for _ in range(len(im_list))]
vlims = [[np.nanmin(iii), np.nanmax(iii)] for iii in im_list]
title_list = ['reference pupil covariance']
cbar_label_list = [ r'$\sigma^2$' ]
util.nice_heatmap_subplots(im_list , xlabel_list, ylabel_list, title_list,cbar_label_list, fontsize=15, vlims=vlims, \
                        cbar_orientation = 'right', axis_off=False, savefig= current_path + f'covariance_of_registered_pixels_mask{phasemask_name}.png')

# no input phase aberrations 
phase_cov = np.eye( np.array(IM).shape[1] )


for m in [43,55,65,75]:
    print('calculating for m = ', m)
    
    disturbance_cmd = poke_amp * M2C_0[m] 

    zwfs.dm.send_data( zwfs.dm_shapes['flat_dm'] + dm_pupil_filt * disturbance_cmd )

    time.sleep( 0.1 )

    residual_list = []
    rmse_list =[]
    c_list =[]
    sig_list = []
    R_list = []

    disturb_list = [disturbance_cmd ]

    Smax_grid = np.arange( 2,100,3)

    for Smax in Smax_grid: 
        
        #U, S, Vt = np.linalg.svd( IM, full_matrices=False)

        #R  = (Vt.T * [1/ss if i < Smax else 0 for i,ss in enumerate(S)])  @ U.T
                
        R_map = (phase_cov @ IM.T @ np.linalg.pinv(IM @ phase_cov.T @ IM.T + noise_cov) )

        U, S, Vt = np.linalg.svd( R_map, full_matrices=False)

        R = (U * np.array([ss if i < Smax else 0 for i,ss in enumerate(S)]))  @ Vt


        """TT_vectors = util.get_tip_tilt_vectors()

        TT_space = M2C @ TT_vectors
            
        U_TT, S_TT, Vt_TT = np.linalg.svd( TT_space, full_matrices=False)

        I2M_TT = U_TT.T @ R 

        M2C_TT = M2C_0.T @ U_TT # since pinned need M2C to go back to 140 dimension vector  

        R_HO = (np.eye(U_TT.shape[0])  - U_TT @ U_TT.T) @ R

        # go to Eigenmodes for modal control in higher order reconstructor
        U_HO, S_HO, Vt_HO = np.linalg.svd( R_HO, full_matrices=False)
        I2M_HO = Vt_HO  
        M2C_HO = M2C_0.T @ (U_HO * S_HO) # since pinned need M2C to go back to 140 dimension vector 
        """
        i = np.mean( zwfs.get_some_frames(number_of_frames=100, apply_manual_reduction=True),axis=0) #z.detection_chain( test_field, FPM_on=True, include_shotnoise=True, ph_per_s_per_m2_per_nm=True, grids_aligned=True, replace_nan_with=0 )
        #o = #z.detection_chain( test_field, FPM_on=False, include_shotnoise=True, ph_per_s_per_m2_per_nm=True, grids_aligned=True, replace_nan_with=0 )
        
        sig = ( i / np.mean( i ) -  I0 / np.mean( I0 ) ) # I0_theory/ np.mean(I0_theory) #

        # update distrubance after measurement 
        #for _ in range(rows_to_jump):
        #    scrn.add_row()
        #disturbance_cmd = np.array( util.create_phase_screen_cmd_for_DM(scrn,  scaling_factor=scrn_scaling_factor   , drop_indicies = corner_indicies, plot_cmd=False) )
        """
        e_TT = I2M_TT @ sig.reshape(-1)[pupil_pixels]
        
        u_TT = e_TT #pid.process( e_TT )
        
        c_TT = M2C_TT @ u_TT 
        
        e_HO = I2M_HO @ sig.reshape(-1)[pupil_pixels]

        u_HO = e_HO #leak.process( e_HO )
        
        c_HO = M2C_HO @ u_HO 
        """
        #c = c_TT + c_HO
        # or if 1/poke amp mult by IM prior use : M2C_0.T  @ R @ sig.reshape(-1)[pupil_pixels]
        c =  poke_amp * M2C_0.T  @ R @ sig.reshape(-1)[pupil_pixels]

        #zwfs.dm.send_data( zwfs.dm_shapes['flat_dm'] + disturbance_cmd - c_HO - c_TT ) # same way to rtc PID 
        #time.sleep(0.05)  
        # only measure residual in the registered pupil on DM 
        residual =  (disturbance_cmd - c)[dm_pupil_filt]
        rmse = np.nanstd( residual )
        
        rmse_list.append( rmse ) 
        
        residual_list.append( disturbance_cmd - c)
        c_list.append( c )
        sig_list.append( sig )
        R_list.append( R )
        
    best_i = np.argmin( rmse_list )


    im_list = [ np.flipud(sig), cmd2opd * util.get_DM_command_in_2D( disturbance_cmd ) , cmd2opd * util.get_DM_command_in_2D( c_list[best_i] ) , \
                cmd2opd * util.get_DM_command_in_2D(residual_list[best_i])  ]
            
    xlabel_list = ['' for _ in range(len(im_list))]
    ylabel_list = ['' for _ in range(len(im_list))]

    dm_mode = cmd2opd *  util.get_DM_command_in_2D( disturbance_cmd)

    vlims=[[np.min(sig_list[best_i]),np.max(sig_list[best_i])]] + [[np.nanmin(dm_mode), np.nanmax(dm_mode)] for _ in im_list[1:]]
    title_list = ['ZWFS signal', 'DM aberration' , 'DM reconstruction', 'residuals']
    cbar_label_list = [ 'normalized signal [adu]', 'OPD [nm]', 'OPD [nm]','OPD [nm]']
    util.nice_heatmap_subplots(im_list , xlabel_list, ylabel_list, title_list,cbar_label_list, fontsize=15, vlims=vlims, \
                            cbar_orientation = 'bottom', axis_off=True, savefig= current_path + f'reco_single_mode_{m}_mask{phasemask_name}.png')

        

    #basis_name = 'fourier' #'zonal'
    plt.figure(figsize=(8,5))
    plt.plot( Smax_grid, 2*np.pi/1050 * cmd2opd * np.array( rmse_list)  )
    #plt.axhline( np.var(test_field.phase[z.wvls[0]][z.pup>0]) , label='initial')
    plt.legend()
    plt.xlabel('Singular value truncation index',fontsize=15)
    plt.ylabel('Reconstructed RMSE [rad]',fontsize=15) 
    plt.gca().tick_params(labelsize=15)   
    plt.savefig(current_path  +f'rmse_vs_svd_trunction_measured_single_mode_{m}_{phasemask_name}.png', bbox_inches='tight', dpi=200)



    write_dict = {
    "disturbance" : disturb_list,
    "signal" : sig_list,
    "svd_trucation_grid":Smax_grid,
    "reconstructor" :R_list ,
    "reco_cmd":c_list ,
    "cmd_rmse":rmse_list ,
    "cmd_residual":residual_list  ,
    "IM":IM,
    }



    # Create a list of HDUs (Header Data Units)
    hdul = fits.HDUList()

    # Add each list to the HDU list as a new extension
    for list_name, data_list in write_dict.items():
        # Convert list to numpy array for FITS compatibility
        data_array = np.array(data_list, dtype=float)  # Ensure it is a float array or any appropriate type

        # Create a new ImageHDU with the data
        hdu = fits.ImageHDU(data_array)

        # Set the EXTNAME header to the variable name
        hdu.header['EXTNAME'] = list_name

        # Append the HDU to the HDU list
        hdul.append(hdu)

    # Write the HDU list to a FITS file
    hdul.writeto( current_path + f'pt2-reco_single_mode_telemetry_{m}_{phasemask_name}_{tstamp}.fits', overwrite=True)
        
        
        
        
    
    
# ----3) put Kolmogorov mode on DM and reconstruct 

current_path = fig_path + exper_path +  "3_reco_kolmogorov/" 
if not os.path.exists(current_path ):
   os.makedirs(current_path)


# check centering 
zwfs.dm.send_data(zwfs.dm_shapes['flat_dm'] )
phasemask_centering_tool.move_relative_and_get_image(zwfs, phasemask, savefigName=fig_path + 'delme.png')

# get new reference 
I0, N0 = util.get_reference_images(zwfs, phasemask, theta_degrees=11.8, number_of_frames=256, \
compass = True, compass_origin=None, savefig= None )

zwfs.dm.send_data(zwfs.dm_shapes['flat_dm'] )


# Which is truncation is best 

Nx_act = 12 # actuators across DM 

D = 1.8 #m effective diameter of the telescope

screen_pixels = Nx_act*2**3  #pixels inthe inital screen before projection onto DM

corner_indicies = [0, Nx_act-1, Nx_act * (Nx_act-1), -1] 

scrn_scaling_factor =  0.1 

rows_to_jump = 2 # how many rows to jump on initial phase screen for each Baldr loop

distance_per_correction = rows_to_jump * D/screen_pixels # effective distance travelled by turbulence per AO iteration 
print(f'{rows_to_jump} rows jumped per AO command in initial phase screen of {screen_pixels} pixels. for {D}m mirror this corresponds to a distance_per_correction = {distance_per_correction}m')

scrn = aotools.infinitephasescreen.PhaseScreenVonKarman(nx_size=screen_pixels, pixel_scale=D/screen_pixels,r0=0.1,L0=12)

disturbance_cmd = util.create_phase_screen_cmd_for_DM(scrn,  scaling_factor=scrn_scaling_factor   , drop_indicies = corner_indicies, plot_cmd=False)

zwfs.dm.send_data( zwfs.dm_shapes['flat_dm'] + dm_pupil_filt * disturbance_cmd )

rmse_list =[]
c_list =[]
sig_list = []
R_list = []
Smax_grid = np.arange( 2,100,3)

for Smax in Smax_grid: 
    
    #U, S, Vt = np.linalg.svd( IM, full_matrices=False)

    #R  = (Vt.T * [1/ss if i < Smax else 0 for i,ss in enumerate(S)])  @ U.T
    R_map = (phase_cov @ IM.T @ np.linalg.pinv(IM @ phase_cov.T @ IM.T + noise_cov) )

    U, S, Vt = np.linalg.svd( R_map, full_matrices=False)

    R = (U * np.array([ss if i < Smax else 0 for i,ss in enumerate(S)]))  @ Vt

    """TT_vectors = util.get_tip_tilt_vectors()

    TT_space = M2C @ TT_vectors
        
    U_TT, S_TT, Vt_TT = np.linalg.svd( TT_space, full_matrices=False)

    I2M_TT = U_TT.T @ R 

    M2C_TT = M2C_0.T @ U_TT # since pinned need M2C to go back to 140 dimension vector  

    R_HO = (np.eye(U_TT.shape[0])  - U_TT @ U_TT.T) @ R

    # go to Eigenmodes for modal control in higher order reconstructor
    U_HO, S_HO, Vt_HO = np.linalg.svd( R_HO, full_matrices=False)
    I2M_HO = Vt_HO  
    M2C_HO = M2C_0.T @ (U_HO * S_HO) # since pinned need M2C to go back to 140 dimension vector 
    """
    i = np.mean( zwfs.get_some_frames(number_of_frames=100, apply_manual_reduction=True),axis=0) #z.detection_chain( test_field, FPM_on=True, include_shotnoise=True, ph_per_s_per_m2_per_nm=True, grids_aligned=True, replace_nan_with=0 )
    #o = #z.detection_chain( test_field, FPM_on=False, include_shotnoise=True, ph_per_s_per_m2_per_nm=True, grids_aligned=True, replace_nan_with=0 )
    
    sig = i / np.mean( i ) -  I0 / np.mean( I0 ) # I0_theory/ np.mean(I0_theory) #

    # update distrubance after measurement 
    #for _ in range(rows_to_jump):
    #    scrn.add_row()
    #disturbance_cmd = np.array( util.create_phase_screen_cmd_for_DM(scrn,  scaling_factor=scrn_scaling_factor   , drop_indicies = corner_indicies, plot_cmd=False) )
    """
    e_TT = I2M_TT @ sig.reshape(-1)[pupil_pixels]
    
    u_TT = e_TT #pid.process( e_TT )
    
    c_TT = M2C_TT @ u_TT 
    
    e_HO = I2M_HO @ sig.reshape(-1)[pupil_pixels]

    u_HO = e_HO #leak.process( e_HO )
    
    c_HO = M2C_HO @ u_HO 
    """
    #c = c_TT + c_HO
    # or if 1/poke amp mult by IM prior use : M2C_0.T  @ R @ sig.reshape(-1)[pupil_pixels]
    c =  poke_amp * M2C_0.T  @ R @ sig.reshape(-1)[pupil_pixels]

    #zwfs.dm.send_data( zwfs.dm_shapes['flat_dm'] + disturbance_cmd - c_HO - c_TT ) # same way to rtc PID 
    #time.sleep(0.05)  
    # only measure residual in the registered pupil on DM 
    residual =  (disturbance_cmd - c)[dm_pupil_filt]
    
    residual_list.append( disturbance_cmd - c)
    rmse = np.nanstd( residual )
    
    rmse_list.append( rmse ) 
    c_list.append( c )
    sig_list.append( sig )
    R_list.append( R )
    
best_i = np.argmin( rmse_list )

use_dm_filt = 1
if use_dm_filt:
    im_list = [ np.flipud(sig_list[best_i]), cmd2opd * util.get_DM_command_in_2D( dm_pupil_filt * disturbance_cmd ) , cmd2opd * util.get_DM_command_in_2D( dm_pupil_filt * c_list[best_i] ) , \
                cmd2opd * util.get_DM_command_in_2D( dm_pupil_filt * residual_list[best_i])  ]
else:
    im_list = [ np.flipud(sig_list[best_i]), cmd2opd * util.get_DM_command_in_2D( dm_pupil_filt * disturbance_cmd ) , cmd2opd * util.get_DM_command_in_2D( dm_pupil_filt * c_list[best_i] ) , \
                cmd2opd * util.get_DM_command_in_2D( dm_pupil_filt * residual_list[best_i])  ]
 
xlabel_list = ['' for _ in range(len(im_list))]
ylabel_list = ['' for _ in range(len(im_list))]
dm_mode = cmd2opd *  util.get_DM_command_in_2D( disturbance_cmd )
vlims=[[np.min(sig_list[best_i]),np.max(sig_list[best_i])]] + [[np.nanmin(dm_mode), np.nanmax(dm_mode)] for _ in im_list[1:]]
title_list = ['ZWFS signal', 'DM aberration' , 'DM reconstruction', 'residuals']
cbar_label_list = [ 'normalized signal [adu]', 'OPD [nm]', 'OPD [nm]','OPD [nm]']
util.nice_heatmap_subplots(im_list , xlabel_list, ylabel_list, title_list,cbar_label_list, fontsize=15, vlims=vlims, \
                        cbar_orientation = 'bottom', axis_off=True, savefig=current_path +  f'reco_Kolmogorov_mask{phasemask_name}.png')

 
plt.figure(figsize=(8,5))
plt.plot( Smax_grid, 2*np.pi/1050 * cmd2opd * np.array( rmse_list)  )
#plt.axhline( np.var(test_field.phase[z.wvls[0]][z.pup>0]) , label='initial')
plt.legend()
plt.xlabel('Singular value truncation index',fontsize=15)
plt.ylabel('Reconstructed RMSE [rad]',fontsize=15) 
plt.gca().tick_params(labelsize=15)   
plt.savefig(current_path +f'rmse_vs_svd_trunction_measured_Kolmogorov_{phasemask_name}.png', bbox_inches='tight', dpi=200)




write_dict = {
"disturbance" : disturb_list,
"signal" : sig_list,
"svd_trucation_grid":Smax_grid,
"reconstructor" :R_list ,
"reco_cmd":c_list ,
"cmd_rmse":rmse_list ,
"cmd_residual":residual_list,  
"IM": IM
}



# Create a list of HDUs (Header Data Units)
hdul = fits.HDUList()

# Add each list to the HDU list as a new extension
for list_name, data_list in write_dict.items():
    # Convert list to numpy array for FITS compatibility
    data_array = np.array(data_list, dtype=float)  # Ensure it is a float array or any appropriate type

    # Create a new ImageHDU with the data
    hdu = fits.ImageHDU(data_array)

    # Set the EXTNAME header to the variable name
    hdu.header['EXTNAME'] = list_name

    # Append the HDU to the HDU list
    hdul.append(hdu)

# Write the HDU list to a FITS file
hdul.writeto( current_path + f'pt3-reco_kolmogorov-r0-0.1_amp{scrn_scaling_factor}_telemetry_{phasemask_name}_{tstamp}.fits', overwrite=True)
    
    
    
# ---- CLOSE THE LOOP?  - start static

current_path = fig_path + exper_path +  "4_closed_loop_static_ab/" 
if not os.path.exists(current_path ):
   os.makedirs(current_path)


# check centering 
zwfs.dm.send_data(zwfs.dm_shapes['flat_dm'] )
phasemask_centering_tool.move_relative_and_get_image(zwfs, phasemask, savefigName=fig_path + 'delme.png')

# get new reference 
I0, N0 = util.get_reference_images(zwfs, phasemask, theta_degrees=11.8, number_of_frames=256, \
compass = True, compass_origin=None, savefig= None )

zwfs.dm.send_data(zwfs.dm_shapes['flat_dm'] )


TT_vectors = util.get_tip_tilt_vectors()

TT_space = M2C_0 @ TT_vectors
    
U_TT, S_TT, Vt_TT = np.linalg.svd( TT_space, full_matrices=False)

I2M_TT = U_TT.T @ R 

M2C_TT = poke_amp * M2C_0.T @ U_TT # since pinned need M2C to go back to 140 dimension vector  

R_HO = (np.eye(U_TT.shape[0])  - U_TT @ U_TT.T) @ R

# go to Eigenmodes for modal control in higher order reconstructor
U_HO, S_HO, Vt_HO = np.linalg.svd( R_HO, full_matrices=False)
I2M_HO = Vt_HO  
M2C_HO = poke_amp *  M2C_0.T @ (U_HO * S_HO) # since pinned need M2C to go back to 140 dimension vector


# pupil outside to sample reference field / strehl 

central_lambda = util.find_central_wavelength(lambda_cut_on=900e-9, lambda_cut_off=1180e-9, T=1900)
print(f"The central wavelength is {central_lambda * 1e9:.2f} nm")


wvl = 1e6 * central_lambda # 0.900 #1.040 # um  
phase_shift = util.get_phasemask_phaseshift( wvl= wvl, depth = phasemask.phasemask_parameters[phasemask_name]['depth'] )
mask_diam = 1e-6 * phasemask.phasemask_parameters[phasemask_name]['diameter']
N0_theory0, I0_theory0 = util.get_theoretical_reference_pupils( wavelength = wvl*1e-6 ,F_number = 21.2, mask_diam = mask_diam,\
                                        diameter_in_angular_units = False,  phaseshift = phase_shift , padding_factor = 4, \
                                        debug= False, analytic_solution = True )

M = I0_theory0.shape[0]
N = I0_theory0.shape[1]

m = zwfs.I0.shape[1]
n = zwfs.I0.shape[0]

# A = pi * r^2 => r = sqrt( A / pi)
new_radius = (zwfs.pupil_pixel_filter.sum()/np.pi)**0.5
x_c = np.mean( np.unravel_index( np.where( zwfs.pupil_pixel_filter ), zwfs.I0.shape)[1])
y_c = np.mean( np.unravel_index( np.where( zwfs.pupil_pixel_filter ), zwfs.I0.shape)[0])

I0_theory = util.interpolate_pupil_to_measurement( N0_theory0, I0_theory0, M, N, m, n, x_c, y_c, new_radius)

N0_theory = util.interpolate_pupil_to_measurement( N0_theory0, N0_theory0, M, N, m, n, x_c, y_c, new_radius)

pupil_outer_perim_filter = (~zwfs.bad_pixel_filter * (abs( I0_theory - N0_theory ) > 0.1 ).reshape(-1) * (~zwfs.pupil_pixel_filter) )

plt.figure()
plt.imshow( pupil_outer_perim_filter.reshape(zwfs.I0.shape) ) 
plt.savefig( current_path + 'outer_pupil_filter.png')




# init our controllers 

rho = 0 * np.ones( I2M_HO.shape[0] )
kp_leak = 0 * np.ones( I2M_HO.shape[0] )
lower_limit_leak = -100 * np.ones( I2M_HO.shape[0] )
upper_limit_leak = 100 * np.ones( I2M_HO.shape[0] )

leak = LeakyIntegrator(rho=rho, kp=kp_leak, lower_limit=lower_limit_leak, upper_limit=upper_limit_leak )

kp = 0. * np.ones( I2M_TT.shape[0] )
ki = 0. * np.ones( I2M_TT.shape[0] )
kd = 0. * np.ones( I2M_TT.shape[0] )
setpoint = np.zeros( I2M_TT.shape[0] )
lower_limit_pid = -100 * np.ones( I2M_TT.shape[0] )
upper_limit_pid = 100 * np.ones( I2M_TT.shape[0] )

pid = PIDController(kp, ki, kd, upper_limit_pid, lower_limit_pid, setpoint)

s_list = []
e_TT_list = []
u_TT_list = []
c_TT_list = []
e_HO_list = []
u_HO_list = []
c_HO_list = []
atm_disturb_list = []
dm_disturb_list = []
rmse_list = []
flux_outside_pupil_list = []
residual_list = []
close_after = 10 

pid.reset() 
leak.reset()

#disturb_basis = util.construct_command_basis( 'fourier_pinned_edges')
disturbance_cmd = 0.3 * TT_vectors[:,0]
zwfs.dm.send_data(zwfs.dm_shapes['flat_dm'] + dm_pupil_filt * disturbance_cmd) # only apply in registered pupil 
time.sleep(0.1)

for it in range(40):
    
    if it > close_after : # close after 
        pid.kp = 1 * np.ones( I2M_TT.shape[0] )
        pid.ki = 0.3 * np.ones( I2M_TT.shape[0] )
        
        #leak.rho[2:5] = 0.2 #* np.ones( I2M_HO.shape[0] )
        #leak.kp[2:5] = 0.5

        im_list =  [  sig, util.get_DM_command_in_2D( cmd2opd * disturbance_cmd),  util.get_DM_command_in_2D( cmd2opd * c_TT),\
                      util.get_DM_command_in_2D( cmd2opd * c_HO),  util.get_DM_command_in_2D( cmd2opd * (disturbance_cmd - c_HO - c_TT) )] 
        xlabel_list = [ "" for _ in im_list]
        ylabel_list = [ "" for _ in im_list]
        title_list = [ "ZWFS signal", "aberration" , "reco. TT", "reco. HO", "residuals"]
        cbar_label_list =  [ "ADU", "OPD [nm]",  "OPD [nm]" ,"OPD [nm]", "OPD [nm]"]
        vlims=[[np.min(sig),np.max(sig)]] + [[np.min(cmd2opd * disturbance_cmd), np.max(cmd2opd * disturbance_cmd)] for _ in im_list[1:]]
        savefig = current_path + 'delme.png' 
        util.nice_heatmap_subplots( im_list , xlabel_list, ylabel_list, title_list,cbar_label_list, \
                            fontsize=15, cbar_orientation = 'bottom', vlims=vlims, axis_off=True, savefig=savefig)
        _ = input('next?') 

    i = np.mean( zwfs.get_some_frames(number_of_frames=20, apply_manual_reduction=True),axis=0) #z.detection_chain( test_field, FPM_on=True, include_shotnoise=True, ph_per_s_per_m2_per_nm=True, grids_aligned=True, replace_nan_with=0 )
    #o = #z.detection_chain( test_field, FPM_on=False, include_shotnoise=True, ph_per_s_per_m2_per_nm=True, grids_aligned=True, replace_nan_with=0 )
    
    sig = i / np.mean( i ) -  I0 / np.mean( I0 ) # I0_theory/ np.mean(I0_theory) #

    # update distrubance after measurement 
    #for _ in range(rows_to_jump):
    #    scrn.add_row()
    #disturbance_cmd = np.array( util.create_phase_screen_cmd_for_DM(scrn,  scaling_factor=scrn_scaling_factor   , drop_indicies = corner_indicies, plot_cmd=False) )


    e_TT = I2M_TT @ sig.reshape(-1)[pupil_pixels]
    
    u_TT = pid.process( e_TT )
    
    c_TT = M2C_TT @ u_TT 
    
    e_HO = I2M_HO @ sig.reshape(-1)[pupil_pixels]

    u_HO = leak.process( e_HO )
    
    c_HO = M2C_HO @ u_HO 

    #c = R @ sig

    zwfs.dm.send_data( zwfs.dm_shapes['flat_dm'] + disturbance_cmd - c_HO - c_TT ) # same way to rtc PID 
    time.sleep(0.05)  
    # only measure residual in the registered pupil on DM 
    residual =  (disturbance_cmd - c_HO - c_TT)[dm_pupil_filt]
    rmse = np.nanstd( residual )
    
    # telemetry 
    s_list.append( sig )
    e_TT_list.append( e_TT )
    u_TT_list.append( u_TT )
    c_TT_list.append( c_TT )
    
    e_HO_list.append( e_HO )
    u_HO_list.append( u_HO )
    c_HO_list.append( c_HO )
    
    atm_disturb_list.append( scrn.scrn )
    dm_disturb_list.append( disturbance_cmd )
    
    residual_list.append( residual )
    rmse_list.append( rmse )
    flux_outside_pupil_list.append( np.sum( sig.reshape(-1)[pupil_outer_perim_filter] ) )
    print( it, f'rmse = {rmse}, flux outside = {flux_outside_pupil_list[-1]}' )



# write telemetry to file 

# Dictionary of lists and their names
lists_dict = {
    "s_list": s_list,
    "e_TT_list": e_TT_list,
    "u_TT_list": u_TT_list,
    "c_TT_list": c_TT_list,
    "e_HO_list": e_HO_list,
    "u_HO_list": u_HO_list,
    "c_HO_list": c_HO_list,
    "pid_kp_list": pid.kp,
    "pid_ki_list": pid.ki,
    "pid_kd_list": pid.kd,
    "leay_kp_list": leak.kp,
    "leay_rho_list": leak.rho,
    "atm_disturb_list": atm_disturb_list,
    "dm_disturb_list": dm_disturb_list,
    "rmse_list": rmse_list,
    "residual_list": residual_list,
    "flux_outside_pupil_list":flux_outside_pupil_list,
    "IM":IM,
    "R" : R,
    "I2M_TT" : I2M_TT,
    "I2M_HO" : I2M_HO,
    "M2C_TT" : M2C_TT,
    "M2C_HO" : M2C_HO
}

# Create a list of HDUs (Header Data Units)
hdul = fits.HDUList()

# Add each list to the HDU list as a new extension
for list_name, data_list in lists_dict.items():
    # Convert list to numpy array for FITS compatibility
    data_array = np.array(data_list, dtype=float)  # Ensure it is a float array or any appropriate type

    # Create a new ImageHDU with the data
    hdu = fits.ImageHDU(data_array)

    # Set the EXTNAME header to the variable name
    hdu.header['EXTNAME'] = list_name

    # Append the HDU to the HDU list
    hdul.append(hdu)

# Write the HDU list to a FITS file
hdul.writeto(current_path + f'closed_loop_TTstatic_telemetry_{itera}.fits', overwrite=True)




    
# ---- CLOSE THE LOOP?  - sdynamic

current_path = fig_path + exper_path +  "5_closed_loop_dynamic_kolmogorov/" 
if not os.path.exists(current_path ):
   os.makedirs(current_path)


# check centering 
zwfs.dm.send_data(zwfs.dm_shapes['flat_dm'] )
phasemask_centering_tool.move_relative_and_get_image(zwfs, phasemask, savefigName=fig_path + 'delme.png')

# get new reference 
I0, N0 = util.get_reference_images(zwfs, phasemask, theta_degrees=11.8, number_of_frames=256, \
compass = True, compass_origin=None, savefig= None )

zwfs.dm.send_data(zwfs.dm_shapes['flat_dm'] )


TT_vectors = util.get_tip_tilt_vectors()

TT_space = M2C_0 @ TT_vectors
    
U_TT, S_TT, Vt_TT = np.linalg.svd( TT_space, full_matrices=False)

I2M_TT = U_TT.T @ R 

M2C_TT = poke_amp * M2C_0.T @ U_TT # since pinned need M2C to go back to 140 dimension vector  

R_HO = (np.eye(U_TT.shape[0])  - U_TT @ U_TT.T) @ R

# go to Eigenmodes for modal control in higher order reconstructor
U_HO, S_HO, Vt_HO = np.linalg.svd( R_HO, full_matrices=False)
I2M_HO = Vt_HO  
M2C_HO = poke_amp *  M2C_0.T @ (U_HO * S_HO) # since pinned need M2C to go back to 140 dimension vector


# pupil outside to sample reference field / strehl 

central_lambda = util.find_central_wavelength(lambda_cut_on=900e-9, lambda_cut_off=1180e-9, T=1900)
print(f"The central wavelength is {central_lambda * 1e9:.2f} nm")


wvl = 1e6 * central_lambda # 0.900 #1.040 # um  
phase_shift = util.get_phasemask_phaseshift( wvl= wvl, depth = phasemask.phasemask_parameters[phasemask_name]['depth'] )
mask_diam = 1e-6 * phasemask.phasemask_parameters[phasemask_name]['diameter']
N0_theory0, I0_theory0 = util.get_theoretical_reference_pupils( wavelength = wvl*1e-6 ,F_number = 21.2, mask_diam = mask_diam,\
                                        diameter_in_angular_units = False,  phaseshift = phase_shift , padding_factor = 4, \
                                        debug= False, analytic_solution = True )

M = I0_theory0.shape[0]
N = I0_theory0.shape[1]

m = zwfs.I0.shape[1]
n = zwfs.I0.shape[0]

# A = pi * r^2 => r = sqrt( A / pi)
new_radius = (zwfs.pupil_pixel_filter.sum()/np.pi)**0.5
x_c = np.mean( np.unravel_index( np.where( zwfs.pupil_pixel_filter ), zwfs.I0.shape)[1])
y_c = np.mean( np.unravel_index( np.where( zwfs.pupil_pixel_filter ), zwfs.I0.shape)[0])

I0_theory = util.interpolate_pupil_to_measurement( N0_theory0, I0_theory0, M, N, m, n, x_c, y_c, new_radius)

N0_theory = util.interpolate_pupil_to_measurement( N0_theory0, N0_theory0, M, N, m, n, x_c, y_c, new_radius)

pupil_outer_perim_filter = (~zwfs.bad_pixel_filter * (abs( I0_theory - N0_theory ) > 0.1 ).reshape(-1) * (~zwfs.pupil_pixel_filter) )

plt.figure()
plt.imshow( pupil_outer_perim_filter.reshape(zwfs.I0.shape) ) 
plt.savefig( current_path + 'outer_pupil_filter.png')




# init our controllers 

cl_try = 4 # which closed loop try are we up to?

rho = 0 * np.ones( I2M_HO.shape[0] )
kp_leak = 0 * np.ones( I2M_HO.shape[0] )
lower_limit_leak = -100 * np.ones( I2M_HO.shape[0] )
upper_limit_leak = 100 * np.ones( I2M_HO.shape[0] )

leak = LeakyIntegrator(rho=rho, kp=kp_leak, lower_limit=lower_limit_leak, upper_limit=upper_limit_leak )

kp = 0. * np.ones( I2M_TT.shape[0] )
ki = 0. * np.ones( I2M_TT.shape[0] )
kd = 0. * np.ones( I2M_TT.shape[0] )
setpoint = np.zeros( I2M_TT.shape[0] )
lower_limit_pid = -100 * np.ones( I2M_TT.shape[0] )
upper_limit_pid = 100 * np.ones( I2M_TT.shape[0] )

pid = PIDController(kp, ki, kd, upper_limit_pid, lower_limit_pid, setpoint)

s_list = []
e_TT_list = []
u_TT_list = []
c_TT_list = []
e_HO_list = []
u_HO_list = []
c_HO_list = []
atm_disturb_list = []
dm_disturb_list = []
rmse_list = []
flux_outside_pupil_list = []
residual_list = []
close_after = 10

pid.reset() 
leak.reset()

#disturb_basis = util.construct_command_basis( 'fourier_pinned_edges')
#disturbance_cmd = 0.3 * TT_vectors[:,0]

# init a phasescreen to roll across DM 

Nx_act = 12 # actuators across DM 

D = 1.8 #m effective diameter of the telescope

screen_pixels = Nx_act*2**3  #pixels inthe inital screen before projection onto DM

corner_indicies = [0, Nx_act-1, Nx_act * (Nx_act-1), -1] 

scrn_scaling_factor =  0.15 

rows_to_jump = 2 # how many rows to jump on initial phase screen for each Baldr loop

distance_per_correction = rows_to_jump * D/screen_pixels # effective distance travelled by turbulence per AO iteration 
print(f'{rows_to_jump} rows jumped per AO command in initial phase screen of {screen_pixels} pixels. for {D}m mirror this corresponds to a distance_per_correction = {distance_per_correction}m')

scrn = aotools.infinitephasescreen.PhaseScreenVonKarman(nx_size=screen_pixels, pixel_scale=D/screen_pixels,r0=0.1,L0=12)

disturbance_cmd = util.create_phase_screen_cmd_for_DM(scrn,  scaling_factor=scrn_scaling_factor   , drop_indicies = corner_indicies, plot_cmd=False)

plt.figure()
plt.imshow( util.get_DM_command_in_2D(disturbance_cmd, Nx_act=12) )
plt.colorbar()
plt.title( 'initial Kolmogorov aberration to apply to DM')
plt.savefig( current_path  + 'initial_phasescreen.png')


zwfs.dm.send_data(zwfs.dm_shapes['flat_dm'] + dm_pupil_filt * disturbance_cmd) # only apply in registered pupil 
time.sleep(0.1)

for it in range(50):
    
    if it > close_after : # close after 
        pid.kp = 0.9 * np.ones( I2M_TT.shape[0] )
        pid.ki = 0.1 * np.ones( I2M_TT.shape[0] )
        
        leak.rho[:30] = 0.1 #* np.ones( I2M_HO.shape[0] )
        leak.kp[:30] = 0.5

        im_list =  [  sig, util.get_DM_command_in_2D( cmd2opd * disturbance_cmd),  util.get_DM_command_in_2D( cmd2opd * c_TT),\
                      util.get_DM_command_in_2D( cmd2opd * c_HO),  util.get_DM_command_in_2D( cmd2opd * (disturbance_cmd - c_HO - c_TT) )] 
        xlabel_list = [ "" for _ in im_list]
        ylabel_list = [ "" for _ in im_list]
        title_list = [ "ZWFS signal", "aberration" , "reco. TT", "reco. HO", "residuals"]
        cbar_label_list =  [ "ADU", "OPD [nm]",  "OPD [nm]" ,"OPD [nm]", "OPD [nm]"]
        vlims=[[np.min(sig),np.max(sig)]] + [[np.min(cmd2opd * disturbance_cmd), np.max(cmd2opd * disturbance_cmd)] for _ in im_list[1:]]
        savefig = current_path + 'delme.png' 
        util.nice_heatmap_subplots( im_list , xlabel_list, ylabel_list, title_list,cbar_label_list, \
                            fontsize=15, cbar_orientation = 'bottom', vlims=vlims, axis_off=True, savefig=savefig)
        _ = input('next?') 

    i = np.mean( zwfs.get_some_frames(number_of_frames=20, apply_manual_reduction=True),axis=0) #z.detection_chain( test_field, FPM_on=True, include_shotnoise=True, ph_per_s_per_m2_per_nm=True, grids_aligned=True, replace_nan_with=0 )
    #o = #z.detection_chain( test_field, FPM_on=False, include_shotnoise=True, ph_per_s_per_m2_per_nm=True, grids_aligned=True, replace_nan_with=0 )
    
    sig = i / np.mean( i ) -  I0 / np.mean( I0 ) # I0_theory/ np.mean(I0_theory) #

    # update distrubance after measurement 
    for _ in range(rows_to_jump):
        scrn.add_row()
    disturbance_cmd = np.array( util.create_phase_screen_cmd_for_DM(scrn,  scaling_factor=scrn_scaling_factor   , drop_indicies = corner_indicies, plot_cmd=False) )


    e_TT = I2M_TT @ sig.reshape(-1)[pupil_pixels]
    
    u_TT = pid.process( e_TT )
    
    c_TT = M2C_TT @ u_TT 
    
    e_HO = I2M_HO @ sig.reshape(-1)[pupil_pixels]

    u_HO = leak.process( e_HO )
    
    c_HO = M2C_HO @ u_HO 

    #c = R @ sig

    zwfs.dm.send_data( zwfs.dm_shapes['flat_dm'] + disturbance_cmd - c_HO - c_TT ) # same way to rtc PID 
    time.sleep(0.05)  
    # only measure residual in the registered pupil on DM 
    residual =  (disturbance_cmd - c_HO - c_TT)[dm_pupil_filt]
    rmse = np.nanstd( residual )
    
    # telemetry 
    s_list.append( sig )
    e_TT_list.append( e_TT )
    u_TT_list.append( u_TT )
    c_TT_list.append( c_TT )
    
    e_HO_list.append( e_HO )
    u_HO_list.append( u_HO )
    c_HO_list.append( c_HO )
    
    atm_disturb_list.append( scrn.scrn )
    dm_disturb_list.append( disturbance_cmd )
    
    residual_list.append( residual )
    rmse_list.append( rmse )
    flux_outside_pupil_list.append( np.sum( sig.reshape(-1)[pupil_outer_perim_filter] ) )
    print( it, f'rmse = {rmse}, flux outside = {flux_outside_pupil_list[-1]}' )



# write telemetry to file 

# Dictionary of lists and their names
lists_dict = {
    "s_list": s_list,
    "e_TT_list": e_TT_list,
    "u_TT_list": u_TT_list,
    "c_TT_list": c_TT_list,
    "e_HO_list": e_HO_list,
    "u_HO_list": u_HO_list,
    "c_HO_list": c_HO_list,
    "pid_kp_list": pid.kp,
    "pid_ki_list": pid.ki,
    "pid_kd_list": pid.kd,
    "leay_kp_list": leak.kp,
    "leay_rho_list": leak.rho,
    "atm_disturb_list": atm_disturb_list,
    "dm_disturb_list": dm_disturb_list,
    "rmse_list": rmse_list,
    "residual_list": residual_list,
    "flux_outside_pupil_list":flux_outside_pupil_list,
    "IM":IM,
    "R" : R,
    "I2M_TT" : I2M_TT,
    "I2M_HO" : I2M_HO,
    "M2C_TT" : M2C_TT,
    "M2C_HO" : M2C_HO
}

# Create a list of HDUs (Header Data Units)
hdul = fits.HDUList()

# Add each list to the HDU list as a new extension
for list_name, data_list in lists_dict.items():
    # Convert list to numpy array for FITS compatibility
    data_array = np.array(data_list, dtype=float)  # Ensure it is a float array or any appropriate type

    # Create a new ImageHDU with the data
    hdu = fits.ImageHDU(data_array)

    # Set the EXTNAME header to the variable name
    hdu.header['EXTNAME'] = list_name

    # Append the HDU to the HDU list
    hdul.append(hdu)

# Write the HDU list to a FITS file
hdul.writeto(current_path + f'closed_loop_dynamic_kol{scrn_scaling_factor}_telemetry_{cl_try}.fits', overwrite=True)


plt.figure(); plt.plot( cmd2opd * np.array(rmse_list) ) ; plt.figure( current_path +f'rmse_try{cl_try}')




































