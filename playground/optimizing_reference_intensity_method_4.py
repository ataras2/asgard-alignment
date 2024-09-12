import numpy as np
import glob 
from astropy.io import fits
import time
import os 
import pandas as pd 
import matplotlib.pyplot as plt 
from scipy.optimize import curve_fit
from scipy.optimize import least_squares
from scipy.ndimage import distance_transform_edt
import importlib
import corner
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

# setup paths 
if not os.path.exists(f'tmp/{tstamp.split("T")[0]}/'):
   os.makedirs(f'tmp/{tstamp.split("T")[0]}/')

fig_path = f'tmp/{tstamp.split("T")[0]}/optimize_ref_int_method_4/' 
data_path = f'tmp/{tstamp.split("T")[0]}/optimize_ref_int_method_4/' 

if not os.path.exists(fig_path):
   os.makedirs(fig_path)


# =====================
#   SETUP
# =====================
# ====== hardware variables
beam = 3
phasemask_name = 'J3'
phasemask_OUT_offset = [1000,1000]  # relative offset (um) to take phasemask out of beam
BFO_pos = 3000 # um (absolute position of detector imgaging lens) 
dichroic_name = "J"
source_name = 'SBB'
DM_serial_number = '17DW019#122' # Syd = '17DW019#122', ANU = '17DW019#053'

# ======  set up source 

# start with source out !

# ======  set up dichroic 

# do manually (COM3 communication issue)

#  ConnectionFailedException: ConnectionFailedException: Cannot open serial port: no such file or directory

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


pupil_crop_region = [160, 220, 110, 185] #[204,268,125, 187] #[None, None, None, None] #[204 -50 ,268+50,125-50, 187+50] 

#init our ZWFS (object that interacts with camera and DM) (old path = home/baldr/Documents/baldr/ANU_demo_scripts/BALDR/)
zwfs = ZWFS.ZWFS(DM_serial_number=DM_serial_number, cameraIndex=0, DMshapes_path = 'DMShapes/', pupil_crop_region=pupil_crop_region ) 

# the sydney BMC multi-3.5 calibrated flat seems shit! Try with just a 

zwfs.deactive_cropping() 
zwfs.set_camera_dit( 0.001 );time.sleep(0.2)
zwfs.set_camera_fps( 200 );time.sleep(0.2)
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
bad_pixels = zwfs.get_bad_pixel_indicies( no_frames = 100, std_threshold = 50 , flatten=False)

# update zwfs bad pixel mask and flattened pixel values 
zwfs.build_bad_pixel_mask( bad_pixels , set_bad_pixels_to = 0 )

## ------- move source back in 
source_selection.set_source(  source_name )
time.sleep(2)

# quick check that dark subtraction works and we have signal
I0 = zwfs.get_image( apply_manual_reduction  = True)
plt.figure(); plt.title('test image \nwith dark subtraction \nand bad pixel mask'); plt.imshow( I0 ); plt.colorbar()
plt.savefig( fig_path + 'delme.png')
plt.close()

print_current_state()


## ------- Checking centering 

# ensure flat DM 
zwfs.dm.send_data(zwfs.dm_shapes['flat_dm'])

# == manual search 

phasemask_centering_tool.move_relative_and_get_image(zwfs, phasemask, savefigName=fig_path + 'delme.png')

# == init pupil region classification  
#init our pupil controller (object that processes ZWFS images and outputs VCM commands)
pupil_ctrl = pupil_control.pupil_controller_1(config_file = None)

#analyse pupil and decide if it is ok. This must be done before reconstructor
pupil_report = pupil_control.analyse_pupil_openloop( zwfs, debug = False, return_report = True, symmetric_pupil=False, std_below_med_threshold=1. )

if pupil_report['pupil_quality_flag'] == 1: 
    zwfs.update_reference_regions_in_img( pupil_report ) # 

# get a zernike basis 
zernike_basis =  util.construct_command_basis( basis='Zernike_pinned_edges', number_of_modes = 40, Nx_act_DM = 12, Nx_act_basis = 12, act_offset=(0,0), without_piston=True)


# --- linear ramps 
# use baldr.

zwfs.dm.send_data( zwfs.dm_shapes['flat_dm'] ); 

recon_data = util.GET_BDR_RECON_DATA_INTERNAL(zwfs, number_amp_samples = 10, amp_max = 0.1,\
number_images_recorded_per_cmd = 10, save_fits = data_path+f'pokeramp_data_MASK_{phasemask_name}_sydney_{tstamp}.fits') 
# recon_data = fits.open( data_path+'recon_data_LARGE_SECONDARY_19-04-2024T12.19.22.fits' )

M2C = util.construct_command_basis( basis='Zernike', number_of_modes = 5, Nx_act_DM = 12,Nx_act_basis =10, act_offset= (0,0), without_piston=True) 
active_dm_actuator_filter = (abs(np.sum( M2C, axis=1 )) > 0 ).astype(bool)                           


plt.figure(figsize=(8,5))
plt.title( 'active_DM_filter')
plt.imshow(util.get_DM_command_in_2D( active_dm_actuator_filter))
plt.colorbar();
#plt.imshow( zwfs.pupil_pixel_filter.reshape(zwfs.I0.shape) * np.sum( P2C_1x1 ,axis=0).reshape(zwfs.I0.shape) ) < - to see a clearer one
plt.savefig( fig_path + 'active_DM_filter_small.png')

zonal_fits = util.PROCESS_BDR_RECON_DATA_INTERNAL(recon_data, bad_pixels = np.where( zwfs.bad_pixel_filter.reshape(zwfs.I0.shape)),\
                                                   active_dm_actuator_filter=active_dm_actuator_filter, debug=False, \
                                                    fig_path = fig_path , poke_amplitude_indx=5, savefits= data_path+f'fitted_pokeramp_data_MASK_{phasemask_name}_sydney_{tstamp}.fits') 

# Check still well centered 
phasemask_centering_tool.move_relative_and_get_image(zwfs, phasemask, savefigName=fig_path + 'delme.png')

#### TESSTING SINGLE ITERATION 
# get residuals from 
#calibrated single pixel to actuator mapping 
P2C_1x1  = zonal_fits['P2C'].data[0] #

#anything registered outside pupil set to zero
P2C_1x1[:, ~zwfs.pupil_pixel_filter.ravel()] = 0

# have a look at the registration in the pixel space 
plt.figure(figsize=(8,5))
plt.title( 'P2C\nregistration in pixel space')
plt.imshow( np.sum( P2C_1x1 ,axis=0).reshape(zwfs.I0.shape) ) 
plt.colorbar();
#plt.imshow( zwfs.pupil_pixel_filter.reshape(zwfs.I0.shape) * np.sum( P2C_1x1 ,axis=0).reshape(zwfs.I0.shape) ) < - to see a clearer one
plt.savefig( fig_path + 'P2C_pixels_registered.png')


# reference intensities 
I0, _ = util.get_reference_images(zwfs, phasemask, theta_degrees=11.8, number_of_frames=256, \
compass = True, compass_origin=None, savefig= fig_path + f'initial_reference_pupils.png' )


# get a series of off mask images 
phasemask.move_relative( [ 200, 200 ] )

N0_list = zwfs.get_some_frames(number_of_frames=1000, apply_manual_reduction=True)

phasemask.move_relative( [ -200, -200 ] )

N0 = np.mean( N0_list ,axis=0)

plt.figure(); plt.imshow( N0) ;plt.savefig(fig_path + 'delme.png')

# check pupil and its registration
fig,ax = plt.subplots(1,2,figsize=(10,5))
ax[0].imshow( N0 ) ; ax[0].set_title('phasemask out')
ax[1].imshow( zwfs.pupil_pixel_filter.reshape(N0.shape)) ; ax[1].set_title('registered pupil')
plt.savefig(fig_path + f'registered_pupil_{phasemask_name}_{tstamp}.png')






# Final test


exper_path = 'flatcal_using_measN0_in_theory_3_at900nm_smaller_DM/'

if not os.path.exists(fig_path + exper_path):
   os.makedirs(fig_path + exper_path)



## ------- Get theoretical reference intensity with phasemask in 

#estimated_strehl = np.max(  I0/np.max(N0) ) / np.max( I0_theory/np.max(N0_theory) )
#print(  f'estimated strehl at {wvl}= {estimated_strehl}' )


# sydney Baldr using DMLP1180 Longpass Dichroic Mirrors/Beamsplitters: 1180 nm Cut-Off Wavelength (Baldr in reflection)
# C-RED 2 cut on at 900nm 
# black body source at 1900 K 
# therefore central wavelength
central_lambda = util.find_central_wavelength(lambda_cut_on=900e-9, lambda_cut_off=1180e-9, T=1900)
print(f"The central wavelength is {central_lambda * 1e9:.2f} nm")


wvl =  0.9 #1e6 * central_lambda # 0.900 #1.040 # um  
phase_shift = util.get_phasemask_phaseshift( wvl= wvl, depth = phasemask.phasemask_parameters[phasemask_name]['depth'] )
mask_diam = 1e-6 * phasemask.phasemask_parameters[phasemask_name]['diameter']
"""N0_theory0, I0_theory0 = util.get_theoretical_reference_pupils( wavelength = wvl*1e-6 ,F_number = 21.2, mask_diam = mask_diam,\
                                        diameter_in_angular_units = False,  phaseshift = phase_shift , padding_factor = 4, \
                                        debug= False, analytic_solution = True )
"""
# get_individual_terms=False, so don't include measured N0 (FPM OUT)
N0_theory0, I0_theory0 = util.get_theoretical_reference_pupils( wavelength = wvl*1e-6 ,F_number = 21.2, mask_diam = mask_diam,\
                                        get_individual_terms=False, diameter_in_angular_units = False,  phaseshift = phase_shift , padding_factor = 4, \
                                        debug= False, analytic_solution = True )

# for including the measured N0 (FPM out)
P_theory0, M_theory0, mu_theory0 = util.get_theoretical_reference_pupils( wavelength = wvl*1e-6 ,F_number = 21.2, mask_diam = mask_diam,\
                                        get_individual_terms=True, diameter_in_angular_units = False,  phaseshift = phase_shift , padding_factor = 4, \
                                        debug= False, analytic_solution = True )

M = P_theory0.shape[0]
N = P_theory0.shape[1]

m = zwfs.I0.shape[1]
n = zwfs.I0.shape[0]

# A = pi * r^2 => r = sqrt( A / pi)
new_radius = (zwfs.pupil_pixel_filter.sum()/np.pi)**0.5
x_c = np.mean( np.unravel_index( np.where( zwfs.pupil_pixel_filter ), zwfs.I0.shape)[1])
y_c = np.mean( np.unravel_index( np.where( zwfs.pupil_pixel_filter ), zwfs.I0.shape)[0])

P_theory = util.interpolate_pupil_to_measurement( N0_theory0, I0_theory0, M, N, m, n, x_c, y_c, new_radius)

M_theory = util.interpolate_pupil_to_measurement( N0_theory0, M_theory0, M, N, m, n, x_c, y_c, new_radius)

mu_theory = util.interpolate_pupil_to_measurement( N0_theory0, mu_theory0, M, N, m, n, x_c, y_c, new_radius)


# with perfect pupil 
I0_theory = ( P_theory**2 + abs(M_theory)**2 + 2* P_theory * abs(M_theory) * np.cos(mu_theory ) )
N0_theory = P_theory**2 

im_list = [I0_theory/np.max(I0_theory) , I0/np.max(I0), I0_theory/np.max(I0_theory) - I0/np.max(I0)]
xlabel_list = [None, None, None]
ylabel_list = [None, None, None]
title_list = ['theory', 'measured', 'residual']
cbar_label_list = [r'normalized intensity',r'normalized intensity', r'normalized intensity'] 
savefig = fig_path + exper_path + f'I0_theory_vs_meas_mask-{phasemask_name}.png' #f'mode_reconstruction_images/phase_reconstruction_example_mode-{mode_indx}_basis-{phase_ctrl.config["basis"]}_ctrl_modes-{phase_ctrl.config["number_of_controlled_modes"]}ctrl_act_diam-{phase_ctrl.config["dm_control_diameter"]}_readout_mode-12x12.png'

util.nice_heatmap_subplots( im_list , xlabel_list, ylabel_list, title_list, cbar_label_list, fontsize=15, axis_off=True, cbar_orientation = 'bottom', savefig=savefig)

# using the measured FPM out intensity

# first scale measured reference pupil appropiately
N0_meas = N0 / np.mean( N0 ) * np.mean( P_theory**2 )
N0_meas[N0_meas < 0] = 0 # enforce non-negative for sqrt
 
I0_theory_w_meas = ( N0_meas + abs(M_theory)**2 + 2* np.sqrt(N0_meas) * abs(M_theory) * np.cos(mu_theory ) )
I0_theory_w_meas.reshape(-1)[zwfs.bad_pixel_filter] = 0

im_list = [I0_theory_w_meas/np.max(I0_theory_w_meas) , I0/np.max(I0), I0_theory_w_meas/np.max(I0_theory_w_meas) - I0/np.max(I0)]
xlabel_list = [None, None, None]
ylabel_list = [None, None, None]
title_list = ['theory', 'measured', 'residual']
cbar_label_list = [r'normalized intensity',r'normalized intensity', r'normalized intensity'] 
savefig = fig_path + exper_path +  f'I0_theory_w_measP_vs_meas_mask-{phasemask_name}.png' #f'mode_reconstruction_images/phase_reconstruction_example_mode-{mode_indx}_basis-{phase_ctrl.config["basis"]}_ctrl_modes-{phase_ctrl.config["number_of_controlled_modes"]}ctrl_act_diam-{phase_ctrl.config["dm_control_diameter"]}_readout_mode-12x12.png'

util.nice_heatmap_subplots( im_list , xlabel_list, ylabel_list, title_list, cbar_label_list, fontsize=15, axis_off=True, cbar_orientation = 'bottom', savefig=savefig)










zwfs.dm.send_data( zwfs.dm_shapes['flat_dm'] ); 

phasemask_centering_tool.move_relative_and_get_image(zwfs, phasemask, savefigName = fig_path + 'delme.png')


I0_before, N0_before = util.get_reference_images(zwfs, phasemask, theta_degrees=11.8, number_of_frames=256, \
    compass = True, compass_origin=None, savefig= fig_path + exper_path + f'initial_reference_pupils.png' )


#pupil_outer_perim_filter = ( (abs( I0 - N0 ) > 0.1 ).reshape(-1) * (~zwfs.pupil_pixel_filter) )
pupil_outer_perim_filter = ( (abs( I0_theory - N0_theory ) > 0.02 ).reshape(-1) * (~zwfs.pupil_pixel_filter) * (~zwfs.bad_pixel_filter) )
# seems better to use the theory one
plt.figure()
plt.imshow( util.get_DM_command_in_2D( P2C_1x1.sum(axis=1)) ) ; plt.savefig(fig_path + exper_path + 'registered_on_DM.png')

plt.figure()
plt.imshow( pupil_outer_perim_filter.reshape(zwfs.I0.shape) ) 
plt.savefig( fig_path + exper_path + 'outer_pupil_filter.png')



rmse_list = [] 
diff_field_list = []
cmd_offset_list = []
cmd_list = []
I0_a_list = []
N0_a_list = []
amp = 0.03 / 5
cmd_offset = np.zeros( 140 )
cmd = zwfs.dm_shapes['flat_dm']
for it in range(15):
    print( it )
    cmd = cmd + amp * cmd_offset

    zwfs.dm.send_data( cmd )
    
    time.sleep(0.1)

    #I0_a, N0_a = util.get_reference_images(zwfs, phasemask, theta_degrees=11.8, number_of_frames=256, \
    #compass = True, compass_origin=None, savefig= None )

    I0_a = np.mean( zwfs.get_some_frames( 256, apply_manual_reduction=True), axis=0 ) 
    #I0_a_list.append( I0_a )
    #N0_a_list.append( N0_a )
    # have to normalize correctly (they N0_meas was normalized prior for I0_theory_w_meas)
    #residual = I0_theory_w_meas / np.max(N0_meas) - I0_a / np.max(N0) #I0_theory/N0_theory - I0_a/N0_a #I0_theory/np.max(I0_theory) - I0_a/np.max(I0_a)
    
    # theory
    i_t_norm = I0_theory_w_meas / np.mean(N0_meas.ravel()[zwfs.pupil_pixels])
    
    # measured
    i_m_norm = I0_a / np.mean(N0.ravel()[zwfs.pupil_pixels]) #I0_theory/N0_theory - I0_a/N0_a #I0_theory/np.max(I0_theory) - I0_a/np.max(I0_a)
    
    residual = i_t_norm - i_m_norm
    
    filt = zwfs.pupil_pixel_filter & np.isfinite(residual.reshape(-1))
    rmse = np.sqrt( np.mean( residual.reshape(-1)[filt]**2 )) # only look at residuals within pupil (outside they go crazy since N0 -> 0)

    print( rmse )
    # when using inverse of N0 have to ensure pixels outside registered pupil are zero (otherwise they blow up)
    cmd_offset = P2C_1x1 @ ( np.nan_to_num( residual.reshape(-1) * filt,0 ) )  
    
    d = util.get_DM_command_in_2D( cmd_offset )[1:-1,1:-1]


    # cmd_offset_raw = 
    # normalize between 0-1
    ###cmd_offset = (cmd_offset_raw - np.nanmin( cmd_offset_raw ) ) / (np.nanmax(  cmd_offset_raw ) - np.nanmin(  cmd_offset_raw ))
    #cmd_offset_n = (cmd_offset_raw - np.nanmin( cmd_offset_raw ) ) / (np.nanmax(  cmd_offset_raw ) - np.nanmin(  cmd_offset_raw ))

    #cmd_offset_unNorm = cmd_offset + cmd_offset_n 
    
    #cmd_offset = (cmd_offset_unNorm - np.nanmin( cmd_offset_unNorm ) ) / (np.nanmax(  cmd_offset_unNorm ) - np.nanmin(  cmd_offset_unNorm ))

    cmd_offset_list.append( cmd_offset )
    cmd_list.append( cmd ) 

    im_list = [i_t_norm , i_m_norm, residual, util.get_DM_command_in_2D( cmd ) ]
    xlabel_list = [None for _ in im_list]
    ylabel_list = [None for _ in im_list]
    title_list = ['theory', 'measured', 'residual', 'command offset']
    cbar_label_list = [r'normalized intensity',r'normalized intensity', r'normalized intensity', 'mapped to actuators'] 
    savefig = fig_path + exper_path + f'calibrating_reference_I0_mask-{phasemask_name}_theory_v_meas_it{it}.png' # 'focus_test/' + f'I0_theory_v_meas_focusOffset-{round(i,2)}_mask-{phasemask_name}.png' #f'mode_reconstruction_images/phase_reconstruction_example_mode-{mode_indx}_basis-{phase_ctrl.config["basis"]}_ctrl_modes-{phase_ctrl.config["number_of_controlled_modes"]}ctrl_act_diam-{phase_ctrl.config["dm_control_diameter"]}_readout_mode-12x12.png'

    util.nice_heatmap_subplots( im_list , xlabel_list, ylabel_list, title_list, cbar_label_list, fontsize=15, axis_off=True, cbar_orientation = 'bottom', savefig=savefig)

    rmse_list.append( rmse)
    # light diffracted outside pupil on edge
    diff_field_list.append( np.var( I0_a.reshape(-1)[pupil_outer_perim_filter]) )
    
    #_ = input(f'it = {it}, rms = {rmse}, go to next?')



I0_after, N0_after = util.get_reference_images(zwfs, phasemask, theta_degrees=11.8, number_of_frames=256, \
compass = True, compass_origin=None, savefig= fig_path + exper_path + f'final_reference_pupils.png' )

plt.figure(figsize=(8,5));  
plt.plot( np.array( diff_field_list)/np.max(diff_field_list), label=r'$\sigma^2_{lobe}$')
plt.plot( np.array(rmse_list)/np.max( rmse_list), label='RMSE')
plt.xlabel('iterations',fontsize=15)
plt.ylabel('Normalized metric',fontsize=15)
plt.legend(fontsize=15)
plt.gca().tick_params(labelsize=15)
plt.savefig( fig_path + exper_path + f'convergence_metrics_{phasemask_name}.png')


zwfs.dm.send_data( zwfs.dm_shapes['flat_dm']  ) 

# best cmd offset
best_cmd = cmd_list[np.argmin( rmse_list)] # [ np.argmax( diff_field_list ) ] # zwfs.dm_shapes['flat_dm'] + amp * cmd_offset_list
pd.Series( best_cmd ).to_csv( fig_path + exper_path + f'calibrated_DMflat_{tstamp}.csv' )

write_list = [I0_a_list, N0_a_list, I0_theory, cmd_list, diff_field_list, rmse_list, P2C_1x1,  filt.astype(int), pupil_outer_perim_filter.astype(int)]
write_list_label = ["I0_a_list", "N0_a_list", "cmd_list", "diff_field_list", "rmse_list", "P2C_1x1", "pupil_filt", "pupil_outer_perim_filter"]

flattening_dm_fits = fits.HDUList( [] )
for i,lab in zip(write_list, write_list_label):
    tmp_fits = fits.PrimaryHDU( i )
    tmp_fits.header.set('EXTNAME',lab)
    flattening_dm_fits.append( tmp_fits )

flattening_dm_fits.writeto(fig_path + exper_path + f'flattening_DM_{tstamp}.fits', overwrite=False) 





I0_after, N0_after = util.get_reference_images(zwfs, phasemask, theta_degrees=11.8, number_of_frames=256, \
compass = True, compass_origin=None, savefig= fig_path + exper_path + f'final_reference_pupils_manual_adj_2.png' )





#zwfs.dm_shapes['flat_dm_original'] = zwfs.dm_shapes['flat_dm'].copy()
#zwfs.dm_shapes['flat_dm'] = best_cmd


# NOW TRY MANUALA ADJUSTMENT for fine adjustment 
#cmd_man_adj = util.shape_dm_manually(zwfs, compass = True , initial_cmd = None ,number_of_frames=5, apply_manual_reduction=True,\
#                   theta_degrees=11.8, flip_dm = True, savefig=fig_path + exper_path + 'manual_adjustment.png') #'manual_adjustment.png')


#pd.Series( cmd_man_adj ).to_csv( fig_path + exper_path + f'BEST_calibrated_DMflat_{tstamp}.csv' )






#zwfs.dm.send_data(  zwfs.dm_shapes['flat_dm'])


plt.figure()
plt.imshow( np.flipud( util.get_DM_command_in_2D( P2C_1x1 @ I0.reshape(-1)/np.max( I0.reshape(-1) )) ))
plt.title( 'P2C @ I0')
plt.colorbar() 
plt.savefig(fig_path + exper_path + f'P2C-I_mask{phasemask_name}.png')




plt.figure()
plt.imshow( np.flipud( util.get_DM_command_in_2D( P2C_1x1 @ (I0.reshape(-1)/np.max( I0.reshape(-1) ) - I0_theory.reshape(-1)/np.max(I0_theory)) )) )
plt.title( 'P2C @ (I0-I0_theory)')
plt.colorbar() 
plt.savefig(fig_path + exper_path + f'P2C-I_Itheory_mask{phasemask_name}.png')










#%% OLDER STUFF / TESTS BELOW 



# apply aberration and look at how residuals change
zwfs.dm.send_data( zwfs.dm_shapes['flat_dm'] + 0.2 * zernike_basis.T[1]); 
time.sleep(0.5)

I0_a, N0_a = util.get_reference_images(zwfs, phasemask, theta_degrees=11.8, number_of_frames=256, \
compass = True, compass_origin=None, savefig= None )

im_list = [I0_theory/np.max(I0_theory) , I0_a/np.max(I0_a), I0_theory/np.max(I0_theory) - I0_a/np.max(I0_a)]
xlabel_list = [None, None, None]
ylabel_list = [None, None, None]
title_list = ['theory', 'measured', 'residual']
cbar_label_list = [r'normalized intensity',r'normalized intensity', r'normalized intensity'] 
savefig = fig_path + f'I0_theory_vs_meas_WITH_ABERRATION_mask-{phasemask_name}.png' #f'mode_reconstruction_images/phase_reconstruction_example_mode-{mode_indx}_basis-{phase_ctrl.config["basis"]}_ctrl_modes-{phase_ctrl.config["number_of_controlled_modes"]}ctrl_act_diam-{phase_ctrl.config["dm_control_diameter"]}_readout_mode-12x12.png'

util.nice_heatmap_subplots( im_list , xlabel_list, ylabel_list, title_list, cbar_label_list, fontsize=15, axis_off=True, cbar_orientation = 'bottom', savefig=savefig)


zwfs.dm.send_data( zwfs.dm_shapes['flat_dm'] ); 


# plot as a function of focus 
# check which focus is best 

if not os.path.exists(fig_path + 'focus_test/'):
   os.makedirs(fig_path + 'focus_test/')

# check focus to apply 
#plt.figure(); plt.imshow( util.get_DM_command_in_2D(zernike_basis.T[2]) ) ; plt.savefig(fig_path + 'delme.png' )

rmse_list = []
for i in np.linspace(-1,1,15): 
    zwfs.dm.send_data( zwfs.dm_shapes['flat_dm'] + i * zernike_basis.T[2]); 
    time.sleep(0.1)

    I0_a, N0_a = util.get_reference_images(zwfs, phasemask, theta_degrees=11.8, number_of_frames=256, \
    compass = True, compass_origin=None, savefig= None )

    rmse = np.sqrt( np.mean( (I0_theory/np.max(I0_theory) - I0_a/np.max(I0_a))**2 ))

    im_list = [I0_theory/np.max(I0_theory) , I0_a/np.max(I0_a), I0_theory/np.max(I0_theory) - I0_a/np.max(I0_a)]
    xlabel_list = [None, None, None]
    ylabel_list = [None, None, None]
    title_list = ['theory', 'measured', f'residual (rmse = {round(rmse, 2)})']
    cbar_label_list = [r'normalized intensity',r'normalized intensity', r'normalized intensity'] 
    savefig = fig_path + 'focus_test/' + f'I0_theory_v_meas_focusOffset-{round(i,2)}_mask-{phasemask_name}.png' #f'mode_reconstruction_images/phase_reconstruction_example_mode-{mode_indx}_basis-{phase_ctrl.config["basis"]}_ctrl_modes-{phase_ctrl.config["number_of_controlled_modes"]}ctrl_act_diam-{phase_ctrl.config["dm_control_diameter"]}_readout_mode-12x12.png'

    util.nice_heatmap_subplots( im_list , xlabel_list, ylabel_list, title_list, cbar_label_list, fontsize=15, axis_off=True, cbar_orientation = 'bottom', savefig=savefig)

    rmse_list.append( rmse)
    _ = input(f'a_2 = {i}, rms = {rmse}, go to next?')

zwfs.dm.send_data( zwfs.dm_shapes['flat_dm'] ); 



I0, N0 = util.get_reference_images(zwfs, phasemask, theta_degrees=11.8, number_of_frames=256, \
compass = True, compass_origin=None, savefig= fig_path + f'initial_reference_pupils.png' )


residual =  I0_theory/np.max(I0_theory) - I0/np.max(I0)

cmd_offset_raw = P2C_1x1 @ residual.reshape(-1) 

# normalize between 0-1
cmd_offset = (cmd_offset_raw - np.nanmin( cmd_offset_raw ) ) / (np.nanmax(  cmd_offset_raw ) - np.nanmin(  cmd_offset_raw ))

im_list = [I0_theory/np.max(I0_theory) , I0/np.max(I0), residual, util.get_DM_command_in_2D( cmd_offset) ]
xlabel_list = [None for _ in im_list]
ylabel_list = [None for _ in im_list]
title_list = ['theory', 'measured', 'residual', 'command offset']
cbar_label_list = [r'normalized intensity',r'normalized intensity', r'normalized intensity', 'mapped to actuators'] 
savefig = fig_path + f'command_proposition-{phasemask_name}.png' #f'mode_reconstruction_images/phase_reconstruction_example_mode-{mode_indx}_basis-{phase_ctrl.config["basis"]}_ctrl_modes-{phase_ctrl.config["number_of_controlled_modes"]}ctrl_act_diam-{phase_ctrl.config["dm_control_diameter"]}_readout_mode-12x12.png'

util.nice_heatmap_subplots( im_list , xlabel_list, ylabel_list, title_list, cbar_label_list, fontsize=15, axis_off=True, cbar_orientation = 'bottom', savefig=savefig)


## 

# Testing different amplitudes of feedback single iteration 
# re-align 
phasemask_centering_tool.move_relative_and_get_image(zwfs, phasemask, savefigName=fig_path + 'delme.png')

if not os.path.exists(fig_path + 'newflat_test/'):
   os.makedirs(fig_path + 'newflat_test/')


rmse_list = [] 
for amp in np.linspace(-0.05, 0.05, 10 ): 
    zwfs.dm.send_data( zwfs.dm_shapes['flat_dm'] + amp * cmd_offset); 
    time.sleep(0.1)

    I0_a, N0_a = util.get_reference_images(zwfs, phasemask, theta_degrees=11.8, number_of_frames=256, \
    compass = True, compass_origin=None, savefig= None )

    rmse = np.sqrt( np.mean( (I0_theory/np.max(I0_theory) - I0_a/np.max(I0_a))**2 ))

    im_list = [I0_theory/np.max(I0_theory) , I0_a/np.max(I0_a), I0_theory/np.max(I0_theory) - I0_a/np.max(I0_a)]
    xlabel_list = [None, None, None]
    ylabel_list = [None, None, None]
    title_list = ['theory', 'measured', f'residual (rmse = {round(rmse, 2)})']
    cbar_label_list = [r'normalized intensity',r'normalized intensity', r'normalized intensity'] 
    savefig = fig_path +'newflat_test/' + 'delme.png' # 'focus_test/' + f'I0_theory_v_meas_focusOffset-{round(i,2)}_mask-{phasemask_name}.png' #f'mode_reconstruction_images/phase_reconstruction_example_mode-{mode_indx}_basis-{phase_ctrl.config["basis"]}_ctrl_modes-{phase_ctrl.config["number_of_controlled_modes"]}ctrl_act_diam-{phase_ctrl.config["dm_control_diameter"]}_readout_mode-12x12.png'

    util.nice_heatmap_subplots( im_list , xlabel_list, ylabel_list, title_list, cbar_label_list, fontsize=15, axis_off=True, cbar_orientation = 'bottom', savefig=savefig)

    rmse_list.append( rmse)
    _ = input(f'a_2 = {amp}, rms = {rmse}, go to next?')




# DO it faster without plotting 

zwfs.dm.send_data( zwfs.dm_shapes['flat_dm']  ) 

phasemask_centering_tool.move_relative_and_get_image(zwfs, phasemask, savefigName = fig_path + 'delme.png')

I0, N0 = util.get_reference_images(zwfs, phasemask, theta_degrees=11.8, number_of_frames=256, \
compass = True, compass_origin=None, savefig= fig_path +'newflat_test/' + f'initial_reference_pupils.png' )



#plt.figure(); plt.imshow(( (abs( I0 - N0 ) > 0.1 ).reshape(-1) * (~zwfs.pupil_pixel_filter) ) .reshape(I0.shape) ); plt.savefig(fig_path + 'delme.png')
#plt.figure(); plt.imshow(( (abs( I0_theory - N0_theory ) > 0.1 ).reshape(-1) * (~zwfs.pupil_pixel_filter) ) .reshape(I0.shape) ); plt.savefig(fig_path + 'delme.png')

#pupil_outer_perim_filter = ( (abs( I0 - N0 ) > 0.1 ).reshape(-1) * (~zwfs.pupil_pixel_filter) )
pupil_outer_perim_filter = ( (abs( I0_theory - N0_theory ) > 0.1 ).reshape(-1) * (~zwfs.pupil_pixel_filter) )
# seems better to use the theory one

rmse_list = [] 
diff_field_list = []
amp = 0.03
cmd_offset = np.zeros( 140 )
no_its = 20
for it in range(no_its):
    print(it) 
    zwfs.dm.send_data( zwfs.dm_shapes['flat_dm'] + amp * cmd_offset); 
    time.sleep(0.1)

    I0_a, N0_a = util.get_reference_images(zwfs, phasemask, theta_degrees=11.8, number_of_frames=50, \
    compass = True, compass_origin=None, savefig= None )

    residual = I0_theory/np.max(I0_theory) - I0_a/np.max(I0_a)
    rmse = np.sqrt( np.mean( residual**2 ))

    rmse_list.append( rmse )
    diff_field_list.append( I0_a.reshape(-1)[pupil_outer_perim_filter])
    
    cmd_offset_raw = P2C_1x1 @ residual.reshape(-1) 

    # normalize between 0-1
    cmd_offset_n = (cmd_offset_raw - np.nanmin( cmd_offset_raw ) ) / (np.nanmax(  cmd_offset_raw ) - np.nanmin(  cmd_offset_raw ))

    cmd_offset = cmd_offset + cmd_offset_n 

    maxcmd = np.max(abs( cmd_offset * amp ) )
 
    if (maxcmd > 0.4) : 
        break
    #print( )
    #_ = input(f'it = {it}, rms = {rmse}, go to next?')



I0, N0 = util.get_reference_images(zwfs, phasemask, theta_degrees=11.8, number_of_frames=256, \
compass = True, compass_origin=None, savefig= fig_path + f'final_reference_pupils_after_{no_its}_iterations.png' )





# testing measuring Strehl on outside of pupil 




pupil_outer_perim_filter = ( (abs( I0 - N0 ) > 0.1 ).reshape(-1) * (~zwfs.pupil_pixel_filter) )
pupil_outer_perim_filter2 = ( (abs( I0_theory - N0_theory ) > 0.1 ).reshape(-1) * (~zwfs.pupil_pixel_filter) )


std_grid = np.linspace(0,0.1,10)
cnt_list_1 = {a:[] for a in std_grid}
cnt_list_2 = {a:[] for a in std_grid}
for a in std_grid:
    for _ in range(10): 
        zwfs.dm.send_data( zwfs.dm_shapes['flat_dm'] + a * np.random.randn(140) )
        time.sleep( 0.1 )
        img = np.mean( zwfs.get_some_frames( number_of_frames=20) ,axis=0)

        cnt_list_1[a].append( np.sum( img.reshape(-1)[pupil_outer_perim_filter] ))
        cnt_list_2[a].append( np.sum( img.reshape(-1)[pupil_outer_perim_filter2] ))


plt.figure(); 
plt.plot(std_grid, [np.mean( cnt_list_1[a]) for a in cnt_list_1] ,label='1'); 
plt.plot(std_grid, [np.mean( cnt_list_2[a]) for a in cnt_list_2]); 
plt.legend(); plt.savefig(fig_path + 'delme.png')