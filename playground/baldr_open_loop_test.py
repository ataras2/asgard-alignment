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
sys.path.append('simBaldr/' )
sys.path.append('pyBaldr/' )  
sys.path.insert(1, '/opt/FirstLightImaging/FliSdk/Python/demo/')
sys.path.insert(1,'/opt/Boston Micromachines/lib/Python3/site-packages/')

from pyBaldr import utilities as util
from pyBaldr import ZWFS
from pyBaldr import phase_control
from pyBaldr import pupil_control

import bmc
import FliSdk_V2

from zaber_motion.ascii import Connection
from asgard_alignment.ZaberMotor import BaldrPhaseMask, LAC10AT4A

# timestamp
tstamp = datetime.datetime.now().strftime("%d-%m-%YT%H.%M.%S")

fig_path = 'tmp/' #'/home/baldr/Documents/baldr/ANU_demo_scripts/BALDR/figures/' 
data_path = 'tmp/' #'/home/baldr/Documents/baldr/ANU_demo_scripts/BALDR/data/' 

# ====== hardware variables
beam = 3
phasemask_name = 'J3'
phasemask_OUT_offset = [1000,1000]  # relative offset (um) to take phasemask out of beam
BFO_pos = 4000 # um (absolute position of detector imgaging lens) 
dichroic_name = "J"
source_name = 'STL'
DM_serial_number = '17DW019#122' # Syd = '17DW019#122', ANU = '17DW019#053'


# ======  set up source 

# start with source out !

# ======  set up dichroic 

# do manually (COM3 communication issue)
"""
#  ConnectionFailedException: ConnectionFailedException: Cannot open serial port: no such file or directory

connection = Connection.open_serial_port("COM3")
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
    pass"""

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
print( phasemask.get_position() )
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
print(focus_motor.get_position())




# ====== Set up and calibrate 

debug = True # plot some intermediate results 


pupil_crop_region = [204,268,125, 187] #[None, None, None, None] #[0, 192, 0, 192] 

#init our ZWFS (object that interacts with camera and DM) (old path = home/baldr/Documents/baldr/ANU_demo_scripts/BALDR/)
zwfs = ZWFS.ZWFS(DM_serial_number=DM_serial_number, cameraIndex=0, DMshapes_path = 'DMShapes/', pupil_crop_region=pupil_crop_region ) 

# the sydney BMC multi-3.5 calibrated flat seems shit! Try with just a 

zwfs.set_camera_dit( 0.001 );time.sleep(0.2)
zwfs.set_camera_fps( 200 );time.sleep(0.2)
zwfs.set_sensitivity('high');time.sleep(0.2)
zwfs.enable_frame_tag(tag = True);time.sleep(0.2)
zwfs.bias_off();time.sleep(0.2)
zwfs.flat_off();time.sleep(0.2)

zwfs.dm_shapes['flat_dm'] = 0.5 * np.ones(140)

zwfs.start_camera()

# !!!! TAKE OUT SOURCE !!!! 
# at sydney move 01 X-LSM150A-SE03 to 133.07mm
zwfs.build_manual_dark()

# get our bad pixels 
bad_pixels = zwfs.get_bad_pixel_indicies( no_frames = 1000, std_threshold = 50 , flatten=False)

# update zwfs bad pixel mask and flattened pixel values 
zwfs.build_bad_pixel_mask( bad_pixels , set_bad_pixels_to = 0 )

plt.figure() ; plt.title('dark'); plt.imshow( zwfs.reduction_dict['dark'][0] ); plt.colorbar() ; plt.savefig(fig_path + 'delme.png' )

plt.figure() ;  plt.title('bad pixels'); plt.imshow( zwfs.bad_pixel_filter.reshape(zwfs.reduction_dict['dark'][0].shape) ); plt.savefig(fig_path + 'delme.png' )


# !!!! PUT IN SOURCE !!!! 

# quick check that dark subtraction works
I0 = zwfs.get_image( apply_manual_reduction  = True)
plt.figure(); plt.title('test image \nwith dark subtraction \nand bad pixel mask'); plt.imshow( I0 ); plt.colorbar()
plt.savefig( fig_path + 'delme.png')


zwfs.dm.send_data(zwfs.dm_shapes['flat_dm'])


# ====== testing reconstruction 
#init our phase controller (object that processes ZWFS images and outputs DM commands)
phase_ctrl = phase_control.phase_controller_1(config_file = None) 
# to change basis : 
#phase_ctrl.change_control_basis_parameters( controller_label = ctrl_method_label, number_of_controlled_modes=phase_ctrl.config['number_of_controlled_modes'], basis_name='Zonal' , dm_control_diameter=None, dm_control_center=None)

#init our pupil controller (object that processes ZWFS images and outputs VCM commands)
pupil_ctrl = pupil_control.pupil_controller_1(config_file = None)

#analyse pupil and decide if it is ok. This must be done before reconstructor
pupil_report = pupil_control.analyse_pupil_openloop( zwfs, debug = True, return_report = True)

if pupil_report['pupil_quality_flag'] == 1: 
    zwfs.update_reference_regions_in_img( pupil_report ) # 




# --- linear ramps 
# use baldr.
recon_data = util.GET_BDR_RECON_DATA_INTERNAL(zwfs, number_amp_samples = 20, amp_max = 0.2,\
number_images_recorded_per_cmd = 4, save_fits = data_path+f'pokeramp_data_MASK_J3_sydney_{tstamp}.fits') 
#recon_data = fits.open( data_path+'recon_data_LARGE_SECONDARY_19-04-2024T12.19.22.fits' )



zwfs.pupil_pixel_filter = ~zwfs.bad_pixel_filter
zwfs.pupil_pixels = np.where( ~zwfs.bad_pixel_filter )[0]

ctrl_method_label = 'ctrl_1'
zwfs.dm.send_data( zwfs.dm_shapes['flat_dm'] )
time.sleep( 0.1 )
# TRY model_2 WITH  method='single_side_poke', or 'double_sided_poke'
#phase_ctrl.change_control_basis_parameters(  number_of_controlled_modes=140, basis_name ='Zonal', dm_control_diameter=None, dm_control_center=None,controller_label=None)
#phase_ctrl.build_control_model_2(zwfs, poke_amp = -0.1, label='ctrl_1', poke_method='double_sided_poke', inverse_method='MAP',  debug = True)

phase_ctrl.build_control_model_2(zwfs, poke_amp = -0.3, label='ctrl_1', poke_method='double_sided_poke', inverse_method='MAP',  debug = True)
#phase_ctrl.build_control_model( zwfs , poke_amp = -0.15, label='ctrl_1', debug = True)  

#phase_ctrl.plot_SVD_modes( zwfs, 'ctrl_1', save_path=fig_path)


# write fits to input into RTC
zwfs.write_reco_fits( phase_ctrl, 'ctrl_1', save_path=data_path)



# can we reconstruct in open loop 
mode_basis = phase_ctrl.config['M2C']  # readability 
I2M = phase_ctrl.ctrl_parameters[ctrl_method_label]['I2M']
IM = phase_ctrl.ctrl_parameters[ctrl_method_label]['IM'] # readability 
# unfiltered CM
CM = phase_ctrl.ctrl_parameters[ctrl_method_label]['CM'] # readability 
R_TT = phase_ctrl.ctrl_parameters[ctrl_method_label]['R_TT'] # readability 
R_HO = phase_ctrl.ctrl_parameters[ctrl_method_label]['R_HO'] # readability 

M2C = phase_ctrl.ctrl_parameters[ctrl_method_label]['M2C_4reco'] # readability  # phase_ctrl.ctrl_parameters[ctrl_method_label]['M2C_4reco']#
I0 = phase_ctrl.ctrl_parameters[ctrl_method_label]['ref_pupil_FPM_in']

poke_amp = phase_ctrl.ctrl_parameters[ctrl_method_label]['poke_amp']
for mode_indx in range( len(M2C)-1 ) :  

    mode_aberration = mode_basis.T[mode_indx]#   M2C.T[mode_indx]
    #plt.imshow( util.get_DM_command_in_2D(amp*mode_aberration));plt.colorbar();plt.show()
    
    dm_cmd_aber = zwfs.dm_shapes['flat_dm'] + amp * mode_aberration 

    zwfs.dm.send_data( dm_cmd_aber )
    time.sleep(0.1)
    raw_img_list = []
    for i in range( 10 ) :
        raw_img_list.append( zwfs.get_image() ) # @D, remember for control_phase method this needs to be flattened and filtered for pupil region
    raw_img = np.median( raw_img_list, axis = 0) 
    # plt.figure() ; plt.imshow( raw_img ) ; plt.savefig( fig_path + f'delme.png') # <- signal?
    
    err_img = phase_ctrl.get_img_err( 1/np.mean(raw_img) * raw_img.reshape(-1)[zwfs.pupil_pixel_filter]  ) 
    # plt.figure() ; plt.hist( err_img, label='meas', alpha=0.3 ) ; plt.hist( IM[mode_indx] , label='from IM', alpha=0.3); plt.legend() ; plt.savefig( fig_path + f'delme.png') # <- should be around zeros

    #mode_res_test : inject err_img from interaction matrix to I2M .. should result in perfect reconstruction  
    #plt.figure(); plt.plot( I2M.T @ IM[2] ); plt.savefig( fig_path + f'delme.png')
    #plt.figure(); plt.plot( I2M.T @ IM[mode_indx]  ,label='reconstructed amplitude'); plt.axvline(mode_indx  , ls=':', color='k', label='mode applied') ; plt.xlabel('mode index'); plt.ylabel('mode amplitude'); plt.legend(); plt.savefig( fig_path + f'delme.png')
    mode_res =  I2M.T @ err_img 


    plt.figure(); plt.plot( mode_res ); plt.axvline(mode_indx  , ls=':', color='k') ; plt.savefig( fig_path + f'delme.png')
    plt.figure(figsize=(8,5));
    plt.plot( mode_res  ,label='reconstructed amplitude');
    app_amp = np.zeros( len( mode_res ) ) 

    app_amp[mode_indx] = amp / poke_amp

    plt.plot( app_amp ,'x', label='applied amplitude');
    plt.axvline(mode_indx  , ls=':', color='k', label='mode applied') ; plt.xlabel('mode index',fontsize=15); 
    plt.ylabel('mode amplitude',fontsize=15); plt.gca().tick_params(labelsize=15) ; plt.legend();
    plt.savefig( fig_path + f'delme.png')

    _ = input('press when ready to see mode reconstruction')
    
    cmd_res = 1/poke_amp * M2C @ mode_res
    
    # WITH RESIDUALS 
    
    im_list = [util.get_DM_command_in_2D( mode_aberration ),1/np.mean(raw_img) * raw_img - I0/np.mean(I0),  util.get_DM_command_in_2D( cmd_res ) ,util.get_DM_command_in_2D( mode_aberration - cmd_res ) ]
    xlabel_list = [None, None, None, None]
    ylabel_list = [None, None, None, None]
    title_list = ['Aberration on DM', 'ZWFS signal', 'reconstructed DM cmd', 'residual']
    cbar_label_list = ['DM command', 'ADU (Normalized)', 'DM command' , 'DM command' ] 
    savefig = fig_path + 'delme.png' #f'mode_reconstruction_images/phase_reconstruction_example_mode-{mode_indx}_basis-{phase_ctrl.config["basis"]}_ctrl_modes-{phase_ctrl.config["number_of_controlled_modes"]}ctrl_act_diam-{phase_ctrl.config["dm_control_diameter"]}_readout_mode-12x12.png'

    util.nice_heatmap_subplots( im_list , xlabel_list, ylabel_list, title_list, cbar_label_list, fontsize=15, axis_off=True, cbar_orientation = 'bottom', savefig=savefig)
    
    _ = input('press when ready to go to next moce ')




a = fits.open( 'tmp/RECONSTRUCTORS_DIT-0.001_gain_high_22-08-2024T21.26.09.fits' ) 





U,S,Vt = np.linalg.svd( IM @ IM.T )
plt.figure(); plt.semilogy( S ); plt.savefig( fig_path + "delme.png")

plt.figure(); plt.imshow( util.get_DM_command_in_2D(U.T[0]) ); plt.savefig( fig_path + "delme.png")



# test we can get on/off phase mask with DM 
fourier_basis = util.construct_command_basis( basis='fourier', number_of_modes = 40, Nx_act_DM = 12, Nx_act_basis = 12, act_offset=(0,0), without_piston=True)
tip = fourier_basis[:,0]
tilt = fourier_basis[:,5]

fig,ax = plt.subplots(1,2)
ax[0].imshow( util.get_DM_command_in_2D( tip ) ); ax[0].set_title('')
ax[1].imshow( util.get_DM_command_in_2D( tilt ) ); ax[1].set_title('')
plt.savefig( fig_path + "delme.png")

zwfs.dm.send_data( zwfs.dm_shapes['flat_dm'] )
time.sleep(0.1)
I0 =  zwfs.get_image( apply_manual_reduction = True )
zwfs.dm.send_data( 0.5 + 2* tip )
time.sleep(0.1)
N0 = zwfs.get_image( apply_manual_reduction = True )
zwfs.dm.send_data( zwfs.dm_shapes['flat_dm'] )

fig,ax = plt.subplots(1,2)
ax[0].imshow( I0 ); ax[0].set_title('FPM ON')
ax[1].imshow( N0 ); ax[1].set_title('FPM OFF')
plt.savefig( fig_path + "delme.png")



M2C = phase_ctrl.config['M2C'] # readability 

I2M = phase_ctrl.ctrl_parameters[ctrl_method_label]['I2M']

IM = phase_ctrl.ctrl_parameters[ctrl_method_label]['IM'] # readability 
# unfiltered CM
CM = phase_ctrl.ctrl_parameters[ctrl_method_label]['CM'] # readability 

U,S,Vt = np.linalg.svd( IM @ IM.T )

plt.figure(); plt.semilogy( S ); plt.savefig( fig_path + "delme.png")

plt.figure(); plt.imshow( phase_ctrl.ctrl_parameters[ctrl_method_label]['ref_pupil_FPM_in'] ); plt.savefig( fig_path + "delme.png")

plt.figure(); plt.plot( I2M.T @ IM[65] ); plt.savefig( fig_path + "delme.png")


# do one poke and check we actually get something is
zwfs.dm.send_data( zwfs.dm_shapes['flat_dm'] )
act = 65
damp = 0.1

time.sleep(0.1)
I0 =  zwfs.get_image( apply_manual_reduction = True )
cmd = zwfs.dm_shapes['flat_dm'].copy() 
cmd[act] += damp 
zwfs.dm.send_data( cmd )
time.sleep(0.1)
Ip = zwfs.get_image( apply_manual_reduction = True )

cmd = zwfs.dm_shapes['flat_dm'].copy() 
cmd[act] -= damp 
zwfs.dm.send_data( cmd )
time.sleep(0.1)
Im = zwfs.get_image( apply_manual_reduction = True )

fig,ax = plt.subplots(1,2)
ax[0].imshow( Ip-I0 ); ax[0].set_title('I(a65+=0.1) - I0')
ax[1].imshow( Im-I0 ); ax[1].set_title('I(a65-=0.1) - I0')
plt.savefig( fig_path + "delme.png")

#plt.figure(); plt.imshow( Ip-I0 ); plt.savefig( fig_path + "delme.png")

