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

# ====== hardware variables
beam = 3
phasemask_name = 'J3'
phasemask_OUT_offset = [1000,1000]  # relative offset (um) to take phasemask out of beam
BFO_pos = 4000 # um (absolute position of detector imgaging lens) 
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


pupil_crop_region = [204,268,125, 187] #[None, None, None, None] #[204 -50 ,268+50,125-50, 187+50] 

#init our ZWFS (object that interacts with camera and DM) (old path = home/baldr/Documents/baldr/ANU_demo_scripts/BALDR/)
zwfs = ZWFS.ZWFS(DM_serial_number=DM_serial_number, cameraIndex=0, DMshapes_path = 'DMShapes/', pupil_crop_region=pupil_crop_region ) 

# the sydney BMC multi-3.5 calibrated flat seems shit! Try with just a 

zwfs.deactive_cropping() # zwfws.set_camera_cropping(r1, r2, c1, c2 ) #<- use this for latency tests 
zwfs.set_camera_fps( 200 );time.sleep(0.2)
zwfs.set_camera_dit( 0.001 );time.sleep(0.2)
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
bad_pixels = zwfs.get_bad_pixel_indicies( no_frames = 1000, std_threshold = 50 , flatten=False)

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
# if you like the improvement 
# phasemask.update_mask_position( phasemask_name )
# phasemask.write_current_mask_positions()

# == automatic search 
"""
phasemask_diameter = 50 # um <- have to ensure grid is at this resolution 
search_radius = 100  # search radius for spiral search (um)
dtheta = np.pi / 20  # angular increment (rad) 
iterations_per_circle = 2*np.pi / dtheta
dr = phasemask_diameter / iterations_per_circle # cover 1 phasemask diameter per circle

# move off phase mask, its good to make sure zwfs object has dark, bad pixel map etc first to see better
phasemask.move_to_mask( phasemask_name )
phasemask.move_relative( [1000,1000] )  # 1mm in each axis
time.sleep(1.2)
reference_img =  np.mean(zwfs.get_some_frames(number_of_frames = 10, apply_manual_reduction = True ) , axis=0 ) # Capture reference image when misaligned
phasemask.move_to_mask( phasemask_name )  # move back to initial_position 
time.sleep(1.2)

# look at savefigName png image while running to see updates ()
phasemask_centering_tool.spiral_search_and_center(zwfs, phasemask, phasemask_name, search_radius, dr, dtheta, \
    reference_img, fine_tune_threshold=2, savefigName='tmp/delme.png', usr_input=True)

"""

#init our pupil controller (object that processes ZWFS images and outputs VCM commands)
pupil_ctrl = pupil_control.pupil_controller_1(config_file = None)

#analyse pupil and decide if it is ok. This must be done before reconstructor
pupil_report = pupil_control.analyse_pupil_openloop( zwfs, debug = True, return_report = True)

if pupil_report['pupil_quality_flag'] == 1: 
    zwfs.update_reference_regions_in_img( pupil_report ) # 


# x,y in compass referenced to DM right (+x), up (+y)
I0, N0 = util.get_reference_images(zwfs, phasemask, theta_degrees=11.8, number_of_frames=256, \
compass = True, compass_origin=None, savefig=fig_path + f'FPM-in-out_{phasemask_name}.png' )


"""zwfs.pupil_pixel_filter =  (N0 > np.mean(N0) + 2 * np.std( N0))
zwfs.pupil_pixel_filter[0,:]=False # first row has frame counts 
zwfs.pupil_pixel_filter = zwfs.pupil_pixel_filter.reshape(-1)
"""
# plot the active pupil region registered 
#img_tmp = np.mean( zwfs.get_some_frames(number_of_frames = 100, apply_manual_reduction = True) , axis =0 )# just to get correct image shape
fig,ax = plt.subplots(1,2,figsize=(10,5))
ax[0].imshow( N0 ) ; ax[0].set_title('phasemask out')
ax[1].imshow( zwfs.pupil_pixel_filter.reshape(N0.shape)) ; ax[1].set_title('registered pupil')
plt.savefig(fig_path + f'pupil_registration_{pupil_crop_region}.png')



#####

# TESTING 

#####


# --- linear ramps 
# use baldr.
recon_data = util.GET_BDR_RECON_DATA_INTERNAL(zwfs, number_amp_samples = 20, amp_max = 0.2,\
number_images_recorded_per_cmd = 200, save_fits = data_path+f'pokeramp_data_MASK_{phasemask_name}_sydney_{tstamp}.fits') 
# recon_data = fits.open( data_path+'recon_data_LARGE_SECONDARY_19-04-2024T12.19.22.fits' )

#M2C = util.construct_command_basis( basis='Zernike', number_of_modes = 5, Nx_act_DM = 12,Nx_act_basis =12, act_offset= (0,0), without_piston=True) 
#active_dm_actuator_filter = (abs(np.sum( M2C, axis=1 )) > 0 ).astype(bool)                           

#zonal_fits = util.PROCESS_BDR_RECON_DATA_INTERNAL(recon_data, bad_pixels = ([],[]), active_dm_actuator_filter=active_dm_actuator_filter, debug=True, fig_path = fig_path , savefits= data_path+f'fitted_pokeramp_data_MASK_{phasemask_name}_sydney_{tstamp}.fits') 



# ===== Improve I0 with focus on DM ?
fourier_basis = util.construct_command_basis( basis='fourier', number_of_modes = 40, Nx_act_DM = 12, Nx_act_basis = 12, act_offset=(0,0), without_piston=True)

# check its focus
plt.figure(); plt.imshow( util.get_DM_command_in_2D( fourier_basis.T[19] )) ;plt.savefig(fig_path + 'focus_DM.png')

int_sum = []
int_sum_N0 = []
int_sum_in_pupil = []
int_sum_N0_in_pupil = []
amp_grid = np.linspace(-1,1,15)

for a in amp_grid:

    zwfs.dm.send_data( zwfs.dm_shapes['flat_dm'] + a * fourier_basis.T[19] ) 
    time.sleep(0.5)
    dm_ab = a * util.get_DM_command_in_2D( fourier_basis.T[19] )
    img = np.mean( zwfs.get_some_frames(number_of_frames = 1000, apply_manual_reduction = True ), axis =0 )

    phasemask.move_relative([100,100])

    imgN0 = np.mean( zwfs.get_some_frames(number_of_frames = 1000, apply_manual_reduction = True ), axis =0 )

    phasemask.move_relative([-100,-100])

    int_sum.append( np.sum(img) )
    int_sum_in_pupil.append( np.sum(img.reshape(-1)[zwfs.pupil_pixel_filter]) )

    int_sum_N0.append( np.sum( imgN0 ) )
    int_sum_N0_in_pupil.append( np.sum(imgN0.reshape(-1)[zwfs.pupil_pixel_filter]) )
    print( f'\na={a}\nsum(img)={[-1]}')
    """
    im_list = [dm_ab ,img ]
    xlabel_list = [None, None]
    ylabel_list = [None, None]
    title_list = ['Aberration on DM', 'Intensity']
    cbar_label_list = [f'DM command (a={round(a,1)})', 'ADU ' ] 
    savefig = fig_path + 'delme.png' #f'mode_reconstruction_images/phase_reconstruction_example_mode-{mode_indx}_basis-{phase_ctrl.config["basis"]}_ctrl_modes-{phase_ctrl.config["number_of_controlled_modes"]}ctrl_act_diam-{phase_ctrl.config["dm_control_diameter"]}_readout_mode-12x12.png'

    util.nice_heatmap_subplots( im_list , xlabel_list, ylabel_list, title_list, cbar_label_list, fontsize=15, axis_off=True, cbar_orientation = 'bottom', savefig=savefig)
    """
    #_ = input('next?')

# go off to get N0 estimate 

"""zwfs.dm.send_data( zwfs.dm_shapes['flat_dm'] + 2*fourier_basis.T[0] ) 
time.sleep(0.5)
N0 = np.mean( zwfs.get_some_frames(number_of_frames = 100, apply_manual_reduction = True ), axis =0 )
zwfs.dm.send_data( zwfs.dm_shapes['flat_dm']  ) 
"""
cmd_2_opd = 3 # um RMS
plt.figure(figsize=(10,8)); 
#plt.axhline( N0/np.sum( N0 ), color='k',ls=':', label=r'$\Sigma N_0(x,y)$')
#plt.axhline( N0.reshape(-1)[zwfs.pupil_pixel_filter] /np.sum( N0 ), color='k',ls='--', label=r'$\Sigma N_0(x,y \in pupil)$')
plt.plot( cmd_2_opd * amp_grid * np.std( fourier_basis.T[19] ) , int_sum_N0/np.sum( N0 ), color='k',ls=':', label=r'$\Sigma N_0(x,y)$')
plt.plot( cmd_2_opd * amp_grid * np.std( fourier_basis.T[19] ),int_sum_N0_in_pupil /np.sum( N0 ), color='k',ls='--', label=r'$\Sigma N_0(x,y \in pupil)$')
plt.plot( cmd_2_opd * amp_grid * np.std( fourier_basis.T[19] ), int_sum/np.sum( N0 ), label=r'$\Sigma I_0(x,y)$' );
plt.plot( cmd_2_opd * amp_grid * np.std( fourier_basis.T[19] ) , int_sum_in_pupil/np.sum( N0 ) , label=r'$\Sigma I_0(x,y \in pupil)$');

plt.legend(fontsize=12)
plt.gca().tick_params(labelsize=15)
plt.xlabel('focus amplitude OPD (um RMS)',fontsize=15)
plt.ylabel(r'$\Sigma I(x,y)$',fontsize=15)
plt.axvline(0)
plt.savefig(fig_path + f'intensity_vs_focus_{pupil_crop_region}_BFO_with_{a_opt}_focus_offset.png',dpi=300, bbox_inches='tight')

# add focus amplitude where gradient was maximum to DM flat 
#a_opt = 0.16#0.5 / cmd_2_opd 
#zwfs.dm_shapes['flat_dm']  = zwfs.dm_shapes['flat_dm'] + 0.16 * fourier_basis.T[19]

# now do pokaramp with fpcus offset 
recon_data = util.GET_BDR_RECON_DATA_INTERNAL(zwfs,number_amp_samples = 20, amp_max = 0.2,\
number_images_recorded_per_cmd = 200, save_fits = data_path+f'pokeramp_data_MASK_{phasemask_name}_{a_opt}focus_offset_sydney_{tstamp}.fits',\
source_selector = None) 






#phasemask_centering_tool.move_relative_and_get_image(zwfs, phasemask, savefigName=fig_path + 'delme.png')

 
# ====== testing reconstruction 





#zwfs.pupil_pixel_filter = ~zwfs.bad_pixel_filter
#zwfs.pupil_pixels = np.where( ~zwfs.bad_pixel_filter )[0]

#init our phase controller (object that processes ZWFS images and outputs DM commands)
zonal_phase_ctrl = phase_control.phase_controller_1(config_file = None, basis_name = 'Zonal', number_of_controlled_modes = 140) 
zernike_phase_ctrl = phase_control.phase_controller_1(config_file = None, basis_name = 'Zernike', number_of_controlled_modes = 20) 
fourier_phase_ctrl = phase_control.phase_controller_1(config_file = None, basis_name = 'fourier', number_of_controlled_modes = 20)

# to change basis : 
#phase_ctrl.change_control_basis_parameters( controller_label = ctrl_method_label, number_of_controlled_modes=phase_ctrl.config['number_of_controlled_modes'], basis_name='Zonal' , dm_control_diameter=None, dm_control_center=None)


zonal_dict = {'controller': zonal_phase_ctrl, 'poke_amp':0.07, 'poke_method':'double_sided_poke', 'inverse_method':'pinv', 'label':'zonal_0.07pokeamp_in-out_pokes_pinv' }
zernike_dict = {'controller': zernike_phase_ctrl, 'poke_amp':0.2, 'poke_method':'double_sided_poke', 'inverse_method':'pinv', 'label':'zernike_0.2pokeamp_in-out_pokes_pinv' }
fourier_dict = {'controller': fourier_phase_ctrl, 'poke_amp':0.2, 'poke_method':'double_sided_poke', 'inverse_method':'pinv', 'label':'fourier_0.2pokeamp_in-out_pokes_pinv' }

build_dict = {
    'zonal':zonal_dict ,
    'zernike':zernike_dict ,
    'fourier':fourier_dict
}

basis = 'fourier'

zwfs.dm.send_data( zwfs.dm_shapes['flat_dm'] )
time.sleep( 0.1 )

# readability 
p = build_dict[basis]['controller']
label = build_dict[basis]['label']


# ====== Building noise model (covariance of detector signals)
p.update_noise_model(zwfs, number_of_frames = 10000 )


p.build_control_model_2(
    zwfs, \
    poke_amp = build_dict[basis]['poke_amp'], label=label, \
    poke_method = build_dict[basis]['poke_method'], inverse_method= build_dict[basis]['inverse_method'],  debug = True \
    )


zwfs.dm.send_data( zwfs.dm_shapes['flat_dm'] )
time.sleep( 0.1 )

# write fits to input into RTC
zwfs.write_reco_fits( p, label, save_path=data_path, save_label=label)

# have a look at the singular values and eigenmodes in DM and detector space 
p.plot_SVD_modes( zwfs, label, save_path=fig_path)

# check reference images
I0 = p.ctrl_parameters[label]['ref_pupil_FPM_in']
N0 = p.ctrl_parameters[label]['ref_pupil_FPM_out']

fig,ax = plt.subplots(1,2,figsize=(10,5))
ax[0].imshow(I0); ax[0].set_title('I0')
ax[1].imshow( N0 ) ; ax[1].set_title('N0')
plt.savefig(fig_path + 'delme.png')

mode_basis = p.config['M2C']  
poke_amp = p.ctrl_parameters[label]['poke_amp']
I2M = p.ctrl_parameters[label]['I2M']
IM = p.ctrl_parameters[label]['IM'] 

# unfiltered CM
CM = p.ctrl_parameters[label]['CM'] 
R_TT = p.ctrl_parameters[label]['R_TT'] 
R_HO = p.ctrl_parameters[label]['R_HO'] 

M2C = p.ctrl_parameters[label]['M2C_4reco'] 
I0 = p.ctrl_parameters[label]['ref_pupil_FPM_in']
N0 = p.ctrl_parameters[label]['ref_pupil_FPM_out']


# Look at the signals in the interaction matrix (filtered by registered pupil pixels)
for m in range( len(IM) ):
    tmp =  zwfs.pupil_pixel_filter.copy()
    imgrid = np.zeros(tmp.shape)
    imgrid[tmp] = IM[m] #

    fig,ax = plt.subplots(1,1,figsize=(5,5))
    im = ax.imshow( imgrid.reshape( zwfs.I0.shape ) ); ax.set_title(f'mode {m}')
    plt.colorbar( im , ax=ax )
    #ax[1].imshow( N0 ) ; ax[1].set_title('N0')
    plt.savefig(fig_path + 'delme.png')
    plt.close() 
    _= input('next?')



# can we do open loop reconstruction ok on our input basis?


# Testing how the number of frames averaged influence reconstructor

"""
# fourier tip to go off phase mask  
fourier_basis = util.construct_command_basis( basis='fourier', number_of_modes = 40, Nx_act_DM = 12, Nx_act_basis = 12, act_offset=(0,0), without_piston=True)
tip = fourier_basis[:,0]

imgs_to_mean =  256

zwfs.dm.send_data(0.5 + 2 * tip ) # move off phase mask 
time.sleep(0.1)
N0_list = zwfs.get_some_frames(number_of_frames = imgs_to_mean, apply_manual_reduction = True ) #REFERENCE INTENSITY WITH FPM OUT
N0 = np.mean( N0_list, axis = 0 )


zwfs.dm.send_data( zwfs.dm_shapes['flat_dm'] )
time.sleep(0.1)
I0_list = zwfs.get_some_frames(number_of_frames = imgs_to_mean, apply_manual_reduction = True ) #REFERENCE INTENSITY WITH FPM IN
I0 = np.mean( I0_list, axis = 0 )
"""
amp = 0.2 # amp to apply to each mode
imgs_to_mean_grid  = np.logspace( 0 , 3, 30)
rmse_dict = {}
for imgs_to_mean in imgs_to_mean_grid:
    print(imgs_to_mean)
    rmse_dict[imgs_to_mean] = {}
    for mode_indx in range( len(IM) ) :#len(M2C)-1 ) :  

        mode_aberration = mode_basis.T[mode_indx]#   M2C.T[mode_indx]
        #plt.imshow( util.get_DM_command_in_2D(amp*mode_aberration));plt.colorbar();plt.show()
        
        dm_cmd_aber = zwfs.dm_shapes['flat_dm'] + amp * mode_aberration 

        zwfs.dm.send_data( dm_cmd_aber )
        time.sleep(0.1)
        #raw_img_list = []
        #for i in range( 10 ) :
        #    raw_img_list.append( zwfs.get_image() ) # @D, remember for control_phase method this needs to be flattened and filtered for pupil region
        raw_img = np.mean( zwfs.get_some_frames(number_of_frames = imgs_to_mean.astype(int), apply_manual_reduction = True ) ,axis=0) #zwfs.get_image()
        # plt.figure() ; plt.imshow( raw_img ) ; plt.savefig( fig_path + f'delme.png') # <- signal?
        
        err_img = p.get_img_err( 1/np.mean(raw_img) * raw_img.reshape(-1)[zwfs.pupil_pixel_filter]  ) 
        
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
        """
        plt.plot( app_amp ,'x', label='applied amplitude');
        plt.axvline(mode_indx  , ls=':', color='k', label='mode applied') ; plt.xlabel('mode index',fontsize=15); 
        plt.ylabel('mode amplitude',fontsize=15); plt.gca().tick_params(labelsize=15) ; plt.legend();
        plt.savefig( fig_path + f'delme.png')

        _ = input('press when ready to see mode reconstruction')
        """
        cmd_res =  M2C @ mode_res  # <-- should be like this. 1/poke_amp * M2C @ mode_res # SHOULD I USE M2C_4reco here? 
        
        # WITH RESIDUALS 
        """
        im_list = [util.get_DM_command_in_2D( amp * mode_aberration  ),raw_img - I0,  util.get_DM_command_in_2D( cmd_res ) ,util.get_DM_command_in_2D( amp * mode_aberration - cmd_res ) ]
        xlabel_list = [None, None, None, None]
        ylabel_list = [None, None, None, None]
        title_list = ['Aberration on DM', 'I-I0', 'reconstructed DM cmd', 'residual']
        cbar_label_list = ['DM command', 'ADU (Normalized)', 'DM command' , 'DM command' ] 
        savefig = fig_path + 'delme.png' #f'mode_reconstruction_images/phase_reconstruction_example_mode-{mode_indx}_basis-{phase_ctrl.config["basis"]}_ctrl_modes-{phase_ctrl.config["number_of_controlled_modes"]}ctrl_act_diam-{phase_ctrl.config["dm_control_diameter"]}_readout_mode-12x12.png'

        util.nice_heatmap_subplots( im_list , xlabel_list, ylabel_list, title_list, cbar_label_list, fontsize=15, axis_off=True, cbar_orientation = 'bottom', savefig=savefig)
        
        _ = input('press when ready to go to next moce ')
        """
        rmse = np.sqrt( np.mean(( amp * mode_aberration - cmd_res  )**2) )
        rmse_dict[imgs_to_mean][mode_indx] = rmse

plt.figure(); 
for mode_indx in [0,1,2]:
    print(mode_indx)
    plt.semilogx(imgs_to_mean_grid,  [rmse_dict[imgs_to_mean][mode_indx] for imgs_to_mean in rmse_dict], label= f'mode index={mode_indx}');
plt.xlabel( ' number of frames averaged' ,fontsize=15)
plt.ylabel(' mode RMSE')
plt.legend() 
plt.savefig(fig_path+'delme.png');



# Test how number of modes (modal leakage) effects residuals

build_dict = {}
Nmode_grid = [9,36,64]
imgs_to_mean = 200
rmse_dict = {}
for m in Nmode_grid: # building controllers with different number of modes (m) in construction
    print(f'mode ={m}')
    basis = f'fourier_{m}'
    phase_ctrl_fourier = phase_control.phase_controller_1(config_file = None, basis_name = 'fourier', number_of_controlled_modes = m) 
    fourier_dict = {'controller': phase_ctrl_fourier, 'poke_amp':0.2, 'poke_method':'double_sided_poke', 'inverse_method':'pinv', 'label':'fourier_0.2pokeamp_in-out_pokes_pinv' }
    build_dict[basis] = fourier_dict

    p = build_dict[basis]['controller']
    label = build_dict[basis]['label']


    # ====== Building noise model (covariance of detector signals)
    p.update_noise_model(zwfs, number_of_frames = 10000 )

    p.build_control_model_2(
        zwfs, \
        poke_amp = build_dict[basis]['poke_amp'], label=label, \
        poke_method = build_dict[basis]['poke_method'], inverse_method= build_dict[basis]['inverse_method'],  debug = True \
        )

    # check reference images
    I0 = p.ctrl_parameters[label]['ref_pupil_FPM_in']
    N0 = p.ctrl_parameters[label]['ref_pupil_FPM_out']

    mode_basis = p.config['M2C']  
    poke_amp = p.ctrl_parameters[label]['poke_amp']
    I2M = p.ctrl_parameters[label]['I2M']
    IM = p.ctrl_parameters[label]['IM'] 

    # unfiltered CM
    CM = p.ctrl_parameters[label]['CM'] 
    R_TT = p.ctrl_parameters[label]['R_TT'] 
    R_HO = p.ctrl_parameters[label]['R_HO'] 

    M2C = p.ctrl_parameters[label]['M2C_4reco'] 
    I0 = p.ctrl_parameters[label]['ref_pupil_FPM_in']
    N0 = p.ctrl_parameters[label]['ref_pupil_FPM_out']

        
    amp = 0.2 # amp to apply to each mode
    imgs_to_mean_grid  = 200

    rmse_dict[m] = {}
    for mode_indx in range( 5 ) :#len(M2C)-1 ) :  

        mode_aberration = mode_basis.T[mode_indx]#   M2C.T[mode_indx]
        #plt.imshow( util.get_DM_command_in_2D(amp*mode_aberration));plt.colorbar();plt.show()
        
        dm_cmd_aber = zwfs.dm_shapes['flat_dm'] + amp * mode_aberration 

        zwfs.dm.send_data( dm_cmd_aber )
        time.sleep(0.1)
        #raw_img_list = []
        #for i in range( 10 ) :
        #    raw_img_list.append( zwfs.get_image() ) # @D, remember for control_phase method this needs to be flattened and filtered for pupil region
        raw_img = np.mean( zwfs.get_some_frames(number_of_frames = imgs_to_mean, apply_manual_reduction = True ) ,axis=0) #zwfs.get_image()
        # plt.figure() ; plt.imshow( raw_img ) ; plt.savefig( fig_path + f'delme.png') # <- signal?
        
        err_img = p.get_img_err( 1/np.mean(raw_img) * raw_img.reshape(-1)[zwfs.pupil_pixel_filter]  ) 
        
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
        """
        plt.plot( app_amp ,'x', label='applied amplitude');
        plt.axvline(mode_indx  , ls=':', color='k', label='mode applied') ; plt.xlabel('mode index',fontsize=15); 
        plt.ylabel('mode amplitude',fontsize=15); plt.gca().tick_params(labelsize=15) ; plt.legend();
        plt.savefig( fig_path + f'delme.png')

        _ = input('press when ready to see mode reconstruction')
        """
        cmd_res =  M2C @ mode_res  # <-- should be like this. 1/poke_amp * M2C @ mode_res # SHOULD I USE M2C_4reco here? 
        
        # WITH RESIDUALS 
        """
        im_list = [util.get_DM_command_in_2D( amp * mode_aberration  ),raw_img - I0,  util.get_DM_command_in_2D( cmd_res ) ,util.get_DM_command_in_2D( amp * mode_aberration - cmd_res ) ]
        xlabel_list = [None, None, None, None]
        ylabel_list = [None, None, None, None]
        title_list = ['Aberration on DM', 'I-I0', 'reconstructed DM cmd', 'residual']
        cbar_label_list = ['DM command', 'ADU (Normalized)', 'DM command' , 'DM command' ] 
        savefig = fig_path + 'delme.png' #f'mode_reconstruction_images/phase_reconstruction_example_mode-{mode_indx}_basis-{phase_ctrl.config["basis"]}_ctrl_modes-{phase_ctrl.config["number_of_controlled_modes"]}ctrl_act_diam-{phase_ctrl.config["dm_control_diameter"]}_readout_mode-12x12.png'

        util.nice_heatmap_subplots( im_list , xlabel_list, ylabel_list, title_list, cbar_label_list, fontsize=15, axis_off=True, cbar_orientation = 'bottom', savefig=savefig)
        
        _ = input('press when ready to go to next moce ')
        """
        rmse = np.sqrt( np.mean(( amp * mode_aberration - cmd_res  )**2) )
        rmse_dict[m][mode_indx] = rmse



plt.figure(); 
for mode_indx in [0,1,2]:
    print(mode_indx)
    plt.plot( Nmode_grid,  [rmse_dict[m][mode_indx] for m in rmse_dict], label= f'correcting mode = {mode_indx}');
plt.xlabel( 'Number of modes corrected' ,fontsize=15)
plt.ylabel('mode RMSE (DM CMD SPACE)')
plt.title('Fourier basis')
plt.legend() 
plt.savefig(fig_path+'delme.png')


# Try correct in eigenspace, filtering modes 


zonal_phase_ctrl = phase_control.phase_controller_1(config_file = None, basis_name = 'Zonal', number_of_controlled_modes = 140) 
fourier_phase_ctrl = phase_control.phase_controller_1(config_file = None, basis_name = 'fourier', number_of_controlled_modes = 100) 

zonal_dict = {'controller': zonal_phase_ctrl, 'poke_amp':0.07, 'poke_method':'double_sided_poke', 'inverse_method':'pinv', 'label':'zonal_0.07pokeamp_in-out_pokes_pinv' }
fourier_dict = {'controller': fourier_phase_ctrl, 'poke_amp':0.2, 'poke_method':'double_sided_poke', 'inverse_method':'pinv', 'label':'fourier_0.2pokeamp_in-out_pokes_pinv' }

build_dict = {
    'zonal':zonal_dict ,
    'fourier':fourier_dict
}

basis = 'zonal'

p = build_dict[basis]['controller']
label = build_dict[basis]['label']


# ====== Building noise model (covariance of detector signals)
p.update_noise_model(zwfs, number_of_frames = 10000 )

p.build_control_model_2(
    zwfs, \
    poke_amp = build_dict[basis]['poke_amp'], label=label, \
    poke_method = build_dict[basis]['poke_method'], inverse_method= build_dict[basis]['inverse_method'],  debug = True \
    )

# check reference images
"""I0 = p.ctrl_parameters[label]['ref_pupil_FPM_in']
N0 = p.ctrl_parameters[label]['ref_pupil_FPM_out']

mode_basis = p.config['M2C']  
poke_amp = p.ctrl_parameters[label]['poke_amp']
I2M = p.ctrl_parameters[label]['I2M']
IM = p.ctrl_parameters[label]['IM'] 

# unfiltered CM
CM = p.ctrl_parameters[label]['CM'] 
R_TT = p.ctrl_parameters[label]['R_TT'] 
R_HO = p.ctrl_parameters[label]['R_HO'] 

M2C = p.ctrl_parameters[label]['M2C_4reco'] 
I0 = p.ctrl_parameters[label]['ref_pupil_FPM_in']
N0 = p.ctrl_parameters[label]['ref_pupil_FPM_out']
"""
poke_amp = p.ctrl_parameters[label]['poke_amp']
IM = p.ctrl_parameters[label]['IM'] 
U, S, Vt = np.linalg.svd( IM.T , full_matrices=False)  # I append rows to IM.. convention is columns.. Thats why I need CM.T etc 


# IM @ CM = I . .CM = Vt.T @ np.diag(1/S) @ U.T
(U @ np.diag(S) @ Vt) @ (Vt.T @ np.diag(1/S) @ U.T)

plt.figure(); plt.imshow( IM.T @ np.linalg.pinv( IM.T )); plt.savefig(fig_path +'delme.png')

dm_pupil_filter =  np.std( IM, axis=1) > 1

"""plt.figure(); plt.imshow( util.get_DM_command_in_2D(Vt[0]));plt.savefig(fig_path+'delme.png')
plt.figure(); plt.semilogy(S); plt.xlabel('eigenmode index'); plt.ylabel('Eigenvalues');plt.savefig(fig_path+'delme.png')

# important to get DM registration of the pupil 
plt.figure(); plt.imshow( util.get_DM_command_in_2D(np.std( IM, axis=1) ) );plt.colorbar(); plt.title(r'$\sigma$'); plt.savefig(fig_path + 'delme.png')
# Set pupil registration threshold at 1sigma in the pixel space 
dm_pupil_filter =  np.std( IM, axis=1) > 1
plt.figure(); plt.imshow( util.get_DM_command_in_2D( dm_pupil_filter ) );plt.colorbar(); plt.title(r'$\sigma$'); plt.savefig(fig_path + 'delme.png')
"""
p.plot_SVD_modes( zwfs, label, save_path=fig_path)

truncation_index = 20
"""Sigma = 1/S 
gains = np.zeros(len(S))
gains[:truncation_index] = 1
I2M = gains * np.diag(Sigma) @ U.T"""
#np.sum(Vt.T[0]**2) = 1 so we need to adjust by poke_amp 
M2C = poke_amp * Vt.T 

#mode_basis = p.config['M2C']  # fourier 
    
amp = 0.4 # amp to apply to each mode
imgs_to_mean  = 200

#plt.figure(); plt.imshow( util.get_DM_command_in_2D(mode_basis.T[0]));plt.savefig(fig_path+'delme.png')

# test we can get on/off phase mask with DM 
fourier_basis = util.construct_command_basis( basis='fourier', number_of_modes = 40, Nx_act_DM = 12, Nx_act_basis = 12, act_offset=(0,0), without_piston=True)

rmse_dict = {}
rmse_baseline_dict = {}
truncation_index_grid = [2,5,10,15,20,25,30,35,40,45,50,55,60,65,70,75,80 ]
mode_indx_grid=range( 10 )

for cnt , truncation_index in enumerate(truncation_index_grid):
    print( f'complete {100 * cnt/len(truncation_index_grid) } %')
    Sigma = 1/S 
    gains = np.zeros(len(S))
    gains[:truncation_index] = 1
    I2M = gains * np.diag(Sigma) @ U.T
    rmse_dict[truncation_index] = {}
    for mode_indx in mode_indx_grid:#len(M2C)-1 ) :  

        # apply the full aberration without pupil filter 
        mode_aberration =  fourier_basis.T[mode_indx]#   M2C.T[mode_indx]
        #plt.imshow( util.get_DM_command_in_2D(amp*mode_aberration));plt.colorbar();plt.show()
        
        dm_cmd_aber = zwfs.dm_shapes['flat_dm'] + amp * mode_aberration 

        zwfs.dm.send_data( dm_cmd_aber )
        time.sleep(0.2)
        #raw_img_list = []
        #for i in range( 10 ) :
        #    raw_img_list.append( zwfs.get_image() ) # @D, remember for control_phase method this needs to be flattened and filtered for pupil region
        raw_img = np.mean( zwfs.get_some_frames(number_of_frames = imgs_to_mean, apply_manual_reduction = True ) ,axis=0) #zwfs.get_image()
        # plt.figure() ; plt.imshow( raw_img ) ; plt.savefig( fig_path + f'delme.png') # <- signal?
        
        err_img = p.get_img_err( 1/np.mean(raw_img) * raw_img.reshape(-1)[zwfs.pupil_pixel_filter]  ) 
        
        # plt.figure() ; plt.hist( err_img, label='meas', alpha=0.3 ) ; plt.hist( IM[mode_indx] , label='from IM', alpha=0.3); plt.legend() ; plt.savefig( fig_path + f'delme.png') # <- should be around zeros

        #mode_res_test : inject err_img from interaction matrix to I2M .. should result in perfect reconstruction  
        #plt.figure(); plt.plot( I2M.T @ IM[2] ); plt.savefig( fig_path + f'delme.png')
        #plt.figure(); plt.plot( I2M.T @ IM[mode_indx]  ,label='reconstructed amplitude'); plt.axvline(mode_indx  , ls=':', color='k', label='mode applied') ; plt.xlabel('mode index'); plt.ylabel('mode amplitude'); plt.legend(); plt.savefig( fig_path + f'delme.png')
        mode_res =  I2M @ err_img 
        """
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
        """  
        cmd_res =  M2C @ mode_res  # <-- should be like this. 1/poke_amp * M2C @ mode_res # SHOULD I USE M2C_4reco here? 
        
        # WITH RESIDUALS 
        
        im_list = [util.get_DM_command_in_2D( amp * mode_aberration  ),raw_img - I0,  util.get_DM_command_in_2D( cmd_res ) ,util.get_DM_command_in_2D( dm_pupil_filter *(amp * mode_aberration - cmd_res) ) ]
        xlabel_list = [None, None, None, None]
        ylabel_list = [None, None, None, None]
        title_list = ['Aberration on DM', 'I-I0', 'reconstructed DM cmd', 'residual']
        cbar_label_list = ['DM command', 'ADU (Normalized)', 'DM command' , 'DM command' ] 
        savefig = fig_path + 'delme.png' #f'mode_reconstruction_images/phase_reconstruction_example_mode-{mode_indx}_basis-{phase_ctrl.config["basis"]}_ctrl_modes-{phase_ctrl.config["number_of_controlled_modes"]}ctrl_act_diam-{phase_ctrl.config["dm_control_diameter"]}_readout_mode-12x12.png'

        util.nice_heatmap_subplots( im_list , xlabel_list, ylabel_list, title_list, cbar_label_list, fontsize=15, axis_off=True, cbar_orientation = 'bottom', savefig=savefig)
        
        _ = input('press when ready to go to next moce ')
        
        # filter in DM pupil for RMSE 
        rmse = np.sqrt( np.mean((  dm_pupil_filter *(amp * mode_aberration - cmd_res)  )**2) )

        rmse_baseline_dict[mode_indx] = np.sqrt( np.mean(  dm_pupil_filter *(amp * mode_aberration )**2 ) ) 
        rmse_dict[truncation_index][mode_indx] = rmse



plt.figure(); 
for mode_indx,col in zip( mode_indx_grid[0:5], ['r','g','b','orange','y']):
    print(mode_indx)
    plt.plot( truncation_index_grid,  [rmse_dict[m][mode_indx] for m in rmse_dict], color=col, label= f'correcting Fourier mode = {mode_indx}');
    plt.axhline( rmse_baseline_dict[mode_indx] , color=col,ls=':')
plt.xlabel( 'Number of modes corrected (in Eigenspace)' ,fontsize=15)
plt.ylabel('mode RMSE (DM CMD SPACE)')
plt.title('Fourier basis')
plt.legend() 
plt.savefig(fig_path+'Fourier_correction_Eigenmode_space_singularvalue_truncation.png')


# test tip/tilt reco and higher order






















# TRY model_2 WITH  method='single_side_poke', or 'double_sided_poke'
#phase_ctrl.change_control_basis_parameters(  number_of_controlled_modes=140, basis_name ='Zonal', dm_control_diameter=None, dm_control_center=None,controller_label=None)
#phase_ctrl.build_control_model_2(zwfs, poke_amp = -0.1, label='ctrl_1', poke_method='double_sided_poke', inverse_method='MAP',  debug = True)
zonal_phase_ctrl.build_control_model_2(zwfs, poke_amp = zonal_poke_amp, label=method_label_1, poke_method=poke_method, inverse_method=inverse_method,  debug = True)
zernike_phase_ctrl.build_control_model_2(zwfs, poke_amp = zernike_poke_amp, label=method_label_1, poke_method=poke_method, inverse_method=inverse_method,  debug = True)
#phase_ctrl.build_control_model( zwfs , poke_amp = -0.15, label='ctrl_1', debug = True)  


# write fits to input into RTC
zwfs.write_reco_fits( zonal_phase_ctrl, method_label_1, save_path=data_path, save_label=method_label_1)

# have a look at the singular values and eigenmodes in DM and detector space 
zonal_phase_ctrl.plot_SVD_modes( zwfs, zonal_method_label_1, save_path=fig_path)


ctrl_method_label = zonal_method_label_1 
phase_ctrl = copy.deepcopy( zonal_phase_ctrl )

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
N0 = phase_ctrl.ctrl_parameters[ctrl_method_label]['ref_pupil_FPM_out']


# check reference images
fig,ax = plt.subplots(1,2,figsize=(10,5))
ax[0].imshow(I0); ax[0].set_title('I0')
ax[1].imshow( N0 ) ; ax[1].set_title('N0')
plt.savefig(fig_path + 'delme.png')

# reconstruct image from IM for particular actuators 
act = 65

for act in range( len(IM) ):
    tmp =  zwfs.pupil_pixel_filter.copy()
    imgrid = np.zeros(tmp.shape)
    imgrid[tmp] = IM[act] #

    fig,ax = plt.subplots(1,1,figsize=(5,5))
    im = ax.imshow( imgrid.reshape( zwfs.I0.shape ) ); ax.set_title(f'act {act}')
    plt.colorbar( im , ax=ax )
    #ax[1].imshow( N0 ) ; ax[1].set_title('N0')
    plt.savefig(fig_path + 'delme.png')
    plt.close() 
    _= input('next?')


print( f'issue that np.sum(p.I0)/np.sum(p.N0) = {np.sum(p.I0)/np.sum(p.N0)}, so loss on mask - this is important')

poke_amp = p.ctrl_parameters[label]['poke_amp']
amp = 0.05 # amp to apply
for mode_indx in range( len(IM) ) :#len(M2C)-1 ) :  

    mode_aberration = mode_basis.T[mode_indx]#   M2C.T[mode_indx]
    #plt.imshow( util.get_DM_command_in_2D(amp*mode_aberration));plt.colorbar();plt.show()
    
    dm_cmd_aber = zwfs.dm_shapes['flat_dm'] + amp * mode_aberration 

    zwfs.dm.send_data( dm_cmd_aber )
    time.sleep(0.1)
    #raw_img_list = []
    #for i in range( 10 ) :
    #    raw_img_list.append( zwfs.get_image() ) # @D, remember for control_phase method this needs to be flattened and filtered for pupil region
    raw_img = zwfs.get_image( apply_manual_reduction = True)  # np.median( raw_img_list, axis = 0) 
    # plt.figure() ; plt.imshow( raw_img ) ; plt.savefig( fig_path + f'delme.png') # <- signal?
    
    err_img = p.get_img_err( 1/np.mean(raw_img) * raw_img.reshape(-1)[zwfs.pupil_pixel_filter]  ) 
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





amp = 0.2

for mode_indx in range( 5 ) :#len(M2C)-1 ) :  

    mode_aberration = mode_basis.T[mode_indx]#   M2C.T[mode_indx]
    #plt.imshow( util.get_DM_command_in_2D(amp*mode_aberration));plt.colorbar();plt.show()
    
    dm_cmd_aber = zwfs.dm_shapes['flat_dm'] + amp * mode_aberration 

    zwfs.dm.send_data( dm_cmd_aber )
    time.sleep(0.1)
    #raw_img_list = []
    #for i in range( 10 ) :
    #    raw_img_list.append( zwfs.get_image() ) # @D, remember for control_phase method this needs to be flattened and filtered for pupil region
    raw_img = np.mean( zwfs.get_some_frames(number_of_frames = imgs_to_mean, apply_manual_reduction = True ) ,axis=0) #zwfs.get_image()
    # plt.figure() ; plt.imshow( raw_img ) ; plt.savefig( fig_path + f'delme.png') # <- signal?
    
    err_img = p.get_img_err( 1/np.mean(raw_img) * raw_img.reshape(-1)[zwfs.pupil_pixel_filter]  ) 
    
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
    
    cmd_res =  M2C @ mode_res  # <-- should be like this. 1/poke_amp * M2C @ mode_res # SHOULD I USE M2C_4reco here? 
    
    # WITH RESIDUALS 
    
    im_list = [util.get_DM_command_in_2D( amp * mode_aberration  ),raw_img - I0,  util.get_DM_command_in_2D( cmd_res ) ,util.get_DM_command_in_2D( amp * mode_aberration - cmd_res ) ]
    xlabel_list = [None, None, None, None]
    ylabel_list = [None, None, None, None]
    title_list = ['Aberration on DM', 'I-I0', 'reconstructed DM cmd', 'residual']
    cbar_label_list = ['DM command', 'ADU (Normalized)', 'DM command' , 'DM command' ] 
    savefig = fig_path + 'delme.png' #f'mode_reconstruction_images/phase_reconstruction_example_mode-{mode_indx}_basis-{phase_ctrl.config["basis"]}_ctrl_modes-{phase_ctrl.config["number_of_controlled_modes"]}ctrl_act_diam-{phase_ctrl.config["dm_control_diameter"]}_readout_mode-12x12.png'

    util.nice_heatmap_subplots( im_list , xlabel_list, ylabel_list, title_list, cbar_label_list, fontsize=15, axis_off=True, cbar_orientation = 'bottom', savefig=savefig)
    
    _ = input('press when ready to go to next moce ')
    