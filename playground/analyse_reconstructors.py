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

#experiment_label = 'experi_good_pupil/'
#if not os.path.exists(fig_path + experiment_label):
#   os.makedirs(fig_path+ experiment_label)

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



zwfs.deactive_cropping() # zwfs.set_camera_cropping(r1, r2, c1, c2 ) #<- use this for latency tests , set back after with zwfs.set_camera_cropping(0, 639, 0, 511 ) 
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
bad_pixels = zwfs.get_bad_pixel_indicies( no_frames = 2000, std_threshold = 25 , flatten=False) # std_threshold = 50

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
pupil_report = pupil_control.analyse_pupil_openloop( zwfs, debug = True, return_report = True, symmetric_pupil=False, std_below_med_threshold=1.4 )

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
plt.savefig(fig_path + f'registered_pupil_{tstamp}.png')


#init our phase controller (object that processes ZWFS images and outputs DM commands)
# its not actually 140 modes since edges pinned 
zonal_phase_ctrl = phase_control.phase_controller_1(config_file = None, basis_name = 'Zonal_pinned_edges', number_of_controlled_modes = 140) 

zernike_phase_ctrl_90 = phase_control.phase_controller_1(config_file = None, basis_name = 'Zernike_pinned_edges', number_of_controlled_modes = 90) 
zernike_phase_ctrl_20 = phase_control.phase_controller_1(config_file = None, basis_name = 'Zernike_pinned_edges', number_of_controlled_modes = 20) 
fourier_phase_ctrl_50 = phase_control.phase_controller_1(config_file = None, basis_name = 'fourier_pinned_edges', number_of_controlled_modes = 50)
fourier_phase_ctrl_20 = phase_control.phase_controller_1(config_file = None, basis_name = 'fourier_pinned_edges', number_of_controlled_modes = 20)

# to change basis : 
#phase_ctrl.change_control_basis_parameters( controller_label = ctrl_method_label, number_of_controlled_modes=phase_ctrl.config['number_of_controlled_modes'], basis_name='Zonal' , dm_control_diameter=None, dm_control_center=None)


zonal_dict = {'controller': zonal_phase_ctrl, 'poke_amp':0.07, 'poke_method':'double_sided_poke', 'inverse_method':'MAP', 'label':'zonal_0.07pokeamp_in-out_pokes_map' }
zernike_dict_90 = {'controller': zernike_phase_ctrl_90, 'poke_amp':0.2, 'poke_method':'double_sided_poke', 'inverse_method':'MAP', 'label':'zernike_0.2pokeamp_in-out_pokes_map' }
zernike_dict_20  = {'controller': zernike_phase_ctrl_20, 'poke_amp':0.2, 'poke_method':'double_sided_poke', 'inverse_method':'MAP', 'label':'zernike_0.2pokeamp_in-out_pokes_map' }
fourier_dict_50 = {'controller': fourier_phase_ctrl_50, 'poke_amp':0.2, 'poke_method':'double_sided_poke', 'inverse_method':'MAP', 'label':'fourier_0.2pokeamp_in-out_pokes_map' }
fourier_dict_20 = {'controller': fourier_phase_ctrl_20, 'poke_amp':0.2, 'poke_method':'double_sided_poke', 'inverse_method':'MAP', 'label':'fourier_0.2pokeamp_in-out_pokes_map' }
fourier_dict_20_pinv = {'controller': fourier_phase_ctrl_20, 'poke_amp':0.2, 'poke_method':'double_sided_poke', 'inverse_method':'pinv', 'label':'fourier_0.2pokeamp_in-out_pokes_pinv' }

build_dict = {
    'zonal':zonal_dict ,
    'zernike_20modes_map':zernike_dict_20,
    'fourier_50modes_map':fourier_dict_50,
    #'fourier_20modes_pinv':fourier_dict_20_pinv,
    'fourier_20modes_map':fourier_dict_20
}

# iter 10 , reduced bad pixel mask threshold from 50 - 25 . WORKED FOR TT!! at least in open loop reconstructor
#           no good pinv but MAP seems critical! used 10k frames for this 
# iter 11 , updated reconstructor fits to include RTT RHO M2C_4reco properly 
# iter 12 , fixed bug with R_TT in zonal basis (with addition of pinned_to_edge basis a case went missing in R_TT construction)
#       update4d zwfs reconstructor fits writing with M2C_reco etc. 
# iter 13 , trying fourier with more modes
# iter 14 , loading to rtc realized cropping state was set wrong from previous latency tests.. even though crop diabled the crop rows/cols still remain
iter = 14 

#subprocess.run()
# build and write them to fits 
for basis in  build_dict:

    current_path = fig_path + f'iter_{iter}_{phasemask_name}/{basis}_reconstructor/' # f'tmp/{tstamp.split("T")[0]}/' #'/home/baldr/Documents/baldr/ANU_demo_scripts/BALDR/figures/' 
    #data_path = f'tmp/{tstamp.split("T")[0]}/' #'/home/baldr/Documents/baldr/ANU_demo_scripts/BALDR/data/' 


    if not os.path.exists(current_path):
        os.makedirs(current_path)
    

    #p = build_dict[basis]['controller']
    label = build_dict[basis]['label']


    # ====== Building noise model (covariance of detector signals)
    build_dict[basis]['controller'].update_noise_model(zwfs, number_of_frames = 10000 )


    build_dict[basis]['controller'].build_control_model_2(
        zwfs, \
        poke_amp = build_dict[basis]['poke_amp'], label=label, \
        poke_method = build_dict[basis]['poke_method'], inverse_method= build_dict[basis]['inverse_method'],  debug = True \
        )


    zwfs.dm.send_data( zwfs.dm_shapes['flat_dm'] )
    time.sleep( 0.1 )


    # write fits to input into RTC
    zwfs.write_reco_fits( build_dict[basis]['controller'], label, save_path=current_path, save_label=label)

    # get an image associated with file of the pupils.
    I0, N0 = util.get_reference_images(zwfs, phasemask, theta_degrees=11.8, number_of_frames=256, \
compass = True, compass_origin=None, savefig=current_path + f'FPM-in-out_{phasemask_name}_{label}.png' )


# anmalyse them all 
for basis in build_dict:

    zwfs.dm.send_data(zwfs.dm_shapes['flat_dm'])

    current_path = fig_path + f'iter_{iter}_{phasemask_name}/{basis}_reconstructor/'  #fig_path + f'{basis}_reconstructor/' # current_path + f'{basis}_reconstructor'

    p = build_dict[basis]['controller']
    label = build_dict[basis]['label']

    # plot and save SVD 
    p.plot_SVD_modes( zwfs, label, save_path=current_path)

    # re-label for readability 
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

    # registered pupil 
    fig,ax = plt.subplots(1,2,figsize=(10,5))
    ax[0].imshow( N0 ) ; ax[0].set_title('phasemask out')
    ax[1].imshow( zwfs.pupil_pixel_filter.reshape(N0.shape)) ; ax[1].set_title('registered pupil')
    plt.savefig(current_path + f'registered_pupil_{tstamp}.png')

    ####
    # Estimate pixel noise, add this to reconstructor 
    ##
    zwfs.dm.send_data( zwfs.dm_shapes['flat_dm'] ) 
    time.sleep(0.05)
    frames = zwfs.get_some_frames(number_of_frames = 1000, apply_manual_reduction = True )

    frame_fits = fits.PrimaryHDU( frames ) 
    frame_fits.header.set('EXTNAME','FRAMES_FPM_IN')
    camera_info_dict = util.get_camera_info(zwfs.camera)
    for k,v in camera_info_dict.items():
        frame_fits.header.set(k,v)
    frame_fits.writeto( current_path + 'FRAMES_FPM_IN.fits')

    img_var = np.var( frames ,axis=0) #zwfs.get_image()
    img_mean = np.mean( frames ,axis=0) #zwfs.get_image()
    im_list = [img_mean, img_var]
    xlabel_list = [None, None]
    ylabel_list = [None, None]
    title_list = [None, None]
    cbar_label_list = [r'$\mu$ [adu]',r'$\sigma^2$ [adu$^2$]'] 
    savefig = current_path + 'image_statistics_FPM_in.png' #f'mode_reconstruction_images/phase_reconstruction_example_mode-{mode_indx}_basis-{phase_ctrl.config["basis"]}_ctrl_modes-{phase_ctrl.config["number_of_controlled_modes"]}ctrl_act_diam-{phase_ctrl.config["dm_control_diameter"]}_readout_mode-12x12.png'

    util.nice_heatmap_subplots( im_list , xlabel_list, ylabel_list, title_list, cbar_label_list, fontsize=15, axis_off=True, cbar_orientation = 'bottom', savefig=savefig)

    # assuming a flat DM => I = I0 => err_signal = 0 and only noise from detector, how much do we reconstruct on the mode? 
    img_std = np.sqrt(img_var.reshape(-1)[zwfs.pupil_pixels] )

    plt.figure(figsize=(8,5))
    std_grid = np.linspace(0,2,10)
    TTerrs = [R_TT @ (a * img_std ) for a in std_grid]
    plt.plot(std_grid,  [abs(e[0]) for e in TTerrs], label = 'tip')
    plt.plot(std_grid,  [abs(e[1]) for e in TTerrs], label = 'tilt')
    plt.ylabel( 'mode error',fontsize=15)
    plt.xlabel( r'$<\sigma_{pixels}>$',fontsize=15)
    plt.legend(fontsize=15) 
    plt.gca().tick_params(labelsize=15)
    plt.savefig( current_path + 'TT_error_vs_image_std.png')

    # Look at the signals in the interaction matrix (filtered by registered pupil pixels)
    for m in np.logspace( 0, np.log10( M2C.shape[1]-2 ), 5 ).astype(int):
        tmp =  zwfs.pupil_pixel_filter.copy()
        imgrid = np.zeros(tmp.shape)
        imgrid[tmp] = IM[m] #

        fig,ax = plt.subplots(1,2,figsize=(8,16))
        im = ax[0].imshow( imgrid.reshape( zwfs.I0.shape ) ); 
        ax[0].set_title(f'ZWFS signal \nin defined pupil')
        ax[1].set_title(f'DM mode : {m}')
        #plt.colorbar( im , ax=ax )
        im1 = ax[1].imshow( util.get_DM_command_in_2D( mode_basis.T[m] ) )
        #ax[1].imshow( N0 ) ; ax[1].set_title('N0')
        plt.savefig(current_path + f'{basis}_IM_{m}_signal.png')
        #plt.close() 
        #_= input('next?')


    # Test R_TT and R_H0 on IM signal 
    err_TT_1 = R_TT @ IM[0]
    err_TT_2 = R_TT @ IM[5]
    err_HO_1 = R_HO @ IM[0] # should be zero 
    err_HO_2 = R_HO @ IM[5]

    plt.figure() 
    plt.plot( err_TT_1, label=r'$RTT.IM_0$')
    plt.plot( err_TT_2,  label=r'$RTT.IM_5$')
    plt.plot( err_HO_1,  label=r'$RHO.IM_0$')
    plt.plot( err_HO_2,  label=r'$RHO.IM_5$')
    plt.axhline( poke_amp, color='k', ls = ":", label = 'probe amplitude')
    plt.legend()
    plt.xlabel('mode index')
    plt.ylabel('reconstructed amplitude')
    plt.savefig( current_path + f'{basis}_amp_reco_test_on_IM.png')

    # apply a mode and try reconstruct it 
    
    if 'zonal' not in basis: # then we use the naitive basis 
        ab_basis = M2C
    else: # then zonal - so we build the fourier basis so apply fourier modes as aberrations (not zonal)
        b = util.construct_command_basis( basis='fourier_pinned_edges', number_of_modes = 40, Nx_act_DM = 12, Nx_act_basis = 12, act_offset=(0,0), without_piston=True)
        ab_basis = poke_amp * b

    a_tt = 1.2 #TT amplitude 
    i_tt = 1 # index 

    a_ho1 = [0.5, 0.1, 1.2] # HO amplitudes
    # 3 random higher order modes 
    i_ho1 = list(np.random.choice(np.arange(2, 11), size=3, replace=False))  #[5,8,2] #indicies 

    # build the total command 
    cmd =  a_tt * ab_basis[:,i_tt] # note M2C here is already scaled by the IM poke amplitude (or should be)
    for a,i in zip( a_ho1, i_ho1):
        cmd += a * ab_basis[:,i] #note that M2C was scaled by poke amplitude - so coefficients are relative to this

    # seperate HO and TT commands to check the reconstructors are working right! 
    # look at higher order and tip/tilt components seperately 
    cmd_HO =  np.zeros( len(cmd) )
    for a,i in zip( a_ho1, i_ho1):
        cmd_HO += a * ab_basis[:,i] #note that M2C was scaled by poke amplitude - so coefficients are relative to this

    cmd_TT = a_tt * ab_basis[:,i_tt]

    #send the DM command 
    zwfs.dm.send_data( zwfs.dm_shapes['flat_dm'] + cmd ) 
    time.sleep(0.1)
    # now get some images 
    raw_img = np.mean( zwfs.get_some_frames(number_of_frames = 10, apply_manual_reduction = True ) ,axis=0) #zwfs.get_image()
       
    err_img = p.get_img_err( 1/np.mean(raw_img) * raw_img.reshape(-1)[zwfs.pupil_pixel_filter]  ) 
        
    # now reconstruct command with R_TT and R_HO 
    cmd_reco_TT = M2C @ R_TT @ err_img # R_TT goes to modal space and then M2C sends mode amplitude to DM command 
 
    cmd_reco_HO = M2C @ R_HO @ err_img

    # filtered pupil error rebuilt in square region ()
    tmp =  zwfs.pupil_pixel_filter.copy()
    imgrid = np.zeros(tmp.shape)
    imgrid[tmp] = err_img #
    imgrid = imgrid.reshape( zwfs.I0.shape )

    im_list = [ util.get_DM_command_in_2D(cmd) , imgrid, util.get_DM_command_in_2D(cmd_TT),\
     util.get_DM_command_in_2D(cmd_reco_TT), util.get_DM_command_in_2D(cmd_TT - cmd_reco_TT)] 
    #
    xlabel_list = [None, None, None, None, None]
    ylabel_list = [None, None, None, None, None]
    vlims= [ [np.nanmin(cmd),np.nanmax(cmd)],  [np.nanmin(imgrid),np.nanmax(imgrid)], \
       [np.nanmin(cmd_TT),np.nanmax(cmd_TT)],  [np.nanmin(cmd_reco_TT),np.nanmax(cmd_reco_TT)],[np.nanmin(cmd_TT),np.nanmax(cmd_TT)]]
    title_list = ['full DM \naberration', 'ZWFS signal' ,'tip/tilt \ncomponent', 'reconstruction', 'residual']
    cbar_label_list = ['DM units', 'ADU (Normalized)','DM units', 'DM units' , 'DM units' ] 
    savefig = current_path + f'{basis}_TT_reconstruction_test.png' #f'mode_reconstruction_images/phase_reconstruction_example_mode-{mode_indx}_basis-{phase_ctrl.config["basis"]}_ctrl_modes-{phase_ctrl.config["number_of_controlled_modes"]}ctrl_act_diam-{phase_ctrl.config["dm_control_diameter"]}_readout_mode-12x12.png'

    util.nice_heatmap_subplots( im_list , xlabel_list, ylabel_list, title_list, cbar_label_list, vlims = vlims, fontsize=15, axis_off=True, cbar_orientation = 'bottom', savefig=savefig)
    
    
    im_list = [ util.get_DM_command_in_2D( cmd ), imgrid, util.get_DM_command_in_2D(cmd_HO),\
     util.get_DM_command_in_2D(cmd_reco_HO), util.get_DM_command_in_2D(cmd_HO - cmd_reco_HO)] 
    #

    xlabel_list = [None, None, None, None, None]
    ylabel_list = [None, None, None, None, None]
    title_list = ['full DM \naberration', 'ZWFS signal' ,'HO \ncomponent', 'reconstruction', 'residual']
    cbar_label_list = ['DM units', 'ADU (Normalized)','DM units', 'DM units' , 'DM units' ] 
    savefig = current_path + f'{basis}_HO_reconstruction_test.png' #f'mode_reconstruction_images/phase_reconstruction_example_mode-{mode_indx}_basis-{phase_ctrl.config["basis"]}_ctrl_modes-{phase_ctrl.config["number_of_controlled_modes"]}ctrl_act_diam-{phase_ctrl.config["dm_control_diameter"]}_readout_mode-12x12.png'

    util.nice_heatmap_subplots( im_list , xlabel_list, ylabel_list, title_list, cbar_label_list, fontsize=15, axis_off=True, cbar_orientation = 'bottom', savefig=savefig)
    
    plt.close()

    # save as fits 
    reco_openloop_fits = fits.HDUList( [] )
    for thing, lab in zip( [cmd, imgrid, cmd_TT, cmd_reco_TT, cmd_HO, cmd_reco_HO], ['cmd', 'zwfs_signal', 'cmd_TT', 'cmd_reco_TT', 'cmd_HO', 'cmd_reco_HO']):

            tmp_fits = fits.PrimaryHDU( thing ) 
            tmp_fits.header.set('EXTNAME',lab)

            camera_info_dict = util.get_camera_info(zwfs.camera)
            for k,v in camera_info_dict.items():
                tmp_fits.header.set(k,v)

            reco_openloop_fits.append( tmp_fits )
    
    reco_openloop_fits.writeto( current_path + 'reco_open_loop_test.fits')

"""
###
## Apply some open loop corrections and look at image - are we stable? 
##
# flat DM reco. R_TT commands and start sending .. are we stable?  
basis = 'fourier_20modes_map'
p = build_dict[basis]['controller']
label = build_dict[basis]['label']

R_TT = p.ctrl_parameters[label]['R_TT'] 
R_HO = p.ctrl_parameters[label]['R_HO'] 

M2C = p.ctrl_parameters[label]['M2C_4reco'] 

zwfs.dm.send_data(zwfs.dm_shapes['flat_dm'])

for i in range(100):
    
    raw_img = np.mean( zwfs.get_some_frames(number_of_frames = 100, apply_manual_reduction = True ) ,axis=0) #zwfs.get_image()

    plt.figure(); plt.imshow( raw_img ); plt.savefig( fig_path + 'delme.png')

    _ = input ( "continue?")

    err_img = p.get_img_err( 1/np.mean(raw_img) * raw_img.reshape(-1)[zwfs.pupil_pixel_filter]  ) 
        
    amp_err = R_TT @ err_img
    print( 'estimated tip/tilt amplitudes = ', amp_err[0], amp_err[1])

    # now reconstruct command with R_TT and R_HO 
    cmd_reco_TT = M2C @ R_TT @ err_img # R_TT goes to modal space and then M2C sends mode amplitude to DM command 

    zwfs.dm.send_data(zwfs.dm_shapes['flat_dm'] + cmd_reco_TT)
    time.sleep(0.05)

"""



# finally we could try some closed loop 
#exit_all()
"""   
fig,ax = plt.subplots( 1,3 )
im0= ax[0].imshow( cmd_reco_TT.reshape(12,12) )
im1=ax[1].imshow( cmd_TT.reshape(12,12) )
im2=ax[2].imshow( cmd_TT.reshape(12,12) - cmd_reco_TT.reshape(12,12) )
plt.colorbar(im0, ax=ax[0])
plt.colorbar(im1, ax=ax[1])
plt.colorbar(im2, ax=ax[2])
ax[0].set_title('TT reconstructed')
ax[1].set_title('TT applied')
ax[2].set_title('residual')


fig,ax = plt.subplots( 1,3 )
im0=ax[0].imshow( cmd_reco_HO.reshape(12,12) )
im1=ax[1].imshow( cmd_HO.reshape(12,12) )
im2=ax[2].imshow( cmd_HO.reshape(12,12) - cmd_reco_HO.reshape(12,12) )
plt.colorbar(im0, ax=ax[0])
plt.colorbar(im1, ax=ax[1])
plt.colorbar(im2, ax=ax[2])
ax[0].set_title('HO reconstructed')
ax[1].set_title('HO applied')
ax[2].set_title('residual')
                



exit_all() 
"""

