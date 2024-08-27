import numpy as np
import glob 
from astropy.io import fits
import time
import os 
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


fig_path = f'tmp/{tstamp.split("T")[0]}/' #'/home/baldr/Documents/baldr/ANU_demo_scripts/BALDR/figures/' 
data_path = f'tmp/{tstamp.split("T")[0]}/' #'/home/baldr/Documents/baldr/ANU_demo_scripts/BALDR/data/' 


if not os.path.exists(fig_path):
   os.makedirs(fig_path)


# =====================
#   SETUP
# =====================
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



#
# other random method to try 
aa = np.std( poke_imgs, axis=(0,2,3) )
 plt.figure() ; plt.imshow( util.get_DM_command_in_2D(np.std( poke_imgs, axis=(0,2,3) ))) ;plt.colorbar(); plt.savefig( 'tmp/delme.png')

zwfs.dm.send_data( zwfs.dm_shapes['flat_dm'])

I0, N0 = util.get_reference_images(zwfs, phasemask, theta_degrees=11.8, number_of_frames=256, \
compass = True, compass_origin=None, savefig= f'tmp/0.delme_before.png' )

zwfs.dm.send_data( zwfs.dm_shapes['flat_dm'] + 0.005 * (aa-np.min(aa)) )

util.get_reference_images(zwfs, phasemask, theta_degrees=11.8, number_of_frames=256, \
compass = True, compass_origin=None, savefig='tmp/0.delme_after.png' )


# =====================
#   OPTIMIZING I0 - INTENSITY REFERENCE WITH PHASEMASK INSERTED IN BEAM
# =====================

experiment_label = 'optimize_ref_int_method_2/iteration_2'

tstamp = datetime.datetime.now().strftime("%d-%m-%YT%H.%M.%S")



fig_path = f'tmp/{tstamp.split("T")[0]}/{experiment_label}/' #'/home/baldr/Documents/baldr/ANU_demo_scripts/BALDR/figures/' 
data_path = f'tmp/{tstamp.split("T")[0]}/{experiment_label}/' #'/home/baldr/Documents/baldr/ANU_demo_scripts/BALDR/data/' 

if not os.path.exists(fig_path):
   os.makedirs(fig_path)



# x,y in compass referenced to DM right (+x), up (+y)
I0, N0 = util.get_reference_images(zwfs, phasemask, theta_degrees=11.8, number_of_frames=256, \
compass = True, compass_origin=None, savefig=fig_path + f'0.FPM-in-out_{phasemask_name}_before.png' )


# --- linear ramps 
# use baldr.
#recon_data = util.GET_BDR_RECON_DATA_INTERNAL(zwfs, number_amp_samples = 20, amp_max = 0.2,\
#number_images_recorded_per_cmd = 200, save_fits = data_path+f'pokeramp_data_MASK_{phasemask_name}_sydney_{tstamp}.fits') 

# recon_data = fits.open( data_path+'recon_data_LARGE_SECONDARY_19-04-2024T12.19.22.fits' )

recon_data = fits.open( '/home/heimdallr/Documents/asgard-alignment/tmp/27-08-2024/pokeramp_data_MASK_J3_sydney_27-08-2024T10.47.02.fits')


#zonal_fits = util.PROCESS_BDR_RECON_DATA_INTERNAL(recon_data, bad_pixels = ([],[]), active_dm_actuator_filter=active_dm_actuator_filter, debug=True, fig_path = fig_path , savefits= data_path+f'fitted_pokeramp_data_MASK_{phasemask_name}_sydney_{tstamp}.fits') 

#print( [z.header['EXTNAME'] for z in zonal_fits ] ) 

# -- prelims of reading in and labelling data 

debug = True 
# poke values used in linear ramp
No_ramps = int(recon_data['SEQUENCE_IMGS'].header['#ramp steps'])
max_ramp = float( recon_data['SEQUENCE_IMGS'].header['in-poke max amp'] )
min_ramp = float( recon_data['SEQUENCE_IMGS'].header['out-poke max amp'] ) 
ramp_values = np.linspace( min_ramp, max_ramp, No_ramps)

flat_dm_cmd = recon_data['FLAT_DM_CMD'].data

Nmodes_poked = int(recon_data[0].header['HIERARCH Nmodes_poked']) # can also see recon_data[0].header['RESHAPE']

Nact =  int(recon_data[0].header['HIERARCH Nact'])  

N0 = recon_data['FPM_OUT'].data
#P = np.sqrt( pupil ) # 
I0 = recon_data['FPM_IN'].data

# the first image is another reference I0 with FPM IN and flat DM
poke_imgs = recon_data['SEQUENCE_IMGS'].data[1:].reshape(No_ramps, 140, I0.shape[0], I0.shape[1])
#normalized
poke_imgs_norm = (poke_imgs - np.mean( poke_imgs ,axis = (0,1) ) ) / np.std( poke_imgs ,axis = (0,1))

# defining our DM pupil 10 actuator diameter circular
M2C = util.construct_command_basis( basis='Zernike', number_of_modes = 5, Nx_act_DM = 12,Nx_act_basis =12, act_offset= (0,0), without_piston=True) 
dm_pupil_filt = (abs(np.sum( M2C, axis=1 )) > 0 ).astype(bool)        

"""plt.figure( ) 
plt.imshow( np.std( poke_imgs ,axis = (0,1)) )
plt.colorbar(label = 'std pixels') 
plt.savefig( fig_path + 'delme.png')"""

# bad pixels above 10sigma or sigma = 0
recomended_bad_pixels = np.where( (np.std( poke_imgs_norm ,axis = (0,1)) > 10) + (np.std( poke_imgs ,axis = (0,1)) == 0 ))
print('recommended bad pixels (high or zero std) at :',recomended_bad_pixels )

if len(recomended_bad_pixels) > 0:
    
    bad_pixel_mask = np.ones(I0.shape)
    for ibad,jbad in list(zip(recomended_bad_pixels, recomended_bad_pixels)):
        bad_pixel_mask[ibad,jbad] = 0
        
    I0 *= bad_pixel_mask
    N0 *= bad_pixel_mask
    poke_imgs  = poke_imgs * bad_pixel_mask
    poke_imgs_norm = poke_imgs_norm * bad_pixel_mask


# amplitude index we use for registration
a0 = len(ramp_values)//2 - 1

# to get good threshold (reliable)
# each actuator should influence slightly the 9 actuators around it, so we would expect around 9 outliers 
the_tenth_list = []
for act_idx in np.where(  dm_pupil_filt ): #Where we define our DM pupil
    d = abs(poke_imgs[a0][act_idx] - poke_imgs[-a0][act_idx]).reshape(-1)
    the_tenth_list.append( np.sort( d[np.isfinite(d)] )[::-1][10] )

# no registration threshold
registration_threshold = np.mean( the_tenth_list ) / 4

Sw_x, Sw_y = 3,3 #+- pixels taken around region of peak influence. PICK ODD NUMBERS SO WELL CENTERED!   
act_img_mask_1x1 = {} #pixel with peak sensitivity to the actuator
act_img_mask_3x3 = {} # 3x3 region around pixel with peak sensitivity to the actuator
poor_registration_list = np.zeros(Nact).astype(bool) # list of actuators in control region that have poor registration 


for act_idx in range(Nact):
    # use difference in normalized image (check plt.figure(); plt.hist(poke_imgs_norm[0][0].reshape(-1)) ;plt.savefig(fig_path + 'delme.png'))
    delta =  poke_imgs[a0][act_idx] - poke_imgs[-a0][act_idx] 

    #plt.figure(); plt.imshow( delta ); plt.colorbar();plt.savefig(fig_path + 'delme.png')
    #act_idx = 65
    #plt.figure(); plt.hist(abs(poke_imgs_norm[a0][act_idx] - poke_imgs_norm[-a0][act_idx]).reshape(-1) ,bins=100) ;plt.yscale('log');plt.savefig(fig_path + 'delme.png')
    
    mask_3x3 = np.zeros( I0.shape )
    mask_1x1 = np.zeros( I0.shape )
    if dm_pupil_filt[act_idx]: #  if we decided actuator has strong influence on ZWFS image, we 
        peak_delta = np.nanmax( abs(delta) ) 
        if peak_delta > registration_threshold:
            i,j = np.unravel_index( np.argmax( abs(delta) ), I0.shape )
            mask_3x3[i-Sw_x-1: i+Sw_x, j-Sw_y-1:j+Sw_y] = 1 # keep centered, 
            mask_1x1[i,j] = 1 
            #mask *= 1/np.sum(mask[i-Sw_x-1: i+Sw_x, j-Sw_y-1:j+Sw_y]) #normalize by #pixels in window 
            act_img_mask_3x3[act_idx] = mask_3x3
            act_img_mask_1x1[act_idx] = mask_1x1
        else:
            poor_registration_list[act_idx] = True
            act_img_mask_3x3[act_idx] = mask_3x3 
            act_img_mask_1x1[act_idx] = mask_1x1 
    else :
        act_img_mask_3x3[act_idx] = mask_3x3 
        act_img_mask_1x1[act_idx] = mask_1x1 
        #act_flag[act_idx] = 0 
if debug:
    plt.figure()
    plt.title('pixel to actuator registration')
    plt.imshow( np.sum( list(act_img_mask_1x1.values()), axis = 0 ) )
    #plt.show()
    plt.savefig(  fig_path + f'1.pixel_to_actuator_registration_{tstamp}.png')  #f'process_fits_1_{tstamp}.png', bbox_inches='tight', dpi=300)


# turn our dictionary to a big pixel to command matrix 
P2C_1x1 = np.array([list(act_img_mask_1x1[act_idx].reshape(-1)) for act_idx in range(Nact)])
P2C_3x3 = np.array([list(act_img_mask_3x3[act_idx].reshape(-1)) for act_idx in range(Nact)])


# check the active region 
fig,ax = plt.subplots(1,1)
ax.imshow( util.get_DM_command_in_2D( dm_pupil_filt  ) )
ax.set_title('active DM region')
ax.grid(True, which='minor',axis='both', linestyle='-', color='k' ,lw=3)
ax.set_xticks( np.arange(12) - 0.5 , minor=True)
ax.set_yticks( np.arange(12) - 0.5 , minor=True)

plt.savefig(  fig_path + f'2.active_DM_actuators_{tstamp}.png', bbox_inches='tight', dpi=300)
#plt.savefig( fig_path + f'active_DM_region_{tstamp}.png' , bbox_inches='tight', dpi=300) 
# check the well registered DM region : 

fig,ax = plt.subplots(1,1)
ax.imshow( util.get_DM_command_in_2D( np.sum( P2C_1x1, axis=1 )))
ax.set_title('well registered actuators')
ax.grid(True, which='minor',axis='both', linestyle='-', color='k',lw=2 )
ax.set_xticks( np.arange(12) - 0.5 , minor=True)
ax.set_yticks( np.arange(12) - 0.5 , minor=True)

plt.savefig(  fig_path + f'3.well_registered_actuators_{tstamp}.png', bbox_inches='tight', dpi=300)

#plt.savefig( fig_path + f'poorly_registered_actuators_{tstamp}.png' , bbox_inches='tight', dpi=300) 

# check poorly registered actuators: 
fig,ax = plt.subplots(1,1)
ax.imshow( util.get_DM_command_in_2D(poor_registration_list) )
ax.set_title('poorly registered actuators')
ax.grid(True, which='minor',axis='both', linestyle='-', color='k', lw=2 )
ax.set_xticks( np.arange(12) - 0.5 , minor=True)
ax.set_yticks( np.arange(12) - 0.5 , minor=True)

plt.savefig(  fig_path + f'4.poorly_registered_actuators_{tstamp}.png', bbox_inches='tight', dpi=300)






def Ic_model_constrained(x, A, B, F, mu):

    # force mu between 0-360 degrees 
    #mu = np.arccos( np.cos( mu ) )

    #penalty = 0
    # F and B are forced to be positive via a fit penality
    #if (F < 0) or (B > 0): # F and mu can be correlated so constrain the quadrants 
    #    penalty = 1e3
    I = A + B * np.cos(F * x + mu) #+ penalty
    return I 

param_dict = {}
cov_dict = {}
fit_residuals = []
nofit_list = []
for act_idx in range(len(flat_dm_cmd)): 

    #Note that if we use the P2C_3x3 we need to normalize it 1/(3*3) * P2C_3x3
    if dm_pupil_filt[act_idx] * ( ~poor_registration_list)[act_idx]:

        # -- we do this with matrix multiplication using  mask_matrix
        #P_i = np.sum( act_img_mask[act_idx] * pupil ) #Flat DM with FPM OUT 
        #P_i = mean_filtered_pupil.copy() # just consider mean pupil! 
    
        I_i = np.array( [P2C_1x1[act_idx] @ poke_imgs[i][act_idx].reshape(-1) for i in  range(len(ramp_values))] ) #np.array( [np.sum( act_img_mask[act_idx] * poke_imgs[i][act_idx] ) for i in range(len(ramp_values))] ) #spatially filtered sum of intensities per actuator cmds 

        #re-label and filter to capture best linear range 
        x_data = ramp_values[2:-2].copy()
        y_data = I_i[2:-2].copy()

        #plt.figure(); plt.plot(x_data, y_data) ; plt.savefig(fig_path+'delme.png')
        #_ = input('asd')
        """
        # brite grid search to get initial values 
        A_grid = np.linspace( np.min(I_i), np.max(I_i), 10)
        B_grid=np.linspace( 0, np.max(I_i) - np.min(I_i), 10) 
        F_grid = np.linspace(5,20,10)
        mu_grid=np.linspace(0,2*np.pi,10)


        A, B , F, mu = np.meshgrid( A_grid, B_grid,  F_grid, mu_grid )

        # Calculate the residuals
        residuals = np.abs(y_data - A[..., np.newaxis] + B[..., np.newaxis] * np.cos(F[..., np.newaxis] * x_data + mu[..., np.newaxis]))

        # Sum the residuals over x to get the total residual for each A, B, F, mu
        total_residuals = np.sum(residuals, axis=-1)

        # Find the indices of the minimum residual
        min_indices = np.unravel_index(np.argmin(total_residuals), total_residuals.shape)

        initial_A = A_grid[min_indices[0]]
        initial_B = B_grid[min_indices[1]]
        initial_F = F_grid[min_indices[2]]
        initial_mu = mu_grid[min_indices[3]]

        initial_guess = [initial_A , initial_B,  initial_F,  initial_mu]
        #initial_guess = [7, 2, 15, 2.4] #[0.5, 0.5, 15, 2.4]  #A_opt, B_opt, F_opt, mu_opt  ( S = A+B*cos(F*x + mu) )

        """
        # Initial guesses for the parameters [A, B, F, mu]
        initial_guess = [np.mean(y_data), -np.ptp(y_data) / 2, 15, 2]  # Adjust as needed

        # Set bounds: A can be any value, B > 0, F > 0, mu between 0 and 2*pi
        bounds = ([-np.inf, -np.inf, 0, -2*np.pi], [np.inf, np.inf, np.inf, 2 * np.pi])

        # Use least_squares to optimize the parameters within the bounds
        #result = least_squares(residuals, initial_guess, bounds=bounds, args=(x_data, y_data))

        # Extract optimized parameters
        #optimal_A, optimal_B, optimal_F, optimal_mu = result.x

        #try:
        if 1:
            # FIT 
            popt, pcov = curve_fit(Ic_model_constrained, x_data, y_data, p0=initial_guess)
            #popt = least_squares(residuals, initial_guess, bounds=bounds, args=(x_data, y_data))
            # Extract the optimized parameters explictly to measure residuals
            A_opt, B_opt, F_opt, mu_opt = popt #.x

            # Calculate the covariance matrix
            #J = popt.jac  # Jacobian matrix at the solution
            #residual_variance = 2 * popt.cost / (len(y_data) - len(initial_guess))
            #pcov = residual_variance * np.linalg.pinv(J.T @ J)


            # STORE FITS 
            param_dict[act_idx] = popt
            cov_dict[act_idx] = pcov 
            # also record fit residuals 
            fit_residuals.append( I_i - Ic_model_constrained(ramp_values, A_opt, B_opt, F_opt, mu_opt) )

            if  debug : # None: #65:
                #plt.cla()
                #PLOT
                fig1 = plt.figure(figsize=(8,5))
                #Plot Data-model
                frame1=fig1.add_axes((.1,.3,.8,.6))
                #xstart, ystart, xend, yend [units are fraction of the image frame, from bottom left corner]
                #frame1.plot(ramp_values, func(ramp_values, A_opt, B_opt, F_opt, mu_opt)/ Pi[-1]**2 , label='fit (actuator 65)') #Noisy data
                #frame1.plot(ramp_values, I_v_ramp / Pi[-1]**2 ,marker = 'o',linestyle='-', label='measured (actuator 65)') #Best fit model
                frame1.plot(x_data, Ic_model_constrained(x_data, A_opt, B_opt, F_opt, mu_opt) , label=f'fit (actuator {act_idx})') #Noisy data
                frame1.plot(x_data, y_data ,marker = 'o',linestyle='-', label=f'measured (actuator {act_idx})') 
                #Remove x-tic labels for the first frame
                plt.grid()
                frame1.legend(fontsize=15)
                #frame1.set_ylabel( 'normalized Intensity \n' + r'[$I/P^2$]',fontsize=15)
                frame1.set_ylabel( 'Intensity',fontsize=15)
                frame1.set_xticklabels([])
                frame1.tick_params(labelsize=14)
                #Residual plot
                #difference = I_v_ramp / Pi[-1]**2  - func(ramp_values, A_opt, B_opt, F_opt, mu_opt)/ Pi[-1]**2 
                difference = y_data - Ic_model_constrained(x_data, A_opt, B_opt, F_opt, mu_opt) 
                frame2=fig1.add_axes((.1,.1,.8,.2))     
                frame2.tick_params(labelsize=14)
                frame2.plot(x_data, difference, color='k')
                plt.grid()
                plt.xlabel('Normalized DM Command ' + r'[$\Delta c$]',fontsize=15)
                plt.ylabel('Residual',fontsize=15)
                #plt.tight_layout()

                if not os.path.exists(fig_path+'individual_fits/'):
                    os.makedirs(fig_path+'individual_fits/')
                plt.savefig(fig_path  + f'individual_fits/intensity_model_vs_measured_act_{act_idx}_norm.png', dpi=300, bbox_inches='tight')


        #except:
        #     print(f'\n!!!!!!!!!!!!\nfit failed for actuator {act_idx}\n!!!!!!!!!!!!\nanalyse plot to try understand why')

        #     """nofit_list.append( act_idx ) 
        #     fig1, ax1 = plt.subplots(1,1)
        #     ax1.plot( ramp_values, S )
        #     ax1.set_title('could not fit this!') """
            


#plt.figure(); plt.plot( pd.DataFrame( param_dict ).T[1], pd.DataFrame( param_dict ).T[3] * 180/np.pi,'.' ); plt.savefig(fig_path + 'delme.png')
if debug: # plot mosaic 

    Nrows = np.ceil( len(param_dict)**0.5).astype(int)
    fig,ax = plt.subplots(Nrows,Nrows,figsize=(20,20))
    axx = ax.reshape(-1)
    for j, a in enumerate(param_dict):
        I_i = np.array( [P2C_1x1[a] @ poke_imgs[i][a].reshape(-1) for i in  range(len(ramp_values))] )
        A_opt, B_opt, F_opt, mu_opt = param_dict[a]

        axx[j].plot( ramp_values, Ic_model_constrained(ramp_values, A_opt, B_opt, F_opt, mu_opt) ,label=f'fit (act{act_idx})') 
        axx[j].plot( ramp_values, I_i ,label=f'measured (act{act_idx})' )
        axx[j].axis('off')
    j=0 #axx index

    plt.savefig( fig_path + f'5.fit_mosaic_{tstamp}.png' , bbox_inches='tight', dpi=300) 
    #plt.show() 


    # CORNER PLOT 
    #labels = ['Q', 'W', 'F', r'$\mu$']
    corner.corner( np.array(list( param_dict.values() )), quantiles=[0.16,0.5,0.84], show_titles=True, labels = ['A', 'B', 'F', r'$\mu$'] ,range = [(0,4*np.mean(y_data)),(-2*(np.max(y_data)-np.min(y_data)), 0 ) , (5,20), (0,6) ] ) # range = [(0,4*np.mean(y_data)),(0, 2*(np.max(y_data)-np.min(y_data)) ) , (5,20), (0,6) ] #, range = [(2*np.min(S), 102*np.max(S)), (0, 2*(np.max(S) - np.min(S)) ), (5, 20), (-3,3)] ) #['Q [adu]', 'W [adu/cos(rad)]', 'F [rad/cmd]', r'$\mu$ [rad]']
    plt.savefig( fig_path + f'6.corner_plot_of_fitted_parameters_{tstamp}.png', bbox_inches='tight', dpi=300)
    plt.show()



#mu_array = np.array([param_dict[act][-1] for act in param_dict])
new_flat = np.nan * flat_dm_cmd.copy()
target_rad =  np.deg2rad( 270 )
for act in param_dict:
    x0 = flat_dm_cmd[act] # reference flat used when ramping actuators
    F = param_dict[act][-2]
    mu = param_dict[act][-1]
    if F > 1:
        # Calibrating new flat = x0+x
        #F.(x0 + x) + mu = target_rad # most sensitive part of cosine curve 
        
        new_flat[act] = (target_rad - mu)*1/F # np.arccos( np.cos(target_rad - mu) ) * 1/F #(3*np.pi/2 - mu) * 1/F  )

        #new_flat[act] = x + x0

# revert any prohibited values to the original DM flat 
#new_flat[(new_flat>1) + (new_flat<0)] = flat_dm_cmd[(new_flat>1) + (new_flat<0)]

# plot 
plt.figure(); plt.imshow( util.get_DM_command_in_2D( new_flat )); plt.colorbar(); plt.savefig(fig_path + f'7.newflat_{tstamp}.png')
plt.figure(); plt.imshow( util.get_DM_command_in_2D( flat_dm_cmd )); plt.colorbar(); plt.savefig(fig_path + f'8.original_flat_{tstamp}.png')


# interpolate
nan_mask = np.isnan( util.get_DM_command_in_2D( new_flat ) )
nearest_index = distance_transform_edt(nan_mask, return_distances=False, return_indices=True)

# Use the indices to replace NaNs with the nearest non-NaN values
filled_data = util.get_DM_command_in_2D( new_flat )[tuple(nearest_index)]

#corner_indices = [0, 12-1, 12 * (12-1), 12*12-1]
corner_flat_indices = [0, 12-1, 12 * (12-1), 12*12-1]

# Remove the corner elements from the flattened array
new_flat_interpolated = np.delete(filled_data.reshape(-1), corner_flat_indices)

plt.figure();plt.imshow(  util.get_DM_command_in_2D(  new_flat_interpolated ) ); plt.colorbar(); plt.savefig(fig_path + f'9.newflat_interp_{tstamp}.png')


zwfs.dm.send_data( new_flat_interpolated  )
time.sleep(0.1)

I0, N0 = util.get_reference_images(zwfs, phasemask, theta_degrees=11.8, number_of_frames=256, \
compass = True, compass_origin=None, savefig=fig_path + f'10.FPM-in-out_{phasemask_name}_after.png' )





# WRITING TO FITS 

output_fits = fits.HDUList( [] )

# reference images 
N0_fits = fits.PrimaryHDU( N0 )
N0_fits.header.set('EXTNAME','FPM OUT REF')
N0_fits.header.set('WHAT IS','ref int. with FPM out')

I0_fits = fits.PrimaryHDU( I0 )
I0_fits.header.set('EXTNAME','FPM IN REF')
I0_fits.header.set('WHAT IS','ref int. with FPM in')

# output fits files 
P2C_fits = fits.PrimaryHDU( np.array([P2C_1x1, P2C_3x3]) )
P2C_fits.header.set('EXTNAME','P2C')
P2C_fits.header.set('WHAT IS','pixel to DM actuator register')
P2C_fits.header.set('index 0','P2C_1x1') 
P2C_fits.header.set('index 1','P2C_3x3')    

#fitted parameters
param_fits = fits.PrimaryHDU( np.array(list( param_dict.values() )) )
param_fits.header.set('EXTNAME','FITTED_PARAMS')
param_fits.header.set('COL0','A [adu]')
param_fits.header.set('COL1','B [adu]')
param_fits.header.set('COL2','F [rad/cmd]')
param_fits.header.set('COL4','mu [rad]')
if len(nofit_list)!=0:
    for i, act_idx in enumerate(nofit_list):
        param_fits.header.set(f'{i}_fit_fail_act', act_idx)
    
#covariances
cov_fits = fits.PrimaryHDU( np.array(list(cov_dict.values())) )
cov_fits.header.set('EXTNAME','FIT_COV')
# residuals 
res_fits = fits.PrimaryHDU( np.array(fit_residuals) )
res_fits.header.set('EXTNAME','FIT_RESIDUALS')

#DM regions 
dm_fit_regions = fits.PrimaryHDU( np.array( [dm_pupil_filt, dm_pupil_filt*(~poor_registration_list), poor_registration_list] ).astype(int) )
dm_fit_regions.header.set('EXTNAME','DM_REGISTRATION_REGIONS')
dm_fit_regions.header.set('registration_threshold',registration_threshold)
dm_fit_regions.header.set('index 0 ','active_DM_region')   
dm_fit_regions.header.set('index 1 ','well registered actuators') 
dm_fit_regions.header.set('index 2 ','poor registered actuators') 
#
new_flat = fits.PrimaryHDU( np.array(new_flat_interpolated ) )
new_flat.header.set('EXTNAME','NEW_FLAT')


for f in [N0_fits, I0_fits, P2C_fits, param_fits, cov_fits,res_fits, dm_fit_regions, new_flat ]:
    output_fits.append( f ) 


output_fits.writeto( fig_path + f'newflat_fit_{tstamp}.fits', overwrite=True )  #data_path + 'ZWFS_internal_calibration.fits'

