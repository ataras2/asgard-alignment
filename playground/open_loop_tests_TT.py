"""

Using more calibrated flat 
update zeropoints 

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
phasemask.update_mask_position( phasemask_name )
phasemask.phasemask.update_all_mask_positions_relative_to_current( phasemask_name, 'phase_positions_beam_3 original_DONT_DELETE.json')
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


tint = 0.005 # SNR ~ 50 with cred2, bypass beamsplitters
fps = 100
zwfs.deactive_cropping() # zwfs.set_camera_cropping(r1, r2, c1, c2 ) #<- use this for latency tests , set back after with zwfs.set_camera_cropping(0, 639, 0, 511 ) 
zwfs.set_camera_dit( tint );time.sleep(0.2)
zwfs.set_camera_fps( fps );time.sleep(0.2)
zwfs.set_sensitivity('high');time.sleep(0.2)
zwfs.enable_frame_tag(tag = True);time.sleep(0.2)
zwfs.bias_off();time.sleep(0.2)
zwfs.flat_off();time.sleep(0.2)



zwfs.start_camera()

## ------- Calibrate detector (dark, badpixels)
# Source should be out
# at sydney move 01 X-LSM150A-SE03 to 133.07mm
zwfs.build_manual_dark()

# get our bad pixels 
# std_threshold = 25 at 1ms is good 
bad_pixels = zwfs.get_bad_pixel_indicies( no_frames = 200, std_threshold = 40 , flatten=False) # std_threshold = 50

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
#itera = 8 # retaking IM to go more closed loop dynamic 

#itera = 1 # 11/9/ - try get a good data set of TT locking 

#itera = 1 # self calibrated dm flat
#itera = 2 # bcm dm flat - see lots of drifts
#itera = 3 # Bmc flat - try go straight to reco , looked better 
#itera = 4 # do static tip / tilt with R_TT - adam in lab 
itera = 5 # close TT on nothing

# timestamp
tstamp = datetime.datetime.now().strftime("%d-%m-%YT%H.%M.%S")

fig_path = f'tmp/{tstamp.split("T")[0]}/'

exper_path = f'open_loop_{itera}/'

current_path = fig_path + exper_path +  "CL_static_1/" 

# setup paths 
if not os.path.exists(fig_path + exper_path ):
   os.makedirs(fig_path + exper_path )


print( 'dm shapes:' ,zwfs.dm_shapes.keys() )

########################
# Define our DM flat

#aa = pd.read_csv( 'DMShapes/BEST_calibrated_DMflat_12-09-2024T10.04.32.csv', header=None)

dm_flat = zwfs.dm_shapes['17DW019#122_FLAT_MAP_COMMANDS']
#zwfs.dm_shapes['17DW019#122_FLAT_MAP_COMMANDS'] # 
########################


zwfs.dm.send_data( dm_flat )
phasemask_centering_tool.move_relative_and_get_image(zwfs, phasemask, savefigName=fig_path + 'delme.png')

# TO update positions and write file if needed
# phasemask.update_mask_position( 'J3' )
# phasemask.update_all_mask_positions_relative_to_current( phasemask_name, 'phase_positions_beam_3 original_DONT_DELETE.json')
# phasemask.write_current_mask_positions()

# == init pupil region classification  
#init our pupil controller (object that processes ZWFS images and outputs VCM commands)
pupil_ctrl = pupil_control.pupil_controller_1(config_file = None)

#analyse pupil and decide if it is ok. This must be done before reconstructor
pupil_report = pupil_control.analyse_pupil_openloop( zwfs, debug = False, return_report = True, symmetric_pupil=False, std_below_med_threshold=1. )

if pupil_report['pupil_quality_flag'] == 1: 
    zwfs.update_reference_regions_in_img( pupil_report ) # 




# last minute check 
# zwfs.dm.send_data(zwfs.dm_shapes['flat_dm'] + 0.1 * zwfs.dm_shapes['four_torres'])
zwfs.dm.send_data( dm_flat )
#phasemask_centering_tool.move_relative_and_get_image(zwfs, phasemask, savefigName=fig_path + 'delme.png')


# BUILD THE RECONSTRUCTOR HERE 
now_tmp = datetime.datetime.now().strftime("%d-%m-%YT%H.%M.%S")
I0_now, N0_now = util.get_reference_images(zwfs, phasemask, theta_degrees=11.8, number_of_frames=256, \
compass = True, compass_origin=None, savefig = fig_path + exper_path + f'FPM-in-out_{phasemask_name}_{now_tmp}.png' )


modal_basis = util.construct_command_basis('Zonal_pinned_edges')


####
# NOTE HERE WE BUILD WITH REAL REGISTRATION 
###
poke_amp = 0.05
IM_list = []
method='double_sided'

if method=='double_sided':

    # the IM zero point 
    I0 = np.mean( zwfs.get_some_frames(number_of_frames = 256, apply_manual_reduction = True ) ,axis = 0 )

    for i,m in enumerate(modal_basis.T):
        print(f'executing cmd {i}/{len(modal_basis)}')
        I_plus_list = []
        I_minus_list = []
        imgs_to_mean = 10
        for sign in [(-1)**n for n in range(10)]: #[-1,1]:
            zwfs.dm.send_data( list( dm_flat+ sign * poke_amp/2 * m )  )
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
            zwfs.dm.send_data( list( dm_flat+ sign * poke_amp * m )  )
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




#def update_IM( ):
    # I0 = N0 + B + 2N0**0.5Bcos(X), IM signal is 2AB( cos(+e)-cos(-e) ) ~ AB e X . X higher order term
    # to update A_n, B_n 


IM= np.array( IM_list ).T # 1/poke_amp * 

zwfs.dm.send_data( dm_flat )

M2C_0 = modal_basis.T


## WE CAN DO ALL THIS WITH DIFFERENT PUPIL REGISTRATIONS - good to test!!! 
pupil_pixel_shift=0
pupil_pixel_filter = np.roll(zwfs.pupil_pixel_filter, shift=pupil_pixel_shift, axis=0)
pupil_pixels = np.where( pupil_pixel_filter )[0]  #pupil_pixels.copy() 

# 16 is measured from data , 1100 rough central wavelength of measurement 
cmd2opd = 16 * 1100 / (2*np.pi) #3200 # to go to cmd space to nm OPD 


    
# ---- Static TT start static


if not os.path.exists(current_path ):
   os.makedirs(current_path)


if 1:

    U,S,Vt = np.linalg.svd( IM.T , full_matrices=True)

    #singular values
    plt.figure() 
    plt.semilogy(S) #/np.max(S))
    #plt.axvline( np.pi * (10/2)**2, color='k', ls=':', label='number of actuators in pupil')
    plt.legend() 
    plt.xlabel('mode index')
    plt.ylabel('singular values')

    plt.savefig(current_path + f'singularvalues_{tstamp}.png', bbox_inches='tight', dpi=200)
    plt.show()
    
    # THE IMAGE MODES 
    n_row = round( np.sqrt( M2C_0.shape[0]) ) - 1
    fig,ax = plt.subplots(n_row  ,n_row ,figsize=(30,30))
    plt.subplots_adjust(hspace=0.1,wspace=0.1)
    for i,axx in enumerate(ax.reshape(-1)):
        # we filtered circle on grid, so need to put back in grid
        tmp =  pupil_pixel_filter.copy()
        vtgrid = np.zeros(tmp.shape)
        vtgrid[tmp] = Vt[i]
        r1,r2,c1,c2 = 10,-10,10,-10
        axx.imshow( vtgrid.reshape(zwfs.I0.shape )[r1:r2,c1:c2] ) #cp_x2-cp_x1,cp_y2-cp_y1) )
        #axx.set_title(f'\n\n\nmode {i}, S={round(S[i]/np.max(S),3)}',fontsize=5)
        #
        axx.text( 10,10,f'{i}',color='w',fontsize=4)
        axx.text( 10,20,f'S={round(S[i]/np.max(S),3)}',color='w',fontsize=4)
        axx.axis('off')
        #plt.legend(ax=axx)
    plt.tight_layout()

    plt.savefig(current_path + f'det_eignmodes_{tstamp}.png',bbox_inches='tight',dpi=200)
    plt.show()
    
    # THE DM MODES 

    # NOTE: if not zonal (modal) i might need M2C to get this to dm space 
    # if zonal M2C is just identity matrix. 
    fig,ax = plt.subplots(n_row, n_row, figsize=(30,30))
    plt.subplots_adjust(hspace=0.1,wspace=0.1)
    for i,axx in enumerate(ax.reshape(-1)):
        axx.imshow( util.get_DM_command_in_2D( M2C_0.T @ U.T[i] ) )
        #axx.set_title(f'mode {i}, S={round(S[i]/np.max(S),3)}')
        axx.text( 1,2,f'{i}',color='w',fontsize=6)
        axx.text( 1,3,f'S={round(S[i]/np.max(S),3)}',color='w',fontsize=6)
        axx.axis('off')
        #plt.legend(ax=axx)
    plt.tight_layout()
    plt.savefig(current_path + f'dm_eignmodes_{tstamp}.png',bbox_inches='tight',dpi=200)
    plt.show()



# check centering 
zwfs.dm.send_data(dm_flat)
phasemask_centering_tool.move_relative_and_get_image(zwfs, phasemask, savefigName=fig_path + 'delme.png')

# get new reference 
I0_now, N0_now = util.get_reference_images(zwfs, phasemask, theta_degrees=11.8, number_of_frames=256, \
compass = True, compass_origin=None, savefig= None )

zwfs.dm.send_data(dm_flat)

# ------ check centering on DM 

zwfs.dm.send_data( dm_flat + 0.1*zwfs.dm_shapes['four_torres'])
#phasemask_centering_tool.move_relative_and_get_image(zwfs, phasemask, savefigName=fig_path + 'delme.png')
time.sleep(0.1)
# check reference 
now_tmp = datetime.datetime.now().strftime("%d-%m-%YT%H.%M.%S")
I0_torre, N0_torre = util.get_reference_images(zwfs, phasemask, theta_degrees=11.8, number_of_frames=256, \
compass = True, compass_origin=None, savefig= fig_path + exper_path + f'DM_centering_four_torres_{now_tmp}.png' )

# --- DM centering
zwfs.dm.send_data( dm_flat + 0.1*zwfs.dm_shapes['Crosshair140'])
#phasemask_centering_tool.move_relative_and_get_image(zwfs, phasemask, savefigName=fig_path + 'delme.png')
time.sleep(0.1)
# check reference 
now_tmp = datetime.datetime.now().strftime("%d-%m-%YT%H.%M.%S")
I0_cross, N0_cross = util.get_reference_images(zwfs, phasemask, theta_degrees=11.8, number_of_frames=256, \
compass = True, compass_origin=None, savefig= fig_path + exper_path + f'DM_centering_crosshairs_{now_tmp}.png' )

zwfs.dm.send_data( dm_flat ) 


# -- define region outside out pupil to sample strehl using theoretical pupil
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

pupil_outer_perim_filter = (~zwfs.bad_pixel_filter * (abs( I0_theory - N0_theory ) > 0.02 ).reshape(-1) * (~zwfs.pupil_pixel_filter) )

plt.figure()
plt.imshow( pupil_outer_perim_filter.reshape(zwfs.I0.shape) ) 
plt.savefig( current_path + 'outer_pupil_filter.png')




# -- Build our matricies 

U, S, Vt = np.linalg.svd( IM, full_matrices=False)

Smax = 80
R  = (Vt.T * [1/ss if i < Smax else 0 for i,ss in enumerate(S)])  @ U.T

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

plot_intermediate = False
no_iterations = 300
close_after = 100 

pid.reset() 
leak.reset()



#disturb_basis = util.construct_command_basis( 'fourier_pinned_edges')
disturbance_cmd = 0. * TT_vectors[:,0]

#dm_pupil_filt * 
zwfs.dm.send_data(dm_flat+ disturbance_cmd) # only apply in registered pupil 
time.sleep(0.1)

kpTT = 1
kiTT = 0
#for kpTT in np.linspace( 0, 1 , 5):
#    for kiTT in [0, 0.2, 0.5]:
#        zwfs.dm.send_data(dm_flat+ disturbance_cmd)

now_tmp = datetime.datetime.now().strftime("%d-%m-%YT%H.%M.%S")
explabel = f'closed_loop_TT_on_nothing_telemetry_{kpTT}_{kiTT}_{now_tmp}'

for it in range(no_iterations):
    
    if it > close_after : # close after 
        pid.kp = kpTT * np.ones( I2M_TT.shape[0] )
        pid.ki = kiTT * np.ones( I2M_TT.shape[0] )
        
        #leak.rho[2:5] = 0.2 #* np.ones( I2M_HO.shape[0] )
        #leak.kp[2:5] = 0.5
        if plot_intermediate :
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

    i = np.mean( zwfs.get_some_frames(number_of_frames=5, apply_manual_reduction=True),axis=0) #z.detection_chain( test_field, FPM_on=True, include_shotnoise=True, ph_per_s_per_m2_per_nm=True, grids_aligned=True, replace_nan_with=0 )
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

    # safety 
    if np.max( c_TT + c_HO ) > 0.8: 
        break
    #c = R @ sig

    zwfs.dm.send_data( dm_flat+ disturbance_cmd - c_HO - c_TT ) # same way to rtc PID 
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
    
    #atm_disturb_list.append( scrn.scrn )
    dm_disturb_list.append( disturbance_cmd )
    
    residual_list.append( residual )
    rmse_list.append( rmse )
    flux_outside_pupil_list.append( np.var( sig.reshape(-1)[pupil_outer_perim_filter] ) )
    print( it, f'rmse = {rmse}, flux outside = {flux_outside_pupil_list[-1]}' )


plt.figure()
plt.plot( e_TT_list )
plt.axvline( close_after ,color='k',ls=':',label='close TT')
plt.legend()
plt.ylabel('mode error')
plt.xlabel('iterations')
#plt.savefig(fig_path + 'delme.png')
plt.savefig(current_path + f'mode_err_{explabel}.png')
# write telemetry to file 


plt.figure()
plt.plot( c_TT_list , alpha=0.1, color='k')
plt.axvline( close_after , color='k', ls=':',label='close TT')
plt.legend()
plt.ylabel('actuator cmds')
plt.xlabel('iterations')
#plt.savefig(fig_path + 'delme.png')
plt.savefig(current_path + f'command_ts_{explabel}.png')
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
hdul.writeto(current_path + f'{explabel}.fits', overwrite=True)





























# # ---- Dynamic TT start static

# current_path = fig_path + exper_path +  "5_CL_tip-tilt/" 
# if not os.path.exists(current_path ):
#    os.makedirs(current_path)


# # check centering 
# zwfs.dm.send_data(dm_flat)
# phasemask_centering_tool.move_relative_and_get_image(zwfs, phasemask, savefigName=fig_path + 'delme.png')

# # get new reference 
# I0_now, N0_now = util.get_reference_images(zwfs, phasemask, theta_degrees=11.8, number_of_frames=256, \
# compass = True, compass_origin=None, savefig= None )

# zwfs.dm.send_data(dm_flat)

# # ------ check centering on DM 

# zwfs.dm.send_data( dm_flat + 0.1*zwfs.dm_shapes['four_torres'])
# #phasemask_centering_tool.move_relative_and_get_image(zwfs, phasemask, savefigName=fig_path + 'delme.png')
# time.sleep(0.1)
# # check reference 
# now_tmp = datetime.datetime.now().strftime("%d-%m-%YT%H.%M.%S")
# I0_now, N0_now = util.get_reference_images(zwfs, phasemask, theta_degrees=11.8, number_of_frames=256, \
# compass = True, compass_origin=None, savefig= fig_path + exper_path + f'DM_centering_four_torres_{now_tmp}.png' )

# # --- DM centering
# zwfs.dm.send_data( dm_flat + 0.1*zwfs.dm_shapes['Crosshair140'])
# #phasemask_centering_tool.move_relative_and_get_image(zwfs, phasemask, savefigName=fig_path + 'delme.png')
# time.sleep(0.1)
# # check reference 
# now_tmp = datetime.datetime.now().strftime("%d-%m-%YT%H.%M.%S")
# I0_now, N0_now = util.get_reference_images(zwfs, phasemask, theta_degrees=11.8, number_of_frames=256, \
# compass = True, compass_origin=None, savefig= fig_path + exper_path + f'DM_centering_crosshairs_{now_tmp}.png' )

# zwfs.dm.send_data( dm_flat ) 


# TT_vectors = util.get_tip_tilt_vectors()

# TT_space = M2C_0 @ TT_vectors
    
# U_TT, S_TT, Vt_TT = np.linalg.svd( TT_space, full_matrices=False)

# I2M_TT = U_TT.T @ R 

# M2C_TT = poke_amp * M2C_0.T @ U_TT # since pinned need M2C to go back to 140 dimension vector  

# R_HO = (np.eye(U_TT.shape[0])  - U_TT @ U_TT.T) @ R

# # go to Eigenmodes for modal control in higher order reconstructor
# U_HO, S_HO, Vt_HO = np.linalg.svd( R_HO, full_matrices=False)
# I2M_HO = Vt_HO  
# M2C_HO = poke_amp *  M2C_0.T @ (U_HO * S_HO) # since pinned need M2C to go back to 140 dimension vector






# # init our controllers 

# rho = 0 * np.ones( I2M_HO.shape[0] )
# kp_leak = 0 * np.ones( I2M_HO.shape[0] )
# lower_limit_leak = -100 * np.ones( I2M_HO.shape[0] )
# upper_limit_leak = 100 * np.ones( I2M_HO.shape[0] )

# leak = LeakyIntegrator(rho=rho, kp=kp_leak, lower_limit=lower_limit_leak, upper_limit=upper_limit_leak )

# kp = 0. * np.ones( I2M_TT.shape[0] )
# ki = 0. * np.ones( I2M_TT.shape[0] )
# kd = 0. * np.ones( I2M_TT.shape[0] )
# setpoint = np.zeros( I2M_TT.shape[0] )
# lower_limit_pid = -100 * np.ones( I2M_TT.shape[0] )
# upper_limit_pid = 100 * np.ones( I2M_TT.shape[0] )

# pid = PIDController(kp, ki, kd, upper_limit_pid, lower_limit_pid, setpoint)

# s_list = []
# e_TT_list = []
# u_TT_list = []
# c_TT_list = []
# e_HO_list = []
# u_HO_list = []
# c_HO_list = []
# atm_disturb_list = []
# dm_disturb_list = []
# rmse_list = []
# flux_outside_pupil_list = []
# residual_list = []
# close_after = 10 

# pid.reset() 
# leak.reset()

# #disturb_basis = util.construct_command_basis( 'fourier_pinned_edges')
# disturbance_cmd = 0.3 * TT_vectors[:,0]
# zwfs.dm.send_data(dm_flat+ dm_pupil_filt * disturbance_cmd) # only apply in registered pupil 
# time.sleep(0.1)

# for it in range(40):
    
#     if it > close_after : # close after 
#         pid.kp = 1 * np.ones( I2M_TT.shape[0] )
#         pid.ki = 0.3 * np.ones( I2M_TT.shape[0] )
        
#         #leak.rho[2:5] = 0.2 #* np.ones( I2M_HO.shape[0] )
#         #leak.kp[2:5] = 0.5

#         im_list =  [  sig, util.get_DM_command_in_2D( cmd2opd * disturbance_cmd),  util.get_DM_command_in_2D( cmd2opd * c_TT),\
#                       util.get_DM_command_in_2D( cmd2opd * c_HO),  util.get_DM_command_in_2D( cmd2opd * (disturbance_cmd - c_HO - c_TT) )] 
#         xlabel_list = [ "" for _ in im_list]
#         ylabel_list = [ "" for _ in im_list]
#         title_list = [ "ZWFS signal", "aberration" , "reco. TT", "reco. HO", "residuals"]
#         cbar_label_list =  [ "ADU", "OPD [nm]",  "OPD [nm]" ,"OPD [nm]", "OPD [nm]"]
#         vlims=[[np.min(sig),np.max(sig)]] + [[np.min(cmd2opd * disturbance_cmd), np.max(cmd2opd * disturbance_cmd)] for _ in im_list[1:]]
#         savefig = current_path + 'delme.png' 
#         util.nice_heatmap_subplots( im_list , xlabel_list, ylabel_list, title_list,cbar_label_list, \
#                             fontsize=15, cbar_orientation = 'bottom', vlims=vlims, axis_off=True, savefig=savefig)
#         _ = input('next?') 

#     i = np.mean( zwfs.get_some_frames(number_of_frames=20, apply_manual_reduction=True),axis=0) #z.detection_chain( test_field, FPM_on=True, include_shotnoise=True, ph_per_s_per_m2_per_nm=True, grids_aligned=True, replace_nan_with=0 )
#     #o = #z.detection_chain( test_field, FPM_on=False, include_shotnoise=True, ph_per_s_per_m2_per_nm=True, grids_aligned=True, replace_nan_with=0 )
    
#     sig = i / np.mean( i ) -  I0 / np.mean( I0 ) # I0_theory/ np.mean(I0_theory) #

#     # update distrubance after measurement 
#     #for _ in range(rows_to_jump):
#     #    scrn.add_row()
#     #disturbance_cmd = np.array( util.create_phase_screen_cmd_for_DM(scrn,  scaling_factor=scrn_scaling_factor   , drop_indicies = corner_indicies, plot_cmd=False) )


#     e_TT = I2M_TT @ sig.reshape(-1)[pupil_pixels]
    
#     u_TT = pid.process( e_TT )
    
#     c_TT = M2C_TT @ u_TT 
    
#     e_HO = I2M_HO @ sig.reshape(-1)[pupil_pixels]

#     u_HO = leak.process( e_HO )
    
#     c_HO = M2C_HO @ u_HO 

#     #c = R @ sig

#     zwfs.dm.send_data( dm_flat+ disturbance_cmd - c_HO - c_TT ) # same way to rtc PID 
#     time.sleep(0.05)  
#     # only measure residual in the registered pupil on DM 
#     residual =  (disturbance_cmd - c_HO - c_TT)[dm_pupil_filt]
#     rmse = np.nanstd( residual )
    
#     # telemetry 
#     s_list.append( sig )
#     e_TT_list.append( e_TT )
#     u_TT_list.append( u_TT )
#     c_TT_list.append( c_TT )
    
#     e_HO_list.append( e_HO )
#     u_HO_list.append( u_HO )
#     c_HO_list.append( c_HO )
    
#     atm_disturb_list.append( scrn.scrn )
#     dm_disturb_list.append( disturbance_cmd )
    
#     residual_list.append( residual )
#     rmse_list.append( rmse )
#     flux_outside_pupil_list.append( np.sum( sig.reshape(-1)[pupil_outer_perim_filter] ) )
#     print( it, f'rmse = {rmse}, flux outside = {flux_outside_pupil_list[-1]}' )



# # write telemetry to file 

# # Dictionary of lists and their names
# lists_dict = {
#     "s_list": s_list,
#     "e_TT_list": e_TT_list,
#     "u_TT_list": u_TT_list,
#     "c_TT_list": c_TT_list,
#     "e_HO_list": e_HO_list,
#     "u_HO_list": u_HO_list,
#     "c_HO_list": c_HO_list,
#     "pid_kp_list": pid.kp,
#     "pid_ki_list": pid.ki,
#     "pid_kd_list": pid.kd,
#     "leay_kp_list": leak.kp,
#     "leay_rho_list": leak.rho,
#     "atm_disturb_list": atm_disturb_list,
#     "dm_disturb_list": dm_disturb_list,
#     "rmse_list": rmse_list,
#     "residual_list": residual_list,
#     "flux_outside_pupil_list":flux_outside_pupil_list,
#     "IM":IM,
#     "R" : R,
#     "I2M_TT" : I2M_TT,
#     "I2M_HO" : I2M_HO,
#     "M2C_TT" : M2C_TT,
#     "M2C_HO" : M2C_HO
# }

# # Create a list of HDUs (Header Data Units)
# hdul = fits.HDUList()

# # Add each list to the HDU list as a new extension
# for list_name, data_list in lists_dict.items():
#     # Convert list to numpy array for FITS compatibility
#     data_array = np.array(data_list, dtype=float)  # Ensure it is a float array or any appropriate type

#     # Create a new ImageHDU with the data
#     hdu = fits.ImageHDU(data_array)

#     # Set the EXTNAME header to the variable name
#     hdu.header['EXTNAME'] = list_name

#     # Append the HDU to the HDU list
#     hdul.append(hdu)

# # Write the HDU list to a FITS file
# hdul.writeto(current_path + f'closed_loop_TTstatic_telemetry_{itera}.fits', overwrite=True)




    
# # ---- CLOSE THE LOOP?  - sdynamic

# current_path = fig_path + exper_path +  "5_closed_loop_dynamic_kolmogorov/" 
# if not os.path.exists(current_path ):
#    os.makedirs(current_path)


# # check centering 
# zwfs.dm.send_data(dm_flat)
# phasemask_centering_tool.move_relative_and_get_image(zwfs, phasemask, savefigName=fig_path + 'delme.png')

# # get new reference 
# I0_now, N0_now = util.get_reference_images(zwfs, phasemask, theta_degrees=11.8, number_of_frames=256, \
# compass = True, compass_origin=None, savefig= None )

# zwfs.dm.send_data(dm_flat)


# TT_vectors = util.get_tip_tilt_vectors()

# TT_space = M2C_0 @ TT_vectors
    
# U_TT, S_TT, Vt_TT = np.linalg.svd( TT_space, full_matrices=False)

# I2M_TT = U_TT.T @ R 

# M2C_TT = poke_amp * M2C_0.T @ U_TT # since pinned need M2C to go back to 140 dimension vector  

# R_HO = (np.eye(U_TT.shape[0])  - U_TT @ U_TT.T) @ R

# # go to Eigenmodes for modal control in higher order reconstructor
# U_HO, S_HO, Vt_HO = np.linalg.svd( R_HO, full_matrices=False)
# I2M_HO = Vt_HO  
# M2C_HO = poke_amp *  M2C_0.T @ (U_HO * S_HO) # since pinned need M2C to go back to 140 dimension vector


# # pupil outside to sample reference field / strehl 

# central_lambda = util.find_central_wavelength(lambda_cut_on=900e-9, lambda_cut_off=1180e-9, T=1900)
# print(f"The central wavelength is {central_lambda * 1e9:.2f} nm")


# wvl = 1e6 * central_lambda # 0.900 #1.040 # um  
# phase_shift = util.get_phasemask_phaseshift( wvl= wvl, depth = phasemask.phasemask_parameters[phasemask_name]['depth'] )
# mask_diam = 1e-6 * phasemask.phasemask_parameters[phasemask_name]['diameter']
# N0_theory0, I0_theory0 = util.get_theoretical_reference_pupils( wavelength = wvl*1e-6 ,F_number = 21.2, mask_diam = mask_diam,\
#                                         diameter_in_angular_units = False,  phaseshift = phase_shift , padding_factor = 4, \
#                                         debug= False, analytic_solution = True )

# M = I0_theory0.shape[0]
# N = I0_theory0.shape[1]

# m = zwfs.I0.shape[1]
# n = zwfs.I0.shape[0]

# # A = pi * r^2 => r = sqrt( A / pi)
# new_radius = (zwfs.pupil_pixel_filter.sum()/np.pi)**0.5
# x_c = np.mean( np.unravel_index( np.where( zwfs.pupil_pixel_filter ), zwfs.I0.shape)[1])
# y_c = np.mean( np.unravel_index( np.where( zwfs.pupil_pixel_filter ), zwfs.I0.shape)[0])

# I0_theory = util.interpolate_pupil_to_measurement( N0_theory0, I0_theory0, M, N, m, n, x_c, y_c, new_radius)

# N0_theory = util.interpolate_pupil_to_measurement( N0_theory0, N0_theory0, M, N, m, n, x_c, y_c, new_radius)

# pupil_outer_perim_filter = (~zwfs.bad_pixel_filter * (abs( I0_theory - N0_theory ) > 0.1 ).reshape(-1) * (~zwfs.pupil_pixel_filter) )

# plt.figure()
# plt.imshow( pupil_outer_perim_filter.reshape(zwfs.I0.shape) ) 
# plt.savefig( current_path + 'outer_pupil_filter.png')




# # init our controllers 

# cl_try = 4 # which closed loop try are we up to?

# rho = 0 * np.ones( I2M_HO.shape[0] )
# kp_leak = 0 * np.ones( I2M_HO.shape[0] )
# lower_limit_leak = -100 * np.ones( I2M_HO.shape[0] )
# upper_limit_leak = 100 * np.ones( I2M_HO.shape[0] )

# leak = LeakyIntegrator(rho=rho, kp=kp_leak, lower_limit=lower_limit_leak, upper_limit=upper_limit_leak )

# kp = 0. * np.ones( I2M_TT.shape[0] )
# ki = 0. * np.ones( I2M_TT.shape[0] )
# kd = 0. * np.ones( I2M_TT.shape[0] )
# setpoint = np.zeros( I2M_TT.shape[0] )
# lower_limit_pid = -100 * np.ones( I2M_TT.shape[0] )
# upper_limit_pid = 100 * np.ones( I2M_TT.shape[0] )

# pid = PIDController(kp, ki, kd, upper_limit_pid, lower_limit_pid, setpoint)

# s_list = []
# e_TT_list = []
# u_TT_list = []
# c_TT_list = []
# e_HO_list = []
# u_HO_list = []
# c_HO_list = []
# atm_disturb_list = []
# dm_disturb_list = []
# rmse_list = []
# flux_outside_pupil_list = []
# residual_list = []
# close_after = 10

# pid.reset() 
# leak.reset()

# #disturb_basis = util.construct_command_basis( 'fourier_pinned_edges')
# #disturbance_cmd = 0.3 * TT_vectors[:,0]

# # init a phasescreen to roll across DM 

# Nx_act = 12 # actuators across DM 

# D = 1.8 #m effective diameter of the telescope

# screen_pixels = Nx_act*2**3  #pixels inthe inital screen before projection onto DM

# corner_indicies = [0, Nx_act-1, Nx_act * (Nx_act-1), -1] 

# scrn_scaling_factor =  0.15 

# rows_to_jump = 2 # how many rows to jump on initial phase screen for each Baldr loop

# distance_per_correction = rows_to_jump * D/screen_pixels # effective distance travelled by turbulence per AO iteration 
# print(f'{rows_to_jump} rows jumped per AO command in initial phase screen of {screen_pixels} pixels. for {D}m mirror this corresponds to a distance_per_correction = {distance_per_correction}m')

# scrn = aotools.infinitephasescreen.PhaseScreenVonKarman(nx_size=screen_pixels, pixel_scale=D/screen_pixels,r0=0.1,L0=12)

# disturbance_cmd = util.create_phase_screen_cmd_for_DM(scrn,  scaling_factor=scrn_scaling_factor   , drop_indicies = corner_indicies, plot_cmd=False)

# plt.figure()
# plt.imshow( util.get_DM_command_in_2D(disturbance_cmd, Nx_act=12) )
# plt.colorbar()
# plt.title( 'initial Kolmogorov aberration to apply to DM')
# plt.savefig( current_path  + 'initial_phasescreen.png')


# zwfs.dm.send_data(dm_flat+ dm_pupil_filt * disturbance_cmd) # only apply in registered pupil 
# time.sleep(0.1)

# for it in range(50):
    
#     if it > close_after : # close after 
#         pid.kp = 0.9 * np.ones( I2M_TT.shape[0] )
#         pid.ki = 0.1 * np.ones( I2M_TT.shape[0] )
        
#         leak.rho[:30] = 0.1 #* np.ones( I2M_HO.shape[0] )
#         leak.kp[:30] = 0.5

#         im_list =  [  sig, util.get_DM_command_in_2D( cmd2opd * disturbance_cmd),  util.get_DM_command_in_2D( cmd2opd * c_TT),\
#                       util.get_DM_command_in_2D( cmd2opd * c_HO),  util.get_DM_command_in_2D( cmd2opd * (disturbance_cmd - c_HO - c_TT) )] 
#         xlabel_list = [ "" for _ in im_list]
#         ylabel_list = [ "" for _ in im_list]
#         title_list = [ "ZWFS signal", "aberration" , "reco. TT", "reco. HO", "residuals"]
#         cbar_label_list =  [ "ADU", "OPD [nm]",  "OPD [nm]" ,"OPD [nm]", "OPD [nm]"]
#         vlims=[[np.min(sig),np.max(sig)]] + [[np.min(cmd2opd * disturbance_cmd), np.max(cmd2opd * disturbance_cmd)] for _ in im_list[1:]]
#         savefig = current_path + 'delme.png' 
#         util.nice_heatmap_subplots( im_list , xlabel_list, ylabel_list, title_list,cbar_label_list, \
#                             fontsize=15, cbar_orientation = 'bottom', vlims=vlims, axis_off=True, savefig=savefig)
#         _ = input('next?') 

#     i = np.mean( zwfs.get_some_frames(number_of_frames=20, apply_manual_reduction=True),axis=0) #z.detection_chain( test_field, FPM_on=True, include_shotnoise=True, ph_per_s_per_m2_per_nm=True, grids_aligned=True, replace_nan_with=0 )
#     #o = #z.detection_chain( test_field, FPM_on=False, include_shotnoise=True, ph_per_s_per_m2_per_nm=True, grids_aligned=True, replace_nan_with=0 )
    
#     sig = i / np.mean( i ) -  I0 / np.mean( I0 ) # I0_theory/ np.mean(I0_theory) #

#     # update distrubance after measurement 
#     for _ in range(rows_to_jump):
#         scrn.add_row()
#     disturbance_cmd = np.array( util.create_phase_screen_cmd_for_DM(scrn,  scaling_factor=scrn_scaling_factor   , drop_indicies = corner_indicies, plot_cmd=False) )


#     e_TT = I2M_TT @ sig.reshape(-1)[pupil_pixels]
    
#     u_TT = pid.process( e_TT )
    
#     c_TT = M2C_TT @ u_TT 
    
#     e_HO = I2M_HO @ sig.reshape(-1)[pupil_pixels]

#     u_HO = leak.process( e_HO )
    
#     c_HO = M2C_HO @ u_HO 

#     #c = R @ sig

#     zwfs.dm.send_data( dm_flat+ disturbance_cmd - c_HO - c_TT ) # same way to rtc PID 
#     time.sleep(0.05)  
#     # only measure residual in the registered pupil on DM 
#     residual =  (disturbance_cmd - c_HO - c_TT)[dm_pupil_filt]
#     rmse = np.nanstd( residual )
    
#     # telemetry 
#     s_list.append( sig )
#     e_TT_list.append( e_TT )
#     u_TT_list.append( u_TT )
#     c_TT_list.append( c_TT )
    
#     e_HO_list.append( e_HO )
#     u_HO_list.append( u_HO )
#     c_HO_list.append( c_HO )
    
#     atm_disturb_list.append( scrn.scrn )
#     dm_disturb_list.append( disturbance_cmd )
    
#     residual_list.append( residual )
#     rmse_list.append( rmse )
#     flux_outside_pupil_list.append( np.sum( sig.reshape(-1)[pupil_outer_perim_filter] ) )
#     print( it, f'rmse = {rmse}, flux outside = {flux_outside_pupil_list[-1]}' )



# # write telemetry to file 

# # Dictionary of lists and their names
# lists_dict = {
#     "s_list": s_list,
#     "e_TT_list": e_TT_list,
#     "u_TT_list": u_TT_list,
#     "c_TT_list": c_TT_list,
#     "e_HO_list": e_HO_list,
#     "u_HO_list": u_HO_list,
#     "c_HO_list": c_HO_list,
#     "pid_kp_list": pid.kp,
#     "pid_ki_list": pid.ki,
#     "pid_kd_list": pid.kd,
#     "leay_kp_list": leak.kp,
#     "leay_rho_list": leak.rho,
#     "atm_disturb_list": atm_disturb_list,
#     "dm_disturb_list": dm_disturb_list,
#     "rmse_list": rmse_list,
#     "residual_list": residual_list,
#     "flux_outside_pupil_list":flux_outside_pupil_list,
#     "IM":IM,
#     "R" : R,
#     "I2M_TT" : I2M_TT,
#     "I2M_HO" : I2M_HO,
#     "M2C_TT" : M2C_TT,
#     "M2C_HO" : M2C_HO
# }

# # Create a list of HDUs (Header Data Units)
# hdul = fits.HDUList()

# # Add each list to the HDU list as a new extension
# for list_name, data_list in lists_dict.items():
#     # Convert list to numpy array for FITS compatibility
#     data_array = np.array(data_list, dtype=float)  # Ensure it is a float array or any appropriate type

#     # Create a new ImageHDU with the data
#     hdu = fits.ImageHDU(data_array)

#     # Set the EXTNAME header to the variable name
#     hdu.header['EXTNAME'] = list_name

#     # Append the HDU to the HDU list
#     hdul.append(hdu)

# # Write the HDU list to a FITS file
# hdul.writeto(current_path + f'closed_loop_dynamic_kol{scrn_scaling_factor}_telemetry_{cl_try}.fits', overwrite=True)


# plt.figure(); plt.plot( cmd2opd * np.array(rmse_list) ) ; plt.figure( current_path +f'rmse_try{cl_try}')




































