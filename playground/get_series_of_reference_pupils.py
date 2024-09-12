

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


data_path = f'tmp/{tstamp.split("T")[0]}/SNR_analysis' 


if not os.path.exists(data_path):
   os.makedirs(data_path)




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


source_selection.set_source(  'none' )
time.sleep(0.2)

## ------- Calibrate detector (dark, badpixels)
# Source should be out
# at sydney move 01 X-LSM150A-SE03 to 133.07mm
zwfs.build_manual_dark()

# get our bad pixels 
bad_pixels = zwfs.get_bad_pixel_indicies( no_frames = 200, std_threshold = 30 , flatten=False) # std_threshold = 50

# update zwfs bad pixel mask and flattened pixel values 
zwfs.build_bad_pixel_mask( bad_pixels , set_bad_pixels_to = 0 )

## ------- move source back in 
source_selection.set_source( source_name )
time.sleep(2)


# check centering 
zwfs.dm.send_data(zwfs.dm_shapes['flat_dm'] )
phasemask_centering_tool.move_relative_and_get_image(zwfs, phasemask, savefigName= data_path + 'delme.png')



tgrid = 1e-3 * np.array( [0.3,0.5,0.7,1,2, 3,5,7,9] )[::-1]
N0_dict = {} 
I0_dict = {} 
for tint in tgrid:
    
    print( tint )
    
    zwfs.set_camera_dit( tint )
    time.sleep(0.2)
    
    source_selection.set_source(  'none' )
    time.sleep(0.2)
    
    ## ------- Calibrate detector (dark, badpixels)
    # Source should be out
    # at sydney move 01 X-LSM150A-SE03 to 133.07mm
    zwfs.build_manual_dark()
    
    # get our bad pixels 
    bad_pixels = zwfs.get_bad_pixel_indicies( no_frames = 200, std_threshold = 2e5 , flatten=False) # std_threshold = 50
    
    # update zwfs bad pixel mask and flattened pixel values 
    zwfs.build_bad_pixel_mask( bad_pixels , set_bad_pixels_to = 0 )
    
    ## ------- move source back in 
    source_selection.set_source( source_name )
    time.sleep(2)
    
    # get a series of on mask images 
    I0_dict["I0_tint"+str(tint)] = zwfs.get_some_frames(number_of_frames=1000, apply_manual_reduction=True)
    
    # get a series of off mask images 
    phasemask.move_relative( [ 200, 0 ] )
    
    N0_dict["N0_tint"+str(tint)] = zwfs.get_some_frames(number_of_frames=1000, apply_manual_reduction=True)
    
    phasemask.move_relative( [ -200, 0 ] )

    time.sleep( 60 ) # we want time between them to watch stability

    
hdulist = fits.HDUList([])

for list_name, data_list in N0_dict.items():
    
    # Convert list to numpy array for FITS compatibility
    data_array = np.array(data_list, dtype=float)  # Ensure it is a float array or any appropriate type

    # Create a new ImageHDU with the data
    hdu = fits.ImageHDU(data_array)

    # Set the EXTNAME header to the variable name
    hdu.header['EXTNAME'] = list_name

    # Append the HDU to the HDU list
    hdulist.append(hdu)

    
for list_name, data_list in I0_dict.items():
    
    # Convert list to numpy array for FITS compatibility
    data_array = np.array(data_list, dtype=float)  # Ensure it is a float array or any appropriate type

    # Create a new ImageHDU with the data
    hdu = fits.ImageHDU(data_array)

    # Set the EXTNAME header to the variable name
    hdu.header['EXTNAME'] = list_name

    # Append the HDU to the HDU list
    hdulist.append(hdu)


hdulist.writeto(data_path + f'N0_I0_vs_DIT_{tstamp}.fits', overwrite=True)
    
    

# analysis 

a = hdulist 

plt.figure(1)
pup_filt =  a[0].data[0] > np.mean( a[0].data[0] ) + 0.5 * np.std( a[0].data[0] )
pup_filt[:,:25] = False # some bad pixels here 

fig,ax = plt.subplots(1,2 )
ax[0].imshow( a[0].data[0] )
ax[1].imshow( pup_filt )
plt.savefig( data_path + f'pupil_filter_{tstamp}.png')    


import matplotlib.colors as colors

# look at the actual variance on the pupil 

snr_list = [];  t_list = []
for i in range(len(a)//2):
    plt.close()
    tint = round( 1e3 * ( float(a[i].header['EXTNAME'].split('tint')[-1] ) ),2)
    t_list.append( tint )
    plt.figure(4)
    SNR = np.mean(  a[i].data ,axis=0 )  / np.std( a[i].data ,axis=0 )
    im = plt.imshow( SNR ) 
    plt.colorbar(im, label='SNR' ) #, norm=colors.LogNorm() )
    plt.title( f"DIT = {tint}ms")

    snr_list.append( SNR[pup_filt] )
    
    plt.savefig(data_path + f'pupil_N0_SNR_DIT-{tint}ms_{tstamp}.png')


    
plt.figure(i+1); 

plt.xlabel('integration time [ms]',fontsize=15)
plt.ylabel('SNR',fontsize=15)
plt.gca().tick_params(labelsize=15)
plt.semilogx( t_list, np.array(snr_list).mean(axis=1), '.')
plt.savefig( data_path + f'SNR_vs_tint_logscale_{tstamp}.png')    
    

# look at I0 (phasemask in)

snr_list = [];  t_list = []
for i in range(len(a)//2, len(a)):
    plt.close()
    tint = round( 1e3 * ( float(a[i].header['EXTNAME'].split('tint')[-1] ) ),2)
    t_list.append( tint )
    plt.figure(4)
    SNR = np.mean(  a[i].data ,axis=0 )  / np.std( a[i].data ,axis=0 )
    im = plt.imshow( SNR ) 
    plt.colorbar(im, label='SNR' ) #, norm=colors.LogNorm() )
    plt.title( f"DIT = {tint}ms")

    snr_list.append( SNR[pup_filt] )
    
    plt.savefig(data_path + f'pupil_I0_SNR_DIT-{tint}ms_{tstamp}.png')


