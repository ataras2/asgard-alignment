import zmq
from astropy.io import fits
import argparse
import time
import os
import datetime
import numpy as np
from asgard_alignment import FLI_Cameras as FLI # depends on xao shm


########################################################################
# standardized format to generate RAW internal flats
# FOR NOW we just run this nievely relying on the user to execute it 
# in the desired conditions (script does not know if / what phasemask 
# it is on). Later we probably want to have a specified region 
# e.g. FLAT FPM STATE : IN/OUT 
#       if IN :
#            MOVE TO FPM MASK 
#       elif OUT :
#           MOVE TO FPM MASK :
#           MOVE 200um +X off mask
#       idea is to consistently sample to same region of the phasemask 
#       for clear / zwfs pupils
########################################################################


tstamp = datetime.datetime.now().strftime("%Y-%m-%dT%H-%M-%S")
tstamp_rough = datetime.datetime.now().strftime("%d-%m-%Y")

parser = argparse.ArgumentParser(description="generate darks for different gain settings on CRED ONE")

parser.add_argument(
    '--data_path',
    type=str,
    default=f"/home/asg/Progs/repos/asgard-alignment/calibration/cal_data/calibration_frames/{tstamp_rough}/",
    help="Path to the directory for storing dark data. Default: %(default)s"
)

parser.add_argument(
    '--gains',
    type=int,
    nargs='+',
    default=[1, 5],
    help="List of gains to apply when collecting raw flat frames. Default: %(default)s"
)

parser.add_argument(
    '--fps',
    type=int,
    nargs='+',
    default=[200,500,1000,1737],
    help="List of frame rates for raw flats. Default: %(default)s"
)


parser.add_argument(
    '--phasemask_in',
    type=int,
    default=0,
    help="focal plane phasemask (FPM) in (1) or out (0). Default: %(default)s"
)



parser.add_argument(
    '--phasemask',
    type=str,
    default="H3",
    help="which phasemask to use (TO DO), if FPM_STATE = 0, then we move 200um off this phasemask. Default: %(default)s"
)

parser.add_argument("--number_of_frames", 
                    type=int, 
                    default=1000, 
                    help="number of frames to take for flats")


parser.add_argument("--host", type=str, default="localhost", help="Server host")
parser.add_argument("--port", type=int, default=5555, help="Server port")
parser.add_argument(
    "--timeout", type=int, default=2000, help="Response timeout in milliseconds"
)



args = parser.parse_args()

context = zmq.Context()
context.socket(zmq.REQ)
mds_socket = context.socket(zmq.REQ)
mds_socket.setsockopt(zmq.RCVTIMEO, args.timeout)
mds_socket.connect( f"tcp://{args.host}:{args.port}")

# to store master darks 
if not os.path.exists( args.data_path ):
    os.makedirs( args.data_path )

# to store raw darks 

if not args.fpm_in:
    raw_flat_rel_pth = "RAW_CLEAR_FLATS/" #+ f"{tstamp_rough}/"

    # TO DO : MOVE TO PHASEMASK THEN APPLY 200um offset 

else args.fpm_in:
    raw_flat_rel_pth = "RAW_ZWFS_FLATS/" #+ f"{tstamp_rough}/"
    # TO DO : MOVE TO PHASEMASK THEN 

if not os.path.exists( args.data_path + raw_flat_rel_pth  ):
    os.makedirs( args.data_path + raw_flat_rel_pth  )

#cc =  shm("/dev/shm/cred1.im.shm")  # testing 
c = FLI.fli()

aduoffset = 1000 # hard coded for consistency 

# default aduoffset to avoid overflow of int on CRED 
c.send_fli_cmd(f"set aduoffset {aduoffset}")

# camera config
config_tmp = c.get_camera_config()

# integration time as i understand for the CRED 1. 
dt = round( 1/float(config_tmp["fps"]) , 5 )

# try turn off source 
message = "on SBB"
mds_socket.send_string(message)
response = mds_socket.recv_string()#.decode("ascii")
print( response )


print(f'turning on internal source and waiting 3s')
time.sleep(3) # wait a bit to settle


flat_hdulist = fits.HDUList([] ) 

print('...getting frames')
#maxfps = 1739 # Hz # c.send_fli_cmd("maxfps")
for gain in args.gain:

    print(f"   gain = {gain}")

    c.send_fli_cmd(f"set gain {gain}")

    # edit our cponfig dict without bothering to comunicate with camera
    config_tmp["gain"] = gain 

    for fps in args.fps:

    c.send_fli_cmd( f"set fps {fps}")

    time.sleep(2)

    flat_list = c.get_some_frames(number_of_frames = args.number_of_frames, apply_manual_reduction=False, timeout_limit = 20000 )

    master_flat = np.mean(flat_list, axis=0) # [adu]
    
    time.sleep(2)

    # ----------writing raw dakrs
    hdu = fits.PrimaryHDU(flat_list)
    if args.phasemask_in:
        hdu.header['EXTNAME'] = 'ZWFS_FLAT_FRAMES'
    else:
        hdu.header['EXTNAME'] = 'CLEAR_FLAT_FRAMES'
    hdu.header['PHASEMASK'] = args.phasemask
    hdu.header['UNITS'] = "ADU"
    hdu.header['DATE'] = tstamp
    hdu.header['SYSTEM'] = "BALDR-HEIM_CRED1"
    for k, v in config_tmp.items():
        hdu.header[k] = v

    hdulist = fits.HDUList([hdu]) #, badpix_hdu, conv_gain_hdu])

    fname_raw = args.data_path + raw_flat_rel_pth + f"raw_{hdu.header['EXTNAME']}_phaasemask-{args.phasemask}_fps-{config_tmp['fps']}_gain-{config_tmp['gain']}_{tstamp}.fits"
    hdulist.writeto(fname_raw, overwrite=True)
    print( f"   wrote raw flats for gain {gain}:\n    {fname_raw}" )

    # ----------Estimate pixelwise conversion gain (e-/ADU)
    # Assumes the sinal is dominated by shot noise
	# The system is linear
	# The bias (offset) has been properly subtracted
    # THIS NEEDS TO BE DONE ON INTERNAL FLAT 
    # dark_minus_bias = np.array( dark_list ) - np.mean(bias_list, axis=0)
    # mean_dark = np.mean( dark_minus_bias, axis=0)
    # var_dark = np.var( dark_minus_bias, axis=0) 
    # with np.errstate(divide='ignore', invalid='ignore'):
    #     conversion_gain = np.where(var_dark > 0, mean_dark / var_dark, 0)

    # conv_gain_hdu = fits.ImageHDU(conversion_gain)
    # conv_gain_hdu.header['EXTNAME'] = f'CONV_GAIN_GAIN-{gain}'
    # conv_gain_hdu.header['UNITS'] = 'e-/ADU'
    # for k, v in config_tmp.items():
    #     conv_gain_hdu.header[k] = v    
    # conv_gain_hdu.header['DATE'] = tstamp

    # ---------- Calculate bad_pixel_map 
    _, bad_pixel_mask = FLI.get_bad_pixels( dark_list, std_threshold = args.std_threshold, mean_threshold = args.mean_threshold)
    badpix_hdu = fits.PrimaryHDU( np.array( bad_pixel_mask).astype(int) )
    badpix_hdu.header['EXTNAME'] = f'BAD_PIXEL_MAP_GAIN-{gain}'  
    for k, v in config_tmp.items():
        badpix_hdu.header[k] = v
    badpix_hdu.header['STD_THR'] = args.std_threshold
    badpix_hdu.header['MEAN_THR'] = args.mean_threshold
    badpix_hdu.header['DATE'] = tstamp



    # ----------writing raw bais
    bias_hdu = fits.PrimaryHDU(bias_list)
    bias_hdu.header['EXTNAME'] = f'BIAS_GAIN-{gain}'  
    bias_hdu.header['UNITS'] = "ADU"
    bias_hdu.header['DATE'] = tstamp
    for k, v in config_tmp.items():
        bias_hdu.header[k] = v

    hdulist = fits.HDUList([bias_hdu])

    fname_raw = args.data_path + raw_bias_rel_pth + f"raw_bias_maxfps-{maxfps}_gain-{config_tmp['gain']}_{tstamp}.fits"
    hdulist.writeto(fname_raw, overwrite=True)
    print( f"   wrote raw bias for gain {gain}:\n    {fname_raw}" )

    # ----------writing MASTER DARK
    master_dark_hdu = fits.PrimaryHDU(master_dark)
    master_dark_hdu.header['EXTNAME'] = f"DARK_GAIN-{gain}"
    master_dark_hdu.header['UNITS'] = "ADU/s"
    master_dark_hdu.header['DATE'] = tstamp
    master_dark_hdu.header['SYSTEM'] = "BALDR-HEIM_CRED1"
    for k, v in config_tmp.items():
        master_dark_hdu.header[k] = v

    print("calculated master dark and appending the fits")
    master_dark_hdulist.append( master_dark_hdu )

    # ----------writing MASTER BIAS
    master_bias_hdu = fits.PrimaryHDU(master_bias)
    master_bias_hdu.header['EXTNAME'] = f"BIAS_GAIN-{gain}"
    master_bias_hdu.header['UNITS'] = "ADU/s"
    master_bias_hdu.header['DATE'] = tstamp
    master_bias_hdu.header['SYSTEM'] = "BALDR-HEIM_CRED1"
    for k, v in config_tmp.items():
        master_bias_hdu.header[k] = v

    print("calculated master bias and appending the fits")
    master_bias_hdulist.append( master_bias_hdu )

    # ---------- MASTER BAD PIXEL MAP
    print("calculated master bad pixel map and appending the fits")
    master_badpixel_hdulist.append( badpix_hdu )

    # ---------- CONVERSION GAIN MAP
    #print("calculated master conversion gain map and appending the fits")
    #master_convgain_hdulist.append( conv_gain_hdu )

fname_dark_master = args.data_path + f"master_darks_adu_p_sec_fps-{config_tmp['fps']}_{tstamp}.fits"
master_dark_hdulist.writeto(fname_dark_master, overwrite=True)
print( f"   wrote master darks:\n    {fname_dark_master}" )

fname_bias_master = args.data_path + f"master_bias_adu_p_sec_fps-{config_tmp['fps']}_{tstamp}.fits"
master_bias_hdulist.writeto(fname_bias_master, overwrite=True)
print( f"   wrote master bias:\n    {fname_bias_master}" )


fname_badpix_master = args.data_path + f"master_bad_pixel_map_fps-{config_tmp['fps']}_{tstamp}.fits"
master_badpixel_hdulist.writeto(fname_badpix_master, overwrite=True)
print( f"   wrote bad pixel mask:\n    {fname_badpix_master}" )

# This gets done on internal flat field 
# fname_convgain_master = args.data_path + f"conversion_gain_map_fps-{config_tmp['fps']}_{tstamp}.fits"
# master_convgain_hdulist.writeto(fname_convgain_master, overwrite=True)
# print( f"   wrote conversion gain:\n    {fname_convgain_master}" )



# ---------- TEST SECTION ----------

print("\n--- Running validation test at gain = mid range ---")

gain = np.max([1,args.max_gain // 2])
# Set gain to 
c.send_fli_cmd(f"set gain {gain}")

# Get camera config and fps
fps = float(config_tmp["fps"])
dt = 1.0 / fps

# Acquire test dark
test_dark = np.mean(c.get_some_frames(number_of_frames=100, apply_manual_reduction=False), axis=0)  # [ADU]

# Load corresponding master bias and master dark
master_bias_test = master_bias_hdulist[f"BIAS_GAIN-{gain}"].data   # gain = 5 → index = 4 (0-based)
master_dark_test = master_dark_hdulist[f"DARK_GAIN-{gain}"].data   # [ADU/s]
bad_pixel_mask = master_badpixel_hdulist[f"BAD_PIXEL_MAP_GAIN-{gain}"].data
# Subtract bias
test_dark_minus_bias = test_dark - master_bias_test  # [ADU]

# Convert master dark to [ADU] by multiplying with dt
expected_dark = master_dark_test * dt  # [ADU]

# Compute pixelwise error and statistics
error_map = test_dark_minus_bias - expected_dark # ADU]
mean_error = np.mean(error_map[~bad_pixel_mask ])
std_error = np.std( error_map[~bad_pixel_mask ])

plt.figure(); plt.imshow( ~bad_pixel_mask * error_map) ;plt.colorbar() ;plt.savefig('delme.png')

print(f"Validation test for gain  = {gain}:")
print(f" - FPS = {fps} -> dt = {dt:.5f} s")
print(f" - Mean pixel error [ADU] = {mean_error:.3f}")
print(f" - Std  pixel error [ADU] = {std_error:.3f}")

# try turn source back on 
#my_controllino.turn_on("SBB")
message = "on SBB"
mds_socket.send_string(message)
response = mds_socket.recv_string()#.decode("ascii")
print( response )
time.sleep(2)

# close camera SHM and set gain = 1 
c.close(erase_file=False)

print(f"DONE.")

