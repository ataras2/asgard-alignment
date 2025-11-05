import zmq
from astropy.io import fits
import argparse
import time
import os
import datetime
import numpy as np
import matplotlib.pyplot as plt 
from asgard_alignment import FLI_Cameras as FLI # depends on xao shm

########################################################################
# standardized format to generate bias, dark and bad pixel maps 
# for Baldr/Heim
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
# parser.add_argument(
#     '--max_gain',
#     type=int,
#     default=5,
#     help="maximum gain to be applied in gain grid for the darks. Default: %(default)s"
# )
parser.add_argument(
    '--gains',
    type=int,
    nargs='+',
    default=[1, 5, 10, 20],
    help="List of gains to apply when collecting dark frames. Default: %(default)s"
)
parser.add_argument(
    '--fps',
    type=float,
    default=None,
    help="frame rate for darks. Default: %(default)s"
)
parser.add_argument(
    '--mean_threshold',
    type=float,
    default=4.0,
    help="mean threshold for calculating bad pixels on darks. Default: %(default)s"
)
parser.add_argument(
    '--std_threshold',
    type=float,
    default=10.0,
    help="std threshold for calculating bad pixels on darks. Default: %(default)s"
)
parser.add_argument("--number_of_frames", 
                    type=int, 
                    default=1000, 
                    help="number of frames to take for dark")

parser.add_argument("--host", type=str, default="192.168.100.2", help="Server host")
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

sleeptime = 60 # seconds <--- required to get ride of persistance of pupils 

# to store master darks 
if not os.path.exists( args.data_path ):
    os.makedirs( args.data_path )

# to store raw darks
for red_pth in ["RAW_DARKS/","RAW_BIAS/","MASTER_BIAS/","MASTER_DARK/","BAD_PIXEL_MAP/"] :
    if not os.path.exists( args.data_path + red_pth  ):
        os.makedirs( args.data_path + red_pth  )
        print(f"made directory {args.data_path + red_pth}")

# raw_dark_rel_pth = "RAW_DARKS/" #+ f"{tstamp_rough}/"
# if not os.path.exists( args.data_path + raw_dark_rel_pth  ):
#     os.makedirs( args.data_path + raw_dark_rel_pth  )

# raw_bias_rel_pth = "RAW_BIAS/" #+ f"{tstamp_rough}/"
# if not os.path.exists( args.data_path + raw_bias_rel_pth ):
#     os.makedirs( args.data_path + raw_bias_rel_pth )

# raw_bias_rel_pth = "MASTER_BIAS/" #+ f"{tstamp_rough}/"
# if not os.path.exists( args.data_path + raw_bias_rel_pth ):
#     os.makedirs( args.data_path + raw_bias_rel_pth )

# raw_bias_rel_pth = "MASTER_DARK/" #+ f"{tstamp_rough}/"
# if not os.path.exists( args.data_path + raw_bias_rel_pth ):
#     os.makedirs( args.data_path + raw_bias_rel_pth )

# raw_bias_rel_pth = "BAD_PIXEL_MAP/" #+ f"{tstamp_rough}/"
# if not os.path.exists( args.data_path + raw_bias_rel_pth ):
#     os.makedirs( args.data_path + raw_bias_rel_pth )

#cc =  shm("/dev/shm/cred1.im.shm")  # testing 
c = FLI.fli()

aduoffset = 1000 # hard coded for consistency 

# default aduoffset to avoid overflow of int on CRED 
c.send_fli_cmd(f"set aduoffset {aduoffset}")

if args.fps is not None:
    c.send_fli_cmd(f"set fps {args.fps}")

# camera config
config_tmp = c.get_camera_config()

# integration time as i understand for the CRED 1. 
dt = round( 1/float(config_tmp["fps"]) , 5 )

# try turn off source 
message = "off SBB"
mds_socket.send_string(message)
response = mds_socket.recv_string()#.decode("ascii")
print( response )

print(f'turning off source and waiting {sleeptime}s')
time.sleep(sleeptime) # wait a bit to settle

# I DECIDED AGAINST MULTIPLE EXTENSIONS IN A FITS HDULIST
# -- INSTEAD KEEP FILES LIGHTWEIGHT AND QUERYABLE FROM FILE NAME
# master_bias_hdulist = fits.HDUList([] ) 
# master_dark_hdulist = fits.HDUList([] ) 
# master_badpixel_hdulist = fits.HDUList([] ) 
# master_convgain_hdulist = fits.HDUList([] ) 
print('...getting frames')
maxfps = 1739 # Hz # c.send_fli_cmd("maxfps")
for gain in args.gains: #np.arange(1,args.max_gain):

    print(f"   gain = {gain}")

    c.send_fli_cmd(f"set gain {gain}")

    # edit our cponfig dict without bothering to comunicate with camera
    config_tmp["gain"] = gain 

    # ----------BIAS 
    c.send_fli_cmd( f"set fps {maxfps}")

    time.sleep(2)

    bias_list = c.get_some_frames(number_of_frames = args.number_of_frames, apply_manual_reduction=False, timeout_limit = 20000 )

    master_bias = np.mean(bias_list, axis=0) # [adu]
    
    time.sleep(2)

    # return to previous fps
    c.send_fli_cmd(f"set fps {config_tmp['fps']}")

    time.sleep(2) 

    # ----------DARK
    dark_list = c.get_some_frames(number_of_frames = args.number_of_frames, apply_manual_reduction=False, timeout_limit = 20000 )

    master_dark = ( np.mean(dark_list, axis=0)  -  master_bias) / dt #[adu/s]

    # ----------writing raw dakrs
    dark_hdu = fits.PrimaryHDU(dark_list)
    dark_hdu.header['EXTNAME'] = 'DARK_FRAMES'  
    dark_hdu.header['UNITS'] = "ADU"
    dark_hdu.header['DATE'] = tstamp
    for k, v in config_tmp.items():
        dark_hdu.header[k] = v


    hdulist = fits.HDUList([dark_hdu]) #, badpix_hdu, conv_gain_hdu])

    fname_raw = args.data_path + f"RAW_DARKS/raw_darks_fps-{config_tmp['fps']}_gain-{config_tmp['gain']}_{tstamp}.fits"
    hdulist.writeto(fname_raw, overwrite=True)
    print( f"   wrote raw darks for gain {gain}:\n    {fname_raw}" )

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


    fname_badpix_master = args.data_path +  f"BAD_PIXEL_MAP/master_bad_pixel_map_fps-{config_tmp['fps']}_gain-{config_tmp['gain']}_{tstamp}.fits"
    badpix_hdu.writeto(fname_badpix_master, overwrite=True)
    print( f"   wrote bad pixel mask:\n    {fname_badpix_master}" )


    # ----------writing raw bais
    bias_hdu = fits.PrimaryHDU(bias_list)
    bias_hdu.header['EXTNAME'] = f'BIAS_GAIN-{gain}'  
    bias_hdu.header['UNITS'] = "ADU"
    bias_hdu.header['DATE'] = tstamp
    for k, v in config_tmp.items():
        bias_hdu.header[k] = v

    hdulist = fits.HDUList([bias_hdu])

    fname_raw = args.data_path +  f"RAW_BIAS/raw_bias_maxfps-{maxfps}_gain-{config_tmp['gain']}_{tstamp}.fits"
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

    # print("calculated master dark and appending the fits")
    # master_dark_hdulist.append( master_dark_hdu )
    fname_dark_master = args.data_path + f"MASTER_DARK/master_darks_adu_p_sec_fps-{config_tmp['fps']}_gain-{config_tmp['gain']}_{tstamp}.fits"
    master_dark_hdu.writeto(fname_dark_master, overwrite=True)
    print( f"   wrote master darks:\n    {fname_dark_master}" )

    # ----------writing MASTER BIAS
    master_bias_hdu = fits.PrimaryHDU(master_bias)
    master_bias_hdu.header['EXTNAME'] = f"BIAS_GAIN-{gain}"
    master_bias_hdu.header['UNITS'] = "ADU"
    master_bias_hdu.header['DATE'] = tstamp
    master_bias_hdu.header['SYSTEM'] = "BALDR-HEIM_CRED1"
    for k, v in config_tmp.items():
        master_bias_hdu.header[k] = v


    fname_bias_master = args.data_path + f"MASTER_BIAS/master_bias_fps-{config_tmp['fps']}_gain-{config_tmp['gain']}_{tstamp}.fits"
    master_bias_hdu.writeto(fname_bias_master, overwrite=True)
    print( f"   wrote master bias:\n    {fname_bias_master}" )


    #print("calculated master bias and appending the fits")
    #master_bias_hdulist.append( master_bias_hdu )

    # ---------- MASTER BAD PIXEL MAP
    #print("calculated master bad pixel map and appending the fits")
    #master_badpixel_hdulist.append( badpix_hdu )

    # ---------- CONVERSION GAIN MAP
    #print("calculated master conversion gain map and appending the fits")
    #master_convgain_hdulist.append( conv_gain_hdu )

# fname_dark_master = args.data_path + f"master_darks_adu_p_sec_fps-{config_tmp['fps']}_{tstamp}.fits"
# master_dark_hdulist.writeto(fname_dark_master, overwrite=True)
# print( f"   wrote master darks:\n    {fname_dark_master}" )

# fname_bias_master = args.data_path + f"master_bias_fps-{config_tmp['fps']}_{tstamp}.fits"
# master_bias_hdulist.writeto(fname_bias_master, overwrite=True)
# print( f"   wrote master bias:\n    {fname_bias_master}" )


# fname_badpix_master = args.data_path + f"master_bad_pixel_map_fps-{config_tmp['fps']}_{tstamp}.fits"
# master_badpixel_hdulist.writeto(fname_badpix_master, overwrite=True)
# print( f"   wrote bad pixel mask:\n    {fname_badpix_master}" )

# This gets done on internal flat field 
# fname_convgain_master = args.data_path + f"conversion_gain_map_fps-{config_tmp['fps']}_{tstamp}.fits"
# master_convgain_hdulist.writeto(fname_convgain_master, overwrite=True)
# print( f"   wrote conversion gain:\n    {fname_convgain_master}" )



# ---------- TEST SECTION ----------

print("\n--- Running validation test at gain = mid range ---")

gain = np.max([1, args.gains[len(args.gains)//2]])
# Set gain to 
c.send_fli_cmd(f"set gain {gain}")

# Get camera config and fps
fps = float(config_tmp["fps"])
dt = 1.0 / fps

# Acquire test dark
test_dark = np.mean(c.get_some_frames(number_of_frames=100, apply_manual_reduction=False), axis=0)  # [ADU]

# Load corresponding master bias and master dark
#master_bias_test = master_bias_hdulist[f"BIAS_GAIN-{gain}"].data   # gain = 5 → index = 4 (0-based)
#master_dark_test = master_dark_hdulist[f"DARK_GAIN-{gain}"].data   # [ADU/s]
#bad_pixel_mask = master_badpixel_hdulist[f"BAD_PIXEL_MAP_GAIN-{gain}"].data

master_bias_test_fits = fits.open( args.data_path + f"MASTER_BIAS/master_bias_fps-{config_tmp['fps']}_gain-{gain}_{tstamp}.fits" )
master_dark_test_fits = fits.open(args.data_path + f"MASTER_DARK/master_darks_adu_p_sec_fps-{config_tmp['fps']}_gain-{gain}_{tstamp}.fits" )
bad_pixel_mask_fits = fits.open(args.data_path + f"BAD_PIXEL_MAP/master_bad_pixel_map_fps-{config_tmp['fps']}_gain-{gain}_{tstamp}.fits" )


master_bias_test = master_bias_test_fits[0].data
master_dark_test = master_dark_test_fits[0].data
bad_pixel_mask = bad_pixel_mask_fits[0].data

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

