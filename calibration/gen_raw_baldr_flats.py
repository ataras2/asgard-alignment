import zmq
from astropy.io import fits
import argparse
import time
import os
import glob
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

# Also to crop pupils and get extensions for each pupil - this could be this step
# or the next for generating master fat ... probably want it here since pupil regions 
# are defined in these raw flats 
# Either way we should ONLY define the pupil regions in one clear script 
# (detect cropped pupils .py) and here read them in from the toml file (or SHM address)
########################################################################

### For reducing the flats
def find_calibration_file_in_dated_subdir(base_path, calib_type, gain):
    """
    Search through dated subdirectories for the first calibration file matching the gain.

    Parameters
    ----------
    base_path : str
        Root calibration directory containing %d-%m-%Y subdirectories.
    calib_type : str
        One of ["MASTER_BIAS/", "MASTER_DARK/", "BAD_PIXEL_MAP/"]
    gain : int
        Gain value to search for.

    Returns
    -------
    filepath : str or None
        Path to the matching calibration file, or None if not found.
    found : bool
        True if file was found, else False.
    """

    assert calib_type in ["MASTER_BIAS/", "MASTER_DARK/", "BAD_PIXEL_MAP/"], \
        f"Unknown calibration type '{calib_type}'"

    if not os.path.exists(base_path):
        return None, False

    # Look for all dated subdirs
    # subdirs = [d for d in os.listdir(base_path)
    #         if os.path.isdir(os.path.join(base_path, d))]

    # for subdir in sorted(subdirs, reverse=True):  # Search newest first
    #     try:
    #         datetime.datetime.strptime(subdir, "%d-%m-%Y")
    #     except ValueError:
    #         continue

    #full_calib_dir = os.path.join(base_path, subdir, calib_type)
    full_calib_dir = os.path.join(base_path, calib_type)
    pattern = os.path.join(full_calib_dir, f"*gain-{gain}_*.fits")
    matches = glob.glob(pattern)

    if matches:
        # Sort matches by file modification time 
        matches.sort(key=os.path.getmtime, reverse=True)
        return matches[0], True

    return None, False 



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
    default=[1, 5, 20],
    help="List of gains to apply when collecting raw flat frames. Default: %(default)s"
)

# at gain = 20, fps = 500, aduoffset = 1000, ADU ~ 5000 in clear pupil.
parser.add_argument(
    '--fps',
    type=int,
    nargs='+',
    default=[500,1000,1739],
    help="List of frame rates for raw flats. Default: %(default)s"
)


parser.add_argument(
    '--phasemask_in',
    type=int,
    default=0,
    help="focal plane phasemask in (1) or out (0). Default: %(default)s"
)

parser.add_argument(
    '--phasemask',
    type=str,
    default="H3",
    help="which phasemask to use (TO DO), if FPM_STATE = 0, then we move 200um off this phasemask. Default: %(default)s"
)

parser.add_argument(
    '--use_baldr_dm_flat',
    action='store_true',
    help="Use the calibrated Baldr DM flat. If not set, the BMC factory flat is used (default)."
)
parser.add_argument("--number_of_frames", 
                    type=int, 
                    default=2000, 
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

if not args.phasemask_in:
    flat_type = "CLEAR" #+ f"{tstamp_rough}/"
    # TO DO : MOVE TO PHASEMASK THEN APPLY 200um offset 

else:
    flat_type = "ZWFS" #+ f"{tstamp_rough}/"
    # TO DO : MOVE TO PHASEMASK THEN 


for red_pth in [f"RAW_{flat_type}_FLATS", f"MASTER_{flat_type}_FLAT",f"CONVERSION_GAIN_{flat_type}"] :
    if not os.path.exists( args.data_path + red_pth  ):
        os.makedirs( args.data_path + red_pth  )
        print(f"made directory {args.data_path + red_pth}")


#cc =  shm("/dev/shm/cred1.im.shm")  # testing 
c = FLI.fli()

aduoffset = 1000 # hard coded for consistency 

# default aduoffset to avoid overflow of int on CRED 
c.send_fli_cmd(f"set aduoffset {aduoffset}")

# camera config
config_tmp = c.get_camera_config()


# DM 
from asgard_alignment.DM_shm_ctrl import dmclass


# DMs
dm = {}
for beam_id in [1,2,3,4]:
    dm[beam_id] = dmclass( beam_id=beam_id )
    # zero all channels
    dm[beam_id].zero_all()
    # activate flat 
    if args.use_baldr_dm_flat:
        dm[beam_id].activate_calibrated_flat()
    else:   
        dm[beam_id].activate_flat()


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
for gain in args.gains:

    print(f"   gain = {gain}")

    c.send_fli_cmd(f"set gain {gain}")

    # edit our cponfig dict without bothering to comunicate with camera
    config_tmp["gain"] = gain 

    for fps in args.fps:
        
        c.send_fli_cmd( f"set fps {fps}")

        config_tmp["fps"] = fps

        time.sleep(2)

        flat_list = c.get_some_frames(number_of_frames = args.number_of_frames, apply_manual_reduction=False, timeout_limit = 20000 )

        #master_flat = np.mean(flat_list, axis=0) # [adu]
        
        time.sleep(2)

        # ----------writing raw flats
        hdu = fits.PrimaryHDU(flat_list)
        if args.phasemask_in:
            hdu.header['EXTNAME'] = 'RAW_INTERNAL_FLAT_BDR_ZWFS'
        else:
            hdu.header['EXTNAME'] = 'RAW_INTERNAL_FLAT_BDR_CLEAR'
        hdu.header['PHASEMASK'] = args.phasemask
        hdu.header['UNITS'] = "ADU"
        hdu.header['DATE'] = tstamp
        hdu.header['SYSTEM'] = "BALDR-HEIM_CRED1"
        for k, v in config_tmp.items():
            hdu.header[k] = v

        # flat_hdulist.append( hdu )

        fname_raw = args.data_path +  f"RAW_{flat_type}_FLATS/raw_flats_{hdu.header['EXTNAME']}_phaasemask-{args.phasemask}_fps-{fps}_gain-{gain}_{tstamp}.fits"
        hdu.writeto(fname_raw, overwrite=True)
        print( f"   wrote raw flats for fps {fps}, gain {gain}:\n    {fname_raw}" )

        dependant_cals = ["MASTER_BIAS/", "MASTER_DARK/", "BAD_PIXEL_MAP/"]
        depend_data = {}
        for label in dependant_cals:
            file, file_found = find_calibration_file_in_dated_subdir(base_path=args.data_path, 
                                                                     calib_type=label, 
                                                                     gain=gain)
            #print( file , file_found)
            if file_found :
                with fits.open(file) as d:
                    depend_data[label] = d[0].data
    
        if len( dependant_cals ) != len( depend_data ):
            print(f"\n======\nMissing calibrations. Need one \n{dependant_cals}, \nonly have \n{depend_data.keys()}")
            missing_flag = 1 
        else:
            missing_flag = 0

            # BIAS has units ADU, DARK has units ADU/s
            master_flat = ( np.mean( flat_list ,axis=0)  \
                - depend_data["MASTER_BIAS/"] \
                - 1/fps * depend_data["MASTER_DARK/"] ) * fps # ADU/s
            
            # set bad pixels to zero
            master_flat[ depend_data["BAD_PIXEL_MAP/"] ] = 0 

            clipping = True 
            if clipping :
                hdu = fits.PrimaryHDU( np.clip( master_flat, 0, np.inf ) ) #ADU/s
            else:
                hdu = fits.PrimaryHDU( master_flat )#ADU/s
            
            if args.phasemask_in:
                hdu.header['EXTNAME'] = 'RAW_INTERNAL_FLAT_BDR_ZWFS'
            else:
                hdu.header['EXTNAME'] = 'RAW_INTERNAL_FLAT_BDR_CLEAR'            
            hdu.header['PHASEMASK'] = args.phasemask
            hdu.header['UNITS'] = "ADU/s"
            hdu.header['DATE'] = tstamp
            hdu.header['SYSTEM'] = "BALDR-HEIM_CRED1"
            for k, v in config_tmp.items():
                hdu.header[k] = v

            fname_master = args.data_path + f"MASTER_{flat_type}_FLAT/master_flat_{hdu.header['EXTNAME']}_phaasemask-{args.phasemask}_{tstamp}.fits"
            hdu.writeto(fname_master, overwrite=True)
            print( f"\n   wrote master flats for gain {gain}:\n    {fname_master}" )

            #import matplotlib.pyplot as plt 
            #plt.figure(); plt.imshow( np.clip( master_flat, 0 , np.inf)  );plt.colorbar(); plt.savefig("delme.png")
            #plt.figure(); plt.imshow(  master_flat  );plt.colorbar(); plt.savefig("delme.png")
 



            # ----------Estimate pixelwise conversion gain (e-/ADU)
            # Assumes the sinal is dominated by shot noise
            # The system is linear
            # The bias (offset) has been properly subtracted
            # THIS NEEDS TO BE DONE ON INTERNAL FLAT 
            master_flat_list = ( np.array( flat_list )  \
                - depend_data["MASTER_BIAS/"] \
                - 1/fps * depend_data["MASTER_DARK/"] ) # ADU 
            
            mean_dark = np.mean( master_flat_list, axis=0)
            var_dark = np.var( master_flat_list, axis=0) 
            with np.errstate(divide='ignore', invalid='ignore'):
                conversion_gain = np.where(var_dark > 0, mean_dark / var_dark, 0)

            conv_gain_hdu = fits.ImageHDU(conversion_gain)
            conv_gain_hdu.header['EXTNAME'] = f'CONV_GAIN_GAIN-{gain}'
            conv_gain_hdu.header['UNITS'] = 'e-/ADU'
            for k, v in config_tmp.items():
                conv_gain_hdu.header[k] = v    
            conv_gain_hdu.header['DATE'] = tstamp

            fname_convgain = args.data_path + f"CONVERSION_GAIN_{flat_type}/conversion_gain_{hdu.header['EXTNAME']}_phaasemask-{args.phasemask}_{tstamp}.fits"
            hdu.writeto(fname_convgain, overwrite=True)

            #import matplotlib.pyplot as plt 
            #plt.figure(); plt.imshow( conversion_gain  );plt.colorbar(); plt.savefig("delme.png")
    

# close DM,camera SHM and set gain = 1 
c.close(erase_file=False)

for beam_id in [1,2,3,4]:
    dm[beam_id].close(erase_file=False)

print(f"DONE.")


