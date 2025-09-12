#!/usr/bin/env python
import numpy as np 
import zmq
import time
import toml
import os 
import argparse
import matplotlib.pyplot as plt
import argparse
import subprocess
import glob

from astropy.io import fits
from scipy.signal import TransferFunction, bode
from types import SimpleNamespace
import asgard_alignment.controllino as co # for turning on / off source \
from asgard_alignment.DM_shm_ctrl import dmclass
import common.DM_basis_functions as dmbases
import asgard_alignment.controllino as co
import common.phasemask_centering_tool as pct
import common.phasescreens as ps 
import pyBaldr.utilities as util 
import pyzelda.ztools as ztools
import datetime
from xaosim.shmlib import shm
from asgard_alignment import FLI_Cameras as FLI


# By default HO in this construction of the IM will always contain zonal actuation of each DM actuator.
# Using LO we can also define our Lower order modes on a Zernike basis where LO 
# is the Noll index up to which modes to consider. These LO modes are probed first
# in the IM and then the HO (zonal) modes are probed  


MDS_port = 5555
MDS_host = "192.168.100.2" # simmode : "127.0.0.1" #'localhost'
context = zmq.Context()
context.socket(zmq.REQ)
socket = context.socket(zmq.REQ)
socket.setsockopt(zmq.RCVTIMEO, 5000)
server_address = f"tcp://{MDS_host}:{MDS_port}"
socket.connect(server_address)
state_dict = {"message_history": [], "socket": socket}




def send_and_get_response(message):
    # st.write(f"Sending message to server: {message}")
    state_dict["message_history"].append(
        f":blue[Sending message to server: ] {message}\n"
    )
    state_dict["socket"].send_string(message)
    response = state_dict["socket"].recv_string()
    if "NACK" in response or "not connected" in response:
        colour = "red"
    else:
        colour = "green"
    # st.markdown(f":{colour}[Received response from server: ] {response}")
    state_dict["message_history"].append(
        f":{colour}[Received response from server: ] {response}\n"
    )

    return response.strip()


def plot2d( thing ):
    plt.figure()
    plt.imshow(thing)
    plt.colorbar()
    plt.savefig('/home/asg/Progs/repos/asgard-alignment/delme.png')
    plt.close()

# split_mode 1 
#aa = shm("/dev/shm/baldr1.im.shm")
#util.nice_heatmap_subplots( [ aa.get_data() ],savefig='delme.png')

parser = argparse.ArgumentParser(description="Interaction and control matricies.")

default_toml = os.path.join("/usr/local/etc/baldr/", "baldr_config_#.toml") 

# Camera shared memory path
parser.add_argument(
    "--global_camera_shm",
    type=str,
    default="/dev/shm/cred1.im.shm",
    help="Camera shared memory path. Default: /dev/shm/cred1.im.shm"
)

# TOML file path; default is relative to the current file's directory.
parser.add_argument(
    "--toml_file",
    type=str,
    default=default_toml,
    help="TOML file pattern (replace # with beam_id) to write/edit. Default: ../config_files/baldr_config.toml (relative to script)"
)

# Beam ids: provided as a comma-separated string and converted to a list of ints.
parser.add_argument(
    "--beam_id",
    type=lambda s: [int(item) for item in s.split(",")],
    default=[2], # 1, 2, 3, 4],
    help="Comma-separated beam IDs to apply. Default: 1,2,3,4"
)

parser.add_argument(
    "--phasemask",
    type=str,
    default="H3",
    help="Comma-separated beam IDs to apply. Default: 1,2,3,4"
)

parser.add_argument(
    "--LO",
    type=int,
    default=2,
    help="Up to what zernike order do we consider Low Order (LO). 2 is for tip/tilt, 3 would be tip,tilt,focus etc). Default: %(default)s"
)


# parser.add_argument(
#     "--basis_name",
#     type=str,
#     default="zonal",
#     help="basis used to build interaction matrix (IM). zonal, zernike, zonal"
# )

# parser.add_argument(
#     "--Nmodes",
#     type=int,
#     default=10,
#     help="number of modes to probe"
# )

parser.add_argument(
    "--poke_amp",
    type=float,
    default=0.05,
    help="amplitude to poke DM modes for building interaction matrix"
)

parser.add_argument(
    "--signal_space",
    type=str,
    default='dm',
    help="what space do we consider the signal on. either dm (uses I2A) or pixel"
)

parser.add_argument(
    "--DM_flat",
    type=str,
    default="baldr",
    help="What flat do we use on the DM during the calibration. either 'baldr' or 'factory'. Default: %(default)s"
)

parser.add_argument(
    '--cam_fps',
    type=int,
    default=1000,
    help="frames per second on camera. Default: %(default)s"
)


parser.add_argument(
    '--cam_gain',
    type=int,
    default=10,
    help="camera gain. Default: %(default)s"
)

parser.add_argument("--fig_path", 
                    type=str, 
                    default='/home/asg/Progs/repos/asgard-alignment/calibration/reports/test/', 
                    help="path/to/output/image/ for the saved figures")



args=parser.parse_args()


# c, dms, darks_dict, I0_dict, N0_dict,  baldr_pupils, I2A = setup(args.beam_id,
#                               args.global_camera_shm, 
#                               args.toml_file) 

NNN= 10 # how many groups of 100 to take for reference images 

I2A_dict = {}
pupil_mask = {}
secondary_mask = {}
exterior_mask = {}
for beam_id in args.beam_id:

    # read in TOML as dictionary for config 
    with open(args.toml_file.replace('#',f'{beam_id}'), "r") as f:
        config_dict = toml.load(f)
        # Baldr pupils from global frame 
        baldr_pupils = config_dict['baldr_pupils']
        I2A_dict[beam_id] = config_dict[f'beam{beam_id}']['I2A']
        
        pupil_mask[beam_id] = config_dict.get(f"beam{beam_id}", {}).get("pupil_mask", {}).get("mask", None)

        secondary_mask[beam_id] = np.array(config_dict.get(f"beam{beam_id}", {}).get("pupil_mask", {}).get("secondary", None) )

        exterior_mask[beam_id] = np.array(config_dict.get(f"beam{beam_id}", {}).get("pupil_mask", {}).get("exterior", None) )




c = FLI.fli(args.global_camera_shm, roi = [None,None,None,None])

# read the data to get directly the number of reads without reset (this is what the buffer is typically set to in non-destructive read mode)
nrs = c.mySHM.get_data().shape[0] 

# mask offset 
rel_offset = 200 

#Clear Pupil
print( 'gettin clear pupils')
N0s = c.get_data( apply_manual_reduction=True  ) #get_some_frames( number_of_frames = 1000,  apply_manual_reduction=True ) 
inner_pupil_filt = {} # strictly inside (not on boundary)

for beam_id in args.beam_id:
    
    r1,r2,c1,c2 = baldr_pupils[f"{beam_id}"]
    
    clear_pupils[beam_id] = N0s[:,r1:r2,c1:c2]

    #bad_pix_mask_tmp = np.array( c.reduction_dict["bad_pixel_mask"][-1][r1:r2,c1:c2] ).astype(bool)

    # move back 
    print( 'Moving FPM back in beam.')
    message = f"moverel BMX{beam_id} {-rel_offset}"
    res = send_and_get_response(message)
    print(res) 
    time.sleep(1)
    message = f"moverel BMY{beam_id} {-rel_offset}"
    res = send_and_get_response(message)
    print(res) 
    time.sleep(5)

    inner_pupil_filt[beam_id] = util.remove_boundary(pupil_mask[beam_id])

    # set as clear pupils where we set exterior and bad pixels to mean interior clear pup signal

    # filter exterior pixels (that risk 1/0 error)
    pixel_filter = secondary_mask[beam_id].astype(bool)  | (~inner_pupil_filt[beam_id].astype(bool) ) #| (~bad_pix_mask_tmp )
    
    normalized_pupils[beam_id] = np.mean( clear_pupils[beam_id] , axis=0) 
    normalized_pupils[beam_id][ pixel_filter  ] = np.mean( np.mean(clear_pupils[beam_id],0)[~pixel_filter]  ) # set exterior and boundary pupils to interior mean




# ZWFS Pupil
input("phasemasks aligned? ensure alignment then press enter")

time.sleep(5)

print( 'Getting ZWFS pupils')
I0s = c.get_data( apply_manual_reduction=True ) #get_some_frames( number_of_frames = 1000,  apply_manual_reduction=True ) 

for beam_id in args.beam_id:

    #I0s = []
    #for _ in range(NNN):
    #    I0s.append(  c_dict[beam_id].get_data( apply_manual_reduction=True ) )
    #I0s = np.array(  I0s ).reshape(-1,  I0s[0].shape[1],  I0s[0].shape[2])

    r1,r2,c1,c2 = baldr_pupils[f"{beam_id}"]
    #cropped_img = interpolate_bad_pixels(img[r1:r2, c1:c2], bad_pixel_mask[r1:r2, c1:c2])
    #cropped_img = [nn[r1:r2,c1:c2] for nn in I0s] #/np.mean(img[r1:r2, c1:c2][pupil_mask[bb]])
    zwfs_pupils[beam_id] = I0s[:,r1:r2,c1:c2] #cropped_img




modal_basis = np.array(  dmbases.zer_bank(2, 10 ) ) 

M2C = modal_basis.copy().reshape(modal_basis.shape[0],-1).T 


# zonal_basis = np.array([dm_shm_dict[beam_id].cmd_2_map2D(ii) for ii in np.eye(140)]) 

# modal_basis = np.array( LO_basis.tolist() +  zonal_basis.tolist() ) 

# M2C = modal_basis.copy().reshape(modal_basis.shape[0],-1).T # mode 2 command matrix 



if args.signal_space.lower() not in ["dm", "pixel"] :
    raise UserWarning("signal space must either be 'dm' or 'pixel'")

#cam_config = c.config

IM = {beam_id:[] for beam_id in args.beam_id}
Iplus_all = {beam_id:[] for beam_id in args.beam_id}
Iminus_all = {beam_id:[] for beam_id in args.beam_id}



#imgs_to_mean = 20 # for each poke we average this number of frames
# for now we use standard get_data mehtod which is 200 frames (april 2025)
for i,m in enumerate(modal_basis):
    print(f'executing cmd {i}/{len(modal_basis)}')
    #if i == args.LO:
    #    input("close Baldr TT and ensure stable. Then press enter.")
    I_plus_list = {beam_id:[] for beam_id in args.beam_id}
    I_minus_list = {beam_id:[] for beam_id in args.beam_id}
    for sign in [(-1)**n for n in range(20)]: #range(10)]: #[-1,1]:
        
        for beam_id in args.beam_id:
            dm_shm_dict[beam_id].set_data(  sign * args.poke_amp/2 * m ) 
        
        #print( "sleep", float(c.config["fps"]) )
        #time.sleep( nbreadworeset / 1 ) #float(c.config["fps"])) # 200 because get data takes 200 frames
        
        # wait for a new buffer to fill before we read the buffer and average it.
        t0 = c.mySHM.get_counter()
        cnt = 0
        while cnt < 2 * nrs : # wait at least 1 buffers before we average buffer 
            t1 = c.mySHM.get_counter()
            cnt = t1 - t0 
            time.sleep( 1/float(c.config['fps']) )
        del cnt, t1, t0 # delete when finished
        
        imgtmp_global = c.get_data(apply_manual_reduction = True )
        # quick version below just for testing . Use full ^ grab above for proper cal.
        #imgtmp_global = np.array([c.get_image(apply_manual_reduction = True ) ,c.get_image(apply_manual_reduction = True )] )#get_data(apply_manual_reduction = True ) # get_some_frames( number_of_frames = imgs_to_mean, apply_manual_reduction = True )

        for beam_id in args.beam_id:
            r1,r2,c1,c2 = baldr_pupils[f'{beam_id}']
            if sign > 0:
                
                I_plus_list[beam_id].append( list( np.mean( imgtmp_global[:,r1:r2,c1:c2], axis = 0)  ) )
                #I_plus *= 1/np.mean( I_plus )

            if sign < 0:
                
                I_minus_list[beam_id].append( list( np.mean( imgtmp_global[:,r1:r2,c1:c2], axis = 0)  ) )
                #I_minus *= 1/np.mean( I_minus )


    for beam_id in args.beam_id:
        I_plus = np.mean( I_plus_list[beam_id], axis = 0).reshape(-1) / normalized_pupils[beam_id].reshape(-1)
        I_minus = np.mean( I_minus_list[beam_id], axis = 0).reshape(-1) /  normalized_pupils[beam_id].reshape(-1)

        #errsig = dm_mask[beam_id] * ( I2A_dict[beam_id] @ ((I_plus - I_minus))  )  / args.poke_amp  # dont use pokeamp norm so I2M maps to naitive DM units (checked in /Users/bencb/Documents/ASGARD/Nice_March_tests/IM_zernike100/SVD_IM_analysis.py)
        #errsig = #( I2A_dict[beam_id] @ ((I_plus - I_minus))  )  / args.poke_amp  # dont use pokeamp norm so I2M maps to naitive DM units (checked in /Users/bencb/Documents/ASGARD/Nice_March_tests/IM_zernike100/SVD_IM_analysis.py)
        
        # Try minimize dependancies, if I2A not calibrated or DM mask then the above fails.. keep simple. We can deal with this in post processing
        
        #############
        #############
        
        # removing seoconary pixels 
        #(~secondary_mask[beam_id].astype(bool)).reshape(-1) 

        if args.signal_space.lower() == 'dm':
            errsig = I2A_dict[beam_id] @ ( float( c.config["gain"] ) / float( c.config["fps"] )  * (I_plus - I_minus)  / args.poke_amp ) # 1 / DMcmd * (s * gain)  projected to DM space
        elif args.signal_space.lower() == 'pixel':
            errsig = ( float( c.config["gain"] ) / float( c.config["fps"] )  * (I_plus - I_minus)  / args.poke_amp ) # 1 / DMcmd * (s * gain)  projected to Pixel space
        
        #############
        #############

        # reenter pokeamp norm <- this is used for detailed analysis sometimes
        #Iplus_all[beam_id].append( I_plus_list )
        #Iminus_all[beam_id].append( I_minus_list )

        IM[beam_id].append( list(  errsig.reshape(-1) ) ) 






hdul = fits.HDUList()

hdu = fits.ImageHDU(IM)
hdu.header['EXTNAME'] = 'IM'
hdu.header['units'] = "sec.gain/DMunit"
hdu.header['phasemask'] = args.phasemask
hdu.header['beam'] = beam_id
hdu.header['poke_amp'] = args.poke_amp
for k,v in c.config.items():
    hdu.header[k] = v 

for ii , ll in zip([r1,r2,c1,c2],["r1","r2","c1","c2"]) :
    hdu.header[ll] = ii
hdu.header["frame_shape"] = f"{r2-r1}x{c2-c1}"

hdul.append(hdu)


# hdu = fits.ImageHDU( dark_fits["DARK_FRAMES"].data )
# hdu.header['EXTNAME'] = 'DARKS'

hdu = fits.ImageHDU(Iplus_all)
hdu.header['EXTNAME'] = 'I+'
hdul.append(hdu)


hdu = fits.ImageHDU(clear_pupils[beam_id])
hdu.header['EXTNAME'] = 'N0'
hdul.append(hdu)

hdu = fits.ImageHDU(zwfs_pupils[beam_id])
hdu.header['EXTNAME'] = 'I0'
hdul.append(hdu)

hdu = fits.ImageHDU(normalized_pupils[beam_id])
hdu.header['EXTNAME'] = 'normalized_pupil'
hdul.append(hdu)


hdu = fits.ImageHDU(Iplus_all)
hdu.header['EXTNAME'] = 'I+'
hdul.append(hdu)

hdu = fits.ImageHDU(Iminus_all)
hdu.header['EXTNAME'] = 'I-'
hdul.append(hdu)

hdu = fits.ImageHDU( np.array(pupil_mask[beam_id]).astype(int)) 
hdu.header['EXTNAME'] = 'PUPIL_MASK'
hdul.append(hdu)

# hdu = fits.ImageHDU( dm_mask[beam_id] )
# hdu.header['EXTNAME'] = 'PUPIL_MASK_DM'
# hdul.append(hdu)

hdu = fits.ImageHDU(modal_basis)
hdu.header['EXTNAME'] = 'M2C'
hdul.append(hdu)

# hdu = fits.ImageHDU(I2M)
# hdu.header['EXTNAME'] = 'I2M'
# hdul.append(hdu)

hdu = fits.ImageHDU(I2A_dict[beam_id])
hdu.header['EXTNAME'] = 'interpMatrix'
hdul.append(hdu)

# hdu = fits.ImageHDU(zwfs_pupils[beam_id])
# hdu.header['EXTNAME'] = 'I0'
# hdul.append(hdu)

# hdu = fits.ImageHDU(clear_pupils[beam_id])
# hdu.header['EXTNAME'] = 'N0'
# hdul.append(hdu)

fits_file = '/home/asg/Videos/' + f'IM_full_{Nmodes}{basis_name}_beam{beam_id}_mask-{args.phasemask}_pokeamp_{args.poke_amp}_fps-{c.config["fps"]}_gain-{c.config["gain"]}.fits' #_{args.phasemask}.fits'
#f'IM_full_{Nmodes}ZERNIKE_beam{beam_id}_mask-H5_pokeamp_{poke_amp}.fits' #_{args.phasemask}.fits'
hdul.writeto(fits_file, overwrite=True)
print(f'wrote telemetry to \n{fits_file}')

