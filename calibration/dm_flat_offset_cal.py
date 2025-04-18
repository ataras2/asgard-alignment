
########################################
## Get real image and interpolate theoretical reference intensity onto it! 

import matplotlib.pyplot as plt 
import os 
import toml
import numpy as np
import datetime
import subprocess
import argparse
import time 
import glob
import zmq
from astropy.io import fits
from asgard_alignment.DM_shm_ctrl import dmclass
from xaosim.shmlib import shm
from pyBaldr import utilities as util 
from asgard_alignment import FLI_Cameras as FLI
parser = argparse.ArgumentParser(description="interpolate theoretical intensity onto measured pupil")

default_toml = os.path.join( "config_files", "baldr_config_#.toml") #os.path.dirname(os.path.abspath(__file__)), "..", "config_files", "baldr_config.toml")

# this is in phasemask centering tools.py just need to update it with SHM .. to do
def move_relative_and_get_image(cam, beam, baldr_pupils, phasemask, savefigName=None, use_multideviceserver=True):
    print(
        f"input savefigName = {savefigName} <- this is where output images will be saved.\nNo plots created if savefigName = None"
    )
    r1,r2,c1,c2 = baldr_pupils[f"{beam}"]
    exit = 0
    while not exit:
        input_str = input('enter "e" to exit, else input relative movement in um: x,y')
        if input_str == "e":
            exit = 1
        else:
            try:
                xy = input_str.split(",")
                x = float(xy[0])
                y = float(xy[1])

                if use_multideviceserver:
                    #message = f"fpm_moveabs phasemask{beam} {[x,y]}"
                    #phasemask.send_string(message)
                    message = f"moverel BMX{beam} {x}"
                    phasemask.send_string(message)
                    response = phasemask.recv_string()
                    print(response)

                    message = f"moverel BMY{beam} {y}"
                    phasemask.send_string(message)
                    response = phasemask.recv_string()
                    print(response)

                else:
                    phasemask.move_relative([x, y])

                time.sleep(0.5)
                img = np.mean(
                    cam.get_data(),
                    axis=0,
                )[r1:r2,c1:c2]

            except:
                print('incorrect input. Try input "1,1" as an example, or "e" to exit')

    plt.close()


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
    help="TOML file to write/edit. Default: ../config_files/baldr_config.toml (relative to script)"
)

# Beam ids: provided as a comma-separated string and converted to a list of ints.
parser.add_argument(
    "--beam_id",
    type=lambda s: [int(item) for item in s.split(",")],
    default=[4],
    help="Comma-separated beam IDs to apply. Default: 1,2,3,4"
)

parser.add_argument(
    "--phasemask",
    type=str,
    default="H3",
    help="which phasemask do we use"
)

parser.add_argument(
    "--start_with_current_baldr_flat", 
    action="store_false",
    default=True,
    help="calibrate the Baldr flat starting with the current baldr flat. If False we beging with the BMC factory flat"
)
#--start_with_current_baldr_flat

args = parser.parse_args()

tstamp = datetime.datetime.now().strftime("%d-%m-%YT%H.%M.%S")
tstamp_rough =  datetime.datetime.now().strftime("%d-%m-%Y")

beam_id = args.beam_id[0]


host = "172.16.8.6"#"localhost"
port = 5555
timeout = 60000
context = zmq.Context()
context.socket(zmq.REQ)
mds_socket = context.socket(zmq.REQ)
mds_socket.setsockopt(zmq.RCVTIMEO, timeout)
mds_socket.connect( f"tcp://{host}:{port}")


pupil_masks = {}
I2A_dict = {}
with open(args.toml_file.replace('#',f'{beam_id}') ) as file:
    config_dict = toml.load(file)

    # Extract the "baldr_pupils" section
    baldr_pupils = config_dict.get("baldr_pupils", {})
    pupil_masks[beam_id] = config_dict.get(f"beam{beam_id}", {}).get("pupil_mask", {}).get("mask", None)
    I2A_dict[beam_id] = config_dict[f'beam{beam_id}']['I2A']


#---------- hardcoded phasemask and system parameters

# dictionary with depths referenced for beam 2 (1-5 goes big to small)
phasemask_parameters = {  
                        "J5": {"depth":0.474 ,  "diameter":32},
                        "J4": {"depth":0.474 ,  "diameter":36}, 
                        "J3": {"depth":0.474 ,  "diameter":44}, 
                        "J2": {"depth":0.474 ,  "diameter":54},
                        "J1": {"depth":0.474 ,  "diameter":65},
                        "H1": {"depth":0.654 ,  "diameter":68},  
                        "H2": {"depth":0.654 ,  "diameter":53}, 
                        "H3": {"depth":0.654 ,  "diameter":44}, 
                        "H4": {"depth":0.654 ,  "diameter":37},
                        "H5": {"depth":0.654 ,  "diameter":31}
                        }

T = 1900 #K lab thermal source temperature 
lambda_cut_on, lambda_cut_off =  1.38, 1.82 # um
wvl = util.find_central_wavelength(lambda_cut_on, lambda_cut_off, T) # central wavelength of Nice setup
F_number = 21.2
coldstop_diam = 4.8
mask_diam = 1.22 * F_number * wvl / phasemask_parameters[args.phasemask]['diameter']
eta = 0.647/4.82 #~= 1.1/8.2 (i.e. UTs) # ratio of secondary obstruction (UTs)


#---------- New Darks 
# run a new set of darks 
get_new_dark = False
if get_new_dark:
    script_path = "/home/asg/Progs/repos/asgard-alignment/calibration/gen_dark_bias_badpix.py"
    try:
        # Run the script and ensure it completes
        with subprocess.Popen(["python", script_path], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True) as process:
            stdout, stderr = process.communicate()  # Wait for process to complete

            if process.returncode == 0:
                print("Script executed successfully!")
                print(stdout)  # Print standard output (optional)
            else:
                print(f"Script failed with error:\n{stderr}")

    except Exception as e:
        print(f"Error running script: {e}")


#---------- CAMERA 
c = FLI.fli() #shm(args.global_camera_shm)
c.build_manual_bias(number_of_frames=200)
c.build_manual_dark(no_frames = 200 , build_bad_pixel_mask=True, kwargs={'std_threshold':20, 'mean_threshold':6} )

# # read in dark and get bad pixels 

# tstamp_rough = datetime.datetime.now().strftime("%d-%m-%Y")
# dark_fits_files = glob.glob(f"/home/asg/Progs/repos/asgard-alignment/calibration/cal_data/calibration_frames/{tstamp_rough}/MASTER_DARK/*fits")
# #"/home/asg/Progs/repos/asgard-alignment/calibration/cal_data/darks/*.fits") 
# most_recent_dark = max(dark_fits_files, key=os.path.getmtime) 

# dark_fits = fits.open( most_recent_dark )

# bad_pixels, bad_pixel_mask = FLI.get_bad_pixels( dark_fits["DARK_FRAMES"].data, std_threshold=10, mean_threshold=10)


# bad_pixel_mask[0][0] = False # the frame tag should not be masked! 
# c.reduction_dict['bad_pixel_mask'].append( (~bad_pixel_mask).astype(int) )
# c.reduction_dict['dark'].append(  dark_fits["MASTER DARK"].data.astype(int) )


# plt.figure(); plt.imshow( np.log10( dark_fits["MASTER DARK"].data ) ) ;plt.colorbar(); plt.savefig('delme.png')
# plt.figure(); plt.imshow( dark_fits["MASTER DARK"].data[r1:r2,c1:c2] ) ;plt.savefig('delme.png')
# plt.figure(); plt.imshow( bad_pixel_mask[r1:r2,c1:c2] ) ;plt.savefig('delme.png')


#---------- DMs
dm_shm_dict = {}
for beam_id in args.beam_id:
    dm_shm_dict[beam_id] = dmclass( beam_id=beam_id )
    # zero all channels
    dm_shm_dict[beam_id].zero_all()
    # activate flat 
    if not args.start_with_current_baldr_flat:
        dm_shm_dict[beam_id].activate_flat()
    else:
        dm_shm_dict[beam_id].activate_calibrated_flat()
    # apply DM flat offset 


#---------- Move to phase mask
message = f"fpm_movetomask phasemask{beam_id} {args.phasemask}"
mds_socket.send_string(message)
response = mds_socket.recv_string()#.decode("ascii")
print(f"moved to phasemask {args.phasemask}")

#---------- Clear pupil reference 
BMX_offset_tmp = 200.0 #um
message = f"moverel BMX{beam_id} {BMX_offset_tmp}"
mds_socket.send_string(message)
response = mds_socket.recv_string()#.decode("ascii")

# if "ACK" in response:
#     print(f"{BMX_offset_tmp}um offset successfully applied to BMX{beam_id}")
# else:
#     print(f"Failed to apply offset to BMX{beam_id}. Response: {response}")

time.sleep(10)


# get images
img = np.mean( c.get_data( apply_manual_reduction=True) , axis=0)
r1,r2,c1,c2 = baldr_pupils[f"{beam_id}"]
meas_pupil = img[r1:r2, c1:c2] #

### CLEAR REFERENCE MEASURED 
N0_m = img[r1:r2, c1:c2]

############## +++++++
plt.figure(); plt.imshow( N0_m / np.max( N0_m[pupil_masks[beam_id]]) ) ; plt.colorbar(); plt.savefig('delme.png')
#plt.figure(); plt.imshow( bad_pixel_mask[r1:r2,c1:c2]) ;plt.savefig('delme.png')

print( meas_pupil.shape)
print( np.array( pupil_masks[beam_id] ).shape)

time.sleep(2)


message = f"moverel BMX{beam_id} {-BMX_offset_tmp}"
mds_socket.send_string(message)
response = mds_socket.recv_string()#.decode("ascii")
# if "ACK" in response:
#     print(f"moving back... {-BMX_offset_tmp}um offset successfully applied to BMX{beam_id}")
# else:
#     print(f"Failed to apply offset to BMX{beam_id}. Response: {response}")


### ENSURE IT'S CENTERED PROPERLY AGAIN 
#move_relative_and_get_image(cam=c, beam=beam_id, baldr_pupils=baldr_pupils, phasemask=mds_socket, savefigName='delme.png', use_multideviceserver=True)
input('remember to move phasemask back in and center well!')

#---------- theoretical intensities mapped to CRED 1

P_theory0 , Ic_theory0 = util.get_theoretical_reference_pupils( wavelength = wvl ,
                                              F_number = F_number , 
                                              mask_diam = mask_diam, 
                                              coldstop_diam=coldstop_diam,
                                              eta = eta, 
                                              diameter_in_angular_units = True, 
                                              get_individual_terms=False, 
                                              phaseshift = util.get_phasemask_phaseshift(wvl=wvl, depth = phasemask_parameters[args.phasemask]['depth'], dot_material='N_1405') , 
                                              padding_factor = 6, 
                                              debug= False, 
                                              analytic_solution = False )


M = P_theory0.shape[0]
N = P_theory0.shape[1]

m = meas_pupil.shape[1]
n = meas_pupil.shape[0]

# A = pi * r^2 => r = sqrt( A / pi)
new_radius = (np.array(pupil_masks[beam_id]).sum()/np.pi)**0.5

x_c0, y_c0 = util.get_mask_center(pupil_masks[beam_id],  method='2') #np.mean(  np.where( pupil_masks[beam_id] )[0])

P_theory_cred1 = util.interpolate_pupil_to_measurement( P_theory0, P_theory0, M, N, m, n, x_c0, y_c0, new_radius)

# CHECK the centers are right!
# fig, ax = plt.subplots()
# mask = np.array( pupil_masks[beam_id])
# ax.imshow(mask, cmap='gray', extent=[-0.5, mask.shape[1]-0.5, mask.shape[0]-0.5, -0.5])

# # Overlay scatter point (assuming y_c, x_c are correct)
# ax.scatter([y_c0], [x_c0], color='red', marker='x', s=100, label='Center?')
# plt.legend()
# plt.savefig('delme.png') 


##---------- FINE ADJUSTMENT TO GET BEST OVERLAP WITH CLEAR PUPIL 
meas_pupil_normed = meas_pupil / np.mean(meas_pupil[pupil_masks[beam_id]])
meas_pupil_normed -= np.min( meas_pupil_normed)
meas_pupil_normed /= np.mean(meas_pupil_normed[pupil_masks[beam_id]])

# Look at residual of the pupil 
#meas_pupil_normed = meas_pupil / np.mean(meas_pupil[pupil_masks[beam_id]])
titles = ['Measured Pupil', "interpolated Theoretical Pupil", r"$\Delta$"]
imgs = [ meas_pupil_normed, P_theory_cred1, meas_pupil_normed-P_theory_cred1]
util.nice_heatmap_subplots(im_list = imgs, title_list=titles) 
plt.savefig('delme.png')

dxgrid = np.linspace(-3,3,15)
dygrid = np.linspace(-3,3,15)
delta = np.zeros( [dxgrid.shape[0], dygrid.shape[0]] )
for i,dx in enumerate(dxgrid):
    x_c = x_c0 + dx
    for j,dy in enumerate(dygrid):
        y_c = y_c0 + dy 
        P_theory_cred1 = util.interpolate_pupil_to_measurement( P_theory0, P_theory0, M, N, m, n, x_c, y_c, new_radius)

        rmse = np.sqrt( ( np.sum ( meas_pupil_normed > 0.5 - P_theory_cred1)**2) )
        delta[i,j] = rmse 


# util.nice_heatmap_subplots( [delta] ,cbar_label_list=["pupil RMSE"], title_list=["RMSE map"], axis_off=True )
# plt.savefig('delme.png')

ib, jb = np.unravel_index( np.argmin(delta) , delta.shape)

x_c = x_c0 + dxgrid[ib]
y_c = y_c0 + dygrid[jb]
P_theory_cred1 = util.interpolate_pupil_to_measurement( P_theory0, P_theory0, M, N, m, n, x_c, y_c, new_radius)
titles = ['Measured Pupil', "interpolated Theoretical Pupil", r"$\Delta$"]
imgs = [ meas_pupil_normed, P_theory_cred1, meas_pupil_normed-P_theory_cred1]

util.nice_heatmap_subplots(im_list = imgs, title_list=titles) 
plt.savefig('delme.png')

    
###### Now get thereotical ZWFS intensity 
#Ic_theory_cred1 = util.interpolate_pupil_to_measurement( P_theory0, Ic_theory0, M, N, m, n, x_c, y_c, new_radius)

#Ic = a^2 + b^2 +2abcos(phi) so can clip and roughly scale the inner pupil by the measured clear pupil (Aprrox!!)
#util.nice_heatmap_subplots(im_list = [ np.clip( meas_pupil_normed, 1 , 1.5) * Ic_theory_cred1], title_list=["I0 THEORY ON MEASUREMENT"]) 
#plt.savefig('delme.png')



I0_t = util.interpolate_pupil_to_measurement( P_theory0, Ic_theory0, M, N, m, n, x_c, y_c, new_radius)


img = np.mean( c.get_data( apply_manual_reduction=True) , axis=0)
r1,r2,c1,c2 = baldr_pupils[f"{beam_id}"]
I0_m = img[r1:r2, c1:c2] #




# # MOVE PHASEMASK OUT MANUALLY 
# img = np.mean( c.get_data() , axis=0)
# N0_m = img[r1:r2, c1:c2]


# img_raw = c.get_data()

# ## Identify bad pixels (this can throw it off!!)
# mean_frame = np.mean(img_raw, axis=0)
# std_frame = np.std(img_raw, axis=0)

# global_mean = np.mean(mean_frame)
# global_std = np.std(mean_frame)
# bad_pixel_map = (np.abs(mean_frame - global_mean) > 5.5 * global_std) | (std_frame > 10 * np.median(std_frame))

# plt.figure()
# plt.imshow( bad_pixel_map[r1:r2, c1:c2] ) #[ crop_pupil_coords[i][2]:crop_pupil_coords[i][3],crop_pupil_coords[i][0]:crop_pupil_coords[i][1]])
# plt.colorbar()
# plt.savefig( "delme.png")



# normalize by mean clear pupil 
I0_m *= 1/np.mean( N0_m[pupil_masks[beam_id]] ) 

I0_m_dm = I2A_dict[beam_id] @ I0_m.reshape(-1)

I0_t_dm = I2A_dict[beam_id] @ I0_t.reshape(-1)

I0_t_dm -= np.nanmin(I0_t_dm)#np.nanmean(I0_t_dm)
I0_t_dm /= (np.nanmax(I0_t_dm)-np.nanmin(I0_t_dm))  #np.nanstd(I0_t_dm)

I0_m_dm -= np.nanmin(I0_m_dm)
I0_m_dm /= (np.nanmax(I0_m_dm) - np.nanmin(I0_m_dm)) 

# the difference 
delta_cmd = I0_m_dm - I0_t_dm 

#######++++++++++
im_list = [ np.nan_to_num( util.get_DM_command_in_2D(ii), 0) for ii in [I0_t_dm,  I0_m_dm ,  delta_cmd ]]
titles = ["I0 THEORY", "I0 MEASURED", r'$\Delta$']
util.nice_heatmap_subplots(im_list = im_list, title_list=titles )
plt.savefig('delme.png')


# try and feedback 

# the basis aberration to try flatten wavefront

cc140 = delta_cmd # 140 length vector 
# convert to 2D for SHM format 
cc = np.nan_to_num( util.get_DM_command_in_2D( delta_cmd ) , 0) 

exterior_mask_rough = util.filter_exterior_annulus(np.array( pupil_masks[beam_id] ), inner_radius=8, outer_radius=12)

util.nice_heatmap_subplots( [pupil_masks[beam_id], I0_m + 1000*np.roll( np.roll( exterior_mask_rough, -1, axis=0), -1, axis=1) ] ) #, axis=0) ] )
plt.savefig('delme.png')

exterior_mask = np.roll( np.roll( exterior_mask_rough, -1, axis=0), -1, axis=1)

# exterior_mask = (abs(I0_m - N0_m/np.mean( N0_m[pupil_masks[beam_id]] ) ) > 0.1 ) * (~np.array(pupil_masks[beam_id]))
# plt.figure(); plt.imshow( exterior_mask)  ;plt.colorbar() ;plt.savefig('delme.png')

I0_list = []
I0_dm_list = []
delta_list = []
rmse = []
exterior_sig = []
amps = np.linspace( -0.07, 0.07, 20)
fig_path = f"/home/asg/Progs/repos/asgard-alignment/calibration/reports/flatdm_trials/{tstamp_rough}/beam{beam_id}/"
if not os.path.exists(fig_path):
    os.makedirs(fig_path)


for aa in amps:
    print(aa)
    dm_shm_dict[beam_id].set_data( aa * cc )
    time.sleep(5)

    img = np.mean( c.get_data(apply_manual_reduction=True) , axis=0)
    # need MASK BAD PIXELS 
    r1,r2,c1,c2 = baldr_pupils[f"{beam_id}"]

    I0_m = img[r1:r2, c1:c2]
    I0_m -= np.nanmin(I0_m)
    I0_m /= (np.nanmax(I0_m) - np.nanmin(I0_m)) 

    exterior_sig.append( np.mean( I0_m[exterior_mask] ) )

    I0_list.append( I0_m )

    I0_m_dm = I2A_dict[beam_id] @ I0_m.reshape(-1)
    I0_dm_list.append( I0_m_dm )

    delta_cmd = I0_m_dm - I0_t_dm 

    delta_list.append( delta_cmd )

    rmse.append( np.sqrt( np.sum( (delta_cmd)**2) ) )

    
    util.nice_heatmap_subplots(im_list = [I0_t, I0_m], title_list=["I0 theory", "I0 measured"], cbar_label_list=["Normalize","Normalize"] )
    plt.savefig(fig_path + f'flatcal_beam{beam_id}_I0s_amp-{aa}.png')

    util.nice_heatmap_subplots(im_list = [util.get_DM_command_in_2D(iii) for iii in [I0_t_dm, I0_m_dm]], title_list=["I0 theory\nprojected on DM", "I0 measured\nprojected on"], cbar_label_list=["Normalize","Normalize"] )
    plt.savefig(fig_path + f'flatcal_beam{beam_id}_I0s_on_DM_amp-{aa}.png')


plt.figure()
plt.plot( amps , exterior_sig )
plt.ylabel(r'$\Sigma$ exterior pixels',fontsize=15)
plt.xlabel(r'a.$\Delta$',fontsize=15)
plt.gca().tick_params(labelsize=15)
plt.savefig(fig_path + f'dm_flat_cal_beam{beam_id}_DifrractedSignal.png')

plt.figure()
plt.plot( amps , rmse )
plt.ylabel('RMSE',fontsize=15)
plt.xlabel(r'a.$\Delta$',fontsize=15)
plt.gca().tick_params(labelsize=15)
plt.savefig(fig_path + f'dm_flat_cal_beam{beam_id}_RMSE.png')

# seems to work best using experior 
ib = np.argmax( exterior_sig ) #np.argmin( rmse )

dm_shm_dict[beam_id].set_data( amps[ib] * cc )

#dm_shm_dict[beam_id].shms[1].set_data( amps[ib] * cc )

if not args.start_with_current_baldr_flat:
    best_baldr_flat_offset_2D = amps[ib] * cc 
    best_baldr_flat_offset = amps[ib] * cc140
else : # we must also add in the previous DM flat
    # add previous flat we used here 
    best_baldr_flat_offset_2D =  amps[ib] * cc + dm_shm_dict[beam_id].cmd_2_map2D( dm_shm_dict[beam_id].get_baldr_flat_offset()  )
    best_baldr_flat_offset = amps[ib] * cc140 + dm_shm_dict[beam_id].get_baldr_flat_offset() 
# savefits 
# Convert to a dictionary

data_dict = {
    "best_DM_flat": best_baldr_flat_offset,
    "flat_used":dm_shm_dict[beam_id].shms[0].get_data(),
    "basis_aberration": np.array(cc),
    "amps":np.array(cc),
    "DM_cmds":np.array( [ a*cc for a in amps] ),
    "I0_theory": np.array( I0_t ),
    "I0_theory_dm": np.array( I0_t_dm ),
    "I2A": np.array( I2A_dict[beam_id] ),
    "I0_list": np.array(I0_list),
    "I0_dm_list": np.array(I0_dm_list),
    "delta_list": np.array(delta_list),
    "rmse": np.array(rmse),
    "exterior_sig": np.array(exterior_sig),
    "pupil_pixels": np.array( pupil_masks[beam_id] ).astype(int),
    "exterior_pixels": np.array( exterior_mask ).astype(int),
}

# Create a FITS HDUList
hdul = fits.HDUList()

# Iterate through the dictionary and add each array as an ImageHDU
for key, value in data_dict.items():
    hdu = fits.ImageHDU(value)
    hdu.header['EXTNAME'] = key  # Set the EXTNAME header
    hdu.header['FACTORY_FLAT_START'] = int( args.start_with_current_baldr_flat )
    hdul.append(hdu)

# Define the output FITS filename
#mask_id = 'H3'
fits_filename = fig_path + f"flat_dm_beam{beam_id}_mask{args.phasemask}_{tstamp}.fits"

# Write to FITS file
hdul.writeto(fits_filename, overwrite=True)

print(f"Saved FITS file as {fits_filename}")



### Save as txt file the flat 
dmshape_save_path = f"/home/asg/Progs/repos/asgard-alignment/DMShapes/"
flat_fname = dmshape_save_path + f"BEAM{beam_id}_FLAT_MAP_OFFSETS_{tstamp}.txt"
np.savetxt(flat_fname, best_baldr_flat_offset, fmt="%.7f")
print(f"saved the new flat {flat_fname}")