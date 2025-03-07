
########################################
## Get real image and interpolate theoretical reference intensity onto it! 

import matplotlib.pyplot as plt 
import os 
import toml
import numpy as np
import argparse
from asgard_alignment.DM_shm_ctrl import dmclass
from xaosim.shmlib import shm
from pyBaldr import utilities as util 

parser = argparse.ArgumentParser(description="interpolate theoretical intensity onto measured pupil")

default_toml = os.path.join( "config_files", "baldr_config_#.toml") #os.path.dirname(os.path.abspath(__file__)), "..", "config_files", "baldr_config.toml")

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
    default=[2],
    help="Comma-separated beam IDs to apply. Default: 1,2,3,4"
)

args = parser.parse_args()

beam_id = args.beam_id[0]

pupil_masks = {}
with open(args.toml_file.replace('#',f'{beam_id}') ) as file:
    pupildata = toml.load(file)

    # Extract the "baldr_pupils" section
    baldr_pupils = pupildata.get("baldr_pupils", {})
    pupil_masks[beam_id] = pupildata.get(f"beam{beam_id}", {}).get("pupil_mask", {}).get("mask", None)


# global camera image shm 
c = shm(args.global_camera_shm)

# # DMs
# dm_shm_dict = {}
# for beam_id in args.beam_id:
#     dm_shm_dict[beam_id] = dmclass( beam_id=beam_id )
#     # zero all channels
#     dm_shm_dict[beam_id].zero_all()
#     # activate flat 
#     dm_shm_dict[beam_id].activate_flat()
#     # apply DM flat offset 


img = np.mean( c.get_data() , axis=0)
r1,r2,c1,c2 = baldr_pupils[f"{beam_id}"]
meas_pupil = img[r1:r2, c1:c2] #

print( meas_pupil.shape)
print( np.array( pupil_masks[beam_id] ).shape)

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
mask = "H3"
F_number = 21.2
coldstop_diam = 4.8
mask_diam = 1.22 * F_number * wvl / phasemask_parameters[mask]['diameter']
eta = 0.647/4.82 #~= 1.1/8.2 (i.e. UTs) # ratio of secondary obstruction (UTs)


P_theory0 , Ic_theory0 = util.get_theoretical_reference_pupils( wavelength = wvl ,
                                              F_number = F_number , 
                                              mask_diam = mask_diam, 
                                              coldstop_diam=coldstop_diam,
                                              eta = eta, 
                                              diameter_in_angular_units = True, 
                                              get_individual_terms=False, 
                                              phaseshift = util.get_phasemask_phaseshift(wvl=wvl, depth = phasemask_parameters[mask]['depth'], dot_material='N_1405') , 
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


#### FINE ADJUSTMENT TO GET BEST OVERLAP WITH CLEAR PUPIL 
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


util.nice_heatmap_subplots( [delta] ,axis_off=False )
plt.savefig('delme.png')

ib, jb = np.unravel_index( np.argmin(delta) , delta.shape)

x_c = x_c0 + dxgrid[ib]
y_c = y_c0 + dygrid[jb]
P_theory_cred1 = util.interpolate_pupil_to_measurement( P_theory0, P_theory0, M, N, m, n, x_c, y_c, new_radius)
titles = ['Measured Pupil', "interpolated Theoretical Pupil", r"$\Delta$"]
imgs = [ meas_pupil_normed, P_theory_cred1, meas_pupil_normed-P_theory_cred1]
util.nice_heatmap_subplots(im_list = imgs, title_list=titles) 
plt.savefig('delme.png')

    
###### Now get thereotical ZWFS intensity 
Ic_theory_cred1 = util.interpolate_pupil_to_measurement( P_theory0, Ic_theory0, M, N, m, n, x_c, y_c, new_radius)


#Ic = a^2 + b^2 +2abcos(phi) so can clip and roughly scale the inner pupil by the measured clear pupil (Aprrox!!)
util.nice_heatmap_subplots(im_list = [ np.clip( meas_pupil_normed, 1 , 1.5) * Ic_theory_cred1], title_list=["I0 THEORY ON MEASUREMENT"]) 
plt.savefig('delme.png')
