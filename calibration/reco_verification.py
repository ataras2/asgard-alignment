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
import argparse
import datetime 
import common.DM_basis_functions as dmbases
from asgard_alignment import FLI_Cameras as FLI
from asgard_alignment.DM_shm_ctrl import dmclass
import pyBaldr.utilities as util 


default_toml = os.path.join("config_files", "baldr_config_#.toml") 

tstamp_rough = datetime.datetime.now().strftime("%d-%m-%Y")
tstamp = datetime.datetime.now().strftime("%d-%m-%YT%H.%M.%S")

parser = argparse.ArgumentParser(description="verificar Baldr reconstructions")

# TOML file path; default is relative to the current file's directory.
parser.add_argument(
    "--toml_file",
    type=str,
    default=default_toml,
    help="TOML file pattern (replace # with beam_id) to write/edit. Default: ../config_files/baldr_config.toml (relative to script)"
)

# Camera shared memory path
parser.add_argument(
    "--global_camera_shm",
    type=str,
    default="/dev/shm/cred1.im.shm",
    help="Camera shared memory path. Default: /dev/shm/cred1.im.shm"
)


# Beam ids: provided as a comma-separated string and converted to a list of ints.
parser.add_argument(
    "--beam_id",
    type=int,
    default=2,
    help="what beam are we considering. Default: %(default)s"
)

parser.add_argument(
    "--phasemask",
    type=str,
    default="H3",
    help="what phasemask was used for building the IM. THis is to search the right entry in the configuration file. Default: %(default)s"
)


parser.add_argument(
    '--output_report_dir',
    type=str,
    default=f'/home/asg/Progs/repos/asgard-alignment/calibration/reports/{tstamp_rough}/reco_verification/',
    help="Output directory for calibration reports. Default: %(default)s"
)


args=parser.parse_args()

results_path = args.output_report_dir + f"beam{args.beam_id}/"

# write the directory to output results and analysis
os.makedirs(results_path, exist_ok=True)

# ===================================
# =========== READ IN CONFIGURATION
# ===================================
with open(args.toml_file.replace('#',f'{args.beam_id}'), "r") as f:

    config_dict = toml.load(f)
    
    # Baldr pupils from global frame 
    baldr_pupils = config_dict['baldr_pupils']
    I2A = np.array( config_dict[f'beam{args.beam_id}']['I2A'] )
    
    # image pixel filters
    pupil_mask = np.array(config_dict.get(f"beam{args.beam_id}", {}).get("pupil_mask", {}).get("mask", None) ).astype(bool)   # matrix bool
    exter_mask = np.array(config_dict.get(f"beam{args.beam_id}", {}).get("pupil_mask", {}).get("exterior", None) ).astype(bool) # matrix bool
    secon_mask = np.array(config_dict.get(f"beam{args.beam_id}", {}).get("pupil_mask", {}).get("secondary", None) ).astype(bool) # matrix bool

    # ctrl model 
    #IM = np.array(config_dict.get(f"beam{args.beam_id}", {}).get(f"{args.phasemask}", {}).get("ctrl_model", None).get("IM", None) ).astype(float)
    #I2M_raw = np.array(config_dict.get(f"beam{args.beam_id}", {}).get(f"{args.phasemask}", {}).get("ctrl_model", None).get("I2M", None) ).astype(float)
    I2M_LO_raw = np.array(config_dict.get(f"beam{args.beam_id}", {}).get(f"{args.phasemask}", {}).get("ctrl_model", None).get("I2M_LO", None) ).astype(float)
    I2M_HO_raw = np.array(config_dict.get(f"beam{args.beam_id}", {}).get(f"{args.phasemask}", {}).get("ctrl_model", None).get("I2M_HO", None) ).astype(float)
    
    M2C_LO = config_dict.get(f"beam{args.beam_id}", {}).get(f"{args.phasemask}", {}).get("ctrl_model", None).get("M2C_LO", None)
    M2C_HO = config_dict.get(f"beam{args.beam_id}", {}).get(f"{args.phasemask}", {}).get("ctrl_model", None).get("M2C_HO", None)

    #M2C = np.array(config_dict.get(f"beam{args.beam_id}", {}).get(f"{args.phasemask}", {}).get("ctrl_model", None).get("M2C", None) ).astype(float)
    I0 = np.array(config_dict.get(f"beam{args.beam_id}", {}).get(f"{args.phasemask}", {}).get("ctrl_model", None).get("I0", None) ).astype(float)
    N0 = np.array(config_dict.get(f"beam{args.beam_id}", {}).get(f"{args.phasemask}", {}).get("ctrl_model", None).get("N0", None) ).astype(float)
    N0i = np.array(config_dict.get(f"beam{args.beam_id}", {}).get(f"{args.phasemask}", {}).get("ctrl_model", None).get("norm_pupil", None) ).astype(float)
    
    # used to normalize exterior and bad pixels in N0 (calculation of N0i)
    inside_edge_filt = np.array(config_dict.get(f"beam{args.beam_id}", {}).get(f"{args.phasemask}", {}).get("ctrl_model", None).get("inner_pupil_filt", None) )#.astype(bool)
    
    # reduction products
    IM_cam_config = config_dict.get(f"beam{args.beam_id}", {}).get(f"{args.phasemask}", {}).get("ctrl_model", None).get("camera_config", None) # dictionary
    
    bad_pixel_mask = np.array( config_dict.get(f"beam{args.beam_id}", {}).get(f"{args.phasemask}", {}).get("ctrl_model", None).get("bad_pixel_mask", None) )#.astype(bool)
    bias = np.array( config_dict.get(f"beam{args.beam_id}", {}).get(f"{args.phasemask}", {}).get("ctrl_model", None).get("bias", None) ).astype(float)
    dark = np.array( config_dict.get(f"beam{args.beam_id}", {}).get(f"{args.phasemask}", {}).get("ctrl_model", None).get("dark", None) ).astype(float)

    dm_flat = np.array( config_dict.get(f"beam{args.beam_id}", {}).get(f"{args.phasemask}", {}).get("ctrl_model", None).get("DM_flat", None) )

    strehl_coe_sec = np.array( config_dict.get(f"beam{args.beam_id}", {}).get("strehl_model", None).get("secondary",None) ).astype(float) #coe, intercept (on central secondary pixel, ADU/s/gain)
    strehl_coe_ext = np.array( config_dict.get(f"beam{args.beam_id}", {}).get("strehl_model", None).get("exterior",None) ).astype(float)#coe, intercept (on median signal, ADU/s/gain)

    cam_config = config_dict.get(f"beam{args.beam_id}", {}).get(f"{args.phasemask}", {}).get("ctrl_model", None).get("camera_config", None)


# ===================================
# =========== DM 
# ===================================
dm = dmclass( beam_id=args.beam_id )

dm.zero_all()
dm.activate_calibrated_flat()
if dm_flat == 'baldr':
    dm.activate_calibrated_flat()
elif dm_flat == 'factory':
    dm.activate_flat()
else:
   raise UserWarning("dm_flat must be baldr or factory")

# zernike basis for TT and some other modes (M2C_LO projects out HO, this should still be fine but lets keep it simple and clearly defined here)
LO_basis = dmbases.zer_bank(2, 5 )


# ===================================
# =========== Camera 
# ===================================
r1,r2,c1,c2 = baldr_pupils[f"{args.beam_id}"] # cropping regions - we do this after reading out full frame to facilitate potentially considering other beams 
c = FLI.fli(args.global_camera_shm, roi = [None,None,None,None]) #baldr_pupils[f'{args.beam_id}'])

# config prior to begin
mode0 = c.config["mode"] 
gain0 = float( c.config["gain"] ) 
fps0 = float( c.config["fps"] ) 
aduoffset0 = int( c.config["aduoffset"] ) 
 
# what we want to set up (same conditions as were used to build IM)
mode =  cam_config["mode"] 
gain = float( cam_config["gain"] ) 
fps = float( cam_config["fps"] ) 
aduoffset = int( cam_config["aduoffset"] ) 

c.send_fli_cmd(f"set mode {mode}")
time.sleep(2)
c.send_fli_cmd(f"set fps {fps}")
time.sleep(2)
c.send_fli_cmd(f"set gain {gain}")
time.sleep(2)
c.send_fli_cmd(f"set aduoffset {aduoffset}")

# ===================================
# =========== Normalization of matricies and vectors 
# ===================================

# Normalize control matricies by current gain and fps 
I2M_LO = gain / fps * I2M_LO_raw
I2M_HO = gain / fps * I2M_HO_raw 

# project reference intensities to DM (quicker for division & subtraction)
N0dm = gain / fps * (I2A @ N0i.reshape(-1)) # these are already reduced #- dark_dm - bias_dm
I0dm = gain / fps * (I2A @ I0.reshape(-1)) # these are already reduced  #- dark_dm - bias_dm
bias_dm = I2A @ bias.reshape(-1)
dark_dm = 1/fps * I2A @ dark.reshape(-1)
badpixmap = I2A @ bad_pixel_mask.astype(int).reshape(-1)

# reduction products on secondary pixels 
bias_sec = bias[secon_mask.astype(bool).reshape(-1)][4]
dark_sec = dark[secon_mask.astype(bool).reshape(-1)][4]

# ===================================
# =========== CHECK
# ===================================
im_list = [util.get_DM_command_in_2D(a) for a in [N0dm, I0dm, bias_dm, dark_dm, badpixmap]]
titles = ['clear pupil on DM','ZWFS pupil on DM','cam bias on DM', 'cam dark on DM', 'bad pixels on DM']

util.nice_heatmap_subplots( im_list = im_list ,title_list=titles, savefig='delme.png') 


# ===================================
# =========== DEFINE ABERRATIONS 
# ===================================
# each aberration has LO and HO component 
no_amp = 0.0 
small_amp = 0.02

noAb_probes = no_amp * LO_basis[0]
# LO_noAb_probes = [noAb_amp * LO_basis[0]]
# HO_noAb_probes = [noAb_amp * LO_basis[0]]

LO_smallAmp_probes = small_amp * LO_basis[:2]
HO_smallAmp_probes = small_amp *  np.array( [np.nan_to_num( util.get_DM_command_in_2D( aaa ), 0 ) for aaa in  np.eye(140) ] ) 

#util.nice_heatmap_subplots( im_list = TT_smallAmp_probes, savefig='delme.png')
#util.nice_heatmap_subplots( im_list = HO_smallAmp_probes[60:65], savefig='delme.png')

### IDEA : Potentially apply one open loop correction before hand 
LO_idx = 0
HO_idx = 65
probes = [noAb_probes, LO_smallAmp_probes[LO_idx], HO_smallAmp_probes[HO_idx] , LO_smallAmp_probes[LO_idx] + HO_smallAmp_probes[HO_idx]]
descr = ["no_aberration", f"small_LO_m{LO_idx}", f"small_HO_m{HO_idx}", f"small_LO_m{LO_idx}_and_HO_m{HO_idx}"]
disturb_line = [no_amp, small_amp, small_amp, small_amp]


# ===================================
# =========== PROBE AND GET DATA 
# ===================================
results = {}  
iterations = 500
for m, probe in enumerate( probes  ) : #noAb_probes ):
    print(f"\n\nstarting to probe {descr[m]} (idx = {m}) \n\n")
    results[m] = {"disturb":probe,"e_LO":[],"e_HO":[],"c_LO":[],"c_HO":[],"e_res_LO":[],"c_res_LO":[]}
    for it in range(iterations):

        if np.mod( it , 50)==0:
            print( f"probe {m} is {100 * round( it/iterations,3) }% complete" )

        # apply prob
        dm.set_data( probe ) 
        
        # get raw image 
        img = c.get_image(apply_manual_reduction=False)


        # project to DM 
        i_dm_raw = I2A @ img[r1:r2,c1:c2].reshape(-1) 
        
        # reduce in DM space 
        i_dm = i_dm_raw - dark_dm - bias_dm 

        # signal
        s =  ( i_dm - I0dm ) / (N0dm)   # 
        
        # project to mode 
        e_LO = I2M_LO @ s
        e_HO = I2M_HO @ s
        
        # project to command 
        c_LO = M2C_LO @ e_LO
        c_HO = M2C_HO @ e_HO 

        # calculate residual 
        e_res_LO = small_amp - e_LO
        
        c_res_LO = probe - c_LO.reshape(12,12)

        # could also look at cross coupling 
        results[m]["e_LO"].append( e_LO )
        results[m]["e_HO"].append( e_HO )
        results[m]["c_LO"].append( c_LO )
        results[m]["c_HO"].append( c_HO )
        results[m]["e_res_LO"].append( e_res_LO )
        results[m]["c_res_LO"].append( c_res_LO )

        time.sleep( 0.1 )

# flatten DM 
dm.zero_all()
dm.activate_calibrated_flat()



# ===================================
# =========== ANALYSIS
# ===================================

#worrying_actuators = {} 
residual_threshold = 0.003 # dm units 

for abmode, probe_descr in enumerate( descr ) :

    # Look at the histograms of TT error
    fs = 15
    kwargs = {'fontsize':fs}
    plt.figure(figsize=(8,8))
    means = []
    stds = []

    for lab, m in zip(["Tip","Tilt"],[0,1] ) :
        plt.hist( [ee[m] for ee in results[abmode]["e_LO"] ], bins = np.linspace( np.min(results[abmode]["e_LO"]), np.max(results[abmode]["e_LO"]), 100), label=lab, histtype='step',lw=1.4)
        means.append( np.mean( [ee[m] for ee in results[m]["e_LO"] ] ) )
        stds.append( np.std( [ee[m] for ee in results[m]["e_LO"] ] ) )

    ax = plt.gca()
    ax.tick_params(labelsize=fs)
    plt.axvline(disturb_line[abmode], ls=':', color='k', lw=2, label='Disturbance')

    # collect the lines you want to print
    lines = [
        f"tip mean : {means[0]:.3f}",
        f"tip std  : {stds[0]:.3f}",
        f"tilt mean: {means[1]:.3f}",
        f"tilt std : {stds[1]:.3f}",
    ]

    # start up near the top of the axes
    x0, y0 = 0.02, 0.98
    dy = 0.05    # vertical spacing in axes coords

    for i, txt in enumerate(lines):
        ax.text(x0, y0 - i*dy, txt,
                transform = ax.transAxes,
                fontsize  = fs,
                va='top', ha='left')

    plt.xlabel("Mode Errors [DM units]", fontsize=fs)
    plt.ylabel("Frequency",              fontsize=fs)
    plt.legend(fontsize=fs, loc='upper right')

    plt.tight_layout()
    plt.savefig(results_path + f"TT_err_{probe_descr}_hist.jpeg", 
                bbox_inches='tight', dpi=200)

    # Look at the map of mean signal and residuals in LO
    cc = np.array( results[abmode]["c_LO"] ).reshape(-1,12,12)
    dist = results[abmode]["disturb"]
    res = dist - cc 

    reco_LO_mean = np.mean( cc , axis=0).reshape(12,12)
    res_LO_mean = np.mean( res ,axis=0).reshape(12,12) 
    res_LO_std = np.std( res ,axis=0).reshape(12,12)

    im_list = [results[abmode]["disturb"], reco_LO_mean, res_LO_mean, res_LO_std]
    vlims = [[np.min(reco_LO_mean), np.max(reco_LO_mean)] for _  in im_list]
    titles = ["disturbance","LO reconstructor mean","LO residual mean","LO residual std"]
    cbar_labels = ["DM Units [0-1]","DM Units [0-1]","DM Units [0-1]","DM Units [0-1]"]
    util.nice_heatmap_subplots( im_list = im_list , title_list=titles, vlims=vlims,  cbar_label_list=cbar_labels,savefig=results_path + f'LO_cmd_residual_map_{probe_descr}.jpeg')


    # Look at the map of mean signal and residuals in HO
    cc = np.array( results[abmode]["c_HO"] ).reshape(-1,12,12)
    dist = results[abmode]["disturb"]
    res = dist - cc 
    

    reco_HO_mean = np.mean( cc , axis=0).reshape(12,12)
    res_HO_mean = np.mean( res ,axis=0).reshape(12,12) 
    res_HO_std = np.std( res ,axis=0).reshape(12,12)

    im_list = [results[abmode]["disturb"], reco_HO_mean, res_HO_mean, res_HO_std]
    vlims = [[np.min(reco_HO_mean), np.max(reco_HO_mean)] for _  in im_list]
    titles = ["disturbance","HO reconstructor mean","HO residual mean","HO residual std"]
    cbar_labels = ["DM Units [0-1]","DM Units [0-1]","DM Units [0-1]","DM Units [0-1]"]
    util.nice_heatmap_subplots( im_list = im_list , title_list=titles, vlims=vlims, cbar_label_list=cbar_labels,savefig=results_path + f'HO_cmd_residual_map_{probe_descr}.jpeg')

    # we get the actuators in HO modes where the residuals are above our specified threshold
    if abmode==0: # just look at noaberration case 
        worrying_actuators = np.where( abs( util.convert_12x12_to_140( res_HO_mean ) ) > residual_threshold )[0]

    print(f"saveing results here : {results_path}")
        


# look at cumulative for no aberrations 
abmode = 0 
cc = np.array( results[abmode]["c_HO"] ).reshape(-1,12,12)
dist = results[abmode]["disturb"]
res = dist - cc 
res_HO_mean = np.mean( res ,axis=0).reshape(12,12) 

plt.figure()
plt.ecdf(res_HO_mean.reshape(-1), label="CDF")
plt.yscale('log')
plt.gca().tick_params(labelsize=fs)
plt.xlabel('HO residual [DM Units]',fontsize=fs)
plt.ylabel("occurrence percentage",fontsize=fs)
plt.savefig(results_path + f'HO_cmd_residual_cumulative_prob_{descr[abmode]}.jpeg', bbox_inches ='tight')


# ===================================
# =========== ACTIONS 
# ===================================

### for "worrying" actuators in HO (mostly occur around edges)
# - set gains to zero in controller
# - filter from I2M_HO matrix
# - filter from M2C_HO matrix 

# pros and cons
# gains are more adapatible in real time - couold be monitored, prone mistakes if someone accidently sets something bad 
# matrix filtering is more permenet (although we can reconfig quiet easily) - avoids mistakes but not as adaptable 

#unique_worry_actuators = np.unique( [worrying_actuators[m] for m in worrying_actuators] )

if len( worrying_actuators ) > 20:
    print("SOMETHING VERY WRONG - TO MANY WORRYING ACTUATORS (HIGH RESIDUALS).. CHECK THE SIGNAL PROCESSING.")

# plot them 
worry_map = np.zeros(140)
worry_map[ worrying_actuators ] = 1 

util.nice_heatmap_subplots( [util.get_DM_command_in_2D( worry_map )], title_list=["worry HO actuators"], savefig='delme.png')


## FILTER M2C MATRIX 
M2C_HO_filtered = np.array( M2C_HO ).copy()
M2C_HO_filtered[:,worrying_actuators]  = 0

util.nice_heatmap_subplots( [M2C_HO_filtered ], title_list=["M2C_HO"], savefig='delme.png')


# # ===================================
# # =========== UPDATE CONFIG FILE 
# # ===================================
# input("about to update toml.. press enter to continue")

# # Update the toml file 
# dict2write = {f"beam{args.beam_id}":{
#     f"{args.phasemask}":{
#         "ctrl_model": {
#             "M2C_HO": M2C_HO_filtered
#             }
#         }
#     }
# }



# # Check if file exists; if so, load and update.
# if os.path.exists(args.toml_file.replace('#',f'{args.beam_id}')):
#     try:
#         current_data = toml.load(args.toml_file.replace('#',f'{args.beam_id}'))
#     except Exception as e:
#         print(f"Error loading TOML file: {e}")
#         current_data = {}
# else:
#     current_data = {}


# current_data = util.recursive_update(current_data, dict2write)

# with open(args.toml_file.replace('#',f'{args.beam_id}'), "w") as f:
#     toml.dump(current_data, f)

# print( f"updated configuration file {args.toml_file.replace('#',f'{args.beam_id}')}")





