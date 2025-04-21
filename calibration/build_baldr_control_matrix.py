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

import common.DM_basis_functions as dmbases
import pyBaldr.utilities as util 


"""
Here we put together the final control config to be read in by RTC

- invert the interaction matrix by chosen meethod
- project to LO/HO matricies as desired
- write to ctrl key currently calibrated I2A in toml file 
- write to ctrl key currently calibrated strehl modes in toml file
- write to ctrl key desired shapes and states of the control system (default values) 

this is a large non-=human readable toml, in the future could put large matricies to fits files
and just keep the paths here. For now, for simplicity, I like EVERYTHING needed to configure
the RTC in one spot. Right here. 

"""


default_toml = os.path.join("config_files", "baldr_config_#.toml") 


parser = argparse.ArgumentParser(description="build control model")

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
    "--inverse_method_LO",
    type=str,
    default="pinv",
    help="Method used for inverting interaction matrix for LO to build control (intensity-mode) matrix I2M"
)


parser.add_argument(
    "--inverse_method_HO",
    type=str,
    default="zonal",
    help="Method used for inverting interaction matrix for HO to build control (intensity-mode) matrix I2M"
)


parser.add_argument("--dont_project_TT_out_HO",
                    dest="project_TT_out_HO",
                    action="store_false",
                    help="Disable projecting TT (or what ever lower order LO is defined as) out of HO (default: enabled)")

## NEED TO CHECK THIS AGAIN - BUG
parser.add_argument("--filter_edge_actuators",
                    dest="filter_edge_actuators",
                    action="store_false",
                    help="Filter actuators that interpolate from edge pixels (default: enabled)")


parser.add_argument("--fig_path", 
                    type=str, 
                    default='~/Downloads/', 
                    help="path/to/output/image/ for the saved figures"
                    )



args=parser.parse_args()

print( "filter_edge_actuators = ",args.filter_edge_actuators)

with open(args.toml_file.replace('#',f'{args.beam_id}'), "r") as f:

    config_dict = toml.load(f)
    
    # Baldr pupils from global frame 
    # baldr_pupils = config_dict['baldr_pupils']
    # I2A = np.array( config_dict[f'beam{beam_id}']['I2A'] )
    
    # # image pixel filters
    # pupil_mask = np.array(config_dict.get(f"beam{beam_id}", {}).get("pupil_mask", {}).get("mask", None) ).astype(bool)   # matrix bool
    # exter_mask = np.array(config_dict.get(f"beam{beam_id}", {}).get("pupil_mask", {}).get("exterior", None) ).astype(bool) # matrix bool
    # secon_mask = np.array(config_dict.get(f"beam{beam_id}", {}).get("pupil_mask", {}).get("secondary", None) ).astype(bool) # matrix bool

    #  read in the current calibrated matricies 
    pupil_mask = np.array(config_dict.get(f"beam{args.beam_id}", {}).get("pupil_mask", {}).get("mask", None) ).astype(bool)   # matrix bool
    I2A = np.array( config_dict[f'beam{args.beam_id}']['I2A'] )
    IM = np.array(config_dict.get(f"beam{args.beam_id}", {}).get(f"{args.phasemask}", {}).get("ctrl_model", None).get("IM", None) ).astype(float)
    M2C = np.array(config_dict.get(f"beam{args.beam_id}", {}).get(f"{args.phasemask}", {}).get("ctrl_model", None).get("M2C", None) ).astype(float)

    # also the current calibrated strehl modes 
    I2rms_sec = np.array(config_dict.get(f"beam{args.beam_id}", {}).get(f"strehl_model", {}).get(f"{args.phasemask}", {}).get("secondary", None)).astype(float)
    I2rms_ext = np.array(config_dict.get(f"beam{args.beam_id}", {}).get(f"strehl_model", {}).get(f"{args.phasemask}", {}).get("exterior", None)).astype(float)
    # # define our Tip/Tilt or lower order mode index on zernike DM basis 
    LO = config_dict.get(f"beam{args.beam_id}", {}).get(f"{args.phasemask}", {}).get("ctrl_model", None).get("LO", None)

    # tight (non-edge) pupil filter
    inside_edge_filt = np.array(config_dict.get(f"beam{args.beam_id}", {}).get(f"{args.phasemask}", {}).get("ctrl_model", None).get("inner_pupil_filt", None) )#.astype(bool)
    
    # these are just for testing things 
    poke_amp = config_dict.get(f"beam{args.beam_id}", {}).get(f"{args.phasemask}", {}).get("ctrl_model", None).get("poke_amp", None)
    camera_config = config_dict.get(f"beam{args.beam_id}", {}).get(f"{args.phasemask}", {}).get("ctrl_model", None).get("camera_config", None)
#util.nice_heatmap_subplots( [ util.get_DM_command_in_2D(a) for a in [IM[65], IM[77] ]],savefig='delme.png')

# define out Tip/Tilt or lower order modes on zernike DM basis
#LO = dmbases.zer_bank(2, LO +1 ) # 12x12 format

IM_LO = IM[:LO]
IM_HO = IM[LO:]


########################################
## LO MODES 
########################################
if args.inverse_method_LO.lower() == 'pinv':
    #I2M = np.linalg.pinv( IM )
    #I2M_LO , I2M_HO = util.project_matrix( I2M , [util.convert_12x12_to_140(t) for t in LO] )
    I2M_LO = np.linalg.pinv( IM_LO ) 

elif args.inverse_method_LO.lower() == 'map': # minimum variance of maximum posterior estimator 
    #phase_cov = np.eye( IM.shape[0] )
    #noise_cov = np.eye( IM.shape[1] ) 
    #I2M = (phase_cov @ IM @ np.linalg.inv(IM.T @ phase_cov @ IM + noise_cov) ).T #have to transpose to keep convention.. although should be other way round
    #I2M = phase_cov @ IM.T @ np.linalg.inv(IM @ phase_cov @ IM.T + noise_cov)
    #I2M_LO , I2M_HO = util.project_matrix( I2M , [util.convert_12x12_to_140(t) for t in LO] )
    phase_cov_LO = np.eye( IM_LO.shape[0] )
    noise_cov_LO = np.eye( IM_LO.shape[1] ) 

    I2M_LO = phase_cov_LO @ IM_LO.T @ np.linalg.inv(IM_LO @ phase_cov_LO @ IM_LO.T + noise_cov_LO)



elif 'svd_truncation' in args.inverse_method_LO.lower() :
    k = int( args.inverse_method.split('truncation-')[-1] ) 

    U,S,Vt = np.linalg.svd( IM_LO, full_matrices=True)

    I2M_LO = util.truncated_pseudoinverse(U, S, Vt, k)
    #I2M_LO , I2M_HO = util.project_matrix( I2M , [util.convert_12x12_to_140(t) for t in LO] )
else:
    raise UserWarning('no inverse method provided for LO')

########################################
## HO MODES 
########################################
if args.inverse_method_HO.lower() == 'pinv':
    #I2M = np.linalg.pinv( IM )
    #I2M_LO , I2M_HO = util.project_matrix( I2M , [util.convert_12x12_to_140(t) for t in LO] )
    I2M_HO = np.linalg.pinv( IM_HO )
    

elif args.inverse_method_HO.lower() == 'map': # minimum variance of maximum posterior estimator 
    #phase_cov = np.eye( IM.shape[0] )
    #noise_cov = np.eye( IM.shape[1] ) 
    #I2M = (phase_cov @ IM @ np.linalg.inv(IM.T @ phase_cov @ IM + noise_cov) ).T #have to transpose to keep convention.. although should be other way round
    #I2M = phase_cov @ IM.T @ np.linalg.inv(IM @ phase_cov @ IM.T + noise_cov)
    #I2M_LO , I2M_HO = util.project_matrix( I2M , [util.convert_12x12_to_140(t) for t in LO] )
    phase_cov_HO = np.eye( IM_HO.shape[0] )
    noise_cov_HO = np.eye( IM_HO.shape[1] ) 

    I2M_HO = phase_cov_HO @ IM_HO.T @ np.linalg.inv(IM_HO @ phase_cov_HO @ IM_HO.T + noise_cov_HO)

elif args.inverse_method_HO.lower() == 'zonal':
    # just literally filter weight the pupil and take inverse of the IM signal on diagonals (dm actuator registered pixels)
    if args.filter_edge_actuators: # do this in the mode space! 
        dm_mask =I2A @ np.array( inside_edge_filt ).reshape(-1) # I2A @ inside_edge_filt )
    else:
        dm_mask = I2A @ np.array( pupil_mask ).reshape(-1)

    # util.nice_heatmap_subplots(  im_list = [util.get_DM_command_in_2D(dm_mask)], savefig='delme.png' )
    I2M_HO = np.diag(  np.array( [dm_mask[i]/IM_HO[i][i] if np.isfinite(1/IM_HO[i][i]) else 0 for i in range(len(IM_HO))]) )


elif 'svd_truncation' in args.inverse_method_HO.lower() :
    k = int( args.inverse_method.split('truncation-')[-1] ) 
    U,S,Vt = np.linalg.svd( IM_HO, full_matrices=True)

    I2M_HO = util.truncated_pseudoinverse(U, S, Vt, k)

    #I2M_LO , I2M_HO = util.project_matrix( I2M , [util.convert_12x12_to_140(t) for t in LO] )
else:
    raise UserWarning('no inverse method provided for HO')




## NED TO CHECK THIS AGAIN - BUG, do this in the ZONAL scope of building I2M_HO
# if args.filter_edge_actuators:
#     # tight mask to restrict edge actuators 
#     dm_mask_144 = np.nan_to_num( util.get_DM_command_in_2D( I2A @ np.array([int(a) for a in inside_edge_filt]) ) ).reshape(-1)
#     # typically 44 actuators 
# else:
#     # puypil mask
#     dm_mask_144 = np.nan_to_num( util.get_DM_command_in_2D( I2A @ np.array( pupil_mask ).reshape(-1) ) ).reshape(-1)
#     # typically 71 actuators 

# filter out exterior actuators in command space (from pupol) - redudant if (args.filter_edge_actuators: # do this in the mode space!)
dm_mask_144 = np.nan_to_num( util.get_DM_command_in_2D( I2A @ np.array( pupil_mask ).reshape(-1) ) ).reshape(-1)

#util.nice_heatmap_subplots( [dm_mask_144.reshape(12,12),dm_tight_mask_144.reshape(12,12)], savefig='delme.png')

# project out in command / mode space 
if args.project_TT_out_HO:
    print("projecting TT out of HO")
    #we only need HO and require len 144x 140 (SHM input x number of actuatorss) which projects out the TT 
    _ , M2C_HO = util.project_matrix( np.nan_to_num( M2C[:,LO:], 0),  (dm_mask_144 * np.nan_to_num(M2C.T[:LO],0) ).reshape(-1,144) )
    #_ , M2C_HO = util.project_matrix( np.nan_to_num( M2C[:,LO:], 0),  np.nan_to_num(M2C[:,:LO],0).reshape(-1,144) )
    M2C_LO , _ = util.project_matrix( np.nan_to_num( M2C[:,:LO], 0),  np.nan_to_num(M2C.T[LO:],0).reshape(-1,144) )
else:
    M2C_LO = M2C[:,:LO]
    M2C_HO = M2C[:,LO:]


# TO DO : FIX M2C PROJECTION ====================
dict2write = {f"beam{args.beam_id}":{f"{args.phasemask}":{"ctrl_model": {
                                               "inverse_method_LO": args.inverse_method_LO,
                                               "inverse_method_HO": args.inverse_method_HO,
                                               "controller_type":"PID",
                                               "sza": np.array(M2C).shape[0],
                                               "szm": np.array(M2C).shape[1],
                                               "szp": np.array(I2M_HO).shape[1],
                                               "I2A": np.array(I2A).tolist(), 
                                               #"I2M": np.array(I2M).tolist(),
                                               "I2M_LO": np.array(I2M_LO.T).tolist(),
                                               "I2M_HO": np.array(I2M_HO.T).tolist(),
                                               "M2C_LO" : np.array(M2C_LO).tolist(),
                                               "M2C_HO" : np.array(M2C_HO).tolist(),
                                               "I2rms_sec" : np.array(I2rms_sec).tolist(),
                                               "I2rms_ext" : np.array(I2rms_ext).tolist(),
                                               "telemetry" : 0,  # do we record telem  - need to add to C++ readin
                                               "auto_close" : 0, # automatically close - need to add to C++ readin
                                               "auto_open" : 1, # automatically open - need to add to C++ readin
                                               "auto_tune" : 0, # automatically tune gains  - need to add to C++ readin
                                               "close_on_strehl_limit": 10,
                                               "open_on_strehl_limit": 0,
                                               "open_on_flux_limit": 0,
                                               "open_on_dm_limit"  : 0.3,
                                               "LO_offload_limit"  : 1,

                                               }
                                            }
                                        }
                                    }



# Check if file exists; if so, load and update.
if os.path.exists(args.toml_file.replace('#',f'{args.beam_id}')):
    try:
        current_data = toml.load(args.toml_file.replace('#',f'{args.beam_id}'))
    except Exception as e:
        print(f"Error loading TOML file: {e}")
        current_data = {}
else:
    current_data = {}


current_data = util.recursive_update(current_data, dict2write)

with open(args.toml_file.replace('#',f'{args.beam_id}'), "w") as f:
    toml.dump(current_data, f)

print( f"updated configuration file {args.toml_file.replace('#',f'{args.beam_id}')}")



# # #### SOME TESTS FOR THE CURIOUS


# # Perform SVD
# U, S, Vt = np.linalg.svd(IM_HO, full_matrices=False)  # shapes: (M, M), (min(M,N),), (min(M,N), N)

# # (a) Plot singular values
# plt.figure(figsize=(6, 4))
# plt.semilogy(S, 'o-')
# plt.title("Singular Values of IM_HO")
# plt.xlabel("Index")
# plt.ylabel("Singular value (log scale)")
# plt.grid(True)
# plt.tight_layout()
# plt.savefig('delme.png')

# # (b) Intensity eigenmodes (Vt)
# plt.figure(figsize=(15, 3))
# for i in range(min(5, Vt.shape[0])):
#     ax = plt.subplot(1, 5, i+1)
#     im = ax.imshow(util.get_DM_command_in_2D(Vt[i]), cmap='viridis')
#     ax.set_title(f"Vt[{i}]")
#     plt.colorbar(im, ax=ax)
# plt.suptitle("First 5 intensity eigenmodes (Vt) mapped to 2D")
# plt.tight_layout()
# plt.savefig('delme.png')


# # (c) System eigenmodes (U)
# plt.figure(figsize=(15, 3))
# for i in range(min(5, U.shape[1])):
#     ax = plt.subplot(1, 5, i+1)
#     im = ax.imshow(util.get_DM_command_in_2D(U[:, i]), cmap='plasma')
#     ax.set_title(f"U[:, {i}]")
#     plt.colorbar(im, ax=ax)
# plt.suptitle("First 5 system eigenmodes (U) mapped to 2D")
# plt.tight_layout()
# plt.savefig('delme.png')

# plt.close()

# # look at reconstructors HO 
# I2M_1 = np.linalg.pinv( IM_HO )

# phase_cov = np.eye( IM_HO.shape[0] )
# noise_cov = 10 * np.eye( IM_HO.shape[1] )
# I2M_2 = (phase_cov @ IM_HO @ np.linalg.inv(IM_HO.T @ phase_cov @ IM_HO + noise_cov) ).T #have to transpose to keep convention.. although should be other way round

# dm_mask = I2A @ np.array( pupil_mask ).reshape(-1)
# I2M_3 = np.diag(  np.array( [dm_mask[i]/IM_HO[i][i] if np.isfinite(1/IM_HO[i][i]) else 0 for i in range(len(IM_HO))]) )

# U,S,Vt = np.linalg.svd( IM_HO, full_matrices=True)

# k= 20 # int( 5**2 * np.pi)
# I2M_4 = util.truncated_pseudoinverse(U, S, Vt, k=50)

# act = 65
# im_list = [util.get_DM_command_in_2D( a) for a in [IM[act], I2M_1@IM[act], I2M_2@IM[act], I2M_3@IM[act], I2M_4@IM[act] ] ]
# titles = ["real resp.", "pinv", "MAP", "zonal", f"svd trunc. (k={k})"]

# util.nice_heatmap_subplots(  im_list , title_list=titles, savefig='delme.png' ) 



# ## LO 
# # look at reconstructors HO 
# I2M_1 = np.linalg.pinv( IM_LO )

# phase_cov = np.eye( IM_LO.shape[0] )
# noise_cov = 10 * np.eye( IM_LO.shape[1] )
# I2M_2 = (phase_cov @ IM_LO @ np.linalg.inv(IM_LO.T @ phase_cov @ IM_LO + noise_cov) ).T #have to transpose to keep convention.. although should be other way round

# dm_mask = I2A @ np.array( pupil_mask ).reshape(-1)
# I2M_3 = np.diag(  np.array( [dm_mask[i]/IM_LO[i][i] if np.isfinite(1/IM_LO[i][i]) else 0 for i in range(len(IM_LO))]) )

# U,S,Vt = np.linalg.svd( IM_LO, full_matrices=True)

# k= 20 # int( 5**2 * np.pi)
# I2M_4 = util.truncated_pseudoinverse(U, S, Vt, k=50)

# act = 0
# im_list = [util.get_DM_command_in_2D( a) for a in [IM[act], I2M_1@IM[act], I2M_2@IM[act], I2M_3@IM[act], I2M_4@IM[act] ] ]
# titles = ["real resp.", "pinv", "MAP", "zonal", f"svd trunc. (k={k})"]

# util.nice_heatmap_subplots(  im_list , title_list=titles, savefig='delme.png' ) 




# # ## TT projection HO / TT 

# TT = dmbases.zer_bank(2, 3)
# util.nice_heatmap_subplots( im_list= [TT[0],TT[1]], savefig='delme.png' ) 

# sig = dm_mask * ( IM[act] - 0.3*util.convert_12x12_to_140(TT[0]) - 0.1*util.convert_12x12_to_140(TT[1]))

# im_list =  [util.get_DM_command_in_2D( sig )]
# im_TT_list = [util.get_DM_command_in_2D( sig )]
# im_HO_list = [util.get_DM_command_in_2D( sig )]

# for I2M in [I2M_1,I2M_2,I2M_3,I2M_4]:

#     I2M_TT , I2M_HO = util.project_matrix( I2M , [util.convert_12x12_to_140(t) for t in TT] )

#     im_list.append( util.get_DM_command_in_2D(  I2M  @ sig ) )
#     im_TT_list.append( util.get_DM_command_in_2D(  I2M_TT @ sig ) )
#     im_HO_list.append( util.get_DM_command_in_2D(  I2M_HO @ sig ) )

# util.nice_heatmap_subplots(  im_list , title_list=titles, savefig='delme.png' ) 

# # util.nice_heatmap_subplots(  im_TT_list , title_list=["TT reco "+t for t in titles], savefig='delme.png' ) 

# # util.nice_heatmap_subplots(  im_HO_list , title_list=["HO reco "+t for t in titles], savefig='delme.png' ) 



# #### ADDITIONAL PROJECTION TESTS

# ### TEST 
# c0 = 0*M2C.T[0]
# i = 0*IM[0]
# act_list = [0, 65, 43]
# for a in act_list:
#     c0 += poke_amp/2 * M2C.T[a] # original command

#     i +=  IM[a] #+ IM[65] # simulating intensity repsonse

# e_LO = 2 * float(camera_config['gain']) / float(camera_config['fps']) * I2M_LO.T @ i
# e_HO = 2 * float(camera_config['gain']) / float(camera_config['fps']) * I2M_HO.T @ i

# # without projection just using HO (which has full rank)
# c_HO = (M2C[:,LO:] @ e_HO).reshape(12,12)
# res = c_HO - c0.reshape(12,12,)
# im_list = [  c0.reshape(12,12), c_HO, dm_mask_144.reshape(12,12) * res]
# vlims = [[np.min(c0), np.max(c0)] for _ in im_list]
# title_list = [ "disturb",  "c_HO'", "res."]
# cbar_title_list = ["DM UNITS","DM UNITS", "DM UNITS"]
# util.nice_heatmap_subplots( im_list = im_list ,title_list=title_list, vlims = vlims, cbar_label_list=  cbar_title_list, savefig='delme.png')

# # proper projection 
# c_LOg = (M2C_LO @ e_LO).reshape(12,12)
# c_HOg = (M2C_HO @ e_HO).reshape(12,12)

# dcmdg = c_LOg + c_HOg

# resg = dcmdg - c0.reshape(12,12)

# im_list = [  c0.reshape(12,12), c_LOg, c_HOg, dcmdg, dm_mask_144.reshape(12,12) * resg]
# vlims = [[np.min(c0), np.max(c0)] for _ in im_list]
# title_list = [ "disturb", "c_LO", "c_HO'","c_LO + c_HO","res."]
# cbar_title_list = ["DM UNITS","DM UNITS", "DM UNITS","DM UNITS","DM UNITS"]
# util.nice_heatmap_subplots( im_list = im_list ,title_list=title_list, vlims=vlims, cbar_label_list=  cbar_title_list, savefig='delme.png')

# print( np.std( dm_mask_144.reshape(12,12) * res ), np.std( dm_mask_144.reshape(12,12) * resg ))



# # In [8]: np.array(config_dict ["beam2"]["H3"]['ctrl_model']['M2C_HO']).shape
# # Out[8]: (144, 142)

# # In [9]: np.array(config_dict ["beam2"]["H3"]['ctrl_model']['M2C_LO']).shape
# # Out[9]: (144, 142)

# # In [10]: np.array(config_dict ["beam2"]["H3"]['ctrl_model']['I2M_LO']).shape
# # Out[10]: (2, 140)

# # In [11]: np.array(config_dict ["beam2"]["H3"]['ctrl_model']['I2M_HO']).shape
# # Out[11]: (140, 140)
