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
    "--LO",
    type=int,
    default=2,
    help="Up to what zernike order do we consider Low Order (LO). 2 is for tip/tilt, 3 would be tip,tilt,focus etc). Default: %(default)s"
)

parser.add_argument(
    "--inverse_method",
    type=str,
    default="zonal",
    help="Method used for inverting interaction matrix to build control (intensity-mode) matrix I2M"
)

parser.add_argument("--fig_path", 
                    type=str, 
                    default='~/Downloads/', 
                    help="path/to/output/image/ for the saved figures"
                    )



args=parser.parse_args()

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


#util.nice_heatmap_subplots( [ util.get_DM_command_in_2D(a) for a in [IM[65], IM[77] ]],savefig='delme.png')

# define out Tip/Tilt or lower order modes on zernike DM basis
LO = dmbases.zer_bank(2, args.LO +1 ) # 12x12 format

if args.inverse_method.lower() == 'pinv':
    I2M = np.linalg.pinv( IM )
    I2M_LO , I2M_HO = util.project_matrix( I2M , [util.convert_12x12_to_140(t) for t in LO] )

elif args.inverse_method.lower() == 'map': # minimum variance of maximum posterior estimator 
    phase_cov = np.eye( IM.shape[0] )
    noise_cov = np.eye( IM.shape[1] ) 
    I2M = (phase_cov @ IM @ np.linalg.inv(IM.T @ phase_cov @ IM + noise_cov) ).T #have to transpose to keep convention.. although should be other way round
    #I2M = phase_cov @ IM.T @ np.linalg.inv(IM @ phase_cov @ IM.T + noise_cov)
    I2M_LO , I2M_HO = util.project_matrix( I2M , [util.convert_12x12_to_140(t) for t in LO] )

elif args.inverse_method.lower() == 'zonal':
    # just literally filter weight the pupil and take inverse of the IM signal on diagonals (dm actuator registered pixels)
    dm_mask = I2A @ np.array( pupil_mask ).reshape(-1)

    I2M = np.diag(  np.array( [dm_mask[i]/IM[i][i] if np.isfinite(1/IM[i][i]) else 0 for i in range(len(IM))]) )
    I2M_LO , I2M_HO = util.project_matrix( I2M , [util.convert_12x12_to_140(t) for t in LO] )

elif 'svd_truncation' in args.inverse_method.lower() :
    k = int( args.inverse_method.split('truncation-')[-1] ) 
    U,S,Vt = np.linalg.svd( IM, full_matrices=True)

    I2M = util.truncated_pseudoinverse(U, S, Vt, k)
    I2M_LO , I2M_HO = util.project_matrix( I2M , [util.convert_12x12_to_140(t) for t in LO] )
else:
    raise UserWarning('no inverse method provided')



# TO DO : FIX M2C PROJECTION ====================
dict2write = {f"beam{args.beam_id}":{f"{args.phasemask}":{"ctrl_model": {
                                               "inverse_method": args.inverse_method,
                                               "controller_type":"PID",
                                               "sza": np.array(M2C).shape[0],
                                               "szm": np.array(M2C).shape[1],
                                               "szp": np.array(I2M).shape[1],
                                               "I2A": np.array(I2A).tolist(), 
                                               "I2M": np.array(I2M).tolist(),
                                               "I2M_LO": np.array(I2M_LO).tolist(),
                                               "I2M_HO": np.array(I2M_HO).tolist(),
                                               "M2C_LO" : np.array(M2C).tolist(),
                                               "M2C_HO" : np.array(M2C).tolist(),
                                               "I2rms_sec" : np.array(I2rms_sec).tolist(),
                                               "I2rms_ext" : np.array(I2rms_ext).tolist(),
                                               "LO": args.LO,
                                               "telemetry" : 0,  # do we record telem  - need to add to C++ readin
                                               "auto_close" : 0, # automatically close - need to add to C++ readin
                                               "auto_open" : 1, # automatically open - need to add to C++ readin
                                               "auto_tune" : 0, # automatically tune gains  - need to add to C++ readin
                                               "close_on_strehl_limit": 10,
                                               "open_on_strehl_limit": 0,
                                               "open_on_flux_limit": 0,
                                               "open_on_dm_limit"  : 0.5,
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

# I2M_1 = np.linalg.pinv( IM )

# phase_cov = np.eye( IM.shape[0] )
# noise_cov = 10 * np.eye( IM.shape[1] )
# I2M_2 = (phase_cov @ IM @ np.linalg.inv(IM.T @ phase_cov @ IM + noise_cov) ).T #have to transpose to keep convention.. although should be other way round

# dm_mask = I2A @ np.array( pupil_mask ).reshape(-1)
# I2M_3 = np.diag(  np.array( [dm_mask[i]/IM[i][i] if np.isfinite(1/IM[i][i]) else 0 for i in range(len(IM))]) )

# U,S,Vt = np.linalg.svd( IM, full_matrices=True)

# k= 20 # int( 5**2 * np.pi)
# I2M_4 = util.truncated_pseudoinverse(U, S, Vt, k=50)

# act = 65
# im_list = [util.get_DM_command_in_2D( a) for a in [IM[act], I2M_1@IM[act], I2M_2@IM[act], I2M_3@IM[act], I2M_4@IM[act] ] ]
# titles = ["real resp.", "pinv", "MAP", "zonal", f"svd trunc. (k={k})"]

# util.nice_heatmap_subplots(  im_list , title_list=titles, savefig='delme.png' ) 


# ## TT projection HO / TT 

# TT = dmbases.zer_bank(2, 3)
# util.nice_heatmap_subplots( im_list= [TT[0],TT[1]], savefig='delme.png' ) 

# sig = dm_mask * ( IM[act] - 0.3*convert_12x12_to_140(TT[0]) - 0.1*convert_12x12_to_140(TT[1]))

# im_list =  [util.get_DM_command_in_2D( sig )]
# im_TT_list = [util.get_DM_command_in_2D( sig )]
# im_HO_list = [util.get_DM_command_in_2D( sig )]

# for I2M in [I2M_1,I2M_2,I2M_3,I2M_4]:

#     I2M_TT , I2M_HO = util.project_matrix( I2M , [convert_12x12_to_140(t) for t in TT] )

#     im_list.append( util.get_DM_command_in_2D(  I2M  @ sig ) )
#     im_TT_list.append( util.get_DM_command_in_2D(  I2M_TT @ sig ) )
#     im_HO_list.append( util.get_DM_command_in_2D(  I2M_HO @ sig ) )

# util.nice_heatmap_subplots(  im_list , title_list=titles, savefig='delme.png' ) 

# util.nice_heatmap_subplots(  im_TT_list , title_list=["TT reco "+t for t in titles], savefig='delme.png' ) 

# util.nice_heatmap_subplots(  im_HO_list , title_list=["HO reco "+t for t in titles], savefig='delme.png' ) 