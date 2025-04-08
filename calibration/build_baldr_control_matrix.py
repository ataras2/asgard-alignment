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

def truncated_pseudoinverse(U, s, Vt, k):
    """
    Compute the pseudoinverse of a matrix using a truncated SVD.

    Parameters:
        U (np.ndarray): Left singular vectors (m x m if full_matrices=True)
        s (np.ndarray): Singular values (vector of length min(m,n))
        Vt (np.ndarray): Right singular vectors (n x n if full_matrices=True)
        k (int): Number of singular values/modes to keep.

    Returns:
        np.ndarray: The truncated pseudoinverse of the original matrix.
    """
    # Keep only the first k modes
    U_k = U[:, :k]      # shape: (m, k)
    s_k = s[:k]         # shape: (k,)
    Vt_k = Vt[:k, :]    # shape: (k, n)

    # Build the inverse of the diagonal matrix with the truncated singular values
    S_inv_k = np.diag(1.0 / s_k)  # shape: (k, k)

    # Compute the truncated pseudoinverse
    IM_trunc_inv = Vt_k.T @ S_inv_k @ U_k.T
    return IM_trunc_inv


def convert_12x12_to_140(arr):
    # Convert input to a NumPy array (if it isn't already)
    arr = np.asarray(arr)
    
    if arr.shape != (12, 12):
        raise ValueError("Input must be a 12x12 array.")
    
    # Flatten the array (row-major order)
    flat = arr.flatten()
    
    # The indices for the four corners in a 12x12 flattened array (row-major order):
    # Top-left: index 0
    # Top-right: index 11
    # Bottom-left: index 11*12 = 132
    # Bottom-right: index 143 (11*12 + 11)
    corner_indices = [0, 11, 132, 143]
    
    # Delete the corner elements from the flattened array
    vector = np.delete(flat, corner_indices)
    
    return vector


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

    #  
    pupil_mask = np.array(config_dict.get(f"beam{args.beam_id}", {}).get("pupil_mask", {}).get("mask", None) ).astype(bool)   # matrix bool
    I2A = np.array( config_dict[f'beam{args.beam_id}']['I2A'] )
    IM = np.array(config_dict.get(f"beam{args.beam_id}", {}).get(f"{args.phasemask}", {}).get("ctrl_model", None).get("IM", None) ).astype(float)
    M2C = np.array(config_dict.get(f"beam{args.beam_id}", {}).get(f"{args.phasemask}", {}).get("ctrl_model", None).get("M2C", None) ).astype(float)

    

# define out Tip/Tilt or lower order modes on zernike DM basis
LO = dmbases.zer_bank(2, args.LO +1 ) # 12x12 format

if args.inverse_method.lower() == 'pinv':
    I2M = np.linalg.pinv( IM )
    I2M_LO , I2M_HO = util.project_matrix( I2M , [convert_12x12_to_140(t) for t in LO] )

elif args.inverse_method.lower() == 'map': # minimum variance of maximum posterior estimator 
    phase_cov = np.eye( IM.shape[0] )
    noise_cov = np.eye( IM.shape[1] ) 
    I2M = (phase_cov @ IM @ np.linalg.inv(IM.T @ phase_cov @ IM + noise_cov) ).T #have to transpose to keep convention.. although should be other way round
    #I2M = phase_cov @ IM.T @ np.linalg.inv(IM @ phase_cov @ IM.T + noise_cov)
    I2M_LO , I2M_HO = util.project_matrix( I2M , [convert_12x12_to_140(t) for t in LO] )

elif args.inverse_method.lower() == 'zonal':
    # just literally filter weight the pupil and take inverse of the IM signal on diagonals (dm actuator registered pixels)
    dm_mask = I2A @ np.array( pupil_mask ).reshape(-1)

    I2M = np.diag(  np.array( [dm_mask[i]/IM[i][i] if np.isfinite(1/IM[i][i]) else 0 for i in range(len(IM))]) )
    I2M_LO , I2M_HO = util.project_matrix( I2M , [convert_12x12_to_140(t) for t in LO] )

elif 'svd_truncation' in args.inverse_method.lower() :
    k = int( args.inverse_method.split('truncation-')[-1] ) 
    U,S,Vt = np.linalg.svd( IM, full_matrices=True)

    I2M = truncated_pseudoinverse(U, S, Vt, k)
    I2M_LO , I2M_HO = util.project_matrix( I2M , [convert_12x12_to_140(t) for t in LO] )
else:
    raise UserWarning('no inverse method provided')


# TO DO : FIX M2C PROJECTION ====================
dict2write = {f"beam{args.beam_id}":{f"{args.phasemask}":{"ctrl_model": {
                                               "inverse_method": args.inverse_method,
                                               "I2M": I2M,
                                               "I2M_LO": I2M_LO,
                                               "I2M_HO": I2M_HO,
                                               "M2C_LO" : M2C,
                                               "M2C_HO" : M2C,
                                               "LO": args.LO,
                                               "telemetry" : 0,  # do we record telem
                                               "auto_close" : 0, # automatically close
                                               "auto_open" : 1, # automatically open
                                               "auto_tune" : 0, # automatically tune gains 
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

I2M_1 = np.linalg.pinv( IM )

phase_cov = np.eye( IM.shape[0] )
noise_cov = 10 * np.eye( IM.shape[1] )
I2M_2 = (phase_cov @ IM @ np.linalg.inv(IM.T @ phase_cov @ IM + noise_cov) ).T #have to transpose to keep convention.. although should be other way round

dm_mask = I2A @ np.array( pupil_mask ).reshape(-1)
I2M_3 = np.diag(  np.array( [dm_mask[i]/IM[i][i] if np.isfinite(1/IM[i][i]) else 0 for i in range(len(IM))]) )

U,S,Vt = np.linalg.svd( IM, full_matrices=True)

k= 20 # int( 5**2 * np.pi)
I2M_4 = truncated_pseudoinverse(U, S, Vt, k=50)

act = 65
im_list = [util.get_DM_command_in_2D( a) for a in [IM[act], I2M_1@IM[act], I2M_2@IM[act], I2M_3@IM[act], I2M_4@IM[act] ] ]
titles = ["real resp.", "pinv", "MAP", "zonal", f"svd trunc. (k={k})"]

util.nice_heatmap_subplots(  im_list , title_list=titles, savefig='delme.png' ) 


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