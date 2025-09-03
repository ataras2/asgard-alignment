import numpy as np 
import matplotlib.pyplot as plt 
import os
from datetime import datetime
from astropy.io import fits

# local asgard-alignment package imports (may require correct enviornment)
import pyBaldr.utilities as util 
import common.DM_basis_functions as dmbases

# writing a FITS file to hold standard  basis configurations 
# for the BMC multi3.5 DM for Asgard/Baldr RTC 
# no piston in basis functions! 

Nmodes = 140
NX = NY = 12 # x,y size of DM surface (includes missing corners of BMC 3.5 DM)

basis_names = ['Zonal','Zernike','Hadamard', 'fourier_pinned_edges','fourier'] #, 'KL' ]
basis_keys = ['zonal','zernike', 'hadamard', 'fourier_10x10', 'fourier_12x12'] #, 'kl']
basis_map = {k:v for k,v in zip( basis_names ,basis_keys )}
res = {} # to hold results 
for bn in basis_names:
    print(f"doing basis {bn}")
    if 'zernike' in bn.strip():
        # use Frantz' zernike definition from xaosim
        basis_tmp = dmbases.zer_bank(2, Nmodes ) # Tip/tilt starts at index 2 here by convenction


    else:
        if 'fourier' in bn.strip() :
            
            basis_tmp0 = util.construct_command_basis( basis=bn, 
                                        number_of_modes = Nmodes, 
                                        Nx_act_DM = 12, 
                                        Nx_act_basis = 12, 
                                        act_offset=(0,0), 
                                        without_piston=True)

        else:
            basis_tmp0 = dmbases.construct_command_basis(basis=bn, 
                                            number_of_modes = Nmodes, 
                                            Nx_act_DM = 12, 
                                            Nx_act_basis = 12, 
                                            act_offset=(0,0), 
                                            without_piston=True)
            
        # convert to 12x12 format consistent with DM server expectations
        basis_tmp = np.array( [np.nan_to_num( util.get_DM_command_in_2D( b ),0) for b in basis_tmp0.T] )


    print( "   basis_tmp.shape = ", basis_tmp.shape )
    if basis_tmp.shape[0] > Nmodes:
        print(f"   trucating to {Nmodes} modes")
    
    # store results in dictionary (flatten 12x12 array and only keep 140)
    res[basis_map[bn]] = basis_tmp[:Nmodes] #.reshape( basis_tmp.shape[0], -1 )[:140] #basis_tmp[:140].

# Save it appropiate fits file in /usr/local/etc/baldr/dm_basis directory 

# ---- Save to a single FITS file with one IMAGE extension per basis (Option A) ----
out_dir = "/usr/local/etc/baldr/dm_basis"
os.makedirs(out_dir, exist_ok=True)  # NOTE: may require elevated permissions
out_path = os.path.join(out_dir, "bmc_multi3p5_dm_bases.fits")

primary_hdu = fits.PrimaryHDU()
ph = primary_hdu.header
ph["CREATOR"]  = "baldr_basis_builder"
ph["DATE"]     = datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ")
ph["NX"]       = NX
ph["NY"]       = NY
ph["DMGEOM"]   = "12x12_missing_corners"
ph["ORDERING"] = "row-major"
ph["NMODES"]   = Nmodes
ph["NOPISTON"] = True
ph["COMMENT"]  = "All basis cubes stored as (NX, NY, NMODES). Mode plane = [:, :, k]."

hdus = [primary_hdu]

for human_name, key in zip(basis_names, basis_keys):
    cube_modes = res[key]  # shape should be (Nmodes, NY, NX)
    if cube_modes.ndim != 3:
        raise ValueError(f"Basis '{key}' has unexpected ndim={cube_modes.ndim}; expected 3.")
    if cube_modes.shape[0] != Nmodes:
        raise ValueError(f"Basis '{key}' has {cube_modes.shape[0]} modes; expected {Nmodes}.")
    if cube_modes.shape[1:] != (NY, NX):
        raise ValueError(f"Basis '{key}' per-mode shape {cube_modes.shape[1:]} != ({NY}, {NX}).")

    # Reorder to (NX, NY, Nmodes) so each mode is a 2D plane along axis 3 (FITS NAXIS3)
    cube_f32 = np.asarray(cube_modes, dtype=np.float32)
    cube_f32 = np.transpose(cube_f32, (2, 1, 0))  # (Nmodes, NY, NX) -> (NX, NY, Nmodes)

    hdu = fits.ImageHDU(data=cube_f32, name=f"BASIS:{key}")
    hh = hdu.header
    hh["BASENAME"] = human_name
    hh["NMODES"]   = cube_f32.shape[2]
    hh["BUNIT"]    = "arb.u."
    hh["ORDERING"] = "row-major"
    hh["NOPISTON"] = True
    hdus.append(hdu)

fits.HDUList(hdus).writeto(out_path, overwrite=True, checksum=True)
print(f"Wrote FITS with {len(hdus)-1} basis extensions to: {out_path}")
# Quick usage note:
#   with fits.open(out_path, memmap=True) as hdul:
#       zern = hdul["BASIS:zernike"].data  # shape = (NX, NY, Nmodes)
#       mode_k = zern[:, :, k]             # 2-D plane for mode k

