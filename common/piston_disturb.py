#!/usr/bin/env python3

import numpy as np
from xaosim.wavefront import atmo_screen
from xaosim.shmlib import shm
import time
import argparse
from astropy.io import fits
import common.phasescreens as ps 
import pyBaldr.utilities as util 
import common.DM_basis_functions as dmbases
from asgard_alignment.DM_shm_ctrl import dmclass
import matplotlib.pyplot as plt
import atexit


parser = argparse.ArgumentParser(description="Applying turbulence to ASGARD BMC multi-3.5 DM.")

parser.add_argument(
    "--beam_id",
    type=lambda s: [int(item) for item in s.split(",")],
    default=[2], # 1, 2, 3, 4],
    help="Comma-separated beam IDs to apply. Default: 1,2,3,4"
)

parser.add_argument(
    '--number_of_iterations',
    type=int,
    default=1000,
    help="how many iterations do we run? %(default)s"
)


parser.add_argument(
    '--max_time',
    type=int,
    default=120,
    help="maximum time to run this for (in seconds)? %(default)s"
)


parser.add_argument(
    '--dist_type',
    type=str,
    default="vib",
    help="vib or 1onf. Default: %(default)s"
)

parser.add_argument(
    '--vib_freq',
    type=float,
    default=20,
    help="vibration frequency Hz. Default: %(default)s"
)

parser.add_argument(
    '--dm_rms',
    type=float,
    default=0.02,
    help="RMS of TT in DM units (0-1). Default: %(default)s"
)

parser.add_argument(
    '--dt',
    type=float,
    default=0.1,
    help="sample time (s). Default: %(default)s"
)

parser.add_argument(
    '--DM_chn',
    type=int,
    default=3,
    help="what channel on DM shared memory (0,1,2,3) to apply the turbulence?. Default: %(default)s"
)


parser.add_argument(
    '--record_telem',
    type=str,
    default=None,
    help="record telemetry? input directory/name.fits to save the fits file if you want,\
          Otherwise None to not record. if number of iterations is > 1e5 than we stop recording! \
          (this is around 200 MB) Default: %(default)s"
)




def plot2d( thing ):
    plt.figure()
    plt.imshow(thing)
    plt.colorbar()
    plt.savefig('/home/asg/Progs/repos/asgard-alignment/delme.png')
    plt.close()



####### START
args=parser.parse_args()

# ## HARD CODED FOR HEIMDALLR/BALDR 
# dm2opd = 7.0 # 7um / DM cmd in OPD (wavespace) for BMC multi3.5
# act_per_pupil = 10 # number of DM actuators across the pupil (Heimdallr/Baldr BMC multi3.5)
# Nx_act = 12  # number of actuators across DM diamerer (BMC multi3.5)
# corner_indicies = [0, 11, 11 * 12, -1] # DM corner indidices for BMC multi3.5 DM 
Nx_scrn = 32 # size of phasescreen that is binned to 12x12 DM cmd

# # CALCULATED VARIABLES 
# # time to pass pupil = D/V [s]
# # no pixel in pupil = Npix # non binned screen 
# # npn-binned pixels rolled per second = Npix / (D/V)  [s]
# # non-binned pixel scale = D/ Npix [m/pixel]
# # dt = (D/V) / Npix [s / pix]

# simulation temporal sampling 
# how many DM actuators we pass per iteration / how many act per pupil <- this is fraction of pupil passed per it
# than multiply this by pupil size in meters and divide by turb velocity (m/s)
dt = args.dt #1e-3 # (args.D_tel / args.V) / Nx_scrn

#print( f"simulation is updating at {dt*1e3}ms which is moving equivilantly {act_per_pupil / args.D_tel * args.V * dt} actuators / sec on the DM")
#er_it = act_per_pupil * args.D_tel / args.V  # We apply turbulence in DM space. So how many DM actuators do we want the screen to roll per iteration?
# r0_wvl = (args.r0)*(args.wvl/0.500)**(6/5) # coherence length at simulation wavelength
# m_remv = args.number_of_modes_removed # shorter <- how many Zernike modes first stage AO removes
modal_basis = dmbases.zer_bank(2, 3 )  # include piston!! 

# pupil on the DM
dm_pup = np.zeros( [12,12] ) # pupil on DM 
X,Y = np.meshgrid( np.arange(0,12), np.arange(0,12))
dm_pup[ (X-5.5)**2 + (Y-5.5)**2 <= 25] = 1
plot2d( dm_pup )



import itertools
import numpy as np

def make_TT_coe(no_its, rms):
    if args.dist_type == '1onf':
        # 1) build the finite sequence
        a = np.cumsum(np.random.randn(no_its))
        a = (a - np.mean(a)) / np.std(a)
        a *= rms
    elif args.dist_type == "vib":
        t = np.linspace(0,args.dt* no_its, no_its)
        a = args.dm_rms * np.sin( 2*np.pi * args.vib_freq * t)
    # 2) return an infinite generator looping over it
    else:
        raise UserWarning("not valid dist_type")
    
    return itertools.cycle(a)


# --- setup once, before your loop:
#no_its = 10000
# TT_disturb_rms = 0.02
# tip_rms  = np.sqrt(2) * TT_disturb_rms
# tilt_rms = np.sqrt(2) * TT_disturb_rms

#tip_stream  = make_TT_coe(args.number_of_iterations, args.dm_rms)
#tilt_stream = make_TT_coe(args.number_of_iterations, args.dm_rms)

piston_stream = make_TT_coe(args.number_of_iterations, args.dm_rms)



# OPEN DMs 
print( 'setting up DMs')
dms = {}
for beam in args.beam_id:
    dms[beam] = dmclass( beam_id=beam )
    # # zero all channels
    dms[beam].zero_all()
    # # activate flat (does this on channel 1)
    #dms[beam].activate_flat()
    dms[beam].activate_calibrated_flat()




## START 

# remember we stop recording telemetry (if requested) when cnt > 1e5
telem = {"t_dm":[], "dm_disturb":[]}
cnt = 0 
start_time = time.time()
time_elapsed = 0
while (not (cnt > args.number_of_iterations) ) and (not (time_elapsed > args.max_time)):
    
    t0 = time.time() #iteration start time
    # --- in your while loop:
    #print(cnt)

    pist_val = next(piston_stream)

    c_pist =  pist_val * np.ones_like( modal_basis[0] )
    
    cmd =  c_pist
    #send command on specified DM channel 
    for beam in args.beam_id:
        if np.std( cmd ) < 0.5:
            dms[beam].shms[args.DM_chn].set_data( cmd ) 
            dms[beam].shms[args.DM_chn].post_sems(1) # post it for the channel (this is important if we are recording telemetry on this channel in a sperate process)
            dms[beam].shm0.post_sems(1) # tell DM to update
        else: 
            raise UserWarning("DM is being driven pretty hard.. are you sure you got the amplitudes right? ")
    # if (args.record_telem is not None) and (cnt < 1e5):
    #     #print('to do - think about safety!')
    #     telem["t_dm"].append( time.time() )  # record specifically when we send the DM command 
    #     telem["dm_disturb"].append( cmd ) 

    t1 = time.time()  #iteration end time 

    if t1-t0 < dt : # wait until we reach the simulation temporal sampling 
        time.sleep( dt - (t1-t0))

    cnt += 1 
    time_elapsed = t1 - start_time


print("DONE - returning DM to flat and closing DM SHM")
for beam in args.beam_id:

    # # zero all channels
    dms[beam].zero_all()
    # # activate flat (does this on channel 1)
    #dms[beam].activate_flat()
    dms[beam].activate_calibrated_flat()
    dms[beam].close(erase_file=False)


# if args.record_telem  : 
#     print("now saving the telemetry file..")
#     # Create the TIME extension as a binary table.
#     #time_col = fits.Column(name='TIME', array=np.array(telem["t_dm"]), format='E')
#     hdu_time = fits.ImageHDU(data=np.array(telem["t_dm"])) 
#     hdu_time.header['EXTNAME'] = 'TIME'
#     hdu_time.header['units'] = 's'

#     # Create the DM_CMD extension as an image HDU from the datacube.
#     hdu_dm = fits.ImageHDU(data= np.array( telem["dm_disturb"] ) )
#     hdu_dm.header['EXTNAME'] = 'DM_CMD'

#     # Create a primary HDU (empty).
#     primary_hdu = fits.PrimaryHDU()

#     # Combine into an HDUList and write to a FITS file.
#     hdulist = fits.HDUList([primary_hdu, hdu_time, hdu_dm])
#     hdulist.writeto(args.record_telem, overwrite=True)

#     print(f"wrote telemetry to fits file: {args.record_telem}")



# def cleanup():
#     for beam in args.beam_id:
#         dms[beam] = dmclass( beam_id=beam )
#         # # zero all channels
#         dms[beam].zero_all()
#         # # activate flat (does this on channel 1)
#         #dms[beam].activate_flat()
#         dms[beam].activate_calibrated_flat()
#         dms[beam].close(erase_file=False)

# # Register the cleanup function to be called at exit.
# atexit.register(cleanup)

# ########### END 




# # print( np.mean( dt ))
# # print( np.std( dt ))
# # plt.figure() ; plt.hist(1e3 * np.array(dt), bins =np.logspace(-2, 2) ) ; 
# # plt.xlabel("iteration time for rolling phasescreen on DM [ms]" ) ; 
# # plt.ylabel("frequency" ) ; 
# # plt.xscale('log')
# # plt.savefig('/home/asg/Progs/repos/asgard-alignment/delme.png')

# # plt.imshow( scrn.scrn ) ;plt.savefig('/home/asg/Progs/repos/asgard-alignment/delme.png')




# ##### OLDER
# # dmc = shm(f"/dev/shm/dm{dmid}")

# # dmid = 1  # DM identifier (1 - 4)
# # chn = 3   # dm channel for turbulence

# # dmc = shm(f"/dev/shm/dm{dmid}")

# # dms = 12  # linear DM size (in actuators)
# # pdiam = 1.8  # telescope aperture diameter (in meters)
# # ntel = 50  # size of the phase screen (x pdiam in meters)

# # screen = atmo_screen(ntel * dms, ntel * pdiam, 0.2, 10.0, fc=6).real

# # wl = 1.65  # wavelength in the H-band (in microns)

# # opd = screen * wl / np.pi / 2

# # dx, dy = 1, 2  # phase screen drift

# # while True:
# #     # shift the OPD screen by said amount
# #     # apply OPD screen to the DM
# #     time.sleep(1)


