import subprocess
from astropy.io import fits 
import toml
import numpy as np 
import matplotlib.pyplot as plt 
import time 
import os 
import argparse
from asgard_alignment import FLI_Cameras as FLI
import pyBaldr.utilities as util

def plot2d( thing ):
    plt.figure()
    plt.imshow(thing)
    plt.colorbar()
    plt.savefig('/home/asg/Progs/repos/asgard-alignment/delme.png')
    plt.close()




default_toml = "/home/asg/Progs/repos/asgard-alignment/config_files/baldr_config_#.toml"
#os.path.join("config_files", "baldr_config_#.toml") 

parser = argparse.ArgumentParser(description="Building an intensity to Strehl model for Baldr ZWFS ")

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
    '--cam_fps',
    type=int,
    default=100,
    help="frames per second on camera. Default: %(default)s"
)

parser.add_argument(
    '--cam_gain',
    type=int,
    default=1,
    help="camera gain. Default: %(default)s"
)

parser.add_argument("--fig_path", 
                    type=str, 
                    default=None, 
                    help="path/to/output/image/ for the saved figures")

args=parser.parse_args()


## TO START - FILL OUT LATER / COPY/PASTE
beam_id = args.beam_id[0]
# move to phasemask 
# optimize alignment 



# Open Strehl proxy pixels 
with open(args.toml_file.replace('#',f'{beam_id}'), "r") as f:
    config_dict = toml.load(f)
    # Baldr pupils from global frame 
    baldr_pupils = config_dict['baldr_pupils']
    I2A = config_dict[f'beam{beam_id}']['I2A']
    
    pupil_mask = np.array(config_dict.get(f"beam{beam_id}", {}).get("pupil_mask", {}).get("mask", None) ) 
    exter_mask = np.array(config_dict.get(f"beam{beam_id}", {}).get("pupil_mask", {}).get("exterior", None) ) 
    secon_mask = np.array(config_dict.get(f"beam{beam_id}", {}).get("pupil_mask", {}).get("secondary", None) )



c_dict = {}
if 1 : #eventually for each beam id
    r1,r2,c1,c2 = baldr_pupils[f'{beam_id}'] 
    c_dict[beam_id] = FLI.fli(args.global_camera_shm, roi = [r1,r2,c1,c2])

c_dict[beam_id].send_fli_cmd(f"set fps {args.cam_fps}")
c_dict[beam_id].send_fli_cmd(f"set gain {args.cam_gain}")

# dark bias badpixels
if 1: 
    c_dict[beam_id].build_manual_bias(number_of_frames=500,
                                        sleeptime = 10)
    
    c_dict[beam_id].build_manual_dark(number_of_frames=500,
                                      build_bad_pixel_mask=True, 
                                      sleeptime = 10,
                                      kwargs={'std_threshold':10, 'mean_threshold':6} )



imgs = {} # keyed by r0 or rms however we want to do it 
timestamps = {} # keyed by r0 or rms however we want to do it 
secon_int = {}
exter_int = {}
dm_rms = {}
dm_p2v = {}
# for r0 in ...
#r0 = 1
for r0 in [0.3, 0.5, 0.7, 1.0]:
    imgs[r0] = []
    timestamps[r0] = []

    fname = f'/home/asg/Videos/test.fits'
    cmd = [
        'python', '/home/asg/Progs/repos/asgard-alignment/common/turbulence.py',
        '--beam_id', f'{beam_id}',
        '--number_of_iterations', '1000',
        '--wvl', '1.65',
        '--D_tel', '1.8',
        '--r0', f'{r0}',
        '--V', '7',
        '--number_of_modes_removed', '0',
        '--DM_chn', '3',
        '--record_telem', fname
    ]



    # Start the process non-blocking
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)

    # Flag to indicate if process is running
    process_running = True

    def check_process():
        global process_running
        # poll() returns None if process is still running
        if proc.poll() is None:
            process_running = True
        else:
            process_running = False

    time.sleep( 2 )
    # Main loop to get images while process running. 
    # we skip frames intentionally rto minimize correlations 
    while process_running:
        print("turbulence.py is still running...getting 100 frames")
        imgs[r0].append(  c_dict[beam_id].get_some_frames(number_of_frames = 100, apply_manual_reduction=True) )
        timestamps[r0].append( time.time() )
        time.sleep(0.5)  # Wait 1 second between checks
        check_process()

    # Once finished, capture the output.
    stdout, stderr = proc.communicate()
    print("turbulence.py finished.")
    print("Output:")
    print(stdout)
    if stderr:
        print("Errors:")
        print(stderr)



    # Run the command and capture the output.
    result = subprocess.run(cmd, capture_output=True, text=True)

    # Print the captured stdout and stderr.
    print("Output from turbulence.py:")
    print(result.stdout)
    if result.stderr:
        print("Errors:")
        print(result.stderr)



    # flatten the samples
    img_list = np.array( imgs[r0] ).reshape(-1, np.array( imgs[r0] ).shape[-2], np.array( imgs[r0] ).shape[-1] )

    # get the intensity in the secondary pixel 
    # secondary mask is 3x3 with center at secondary center (index 4! )
    # Normalized to ADU / s / gain!!! 
    secon_int[r0] = np.array( [args.cam_fps/args.cam_gain * i[secon_mask.astype(bool)][4] for i in img_list])

    # look at mean signal (ADU/s/gain) in pupil exterior pixels 
    exter_int[r0] = np.array( [args.cam_fps/args.cam_gain * np.mean( i[secon_mask.astype(bool)] ) for i in img_list])

    # pupil on the DM
    dm_pup = np.zeros( [12,12] ) # pupil on DM 
    X,Y = np.meshgrid( np.arange(0,12), np.arange(0,12))
    dm_pup[ (X-5.5)**2 + (Y-5.5)**2 <= 25] = 1
    # plot2d( dm_pup )


    # read in telemetry 
    rms = [] # rms DM cmd
    p2v = [] # peak-to-valley DM cmd 
    with fits.open( f"{fname}" ) as d:
        print(d.info())

        dm_rms[r0] = np.array( [np.std( c[dm_pup.astype(bool)] ) for c in d["DM_CMD"].data ] )
        dm_p2v[r0] = np.array( [np.max(  c[dm_pup.astype(bool)] ) - np.min( c[dm_pup.astype(bool)] ) for c in d["DM_CMD"].data ] ) 

    print( f"for r0 = {r0} , dm rms = {np.mean( dm_rms[r0] )}")
    print( f"for r0 = {r0} , dm p2v = {np.mean( dm_p2v[r0] )}")




rms_mean = np.array( [np.mean( v ) for _,v in dm_rms.items() ] ) # [2:]
sec_mean = np.array( [np.mean( v ) for _,v in secon_int.items() ] )# [2:]
ext_mean = np.array( [np.mean( v ) for _,v in exter_int.items() ] )# [2:] 

rms_std = np.array( [np.std( v ) for _,v in dm_rms.items() ] ) #[2:] 
sec_std = np.array( [np.std( v ) for _,v in secon_int.items() ] ) #[2:3]
ext_std = np.array( [np.std( v ) for _,v in exter_int.items() ] ) # [2:]

# Fit a linear model for sec vs. rms.
coef_sec = np.polyfit( sec_mean,rms_mean, 1)  # [slope, intercept]

# Fit a linear model for ext vs. rms.
coef_ext = np.polyfit( ext_mean,rms_mean,  1)

print("") 



dict2write = {f"beam{beam_id}":{"strehl_model": {"secondary":np.diag( coef_sec ), "exterior": np.diag( coef_ext )}}}

# Check if file exists; if so, load and update.
if os.path.exists(args.toml_file.replace('#',f'{beam_id}')):
    try:
        current_data = toml.load(args.toml_file.replace('#',f'{beam_id}'))
    except Exception as e:
        print(f"Error loading TOML file: {e}")
        current_data = {}
else:
    current_data = {}


current_data = util.recursive_update(current_data, dict2write)

with open(args.toml_file.replace('#',f'{beam_id}'), "w") as f:
    toml.dump(current_data, f)

print( f"updated configuration file {args.toml_file.replace('#',f'{beam_id}')}")


# Plot errorbars:
fs = 15
plt.figure(figsize=(8,5))
plt.errorbar(sec_mean, rms_mean, yerr=rms_std, xerr=sec_std, fmt='o', color='blue', label='Secondary pixel')
plt.errorbar(ext_mean, rms_mean,  yerr=rms_std, xerr=ext_std, fmt='s', color='red', label='Exterior pixels')

# use the models to plot lines
fit_ext = np.poly1d(coef_ext)
fit_sec = np.poly1d(coef_sec)

# Generate x-values for plotting the fitted lines.

xsec = np.linspace(sec_mean.min(), sec_mean.max(), 100)
xext = np.linspace(ext_mean.min(), ext_mean.max(), 100)
plt.plot(xsec, fit_sec(xsec), 'b--', label=f'Secondary pixel Fit: y={coef_sec[0]:.2e}x+{coef_sec[1]:.2e}')
plt.plot(xext, fit_ext(xext), 'r--', label=f'Exterior pixels Fit: y={coef_ext[0]:.2e}x+{coef_ext[1]:.2e}')
plt.gca().tick_params(labelsize=fs)
plt.ylabel('DM RMS [DM units]',fontsize=fs)
plt.xlabel('signal [ADU/s/gain/pixel]',fontsize=fs)
plt.legend()
#plt.xscale('log')
#plt.savefig("delme.png", bbox_inches='tight',dpi=200)
if args.fig_path is not None:
    plt.savefig(args.fig_path + f"strehl_model_beam{beam_id}.png", bbox_inches='tight',dpi=200)
plt.close()





# # coef_sec = np.polyfit(rms_mean, sec_mean, 1)  # [slope, intercept]
# # fit_sec = np.poly1d(coef_sec)

# # # Fit a linear model for ext vs. rms.
# # coef_ext = np.polyfit(rms_mean, ext_mean, 1)
# # fit_ext = np.poly1d(coef_ext)

# data = {
#     "DM_RMS_MEAN": rms_mean,
#     "SECONDARY_MEAN": sec_mean,
#     "EXTERIOR_MEAN": ext_mean,
#     "DM_RMS_STD":  rms_std,
#     "SECONDARY_STD":  sec_std,
#     "EXTERIOR_STD":  ext_std,
#     "SEC_MODEL_FIT" : coef_sec ,
#     "EXT_MODEL_FIT" : coef_ext ,
# }

# # Define units for each field (adjust these as needed)
# units = {
#     "DM_RMS_MEAN": "DM UNITS",
#     "SECONDARY_MEAN": "ADU/s/gain/pixel",
#     "EXTERIOR_MEAN": "ADU/s/gain/pixel",
#     "RMS_STD":  "ADU/s/gain/pixel",
#     "SECONDARY_MEAN":  "ADU/s/gain/pixel",
#     "EXTERIOR_MEAN":  "ADU/s/gain/pixel",
#     "SEC_MODEL_FIT": "linear",
#     "EXT_MODEL_FIT": "linear",
# }

# # Create a primary HDU (can be empty)
# primary_hdu = fits.PrimaryHDU()

# # Prepare a list of HDUs
# hdus = [primary_hdu]

# # For each key in our data dictionary, create a binary table HDU.
# for key, array in data.items():
#     # Create a column for the 1D array; using 'E' for 32-bit float format.
#     col = fits.Column(name=key, format='E', array=array)
#     hdu = fits.BinTableHDU.from_columns([col])
#     hdu.header['EXTNAME'] = key         # Set the extension name.
#     hdu.header['BUNIT'] = units.get(key, "")  # Set the unit.
#     hdus.append(hdu)

# # Create an HDUList and write the FITS file.
# hdulist = fits.HDUList(hdus)
# hdulist.writeto(args.fig_path + f"strehl_model_telem_beam_{beam_id}_phasemask{args.phasemask}.fits", overwrite=True)


