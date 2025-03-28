import subprocess
from astropy.io import fits 
import toml
import numpy as np 
import matplotlib.pyplot as plt 
import time 
import os 
import argparse
from asgard_alignment import FLI_Cameras as FLI

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
c_dict[beam_id].send_fli_cmd(f"set fps {args.cam_gain}")

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
for r0 in [0.08, 0.2, 0.5, 1]:
    imgs[r0] = []
    timestamps[r0] = []

    fname = f'/home/asg/Videos/test.fits'
    cmd = [
        'python', '/home/asg/Progs/repos/asgard-alignment/playground/baldr_CL/turbulence.py',
        '--beam_id', f'{beam_id}',
        '--number_of_iterations', '20',
        '--wvl', '1.65',
        '--D_tel', '1.8',
        '--r0', f'{r0}',
        '--V', '0.2',
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

    # look at median signal (ADU/s/gain) in pupil exterior pixels 
    exter_int[r0] = np.array( [args.cam_fps/args.cam_gain * np.median( i[secon_mask.astype(bool)] ) for i in img_list])

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

        dm_rms[r0] = np.array( [np.std( dm_pup * c ) for c in d["DM_CMD"].data ] )
        dm_p2v[r0] = np.array( [np.max( dm_pup * c ) - np.min( dm_pup * c ) for c in d["DM_CMD"].data ] ) 

    print( f"for r0 = {r0} , dm rms = {np.mean( dm_rms[r0] )}")
    print( f"for r0 = {r0} , dm p2v = {np.mean( dm_p2v[r0] )}")




rms_mean = np.array( [np.mean( v ) for _,v in dm_rms.items() ] )  
sec_mean = np.array( [np.mean( v ) for _,v in secon_int.items() ] )  