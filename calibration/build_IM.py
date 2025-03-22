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

from astropy.io import fits
from scipy.signal import TransferFunction, bode
from types import SimpleNamespace
import asgard_alignment.controllino as co # for turning on / off source \
from asgard_alignment.DM_shm_ctrl import dmclass
import common.DM_basis_functions as dmbases
import asgard_alignment.controllino as co
import common.phasemask_centering_tool as pct
import common.phasescreens as ps 
import pyBaldr.utilities as util 
import pyzelda.ztools as ztools
import datetime
from xaosim.shmlib import shm
from asgard_alignment import FLI_Cameras as FLI

MDS_port = 5555
MDS_host = 'localhost'
context = zmq.Context()
context.socket(zmq.REQ)
socket = context.socket(zmq.REQ)
socket.setsockopt(zmq.RCVTIMEO, 5000)
server_address = f"tcp://{MDS_host}:{MDS_port}"
socket.connect(server_address)
state_dict = {"message_history": [], "socket": socket}



# PID and leaky integrator copied from /Users/bencb/Documents/asgard-alignment/playground/open_loop_tests_HO.py
class PIDController:
    def __init__(self, kp=None, ki=None, kd=None, upper_limit=None, lower_limit=None, setpoint=None):
        if kp is None:
            kp = np.zeros(1)
        if ki is None:
            ki = np.zeros(1)
        if kd is None:
            kd = np.zeros(1)
        if lower_limit is None:
            lower_limit = np.zeros(1)
        if upper_limit is None:
            upper_limit = np.ones(1)
        if setpoint is None:
            setpoint = np.zeros(1)

        self.kp = np.array(kp)
        self.ki = np.array(ki)
        self.kd = np.array(kd)
        self.lower_limit = np.array(lower_limit)
        self.upper_limit = np.array(upper_limit)
        self.setpoint = np.array(setpoint)
        self.ctrl_type = 'PID'
        
        size = len(self.kp)
        self.output = np.zeros(size)
        self.integrals = np.zeros(size)
        self.prev_errors = np.zeros(size)

    def process(self, measured):
        measured = np.array(measured)
        size = len(self.setpoint)

        if len(measured) != size:
            raise ValueError(f"Input vector size must match setpoint size: {size}")

        # Check all vectors have the same size
        error_message = []
        for attr_name in ['kp', 'ki', 'kd', 'lower_limit', 'upper_limit']:
            if len(getattr(self, attr_name)) != size:
                error_message.append(attr_name)
        
        if error_message:
            raise ValueError(f"Input vectors of incorrect size: {' '.join(error_message)}")

        if len(self.integrals) != size:
            print("Reinitializing integrals, prev_errors, and output to zero with correct size.")
            self.integrals = np.zeros(size)
            self.prev_errors = np.zeros(size)
            self.output = np.zeros(size)

        for i in range(size):
            error = measured[i] - self.setpoint[i]  # same as rtc
            
            if self.ki[i] != 0: # ONLY INTEGRATE IF KI IS NONZERO!! 
                self.integrals[i] += error
                self.integrals[i] = np.clip(self.integrals[i], self.lower_limit[i], self.upper_limit[i])

            derivative = error - self.prev_errors[i]
            self.output[i] = (self.kp[i] * error +
                              self.ki[i] * self.integrals[i] +
                              self.kd[i] * derivative)
            self.prev_errors[i] = error

        return self.output

    def set_all_gains_to_zero(self):
        self.kp = np.zeros( len(self.kp ))
        self.ki = np.zeros( len(self.ki ))
        self.kd = np.zeros( len(self.kd ))
        
    def reset(self):
        self.integrals.fill(0.0)
        self.prev_errors.fill(0.0)
        self.output.fill(0.0)
        
    def get_transfer_function(self, mode_index=0):
        """
        Returns the transfer function for the specified mode index.

        Parameters:
        - mode_index: Index of the mode for which to get the transfer function (default is 0).
        
        Returns:
        - scipy.signal.TransferFunction: Transfer function object.
        """
        if mode_index >= len(self.kp):
            raise IndexError("Mode index out of range.")
        
        # Extract gains for the selected mode
        kp = self.kp[mode_index]
        ki = self.ki[mode_index]
        kd = self.kd[mode_index]
        
        # Numerator and denominator for the PID transfer function: G(s) = kp + ki/s + kd*s
        # Which can be expressed as G(s) = (kd*s^2 + kp*s + ki) / s
        num = [kd, kp, ki]  # coefficients of s^2, s, and constant term
        den = [1, 0]        # s term in the denominator for integral action
        
        return TransferFunction(num, den)

    def plot_bode(self, mode_index=0):
        """
        Plots the Bode plot for the transfer function of a specified mode.

        Parameters:
        - mode_index: Index of the mode for which to plot the Bode plot (default is 0).
        """
        # Get transfer function
        tf = self.get_transfer_function(mode_index)

        # Generate Bode plot data
        w, mag, phase = bode(tf)
        
        # Plot magnitude and phase
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 6))
        
        # Magnitude plot
        ax1.semilogx(w, mag)  # Bode magnitude plot
        ax1.set_title(f"Bode Plot for Mode {mode_index}")
        ax1.set_ylabel("Magnitude (dB)")
        ax1.grid(True, which="both", linestyle="--", linewidth=0.5)

        # Phase plot
        ax2.semilogx(w, phase)  # Bode phase plot
        ax2.set_xlabel("Frequency (rad/s)")
        ax2.set_ylabel("Phase (degrees)")
        ax2.grid(True, which="both", linestyle="--", linewidth=0.5)

        plt.tight_layout()
        plt.show()




def init_telem_dict(): 
    # i_list is intensity measured on the detector
    # i_dm_list is intensity interpolated onto DM actuators - it is used only in zonal_interp control methods 
    # s_list is processed intensity signal used in the control loop (e.g. I - I0)
    # e_* is control error signals 
    # u_* is control signals (e.g. after PID control)
    # c_* is DM command signals 
    telemetry_dict = {
        "time" : [],
        "i_list" : [],
        "i_dm_list":[], 
        "s_list" : [],
        "e_TT_list" : [],
        "u_TT_list" : [],
        "c_TT_list" : [], # the next TT cmd to send to ch2
        "e_HO_list" : [],
        "u_HO_list" : [], 
        "c_HO_list" : [], # the next H0 cmd to send to ch2 
        "current_dm_ch0" : [], # the current DM cmd on ch1
        "current_dm_ch1" : [], # the current DM cmd on ch2
        "current_dm_ch2" : [], # the current DM cmd on ch3
        "current_dm_ch3" : [], # the current DM cmd on ch4
        "current_dm":[], # the current DM cmd (sum of all channels)
        "modal_disturb_list":[],
        "dm_disturb_list" : []
        # "dm_disturb_list" : [],
        # "rmse_list" : [],
        # "flux_outside_pupil_list" : [],
        # "residual_list" : [],
        # "field_phase" : [],
        # "strehl": []
    }
    return telemetry_dict



### using SHM camera structure
def move_relative_and_get_image(cam, beam, baldr_pupils, phasemask, savefigName=None, use_multideviceserver=True,roi=[None,None,None,None]):
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
                if savefigName != None:
                    plt.figure()
                    plt.imshow( np.log10( img[roi[0]:roi[1],roi[2]:roi[3]] ) )
                    plt.colorbar()
                    plt.savefig(savefigName)
            except:
                print('incorrect input. Try input "1,1" as an example, or "e" to exit')

    plt.close()


def send_and_get_response(message):
    # st.write(f"Sending message to server: {message}")
    state_dict["message_history"].append(
        f":blue[Sending message to server: ] {message}\n"
    )
    state_dict["socket"].send_string(message)
    response = state_dict["socket"].recv_string()
    if "NACK" in response or "not connected" in response:
        colour = "red"
    else:
        colour = "green"
    # st.markdown(f":{colour}[Received response from server: ] {response}")
    state_dict["message_history"].append(
        f":{colour}[Received response from server: ] {response}\n"
    )

    return response.strip()


def setup(beam_ids, global_camera_shm, toml_file) :

    NNN = 10 # number of time get_data() called / appended

    print( 'setting up controllino and MDS ZMQ communication')

    controllino_port = '172.16.8.200'

    myco = co.Controllino(controllino_port)


    print( 'Reading in configurations') 

    I2A_dict = {}
    for beam_id in beam_ids:

        # read in TOML as dictionary for config 
        with open(toml_file.replace('#',f'{beam_id}'), "r") as f:
            config_dict = toml.load(f)
            # Baldr pupils from global frame 
            baldr_pupils = config_dict['baldr_pupils']
            I2A_dict[beam_id] = config_dict[f'beam{beam_id}']['I2A']


    # Set up global camera frame SHM 
    print('Setting up camera. You should manually set up camera settings before hand')
    c = shm(global_camera_shm)

    # set up DM SHMs 
    print( 'setting up DMs')
    dm_shm_dict = {}
    for beam_id in beam_ids:
        dm_shm_dict[beam_id] = dmclass( beam_id=beam_id )
        # zero all channels
        dm_shm_dict[beam_id].zero_all()
        # activate flat (does this on channel 1)
        dm_shm_dict[beam_id].activate_flat()
        # apply dm flat offset (does this on channel 2)
        #dm_shm_dict[beam_id].set_data( np.array( dm_flat_offsets[beam_id] ) )
    


    # Get Darks
    print( 'getting Darks')
    myco.turn_off("SBB")
    time.sleep(15)
    darks = []
    for _ in range(NNN):
        darks.append(  c.get_data() )

    darks = np.array( darks ).reshape(-1, darks[0].shape[1], darks[0].shape[2])

    myco.turn_on("SBB")
    time.sleep(10)

    # crop for each beam
    dark_dict = {}
    for beam_id in beam_ids:
        r1,r2,c1,c2 = baldr_pupils[f"{beam_id}"]
        cropped_imgs = [nn[r1:r2,c1:c2] for nn in darks]
        dark_dict[beam_id] = cropped_imgs


    # Get reference pupils (later this can just be a SHM address)
    zwfs_pupils = {}
    clear_pupils = {}
    rel_offset = 200.0 #um phasemask offset for clear pupil
    print( 'Moving FPM out to get clear pupils')
    for beam_id in beam_ids:
        message = f"moverel BMX{beam_id} {rel_offset}"
        res = send_and_get_response(message)
        print(res) 
        time.sleep( 1 )
        message = f"moverel BMY{beam_id} {rel_offset}"
        res = send_and_get_response(message)
        print(res) 
        time.sleep(10)


    #Clear Pupil
    print( 'gettin clear pupils')
    N0s = []
    for _ in range(NNN):
         N0s.append(  c.get_data() )
    N0s = np.array(  N0s ).reshape(-1,  N0s[0].shape[1],  N0s[0].shape[2])


    for beam_id in beam_ids:
        r1,r2,c1,c2 = baldr_pupils[f"{beam_id}"]
        cropped_imgs = [nn[r1:r2,c1:c2] for nn in N0s]
        clear_pupils[beam_id] = cropped_imgs

        # move back 
        print( 'Moving FPM back in beam.')
        message = f"moverel BMX{beam_id} {-rel_offset}"
        res = send_and_get_response(message)
        print(res) 
        time.sleep(1)
        message = f"moverel BMY{beam_id} {-rel_offset}"
        res = send_and_get_response(message)
        print(res) 
        time.sleep(10)


    # check the alignment is still ok 
    beam = int( input( "\ndo you want to check the phasemasks for a beam. Enter beam number (1,2,3,4) or 0 to continue\n") )
    while beam :
        save_tmp = 'delme.png'
        print(f'open {save_tmp } to see generated images after each iteration')
        
        move_relative_and_get_image(cam=c, beam=beam, baldr_pupils = baldr_pupils, phasemask=state_dict["socket"],  savefigName = save_tmp, use_multideviceserver=True )
        
        beam = int( input( "\ndo you want to check the phasemasks for a beam. Enter beam number (1,2,3,4) or 0 to continue\n") )
    

    # ZWFS Pupil
    print( 'Getting ZWFS pupils')
    I0s = []
    for _ in range(NNN):
        I0s.append(  c.get_data() )
    I0s = np.array(  I0s ).reshape(-1,  I0s[0].shape[1],  I0s[0].shape[2])

    for beam_id in beam_ids:
        r1,r2,c1,c2 = baldr_pupils[f"{beam_id}"]
        #cropped_img = interpolate_bad_pixels(img[r1:r2, c1:c2], bad_pixel_mask[r1:r2, c1:c2])
        cropped_img = [nn[r1:r2,c1:c2] for nn in I0s] #/np.mean(img[r1:r2, c1:c2][pupil_masks[bb]])
        zwfs_pupils[beam_id] = cropped_img

    return c, dm_shm_dict, dark_dict, zwfs_pupils, clear_pupils, baldr_pupils, I2A_dict


def process_signal( i, I0, N0):
    # must be same as model cal. import from common module
    # i is intensity, I0 reference intensity (zwfs in), N0 clear pupil (zwfs out)
    return ( i - I0 ) / N0 


parser = argparse.ArgumentParser(description="Interaction and control matricies.")

default_toml = os.path.join("config_files", "baldr_config_#.toml") 

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
    "--basis_name",
    type=str,
    default="zernike",
    help="basis used to build interaction matrix (IM)"
)

parser.add_argument(
    "--Nmodes",
    type=int,
    default=10,
    help="number of modes to probe"
)

parser.add_argument(
    "--poke_amp",
    type=float,
    default=0.04,
    help="amplitude to poke DM modes for building interaction matrix"
)


parser.add_argument(
    "--inverse_method",
    type=str,
    default="pinv",
    help="Method used for inverting interaction matrix to build control (intensity-mode) matrix I2M"
)

parser.add_argument("--fig_path", 
                    type=str, 
                    default='', 
                    help="path/to/output/image/ for the saved figures")




args=parser.parse_args()

# c, dms, darks_dict, I0_dict, N0_dict,  baldr_pupils, I2A = setup(args.beam_id,
#                               args.global_camera_shm, 
#                               args.toml_file) 

NNN= 10 # how many groups of 100 to take for reference images 

I2A_dict = {}
pupil_masks = {}
for beam_id in args.beam_id:

    # read in TOML as dictionary for config 
    with open(args.toml_file.replace('#',f'{beam_id}'), "r") as f:
        config_dict = toml.load(f)
        # Baldr pupils from global frame 
        baldr_pupils = config_dict['baldr_pupils']
        I2A_dict[beam_id] = config_dict[f'beam{beam_id}']['I2A']
        
        pupil_masks[beam_id] = config_dict.get(f"beam{beam_id}", {}).get("pupil_mask", {}).get("mask", None)

# Set up global camera frame SHM 
#print('Setting up camera. You should manually set up camera settings before hand')

I2A_dict = {}
pupil_masks = {}
for beam_id in [1,2,3,4]:

    # read in TOML as dictionary for config 
    with open(args.toml_file.replace('#',f'{beam_id}'), "r") as f:
        config_dict = toml.load(f)
        # Baldr pupils from global frame 
        baldr_pupils = config_dict['baldr_pupils']
        I2A_dict[beam_id] = config_dict[f'beam{beam_id}']['I2A']
        
        pupil_masks[beam_id] = config_dict.get(f"beam{beam_id}", {}).get("pupil_mask", {}).get("mask", None)




# #---------- New Darks 
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


# get darks and bad pixels 
dark_fits_files = glob.glob("/home/asg/Progs/repos/asgard-alignment/calibration/cal_data/darks/*.fits") 
most_recent_dark = max(dark_fits_files, key=os.path.getmtime) 

dark_fits = fits.open( most_recent_dark )

bad_pixels, bad_pixel_mask = FLI.get_bad_pixels( dark_fits["DARK_FRAMES"].data, std_threshold=10, mean_threshold=10)
bad_pixel_mask[0][0] = False # the frame tag should not be masked! 


c_dict = {}
for beam_id in args.beam_id:
    r1,r2,c1,c2 = baldr_pupils[f'{beam_id}']
    c_dict[beam_id] = FLI.fli(args.global_camera_shm, roi = [r1,r2,c1,c2])
    c_dict[beam_id].reduction_dict['bad_pixel_mask'].append( (~bad_pixel_mask).astype(int)[r1:r2, c1:c2] )
    c_dict[beam_id].reduction_dict['dark'].append(  dark_fits["MASTER DARK"].data.astype(int)[r1:r2, c1:c2] )

#c_dict[beam_id].build_dark( no_frames = 100)
#    c_dict[beam_id].reduction_dict['bad_pixel_mask'].append( (~bad_pixel_mask).astype(int)[r1:r2, c1:c2] )
#    c_dict[beam_id].reduction_dict['dark'].append(  dark_fits["MASTER DARK"].data.astype(int)[r1:r2, c1:c2] )


# fps = c_dict[beam_id].send_fli_cmd("fps")

# set up DM SHMs 
print( 'setting up DMs')
dm_shm_dict = {}
for beam_id in args.beam_id:
    dm_shm_dict[beam_id] = dmclass( beam_id=beam_id )
    # zero all channels
    dm_shm_dict[beam_id].zero_all()
    
    # activate flat (does this on channel 1)
    dm_shm_dict[beam_id].activate_flat()

    # apply dm flat + calibrated offset (does this on channel 1)
    #dm_shm_dict[beam_id].activate_calibrated_flat()
    




# # Move to phase mask
# for beam_id in args.beam_id:
#     message = f"fpm_movetomask phasemask{beam_id} {args.phasemask}"
#     res = send_and_get_response(message)
#     print(f"moved to phasemask {args.phasemask} with response: {res}")

time.sleep(1)

# # Get reference pupils (later this can just be a SHM address)
# zwfs_pupils = {}
# clear_pupils = {}
# rel_offset = 200.0 #um phasemask offset for clear pupil
# print( 'Moving FPM out to get clear pupils')
# for beam_id in args.beam_id:
#     message = f"moverel BMX{beam_id} {rel_offset}"
#     res = send_and_get_response(message)
#     print(res) 
#     time.sleep( 1 )
#     message = f"moverel BMY{beam_id} {rel_offset}"
#     res = send_and_get_response(message)
#     print(res) 

# time.sleep(10)


# #Clear Pupil
# for beam_id in args.beam_id:

#     print( 'gettin clear pupils')
#     N0s = []
#     for _ in range(NNN):
#         N0s.append(  c_dict[beam_id].get_data(apply_manual_reduction=True) )
#     N0s = np.array(  N0s ).reshape(-1,  N0s[0].shape[1],  N0s[0].shape[2])

#     r1,r2,c1,c2 = baldr_pupils[f"{beam_id}"]
#     #cropped_imgs = [nn[r1:r2,c1:c2] for nn in N0s]
#     clear_pupils[beam_id] = N0s

#     # move back 
#     print( 'Moving FPM back in beam.')
#     message = f"moverel BMX{beam_id} {-rel_offset}"
#     res = send_and_get_response(message)
#     print(res) 
#     time.sleep(1)
#     message = f"moverel BMY{beam_id} {-rel_offset}"
#     res = send_and_get_response(message)
#     print(res) 
#     time.sleep(10)


# # check the alignment is still ok 
# #input('ensure mask is realigned')

# # beam = int( input( "\ndo you want to check the phasemasks for a beam. Enter beam number (1,2,3,4) or 0 to continue\n") )
# # while beam :
# #     save_tmp = 'delme.png'
# #     print(f'open {save_tmp } to see generated images after each iteration')
    
# #     move_relative_and_get_image(cam=c, beam=beam, baldr_pupils = baldr_pupils, phasemask=state_dict["socket"],  savefigName = save_tmp, use_multideviceserver=True )
    
# #     beam = int( input( "\ndo you want to check the phasemasks for a beam. Enter beam number (1,2,3,4) or 0 to continue\n") )


# # ZWFS Pupil
# for beam_id in args.beam_id:

#     print( 'Getting ZWFS pupils')
#     I0s = []
#     for _ in range(NNN):
#         I0s.append(  c_dict[beam_id].get_data( apply_manual_reduction=True ) )
#     I0s = np.array(  I0s ).reshape(-1,  I0s[0].shape[1],  I0s[0].shape[2])

#     #r1,r2,c1,c2 = baldr_pupils[f"{beam_id}"]
#     #cropped_img = interpolate_bad_pixels(img[r1:r2, c1:c2], bad_pixel_mask[r1:r2, c1:c2])
#     #cropped_img = [nn[r1:r2,c1:c2] for nn in I0s] #/np.mean(img[r1:r2, c1:c2][pupil_masks[bb]])
#     zwfs_pupils[beam_id] = I0s #cropped_img


# #dark = np.mean( darks_dict[beam_id],axis=0)

# I0 = {}
# N0 = {}
# I0_dm = {}
# N0_dm = {}
# dark_dm = {}
# dm_act_filt = {}
# dm_mask = {}
# for beam_id in args.beam_id:
#     N0[beam_id] = np.mean(clear_pupils[beam_id],axis=0)
#     I0[beam_id] = np.mean(zwfs_pupils[beam_id],axis=0)
#     #interpMatrix = I2A_dict[beam_id]
#     I0_dm[beam_id] = I2A_dict[beam_id] @ I0[beam_id].reshape(-1)
#     N0_dm[beam_id] = I2A_dict[beam_id] @ N0[beam_id].reshape(-1)

#     dark_dm[beam_id] = I2A_dict[beam_id] @ c_dict[beam_id].reduction_dict['dark'][-1].reshape(-1)

# dm_act_filt = {}
# dm_mask = {}
# for beam_id in args.beam_id:
#     ### IMPORTANT THIS IS WHERE WE FILTER ACTUATOR SPACE IN IM 
#     dm_mask[beam_id] = I2A_dict[beam_id] @  np.array(pupil_masks[beam_id] ).reshape(-1)
#     dm_act_filt[beam_id] = dm_mask[beam_id] > 0.95 # ignore actuators on the edge! 


# # checks 
# cbar_label_list = ['[adu]','[adu]', '[adu]']
# title_list = ['DARK','I0', 'N0']
# xlabel_list = ['','','']
# ylabel_list = ['','','']

# util.nice_heatmap_subplots( im_list = [ c_dict[beam_id].reduction_dict['dark'][-1], I0[beam_id] , N0  ], 
#                             cbar_label_list = cbar_label_list,
#                             title_list=title_list,
#                             xlabel_list=xlabel_list,
#                             ylabel_list=ylabel_list,
#                             savefig='delme.png' )


basis_name = args.basis_name #"zonal" #"ZERNIKE"

if basis_name == "zernike":
    Nmodes = args.Nmodes
    modal_basis = dmbases.zer_bank(2, Nmodes+2 )
elif basis_name == "zonal":
    Nmodes = 140
    modal_basis = np.array([dm_shm_dict[beam_id].cmd_2_map2D(ii) for ii in np.eye(Nmodes)]) 
else:
    raise UserWarning("invalid basis name.")

M2C = modal_basis.copy() # mode 2 command matrix 

phase_cov = np.eye( 140 ) #np.array(IM).shape[0] )
noise_cov = np.eye( Nmodes ) #np.array(IM).shape[1] )



### Just doing 1 for now
#############
beam_id = args.beam_id[0]
############


#r1,r2,c1,c2 = baldr_pupils[f'{beam_id}']

cam_config = c_dict[beam_id].get_camera_config()


# # check they interpolate ok onto DM 
# cbar_label_list = ['[adu]','[adu]', '[adu]','[unitless]']
# title_list = ['DARK DM','I0 DM', 'N0 DM','(I-I0)/N0']
# xlabel_list = ['','','','']
# ylabel_list = ['','','','']
# # intensity from first measurement 
# idm = I2A_dict[beam_id] @  zwfs_pupils[beam_id][0].reshape(-1)

# im_list = [util.get_DM_command_in_2D(a) for a in [ dark_dm[beam_id], I0_dm[beam_id] , N0_dm[beam_id] ,(idm -I0_dm[beam_id]) /  N0_dm[beam_id] ]]
# util.nice_heatmap_subplots( im_list = im_list, 
#                             cbar_label_list = cbar_label_list,
#                             title_list=title_list,
#                             xlabel_list=xlabel_list,
#                             ylabel_list=ylabel_list,
#                             savefig='delme.png' )

# #dms[beam_id].zero_all()
# time.sleep(1)
# #dms[beam_id].activate_flat()


IM = []
Iplus_all = []
Iminus_all = []
imgs_to_mean = 20 # for each poke we average this number of frames
for i,m in enumerate(modal_basis):
    print(f'executing cmd {i}/{len(modal_basis)}')
    I_plus_list = []
    I_minus_list = []
    for sign in [(-1)**n for n in range(10)]: #[-1,1]:

        dm_shm_dict[beam_id].set_data(  sign * args.poke_amp/2 * m ) 
        
        time.sleep(2/float(cam_config["fps"]))

        if sign > 0:
            I_plus_list.append( list( np.mean( c_dict[beam_id].get_some_frames( number_of_frames = imgs_to_mean, apply_manual_reduction = True ),axis = 0)  ) )
            #I_plus *= 1/np.mean( I_plus )
        if sign < 0:
            I_minus_list.append( list( np.mean( c_dict[beam_id].get_some_frames( number_of_frames = imgs_to_mean, apply_manual_reduction = True ),axis = 0)  ) )
            #I_minus *= 1/np.mean( I_minus )

    I_plus = np.mean( I_plus_list, axis = 0).reshape(-1) #/ N0[beam_id].reshape(-1)
    I_minus = np.mean( I_minus_list, axis = 0).reshape(-1) #/  N0[beam_id].reshape(-1)

    #errsig = dm_mask[beam_id] * ( I2A_dict[beam_id] @ ((I_plus - I_minus))  )  / args.poke_amp  # dont use pokeamp norm so I2M maps to naitive DM units (checked in /Users/bencb/Documents/ASGARD/Nice_March_tests/IM_zernike100/SVD_IM_analysis.py)
    #errsig = #( I2A_dict[beam_id] @ ((I_plus - I_minus))  )  / args.poke_amp  # dont use pokeamp norm so I2M maps to naitive DM units (checked in /Users/bencb/Documents/ASGARD/Nice_March_tests/IM_zernike100/SVD_IM_analysis.py)
    
    # Try minimize dependancies, if I2A not calibrated or DM mask then the above fails.. keep simple. We can deal with this in post processing
    errsig = (I_plus - I_minus)  / args.poke_amp

    # reenter pokeamp norm
    Iplus_all.append( I_plus_list )
    Iminus_all.append( I_minus_list )

    IM.append( list(  errsig.reshape(-1) ) ) 

# intensity to mode matrix 
if args.inverse_method == 'pinv':
    I2M = np.linalg.pinv( IM )

elif args.inverse_method == 'MAP': # minimum variance of maximum posterior estimator 
    I2M = (phase_cov @ IM @ np.linalg.inv(IM.T @ phase_cov @ IM + noise_cov) ).T #have to transpose to keep convention.. although should be other way round
else:
    raise UserWarning('no inverse method provided')




U,S,Vt = np.linalg.svd( IM, full_matrices=True)
#singular values
plt.figure()
plt.semilogy(S) #/np.max(S))
#plt.axvline( np.pi * (10/2)**2, color='k', ls=':', label='number of actuators in pupil')
plt.legend()
plt.xlabel('mode index')
plt.ylabel('singular values')
plt.savefig(f'{args.fig_path}' + f'IM_singularvalues_beam{beam_id}.png', bbox_inches='tight', dpi=200)
plt.close()


# n_row = round( np.sqrt( M2C.shape[0]) ) - 1

# fig,ax = plt.subplots(n_row, n_row, figsize=(15,15))
# plt.subplots_adjust(hspace=0.1,wspace=0.1)
# for i,axx in enumerate(ax.reshape(-1)):
#     axx.imshow( M2C.T @ U.T[i]  )
#     #axx.set_title(f'mode {i}, S={round(S[i]/np.max(S),3)}')
#     axx.text( 1,2,f'{i}',color='w',fontsize=6)
#     axx.text( 1,3,f'S={round(S[i]/np.max(S),3)}',color='w',fontsize=6)
#     axx.axis('off')
#     #plt.legend(ax=axplt.tight_layout()
# if save_path is not None:
#     plt.savefig(save_path +  f'cam_eignmodes_{pokeamp}.png',bbox_inches='tight',dpi=200)
# plt.show()



# fig,ax = plt.subplots(n_row, n_row, figsize=(15,15))
# plt.subplots_adjust(hspace=0.1,wspace=0.1)
# for i,axx in enumerate(ax.reshape(-1)):
#     axx.imshow( get_DM_command_in_2D( Vt[i] )  )
#     #axx.set_title(f'mode {i}, S={round(S[i]/np.max(S),3)}')
#     axx.text( 1,2,f'{i}',color='w',fontsize=6)
#     axx.text( 1,3,f'S={round(S[i]/np.max(S),3)}',color='w',fontsize=6)
#     axx.axis('off')
#     #plt.legend(ax=a

# if save_path is not None:
#     plt.savefig(save_path +  f'DM_eignmodes_{pokeamp}.png',bbox_inches='tight',dpi=200)
# plt.show()


## reset DMs 
dm_shm_dict[beam_id].zero_all()
# apply dm flat + calibrated offset (does this on channel 1)
dm_shm_dict[beam_id].activate_calibrated_flat()


# dms[beam_id].zero_all()
# time.sleep(1)
# dms[beam_id].activate_flat()


# save the IM to fits for later analysis 

hdul = fits.HDUList()

hdu = fits.ImageHDU(IM)
hdu.header['EXTNAME'] = 'IM'
hdu.header['phasemask'] = args.phasemask
hdu.header['sig'] = 'I(a/2)-I(-a/2)'
hdu.header['beam'] = beam_id
hdu.header['poke_amp'] = args.poke_amp
for k,v in cam_config.items():
    hdu.header[k] = v 

hdul.append(hdu)


hdu = fits.ImageHDU( dark_fits["DARK_FRAMES"].data )
hdu.header['EXTNAME'] = 'DARKS'


hdu = fits.ImageHDU(Iplus_all)
hdu.header['EXTNAME'] = 'I+'
hdul.append(hdu)

hdu = fits.ImageHDU(Iminus_all)
hdu.header['EXTNAME'] = 'I-'
hdul.append(hdu)

hdu = fits.ImageHDU( np.array(pupil_masks[beam_id]).astype(int)) 
hdu.header['EXTNAME'] = 'PUPIL_MASK'
hdul.append(hdu)

hdu = fits.ImageHDU( dm_mask[beam_id] )
hdu.header['EXTNAME'] = 'PUPIL_MASK_DM'
hdul.append(hdu)

hdu = fits.ImageHDU(modal_basis)
hdu.header['EXTNAME'] = 'M2C'
hdul.append(hdu)

hdu = fits.ImageHDU(I2M)
hdu.header['EXTNAME'] = 'I2M'
hdul.append(hdu)

hdu = fits.ImageHDU(I2A_dict[beam_id])
hdu.header['EXTNAME'] = 'interpMatrix'
hdul.append(hdu)

# hdu = fits.ImageHDU(zwfs_pupils[beam_id])
# hdu.header['EXTNAME'] = 'I0'
# hdul.append(hdu)

# hdu = fits.ImageHDU(clear_pupils[beam_id])
# hdu.header['EXTNAME'] = 'N0'
# hdul.append(hdu)

fits_file = '/home/asg/Videos/' + f'IM_full_{Nmodes}{basis_name}_beam{beam_id}_mask-{args.phasemask}_pokeamp_{args.poke_amp}.fits' #_{args.phasemask}.fits'
#f'IM_full_{Nmodes}ZERNIKE_beam{beam_id}_mask-H5_pokeamp_{poke_amp}.fits' #_{args.phasemask}.fits'
hdul.writeto(fits_file, overwrite=True)
print(f'wrote telemetry to \n{fits_file}')


#SCP AUTOMATICALLY TO MY MACHINE 
# remote_file = fits_file   # The file you want to transfer
# remote_user = "bencb"  # Your username on the target machine
# remote_host = "10.106.106.34"  
# # (base) bencb@cos-076835 Downloads % ifconfig | grep "inet " | grep -v 127.0.0.1
# # 	inet 192.168.20.5 netmask 0xffffff00 broadcast 192.168.20.255
# # 	inet 10.200.32.250 --> 10.200.32.250 netmask 0xffffffff
# # 	inet 10.106.106.34 --> 10.106.106.33 netmask 0xfffffffc

# remote_path = "/Users/bencb/Downloads/"  # Destination path on your computer

# # Construct the SCP command
# scp_command = f"scp {remote_file} {remote_user}@{remote_host}:{remote_path}"

# # Execute the SCP command
# try:
#     subprocess.run(scp_command, shell=True, check=True)
#     print(f"File {remote_file} successfully transferred to {remote_user}@{remote_host}:{remote_path}")
# except subprocess.CalledProcessError as e:
#     print(f"Error transferring file: {e}")

