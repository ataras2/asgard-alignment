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

default_toml = os.path.join("/usr/local/etc/baldr/", "baldr_config_#.toml") 

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
    default=5,
    help="camera gain. Default: %(default)s"
)


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
# print('Setting up camera. You should manually set up camera settings before hand')


# # get darks and bad pixels 
# dark_fits_files = glob.glob("/home/asg/Progs/repos/asgard-alignment/calibration/cal_data/darks/*.fits") 
# most_recent_dark = max(dark_fits_files, key=os.path.getmtime) 

# dark_fits = fits.open( most_recent_dark )

# bad_pixels, bad_pixel_mask = FLI.get_bad_pixels( dark_fits["DARK_FRAMES"].data, std_threshold=10, mean_threshold=10)
# bad_pixel_mask[0][0] = False # the frame tag should not be masked! 

c_dict = {}
for beam_id in args.beam_id:
    r1,r2,c1,c2 = baldr_pupils[f'{beam_id}']
    c_dict[beam_id] = FLI.fli(args.global_camera_shm, roi = [r1,r2,c1,c2])

time.sleep(1)
c_dict[beam_id].send_fli_cmd(f"set fps {args.cam_fps}")
time.sleep(1)
c_dict[beam_id].send_fli_cmd(f"set gain {args.cam_gain}")
time.sleep(1)

for beam_id in args.beam_id:
    c_dict[beam_id].build_manual_bias(number_of_frames=500)
    c_dict[beam_id].build_manual_dark(number_of_frames=500, 
                                      apply_manual_reduction=True,
                                      build_bad_pixel_mask=True, 
                                      sleeptime = 20,
                                      kwargs={'std_threshold':10, 'mean_threshold':6} )
    #c_dict[beam_id].reduction_dict['bad_pixel_mask'].append( (~bad_pixel_mask).astype(int)[r1:r2, c1:c2] )
    #c_dict[beam_id].reduction_dict['dark'].append(  dark_fits["MASTER DARK"].data.astype(int)[r1:r2, c1:c2] )


# set up DM SHMs 
print( 'setting up DMs')
dm_shm_dict = {}
for beam_id in args.beam_id:
    dm_shm_dict[beam_id] = dmclass( beam_id=beam_id )
    # zero all channels
    dm_shm_dict[beam_id].zero_all()
    
    # activate flat (does this on channel 1)
    #dm_shm_dict[beam_id].activate_flat()

    # apply dm flat + calibrated offset (does this on channel 1)
    dm_shm_dict[beam_id].activate_calibrated_flat()
    




# Move to phase mask
for beam_id in args.beam_id:
    message = f"fpm_movetomask phasemask{beam_id} {args.phasemask}"
    res = send_and_get_response(message)
    print(f"moved to phasemask {args.phasemask} with response: {res}")

time.sleep(5)

# Get reference pupils (later this can just be a SHM address)
zwfs_pupils = {}
clear_pupils = {}
rel_offset = 200.0 #um phasemask offset for clear pupil
print( 'Moving FPM out to get clear pupils')
for beam_id in args.beam_id:
    message = f"moverel BMX{beam_id} {rel_offset}"
    res = send_and_get_response(message)
    print(res) 
    time.sleep( 1 )
    message = f"moverel BMY{beam_id} {rel_offset}"
    res = send_and_get_response(message)
    print(res) 

time.sleep(10)


#Clear Pupil
for beam_id in args.beam_id:

    print( 'gettin clear pupils')
    #N0s = []
    #for _ in range(NNN):
    N0s = c_dict[beam_id].get_some_frames(number_of_frames = 1000, apply_manual_reduction=True) 
    #  N0s = np.array(  N0s ).reshape(-1,  N0s[0].shape[1],  N0s[0].shape[2])

    r1,r2,c1,c2 = baldr_pupils[f"{beam_id}"]
    #cropped_imgs = [nn[r1:r2,c1:c2] for nn in N0s]
    clear_pupils[beam_id] = N0s

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
input('ensure mask is realigned')

# beam = int( input( "\ndo you want to check the phasemasks for a beam. Enter beam number (1,2,3,4) or 0 to continue\n") )
# while beam :
#     save_tmp = 'delme.png'
#     print(f'open {save_tmp } to see generated images after each iteration')
    
#     move_relative_and_get_image(cam=c, beam=beam, baldr_pupils = baldr_pupils, phasemask=state_dict["socket"],  savefigName = save_tmp, use_multideviceserver=True )
    
#     beam = int( input( "\ndo you want to check the phasemasks for a beam. Enter beam number (1,2,3,4) or 0 to continue\n") )


# ZWFS Pupil
for beam_id in args.beam_id:

    print( 'Getting ZWFS pupils')
    #I0s = []
    #for _ in range(NNN):
    #    I0s.append(  c_dict[beam_id].get_data( apply_manual_reduction=True ) )
    I0s = c_dict[beam_id].get_some_frames(number_of_frames = 1000, apply_manual_reduction=True)
    #I0s = np.array(  I0s ).reshape(-1,  I0s[0].shape[1],  I0s[0].shape[2])

    #r1,r2,c1,c2 = baldr_pupils[f"{beam_id}"]
    #cropped_img = interpolate_bad_pixels(img[r1:r2, c1:c2], bad_pixel_mask[r1:r2, c1:c2])
    #cropped_img = [nn[r1:r2,c1:c2] for nn in I0s] #/np.mean(img[r1:r2, c1:c2][pupil_masks[bb]])
    zwfs_pupils[beam_id] = I0s #cropped_img





#dark = np.mean( darks_dict[beam_id],axis=0)

I0 = {}
N0 = {}
I0_dm = {}
N0_dm = {}
dark_dm = {}
dm_act_filt = {}
dm_mask = {}
for beam_id in args.beam_id:
    N0[beam_id] = np.mean(clear_pupils[beam_id],axis=0)
    I0[beam_id] = np.mean(zwfs_pupils[beam_id],axis=0)
    #interpMatrix = I2A_dict[beam_id]
    I0_dm[beam_id] = I2A_dict[beam_id] @ I0[beam_id].reshape(-1)
    N0_dm[beam_id] = I2A_dict[beam_id] @ N0[beam_id].reshape(-1)

    dark_dm[beam_id] = I2A_dict[beam_id] @ c_dict[beam_id].reduction_dict['dark'][-1].reshape(-1)


    ### IMPORTANT THIS IS WHERE WE FILTER ACTUATOR SPACE IN IM 
    dm_mask[beam_id] = I2A_dict[beam_id] @  np.array(pupil_masks[beam_id] ).reshape(-1)
    dm_act_filt[beam_id] = dm_mask[beam_id] > 0.95 # ignore actuators on the edge! 


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


basis_type = "ZERNIKE"
Nmodes = 3
#modal_basis = np.array([dm_shm_dict[beam_id].cmd_2_map2D(ii) for ii in np.eye(Nmodes)]) #
modal_basis = dmbases.zer_bank(2, Nmodes+2 )
M2C = modal_basis.copy() # mode 2 command matrix 
poke_amp = 0.04 #0.02
inverse_method = 'pinv'
phase_cov = np.eye( 140 ) #np.array(IM).shape[0] )
noise_cov = np.eye( Nmodes ) #np.array(IM).shape[1] )



### Just doing 1 for now
#############
beam_id = args.beam_id[0]
############


#r1,r2,c1,c2 = baldr_pupils[f'{beam_id}']



# check they interpolate ok onto DM 
cbar_label_list = ['[adu]','[adu]', '[adu]','[unitless]']
title_list = ['DARK DM','I0 DM', 'N0 DM','(I-I0)/N0']
xlabel_list = ['','','','']
ylabel_list = ['','','','']
# intensity from first measurement 
idm = I2A_dict[beam_id] @  zwfs_pupils[beam_id][0].reshape(-1)

im_list = [util.get_DM_command_in_2D(a) for a in [ dark_dm[beam_id], I0_dm[beam_id] , N0_dm[beam_id] ,(idm -I0_dm[beam_id]) /  N0_dm[beam_id] ]]
util.nice_heatmap_subplots( im_list = im_list, 
                            cbar_label_list = cbar_label_list,
                            title_list=title_list,
                            xlabel_list=xlabel_list,
                            ylabel_list=ylabel_list,
                            savefig='delme.png' )

#dms[beam_id].zero_all()
time.sleep(1)
#dms[beam_id].activate_flat()


IM = []
Iplus_all = []
Iminus_all = []
for i,m in enumerate(modal_basis):
    print(f'executing cmd {i}/{len(modal_basis)}')
    I_plus_list = []
    I_minus_list = []
    imgs_to_mean = 10
    for sign in [(-1)**n for n in range(10)]: #[-1,1]:

        dm_shm_dict[beam_id].set_data(  sign * poke_amp/2 * m ) 
        NN = 10 # number frames to average for each poke
        time.sleep(NN/args.cam_fps)
        if sign > 0:
            I_plus_list.append( list( np.mean( c_dict[beam_id].get_some_frames(number_of_frames=NN, apply_manual_reduction = True ),axis = 0)  ) )
            #I_plus *= 1/np.mean( I_plus )
        if sign < 0:
            I_minus_list.append( list( np.mean( c_dict[beam_id].get_some_frames(number_of_frames=NN, apply_manual_reduction = True ),axis = 0)  ) )
            #I_minus *= 1/np.mean( I_minus )

    I_plus = np.mean( I_plus_list, axis = 0).reshape(-1) / N0[beam_id].reshape(-1)
    I_minus = np.mean( I_minus_list, axis = 0).reshape(-1) /  N0[beam_id].reshape(-1)

    numerator = np.mean( I_plus_list, axis = 0).reshape(-1)
    denominator = N0[beam_id].reshape(-1)
    I_plus = np.divide(numerator, denominator, out=np.zeros_like(numerator), where=denominator!=0)
    
    numerator = np.mean( I_minus_list, axis = 0).reshape(-1)
    I_minus = np.divide(numerator, denominator, out=np.zeros_like(numerator), where=denominator!=0)
    
    errsig = dm_mask[beam_id] * ( I2A_dict[beam_id] @ ((I_plus - I_minus))  )  / poke_amp  # dont use pokeamp norm so I2M maps to naitive DM units (checked in /Users/bencb/Documents/ASGARD/Nice_March_tests/IM_zernike100/SVD_IM_analysis.py)
    
    # reenter pokeamp norm
    Iplus_all.append( I_plus_list )
    Iminus_all.append( I_minus_list )

    IM.append( list(  errsig.reshape(-1) ) ) 

# intensity to mode matrix 
if inverse_method == 'pinv':
    I2M = np.linalg.pinv( IM )

elif inverse_method == 'MAP': # minimum variance of maximum posterior estimator 
    I2M = (phase_cov @ IM @ np.linalg.inv(IM.T @ phase_cov @ IM + noise_cov) ).T #have to transpose to keep convention.. although should be other way round
else:
    raise UserWarning('no inverse method provided')


## reset DMs 
dm_shm_dict[beam_id].zero_all()
# apply dm flat + calibrated offset (does this on channel 1)
dm_shm_dict[beam_id].activate_calibrated_flat()


# dms[beam_id].zero_all()
# time.sleep(1)
# dms[beam_id].activate_flat()


# save the IM to fits for later analysis 

cam_config = c_dict[beam_id].get_camera_config()

hdul = fits.HDUList()

hdu = fits.ImageHDU(IM)
hdu.header['EXTNAME'] = 'IM'
hdu.header['phasemask'] = args.phasemask
hdu.header['sig'] = 'I(a/2)-I(-a/2)'
hdu.header['beam'] = beam_id
hdu.header['poke_amp'] = poke_amp
for k,v in cam_config.items():
    hdu.header[k] = v 

hdul.append(hdu)


# hdu = fits.ImageHDU(   ) #dark_fits["DARK_FRAMES"].data )
# hdu.header['EXTNAME'] = 'DARKS'


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

hdu = fits.ImageHDU(zwfs_pupils[beam_id])
hdu.header['EXTNAME'] = 'I0'
hdul.append(hdu)

hdu = fits.ImageHDU(clear_pupils[beam_id])
hdu.header['EXTNAME'] = 'N0'
hdul.append(hdu)

fits_file = '/home/asg/Videos/' + f'IM_full_{Nmodes}{basis_type}_beam{beam_id}_mask-{args.phasemask}_pokeamp_{poke_amp}.fits' #_{args.phasemask}.fits'
#f'IM_full_{Nmodes}ZERNIKE_beam{beam_id}_mask-H5_pokeamp_{poke_amp}.fits' #_{args.phasemask}.fits'
hdul.writeto(fits_file, overwrite=True)
print(f'wrote telemetry to \n{fits_file}')


#SCP AUTOMATICALLY TO MY MACHINE 
remote_file = fits_file   # The file you want to transfer
remote_user = "bencb"  # Your username on the target machine
remote_host = "10.106.106.34"  
# (base) bencb@cos-076835 Downloads % ifconfig | grep "inet " | grep -v 127.0.0.1
# 	inet 192.168.20.5 netmask 0xffffff00 broadcast 192.168.20.255
# 	inet 10.200.32.250 --> 10.200.32.250 netmask 0xffffffff
# 	inet 10.106.106.34 --> 10.106.106.33 netmask 0xfffffffc

remote_path = "/Users/bencb/Downloads/"  # Destination path on your computer

# Construct the SCP command
scp_command = f"scp {remote_file} {remote_user}@{remote_host}:{remote_path}"

# Execute the SCP command
try:
    subprocess.run(scp_command, shell=True, check=True)
    print(f"File {remote_file} successfully transferred to {remote_user}@{remote_host}:{remote_path}")
except subprocess.CalledProcessError as e:
    print(f"Error transferring file: {e}")






## SOME CHECKS 
print( f'condition of IM = {np.linalg.cond(IM)}')

imgs = [  util.get_DM_command_in_2D( i) for i in IM ][:7]
titles = ['' for _ in imgs]
cbars = ['' for _ in imgs]
xlabel_list = ['' for _ in imgs]
ylabel_list = ['' for _ in imgs]
util.nice_heatmap_subplots( imgs ,title_list=titles,xlabel_list=xlabel_list, ylabel_list=ylabel_list, cbar_label_list=cbars, fontsize=15, cbar_orientation = 'bottom', axis_off=True, vlims=None, savefig='delme.png' )




# get some new images and project them onto the modes 
i =  c_dict[beam_id].get_data( apply_manual_reduction=True)

ss = (I2A_dict[beam_id] @ ((i - I0[beam_id] ) / N0[beam_id]).reshape( -1, len(i) ) )
sig = np.array( [dm_act_filt[beam_id] * ii for ii in ss.T]).T

reco = I2M.T @ sig

Nplots = 5
fig,ax = plt.subplots(Nplots,1,figsize=(5,10),sharex=True)
for m,axx in zip( poke_amp * reco[:Nplots], ax.reshape(-1)):
    axx.hist( m , label=f'mode {m}')
    axx.set_ylabel('frequency')
    axx.axvline(0, color='k',ls=':')
ax[0].set_title('reconstruction on zero input aberrations')
ax[-1].set_xlabel('reconstructed mode amplitude\n[DM units]')
plt.savefig('delme.png') 



##################

pp = 0.04
m = 0
abb = pp * modal_basis[m] 
dm_shm_dict[beam_id].set_data(  abb ) 

time.sleep(1)

i = np.mean( c_dict[beam_id].get_data( apply_manual_reduction=True ), axis=0)

numerator =  dm_mask[beam_id] * (I2A_dict[beam_id] @ (i - I0[beam_id] ).reshape(-1)) 
denominator = dm_mask[beam_id] * ( I2A_dict[beam_id] @ ( N0[beam_id].reshape(-1) ) )
sig =  np.divide(numerator, denominator, out=np.zeros_like(numerator), where=denominator!=0)

#sig =  dm_mask[beam_id] * (I2A_dict[beam_id] @ ((i - I0[beam_id] ) / N0[beam_id]).reshape(-1) )

err = I2M.T @ sig

reco =  (M2C.T @ err).T 

res = abb - reco 

rmse = np.sqrt( np.mean( (res)**2 ))

dmfilt_12x12 = util.get_DM_command_in_2D( dm_act_filt[beam_id] )
#im_list = [i, util.get_DM_command_in_2D(sig), dm_act_filt[beam_id] * reco, dm_act_filt[beam_id] * res ]
im_list = [abb ,  i.T , util.get_DM_command_in_2D(sig), reco,  res ]

title_list = ["disturbance", "intensity", "signal", "reco.", "residual"]
vlims = [[np.nanmin(thing), np.nanmax(thing)] for thing in im_list[:-1]] 
vlims.append( vlims[-1] )
cbar_label_list = ["DM UNITS", "ADU", "ADU", "DM UNITS", "DM UNITS"]
util.nice_heatmap_subplots( im_list, title_list = title_list, cbar_label_list=cbar_label_list, vlims= vlims, savefig='delme.png')



################# LOOKING AT ERROR SIGNAL 
m = 0
errgrid = []
resgrid = []
ampgrid = np.linspace(-0.2, 0.2, 100)
for pp in ampgrid:
    print(pp)
    abb = pp * modal_basis[m] 
    dm_shm_dict[beam_id].set_data(  abb ) 

    time.sleep(2)

    i = np.mean( c_dict[beam_id].get_data( apply_manual_reduction=True ), axis=0)

    #sig =  dm_mask[beam_id] * (I2A_dict[beam_id] @ ((i - I0[beam_id] ) / N0[beam_id]).reshape(-1) )
    # Safer way to do it (maybe slower)
    numerator =  dm_mask[beam_id] * (I2A_dict[beam_id] @ (i - I0[beam_id] ).reshape(-1)) 
    denominator = dm_mask[beam_id] * ( I2A_dict[beam_id] @ ( N0[beam_id].reshape(-1) ) )
    sig =  np.divide(numerator, denominator, out=np.zeros_like(numerator), where=denominator!=0)

    err = I2M.T @ sig

    reco =  (M2C.T @ err).T 

    res = abb - reco 

    errgrid.append( err[m] )
    resgrid.append( np.std( res[m] ) )



cam_config = c_dict[beam_id].get_camera_config()

hdul = fits.HDUList()

hdu = fits.ImageHDU(IM)
hdu.header['EXTNAME'] = 'IM'
hdu.header['phasemask'] = args.phasemask
hdu.header['sig'] = 'I(a/2)-I(-a/2)'
hdu.header['beam'] = beam_id
hdu.header['poke_amp'] = poke_amp
for k,v in cam_config.items():
    hdu.header[k] = v 

hdul.append(hdu)

hdu = fits.ImageHDU(ampgrid)
hdu.header['EXTNAME'] = 'disturb_grid'
hdu.header['mode_index'] = m
hdul.append(hdu)

hdu = fits.ImageHDU(errgrid)
hdu.header['EXTNAME'] = 'err_grid'
hdul.append(hdu)

hdu = fits.ImageHDU(resgrid)
hdu.header['EXTNAME'] = 'resgrid'
hdul.append(hdu)

fits_file = '/home/asg/Videos/' + f'capture_range_{basis_type}_mode{m}_beam{beam_id}_mask-{args.phasemask}_pokeamp_{poke_amp}.fits' #_{args.phasemask}.fits'
#f'IM_full_{Nmodes}ZERNIKE_beam{beam_id}_mask-H5_pokeamp_{poke_amp}.fits' #_{args.phasemask}.fits'
hdul.writeto(fits_file, overwrite=True)
print(f'wrote telemetry to \n{fits_file}')


plt.figure(figsize=(8,5))
plt.plot( 7000 * ampgrid, 7000 * np.array(errgrid), color='k', label="Measured") 
plt.plot( 7000 * ampgrid, 7000 * ampgrid, color="r", ls=":", label="1:1") 
plt.xlim([-1000,1000])
plt.ylim([1.2*np.min(7000 * np.array(errgrid)),-1.2*np.min(7000 * np.array(errgrid))])
plt.axvline(0, color='k',ls=':')
plt.axhline(0, color='k',ls=':')
plt.legend(fontsize=13)
plt.text( 100, -150 , f"phase mask : {args.phasemask}", color='k',alpha=0.5, fontsize=15)
plt.gca().tick_params(labelsize=13)
#plt.xlabel("Tip/Tilt Disturbance [nm RMS]",fontsize=15)
plt.xlabel("Focus Disturbance [nm RMS]",fontsize=15)
plt.ylabel("Error Signal [nm RMS]",fontsize=15)
plt.savefig('delme.png')



# print( f'mean mode reco = {np.mean( reco , axis = 1)}') 
# print( f'std mode reco = {np.std( reco, axis = 1 )}')

#plt.figure(); plt.imshow( np.cov( IM ) / poke_amp ) ;  plt.colorbar(); plt.savefig('delme.png')

# ##############
# # lets look at the command recosntruction
# delta_cmd = (M2C.T @ reco).T

# i= 30
# plt.figure() ; plt.imshow( delta_cmd[3]); plt.colorbar(); plt.savefig('delme.png')

# ##################
# ### Apply command and see if reconstruct amplitude 

# errs = []
# pokegrid = np.linspace( -2*poke_amp , 2*poke_amp, 8) 
# m = 1 
# for pp in pokegrid:
#     print(f'poke mode {m} with {pp}amp')
#     abb =  pp * modal_basis[m] 
#     time.sleep(2)
#     dm_shm_dict[beam_id].set_data(  abb ) 
#     time.sleep( 10 )

#     # cropped_image = np.mean( c.get_data(), axis = 0)[r1:r2, c1:c2]

#     # i_dm = interpMatrix @ cropped_image.reshape(-1)


#     # get some new images and project them onto the modes 
#     i = c_dict[beam_id].get_data( apply_manual_reduction=True )
    

#     sig =  ( dm_mask[np.newaxis] * (interpMatrix @ ((i - I0 ) / N0).reshape(-1,100)).T ).T


#     # current model has no normalization 
#     #sig = process_signal( i_dm, I0_dm, N0_dm )

#     #plt.imshow( util.get_DM_command_in_2D(sig) ); plt.savefig('delme.png')
#     # (4) apply linear model to get reconstructor 
#     e_HO = poke_amp**2 * I2M.T @ sig #slopes * sig + intercepts
#     errs.append( e_HO )


# errs = np.array(errs)

# plt.figure()
# for mm in range(errs.shape[1]):
#     if mm == m:
#         ls='-'
#     else:
#         ls=':'
#     mean_err =   np.mean( errs[:,mm,:] , axis=-1)
#     plt.plot( pokegrid, mean_err, label=f'mode {mm}',ls=ls)

# plt.legend() 
# plt.savefig('delme.png')

# err_correction , _ = np.polyfit(  np.mean( errs[:,m,:] , axis=-1), pokegrid, 1 )


# adj_norm = err_correction * poke_amp
# print(e_HO)


# abb =  poke_amp * modal_basis[m] 
# time.sleep(2)
# dms[beam_id].shms[1].set_data(  abb ) 
# time.sleep( 10 )

# # cropped_image = np.mean( c.get_data(), axis = 0)[r1:r2, c1:c2]

# # i_dm = interpMatrix @ cropped_image.reshape(-1)


# # get some new images and project them onto the modes 
# ifull = c.get_data()
# i = np.array( [ii[r1:r2,c1:c2] for ii in ifull] )

# sig = interpMatrix @ ((i - I0 ) / N0).reshape(-1,100)


# # current model has no normalization 
# #sig = process_signal( i_dm, I0_dm, N0_dm )

# #plt.imshow( util.get_DM_command_in_2D(sig) ); plt.savefig('delme.png')
# # (4) apply linear model to get reconstructor 
# e_HO_all = adj_norm * I2M.T @ sig #slopes * sig + intercepts

# print( f'mean mode err for mode {m} = {np.mean( e_HO_all[m,:])}, applied mode amp = {poke_amp}' )

# e_HO = np.mean( e_HO_all, axis = 1 )

# delta_cmd =   (M2C.T @ e_HO).T
 


# dm_filt = util.get_DM_command_in_2D( util.get_circle_DM_command(radius=5, Nx_act=12) )
# imgs = [ abb , np.mean( (i - I0 ) / N0, axis=0), dm_filt * delta_cmd, dm_filt*(abb-delta_cmd) ]
# titles = ['DM disturbance', 'ZWFS signal', 'DM reconstruction', 'DM residual']
# cbars = ['' for _ in imgs]
# xlabel_list = ['' for _ in imgs]
# ylabel_list = ['' for _ in imgs]
# util.nice_heatmap_subplots( imgs ,
#                            title_list=titles,
#                            xlabel_list=xlabel_list,
#                            ylabel_list=ylabel_list,
#                            cbar_label_list=cbars, 
#                            fontsize=15, 
#                            cbar_orientation = 'bottom', 
#                            axis_off=True, 
#                            vlims=None, 
#                            savefig='delme.png' )

# inputIgnore = input('check residuals are ok in delme.png')

# plt.close()







### TRY CLOSE LOOP 

# basic setup 
no_its = 500#2000 #500
record_telemetry = True
Nmodes_removed = 14

close_after = 100 #1000 #100
disturbances_on = True #False 

## disturbance (if disturbances_on )

# Kolmogorov Phasescreen 
D = 1.8
act_per_it = 0.2 # how many actuators does the screen pass per iteration 
V = 10 / act_per_it  / D #m/s (10 actuators across pupil on DM)
corner_indicies = [0, 11, 11 * 12, -1] # DM corner indidices
scaling_factor = 0.1
phasescreen_nx_size = int( 12 / act_per_it )
scrn = ps.PhaseScreenVonKarman(nx_size= phasescreen_nx_size , 
                            pixel_scale= D / 12, 
                            r0=0.1, 
                            L0=12,
                            random_seed=1)



## additional TT disturbances (lab turbulance) - 1/f spectra
a_tip = np.cumsum( np.random.randn( no_its ) )
a_tilt = np.cumsum( np.random.randn( no_its ) )

a_piston = np.cumsum( np.random.randn( no_its ) )

# 0 mean , unity variance
a_tip -= np.mean( a_tip )
a_tip /= np.std( a_tip )

a_tilt -= np.mean( a_tilt )
a_tilt /= np.std( a_tilt )

a_piston-= np.mean( a_piston )
a_piston /= np.std( a_piston )

piston_rms = 0.03

# define our TT rms in DM units
TT_disturb_rms = 0.03 #0.02

tip_rms = np.sqrt(2) * TT_disturb_rms
tilt_rms = np.sqrt(2) * TT_disturb_rms
a_tip *= tip_rms
a_tilt *= tilt_rms
# piston
a_piston *= piston_rms


modal_disturbances = []
for ii in range(no_its):
    if disturbances_on:
        scrn.add_row()

        dm_scrn = util.create_phase_screen_cmd_for_DM(scrn,  scaling_factor=scaling_factor, drop_indicies = [0, 11, 11 * 12, -1] , plot_cmd=False)

        dm_scrn_2D = np.nan_to_num( util.get_DM_command_in_2D(dm_scrn), 0 )
        modal_disturbances.append( [np.sum( m * dm_scrn_2D ) / np.sum( modal_basis[1]>0 )  if i > Nmodes else 0 for i,m in enumerate( modal_basis )] )
        # add the additional tip tilt 
        modal_disturbances[-1][0] = a_tip[ii]
        modal_disturbances[-1][1] = a_tilt[ii]
    else:
        modal_disturbances.append( [0 for i,m in enumerate( modal_basis )] ) 


#def close_loop( a_ki = 0.2 ): 
piston_offset_grid = np.arange(-0.5, 0.0, 0.05)
for piston_offset in piston_offset_grid:# 
#if 1: 
    for a_ki in [0.0, 0.8]: # 0.1, 0.3, 0.5 ,0.7, 0.9, 1.0, 1.2, 1.5]:
        #print( a_ki )
        cnt = 0 # start count again (loop closes when cnt == close_after)
        if 1:   
            dm_shm_dict[beam_id].zero_all()
            # activate flat (does this on channel 1)
            dm_shm_dict[beam_id].activate_calibrated_flat()

            I0_dm = I2A_dict[beam_id] @ I0[beam_id].reshape(-1)
            N0_dm = I2A_dict[beam_id] @ N0[beam_id].reshape(-1)

            # Flat DM
            #dms[beam_id].zero_all()
            time.sleep(1)
            #dms[beam_id].activate_flat()

            closed = True


            # Telemetry 
            telem = SimpleNamespace( **init_telem_dict() )

            # PID Controller
            N = I2M.shape[1]
            kp = 0. * np.ones( N)
            ki = 0. * np.ones( N )
            kd = 0. * np.ones( N )
            setpoint = np.zeros( N )
            lower_limit_pid = -100 * np.ones( N )
            upper_limit_pid = 100 * np.ones( N )

            ctrl_HO = PIDController(kp, ki, kd, upper_limit_pid, lower_limit_pid, setpoint)

            # gain to apply after "close_after" iterations

            ki_array =  0 * np.linspace(1e-4, 0.5, N)[::-1]
            ki_array[0] = a_ki
            ki_array[1] = a_ki


            t0 = 1e6 * time.time() #us

            # To add a static aberration at beginning 
            #disturbance2D = -0.03 * M2C[0]
            #dm_shm_dict[beam_id].shms[2].set_data( disturbance2D  ) 
            #dm_shm_dict[beam_id].shm0.post_sems(1)
            
            while closed and (cnt < no_its):
                
                if disturbances_on:
                    disturbance = M2C.T @ modal_disturbances[cnt]

                    #ensure correct format for SHM 
                    #disturbance2D = a_piston[cnt] + disturbance #np.nan_to_num( util.get_DM_command_in_2D(disturbance), 0 )
                    disturbance2D = piston_offset + disturbance #np.nan_to_num( util.get_DM_command_in_2D(disturbance), 0 )

                    #apply atm disturbance on channel 2 
                    dm_shm_dict[beam_id].shms[2].set_data( disturbance2D  ) 
                    #dm_shm_dict[beam_id].shms[2].set_data( disturbance  ) 
                    #dm_shm_dict[beam_id].shm0.post_sems(1) ## <- we do this at the end
                #print(cnt)
                #time.sleep(3) 

                # Cropped pupil region
                i = c_dict[beam_id].get_image( apply_manual_reduction=True ) #np.mean( c.get_data(), axis = 0)[r1:r2, c1:c2]

                # interpolate intensities to registered DM pixels 
                #i_dm = I2A_dict[beam_id] @ cropped_image.reshape(-1)

                # normalise  ### IT WILL BE QUICKER TO DO THIS DIRECTLY IN DM SPACE (CONVERT PRIOR) -VERIFY FIRST 
                #sig = dm_mask[beam_id] * (I2A_dict[beam_id] @ ((i - I0[beam_id] ) / N0[beam_id]).reshape(-1) ) #process_signal( i_dm, I0_dm, N0_dm)
                numerator =  dm_mask[beam_id] * (I2A_dict[beam_id] @ (i - I0[beam_id] ).reshape(-1)) 
                denominator = dm_mask[beam_id] * ( I2A_dict[beam_id] @ ( N0[beam_id].reshape(-1) ) )
                
                sig =  np.divide(numerator, denominator, out=np.zeros_like(numerator), where=denominator!=0)

                # apply linear reconstructor to signal to get modal errors
                e_HO = I2M.T @ sig 

                #print( f'e_HO(first ten) = {e_HO[:10]}' )

                # apply PID controller 
                u_HO = ctrl_HO.process( e_HO )
                

                #print( f'u_HO(first ten) = {u_HO[:10]}' )

                #print( f'ctrl_HO.output(first ten) = {ctrl_HO.output[:10]}' )

                #print( f'ctrl_HO.integrals(first ten) = {ctrl_HO.integrals[:10]}' )

                # forcefully remove piston 
                #u_HO -= np.mean( u_HO )
                
                # send command (filter piston)
                #delta_cmd = np.zeros( len(zwfs_ns.dm.dm_flat ) )
                #delta_cmd zwfs_ns.reco.linear_zonal_model.act_filt_recommended ] = u_HO
                # factor of 2 since reflection
                delta_cmd = (M2C.T @ u_HO).T  #[ dm_act_filt[beam_id] ] = u_HO[ dm_act_filt[beam_id]  ]
                # be careful with tranposes , made previouos mistake tip was tilt etc (90 degree erroneous rotation)
                #cmd = -delta_cmd #disturbance - delta_cmd 

                print( cnt, np.std( delta_cmd ) , ki )

                # record telemetry with the new image and processed signals but the current DM commands (not updated)
                if record_telemetry :
                    
                    telem.time.append( time.time()*1e6 - t0  ) #datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S') )
                    telem.i_list.append( i )
                    #telem.i_dm_list.append( i_dm )
                    telem.s_list.append( sig )

                    telem.e_TT_list.append( np.zeros( len(e_HO) ) )
                    telem.u_TT_list.append( np.zeros( len(e_HO) ) )
                    telem.c_TT_list.append( np.zeros( len(delta_cmd) ) )

                    telem.e_HO_list.append( e_HO )
                    telem.u_HO_list.append( u_HO )
                    telem.c_HO_list.append( delta_cmd ) # the next DM command to be applied to channel 2 (default of dm_shm_dict[beam_id].set_data()  )

                    telem.current_dm_ch0.append( dm_shm_dict[beam_id].shms[0].get_data() )
                    telem.current_dm_ch1.append( dm_shm_dict[beam_id].shms[1].get_data() )
                    telem.current_dm_ch2.append( dm_shm_dict[beam_id].shms[2].get_data() )
                    telem.current_dm_ch3.append( dm_shm_dict[beam_id].shms[3].get_data() )
                    # sum of all DM channels (Full command currently applied to DM)
                    telem.current_dm.append( dm_shm_dict[beam_id].shm0.get_data() )

                    telem.modal_disturb_list.append( modal_disturbances[cnt] )
                    

                # if np.std( delta_cmd ) > 0.2:
                #     print('going bad')
                #     dm_shm_dict[beam_id].zero_all()
                #     dm_shm_dict[beam_id].activate_calibrated_flat()
                #     closed = False

                # update control feedback on DM channel 1. Total DM command is sum of all channels 
                dm_shm_dict[beam_id].shms[1].set_data( -delta_cmd ) 
                dm_shm_dict[beam_id].shm0.post_sems(1) # post semaphore to update DM 

                #Carefull - controller integrates even so this doesnt work
                if cnt == close_after :
                    #close tip/tilt only 
                    #ctrl_HO.ki[:2] = ki_v * np.ones(len(ki))
                    ctrl_HO.ki = ki_array

                cnt+=1


            # Finsh - Flatten DMs
            dm_shm_dict[beam_id].zero_all()
            dm_shm_dict[beam_id].activate_calibrated_flat()


        for i in [0,1]:
            print( f"mode {i} RMS before = ", np.std( np.array( telem.e_HO_list )[:close_after, 1 ] ) )

            print( f"mode {i} RMS after = ",np.std( np.array( telem.e_HO_list )[close_after:, 1 ] ) )

            print( f"mode {i} mean err before = ", np.mean( np.array( telem.e_HO_list )[:close_after, 1 ] ) )

            print( f"mode {i} mean err after = ",np.mean( np.array( telem.e_HO_list )[close_after:, 1 ] ) )

        #plt.figure(); plt.imshow(  telem.i_list[-1] ) ;plt.savefig('delme.png')
        

        #plt.figure(); plt.imshow(  telem.i_list[-1] ) ;plt.savefig('delme.png')

        # save telemetry
        runn=f'TT_ki-{a_ki}_pistonOffset_{piston_offset}_with_baldrDMflat_disturbs-{disturbances_on}_CL_sysID_v1'
        # Create a list of HDUs (Header Data Units)
        hdul = fits.HDUList()

        hdu = fits.ImageHDU(IM)
        hdu.header['EXTNAME'] = 'IM'
        hdul.append(hdu)

        hdu = fits.ImageHDU(M2C)
        hdu.header['EXTNAME'] = 'M2C'
        hdul.append(hdu)


        hdu = fits.ImageHDU(I2M)
        hdu.header['EXTNAME'] = 'I2M'
        hdu.header['pokeamp'] = f'{poke_amp}'
        hdul.append(hdu)

        hdu = fits.ImageHDU(I2A_dict[beam_id])
        hdu.header['EXTNAME'] = 'interpMatrix'
        hdul.append(hdu)


        hdu = fits.ImageHDU(dm_shm_dict[beam_id].shms[0].get_data())
        hdu.header['EXTNAME'] = 'DM_FLAT_OFFSET'
        hdul.append(hdu)

        hdu = fits.ImageHDU(ctrl_HO.kp)
        hdu.header['EXTNAME'] = 'Kp'
        hdu.header['close_after'] = f'{close_after}'
        hdul.append(hdu)

        hdu = fits.ImageHDU(ctrl_HO.ki)
        hdu.header['EXTNAME'] = 'Ki'
        hdu.header['close_after'] = f'{close_after}'
        hdul.append(hdu)

        hdu = fits.ImageHDU(ctrl_HO.kd)
        hdu.header['EXTNAME'] = 'Kd'
        hdu.header['close_after'] = f'{close_after}'
        hdul.append(hdu)

        # Add each list to the HDU list as a new extension
        for list_name, data_list in vars(telem).items():
            # Convert list to numpy array for FITS compatibility
            data_array = np.array(data_list, dtype=float)  # Ensure it is a float array or any appropriate type

            # Create a new ImageHDU with the data
            hdu = fits.ImageHDU(data_array)

            # Set the EXTNAME header to the variable name
            hdu.header['EXTNAME'] = list_name

            # Append the HDU to the HDU list
            hdul.append(hdu)


        # Write the HDU list to a FITS file
        tele_pth = f'/home/asg/Videos/TT_with_static_piston_{TT_disturb_rms}rms/' #f'/home/asg/Videos/TT_on_1onf_TT_disturb_w_piston_long_{TT_disturb_rms}rms/'
        if not os.path.exists( tele_pth ):
            os.makedirs( tele_pth )

        fits_file = tele_pth + f'CL_beam{beam_id}_mask{args.phasemask}_{runn}.fits' #_{args.phasemask}.fits'
        hdul.writeto(fits_file, overwrite=True)
        print(f'wrote telemetry to \n{fits_file}')



# import numpy as np
# import scipy.signal as signal
# import matplotlib.pyplot as plt
# from astropy.io import fits

# # Load telemetry FITS file
# fits_file = '/home/asg/Videos/CL_beam2_TT_CL_sysID_v1.fits'
# hdul = fits.open(fits_file)

# # Extract telemetry data
# e_HO = hdul['e_HO_list'].data  # Modal errors (size: no_its x N_modes)
# u_HO = hdul['u_HO_list'].data  # Control commands (size: no_its x N_modes)
# modal_disturbances = hdul['modal_disturb_list'].data  # Injected disturbances (size: no_its x N_modes)

# # Sampling frequency (estimated from loop time)
# dt = 3  # Loop time in seconds
# fs = 1 / dt  # Sampling frequency

# N_modes = e_HO.shape[1]  # Number of controlled modes
# f, Pdd = signal.welch(modal_disturbances, fs=fs, axis=0, nperseg=256)  # Disturbance PSD
# _, Ped = signal.csd(e_HO, modal_disturbances, fs=fs, axis=0, nperseg=256)  # CSD between errors and disturbances
# _, Peu = signal.csd(e_HO, u_HO, fs=fs, axis=0, nperseg=256)  # CSD between errors and commands
# _, Puu = signal.welch(u_HO, fs=fs, axis=0, nperseg=256)  # Control command PSD

# # Compute the closed-loop transfer function H(jω)
# Hjw = Ped / Pdd  

# # Estimate the open-loop transfer function G(jω)
# Cjw = np.zeros_like(Hjw)  # Placeholder for controller transfer function (to be defined)
# Gjw = Hjw / (1 - Hjw * Cjw)  

# # Find maximum stable ki for each mode
# ki_max = np.zeros(N_modes)

# for mode in range(N_modes):
#     for ki in np.logspace(-3, 1, 50):  # Sweep ki values
#         Cjw[:, mode] = ki / (1j * 2 * np.pi * f)  # Integral control transfer function
#         Gjw_test = Hjw[:, mode] / (1 - Hjw[:, mode] * Cjw[:, mode])
        
#         # Find frequencies where the open-loop gain is close to 1
#         unity_gain_indices = np.where(np.abs(Gjw_test * Cjw[:, mode]) >= 1)[0]

#         # Stability condition: |G(jω)C(jω)| < 1
#         # For a system to be robustly stable, it is not enough to just be greater than -180°. Instead, we define phase margin as the difference between -180° and the actual phase at unity gain
#         if len(unity_gain_indices) > 0:
            
#             min_phase_at_unity_gain = np.nanmin(np.angle(Gjw_test[unity_gain_indices] * Cjw[unity_gain_indices, mode], deg=True))
#             print(min_phase_at_unity_gain )
#             if min_phase_at_unity_gain > -150:  # Ensures at least 30° phase margin
#                 ki_max[modpase] = ki
                

# # Plot results
# plt.figure(figsize=(8, 5))
# plt.semilogy(range(N_modes), ki_max, 'o-', label='Max Stable $k_i$')
# plt.xlabel("Mode Index")
# plt.ylabel("Max Stable Integral Gain $k_i$")
# plt.title("Estimated Maximum Stable Integral Gains")
# plt.grid()
# plt.legend()
# plt.show()

# print("Estimated max stable gains:", ki_max)




##### SOME CLEAN UP FUNCTIONS TO TEST 
# def generate_tt_spectrum(no_its, rms=0.02, seed=None):
#     if seed is not None:
#         np.random.seed(seed)
#     signal = np.cumsum(np.random.randn(no_its))
#     signal -= np.mean(signal)
#     signal /= np.std(signal)
#     return signal * rms

# def build_modal_disturbances(no_its, modal_basis, scrn, scaling_factor, Nmodes, 
#                              a_tip=None, a_tilt=None, a_piston=None, disturbances_on=True):
#     n_modes = len(modal_basis)
#     modal_disturbances = np.zeros((no_its, n_modes))
    
#     for ii in range(no_its):
#         if disturbances_on:
#             scrn.add_row()
#             dm_scrn = util.create_phase_screen_cmd_for_DM(
#                 scrn,
#                 scaling_factor=scaling_factor,
#                 drop_indicies=[0, 11, 11 * 12, -1],
#                 plot_cmd=False
#             )
#             dm_scrn_2D = np.nan_to_num(util.get_DM_command_in_2D(dm_scrn), 0)
#             for i, m in enumerate(modal_basis):
#                 if i > Nmodes:
#                     modal_disturbances[ii, i] = np.sum(m * dm_scrn_2D) / np.sum(modal_basis[1] > 0)
#             if a_tip is not None:
#                 modal_disturbances[ii, 0] = a_tip[ii]
#             if a_tilt is not None:
#                 modal_disturbances[ii, 1] = a_tilt[ii]
#         else:
#             modal_disturbances[ii, :] = 0.0

#     return modal_disturbances

# def save_telemetry(telem, ctrl_HO, beam_id, args, I0, N0, I2M, I2A_dict, M2C, poke_amp, dm_shm_dict, runn, tele_pth):
#     hdul = fits.HDUList()

#     # Save configuration data
#     for name, array in zip(['IM', 'M2C', 'I2M', 'interpMatrix'],
#                            [I0[beam_id], M2C, I2M, I2A_dict[beam_id]]):
#         hdu = fits.ImageHDU(array)
#         hdu.header['EXTNAME'] = name
#         hdul.append(hdu)

#     # Save controller gains
#     for name, arr in zip(['Kp', 'Ki', 'Kd'], [ctrl_HO.kp, ctrl_HO.ki, ctrl_HO.kd]):
#         hdu = fits.ImageHDU(arr)
#         hdu.header['EXTNAME'] = name
#         hdu.header['close_after'] = str(close_after)
#         hdul.append(hdu)

#     # Save DM flat
#     dm_flat = dm_shm_dict[beam_id].shms[0].get_data()
#     hdu = fits.ImageHDU(dm_flat)
#     hdu.header['EXTNAME'] = 'DM_FLAT_OFFSET'
#     hdul.append(hdu)

#     # Save telemetry
#     for list_name, data_list in vars(telem).items():
#         data_array = np.array(data_list, dtype=float)
#         hdu = fits.ImageHDU(data_array)
#         hdu.header['EXTNAME'] = list_name
#         hdul.append(hdu)

#     if not os.path.exists(tele_pth):
#         os.makedirs(tele_pth)

#     fits_file = os.path.join(tele_pth, f'CL_beam{beam_id}_mask{args.phasemask}_{runn}.fits')
#     hdul.writeto(fits_file, overwrite=True)
#     print(f'wrote telemetry to \n{fits_file}')





######### PRIOR SIMULATIONS OF THIS METHOD 

# import numpy as np
# import scipy.signal as signal
# import matplotlib.pyplot as plt
# from astropy.io import fits
# from scipy.signal import welch, csd, dlti, dstep
# # ------------------------------
# # 1. Define Synthetic System Model
# # ------------------------------

# N_modes = 20  
# no_its = 1000 
# dt = 3.  
# fs = 1 / dt  
# f = np.fft.rfftfreq(no_its, d=dt)  # I redefine this l8er

# # Synthetic system response (random stable dynamics per mode)
# modal_poles = np.linspace(0.5, 0.9, N_modes)

# # Create synthetic modal transfer functions
# system_tf = [dlti([1], [1, -p], dt=dt) for p in modal_poles]

# #-----------------------
# # 2. Define Control Parameters
# # ------------------------------
# ki_values = np.linspace(0.01, 1.0, N_modes)  # Range of integral gains
# kp_values = np.zeros(N_modes)  # No proportional gain
# kd_values = np.zeros(N_modes)  # No derivative gain

# # PID Controller Transfer Function: C(s) =  ki / s
# Cjw = np.zeros((len(f), N_modes), dtype=complex)
# for i in range(N_modes):
#     Cjw[:, i] = ki_values[i] / (1j * 2 * np.pi * f)

# # ------------------------------
# # 3. Generate Synthetic Disturbances
# # ------------------------------

# # Generate Kolmogorov turbulence disturbances
# modal_disturbances = np.cumsum(np.random.randn(no_its, N_modes), axis=0)

# # Add synthetic tip/tilt disturbances
# tip_rms = 0.02
# tilt_rms = 0.02
# modal_disturbances[:, 0] += np.cumsum(np.random.randn(no_its)) * tip_rms
# modal_disturbances[:, 1] += np.cumsum(np.random.randn(no_its)) * tilt_rms

# # Normalize disturbances
# modal_disturbances -= np.mean(modal_disturbances, axis=0)
# modal_disturbances /= np.std(modal_disturbances, axis=0)

# # ------------------------------
# # 4. Simulate Closed-Loop System
# # ------------------------------
# e_HO = np.zeros((no_its, N_modes))  # Modal errors
# u_HO = np.zeros((no_its, N_modes))  # Control commands

# for mode in range(N_modes):
#     sys = system_tf[mode]
    
#     # Simulate the response to disturbance with feedback control
#     _, eeee = signal.dlsim(sys, modal_disturbances[:, mode])
    
#     e_HO[:, mode] = eeee.ravel()

#     # Apply PID control
#     u_HO[:, mode] = -ki_values[mode] * np.cumsum(e_HO[:, mode])  # Integral control

# # ------------------------------
# # 5. Save Telemetry as FITS File
# # ------------------------------
# telem_dict = {
#     "e_HO_list": e_HO,
#     "u_HO_list": u_HO,
#     "modal_disturb_list": modal_disturbances
# }

# fits_file = 'synthetic_CL_telemetry.fits'
# hdul = fits.HDUList()

# for key, data in telem_dict.items():
#     hdu = fits.ImageHDU(data)
#     hdu.header['EXTNAME'] = key
#     hdul.append(hdu)

# #hdul.writeto(fits_file, overwrite=True)
# print(f"Saved synthetic telemetry to {fits_file}")

# # ------------------------------
# # 6. Run Gain Estimation Algorithm on Synthetic Data
# # ------------------------------
# #hdul = fits.open(fits_file)

# e_HO = hdul['e_HO_list'].data
# u_HO = hdul['u_HO_list'].data
# modal_disturbances = hdul['modal_disturb_list'].data

# f, Pdd = signal.welch(modal_disturbances, fs=fs, axis=0, nperseg=256)
# _, Ped = signal.csd(e_HO, modal_disturbances, fs=fs, axis=0, nperseg=256)
# _, Peu = signal.csd(e_HO, u_HO, fs=fs, axis=0, nperseg=256)
# _, Puu = signal.welch(u_HO, fs=fs, axis=0, nperseg=256)

# Hjw = Ped / Pdd  

# #from scipy.interpolate import interp1d


# # Now perform gain calculation

# # Recompute Cjw using the Welch-derived frequency bins
# Cjw = np.zeros((len(f), N_modes), dtype=complex)
# for i in range(N_modes):
#     Cjw[:, i] = ki_values[i] / (1j * 2 * np.pi * f)  # Now using Welch 


# Gjw = Hjw / (1 - Hjw * Cjw) # CL TF

# # lets look at some of the estimated TF's
# plt.figure(); plt.loglog(f,  abs(Gjw ));plt.show()
# plt.figure(); plt.plot(f, np.angle(Gjw ));plt.show()
# plt.figure(); plt.loglog(f,  abs(Hjw ));plt.show()
# plt.figure(); plt.plot(f, np.angle(Hjw ));plt.show()
# plt.figure(); plt.loglog(f,  abs(Gjw * Cjw)) ; plt.show() 

# # find max gain per mode to reach unity gain 
# ki_max = np.zeros(N_modes)
# for mode in range(N_modes):
#     for ki in np.logspace(-3, 1, 50):
#         Cjw[:, mode] = ki / (1j * 2 * np.pi * f)
#         Gjw_test = Hjw[:, mode] / (1 - Hjw[:, mode] * Cjw[:, mode])
        
#         if np.nanmax(np.abs(Gjw_test * Cjw[:, mode])) < 1:
#             ki_max[mode] = ki
#         else:
#             break

# # Plot results
# plt.figure(figsize=(8, 5))
# plt.semilogy(range(N_modes), ki_max, 'o-', label='Max Stable $k_i$')
# plt.xlabel("Mode Index")
# plt.ylabel("Max Stable Integral Gain $k_i$")
# plt.title("Estimated Maximum Stable Integral Gains (Synthetic System)")
# plt.grid()
# plt.legend()
# plt.show()

# print("Estimated max stable gains (synthetic system):", ki_max)



# tt_raw =[]
# tt_red =[]
# for _ in range(1000):
# 	t0 = time.time(); a= c.get_image(apply_manual_reduction=False); t1=time.time(); tt_raw.append(t1-t0)
# t0 = time.time(); a=c.get_image(apply_manual_reduction=True); t1=time.time(); tt_red.append(t1-t0)
# plt.figure()
# plt.hist(tt_raw, bins = np.logspace(-5,-1,20) , alpha=0.5, label="raw")
# plt.hist(tt_red, bins = np.logspace(-5,-1,20) , alpha=0.5, label="reduced")
# plt.xscale("log")
# plt.xlabel(r"frame polling delay [s]") 
# plt.ylabel("frequency [Hz]")
# plt.savefig("delme.png")
