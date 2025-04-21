# read in config 

# I2M , I2A , Strehl , secondary mask
import numpy as np 
import toml
import argparse
import threading
import zmq
import time
import toml
import os 
import matplotlib.pyplot as plt
import glob
import subprocess 
from astropy.io import fits
from scipy.signal import TransferFunction, bode
from types import SimpleNamespace
from asgard_alignment import FLI_Cameras as FLI
from asgard_alignment.DM_shm_ctrl import dmclass
import pyBaldr.utilities as util 

default_toml = "/home/asg/Progs/repos/asgard-alignment/config_files/baldr_config_#.toml"

parser = argparse.ArgumentParser(description="closed")


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

parser.add_argument(
    "--number_of_iterations",
    type=int,
    default=1000,
    help="number of iterations to run"
)

# Beam ids: provided as a comma-separated string and converted to a list of ints.
parser.add_argument(
    "--beam_id",
    type=int,
    default=1,
    help="beam id (integrer)"
)

parser.add_argument(
    "--phasemask",
    type=str,
    default="H3",
    help="Comma-separated beam IDs to apply. Default: 1,2,3,4"
)


parser.add_argument(
    '--kp_LO',
    type=float,
    default=0.0,
    help="proportional gain to use for each mode. Default: %(default)s"
)

parser.add_argument(
    '--ki_LO',
    type=float,
    default=0.0,
    help="integral gain to use for each mode. Default: %(default)s"
)

parser.add_argument(
    '--kd_LO',
    type=float,
    default=0.0,
    help="differential gain to use for each mode. Default: %(default)s"
)



parser.add_argument(
    '--kp_HO',
    type=float,
    default=0.0,
    help="proportional gain to use for each mode. Default: %(default)s"
)

parser.add_argument(
    '--ki_HO',
    type=float,
    default=0.0,
    help="integral gain to use for each mode. Default: %(default)s"
)

parser.add_argument(
    '--kd_HO',
    type=float,
    default=0.0,
    help="differential gain to use for each mode. Default: %(default)s"
)

parser.add_argument(
    '--cam_fps',
    type=int,
    default=1000,
    help="frames per second on camera. Default: %(default)s"
)


parser.add_argument(
    '--cam_gain',
    type=int,
    default=10,
    help="camera gain. Default: %(default)s"
)

parser.add_argument("--fig_path", 
                    type=str, 
                    default=None, 
                    help="path/to/output/image/ for the saved figures")



#### TURBULENCE

# parser.add_argument(
#     '--number_of_turb_iterations',
#     type=int,
#     default=200,
#     help="how many iterations do we run? %(default)s"
# )

# parser.add_argument(
#     '--wvl',
#     type=float,
#     default=1.65,
#     help="simulation wavelength (um). Default: %(default)s"
# )

# parser.add_argument(
#     '--D_tel',
#     type=float,
#     default=1.8,
#     help="telescope diameter for simulation. Default: %(default)s"
# )

# parser.add_argument(
#     '--r0',
#     type=float,
#     default=0.2,
#     help="Fried paraameter (coherence length) of turbulence (in meters) at 500nm. This gets scaled by the simulation wavelength r0~(wvl/0.5)**(6/5). Default: %(default)s"
# )


# parser.add_argument(
#     '--V',
#     type=float,
#     default=0.20,
#     help="equivilant turbulence velocity (m/s) assuming pupil on DM has a 10 acturator diameter, and the input telescope diameter (D_tel). Default: %(default)s"
# )


# parser.add_argument(
#     '--number_of_modes_removed',
#     type=int,
#     default=0,
#     help="number of Zernike modes removed from Kolmogorov phasescreen to simulate first stage AO. This can slow it down for large number of modes. For reference Naomi is typically 7-14. Default: %(default)s"
# )

# parser.add_argument(
#     '--DM_chn',
#     type=int,
#     default=3,
#     help="what channel on DM shared memory (0,1,2,3) to apply the turbulence?. Default: %(default)s"
# )


# parser.add_argument(
#     '--record_turb_telem',
#     type=str,
#     default=None,
#     help="record telemetry? input directory/name.fits to save the fits file if you want,\
#           Otherwise None to not record. if number of iterations is > 1e5 than we stop recording! \
#           (this is around 200 MB) Default: %(default)s"
# )


parser.add_argument(
    '--folder_pth',
    type=str,
    default=f'/home/asg/Videos/test/',
    help="folder to save telemetry in. Default: %(default)s"
)



args=parser.parse_args()

#####################################################
########## JUST DO 1
beam_id = args.beam_id



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
        
    def reset_single_mode(self , mode):
        self.integrals[mode] = 0.0
        self.prev_errors[mode] = 0.0
        self.output[mode] = 0.0
        s
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
        "time_dm" : [],
        "time_cam" : [],
        "i" : [],
        "i_dm":[], 
        "s" : [],
        "e_TT" : [],
        "u_TT" : [],
        "c_TT" : [], # the next TT cmd to send to ch2
        "e_HO" : [],
        "u_HO" : [], 
        "c_HO" : [], # the next H0 cmd to send to ch2 
        "current_dm_ch0" : [], # the current DM cmd on ch1
        "current_dm_ch1" : [], # the current DM cmd on ch2
        "current_dm_ch2" : [], # the current DM cmd on ch3
        "current_dm_ch3" : [], # the current DM cmd on ch4
        "current_dm":[], # the current DM cmd (sum of all channels)
        "modal_disturb_list":[],
        "dm_disturb_list" : [],
        "exterior_sig": [],
        "secondary_sig": [],
    }
    return telemetry_dict





def run_script(command, stop_event, timeout=None):
    try:
        # Start the process
        process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        
        # Instead of blocking on communicate(), poll periodically:
        while process.poll() is None:
            if stop_event.is_set():
                print("Stop signal received. Terminating subprocess...")
                process.terminate()  # or process.kill() if necessary
                break
            time.sleep(0.1)  # Check every 100ms
        
        # Once finished or terminated, gather output.
        stdout, stderr = process.communicate()
        if process.returncode != 0:
            print(f"Script failed with return code {process.returncode}: {stderr}")
        else:
            print(stdout)
    except Exception as e:
        print(f"Error running script: {e}")



import atexit
def cleanup():
    print("Running cleanup code...")
    keep_going = False 
    dm.zero_all()
    dm.activate_calibrated_flat()

    
# Register the cleanup function to be called on normal program termination.
atexit.register(cleanup)


with open(args.toml_file.replace('#',f'{beam_id}'), "r") as f:

    config_dict = toml.load(f)
    
    # Baldr pupils from global frame 
    baldr_pupils = config_dict['baldr_pupils']
    I2A = np.array( config_dict[f'beam{beam_id}']['I2A'] )
    
    # image pixel filters
    pupil_mask = np.array(config_dict.get(f"beam{beam_id}", {}).get("pupil_mask", {}).get("mask", None) ).astype(bool)   # matrix bool
    exter_mask = np.array(config_dict.get(f"beam{beam_id}", {}).get("pupil_mask", {}).get("exterior", None) ).astype(bool) # matrix bool
    secon_mask = np.array(config_dict.get(f"beam{beam_id}", {}).get("pupil_mask", {}).get("secondary", None) ).astype(bool) # matrix bool

    # ctrl model 
    IM = np.array(config_dict.get(f"beam{beam_id}", {}).get(f"{args.phasemask}", {}).get("ctrl_model", None).get("IM", None) ).astype(float)
    I2M_raw = np.array(config_dict.get(f"beam{beam_id}", {}).get(f"{args.phasemask}", {}).get("ctrl_model", None).get("I2M", None) ).astype(float)
    I2M_LO_raw = np.array(config_dict.get(f"beam{beam_id}", {}).get(f"{args.phasemask}", {}).get("ctrl_model", None).get("I2M_LO", None) ).astype(float)
    I2M_HO_raw = np.array(config_dict.get(f"beam{beam_id}", {}).get(f"{args.phasemask}", {}).get("ctrl_model", None).get("I2M_HO", None) ).astype(float)
    M2C = np.array(config_dict.get(f"beam{beam_id}", {}).get(f"{args.phasemask}", {}).get("ctrl_model", None).get("M2C", None) ).astype(float)
    M2C_LO = np.array(config_dict.get(f"beam{beam_id}", {}).get(f"{args.phasemask}", {}).get("ctrl_model", None).get("M2C_LO", None) ).astype(float)
    M2C_HO = np.array(config_dict.get(f"beam{beam_id}", {}).get(f"{args.phasemask}", {}).get("ctrl_model", None).get("M2C_HO", None) ).astype(float)
    I0 = np.array(config_dict.get(f"beam{beam_id}", {}).get(f"{args.phasemask}", {}).get("ctrl_model", None).get("I0", None) ).astype(float)
    N0 = np.array(config_dict.get(f"beam{beam_id}", {}).get(f"{args.phasemask}", {}).get("ctrl_model", None).get("N0", None) ).astype(float)
    N0i = np.array(config_dict.get(f"beam{beam_id}", {}).get(f"{args.phasemask}", {}).get("ctrl_model", None).get("norm_pupil", None) ).astype(float)
    
    # used to normalize exterior and bad pixels in N0 (calculation of N0i)
    inside_edge_filt = np.array(config_dict.get(f"beam{beam_id}", {}).get(f"{args.phasemask}", {}).get("ctrl_model", None).get("inner_pupil_filt", None) )#.astype(bool)
    
    # reduction products
    IM_cam_config = config_dict.get(f"beam{beam_id}", {}).get(f"{args.phasemask}", {}).get("ctrl_model", None).get("camera_config", None) # dictionary
    
    bad_pixel_mask = np.array( config_dict.get(f"beam{beam_id}", {}).get(f"{args.phasemask}", {}).get("ctrl_model", None).get("bad_pixel_mask", None) )#.astype(bool)
    bias = np.array( config_dict.get(f"beam{beam_id}", {}).get(f"{args.phasemask}", {}).get("ctrl_model", None).get("bias", None) ).astype(float)
    dark = np.array( config_dict.get(f"beam{beam_id}", {}).get(f"{args.phasemask}", {}).get("ctrl_model", None).get("dark", None) ).astype(float)

    dm_flat = np.array( config_dict.get(f"beam{beam_id}", {}).get(f"{args.phasemask}", {}).get("ctrl_model", None).get("DM_flat", None) )

    strehl_coe_sec = np.array( config_dict.get(f"beam{beam_id}", {}).get("strehl_model", None).get("secondary",None) ).astype(float) #coe, intercept (on central secondary pixel, ADU/s/gain)
    strehl_coe_ext = np.array( config_dict.get(f"beam{beam_id}", {}).get("strehl_model", None).get("exterior",None) ).astype(float)#coe, intercept (on median signal, ADU/s/gain)


# Start running turbulence.py 

#python common/turbulence.py --number_of_modes_removed 0 --r0 0.2 --V 0.4 --number_of_iterations 2000

# # --- Build the Command Argument List ---
# turb_args = [
#     "--number_of_iterations", f"{args.number_of_turb_iterations}",
#     "--record_telem", f"{args.record_turb_telem}",
#     "--wvl", f"{args.wvl}",
#     "--D_tel", f"{args.D_tel}",
#     "--r0", f"{args.r0}",
#     "--V", f"{args.V}",
#     "--number_of_modes_removed", f"{args.number_of_modes_removed}",
#     "--DM_chn", f"{args.DM_chn}",
# ]


# cmd = ["python", "common/turbulence.py"] + ["--beam_id", f"{beam_id}"] + turb_args
# # Create an Event to signal stopping the thread.
# stop_event = threading.Event()

# print(f"putting turbulence on DM channel {args.DM_chn}")
# #run_script(cmd) #<- Use this in a thread! 
# thread = threading.Thread(target=run_script, args=(cmd, stop_event, 10))
# thread.start()

# Let the thread run for some time, then signal it to stop.
# time.sleep(5)  # Allow script to run for 5 seconds.
# stop_event.set()

# # Optionally, join the thread to wait for it to finish.
# thread.join()
#print("Thread terminated safely.")


#bad_pixel_mask = bad_pixel_mask == "True"
# Camera 
c = FLI.fli(args.global_camera_shm, roi = baldr_pupils[f'{beam_id}'])

#cam_config = c.get_camera_config()

gain = float( c.config["gain"] ) 
fps = float( c.config["fps"] ) 
print(f"gain = {gain}, fps = {fps}")

# settings when building IM (these were the ones used to normalize frames such as I0,)
# gain0 = float( IM_cam_config["gain"] ) 
# fps0 = float( IM_cam_config["fps"] ) 



#util.nice_heatmap_subplots( [util.get_DM_command_in_2D( I2M @ IM[65])] , savefig='delme.png')

#plt.figure(); plt.imshow( util.get_DM_command_in_2D( dmtight_filt ) ) ;plt.savefig('delme.png')
#np.sum(dmtight_filt)

# DM 
dm = dmclass( beam_id=beam_id )
#dm.zero_all()

dm.activate_calibrated_flat()
if dm_flat == 'baldr':
    dm.activate_calibrated_flat()
elif dm_flat == 'factory':
    dm.activate_flat()
#else:
#    raise UserWarning("dm_flat must be baldr or factory")


# def init_pyRTC():

# Normalize control matricies by current gain and fps 
I2M = gain / fps * I2M_raw 
I2M_LO = gain / fps * I2M_LO_raw
I2M_HO = gain / fps * I2M_HO_raw 

# project reference intensities to DM (quicker for division & subtraction)
N0dm = gain / fps * (I2A @ N0i.reshape(-1)) # these are already reduced #- dark_dm - bias_dm
I0dm = gain / fps * (I2A @ I0.reshape(-1)) # these are already reduced  #- dark_dm - bias_dm
bias_dm = I2A @ bias.reshape(-1)
dark_dm = I2A @ dark.reshape(-1)
badpixmap = I2A @ bad_pixel_mask.astype(int).reshape(-1)

# reduction products on secondary pixels 
bias_sec = bias[secon_mask.astype(bool).reshape(-1)][4]
dark_sec = dark[secon_mask.astype(bool).reshape(-1)][4]

#util.nice_heatmap_subplots( im_list = [util.get_DM_command_in_2D(a) for a in [N0dm, I0dm, bias_dm, dark_dm, badpixmap]] , savefig='delme.png') 

#util.nice_heatmap_subplots( im_list = [M2C_LO.T[0].reshape(12,12), M2C_LO.T[1].reshape(12,12), M2C_HO.T[65].reshape(12,12)] , savefig='delme.png') 

#util.nice_heatmap_subplots( im_list = [util.get_DM_command_in_2D(a) for a in [I2M_LO[0],I2M_LO[1],I2M_HO[65]]] , savefig='delme.png') 


######################################
# ZONAL - Multiple Actuators 
######################################

###########################################
dmtight_mask = I2A @ np.array([int(a) for a in inside_edge_filt])
#dm_mask = I2A @ np.array( pupil_mask ).reshape(-1)

# doing a tight filter (~44 modes)
# testing doing this in build_baldr_control_matrix.py
#I2M = dmtight_mask[:,np.newaxis] * I2M
#I2M_HO = dmtight_mask[:,np.newaxis] * I2M_HO
###########################################


#util.nice_heatmap_subplots( [ util.get_DM_command_in_2D( dm_mask) ] , savefig='delme.png')

# control matrix (zonal) - normalized by current gain and fps
# D = np.diag( gain / fps * np.array( [dm_mask[i]/IM[i][i] if np.isfinite(1/IM[i][i]) else 0 for i in range(len(IM))]) )

#util.nice_heatmap_subplots( [ D ] , savefig='delme.png')

dm2opd = 7000 # nm / DM cmd

# Telemetry 
telem = init_telem_dict() #)

# PID Controller (this can be another toml)
N_HO = np.array(I2M_HO).shape[0]
kp = args.kp_HO * np.ones( N_HO)
ki = args.ki_HO * np.ones( N_HO )
kd = args.kd_HO * np.ones( N_HO )
setpoint = np.zeros( N_HO )
lower_limit_pid = -1 * np.ones( N_HO )
upper_limit_pid = 1 * np.ones( N_HO )

ctrl_HO = PIDController(kp, ki, kd, upper_limit_pid, lower_limit_pid, setpoint)

N_LO = np.array(I2M_LO).shape[0]
kp = args.kp_LO * np.ones( N_LO)
ki =  args.ki_LO * np.ones( N_LO )
kd = args.kd_LO * np.ones( N_LO )
setpoint = np.zeros( N_LO )
lower_limit_pid = -0.4 * np.ones( N_LO )
upper_limit_pid = 0.4 * np.ones( N_LO )

ctrl_LO = PIDController(kp, ki, kd, upper_limit_pid, lower_limit_pid, setpoint)

close_after = 0

##strehl model i_s = coe0 * rms + coe1 
# coe_0 = (gain / fps * strehl_coe_ext[0] )
# coe_1 = (gain / fps * strehl_coe_ext[1])

#u = 0
#bad_ones = [ 26,  37,  38,  53,  63,  65,  66,  75,  76,  78,  79,  87,  90, 98, 101, 102]
telem = False #init_telem_dict() #False
naughty_list = {a:0 for a in range(140)}

keep_going = True
it = 0

#         return {
#         'N0dm': N0dm,
#         'I0dm': I0dm,
#         'bias_dm': bias_dm,
#         'dark_dm': dark_dm,
#         'ctrl_HO': ctrl_HO,
#         'ctrl_LO': ctrl_LO,
#     }


# init_pyRTC() 


loop_speed = 0.004 # s (250 Hz)
c.mySHM.catch_up_with_sem(c.semid)
dm.shm0.catch_up_with_sem(-1)
#for it in range(args.number_of_iterations):
while keep_going:   
    #time.sleep(0.005) # with 1kHz


    #for it in range(args.number_of_iterations):
    # raw intensity 
    #time.sleep(0.1)
    
    i = c.get_image(apply_manual_reduction=False) # we don't reduce in pixel space, but rather DM space to reduce number of operations 
    t0 = time.time()

    # model of the turbulence (in DM units)
    #sss = (i[secon_mask.astype(bool)][4] - bias_sec - (1/ fps * dark_sec) )
    #dm_rms_est =  gain / fps * np.sum( strehl_coe_ext @ [sss, 1] )
    #Sest = np.exp( - (2 * np.pi * dm2opd * dm_rms_est / (1.65e-6) )**2)

    #print( Sest)

    # go to dm space subtracting dark (ADU/s) and bias (ADU) there
    idm = (I2A @ i.reshape(-1))  - gain / float(IM_cam_config["gain"]) * 1/fps * dark_dm - bias_dm

    # adu normalized pupil signal 
    s =  ( idm - I0dm ) / (N0dm)   # 

    # error
    e_LO = I2M_LO @ s 
    e_HO = I2M_HO @ s 

    # ctrl 
    u_LO = ctrl_LO.process( e_LO )
    u_HO = ctrl_HO.process( e_HO )
    # if it > close_after:
    #     u = ctrl_HO.process( e )
    # else:
    #     u = 0 * e 

    #u[bad_ones] = 0

    if np.max( abs( e_LO ) ) > 1:
        print("LO going bad - flatten")
        dm.set_data( dm.cmd_2_map2D( np.zeros( len(u_HO)) ) )
        #dm.activate_calibrated_flat()
        ctrl_LO.reset( )

    # safety
    if np.max( abs( u_HO ) ) > 0.3:
        culprit = np.where( abs( u_HO ) == np.max( abs( u_HO ) )  )[0][0]
        print(f"beam {args.beam_id} broke by act.{culprit} , reseting") # , reducing gain act {culprit} by half")
        #ctrl_HO.ki[culprit] *= 0.5 
        dm.set_data( dm.cmd_2_map2D( np.zeros( len(u_HO)) ) )
        #dm.activate_calibrated_flat()
        ctrl_HO.reset( )

        #ctrl_HO.reset_single_mode(culprit )
        naughty_list[culprit] += 1
        #if naughty_list[culprit] > 5:
        #    #keep_going = False
        #    #print("ENDING")

        if naughty_list[culprit] > 30:
            keep_going = False
            print("ENDING")

        
        if naughty_list[culprit] > 4:
           print(f"turn off naughty atuator {culprit}")
           ctrl_HO.ki[culprit] *= 0.0 # turn off gain for this actuator
        #break
        # ctrl_HO.reset_single_mode(

    c_LO = -1* M2C_LO @ u_LO
    c_HO = -1 * M2C_HO @ u_HO

    #util.nice_heatmap_subplots( im_list=[( M2C_LO.T[0] *0.01).reshape(12,12), c_LO.reshape(12,12) ], savefig='delme.png')
    #u -= np.mean( u ) # Forcefully remove piston! 
    # reconstruction

    dcmd = c_HO + c_LO
    #dcmd = -1 *  dm.cmd_2_map2D( u_HO ) 
    t1 = time.time()

    dm.set_data( dcmd )


    if telem:
        telem["time_cam"].append( t0 )
        telem["time_dm"].append( t1 )
        telem["i"].append( i.copy() )
        telem["e_TT"].append( e_LO.copy() )
        telem["u_TT"].append( u_LO.copy() )
        telem["c_TT"].append( c_LO.copy() )
        telem["e_HO"].append( e_HO.copy() )
        telem["u_HO"].append( u_HO.copy() )
        telem["c_HO"].append( c_HO.copy() )

        #telem["current_dm_ch0"].append( dm.shms[0].get_data() ) 
        telem["current_dm_ch1"].append( dm.shms[1].get_data().copy() ) 
        telem["current_dm_ch2"].append( dm.shms[2].get_data().copy() ) 
        telem["current_dm_ch3"].append( dm.shms[3].get_data().copy() ) 
        telem["exterior_sig"].append( i[secon_mask.astype(bool)].copy() )
        telem["secondary_sig"].append( i[secon_mask.astype(bool)].copy() )

    t1 = time.time()

    #print(it, t1-t0, np.max(abs(e_HO)), np.max(abs(u_HO))) #m_rms_est ,
    
    #if 1/fps - (t1-t0) > 0 : # 1kHz
    #    time.sleep( 1/fps - (t1-t0) )
    if loop_speed - (t1-t0) > 0 :
        time.sleep( loop_speed - (t1-t0) )
    
    #print(it,u_LO.max(),u_HO.max())
    it += 1


ctrl_LO.reset( )
ctrl_HO.reset( )
dm.zero_all()
dm.activate_calibrated_flat()





# ######################################
# # ANALYSIS
# ######################################

# # image_lists = [ telem["i"] , telem["current_dm_ch2"], telem["current_dm_ch3"]]
# # plot_titles = ["intensity","disturbance", "reconstruction" ]
# # cbar_labels = ["ADU", "DM UNITS", "DM UNITS"]
# # util.display_images_with_slider(image_lists, plot_titles=plot_titles, cbar_labels=cbar_labels)

# fig, ax = plt.subplots( 1,4, figsize=(12,5) )
# ax[0].plot( [np.nanstd( tt) for tt in telem["current_dm_ch2"]  ], label='ch2')
# ax[1].plot( [np.std( tt) for tt in telem["current_dm_ch3"]  ], label='ch3')
# ax[2].plot( [  tt[secon_mask.astype(bool)][4]  for tt in telem["i"]  ], label='secondary')
# #ax[3].plot( [np.mean( tt )  for tt in telem["exterior_sig"]  ], label='exterior')


# fig, ax = plt.subplots( 1,4, figsize=(12,5) )
# ax[0].plot( [ tt[8,8] for tt in telem["current_dm_ch2"]  ], label='ch2')
# ax[1].plot( [tt[8,8] for tt in telem["current_dm_ch3"]  ], label='ch3')
# plt.legend()
# plt.savefig('delme.png')

# telem["secondary_sig"]



# if telem:
#     # save telemetry
#     runn=f"static_fps-{c.config['fps']}_gain-fps-{c.config['gain']}" 
#     # Create a list of HDUs (Header Data Units)
#     hdul = fits.HDUList()

#     hdu = fits.ImageHDU(IM)
#     hdu.header['EXTNAME'] = 'IM'
#     hdul.append(hdu)

#     # hdu = fits.ImageHDU(M2C)
#     # hdu.header['EXTNAME'] = 'M2C'
#     # hdul.append(hdu)


#     hdu = fits.ImageHDU(I2M)
#     hdu.header['EXTNAME'] = 'I2M'
#     hdul.append(hdu)

#     hdu = fits.ImageHDU(I2A)
#     hdu.header['EXTNAME'] = 'interpMatrix'
#     hdul.append(hdu)


#     hdu = fits.ImageHDU(dm.shms[0].get_data())
#     hdu.header['EXTNAME'] = 'DM_FLAT_OFFSET'
#     hdul.append(hdu)

#     hdu = fits.ImageHDU(ctrl_HO.kp)
#     hdu.header['EXTNAME'] = 'Kp'
#     hdul.append(hdu)

#     hdu = fits.ImageHDU(ctrl_HO.ki)
#     hdu.header['EXTNAME'] = 'Ki'
#     hdul.append(hdu)

#     hdu = fits.ImageHDU(ctrl_HO.kd)
#     hdu.header['EXTNAME'] = 'Kd'
#     hdul.append(hdu)

#     hdu = fits.ImageHDU(ctrl_HO.kd)
#     hdu.header['EXTNAME'] = 'Kd'
#     hdul.append(hdu)

#     hdu = fits.ImageHDU(pupil_mask.astype(int))
#     hdu.header['EXTNAME'] = 'ext'
#     hdul.append(hdu)
#     # Add each list to the HDU list as a new extension
#     for list_name, data_list in telem.items() :##zip(["time","i","err", "reco", "disturb", "secondary_sig"] ,[   telem["time"], telem["i"],telem["e_HO"], telem["current_dm_ch2"],telem["current_dm_ch3"], telem["secondary_sig"]] ) : # telem.items():
#         # Convert list to numpy array for FITS compatibility
#         data_array = np.array(data_list, dtype=float)  # Ensure it is a float array or any appropriate type

#         # Create a new ImageHDU with the data
#         hdu = fits.ImageHDU(data_array)

#         # Set the EXTNAME header to the variable name
#         hdu.header['EXTNAME'] = list_name

#         # Append the HDU to the HDU list
#         hdul.append(hdu)


#     # Write the HDU list to a FITS file

#     tele_pth = args.folder_pth 
#     if not os.path.exists( tele_pth ):
#         os.makedirs( tele_pth )

#     fits_file = tele_pth + f'CL_beam{beam_id}_mask{args.phasemask}_{runn}.fits' #_{args.phasemask}.fits'
#     hdul.writeto(fits_file, overwrite=True)
#     print(f'wrote telemetry to \n{fits_file}')







# ######################################
# # ZONAL - 1 actuator 
# ######################################

# # # control matrix 
# # gain = float( cam_config["gain"] ) 
# # fps = float( cam_config["fps"] ) 

# # dm_mask = I2A @  np.array( pupil_mask ).reshape(-1)

# # act = 65
# # dm_mask *= 0
# # dm_mask[act] = 1

# # # control matrix (zonal) - normalized by current gain and fps
# # D = np.diag( gain / fps * np.array( [dm_mask[i]/IM[i][i] if np.isfinite(1/IM[i][i]) else 0 for i in range(len(IM))]) )

# # # 9 ms seems to be fastest I can do 
# # method = 'zonal'

# # ### PUT STATIC ABERRATION 
# # pp = 0.08
# # modal_basis = np.array([dm.cmd_2_map2D(ii) for ii in np.eye(140)]) 
# # abb = pp * modal_basis[act] 
# # dm.shms[3].set_data(  abb )  # static aberration on channel 3
# # dm.shm0.post_sems(1)

# # # Telemetry 
# # telem = SimpleNamespace( **init_telem_dict() )

# # # PID Controller (this can be another toml)
# # N = I2M.shape[1]
# # kp = 0. * np.ones( N)
# # ki = 0.2 * np.ones( N )
# # kd = 0. * np.ones( N )
# # setpoint = np.zeros( N )
# # lower_limit_pid = -100 * np.ones( N )
# # upper_limit_pid = 100 * np.ones( N )

# # ctrl_HO = PIDController(kp, ki, kd, upper_limit_pid, lower_limit_pid, setpoint)
# # #u = 0
# # for it in range(1000):
# #     time.sleep( 1.5/fps )
# #     t0 = time.time()
# #     # raw intensity 
# #     i = c.get_image(apply_manual_reduction=False) # we don't reduce in pixel space, but rather DM space to reduce number of operations 

# #     # go to dm space subtracting dark (ADU/s) and bias (ADU) there
# #     idm = (I2A @ i.reshape(-1))  - 1/fps * dark_dm - bias_dm

# #     # adu normalized pupil signal 
# #     s =  ( idm - I0dm ) / N0dm   # 

# #     # error
# #     e = D @ s 

# #     # ctrl 
# #     u = ctrl_HO.process( e )

# #     # safety
# #     if np.max( abs( u ) ) > 0.4:
# #         print("broke")
# #         dm.zero_all()
# #         dm.activate_calibrated_flat()
# #         break

# #     # reconstruction
# #     dcmd = -1 *  dm.cmd_2_map2D(u) ### DOUBLE CHECK THIS

# #     # update dm 
# #     dm.set_data( dcmd )

# #     t1 = time.time()
# #     print(t1-t0, e[act], u[act])

# # dm.zero_all()
# # dm.activate_calibrated_flat()

# # #util.nice_heatmap_subplots( [dcmd - abb ] , savefig='delme.png')


