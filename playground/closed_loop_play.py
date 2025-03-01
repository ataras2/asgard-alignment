

import numpy as np 
import matplotlib.pyplot as plt  
from types import SimpleNamespace
import argparse
import os 
import datetime
import zmq
import toml
import time 
from astropy.io import fits
from scipy.signal import TransferFunction, bode
from xaosim.shmlib import shm
from asgard_alignment.DM_shm_ctrl import dmclass
import common.DM_basis_functions as dmbases
from common import phasemask_centering_tool as pct
import pyBaldr.utilities as util 
try:
    from asgard_alignment import controllino as co
    myco = co.Controllino('172.16.8.200')
    controllino_available = True
    print('controllino connected')
    
except:
    print('WARNING Controllino cannot connect. WILL NOT MOVE SOURCE OUT FOR DARK')
    controllino_available = False 




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
        "current_dm":[] # the current DM cmd (sum of all channels)
        # "atm_disturb_list" : [],
        # "dm_disturb_list" : [],
        # "rmse_list" : [],
        # "flux_outside_pupil_list" : [],
        # "residual_list" : [],
        # "field_phase" : [],
        # "strehl": []
    }
    return telemetry_dict



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



def process_signal( i, I0, N0):
    # must be same as model cal. import from common module
    # i is intensity, I0 reference intensity (zwfs in), N0 clear pupil (zwfs out)
    return ( i - I0 ) / N0 

tstamp = datetime.datetime.now().strftime("%d-%m-%YT%H.%M.%S")
tstamp_rough =  datetime.datetime.now().strftime("%d-%m-%Y")


# setting up socket to ZMQ communication to multi device server
parser = argparse.ArgumentParser(description="Baldr Pupil Fit Configuration.")

state_dict = {} # zmq states 

default_toml = os.path.join( "config_files", "baldr_config_#.toml") #os.path.dirname(os.path.abspath(__file__)), "..", "config_files", "baldr_config.toml")

# setting up socket to ZMQ communication to multi device server
parser.add_argument("--host", type=str, default="localhost", help="Server host")
parser.add_argument("--port", type=int, default=5555, help="Server port")
parser.add_argument(
    "--timeout", type=int, default=5000, help="Response timeout in milliseconds"
)

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
    help="TOML file to write/edit. Default: ../config_files/baldr_config.toml (relative to script)"
)

# Beam ids: provided as a comma-separated string and converted to a list of ints.
parser.add_argument(
    "--beam_id",
    type=lambda s: [int(item) for item in s.split(",")],
    default=[2],
    help="Comma-separated beam IDs to apply. Default: 1,2,3,4"
)

parser.add_argument(
    "--phasemask",
    type=str,
    default="H5", #### should update this 
    help="select which phasemask we're calibrating the model for. Default: None - in which case the user is prompted to enter the phasemask"
)


args=parser.parse_args()

pupil_mask = {}
exterior_mask = {}
dm_flat_offsets = {}
I2M = {} # intensity 2 modes 
reconstructor_model = {} # model name instructs how to apply M2C
M2C = {} # mode 2 command
I0 = {} # zwfs pupil
N0 = {} # clear pupil
dm_act_filt = {} # filter for registered DM actuators (that have response in WFS)
res = {}
for beam_id in args.beam_id:

    # read in TOML as dictionary for config 
    with open(args.toml_file.replace('#',f'{beam_id}'), "r") as f:
        config_dict = toml.load(f)

        # Baldr pupils from global frame 
        baldr_pupils = config_dict['baldr_pupils']
        # boolean filter of the registered (clear) pupil
        # for each local (cropped) baldr pupil frame 
        pupil_mask[beam_id] = config_dict[f'beam{beam_id}']["pupil_mask"] 
        # boolean filter for outside pupil where we see scattered
        # light with phase mask in (basic Strehl proxy)
        exterior_mask[beam_id] =  config_dict[f'beam{beam_id}']["pupil_mask"] 
        # flat offset for DMs (mainly focus)
        dm_flat_offsets[beam_id] = config_dict[f'beam{beam_id}']["DM_flat_offset"] 
        # intensities interpolation matrix to registered DM actuator (from local pupil frame)
        I2M[beam_id] = np.array( config_dict[f'beam{beam_id}']['I2M'] )
        #reco_dict[beam_id] = config_dict[f'beam{beam_id}']['reconstructor_model']
        reconstructor_model[beam_id] = config_dict[f'beam{beam_id}']['reconstructor_model'][f'{args.phasemask}']['reconstructor_model'] 
        M2C[beam_id] = np.array( config_dict[f'beam{beam_id}']['reconstructor_model'][f'{args.phasemask}']['M2C'] ) #[ coes, interc]
        N0[beam_id] = np.array( config_dict[f'beam{beam_id}']['reconstructor_model'][f'{args.phasemask}']['I0'] ).astype(float)
        I0[beam_id] = np.array( config_dict[f'beam{beam_id}']['reconstructor_model'][f'{args.phasemask}']['N0'] ).astype(float)
        dm_act_filt[beam_id] = np.array( config_dict[f'beam{beam_id}']['reconstructor_model'][f'{args.phasemask}']['linear_model_dm_actuator_filter'] )

        res[beam_id] = np.array( config_dict[f'beam{beam_id}']['reconstructor_model'][f'{args.phasemask}']["linear_model_train_residuals"] )


#####
##### --- manually set up camera settings before hand
#####
print( 'You should manually set up camera settings before hand')

# Set up global camera frame SHM 
c = shm(args.global_camera_shm)

# set up DM SHMs 
dm_shm_dict = {}
for beam_id in args.beam_id:
    dm_shm_dict[beam_id] = dmclass( beam_id=beam_id )
    # zero all channels
    dm_shm_dict[beam_id].zero_all()
    # activate flat (does this on channel 1)
    dm_shm_dict[beam_id].activate_flat()
    # apply dm flat offset (does this on channel 2)
    #dm_shm_dict[beam_id].set_data( np.array( dm_flat_offsets[beam_id] ) )


# Get Darks 
if controllino_available:
    
    myco.turn_off("SBB")
    time.sleep(10)
    
    dark_raw = c.get_data()

    myco.turn_on("SBB")
    time.sleep(10)

    #bad_pixel_mask = get_bad_pixel_indicies( dark_raw, std_threshold = 20, mean_threshold=6)
else:
    dark_raw = c.get_data()

    #bad_pixel_mask = get_bad_pixel_indicies( dark_raw, std_threshold = 20, mean_threshold=6)



# # check phasemask alignment 
# beam = int( input( "do you want to check the phasemasks for a beam. Enter beam number (1,2,3,4) or 0 to continue") )
# while beam :
#     print( 'we save images as delme.png in asgard-alignment project - open it!')
#     img = np.sum( c.get_data()  , axis = 0 ) 
#     r1,r2,c1,c2 = baldr_pupils[str(beam)]
#     #print( r1,r2,c1,c2  )
#     plt.figure(); plt.imshow( np.log10( img[r1:r2,c1:c2] ) ) ; plt.colorbar(); plt.savefig('delme.png')

#     # time.sleep(5)

#     # manual centering 
#     move_relative_and_get_image(cam=c, beam=beam, baldr_pupils = baldr_pupils, phasemask=state_dict["socket"],  savefigName='delme.png', use_multideviceserver=True, roi=[r1,r2,c1,c2 ])

#     beam = int( input( "do you want to check the phasemask alignment for a particular beam. Enter beam number (1,2,3,4) or 0 to continue") )




##### CURRENTLY ONLY DEALING WITH ONE BEAM AT A TIME. 


beam_id = 2 


r1,r2,c1,c2 = baldr_pupils[f"{beam_id}"]

disturbance = np.zeros( 140 )


# reference intensities interpolated onto registered DM actuators in pixel space
I0_dm = I2M[beam_id] @ I0[beam_id].reshape(-1) #np.array( config['interpolated_I0'] )
N0_dm = I2M[beam_id] @ N0[beam_id].reshape(-1) #np.array( config['interpolated_N0'] )

# Control model
if reconstructor_model[beam_id] == 'zonal_linear':
    slopes = np.array( [ m[0] for m in M2C[beam_id].T] )
    intercepts = np.array( [ m[1] for m in M2C[beam_id].T] )
else:
    raise UserWarning(f"reconstructor_model:{reconstructor_model} undefined. Try, for example zonal_linear")

# Controller
N = 140 
kp = 0. * np.ones( N)
ki = 0.3 * np.ones( N )
kd = 0. * np.ones( N )
setpoint = np.zeros( N )
lower_limit_pid = -100 * np.ones( N )
upper_limit_pid = 100 * np.ones( N )

ctrl_HO = PIDController(kp, ki, kd, upper_limit_pid, lower_limit_pid, setpoint)


record_telemetry = True
telem = SimpleNamespace( **init_telem_dict() )

closed = True 
cnt = 0
delta_cmd= np.zeros( 140 )
while closed and (cnt < 200):
    print(cnt)
    time.sleep(3) # sleep for 

    # Crop the pupil region
    cropped_image = np.mean( c.get_data(), axis = 0)[r1:r2, c1:c2]

    # (2) interpolate intensities to DM 
    i_dm = I2M[beam_id] @ cropped_image.reshape(-1)

    # (3) normalise 
    # current model has no normalization 
    sig = process_signal( i_dm, I0_dm, N0_dm)
    
    # (4) apply linear model to get reconstructor 
    e_HO = slopes * sig + intercepts

    # PID 
    u_HO = ctrl_HO.process( e_HO )
    
    # forcefully remove piston 
    u_HO -= np.mean( u_HO )
    
    # send command (filter piston)
    #delta_cmd = np.zeros( len(zwfs_ns.dm.dm_flat ) )
    #delta_cmd zwfs_ns.reco.linear_zonal_model.act_filt_recommended ] = u_HO
    delta_cmd[ dm_act_filt[beam_id] ] = u_HO[ dm_act_filt[beam_id]  ]

    cmd = -delta_cmd #disturbance - delta_cmd 

    # record telemetry with the new image and processed signals but the current DM commands (not updated)
    if record_telemetry :
        telem.i_list.append( cropped_image )
        telem.i_dm_list.append( i_dm )
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
        telem.current_dm.append( dm_shm_dict[beam_id].shms0.get_data() )

    if np.std( delta_cmd ) > 0.15:
        print('going bad')
        dm_shm_dict[beam_id].zero_all()
        dm_shm_dict[beam_id].activate_flat()
        closed = False


    # reformat for SHM 
    cmd_shm = np.nan_to_num( dm_shm_dict[beam_id].cmd_2_map2D( cmd ) ) 
    
    #send the command off 
    dm_shm_dict[beam_id].set_data( cmd_shm ) # on Channel 2 

    cnt+=1


dm_shm_dict[beam_id].zero_all()
dm_shm_dict[beam_id].activate_flat()


# save telemetry

# Create a list of HDUs (Header Data Units)
hdul = fits.HDUList()

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
fits_file = '/home/asg/Videos/' + f'CL_{beam_id}.fits' #_{args.phasemask}.fits'
hdul.writeto(fits_file, overwrite=True)

print('wrote telemetry to \n{fits_file}')