import numpy as np 
import zmq
import time
import toml
import os 
import argparse
import matplotlib.pyplot as plt
import argparse
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
from xaosim.shmlib import shm


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

args=parser.parse_args()

c, dms, darks_dict, I0_dict, N0_dict,  baldr_pupils, I2A = setup(args.beam_id,
                              args.global_camera_shm, 
                              args.toml_file) 


#############
beam_id = 2 
############

r1,r2,c1,c2 = baldr_pupils[f'{beam_id}']

dark = np.mean( darks_dict[beam_id],axis=0)
I0 = np.mean(I0_dict[beam_id],axis=0)
N0 = np.mean(N0_dict[beam_id],axis=0)
interpMatrix = I2A[beam_id]
# checks 
cbar_label_list = ['[adu]','[adu]', '[adu]']
title_list = ['DARK','I0', 'N0']
xlabel_list = ['','','']
ylabel_list = ['','','']



util.nice_heatmap_subplots( im_list = [  dark, I0 - dark, N0-dark ] , 
                            cbar_label_list = cbar_label_list,
                            title_list=title_list,
                            xlabel_list=xlabel_list,
                            ylabel_list=ylabel_list,
                            savefig='delme.png' )


Nmodes = 30
modal_basis = dmbases.zer_bank(2, Nmodes+2 )
M2C = modal_basis.copy() # mode 2 command matrix 
poke_amp = 0.02
inverse_method = 'pinv'
phase_cov = np.eye( 140 ) #np.array(IM).shape[0] )
noise_cov = np.eye( Nmodes ) #np.array(IM).shape[1] )

I0_dm = interpMatrix @ I0.reshape(-1)
N0_dm = interpMatrix @ N0.reshape(-1)


IM = []
for i,m in enumerate(modal_basis):
    print(f'executing cmd {i}/{len(modal_basis)}')
    I_plus_list = []
    I_minus_list = []
    imgs_to_mean = 10
    for sign in [(-1)**n for n in range(10)]: #[-1,1]:
        dms[beam_id].set_data(  sign * poke_amp/2 * m ) 
        time.sleep(2)
        if sign > 0:
            I_plus_list.append( list( np.mean( c.get_data( ),axis = 0)[r1:r2,c1:c2]  ) )
            #I_plus *= 1/np.mean( I_plus )
        if sign < 0:
            I_minus_list.append( list( np.mean( c.get_data( ),axis = 0)[r1:r2,c1:c2]  ) )
            #I_minus *= 1/np.mean( I_minus )

    I_plus = np.mean( I_plus_list, axis = 0).reshape(-1) / N0.reshape(-1)
    I_minus = np.mean( I_minus_list, axis = 0).reshape(-1) /  N0.reshape(-1)

    errsig = interpMatrix @ (I_plus - I_minus) / poke_amp

    IM.append( list(  errsig.reshape(-1) ) ) 

# intensity to mode matrix 
if inverse_method == 'pinv':
    I2M = np.linalg.pinv( IM )

elif inverse_method == 'MAP': # minimum variance of maximum posterior estimator 
    I2M = (phase_cov @ IM @ np.linalg.inv(IM.T @ phase_cov @ IM + noise_cov) ).T #have to transpose to keep convention.. although should be other way round


dms[beam_id].zero_all()
dms[beam_id].activate_flat()


## SOME CHECKS 
print( f'condition of IM = {np.linalg.cond(IM)}')

imgs = [util.get_DM_command_in_2D( i) for i in IM ][:7]
titles = ['' for _ in imgs]
cbars = ['' for _ in imgs]
xlabel_list = ['' for _ in imgs]
ylabel_list = ['' for _ in imgs]
util.nice_heatmap_subplots( imgs ,title_list=titles,xlabel_list=xlabel_list, ylabel_list=ylabel_list, cbar_label_list=cbars, fontsize=15, cbar_orientation = 'bottom', axis_off=True, vlims=None, savefig='delme.png' )



# get some new images and project them onto the modes 
ifull = c.get_data()
i = np.array( [ii[r1:r2,c1:c2] for ii in ifull] )

sig = interpMatrix @ ((i - I0 ) / N0).reshape(-1,100)

reco = I2M.T @ sig

print( f'mean mode reco = {np.mean( reco , axis = 1)}') 
print( f'std mode reco = {np.std( reco, axis = 1 )}')

plt.figure(); plt.imshow( np.cov( IM ) / poke_amp ) ;  plt.colorbar(); plt.savefig('delme.png')


# lets look at the command recosntruction
delta_cmd = (M2C.T @ reco).T

i= 30
plt.figure() ; plt.imshow( delta_cmd[22]); plt.colorbar(); plt.savefig('delme.png')


### Apply command and see if reconstruct amplitude 
dms[beam_id].zero_all()
dms[beam_id].activate_flat()

abb = 1 * poke_amp * modal_basis[0] 
time.sleep(2)
dms[beam_id].shms[1].set_data(  abb ) 
time.sleep( 10 )

cropped_image = np.mean( c.get_data(), axis = 0)[r1:r2, c1:c2]

i_dm = interpMatrix @ cropped_image.reshape(-1)

# current model has no normalization 
sig = process_signal( i_dm, I0_dm, N0_dm )

#plt.imshow( util.get_DM_command_in_2D(sig) ); plt.savefig('delme.png')
# (4) apply linear model to get reconstructor 
e_HO = I2M.T @ sig #slopes * sig + intercepts

print(e_HO)

delta_cmd = 2 * poke_amp * (M2C.T @ e_HO).T
 
dm_filt = util.get_DM_command_in_2D( util.get_circle_DM_command(radius=5, Nx_act=12) )
imgs = [ abb ,util.get_DM_command_in_2D(sig), dm_filt * delta_cmd, dm_filt*(abb-delta_cmd) ]
titles = ['DM disturbance', 'ZWFS signal', 'DM reconstruction', 'DM residual']
cbars = ['' for _ in imgs]
xlabel_list = ['' for _ in imgs]
ylabel_list = ['' for _ in imgs]
util.nice_heatmap_subplots( imgs ,title_list=titles,xlabel_list=xlabel_list, ylabel_list=ylabel_list, cbar_label_list=cbars, fontsize=15, cbar_orientation = 'bottom', axis_off=True, vlims=None, savefig='delme.png' )



### TRY CLOSE LOOP 

dms[beam_id].zero_all()
dms[beam_id].activate_flat()

I0_dm = interpMatrix @ I0.reshape(-1)
N0_dm = interpMatrix @ N0.reshape(-1)

record_telemetry = True


#prepare phasescreen to put on DM 
D = 1.8
act_per_it = 0.2 # how many actuators does the screen pass per iteration 
V = 10 / act_per_it  / D #m/s (10 actuators across pupil on DM)
#scaling_factor = 0.2
I0_indicies = 10 # how many reference pupils do we get?

corner_indicies = [0, 11, 11 * 12, -1] # DM corner indidices

scaling_factor = 0.13
#number_of_rolls = 500



close_after = 20
ki_grid = [0.2, 0.5, 0.8, 0.9, 0.97]

for ki_v in ki_grid:
    print(f"\n\nSETTING ki={ki_v}\n\n")
    dms[beam_id].zero_all()
    time.sleep(1)
    dms[beam_id].activate_flat()

    telem = SimpleNamespace( **init_telem_dict() )

    # Controller
    N = I2M.shape[1]
    kp = 0. * np.ones( N)
    ki = 0. * np.ones( N )
    kd = 0. * np.ones( N )
    setpoint = np.zeros( N )
    lower_limit_pid = -100 * np.ones( N )
    upper_limit_pid = 100 * np.ones( N )



    ctrl_HO = PIDController(kp, ki, kd, upper_limit_pid, lower_limit_pid, setpoint)


    closed = True 
    cnt = 0
    delta_cmd= np.zeros( 140 )
    scrn = ps.PhaseScreenVonKarman(nx_size= int( 12 / act_per_it ) , 
                               pixel_scale= D / 12, 
                               r0=0.1, 
                               L0=12,
                               random_seed=1)
    while closed and (cnt < 100):

        scrn.add_row()
        # bin phase screen onto DM space 
        dm_scrn = util.create_phase_screen_cmd_for_DM(scrn,  scaling_factor=scaling_factor, drop_indicies = [0, 11, 11 * 12, -1] , plot_cmd=False)
        # update DM command 
        # put in SHM format 140 1D cmd -> 144 square 
        twoDized = np.nan_to_num( util.get_DM_command_in_2D(dm_scrn), 0 )

        # apply atm disturbance on channel 2 
        dms[beam_id].shms[2].set_data( twoDized  ) 

        print(cnt)
        time.sleep(3) # sleep for 

        # Crop the pupil region
        cropped_image = np.mean( c.get_data(), axis = 0)[r1:r2, c1:c2]

        # (2) interpolate intensities to DM 
        i_dm = interpMatrix @ cropped_image.reshape(-1)

        # (3) normalise 
        # current model has no normalization 
        sig = process_signal( i_dm, I0_dm, N0_dm)
        
        # (4) apply linear model to get reconstructor 
        e_HO = I2M.T @ sig #slopes * sig + intercepts

        print( f'e_HO = {e_HO[:10]}' )
        # turn off tip/tilt
        #e_HO[:2] = 0
        # correct first 8 HO modes 
        #e_HO[10:] = 0
        # PID 
        u_HO = ctrl_HO.process( e_HO )
        
        # forcefully remove piston 
        #u_HO -= np.mean( u_HO )
        
        # send command (filter piston)
        #delta_cmd = np.zeros( len(zwfs_ns.dm.dm_flat ) )
        #delta_cmd zwfs_ns.reco.linear_zonal_model.act_filt_recommended ] = u_HO
        # factor of 2 since reflection
        delta_cmd = 2 * poke_amp * (M2C.T @ u_HO).T  #[ dm_act_filt[beam_id] ] = u_HO[ dm_act_filt[beam_id]  ]
        # be careful with tranposes , made previouos mistake tip was tilt etc (90 degree erroneous rotation)
        #cmd = -delta_cmd #disturbance - delta_cmd 

        print( np.std( delta_cmd ) )

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

            telem.current_dm_ch0.append( dms[beam_id].shms[0].get_data() )
            telem.current_dm_ch1.append( dms[beam_id].shms[1].get_data() )
            telem.current_dm_ch2.append( dms[beam_id].shms[2].get_data() )
            telem.current_dm_ch3.append( dms[beam_id].shms[3].get_data() )
            # sum of all DM channels (Full command currently applied to DM)
            telem.current_dm.append( dms[beam_id].shm0.get_data() )

        if np.std( delta_cmd ) > 0.2:
            print('going bad')
            dms[beam_id].zero_all()
            dms[beam_id].activate_flat()
            closed = False

        # Carefull - controller integrates even so this doesnt work
        if cnt > close_after :
            ctrl_HO.ki = ki_v * np.ones(len(ki))
            # turn off tip/tilt
            #ctrl_HO.ki[:2] = 0
            # correct first 8 HO modes 
            #ki[30:] = 0
            # reformat for SHM 
            #cmd_shm = np.nan_to_num( dms[beam_id].cmd_2_map2D( cmd ) ) 
            
            #send the command off 
            # apply to ch1 so we can apply impulse aberrations on ch2 
        
        dms[beam_id].shms[1].set_data( -delta_cmd  ) #cmd_shm ) # on Channel 2 


        cnt+=1



    dms[beam_id].zero_all()
    dms[beam_id].activate_flat()

    # plt.figure() ; plt.imshow( telem.u_HO_list) ;plt.colorbar(); plt.savefig('delme.png')

    # plt.figure() ; plt.plot( [t[3] for t in telem.u_HO_list]); plt.savefig('delme.png')

    # plt.figure(); plt.imshow( cmd) ;plt.colorbar(); plt.savefig('delme.png')
    # # plt.figure(); plt.imshow( telem.e_HO_list ) ;plt.colorbar(); plt.savefig('delme.png')

    # plt.figure(); plt.imshow( telem.current_dm_ch2[-2] ) ;plt.colorbar(); plt.savefig('delme.png')



    # save telemetry
    runn=f'kol_truncate30modes_wTT_100its_{act_per_it}actdelay_ki{ki_v}_scrnScale{scaling_factor}'
    # Create a list of HDUs (Header Data Units)
    hdul = fits.HDUList()

    hdu = fits.ImageHDU(IM)
    hdu.header['EXTNAME'] = 'IM'
    hdul.append(hdu)

    hdu = fits.ImageHDU(I2M)
    hdu.header['EXTNAME'] = 'I2M'
    hdul.append(hdu)

    hdu = fits.ImageHDU(interpMatrix)
    hdu.header['EXTNAME'] = 'interpMatrix'
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
    fits_file = '/home/asg/Videos/' + f'CL_zernike_{Nmodes}modes_beam{beam_id}_v{runn}_working.fits' #_{args.phasemask}.fits'
    hdul.writeto(fits_file, overwrite=True)

    print(f'wrote telemetry to \n{fits_file}')