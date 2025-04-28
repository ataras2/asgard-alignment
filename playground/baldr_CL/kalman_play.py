
import numpy as np #(version 2.1.1 works but incompatiple with numba)
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
from scipy.stats import pearsonr
import pickle
from types import SimpleNamespace
from sklearn.linear_model import LinearRegression
import importlib # reimport package after edits: importlib.reload(bldr)
import os
import datetime
import scipy.interpolate as interpolate
from statsmodels.tsa.ar_model import AutoReg
import numpy as np
from scipy.linalg import block_diag
import time 
import toml
import argparse 
from asgard_alignment import FLI_Cameras as FLI
import common.phasescreens as ps 
import pyBaldr.utilities as util 
import common.DM_basis_functions as dmbases
from asgard_alignment.DM_shm_ctrl import dmclass



###############################################################################
# Build overall measurement matrix H_total
###############################################################################
def build_measurement_matrix(n_act, lags=20):
    """
    Build a measurement matrix H_total of shape (n_act, n_act*lags)
    that extracts the first state element (the current phase) for each actuator.
    """
    H_blocks = [np.hstack([np.eye(1), np.zeros((1, lags-1))]) for _ in range(n_act)]
    H_total = block_diag(*H_blocks)
    return H_total

###############################################################################
# Fit AR(20) model using statsmodels for each actuator
###############################################################################
def fit_AR_models(phi_history, lags=20):
    """
    Fits an AR(lags) model to the calibration time series for each actuator.
    
    Parameters:
        phi_history : numpy.ndarray of shape (N, n_act)
                      where N is the number of calibration iterations and
                      n_act is the number of DM actuators (e.g., 140).
        lags : int, the AR order (default 20).
        
    Returns:
        ar_params : numpy.ndarray of shape (n_act, lags+1)
                    Each row contains [intercept, a1, ..., a_lags] for that actuator.
        noise_var : numpy.ndarray of shape (n_act,)
                    Estimated residual variance for each actuator.
    """
    N, n_act = phi_history.shape
    ar_params = np.zeros((n_act, lags+1))
    noise_var = np.zeros(n_act)
    for i in range(n_act):
        ts = phi_history[:, i]
        # Fit AR model of specified order
        model = AutoReg(ts, lags=lags, old_names=False)
        res = model.fit()
        ar_params[i, :] = res.params  # first element is intercept, then coefficients
        noise_var[i] = res.sigma2
    return ar_params, noise_var


###############################################################################
# Build block-diagonal state-space matrices for all actuators
###############################################################################
def build_block_state_space(ar_params_all, noise_var_all, lags=20):
    """
    Given AR model parameters for each actuator, build the overall block-diagonal
    state-transition matrix A_total and process noise covariance Q_total.
    
    Parameters:
        ar_params_all : numpy.ndarray of shape (n_act, lags+1)
        noise_var_all : numpy.ndarray of shape (n_act,)
        lags : int, AR order.
    
    Returns:
        A_total : numpy.ndarray of shape (n_act*lags, n_act*lags)
        Q_total : numpy.ndarray of shape (n_act*lags, n_act*lags)
    """
    n_act = ar_params_all.shape[0]
    A_blocks = []
    Q_blocks = []
    for i in range(n_act):
        A_comp, Q_comp = build_state_space_from_AR(ar_params_all[i, :], noise_var_all[i], lags=lags)
        A_blocks.append(A_comp)
        Q_blocks.append(Q_comp)
    A_total = block_diag(*A_blocks)
    Q_total = block_diag(*Q_blocks)
    return A_total, Q_total



###############################################################################
# Kalman filter class (state-space version with block matrices)
###############################################################################
class BlockKalmanFilter:
    def __init__(self, A, Q, R, x0, P0):
        """
        A, Q, P0: state-space matrices of shape (n, n) where n = n_act * lags.
        R: measurement noise covariance of shape (n_act, n_act)
        x0: initial state (n x 1)
        """
        self.A = A
        self.Q = Q
        self.R = R
        self.x = x0
        self.P = P0
        self.K = 0

    def predict(self):
        self.x = self.A @ self.x
        self.P = self.A @ self.P @ self.A.T + self.Q
        return self.x

    def update(self, z, H):
        y = z - H @ self.x  # z is opd measurement (subtracting DM cmd estimate of opd. H just filters x for the mpst recent opd estimate (state) 
        S = H @ self.P @ H.T + self.R
        K = self.P @ H.T @ np.linalg.inv(S)
        self.K = K
        self.x = self.x + K @ y
        self.P = (np.eye(len(self.x)) - K @ H) @ self.P
        return self.x





###############################################################################
# Build companion state-space matrices for an AR(p) model (p=lags)

def build_state_space_from_AR(ar_params, noise_var, lags=20):
    """
    Constructs the companion matrix A and process noise covariance Q for one actuator
    given the AR model parameters.
    
    For an AR(p) process:
        phi[t] = mu + a1 phi[t-1] + a2 phi[t-2] + ... + ap phi[t-p] + w[t]
    The state vector is:
        x[t] = [ phi[t], phi[t-1], ..., phi[t-p+1] ]^T,
    and the companion matrix is:
        A = [ a1, a2, ..., ap ]
            [ 1,  0, ..., 0  ]
            [ 0,  1, ..., 0  ]
            ...
            [ 0,  0, ..., 1  ]
    The process noise is assumed to affect only the first state element.
    
    Parameters:
        ar_params : array-like, shape (lags+1,)
                    [intercept, a1, ..., a_p]
        noise_var : float, estimated variance of the residual (w[t])
        lags : int, AR order.
    
    Returns:
        A_comp : numpy.ndarray of shape (lags, lags)
        Q_comp : numpy.ndarray of shape (lags, lags)
    """
    A_comp = np.zeros((lags, lags))
    # Use the AR coefficients (skip intercept) in the first row.
    A_comp[0, :] = ar_params[1:]
    # Place ones on the first subdiagonal.
    if lags > 1:
        A_comp[1:, :-1] = np.eye(lags-1)
    # Process noise only enters the first state element.
    Q_comp = np.zeros((lags, lags))
    Q_comp[0, 0] = noise_var
    return A_comp, Q_comp

def remove_corners_and_flatten(arr):
    """
    Remove the four corner elements from a 12x12 array and flatten the result.

    """
    if arr.shape != (12, 12):
        raise ValueError("Input array must be of shape (12, 12)")
        
    # Create a mask that is True everywhere except at the four corners.
    mask = np.ones((12, 12), dtype=bool)
    mask[0, 0] = False        # top left
    mask[0, -1] = False       # top right
    mask[-1, 0] = False       # bottom left
    mask[-1, -1] = False      # bottom right
    
    # Flatten the array using the mask
    flattened = arr[mask]
    return flattened


def quickplot(thing, hist=False):
    plt.figure()
    if (len(np.array(thing).shape) < 2) and not hist:
        plt.plot(thing)
    elif (len(np.array(thing).shape) == 2) and not hist:
        plt.imshow(thing)  
    elif (len(np.array(thing).shape) < 2) and hist:
        plt.hist( thing )
    plt.savefig('delme.png')





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


default_toml = os.path.join("/usr/local/etc/baldr/", "baldr_config_#.toml")

parser = argparse.ArgumentParser(description="kalman filter for Baldr")

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
    help="TOML file pattern (replace # with args.beam_id) to write/edit. Default: ../config_files/baldr_config.toml (relative to script)"
)

parser.add_argument(
    "--beam_id",
    type=int, #lambda s: [int(item) for item in s.split(",")],
    default=2,
    help="beam IDs to apply. Default: %(default)s"
)

parser.add_argument(
    "--phasemask",
    type=str,
    default="H3",
    help="Comma-separated beam IDs to apply. Default: 1,2,3,4"
)


parser.add_argument(
    '--number_of_iterations',
    type=int,
    default=1000,
    help="how many iterations do we run? %(default)s"
)


parser.add_argument(
    '--wvl',
    type=float,
    default=1.65,
    help="simulation wavelength (um). Default: %(default)s"
)

parser.add_argument(
    '--D_tel',
    type=float,
    default=1.8,
    help="telescope diameter for simulation. Default: %(default)s"
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


parser.add_argument(
    '--DM_chn',
    type=int,
    default=3,
    help="what channel on DM shared memory (0,1,2,3) to apply the turbulence?. Default: %(default)s"
)

args=parser.parse_args()


with open(args.toml_file.replace('#',f'{args.beam_id}'), "r") as f:

    config_dict = toml.load(f)
    
    # Baldr pupils from global frame 
    baldr_pupils = config_dict['baldr_pupils']
    I2A = np.array( config_dict[f'beam{args.beam_id}']['I2A'] )
    
    # image pixel filters
    pupil_mask = np.array(config_dict.get(f"beam{args.beam_id}", {}).get("pupil_mask", {}).get("mask", None) ).astype(bool)   # matrix bool
    exter_mask = np.array(config_dict.get(f"beam{args.beam_id}", {}).get("pupil_mask", {}).get("exterior", None) ).astype(bool) # matrix bool
    secon_mask = np.array(config_dict.get(f"beam{args.beam_id}", {}).get("pupil_mask", {}).get("secondary", None) ).astype(bool) # matrix bool

    # ctrl model 
    IM = np.array(config_dict.get(f"beam{args.beam_id}", {}).get(f"{args.phasemask}", {}).get("ctrl_model", None).get("IM", None) ).astype(float)
    I2M_raw = np.array(config_dict.get(f"beam{args.beam_id}", {}).get(f"{args.phasemask}", {}).get("ctrl_model", None).get("I2M", None) ).astype(float)
    I2M_LO_raw = np.array(config_dict.get(f"beam{args.beam_id}", {}).get(f"{args.phasemask}", {}).get("ctrl_model", None).get("I2M_LO", None) ).astype(float)
    I2M_HO_raw = np.array(config_dict.get(f"beam{args.beam_id}", {}).get(f"{args.phasemask}", {}).get("ctrl_model", None).get("I2M_HO", None) ).astype(float)
    M2C = np.array(config_dict.get(f"beam{args.beam_id}", {}).get(f"{args.phasemask}", {}).get("ctrl_model", None).get("M2C", None) ).astype(float)
    I0 = np.array(config_dict.get(f"beam{args.beam_id}", {}).get(f"{args.phasemask}", {}).get("ctrl_model", None).get("I0", None) ).astype(float)
    N0 = np.array(config_dict.get(f"beam{args.beam_id}", {}).get(f"{args.phasemask}", {}).get("ctrl_model", None).get("N0", None) ).astype(float)
    N0i = np.array(config_dict.get(f"beam{args.beam_id}", {}).get(f"{args.phasemask}", {}).get("ctrl_model", None).get("norm_pupil", None) ).astype(float)
    
    # used to normalize exterior and bad pixels in N0 (calculation of N0i)
    inside_edge_filt = np.array(config_dict.get(f"beam{args.beam_id}", {}).get(f"{args.phasemask}", {}).get("ctrl_model", None).get("inner_pupil_filt", None) )#.astype(bool)
    
    # reduction products
    IM_cam_config = config_dict.get(f"beam{args.beam_id}", {}).get(f"{args.phasemask}", {}).get("ctrl_model", None).get("camera_config", None) # dictionary
    
    bad_pixel_mask = np.array( config_dict.get(f"beam{args.beam_id}", {}).get(f"{args.phasemask}", {}).get("ctrl_model", None).get("bad_pixel_mask", None) )#.astype(bool)
    bias = np.array( config_dict.get(f"beam{args.beam_id}", {}).get(f"{args.phasemask}", {}).get("ctrl_model", None).get("bias", None) ).astype(float)
    dark = np.array( config_dict.get(f"beam{args.beam_id}", {}).get(f"{args.phasemask}", {}).get("ctrl_model", None).get("dark", None) ).astype(float)

    dm_flat = np.array( config_dict.get(f"beam{args.beam_id}", {}).get(f"{args.phasemask}", {}).get("ctrl_model", None).get("DM_flat", None) )

    strehl_coe_sec = np.array( config_dict.get(f"beam{args.beam_id}", {}).get("strehl_model", None).get("secondary",None) ).astype(float) #coe, intercept (on central secondary pixel, ADU/s/gain)
    strehl_coe_ext = np.array( config_dict.get(f"beam{args.beam_id}", {}).get("strehl_model", None).get("exterior",None) ).astype(float)#coe, intercept (on median signal, ADU/s/gain)




## HARD CODED FOR HEIMDALLR/BALDR 
dm2opd = 7.0 # 7um / DM cmd in OPD (wavespace) for BMC multi3.5
act_per_pupil = 10 # number of DM actuators across the pupil (Heimdallr/Baldr BMC multi3.5)
Nx_act = 12  # number of actuators across DM diamerer (BMC multi3.5)
corner_indicies = [0, 11, 11 * 12, -1] # DM corner indidices for BMC multi3.5 DM 
Nx_scrn = 32 # size of phasescreen that is binned to 12x12 DM cmd

## CAMERA
c = FLI.fli(args.global_camera_shm, roi = baldr_pupils[f'{args.beam_id}'])
cam_config = c.get_camera_config()
gain = float( cam_config["gain"] ) 
fps = float( cam_config["fps"] ) 

# CALCULATED VARIABLES 
N0dm = gain / fps * (I2A @ N0i.reshape(-1)) # these are already reduced #- dark_dm - bias_dm
I0dm = gain / fps * (I2A @ I0.reshape(-1)) # these are already reduced  #- dark_dm - bias_dm
bias_dm = I2A @ bias.reshape(-1)
dark_dm = I2A @ dark.reshape(-1)

# OPEN DMs 
dm = dmclass( beam_id=args.beam_id )
# # zero all channels
dm.zero_all()
# # activate flat (does this on channel 1)
dm.activate_calibrated_flat()


## ROLL PHASESCREENS AND LOOK AT MODEL 

telem = {}
lin_fit_coes = {}
lin_fit_res = {}
cmds = {}
r0_grid = [0.05, 0.1, 0.5, 2.0]

for r0 in r0_grid: #[0.1, 0.3, 0.5, 1]:
    r0_wvl = (r0)*(args.wvl/0.500)**(6/5) # coherence length at simulation wavelength
    scrn = ps.PhaseScreenVonKarman(nx_size= Nx_scrn, 
                                pixel_scale= args.D_tel / Nx_scrn, 
                                r0=r0_wvl, 
                                L0=25, # paranal outerscale median (m)
                                random_seed=1) # Kolmogorov phase screen in radians

    telem[r0] = {"time":[], "signal":[], "disturbance":[]}
    lin_fit_coes[r0] = []   #each entry corresponds to actuator
    cmds[r0] = []
    for it in range(args.number_of_iterations):

        print(f"Calibration iteration {it}")

        # roll the pahse screen
        for _ in range(10):
            scrn.add_row() 
        # bin it to DM dimensions
        dm_scrn = util.bin_phase_screen(phase_screen=scrn.scrn, out_size=Nx_act)

        # conver to OPD 
        opd = dm_scrn * args.wvl / (2*np.pi) # um

        # convert to a DM command 
        cmd = opd / dm2opd # BMC DM units (0-1)

        # forcefully remove piston  
        cmd -= np.mean( cmd )
        cmds[r0].append( cmd )
        #send command on specified DM channel 
        if np.std( cmd ) < 0.5:
            dm.shms[args.DM_chn].set_data( cmd ) 
            dm.shm0.post_sems(1) # tell DM to update
        else: 
            raise UserWarning("DM is being driven pretty hard.. are you sure you got the amplitudes right? ")
        
        time.sleep(0.01)

        t0 = time.time()

        # raw intensity 
        i = c.get_image(apply_manual_reduction=False) # we don't reduce in pixel space, but rather DM space to reduce number of operations 
        
        # go to dm space subtracting dark (ADU/s) and bias (ADU) there
        idm = (I2A @ i.reshape(-1)) - 1/fps * dark_dm - bias_dm

        # Interpolate intensity onto the DM actuator grid.
        # adu normalized pupil signal 
        s =  ( idm - I0dm ) / (N0dm)   # 

        telem[r0]['time'].append( t0 )
        telem[r0]['signal'].append( s )
        telem[r0]["disturbance"].append( remove_corners_and_flatten(dm.shms[args.DM_chn].get_data()) )

        # error
        # e = I2M @ s 


    # ---------------------------
    # (a) CALIBRATE LINEAR MODEL FOR H (with intercept - this accounts for Strehl bias)
    # ---------------------------
    # We assume during calibration (E=0):
    #      (dm_cmd-dm_flat) = H * i_dm + c
    # For each actuator, we fit a linear regression with intercept.


    #train
    nn=len(telem[r0]["signal"])
    X_cal = np.array(telem[r0]["signal"])[:nn//2]  # shape: (iterations, 140)
    Y_cal = np.array(telem[r0]["disturbance"])[:nn//2] 

    #test set to estimate covariance and R matric
    X_test= np.array(telem[r0]["signal"])[nn//2:]
    Y_test= np.array(telem[r0]["disturbance"])[nn//2:] 

    lin_models = []
    slope,interc=[],[]
    Y_pred = np.zeros_like(Y_cal)
    for act in range(Y_cal.shape[1]):
        
        stmp, ctmp = np.polyfit(X_cal[:, act], Y_cal[:, act],deg=1)
        slope.append(stmp)
        interc.append(ctmp)
        # lr = LinearRegression(fit_intercept=True)
        # lr.fit(X_cal[:, act], Y_cal[:, act])
        # Y_pred[:, act] = lr.predict(X_cal)
        # lin_models.append(lr)


    # look at test set and get residual covariance 
    Y_predtest = np.array( slope ) * X_test + np.array( interc ) 
    Y_pred = np.array( slope )  * X_cal + np.array( interc ) 
    # Y_predtest = np.zeros_like(Y_test)
    # for act in range(Y_test.shape[1]):
    #     Y_predtest[:, act] = lin_models[act].predict(X_test[:, act])


    # --- Estimate the measurement noise covariance R ---
    residuals = Y_test - Y_predtest   # shape: (calibration_iterations, 140)
    residual_variances = np.var(residuals, axis=0)


    lin_fit_coes[r0].append( (slope,interc) ) # each entry corresponds to an actuator 
    lin_fit_res[r0] =  residuals 

quickplot(np.array( [thing[0] for thing in lin_fit_coes[r0]]).reshape(-1), hist=True)

residual_variances = np.var(lin_fit_res[r0], axis=0)
#############################
# Construct the measurement noise covariance matrix as a diagonal matrix.
R_est = dm2opd * np.diag(residual_variances) # um^2

print("Estimated R matrix shape:", R_est.shape)



cmd_rms = [np.std( cc ) for cc in cmds.values()]
inter_mean = [] 

plt.figure()
for i , r0 in enumerate(r0_grid) :
    ss = np.array( [thing[1] for thing in lin_fit_coes[r0]]).reshape(-1)
    filt = abs(ss) < 1
    plt.hist( abs(ss), bins = np.logspace(-5,2,50), alpha =0.5,label=f"rms={cmd_rms[i]}um") 
    plt.axvline(np.mean( abs(ss)[filt] ))
    inter_mean.append(np.mean( abs(ss)[filt] ))
plt.xscale('log')
plt.xlabel('reference signal Strehl bias')
plt.legend()
plt.savefig('delme.png')

kwargs={"fontsize":15}
plt.figure(); 
plt.plot(1e3*np.array(cmd_rms), 1e3*dm2opd *  np.array( inter_mean ) ); 
plt.gca().tick_params(labelsize=15)
plt.xlabel("OPD [nm RMS]",kwargs)
plt.ylabel("Fitted intercept",kwargs)
plt.savefig('delme.png')


r0 = 0.5
cbar_label = ["Fitted Slopes [DM units/signal]","Fitted Intercepts [DM units]"]
imgs = [7000 * util.get_DM_command_in_2D( thing[0] ) for thing in lin_fit_coes[r0]] + [7000 * util.get_DM_command_in_2D( thing[1] ) for thing in lin_fit_coes[r0]]
util.nice_heatmap_subplots( im_list=imgs, cbar_label_list=cbar_label  )
plt.savefig('delme.png')

cbar_label = ["Fitted Slopes [nm RMS/signal]","Fitted Intercepts [nm RMS]"]
imgs = [7000 * util.get_DM_command_in_2D( thing[0] ) for thing in lin_fit_coes[r0]] + [7000 * util.get_DM_command_in_2D( thing[1] ) for thing in lin_fit_coes[r0]]
util.nice_heatmap_subplots( im_list=imgs, cbar_label_list=cbar_label  )
plt.savefig('delme.png')





######### NOW CALIBRATE STATE TRANSITION MODEL BY FITTING AR MODEL

phi=[]

#flatten DM
r0 = 0.5
slopes = np.array([thing[0]  for thing in lin_fit_coes[r0]])[0]
interc = np.array([thing[1]  for thing in lin_fit_coes[r0]])[0]

r0_wvl = (r0)*(args.wvl/0.500)**(6/5) # coherence length at simulation wavelength
scrn = ps.PhaseScreenVonKarman(nx_size= Nx_scrn, 
                            pixel_scale= args.D_tel / Nx_scrn, 
                            r0=r0_wvl, 
                            L0=25, # paranal outerscale median (m)
                            random_seed=1) # Kolmogorov phase screen in radians

for it in range(args.number_of_iterations):
    print(it)
    # roll screen
    #for _ in range(10):
    scrn.add_row()
    

    # bin it to DM dimensions
    dm_scrn = util.bin_phase_screen(phase_screen=scrn.scrn, out_size=Nx_act)

    # conver to OPD 
    opd = dm_scrn * args.wvl / (2*np.pi) # um

    # convert to a DM command 
    cmd = opd / dm2opd # BMC DM units (0-1)

    # forcefully remove piston  
    cmd -= np.mean( cmd )
    cmds[r0].append( cmd )
    #send command on specified DM channel 
    if np.std( cmd ) < 0.5:
        dm.shms[args.DM_chn].set_data( cmd ) 
        dm.shm0.post_sems(1) # tell DM to update
    else: 
        raise UserWarning("DM is being driven pretty hard.. are you sure you got the amplitudes right? ")
    
    time.sleep(0.1)
    # raw intensity 
    i = c.get_image(apply_manual_reduction=False) # we don't reduce in pixel space, but rather DM space to reduce number of operations 
    
    # go to dm space subtracting dark (ADU/s) and bias (ADU) there
    idm = (I2A @ i.reshape(-1)) - 1/fps * dark_dm - bias_dm

    # Interpolate intensity onto the DM actuator grid.
    # adu normalized pupil signal 
    s =  ( idm - I0dm ) / (N0dm)   # 

    # phase opd reconstruction from known DM response
    phi.append(  dm2opd * (slopes * s + interc) )


# units of phi are um, s = ADu/ADU (unitless)
phi= np.array(phi)

### build AR model of atmosphere !! ??

lags=20

ar_params, noise_var = fit_AR_models(np.array(phi), lags=lags)
# Build block–diagonal state-space matrices for all actuators
A_total, Q_total = build_block_state_space(ar_params, noise_var, lags=lags)

# test the state transitiom model 

iterations, n_act = phi.shape
print(f"phi shape: {phi.shape}, expected iterations: {iterations}, n_act: {n_act}")

# Prepare arrays to store predictions and residuals for t from lags to iterations-1.
phi_pred = np.zeros((iterations - lags, n_act))
residuals = np.zeros((iterations - lags, n_act))

for i in range(n_act):
    for t in range(lags, iterations):
        # Predict phi[t, i] using the AR(20) model for actuator i:
        # phi_pred = intercept + a1*phi[t-1] + a2*phi[t-2] + ... + a_lags*phi[t-lags]
        # Note: We use phi[t-lags:t, i] in reverse order so that the most recent (phi[t-1]) multiplies a1, etc.
        phi_pred[t - lags, i] = ar_params[i, 0] + np.dot(ar_params[i, 1:], phi[t-lags:t, i][::-1])
        # Residual: actual minus predicted
        residuals[t - lags, i] = phi[t, i] - phi_pred[t - lags, i]


actuator_index = 65

plt.figure()
plt.hist(residuals[:, actuator_index], bins=30)
plt.xlabel('Residual')
plt.ylabel('Frequency')
plt.title(f'Histogram of Residuals for Actuator {actuator_index}')
plt.savefig('delme.png')


plt.figure()
plt.plot(residuals[:, actuator_index])
plt.xlabel('Time Index (t - lags)')
plt.ylabel('Residual')
plt.title(f'Time Series of Residuals for Actuator {actuator_index}')
plt.savefig('delme.png')






# -------------------------------------------------------------------
# CLOSED-LOOP SIMULATION: Close the loop on the input phase screen.
# -------------------------------------------------------------------

n_state = A_total.shape[0]  # = no_actuators * lags

# Build measurement matrix H: each actuator measurement corresponds to the first element of its state block.
# 0----> z_est = H @ x + noise , where x is  sstate, z is measurement
#  y = z_meas - z_est ## inovation !!
# in our case x is opd, z is opd , so H just takes (filters) most recent x_est 
H_total = build_measurement_matrix(n_act=phi.shape[1], lags=lags)


# Initialize overall state: assume zero initial state for each actuator (20 lags per actuator)
x0_total = np.zeros((n_state, 1))
P0_total = 1e-7 * np.eye(n_state)

# Create block Kalman filter instance
bkf = BlockKalmanFilter(A=A_total, Q=Q_total, R=R_est, x0=x0_total, P0=P0_total)


#x_est_history = []
#dm_cmd_history = []
# strehl_history = []
# strehl_before_history = []
#P_history=[] # too big
#K_history=[] # too big

# sec_sig = []
# dm_feedback = []
# dm_disturb = []

telem = init_telem_dict() #)

# n_act: number of DM actuators (should be 140)
n_act = phi.shape[1]  # from your previous calibration, phi is (iterations, 140)

# Initialize "true" atmospheric state for each actuator in its AR(20) representation.
# For each actuator, the state is a vector of length 'lags' (here 20).
# We assume the initial state is zero.
x_true_total = np.zeros((n_act * lags, 1))


close_after = 100
for it in range(args.number_of_iterations):
    
    print(it)


    scrn.add_row()
    
    # bin it to DM dimensions
    dm_scrn = util.bin_phase_screen(phase_screen=scrn.scrn, out_size=Nx_act)

    # conver to OPD 
    opd = dm_scrn * args.wvl / (2*np.pi) # um

    # convert to a DM command 
    cmd = opd / dm2opd # BMC DM units (0-1)

    # forcefully remove piston  
    cmd -= np.mean( cmd )

    #send command on specified DM channel 
    if np.std( cmd ) < 0.4:
        dm.shms[args.DM_chn].set_data( cmd ) 
        dm.shm0.post_sems(1) # tell DM to update
    else: 
        raise UserWarning("DM is being driven pretty hard.. are you sure you got the amplitudes right? ")
    
    time.sleep(0.1)

    # raw intensity 
    i = c.get_image(apply_manual_reduction=False) # we don't reduce in pixel space, but rather DM space to reduce number of operations 
    
    sec_sig =  i[secon_mask.astype(bool)][4] #- 1/fps * dark[secon_mask.astype(bool)][4] - bias[secon_mask.astype(bool)][4]

    # go to dm space subtracting dark (ADU/s) and bias (ADU) there
    idm = (I2A @ i.reshape(-1)) - 1/fps * dark_dm - bias_dm

    # Interpolate intensity onto the DM actuator grid.
    # adu normalized pupil signal 
    s =  ( idm - I0dm ) / (N0dm)   # 


    # (3) Reconstruct the total phase using the calibrated linear model:
    # Here, s (slope vector dm command unit per adu) and c (intercept vector) were determined during calibration.
    # units are opd (m)
    phi_total =  dm2opd * (slopes * s + interc) 

    # (4) Subtract the known DM contribution:
    # The DM-induced phase is: phi_dm = f * cmd (with f = zwfs_ns.dm.opd_per_cmd in our simple case)
    # So, the effective measurement for the atmosphere is:
    # units = OPD
    z_meas = phi_total - dm2opd * remove_corners_and_flatten( dm.shms[args.DM_chn].get_data() )

    z_meas = z_meas.reshape((n_act, 1))
    #x_est_total = bkf.update(z_meas, H_total)
    
    if it > close_after:     
        # Run the Kalman filter update:
        bkf.predict() # predict hte next state using state transition

        # to understand how H works 
        # aaa=x_true_total.reshape( n_act , lags )
        #aaa[:,0]=1
        #H_total @ aaa.reshape(n_act*lags)
        #filters only for the most recent lag in x state
        x_est_total = bkf.update(z_meas, H_total) # update this with the measurement 
        
        
        #x_est_history.append(x_est_total.copy())
        #P_history.append(bkf.P)
        #K_history.append(bkf.K)

        # Extract the estimated atmospheric phase for each actuator (first element of each state block):
        phi_est = x_est_total.reshape(n_act, lags)[:, 0].reshape((n_act, 1))
        
        # Compute the DM command to cancel the estimated atmospheric phase.
        # The DM-induced phase is given by: φ_dm = opd_per_cmd * dm_cmd.
        # To cancel φ_atmosphere, set: dm_cmd = - (φ_est / opd_per_cmd)
        dm_cmd = - (1.0 / dm2opd) * phi_est

        dm.set_data( dm.cmd_2_map2D( dm_cmd.ravel()) ) # feedback channel 2 
        dm.shm0.post_sems(1) # tell DM to update

    
    if telem:
        telem["time_cam"].append( time.time())
        #telem["time_dm"].append( t1 )
        telem["i"].append( i.copy() )
        telem["s"].append( s )
        #telem["e_HO"].append( e.copy() )
        #telem["u_HO"].append( u.copy() )
        
        #telem["current_dm_ch0"].append( dm.shms[0].get_data() ) 
        #telem["current_dm_ch1"].append( dm.shms[1].get_data().copy() ) 
        telem["current_dm_ch2"].append( dm.shms[2].get_data().copy() ) 
        telem["current_dm_ch3"].append( dm.shms[3].get_data().copy() ) 
        telem["exterior_sig"].append( i[exter_mask.astype(bool)].copy() )
        telem["secondary_sig"].append( sec_sig )
  

dm.zero_all()
dm.activate_calibrated_flat()

from astropy.io import fits
# save telemetry
runn=f"KALMAN_r0-{r0}_fps-{cam_config['fps']}_gain-fps-{cam_config['gain']}" 
# Create a list of HDUs (Header Data Units)
hdul = fits.HDUList()


hdu = fits.ImageHDU(ar_params)
hdu.header['EXTNAME'] = 'AR'
hdul.append(hdu)


hdu = fits.ImageHDU(list([slopes]) + list([interc]))
hdu.header['EXTNAME'] = 'MEAS_MODEL'
hdul.append(hdu)


hdu = fits.ImageHDU(I2A)
hdu.header['EXTNAME'] = 'interpMatrix'
hdul.append(hdu)

hdu = fits.ImageHDU(dm.shms[0].get_data())
hdu.header['EXTNAME'] = 'DM_FLAT_OFFSET'
hdul.append(hdu)


hdu = fits.ImageHDU(pupil_mask.astype(int))
hdu.header['EXTNAME'] = 'ext'
hdul.append(hdu)
# Add each list to the HDU list as a new extension
for list_name, data_list in telem.items() :##zip(["time","i","err", "reco", "disturb", "secondary_sig"] ,[   telem["time"], telem["i"],telem["e_HO"], telem["current_dm_ch2"],telem["current_dm_ch3"], telem["secondary_sig"]] ) : # telem.items():
    # Convert list to numpy array for FITS compatibility
    data_array = np.array(data_list, dtype=float)  # Ensure it is a float array or any appropriate type

    # Create a new ImageHDU with the data
    hdu = fits.ImageHDU(data_array)

    # Set the EXTNAME header to the variable name
    hdu.header['EXTNAME'] = list_name

    # Append the HDU to the HDU list
    hdul.append(hdu)


# Write the HDU list to a FITS file

tele_pth = "/home/asg/Videos/KALMAN/" #args.folder_pth 
if not os.path.exists( tele_pth ):
    os.makedirs( tele_pth )

fits_file = tele_pth + f'CL_beam{args.beam_id}_mask{args.phasemask}_{runn}.fits' #_{args.phasemask}.fits'
hdul.writeto(fits_file, overwrite=True)
print(f'wrote telemetry to \n{fits_file}')

