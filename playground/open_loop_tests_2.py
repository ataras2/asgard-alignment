#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep  9 13:39:11 2024

@author: bencb
"""








import numpy as np 
from astropy.io import fits
import aotools 

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
            self.integrals[i] += error
            self.integrals[i] = np.clip(self.integrals[i], self.lower_limit[i], self.upper_limit[i])

            derivative = error - self.prev_errors[i]
            self.output[i] = (self.kp[i] * error +
                              self.ki[i] * self.integrals[i] +
                              self.kd[i] * derivative)
            self.prev_errors[i] = error

        return self.output

    def reset(self):
        self.integrals.fill(0.0)
        self.prev_errors.fill(0.0)
        
        

class LeakyIntegrator:
    def __init__(self, rho=None, lower_limit=None, upper_limit=None, kp=None):
        # If no arguments are passed, initialize with default values
        if rho is None:
            self.rho = []
            self.lower_limit = []
            self.upper_limit = []
            self.kp = []
        else:
            if len(rho) == 0:
                raise ValueError("Rho vector cannot be empty.")
            if len(lower_limit) != len(rho) or len(upper_limit) != len(rho):
                raise ValueError("Lower and upper limit vectors must match rho vector size.")
            if kp is None or len(kp) != len(rho):
                raise ValueError("kp vector must be the same size as rho vector.")

            self.rho = np.array(rho)
            self.output = np.zeros(len(rho))
            self.lower_limit = np.array(lower_limit)
            self.upper_limit = np.array(upper_limit)
            self.kp = np.array(kp)  # kp is a vector now

    def process(self, input_vector):
        input_vector = np.array(input_vector)

        # Error checks
        if len(input_vector) != len(self.rho):
            raise ValueError("Input vector size must match rho vector size.")

        size = len(self.rho)
        error_message = ""

        if len(self.rho) != size:
            error_message += "rho "
        if len(self.lower_limit) != size:
            error_message += "lower_limit "
        if len(self.upper_limit) != size:
            error_message += "upper_limit "
        if len(self.kp) != size:
            error_message += "kp "

        if error_message:
            raise ValueError("Input vectors of incorrect size: " + error_message)

        if len(self.output) != size:
            print(f"output.size() != size.. reinitializing output to zero with correct size")
            self.output = np.zeros(size)

        # Process with the kp vector
        self.output = self.rho * self.output + self.kp * input_vector
        self.output = np.clip(self.output, self.lower_limit, self.upper_limit)

        return self.output

    def reset(self):
        self.output = np.zeros(len(self.rho))

        




# 
itera = 1 # first try here 

exper_path = f'open_loop_{itera}/'

if not os.path.exists(fig_path + exper_path):
   os.makedirs(fig_path + exper_path)


"""# reco file is with cal dm flat, reco file 2 with bmc cal 
reco_file = 'tmp/09-09-2024/iter_4_J3/zonal_reconstructor/RECONSTRUCTORS_zonal_0.07pokeamp_in-out_pokes_map_DIT-0.001_gain_high_09-09-2024T08.28.05.fits'
#"/home/heimdallr/Documents/asgard-alignment/tmp/09-09-2024/iter_3_J3/fourier_90modes_pinv_reconstructor/RECONSTRUCTORS_fourier90_0.2pokeamp_in-out_pokes_pinv_DIT-0.001_gain_high_09-09-2024T07.38.23.fits"
#"/home/heimdallr/Documents/asgard-alignment/tmp/09-09-2024/iter_4_J3/fourier_90modes_pinv_reconstructor/RECONSTRUCTORS_fourier90_0.2pokeamp_in-out_pokes_pinv_DIT-0.001_gain_high_09-09-2024T08.30.27.fits"
#"/home/heimdallr/Documents/asgard-alignment/tmp/09-09-2024/iter_3_J3/fourier_90modes_pinv_reconstructor/RECONSTRUCTORS_fourier90_0.2pokeamp_in-out_pokes_pinv_DIT-0.001_gain_high_09-09-2024T07.38.23.fits"
#"/home/heimdallr/Documents/asgard-alignment/tmp/09-09-2024/iter_3_J3/zonal_reconstructor/RECONSTRUCTORS_zonal_0.07pokeamp_in-out_pokes_map_DIT-0.001_gain_high_09-09-2024T07.35.58.fits" #'/home/heimdallr/Documents/asgard-alignment/tmp/08-09-2024/iter_1_J3/zonal_reconstructor/RECONSTRUCTORS_zonal_0.07pokeamp_in-out_pokes_map_DIT-0.001_gain_high_08-09-2024T18.17.34.fits'
#reco_file2 = '/home/heimdallr/Documents/asgard-alignment/tmp/08-09-2024/iter_3_J3/zonal_reconstructor/RECONSTRUCTORS_zonal_0.07pokeamp_in-out_pokes_map_DIT-0.001_gain_high_08-09-2024T20.54.08.fits'
ff = fits.open(reco_file) 
#ff2 = fits.open( reco_file2)

# _,S1, _ = np.linalg.svd( ff['IM'].data)
# _,S2, _ = np.linalg.svd( ff2['IM'].data)

# plt.figure(); plt.plot( S1 ); plt.plot( S2, label='S2'); plt.savefig(fig_path + 'delme.png')

# poke_amp = ff['INFO'].header['poke_amplitude']    

"""


# last minute check 
# zwfs.dm.send_data(zwfs.dm_shapes['flat_dm'] + 0.1 * zwfs.dm_shapes['four_torres'])
zwfs.dm.send_data(zwfs.dm_shapes['flat_dm'] )
phasemask_centering_tool.move_relative_and_get_image(zwfs, phasemask, savefigName=fig_path + 'delme.png')


# BUILD THE RECONSTRUCTOR HERE 
I0, N0 = util.get_reference_images(zwfs, phasemask, theta_degrees=11.8, number_of_frames=256, \
compass = True, compass_origin=None, savefig= fig_path + exper_path + f'FPM-in-out_{phasemask_name}_{label}.png' )


modal_basis = util.construct_command_basis('Zonal_pinned_edges')


####
# NOTE HERE WE BUILD WITH REAL REGISTRATION 
###
poke_amp = 0.07
IM_list = []
for i,m in enumerate(modal_basis.T):
    print(f'executing cmd {i}/{len(modal_basis)}')
    I_plus_list = []
    I_minus_list = []
    imgs_to_mean = 10
    for sign in [(-1)**n for n in range(10)]: #[-1,1]:
        zwfs.dm.send_data( list( zwfs.dm_shapes['flat_dm'] + sign * poke_amp/2 * m )  )
        time.sleep(0.01)
        if sign > 0:
            I_plus_list += zwfs.get_some_frames(number_of_frames = imgs_to_mean, apply_manual_reduction = True )
            #I_plus *= 1/np.mean( I_plus )
        if sign < 0:
            I_minus_list += zwfs.get_some_frames(number_of_frames = imgs_to_mean, apply_manual_reduction = True )
            #I_minus *= 1/np.mean( I_minus )

    I_plus = np.mean( I_plus_list, axis = 0).reshape(-1)  # flatten so can filter with pupil_pixels
    I_plus *= 1/np.mean( I_plus )

    I_minus = np.mean( I_minus_list, axis = 0).reshape(-1)  # flatten so can filter with pupil_pixels
    I_minus *= 1/np.mean( I_minus )

    errsig = (I_plus - I_minus)[np.array( zwfs.pupil_pixels )]
    IM_list.append( list(  errsig.reshape(-1) ) ) #toook out 1/poke_amp *

IM= np.array( IM_list ).T # 1/poke_amp * 

zwfs.dm.send_data(zwfs.dm_shapes['flat_dm'] )

M2C_0 = modal_basis.T


bb = util.construct_command_basis('Zernike' ,without_piston=False)[:,0]

dm_pupil_filt =  bb > 0  # np.std((M2C.T @ IM.T) ,axis=1) > 4

def plot_DM( cmd , save_path = fig_path + 'delme.png'):

    plt.figure(); plt.imshow( util.get_DM_command_in_2D( cmd  ) ); plt.colorbar() ; plt.savefig(save_path)

#plot_DM( dm_pupil_filt ) 


## WE CAN DO ALL THIS WITH DIFFERENT PUPIL REGISTRATIONS!!! 
pupil_pixel_shift=0
pupil_pixel_filter = numpy.roll(zwfs.pupil_pixel_filter, shift=pupil_pixel_shift, axis=0)
pupil_pixels = np.where( pupil_pixel_filter )  #pupil_pixels.copy() 


cmd2opd = 3200 # to go to cmd space to nm OPD 


# ---- 1) reconstruct previous IM signal 


current_path = fig_path + exper_path +  "1_reco_IM_signal/" 
if not os.path.exists(current_path ):
   os.makedirs(current_path)


U, S, Vt = np.linalg.svd( IM, full_matrices=False)

rmse_list =[]
c_list =[]
R_list = []
residual_list = [] 

Smax_grid = np.arange( 2,100,3)


m=65 # mode to put on
# put on IM signal with a bit of noise (1/2 std )
noise_sigma = 0.5 * np.std( IM )

sig =  IM.T[i] + noise_sigma * np.random.rand( IM.T[m].shape[0] ) 
sig_list = [sig] 

disturbance_cmd  =  poke_amp * M2C_0[m]    
disturb_list = [disturbance_cmd]

for Smax in Smax_grid: 
    
    U, S, Vt = np.linalg.svd( IM, full_matrices=False)

    R  = (Vt.T * [1/ss if i < Smax else 0 for i,ss in enumerate(S)])  @ U.T
    
    c = poke_amp * M2C_0.T @ R @ sig.reshape(-1) #dont need pupil filter as its inbuilt in IM 
    
    residual =  (disturbance_cmd - c)[dm_pupil_filt]

    rmse = np.nanstd( residual )
    
    residual_list.append( residual )    
    rmse_list.append( rmse ) 
    c_list.append( c )
    R_list.append( R )
    
best_i = np.argmin( rmse_list )

im_list = [ np.flipud(sig), cmd2opd * util.get_DM_command_in_2D( disturbance_cmd ) , cmd2opd * util.get_DM_command_in_2D( c_list[best_i] ) , \
            cmd2opd * util.get_DM_command_in_2D(residual_list[best_i]) ) ]
xlabel_list = ['' for _ in range(len(im_list))]
ylabel_list = ['' for _ in range(len(im_list))]

dm_mode = cmd2opd *  util.get_DM_command_in_2D( disturbance_cmd)

vlims=[[np.min(sig_list[best_i]),np.max(sig_list[best_i])]] + [[np.nanmin(dm_mode), np.nanmax(dm_mode)] for _ in im_list[1:]]
title_list = ['ZWFS signal', 'DM aberration' , 'DM reconstruction', 'residuals']
cbar_label_list = [ 'normalized signal [adu]', 'OPD [nm]', 'OPD [nm]','OPD [nm]']
util.nice_heatmap_subplots(im_list , xlabel_list, ylabel_list, title_list,cbar_label_list, fontsize=15, vlims=vlims, \
                        cbar_orientation = 'bottom', axis_off=True, savefig= current_path + f'reco_IM_signal_{m}_mask{phasemask_name}.png')

    
#basis_name = 'fourier' #'zonal'
plt.figure(figsize=(8,5))
plt.plot( Smax_grid, 2*np.pi/1050 * cmd2opd * np.array( rmse_list)  )
#plt.axhline( np.var(test_field.phase[z.wvls[0]][z.pup>0]) , label='initial')
plt.legend()
plt.xlabel('Singular value truncation index',fontsize=15)
plt.ylabel('Reconstructed RMSE [rad]',fontsize=15) 
plt.gca().tick_params(labelsize=15)   
plt.savefig(current_path + f'rmse_vs_svd_trunction_measured_IM_signal_{m}_{phasemask_name}.png', bbox_inches='tight', dpi=200)


write_dict = {
"disturbance" : disturb_list,
"signal" : sig_list,
"svd_trucation_grid":Smax_grid,
"reconstructor" :R_list ,
"reco_cmd":c_list ,
"cmd_rmse":rmse_list ,
"cmd_residual":residual_list  
}



# Create a list of HDUs (Header Data Units)
hdul = fits.HDUList()

# Add each list to the HDU list as a new extension
for list_name, data_list in write_dict.items():
    # Convert list to numpy array for FITS compatibility
    data_array = np.array(data_list, dtype=float)  # Ensure it is a float array or any appropriate type

    # Create a new ImageHDU with the data
    hdu = fits.ImageHDU(data_array)

    # Set the EXTNAME header to the variable name
    hdu.header['EXTNAME'] = list_name

    # Append the HDU to the HDU list
    hdul.append(hdu)

# Write the HDU list to a FITS file
hdul.writeto( current_path + f'pt1-reco_IM_signal_telemetry_{phasemask_name}_{tstamp}.fits', overwrite=True)



# ---- 2) poke mode on reconstructor basis and reconstruct 



current_path = fig_path + exper_path +  "2_reco_on_basis_signal/" 
if not os.path.exists(current_path ):
   os.makedirs(current_path)


m = 65
disturbance_cmd = poke_amp * M2C_0[m] 
zwfs.dm.send_data( zwfs.dm_shapes['flat_dm'] + dm_pupil_filt * disturbance_cmd )


residual_list = []
rmse_list =[]
c_list =[]
sig_list = []
R_list = []

disturb_list = [disturbance_cmd ]

Smax_grid = np.arange( 2,100,3)

for Smax in Smax_grid: 
    
    U, S, Vt = np.linalg.svd( IM, full_matrices=False)

    R  = (Vt.T * [1/ss if i < Smax else 0 for i,ss in enumerate(S)])  @ U.T

    """TT_vectors = util.get_tip_tilt_vectors()

    TT_space = M2C @ TT_vectors
        
    U_TT, S_TT, Vt_TT = np.linalg.svd( TT_space, full_matrices=False)

    I2M_TT = U_TT.T @ R 

    M2C_TT = M2C_0.T @ U_TT # since pinned need M2C to go back to 140 dimension vector  

    R_HO = (np.eye(U_TT.shape[0])  - U_TT @ U_TT.T) @ R

    # go to Eigenmodes for modal control in higher order reconstructor
    U_HO, S_HO, Vt_HO = np.linalg.svd( R_HO, full_matrices=False)
    I2M_HO = Vt_HO  
    M2C_HO = M2C_0.T @ (U_HO * S_HO) # since pinned need M2C to go back to 140 dimension vector 
    """
    i = np.mean( zwfs.get_some_frames(number_of_frames=100, apply_manual_reduction=True),axis=0) #z.detection_chain( test_field, FPM_on=True, include_shotnoise=True, ph_per_s_per_m2_per_nm=True, grids_aligned=True, replace_nan_with=0 )
    #o = #z.detection_chain( test_field, FPM_on=False, include_shotnoise=True, ph_per_s_per_m2_per_nm=True, grids_aligned=True, replace_nan_with=0 )
    
    sig = i / np.mean( i ) -  I0 / np.mean( I0 ) # I0_theory/ np.mean(I0_theory) #

    # update distrubance after measurement 
    #for _ in range(rows_to_jump):
    #    scrn.add_row()
    #disturbance_cmd = np.array( util.create_phase_screen_cmd_for_DM(scrn,  scaling_factor=scrn_scaling_factor   , drop_indicies = corner_indicies, plot_cmd=False) )
    """
    e_TT = I2M_TT @ sig.reshape(-1)[pupil_pixels]
    
    u_TT = e_TT #pid.process( e_TT )
    
    c_TT = M2C_TT @ u_TT 
    
    e_HO = I2M_HO @ sig.reshape(-1)[pupil_pixels]

    u_HO = e_HO #leak.process( e_HO )
    
    c_HO = M2C_HO @ u_HO 
    """
    #c = c_TT + c_HO
    # or if 1/poke amp mult by IM prior use : M2C_0.T  @ R @ sig.reshape(-1)[pupil_pixels]
    c =  poke_amp * M2C_0.T  @ R @ sig.reshape(-1)[pupil_pixels]

    #zwfs.dm.send_data( zwfs.dm_shapes['flat_dm'] + disturbance_cmd - c_HO - c_TT ) # same way to rtc PID 
    #time.sleep(0.05)  
    # only measure residual in the registered pupil on DM 
    residual =  (disturbance_cmd - c)[dm_pupil_filt]
    rmse = np.nanstd( residual )
    
    rmse_list.append( rmse ) 
    
    residual_list.append( disturbance_cmd - c)
    c_list.append( c )
    sig_list.append( sig )
    R_list.append( R )
    
best_i = np.argmin( rmse_list )


im_list = [ np.flipud(sig), cmd2opd * util.get_DM_command_in_2D( disturbance_cmd ) , cmd2opd * util.get_DM_command_in_2D( c_list[best_i] ) , \
            cmd2opd * util.get_DM_command_in_2D(residual_list[best_i]) ) ]
           
xlabel_list = ['' for _ in range(len(im_list))]
ylabel_list = ['' for _ in range(len(im_list))]

dm_mode = cmd2opd *  util.get_DM_command_in_2D( disturbance_cmd)

vlims=[[np.min(sig_list[best_i]),np.max(sig_list[best_i])]] + [[np.nanmin(dm_mode), np.nanmax(dm_mode)] for _ in im_list[1:]]
title_list = ['ZWFS signal', 'DM aberration' , 'DM reconstruction', 'residuals']
cbar_label_list = [ 'normalized signal [adu]', 'OPD [nm]', 'OPD [nm]','OPD [nm]']
util.nice_heatmap_subplots(im_list , xlabel_list, ylabel_list, title_list,cbar_label_list, fontsize=15, vlims=vlims, \
                        cbar_orientation = 'bottom', axis_off=True, savefig= current_path + f'reco_single_mode_{m}_mask{phasemask_name}.png')

    

#basis_name = 'fourier' #'zonal'
plt.figure(figsize=(8,5))
plt.plot( Smax_grid, 2*np.pi/1050 * cmd2opd * np.array( rmse_list)  )
#plt.axhline( np.var(test_field.phase[z.wvls[0]][z.pup>0]) , label='initial')
plt.legend()
plt.xlabel('Singular value truncation index',fontsize=15)
plt.ylabel('Reconstructed RMSE [rad]',fontsize=15) 
plt.gca().tick_params(labelsize=15)   
plt.savefig(current_path  +f'rmse_vs_svd_trunction_measured_single_mode_{m}_{phasemask_name}.png', bbox_inches='tight', dpi=200)



write_dict = {
"disturbance" : disturb_list,
"signal" : sig_list,
"svd_trucation_grid":Smax_grid,
"reconstructor" :R_list ,
"reco_cmd":c_list ,
"cmd_rmse":rmse_list ,
"cmd_residual":residual_list  
}



# Create a list of HDUs (Header Data Units)
hdul = fits.HDUList()

# Add each list to the HDU list as a new extension
for list_name, data_list in write_dict.items():
    # Convert list to numpy array for FITS compatibility
    data_array = np.array(data_list, dtype=float)  # Ensure it is a float array or any appropriate type

    # Create a new ImageHDU with the data
    hdu = fits.ImageHDU(data_array)

    # Set the EXTNAME header to the variable name
    hdu.header['EXTNAME'] = list_name

    # Append the HDU to the HDU list
    hdul.append(hdu)

# Write the HDU list to a FITS file
hdul.writeto( current_path + f'pt2-reco_single_mode_telemetry_{m}_{phasemask_name}_{tstamp}.fits', overwrite=True)
    
    
    
    
    
    
# ----3) put Kolmogorov mode on DM and reconstruct 

current_path = fig_path + exper_path +  "3_reco_kolmogorov/" 
if not os.path.exists(current_path ):
   os.makedirs(current_path)

# Which is truncation is best 

Nx_act = 12 # actuators across DM 

D = 1.8 #m effective diameter of the telescope

screen_pixels = Nx_act*2**3  #pixels inthe inital screen before projection onto DM

corner_indicies = [0, Nx_act-1, Nx_act * (Nx_act-1), -1] 

scrn_scaling_factor =  0.1 

rows_to_jump = 2 # how many rows to jump on initial phase screen for each Baldr loop

distance_per_correction = rows_to_jump * D/screen_pixels # effective distance travelled by turbulence per AO iteration 
print(f'{rows_to_jump} rows jumped per AO command in initial phase screen of {screen_pixels} pixels. for {D}m mirror this corresponds to a distance_per_correction = {distance_per_correction}m')

scrn = aotools.infinitephasescreen.PhaseScreenVonKarman(nx_size=screen_pixels, pixel_scale=D/screen_pixels,r0=0.1,L0=12)

disturbance_cmd = util.create_phase_screen_cmd_for_DM(scrn,  scaling_factor=scrn_scaling_factor   , drop_indicies = corner_indicies, plot_cmd=False)

zwfs.dm.send_data( zwfs.dm_shapes['flat_dm'] + dm_pupil_filt * disturbance_cmd )

rmse_list =[]
c_list =[]
sig_list = []
R_list = []
Smax_grid = np.arange( 2,100,3)

for Smax in Smax_grid: 
    
    U, S, Vt = np.linalg.svd( IM, full_matrices=False)

    R  = (Vt.T * [1/ss if i < Smax else 0 for i,ss in enumerate(S)])  @ U.T

    """TT_vectors = util.get_tip_tilt_vectors()

    TT_space = M2C @ TT_vectors
        
    U_TT, S_TT, Vt_TT = np.linalg.svd( TT_space, full_matrices=False)

    I2M_TT = U_TT.T @ R 

    M2C_TT = M2C_0.T @ U_TT # since pinned need M2C to go back to 140 dimension vector  

    R_HO = (np.eye(U_TT.shape[0])  - U_TT @ U_TT.T) @ R

    # go to Eigenmodes for modal control in higher order reconstructor
    U_HO, S_HO, Vt_HO = np.linalg.svd( R_HO, full_matrices=False)
    I2M_HO = Vt_HO  
    M2C_HO = M2C_0.T @ (U_HO * S_HO) # since pinned need M2C to go back to 140 dimension vector 
    """
    i = np.mean( zwfs.get_some_frames(number_of_frames=100, apply_manual_reduction=True),axis=0) #z.detection_chain( test_field, FPM_on=True, include_shotnoise=True, ph_per_s_per_m2_per_nm=True, grids_aligned=True, replace_nan_with=0 )
    #o = #z.detection_chain( test_field, FPM_on=False, include_shotnoise=True, ph_per_s_per_m2_per_nm=True, grids_aligned=True, replace_nan_with=0 )
    
    sig = i / np.mean( i ) -  I0 / np.mean( I0 ) # I0_theory/ np.mean(I0_theory) #

    # update distrubance after measurement 
    #for _ in range(rows_to_jump):
    #    scrn.add_row()
    #disturbance_cmd = np.array( util.create_phase_screen_cmd_for_DM(scrn,  scaling_factor=scrn_scaling_factor   , drop_indicies = corner_indicies, plot_cmd=False) )
    """
    e_TT = I2M_TT @ sig.reshape(-1)[pupil_pixels]
    
    u_TT = e_TT #pid.process( e_TT )
    
    c_TT = M2C_TT @ u_TT 
    
    e_HO = I2M_HO @ sig.reshape(-1)[pupil_pixels]

    u_HO = e_HO #leak.process( e_HO )
    
    c_HO = M2C_HO @ u_HO 
    """
    #c = c_TT + c_HO
    # or if 1/poke amp mult by IM prior use : M2C_0.T  @ R @ sig.reshape(-1)[pupil_pixels]
    c =  poke_amp * M2C_0.T  @ R @ sig.reshape(-1)[pupil_pixels]

    #zwfs.dm.send_data( zwfs.dm_shapes['flat_dm'] + disturbance_cmd - c_HO - c_TT ) # same way to rtc PID 
    #time.sleep(0.05)  
    # only measure residual in the registered pupil on DM 
    residual =  (disturbance_cmd - c)[dm_pupil_filt]
    
    residual_list.append( disturbance_cmd - c)
    rmse = np.nanstd( residual )
    
    rmse_list.append( rmse ) 
    c_list.append( c )
    sig_list.append( sig )
    R_list.append( R )
    
best_i = np.argmin( rmse_list )

use_dm_filt = 1
if use_dm_filt:
    im_list = [ np.flipud(sig_list[best_i]), cmd2opd * util.get_DM_command_in_2D( dm_pupil_filt * disturbance_cmd ) , cmd2opd * util.get_DM_command_in_2D( dm_pupil_filt * c_list[best_i] ) , \
                cmd2opd * util.get_DM_command_in_2D( dm_pupil_filt * residual_list[best_i]) ) ]
else:
    im_list = [ np.flipud(sig_list[best_i]), cmd2opd * util.get_DM_command_in_2D( dm_pupil_filt * disturbance_cmd ) , cmd2opd * util.get_DM_command_in_2D( dm_pupil_filt * c_list[best_i] ) , \
                cmd2opd * util.get_DM_command_in_2D( dm_pupil_filt * residual_list[best_i]) ) ]
 
xlabel_list = ['' for _ in range(len(im_list))]
ylabel_list = ['' for _ in range(len(im_list))]
dm_mode = cmd2opd *  util.get_DM_command_in_2D( disturbance_cmd )
vlims=[[np.min(sig_list[best_i]),np.max(sig_list[best_i])]] + [[np.nanmin(dm_mode), np.nanmax(dm_mode)] for _ in im_list[1:]]
title_list = ['ZWFS signal', 'DM aberration' , 'DM reconstruction', 'residuals']
cbar_label_list = [ 'normalized signal [adu]', 'OPD [nm]', 'OPD [nm]','OPD [nm]']
util.nice_heatmap_subplots(im_list , xlabel_list, ylabel_list, title_list,cbar_label_list, fontsize=15, vlims=vlims, \
                        cbar_orientation = 'bottom', axis_off=True, savefig=current_path +  f'reco_Kolmogorov_mask{phasemask_name}.png')

 
#basis_name = 'fourier' #'zonal'
plt.figure(figsize=(8,5))
plt.plot( Smax_grid, 2*np.pi/1050 * cmd2opd * np.array( rmse_list)  )
#plt.axhline( np.var(test_field.phase[z.wvls[0]][z.pup>0]) , label='initial')
plt.legend()
plt.xlabel('Singular value truncation index',fontsize=15)
plt.ylabel('Reconstructed RMSE [rad]',fontsize=15) 
plt.gca().tick_params(labelsize=15)   
plt.savefig(current_path +f'rmse_vs_svd_trunction_measured_Kolmogorov_{phasemask_name}.png', bbox_inches='tight', dpi=200)






write_dict = {
"disturbance" : disturb_list,
"signal" : sig_list,
"svd_trucation_grid":Smax_grid,
"reconstructor" :R_list ,
"reco_cmd":c_list ,
"cmd_rmse":rmse_list ,
"cmd_residual":residual_list  
}



# Create a list of HDUs (Header Data Units)
hdul = fits.HDUList()

# Add each list to the HDU list as a new extension
for list_name, data_list in write_dict.items():
    # Convert list to numpy array for FITS compatibility
    data_array = np.array(data_list, dtype=float)  # Ensure it is a float array or any appropriate type

    # Create a new ImageHDU with the data
    hdu = fits.ImageHDU(data_array)

    # Set the EXTNAME header to the variable name
    hdu.header['EXTNAME'] = list_name

    # Append the HDU to the HDU list
    hdul.append(hdu)

# Write the HDU list to a FITS file
hdul.writeto( current_path + f'pt3-reco_kolmogorov-r0-0.1_amp{scrn_scaling_factor}_telemetry_{phasemask_name}_{tstamp}.fits', overwrite=True)
    
    
    













# As a check look at reconstructor on FOurier basis and see if we can reco it


plt.figure(); 
plt.plot( (reco_data_f['M2C_4RECO'].data.T[m] @ (1/poke_amp * M2C.T @ IM.T)) ,label='reco'); 
plt.plot( reco_data_f['IM'].data[m] ,label='True')
plt.legend() 
plt.savefig(fig_path+'delme.png')

m = 10
dm_mode = util.get_DM_command_in_2D( reco_data_f['M2C_4RECO'].data.T[m] ) # outer actuators pinned so just look at inner

#reco_dict['zonal']['poke_amp']**2 / 2 

reco_TT = util.get_DM_command_in_2D( 1/2 * ( M2C_TT @ I2M_TT @ reco_data_f['IM'].data[m] ) )

reco_HO = util.get_DM_command_in_2D( 1/2 *  ( M2C_HO @ I2M_HO @ reco_data_f['IM'].data[m] ) )

residual = dm_mode - reco_TT - reco_HO


cmd2opd = 3200
im_list =  [ cmd2opd * dm_mode, cmd2opd * reco_TT, cmd2opd * reco_HO, cmd2opd * residual ] 
xlabel_list = [ "" for _ in im_list]
ylabel_list = [ "" for _ in im_list]
title_list = [ "aberration", "reco. TT", "reco. HO", "residuals"]
cbar_label_list =  [ "OPD [nm]" for _ in im_list]

savefig = fig_path + 'delme.png' # exper_path + f'reconstruct_fourier_mode{m}_with_pinned_zonal_basis_reco_maskJ3.png'
util.nice_heatmap_subplots( im_list , xlabel_list, ylabel_list, title_list,cbar_label_list, \
                      fontsize=15,vlims=vlims, cbar_orientation = 'bottom', axis_off=True, savefig=savefig)

print(' rmse before =', np.nanstd( dm_mode) )

print(' rmse after =', np.nanstd( residual)) 








# init a phasescreen to roll across DM 

Nx_act = 12 # actuators across DM 

D = 1.8 #m effective diameter of the telescope

screen_pixels = Nx_act*2**3  #pixels inthe inital screen before projection onto DM

corner_indicies = [0, Nx_act-1, Nx_act * (Nx_act-1), -1] 

scrn_scaling_factor =  0.1 

rows_to_jump = 2 # how many rows to jump on initial phase screen for each Baldr loop

distance_per_correction = rows_to_jump * D/screen_pixels # effective distance travelled by turbulence per AO iteration 
print(f'{rows_to_jump} rows jumped per AO command in initial phase screen of {screen_pixels} pixels. for {D}m mirror this corresponds to a distance_per_correction = {distance_per_correction}m')

scrn = aotools.infinitephasescreen.PhaseScreenVonKarman(nx_size=screen_pixels, pixel_scale=D/screen_pixels,r0=0.1,L0=12)

disturbance_cmd = util.create_phase_screen_cmd_for_DM(scrn,  scaling_factor=scrn_scaling_factor   , drop_indicies = corner_indicies, plot_cmd=False)

plt.figure()
plt.imshow( util.get_DM_command_in_2D(disturbance_cmd, Nx_act=12) )
plt.colorbar()
plt.title( 'initial Kolmogorov aberration to apply to DM')
plt.savefig(fig_path + exper_path + 'initial_phasescreen.png')


# get theoretical pupil and create filter for outer perimeter to include in our telemetry 
# sydney Baldr using DMLP1180 Longpass Dichroic Mirrors/Beamsplitters: 1180 nm Cut-Off Wavelength (Baldr in reflection)
# C-RED 2 cut on at 900nm 
# black body source at 1900 K 
# therefore central wavelength
central_lambda = util.find_central_wavelength(lambda_cut_on=900e-9, lambda_cut_off=1180e-9, T=1900)
print(f"The central wavelength is {central_lambda * 1e9:.2f} nm")


wvl = 0.95 # 1e6 * central_lambda # 0.900 #1.040 # um  
phase_shift = util.get_phasemask_phaseshift( wvl= wvl, depth = phasemask.phasemask_parameters[phasemask_name]['depth'] )
mask_diam = 1e-6 * phasemask.phasemask_parameters[phasemask_name]['diameter']
N0_theory0, I0_theory0 = util.get_theoretical_reference_pupils( wavelength = wvl*1e-6 ,F_number = 21.2, mask_diam = mask_diam,\
                                        diameter_in_angular_units = False,  phaseshift = phase_shift , padding_factor = 4, \
                                        debug= False, analytic_solution = True )

M = I0_theory0.shape[0]
N = I0_theory0.shape[1]

m = zwfs.I0.shape[1]
n = zwfs.I0.shape[0]

# A = pi * r^2 => r = sqrt( A / pi)
new_radius = (zwfs.pupil_pixel_filter.sum()/np.pi)**0.5
x_c = np.mean( np.unravel_index( np.where( zwfs.pupil_pixel_filter ), zwfs.I0.shape)[1])
y_c = np.mean( np.unravel_index( np.where( zwfs.pupil_pixel_filter ), zwfs.I0.shape)[0])

I0_theory = util.interpolate_pupil_to_measurement( N0_theory0, I0_theory0, M, N, m, n, x_c, y_c, new_radius)

N0_theory = util.interpolate_pupil_to_measurement( N0_theory0, N0_theory0, M, N, m, n, x_c, y_c, new_radius)





"""im_list = [I0_theory/np.max(I0_theory) , I0/np.max(I0), I0_theory/np.max(I0_theory) - I0/np.max(I0)]
xlabel_list = [None, None, None]
ylabel_list = [None, None, None]
title_list = ['theory', 'measured', 'residual']
cbar_label_list = [r'normalized intensity',r'normalized intensity', r'normalized intensity'] 
savefig = fig_path + f'I0_theory_vs_meas_mask-{phasemask_name}.png' #f'mode_reconstruction_images/phase_reconstruction_example_mode-{mode_indx}_basis-{phase_ctrl.config["basis"]}_ctrl_modes-{phase_ctrl.config["number_of_controlled_modes"]}ctrl_act_diam-{phase_ctrl.config["dm_control_diameter"]}_readout_mode-12x12.png'

util.nice_heatmap_subplots( im_list , xlabel_list, ylabel_list, title_list, cbar_label_list, fontsize=15, axis_off=True, cbar_orientation = 'bottom', savefig=savefig)
"""

pupil_outer_perim_filter = (~zwfs.bad_pixel_filter * (abs( I0_theory - N0_theory ) > 0.1 ).reshape(-1) * (~zwfs.pupil_pixel_filter) )

plt.figure()
plt.imshow( pupil_outer_perim_filter.reshape(zwfs.I0.shape) ) 
plt.savefig( fig_path + exper_path + 'outer_pupil_filter.png')

# last minute check 
zwfs.dm.send_data(zwfs.dm_shapes['flat_dm'] )
phasemask_centering_tool.move_relative_and_get_image(zwfs, phasemask, savefigName=fig_path + 'delme.png')



# create reference pupil 
I0, N0 = util.get_reference_images(zwfs, phasemask, theta_degrees=11.8, number_of_frames=256, \
compass = True, compass_origin=None, savefig= fig_path + exper_path + f'initial_reference_pupils.png' )



itera = 3

# init our controllers 

rho = 0 * np.ones( I2M_HO.shape[0] )
kp_leak = 0 * np.ones( I2M_HO.shape[0] )
lower_limit_leak = -100 * np.ones( I2M_HO.shape[0] )
upper_limit_leak = 100 * np.ones( I2M_HO.shape[0] )

leak = LeakyIntegrator(rho=rho, kp=kp_leak, lower_limit=lower_limit_leak, upper_limit=upper_limit_leak )

kp = 0. * np.ones( I2M_TT.shape[0] )
ki = 0. * np.ones( I2M_TT.shape[0] )
kd = 0. * np.ones( I2M_TT.shape[0] )
setpoint = np.zeros( I2M_TT.shape[0] )
lower_limit_pid = -100 * np.ones( I2M_TT.shape[0] )
upper_limit_pid = 100 * np.ones( I2M_TT.shape[0] )

pid = PIDController(kp, ki, kd, upper_limit_pid, lower_limit_pid, setpoint)

s_list = []
e_TT_list = []
u_TT_list = []
c_TT_list = []
e_HO_list = []
u_HO_list = []
c_HO_list = []
atm_disturb_list = []
dm_disturb_list = []
rmse_list = []
flux_outside_pupil_list = []
residual_list = []
close_after = 10 

pid.reset() 
leak.reset()


disturbance_cmd = 0.5 * reco_data_f['M2C_4RECO'].data.T[0] #+ reco_data_f['M2C_4RECO'].data.T[6] 
zwfs.dm.send_data(zwfs.dm_shapes['flat_dm'] + dm_pupil_filt * disturbance_cmd) # only apply in registered pupil 
time.sleep(0.1)

for it in range(40):
    
    if it > close_after : # close after 
        pid.kp = 1 * np.ones( I2M_TT.shape[0] )
        pid.ki = 0.3 * np.ones( I2M_TT.shape[0] )
        
        #leak.rho[2:5] = 0.2 #* np.ones( I2M_HO.shape[0] )
        #leak.kp[2:5] = 0.5

        cmd2opd = 3200
        im_list =  [  sig, util.get_DM_command_in_2D( cmd2opd * disturbance_cmd),  util.get_DM_command_in_2D( cmd2opd * c_TT),\
                      util.get_DM_command_in_2D( cmd2opd * c_HO),  util.get_DM_command_in_2D( cmd2opd * (disturbance_cmd - c_HO - c_TT) )] 
        xlabel_list = [ "" for _ in im_list]
        ylabel_list = [ "" for _ in im_list]
        title_list = [ "ZWFS signal", "aberration" , "reco. TT", "reco. HO", "residuals"]
        cbar_label_list =  [ "ADU", "OPD [nm]",  "OPD [nm]" ,"OPD [nm]", "OPD [nm]"]
        vlims=[[np.min(sig),np.max(sig)]] + [[np.min(cmd2opd * disturbance_cmd), np.max(cmd2opd * disturbance_cmd)] for _ in im_list[1:]]
        savefig = fig_path + 'delme.png' 
        util.nice_heatmap_subplots( im_list , xlabel_list, ylabel_list, title_list,cbar_label_list, \
                            fontsize=15, cbar_orientation = 'bottom', vlims=vlims, axis_off=True, savefig=savefig)
        _ = input('next?') 

    i = np.mean( zwfs.get_some_frames(number_of_frames=20, apply_manual_reduction=True),axis=0) #z.detection_chain( test_field, FPM_on=True, include_shotnoise=True, ph_per_s_per_m2_per_nm=True, grids_aligned=True, replace_nan_with=0 )
    #o = #z.detection_chain( test_field, FPM_on=False, include_shotnoise=True, ph_per_s_per_m2_per_nm=True, grids_aligned=True, replace_nan_with=0 )
    
    sig = i / np.mean( i ) -  I0 / np.mean( I0 ) # I0_theory/ np.mean(I0_theory) #

    # update distrubance after measurement 
    #for _ in range(rows_to_jump):
    #    scrn.add_row()
    #disturbance_cmd = np.array( util.create_phase_screen_cmd_for_DM(scrn,  scaling_factor=scrn_scaling_factor   , drop_indicies = corner_indicies, plot_cmd=False) )


    e_TT = I2M_TT @ sig.reshape(-1)[pupil_pixels]
    
    u_TT = pid.process( e_TT )
    
    c_TT = M2C_TT @ u_TT 
    
    e_HO = I2M_HO @ sig.reshape(-1)[pupil_pixels]

    u_HO = leak.process( e_HO )
    
    c_HO = M2C_HO @ u_HO 

    #c = R @ sig

    zwfs.dm.send_data( zwfs.dm_shapes['flat_dm'] + disturbance_cmd - c_HO - c_TT ) # same way to rtc PID 
    time.sleep(0.05)  
    # only measure residual in the registered pupil on DM 
    residual =  (disturbance_cmd - c_HO - c_TT)[dm_pupil_filt]
    rmse = np.nanstd( residual )
    
    # telemetry 
    s_list.append( sig )
    e_TT_list.append( e_TT )
    u_TT_list.append( u_TT )
    c_TT_list.append( c_TT )
    
    e_HO_list.append( e_HO )
    u_HO_list.append( u_HO )
    c_HO_list.append( c_HO )
    
    atm_disturb_list.append( scrn.scrn )
    dm_disturb_list.append( disturbance_cmd )
    
    residual_list.append( residual )
    rmse_list.append( rmse )
    flux_outside_pupil_list.append( np.sum( sig.reshape(-1)[pupil_outer_perim_filter] ) )
    print( it, f'rmse = {rmse}, flux outside = {flux_outside_pupil_list[-1]}' )



# write telemetry to file 

# Dictionary of lists and their names
lists_dict = {
    "s_list": s_list,
    "e_TT_list": e_TT_list,
    "u_TT_list": u_TT_list,
    "c_TT_list": c_TT_list,
    "e_HO_list": e_HO_list,
    "u_HO_list": u_HO_list,
    "c_HO_list": c_HO_list,
    "pid_kp_list": pid.kp,
    "pid_ki_list": pid.ki,
    "pid_kd_list": pid.kd,
    "leay_kp_list": leak.kp,
    "leay_rho_list": leak.rho,
    "atm_disturb_list": atm_disturb_list,
    "dm_disturb_list": dm_disturb_list,
    "rmse_list": rmse_list,
    "residual_list": residual_list,
    "flux_outside_pupil_list":flux_outside_pupil_list
}

# Create a list of HDUs (Header Data Units)
hdul = fits.HDUList()

# Add each list to the HDU list as a new extension
for list_name, data_list in lists_dict.items():
    # Convert list to numpy array for FITS compatibility
    data_array = np.array(data_list, dtype=float)  # Ensure it is a float array or any appropriate type

    # Create a new ImageHDU with the data
    hdu = fits.ImageHDU(data_array)

    # Set the EXTNAME header to the variable name
    hdu.header['EXTNAME'] = list_name

    # Append the HDU to the HDU list
    hdul.append(hdu)

# Write the HDU list to a FITS file
hdul.writeto(fig_path + exper_path + f'telemetry_{itera}.fits', overwrite=True)





"""



                I_plus = np.mean( I_plus_list, axis = 0).reshape(-1)  # flatten so can filter with pupil_pixels
                I_plus *= 1/np.mean( I_plus )

                I_minus = np.mean( I_minus_list, axis = 0).reshape(-1)  # flatten so can filter with pupil_pixels
                I_minus *= 1/np.mean( I_minus )

                errsig = (I_plus - I_minus)[np.array( pupil_pixels )]
                IM.append( list(  errsig.reshape(-1) ) ) #toook out 1/poke_amp *

"""





