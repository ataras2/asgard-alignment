from . import utilities as util 
#from . import hardware 

import matplotlib.pyplot as plt 
import numpy as np
import time 
import datetime
class phase_controller_1():
    """
    linear interaction model on internal calibration source
    """
   
    def __init__(self, basis_name = 'Zonal', number_of_controlled_modes = 140 , config_file = None):
        
        # if config file  provided we ignore basis_name and number_of_controlled_modes
        if type(config_file)==str:
            if config_file.split('.')[-1] == 'json':
                with open('data.json') as json_file:
                    self.config  = json.load(json_file)
            else:
                raise TypeError('input config_file is not a json file')
               
        elif config_file==None:
            # class generic controller config parameters
            self.config = {}
            self.config['telescopes'] = ['AT1']
            self.config['fpm'] = 1  #0 for off, positive integer for on a particular phase dot
            self.config['basis'] = basis_name  # either Zernike, Zonal, KL, fourier, or WFS_Eigenmodes
            self.config['number_of_controlled_modes'] = number_of_controlled_modes # number of controlled modes
            self.config['source'] = 1
           
            self.ctrl_parameters = {} # empty dictionary cause we have none
            #for tel in self.config['telescopes']:
            
            self.config['Kp'] = [0 for _ in range( self.config['number_of_controlled_modes'] )] # proportional gains
            self.config['Ki'] = [0 for _ in range( self.config['number_of_controlled_modes'] )] # integral gains

            self.config['dm_control_diameter'] = 12 # diameter (actuators) of active actuators
            self.config['dm_control_center'] = [0,0] # in-case of mis-alignments we can offset control basis on DM
            # all reference intensities normalized by sum of pixels in cropping region. 
            self.I0 = None #reference intensity filtered over defined pupil with FPM IN (1D array)
            self.N0 = None #reference intensity filtered over defined pupil with FPM OUT (1D array)
            self.I0_2D = None # reference intensity with FPM IN without filtering (2D array)
            self.N0_2D = None # reference intensity with FPM OUT without filtering (2D array)
            self.b = None #ZWFS gain in pixel space filtered over defined pupil - needs to be calculated with I0, N0 measurement 
            self.b_2D = None #ZWFS gain in pixel space filtered over defined pupil - needs to be calculated with I0, N0 measurement 

            self.theta = 3.14/180 * 110 # (radian) phase shift of phase mask 
            self.FPM_diam = 49e-6 # (m) - focal plane mask dot diameter 
            self.FPM_depth = 0.932e-6 # (m) - focal plane mask dot diameter 
            self.fratio = 21.2 # focal length / D of ZWFS.
            self.wvl0 = 1.29e-6 # (m) - central wavelength of system (default is center of SLS202 L/M thermal source measured in CRED3 detector)           

            # mode to command matrix
            M2C = util.construct_command_basis( basis=self.config['basis'] , \
                    number_of_modes = self.config['number_of_controlled_modes'],\
                        Nx_act_DM = 12,\
                           Nx_act_basis = self.config['dm_control_diameter'],\
                               act_offset= self.config['dm_control_center'], without_piston=True) # cmd = M2C @ mode_vector, i.e. mode to command matrix
           
                
            # used for estimating MAP inverse of interaction matrix  (minimum variance esti.)
            self.noise_cov = None # should be in pixel space of the filtered pupil 
            self.phase_cov = None # should be in mode space

            self.config['M2C'] = M2C # 
                #self.config[tel]['control_parameters'] = [] # empty list cause we have none
            self.config['active_actuator_filter'] = (abs(np.sum( self.config['M2C'], axis=1 )) > 0 ).astype(bool)
           

            self.config['active_actuator_indicies'] = np.where( abs(np.sum( self.config['M2C'], axis=1 )) > 0 )[0]
             
#plt.imshow( util.get_DM_command_in_2D( abs(np.sum( phase_ctrl.config['M2C'], axis=1 )) > 0 ) ); plt.show() #<-this can be used to check the actuator filter makes sense. 

        else:
            raise TypeError( 'config_file type is wrong. It must be None or a string indicating the config file to read in' )


    def change_control_basis_parameters(self,  number_of_controlled_modes, basis_name ,dm_control_diameter=None, dm_control_center=None,controller_label=None):
        # standize updating of control basis parameters so no inconsistencies
        # warning no error checking here! 
        # controller_label only needs to be provided if updating to 'WFS_Eigenmodes' because it needs the covariance of the IM which is stored in self.ctrl_parameters[controller_label]['IM']

        if basis_name != 'WFS_Eigenmodes':
            self.config['basis'] = basis_name
            self.config['number_of_controlled_modes'] = number_of_controlled_modes

            if dm_control_diameter!=None: 
                self.config['dm_control_diameter'] =  dm_control_diameter
            if dm_control_center!=None:
                self.config['dm_control_center']  =  dm_control_center
        
            # mode to command matrix
            M2C = util.construct_command_basis( basis=self.config['basis'] , \
                        number_of_modes = self.config['number_of_controlled_modes'],\
                            Nx_act_DM = 12,\
                               Nx_act_basis = self.config['dm_control_diameter'],\
                                   act_offset= self.config['dm_control_center'], without_piston=True)

            self.config['M2C'] = M2C 

            self.config['Kp'] = [0 for _ in range( self.config['number_of_controlled_modes'] )] # proportional gains
            self.config['Ki'] = [0 for _ in range( self.config['number_of_controlled_modes'] )] # integral gains
            self.config['active_actuator_filter'] = (abs(np.sum( self.config['M2C'], axis=1 )) > 0 ).astype(bool)
 
            self.config['active_actuator_indicies'] = np.where( abs(np.sum( self.config['M2C'], axis=1 )) > 0 )[0]

        elif  basis_name == 'WFS_Eigenmodes':  
            self.config['basis'] = basis_name
            # treated differently because we need covariance of interaction matrix to build the modes. 
            # number of modes infered by size of IM covariance matrix 
            try:
                IM = self.ctrl_parameters[controller_label]['IM']  #interaction matrix
            except:
                raise TypeError( 'nothing found in self.ctrl_parameters[controller_label]["IM"]' ) 

            M2C_old = self.config['M2C']

            # some checks its a covariance matrix
            if not hasattr(IM, '__len__'):
                raise TypeError('IM needs to be 2D matrix' )
                if len(np.array(IM).shape) != 2:
                    raise TypeError('IM  needs to be 2D matrix')

            #spectral decomposition 
            IM_covariance = np.cov( IM )
            U,S,UT = np.linalg.svd( IM_covariance )


            # project our old modes represented in the M2C matrix to new ones
            M2C_unnormalized  = M2C_old @ U.T

            # normalize 
            M2C = [] 
            for m in np.array(M2C_unnormalized).T:
                M2C.append( 1/np.sqrt( np.sum( m*m ) ) * m  ) # normalize all modes <m|m>=1

            M2C = np.array(M2C).T            

            if len(M2C.T)  !=  self.config['number_of_controlled_modes']:
                print( '\n\n==========\nWARNING: number of mode  self.config["number_of_controlled_modes"] != len( IM )')

            if dm_control_diameter!=None: 
                print( ' cannot update dm control diameter with eigenmodes, it inherits it from the previous basis') 
            if dm_control_center!=None:
                print( ' cannot update dm control center with eignmodes, it inherits it from the previous basis') 

            self.config['M2C'] = M2C 

            self.config['Kp'] = [0 for _ in range( self.config['number_of_controlled_modes'] )] # proportional gains
            self.config['Ki'] = [0 for _ in range( self.config['number_of_controlled_modes'] )] # integral gains
            self.config['active_actuator_filter'] = (abs(np.sum( self.config['M2C'], axis=1 )) > 0 ).astype(bool)
 
            self.config['active_actuator_indicies'] = np.where( abs(np.sum( self.config['M2C'], axis=1 )) > 0 )[0]



    def build_control_model_2(self, ZWFS, poke_amp = -0.15, label='ctrl_1', poke_method='single_sided_poke', inverse_method='MAP', debug = True):
        # newer version without reliance on FPM out. (before sydney test)

        # fourier tip to go off phase mask  
        fourier_basis = util.construct_command_basis( basis='fourier', number_of_modes = 40, Nx_act_DM = 12, Nx_act_basis = 12, act_offset=(0,0), without_piston=True)
        tip = fourier_basis[:,0]

        imgs_to_mean =  256

        ZWFS.dm.send_data(0.5 + 2 * tip ) # move off phase mask 
        time.sleep(0.1)
        N0_list = ZWFS.get_some_frames(number_of_frames = imgs_to_mean, apply_manual_reduction = True ) #REFERENCE INTENSITY WITH FPM OUT
        N0 = np.mean( N0_list, axis = 0 )


        ZWFS.dm.send_data( ZWFS.dm_shapes['flat_dm'] )
        time.sleep(0.1)
        I0_list = ZWFS.get_some_frames(number_of_frames = imgs_to_mean, apply_manual_reduction = True ) #REFERENCE INTENSITY WITH FPM IN
        I0 = np.mean( I0_list, axis = 0 )

        # === ADD ATTRIBUTES 
        self.I0 = I0.reshape(-1)[np.array( ZWFS.pupil_pixels )] / np.mean( I0 ) # append reference intensity over defined     pupil with FPM IN 
        self.N0 = N0.reshape(-1)[np.array( ZWFS.pupil_pixels )] / np.mean( N0 ) # append reference intensity over defined pupil with FPM OUT 

        # === also add the unfiltered so we can plot and see them easily on square grid after 
        self.I0_2D = I0 / np.mean( I0 ) # 2D array (not filtered by pupil pixel filter)  
        self.N0_2D = N0 / np.mean( N0 )# append reference intensity over defined pupil with FPM OUT 

        modal_basis = self.config['M2C'].copy().T # more readable
        IM=[] # init our raw interaction matrix 

        if poke_method=='single_sided_poke': # just poke one side  
            for i,m in enumerate(modal_basis):
                print(f'executing cmd {i}/{len(modal_basis)}')           
                ZWFS.dm.send_data( list( ZWFS.dm_shapes['flat_dm'] + poke_amp * m )  )
                time.sleep(0.1)
                img_list = ZWFS.get_some_frames(number_of_frames = imgs_to_mean, apply_manual_reduction = True ) # to take median of 
                I = np.mean( img_list, axis = 0).reshape(-1) 

                # IMPORTANT : we normalize by mean over total image region (post reduction) (NOT FILTERED )... 
                I *= 1/np.mean( I ) # we normalize by mean over total region! 
                
                # then filter for getting error signal 
                errsig =  self.get_img_err( I[np.array( ZWFS.pupil_pixels )] )

                IM.append( list(  errsig.reshape(-1) ) ) #toook out 1/poke_amp *

        elif poke_method=='double_sided_poke':
            for i,m in enumerate(modal_basis):
                print(f'executing cmd {i}/{len(modal_basis)}')
                I_plus_list = []
                I_minus_list = []
                imgs_to_mean = 10
                for sign in [(-1)**n for n in range(10)]: #[-1,1]:
                    ZWFS.dm.send_data( list( ZWFS.dm_shapes['flat_dm'] + sign * poke_amp/2 * m )  )
                    time.sleep(0.05)
                    if sign > 0:
                        I_plus_list += ZWFS.get_some_frames(number_of_frames = imgs_to_mean, apply_manual_reduction = True )
                        #I_plus *= 1/np.mean( I_plus )
                    if sign < 0:
                        I_minus_list += ZWFS.get_some_frames(number_of_frames = imgs_to_mean, apply_manual_reduction = True )
                        #I_minus *= 1/np.mean( I_minus )

                I_plus = np.mean( I_plus_list, axis = 0).reshape(-1)  # flatten so can filter with ZWFS.pupil_pixels
                I_plus *= 1/np.mean( I_plus )

                I_minus = np.mean( I_minus_list, axis = 0).reshape(-1)  # flatten so can filter with ZWFS.pupil_pixels
                I_minus *= 1/np.mean( I_minus )

                errsig = (I_plus - I_minus)[np.array( ZWFS.pupil_pixels )]
                IM.append( list(  errsig.reshape(-1) ) ) #toook out 1/poke_amp *

        else:
            raise TypeError( ' no matching method for building control model. Try (for example) method="single_side_poke"')
            #U, S, Vt = np.linalg.svd( IM , full_matrices=True)

        # convert to array 
        IM = np.array( IM )  

        # intensity to mode matrix 
        if inverse_method == 'pinv':
            I2M = np.linalg.pinv( IM )

        elif inverse_method == 'MAP': # minimum variance of maximum posterior estimator 
            if not hasattr(self.noise_cov, '__len__'): 
                noise_cov = np.eye( np.array(IM).shape[1] ) #built along IM cols which correspond to pupil fitered pixels
                print( 'phase controller does not have noise covariance matrix.\nbuild it with the command update_noise_model( self, zwfs, number_of_frames = 1000 )')
            else:
                noise_cov = np.array( self.noise_cov )

            if not hasattr(self.phase_cov, '__len__'): 
                phase_cov = np.eye( np.array(IM).shape[0] )
                print( 'phase controller does not have phase covariance matrix. Using indentity.')
            else: 
                phase_cov = np.array( self.phase_cov )

            #minimum variance of maximum posterior estimator 
            I2M = (phase_cov @ IM @ np.linalg.inv(IM.T @ phase_cov @ IM + noise_cov) ).T #have to transpose to keep convention.. although should be other way round
            
        
        # get tip/tilt reconstructors from I2M matrix (IN MODAL SPACE!)
        # NOTE: This ASSUMES that index 0, 1 correspond to tip/tilt

        if self.config['basis']=='Zonal':
            # if zonal then modes are actuators (in DM space), mode to command (M2C) is identity and CM = I2M.T.

            zernike_basis = util.construct_command_basis( basis='Zernike', number_of_modes = 20, Nx_act_DM = 12, Nx_act_basis = 12, act_offset=(0,0), without_piston=True)
            tip = zernike_basis[:,0]
            tilt = zernike_basis[:,1]
            # Another option would be to get eigen modes U, S, Vt = svd(IM@IM.T), U[0], U[1] as tip/tilt on zonal basis? 
            R_TT, R_HO = util.project_matrix( I2M.T , projection_vector_list = [tip, tilt] )

        elif self.config['basis']!='Zonal':
            # then in some modal space
            # we can safely assume first and second indicies are tip/tilt 
            # we project onto modal space defined by intensity to mode matrix I2M 
            tip = np.zeros( I2M.shape[1] )
            tip[0] = 1
            tilt = np.zeros( I2M.shape[1] )
            tilt[1] = 1
        
            R_TT, R_HO = util.project_matrix( I2M.T , projection_vector_list = [tip, tilt] )
        
        """elif self.config['basis']=='fourier':
            # seperate case si
            fourier_basis = util.construct_command_basis( basis='fourier', number_of_modes = 20, Nx_act_DM = 12, Nx_act_basis = 12, act_offset=(0,0), without_piston=True)
            tip = fourier_basis[:,0]
            tilt = fourier_basis[:,3]
            R_TT, R_HO = util.project_matrix( I2M.T , projection_vector_list = [tip, tilt] )
        """

        # class specific controller parameters
        ctrl_parameters = {}

        ctrl_parameters['active'] = 0 # 0 if unactive, 1 if active (should only have one active phase controller)

        ctrl_parameters['ref_pupil_FPM_out'] = N0 # <------- NOT INCLUDED! 

        ctrl_parameters['poke_amp'] = poke_amp # amplitude used for IM reconstruction
        
        ctrl_parameters['ref_pupil_FPM_in'] = I0

        ctrl_parameters['pupil_pixels'] = ZWFS.pupil_pixels

        ctrl_parameters['secondary_pupil_pixels'] = ZWFS.secondary_pixels

        ctrl_parameters['outside_pupil_pixels'] = ZWFS.outside_pixels

        ctrl_parameters['dm_center_ref_pixels'] = ZWFS.dm_center_ref_pixels

        ctrl_parameters['M2C_4reco'] = poke_amp * self.config['M2C']  # mode to command matrix normalized to IM poke amps

        ctrl_parameters['IM'] = IM # interaction matrix
       
        ctrl_parameters['I2M'] = I2M # intensity to mode matrix 
        
        ### ASSUMES INDEX 0, 1 IN BASIS CORRESPOND TO TIP/TILT <- This is not the case for Fourier basis!! 
        ctrl_parameters['R_TT'] = R_TT # projection of signal to tip / tilt mode amplitudes
        
        ctrl_parameters['R_HO'] = R_TT # projection of signal to higher order mode amplitudes
        # e.g. cmd_TT ~ M2C @ R_TT @ signal
        ### 
        
        #unfiltered control matrix (note in zonal method M2C is just identity matrix)
        CM = ctrl_parameters['M2C_4reco'] @ I2M.T

        ctrl_parameters['CM'] = CM # control matrix (intensity to DM cmd)
       
        ctrl_parameters['P2C'] = None # pixel to cmd registration (i.e. what region)
       
        self.ctrl_parameters[label] = ctrl_parameters
       





    def build_control_model_dontuse(self, ZWFS, poke_amp = -0.15, label='ctrl_1', debug = True):
        

        # remember that ZWFS.get_image automatically crops at corners ZWFS.pupil_crop_region
        ZWFS.states['busy'] = 1
        #update_references = int( input('get new reference intensities (1/0)') )

        imgs_to_median = 10 # how many images do we take the median of to build signal the reference signals        
        if 1 : #update_references : #| ( (self.I0==None) | (self.N0==None) ):


            # check other states match such as source etc
        
            # =========== PHASE MASK OUT 
            #hardware.set_phasemask( phasemask = 'out' ) # THIS DOES NOTHING SINCE DONT HAVE MOTORS YET.. for now we just look at the pupil so we can manually move phase mask in and out. 
            _ = input('MANUALLY MOVE PHASE MASK OUT OF BEAM, PRESS ANY KEY TO BEGIN' )
            util.watch_camera(ZWFS, frames_to_watch = 50, time_between_frames=0.05)

            #ZWFS.states['fpm'] = 0
    
            N0_list = []
            for _ in range(imgs_to_median):
                N0_list.append( ZWFS.get_image(  ) ) #REFERENCE INTENSITY WITH FPM IN
            N0 = np.median( N0_list, axis = 0 ) 
            #put self.config['fpm'] phasemask on-axis (for now I only have manual adjustment)

            # =========== PHASE MASK IN 
            #hardware.set_phasemask( phasemask = 'posX' ) # # THIS DOES NOTHING SINCE DONT HAVE MOTORS YET.. for now we just look at the pupil so we can manually move phase mask in and out. 
            _ = input('MANUALLY MOVE PHASE MASK BACK IN, PRESS ANY KEY TO BEGIN' )
            util.watch_camera(ZWFS, frames_to_watch = 50, time_between_frames=0.05)

            #ZWFS.states['fpm'] = self.config['fpm']

            I0_list = []
            for _ in range(imgs_to_median):
                I0_list.append( ZWFS.get_image(  ) ) #REFERENCE INTENSITY WITH FPM IN
            I0 = np.median( I0_list, axis = 0 ) 
        
            # === ADD ATTRIBUTES 
            self.I0 = I0.reshape(-1)[np.array( ZWFS.pupil_pixels )] / np.mean( I0 ) # append reference intensity over defined     pupil with FPM IN 
            self.N0 = N0.reshape(-1)[np.array( ZWFS.pupil_pixels )] / np.mean( N0 ) # append reference intensity over defined pupil with FPM OUT 

            # === also add the unfiltered so we can plot and see them easily on square grid after 
            self.I0_2D = I0 / np.mean( I0 ) # 2D array (not filtered by pupil pixel filter)  
            self.N0_2D = N0 / np.mean( N0 ) # 2D array (not filtered by pupil pixel filter)

        #  also update b attribute in phase controller
        # !!!!!!!!!!!!!!!!!
        # ---------------- GENERALLY WORKS, COMMENTED OUT FOR NOW TO SPEED THINGS UP!
        #self.update_b( ZWFS, self.I0_2D, self.N0_2D )
        

        modal_basis = self.config['M2C'].copy().T # more readable
        IM=[] # init our raw interaction matrix 

        for i,m in enumerate(modal_basis):

            print(f'executing cmd {i}/{len(modal_basis)}')
            
            ZWFS.dm.send_data( list( ZWFS.dm_shapes['flat_dm'] + poke_amp * m )  )
            time.sleep(0.05)
            img_list = []  # to take median of 
            for _ in range(imgs_to_median):
                img_list.append( ZWFS.get_image( ).reshape(-1)[np.array( ZWFS.pupil_pixels )] )
                time.sleep(0.01)
            I = np.median( img_list, axis = 0) 

            if (I.shape == self.I0.shape) & (I.shape == self.N0.shape):
                # !NOTE! we take median of pupil reference intensity with FPM out (self.N0)
                # we do this cause we're lazy and do not want to manually adjust FPM every iteration (we dont have motors here) 
                # real system prob does not want to do this and normalize pixel wise. 
 
                # HAVE TO NORMALIZE BEFORE SENDING TO get_img_err
                I *= 1/np.mean( I ) 
                errsig =  self.get_img_err( I ) #np.array( ( (I - self.I0) / np.median( self.N0 ) ) )
            else: 
                raise TypeError(" reference intensity shapes do not match shape of current measured intensity. Check phase_controller.I0 and/or phase_controller.N0 attributes. Workaround would be to retake these. ")
            
            #D delta_c = delta_I => D *poke_amp * I = delta_I -> D = 1/poke_amp * delta_I
            IM.append( list( 1/poke_amp * errsig.reshape(-1) ) ) # toook out 1/poke_amp *

        # FLAT DM WHEN DONE
        ZWFS.dm.send_data( list( ZWFS.dm_shapes['flat_dm'] ) )
               
        # SVD
        U, S, Vt = np.linalg.svd( IM , full_matrices=True)

        # filter number of modes in eigenmode space when using zonal control  
        if self.config['basis'] == 'Zonal': # then we filter number of modes in the eigenspace of IM 
            S_filt = S > 0 #(i.e. we dont filter here) #S >= np.min(S) # we consider the highest eigenvalues/vectors up to the number_of_controlled_modes
            Sigma = np.zeros( np.array(IM).shape, float)
            np.fill_diagonal(Sigma, S[S_filt], wrap=False) #

        else: # else #modes decided by the construction of modal basis. We may change their gains later
            S_filt = S > 0 # S > S[ np.min( np.where( abs(np.diff(S)) < 1e-2 )[0] ) ]
            Sigma = np.zeros( np.array(IM).shape, float)
            np.fill_diagonal(Sigma, S[S_filt], wrap=False) #

        if debug:
            #plotting DM eigenmodes to see which to filter 
            print( ' we can only easily plot eigenmodes if pupil_pixels is square region!' )
            """
            fig,ax = plt.subplots(6,6,figsize=(15,15))
            
            for i,axx in enumerate( ax.reshape(-1) ) :
                axx.set_title(f'eigenmode {i}')
                axx.imshow( util.get_DM_command_in_2D( U[:,i] ) )
            plt.suptitle( 'DM EIGENMODES' ) #f"rec. cutoff at i={np.min( np.where( abs(np.diff(S)) < 1e-2 )[0])}", fontsize=14)
            plt.show()
            """


        # intensity to mode matrix 
        I2M = np.linalg.pinv( U @ Sigma @ Vt ) # C = A @ M #1/abs(poke_amp)
        
        #control matrix (note in zonal method M2C is just identity matrix)
        CM = self.config['M2C'] @ I2M.T # < - To check with normalization of modes

        # class specific controller parameters
        ctrl_parameters = {}
       
        ctrl_parameters['active'] = 0 # 0 if unactive, 1 if active (should only have one active phase controller)

        ctrl_parameters['ref_pupil_FPM_out'] = N0

        ctrl_parameters['ref_pupil_FPM_in'] = I0

        ctrl_parameters['pupil_pixels'] = ZWFS.pupil_pixels

        ctrl_parameters['secondary_pupil_pixels'] = ZWFS.secondary_pixels

        ctrl_parameters['outside_pupil_pixels'] = ZWFS.outside_pixels

        ctrl_parameters['dm_center_ref_pixels'] = ZWFS.dm_center_ref_pixels

        ctrl_parameters['IM'] = IM # interaction matrix
       
        ctrl_parameters['I2M'] = I2M # intensity to mode matrix 

        ctrl_parameters['M2C_4reco'] = poke_amp * self.config['M2C']  # mode to command matrix normalized to IM poke amps

        ctrl_parameters['CM'] = CM # control matrix (intensity to DM cmd)
       
        ctrl_parameters['P2C'] = None # pixel to cmd registration (i.e. what region)

        ZWFS.states['busy'] = 0
       
        self.ctrl_parameters[label] = ctrl_parameters
       
        if debug: # plot covariance of interaction matrix 
            plt.figure() 
            plt.title( 'Covariance of Interaciton Matrix' )
            plt.imshow( np.cov( self.ctrl_parameters[label]['IM'] ) )
            plt.colorbar()
            plt.show()




    def update_noise_model( self, zwfs, number_of_frames = 1000 ):
        cov_matrix = zwfs.estimate_noise_covariance( number_of_frames = 1000, where = 'pupil' )
        self.noise_cov = cov_matrix


    def update_b( self, ZWFS, I0_2D, N0_2D ):
        #I0 = reference image with FPM in (2D array - CANNOT be flatttened 1D array)
        #N0 = reference image with FPM out (2D array - CANNOT be flatttened 1D array)
        # even though I0_2D, N0_2D is attribute in self we allow user to specify new ones to update b
        image_filter = ZWFS.refpeak_pixel_filter | ZWFS.outside_pixel_filter

        b_pixel_space = util.fit_b_pixel_space(I0_2D, self.theta, image_filter , debug=False)
        # full fit over ZWFS image 
        self.b_2D =  b_pixel_space
        #flattened and filtered in pupil space 
        self.b = b_pixel_space.reshape(-1)[ZWFS.pupil_pixel_filter] 




    def get_img_err( self , img_normalized ):
        # input img has to be normalized by *mean or sum* of entire raw image in cropped region 
        # flattened and filtered in pupil region 
        #(i.e. get raw img then apply img.reshape(-1)[ ZWFS.pupil_pixels])

        # THIS MAY BE BETTER IMPLEMENTED IN zwfs module. 



        # experiment iwth ML estimator 
        # errsig =  np.array( ( (img_normalized - self.I0) / (2 * self.b * np.sin( self.theta ) ) ) )
        
        errsig =  np.array( img_normalized - self.I0 ) 

        return(  errsig )


    def update_FPM_OUT_reference(self, ZWFS, N0_2D):
        #N0_2D must be a 2D array of the estimate of the pupil intensity with the FPM mask out.
        #!!!! IMPORTANT !!! WE DO NOT NORMALIZE HERE - SO I0_2D SHOULD BE NORMALIZED BY MEAN OVER FULL PUPIL IMAGE REGION ON INPUT!
        #ZWFS must be provided since this holds the pixel filter for the wavefront sensing 

        # we should do some checking here that the input N0_2D matches number of pixels from ZWFS.get_image() 
        r1, r2, c1, c2 = self.pupil_crop_region
        if N0_2D.shape != [r2-r1, c2-c1]:
            raise TypeError(f'N0_2D needs to have shape {[r2-r1,c2-c1]} to match image size produced from the input ZWFS object.')  

        self.N0 =  N0_2D.reshape(-1)[np.array( ZWFS.pupil_pixels )]
        self.N0_2D =  N0_2D   
        

    def update_FPM_IN_reference(self, ZWFS, I0_2D):
        #i0_2D must be a 2D array of the estimate of the pupil intensity with the FPM mask out.
        #!!!! IMPORTANT !!! WE DO NOT NORMALIZE HERE - SO I0_2D SHOULD BE NORMALIZED BY MEAN OVER FULL PUPIL IMAGE REGION ON INPUT!
        #ZWFS must be provided since this holds the pixel filter for the wavefront sensing 

        # we should do some checking here that the input N0_2D matches number of pixels from ZWFS.get_image() 
        r1, r2, c1, c2 = self.pupil_crop_region
        if np.array(I0_2D).shape != [r2-r1, c2-c1]:
            raise TypeError(f'N0_2D needs to have shape {[r2-r1,c2-c1]} to match image size produced from the input ZWFS object.')  

        self.I0 = I0_2D.reshape(-1)[np.array( ZWFS.pupil_pixels )]
        self.I0_2D =  I0_2D   


    def measure_FPM_OUT_reference(self, ZWFS) : 
        
        imgs_to_median = 10 # how many images do we take the median of to build signal the reference signals 
        # check other states match such as source etc

        # =========== PHASE MASK OUT 
        #hardware.set_phasemask( phasemask = 'out' ) # THIS DOES NOTHING SINCE DONT HAVE MOTORS YET.. for now we just look at the pupil so we can manually move phase mask in and out. 
        _ = input('MANUALLY MOVE PHASE MASK OUT OF BEAM, PRESS ANY KEY TO BEGIN' )
        util.watch_camera(ZWFS, frames_to_watch = 50, time_between_frames=0.05)

        ZWFS.states['fpm'] = 0

        N0_list = []
        for _ in range(imgs_to_median):
            N0_list.append( ZWFS.get_image(  ) ) #REFERENCE INTENSITY WITH FPM IN
        N0 = np.median( N0_list, axis = 0 ) 
        #put self.config['fpm'] phasemask on-axis (for now I only have manual adjustment)

        # === ADD ATTRIBUTES 
        self.N0 =  1/np.mean( N0 ) * N0.reshape(-1)[np.array( ZWFS.pupil_pixels )] # append reference intensity over defined pupil with FPM OUT 

        # === also add the unfiltered so we can plot and see them easily on square grid after 
        self.N0_2D = 1/np.mean( N0 ) * N0 # 2D array (not filtered by pupil pixel filter) 

    def measure_FPM_IN_reference(self, ZWFS):

        imgs_to_median = 10 # how many images do we take the median of to build signal the reference signals 
        # =========== PHASE MASK IN 
        #hardware.set_phasemask( phasemask = 'posX' ) # # THIS DOES NOTHING SINCE DONT HAVE MOTORS YET.. for now we just look at the pupil so we can manually move phase mask in and out. 
        _ = input('MANUALLY MOVE PHASE MASK BACK IN, PRESS ANY KEY TO BEGIN' )
        util.watch_camera(ZWFS, frames_to_watch = 50, time_between_frames=0.05)

        ZWFS.states['fpm'] = self.config['fpm']

        I0_list = []
        for _ in range(imgs_to_median):
            I0_list.append( ZWFS.get_image(  ) ) #REFERENCE INTENSITY WITH FPM IN
        I0 = np.median( I0_list, axis = 0 ) 
        
        # === ADD ATTRIBUTES 
        self.I0 = 1/np.mean( I0 ) * I0.reshape(-1)[np.array( ZWFS.pupil_pixels )] # append reference intensity over defined pupil with FPM IN 

        # === also add the unfiltered so we can plot and see them easily on square grid after 
        self.I0_2D = 1/np.mean( I0 ) * I0 # 2D array (not filtered by pupil pixel filter)  
  

    def control_phase(self, img, controller_name ):
        # look for active ctrl_parameters, return label
        im_err = self.get_img_err( img )
        cmd = im_err @ self.ctrl_parameters[controller_name]['CM'] 
       
        return( cmd )


    def plot_SVD_modes(self, ZWFS, ctrl_label, save_path=None):

        tstamp = datetime.datetime.now().strftime("%d-%m-%YT%H.%M.%S")

        IM = self.ctrl_parameters[ctrl_label]['IM']
        M2C = self.config['M2C']
        pupil_pixels = self.ctrl_parameters[ctrl_label]['pupil_pixels']
        U,S,Vt = np.linalg.svd( IM , full_matrices=True)

        #singular values
        plt.figure() 
        plt.semilogy(S) #/np.max(S))
        #plt.axvline( np.pi * (10/2)**2, color='k', ls=':', label='number of actuators in pupil')
        plt.legend() 
        plt.xlabel('mode index')
        plt.ylabel('singular values')
        if save_path!=None:
            plt.savefig(save_path + f'singularvalues_{tstamp}.png', bbox_inches='tight', dpi=300)
        plt.show()
        
        # THE IMAGE MODES 

        fig,ax = plt.subplots(8,8,figsize=(30,30))
        plt.subplots_adjust(hspace=0.1,wspace=0.1)
        for i,axx in enumerate(ax.reshape(-1)):
            # we filtered circle on grid, so need to put back in grid
            tmp =  ZWFS.pupil_pixel_filter.copy()
            vtgrid = np.zeros(tmp.shape)
            vtgrid[tmp] = Vt[i]
            axx.imshow( vtgrid.reshape(ZWFS.I0.shape ) ) #cp_x2-cp_x1,cp_y2-cp_y1) )
            #axx.set_title(f'\n\n\nmode {i}, S={round(S[i]/np.max(S),3)}',fontsize=5)
            axx.text( 10,10,f'{i}',color='w',fontsize=4)
            axx.text( 10,20,f'S={round(S[i]/np.max(S),3)}',color='w',fontsize=4)
            axx.axis('off')
            #plt.legend(ax=axx)
        plt.tight_layout()
        if save_path!=None:
            plt.savefig(save_path + f'det_eignmodes_{tstamp}.png',bbox_inches='tight',dpi=300)
        plt.show()
        
        # THE DM MODES 

        # NOTE: if not zonal (modal) i might need M2C to get this to dm space 
        # if zonal M2C is just identity matrix. 
        fig,ax = plt.subplots(8,8,figsize=(30,30))
        plt.subplots_adjust(hspace=0.1,wspace=0.1)
        for i,axx in enumerate(ax.reshape(-1)):
            axx.imshow( util.get_DM_command_in_2D( M2C @ U.T[i] ) )
            #axx.set_title(f'mode {i}, S={round(S[i]/np.max(S),3)}')
            axx.text( 1,2,f'{i}',color='w',fontsize=6)
            axx.text( 1,3,f'S={round(S[i]/np.max(S),3)}',color='w',fontsize=6)
            axx.axis('off')
            #plt.legend(ax=axx)
        plt.tight_layout()
        if save_path!=None:
            plt.savefig(save_path + f'dm_eignmodes_{tstamp}.png',bbox_inches='tight',dpi=300)
        plt.show()
























