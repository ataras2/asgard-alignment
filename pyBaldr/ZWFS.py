#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr  4 11:53:02 2024

@author: bencb

ZWFS class can only really get an image and send a command to the DM and/or update the camera settings



"""


import numpy as np
import os
import glob 
import sys  
import pandas as pd 
import time
import datetime 
import matplotlib.pyplot as plt 
from astropy.io import fits 

sys.path.insert(1, '/opt/FirstLightImaging/FliSdk/Python/demo/')
sys.path.insert(1,'/opt/Boston Micromachines/lib/Python3/site-packages/')

import bmc
import FliSdk_V2
from . import utilities as util

                


class ZWFS():
    # for now one camera, one DM per object - real system will have 1 camera, 4 DMs!
    def __init__(self, DM_serial_number, cameraIndex=0, DMshapes_path='DMShapes/', pupil_crop_region=[None, None, None, None] ):
       
        # connecting to camera
        camera = FliSdk_V2.Init() # init camera object
        listOfGrabbers = FliSdk_V2.DetectGrabbers(camera)
        listOfCameras = FliSdk_V2.DetectCameras(camera)
        # print some info and exit if nothing detected
        if len(listOfGrabbers) == 0:
            print("No grabber detected, exit.")
            FliSdk_V2.Exit(camera)
        if len(listOfCameras) == 0:
            print("No camera detected, exit.")
            FliSdk_V2.Exit(camera)
        for i,s in enumerate(listOfCameras):
            print("- index:" + str(i) + " -> " + s)
       
        print(f'--->using cameraIndex={cameraIndex}')
        # set the camera
        camera_err_flag = FliSdk_V2.SetCamera(camera, listOfCameras[cameraIndex])
        if not camera_err_flag:
            print("Error while setting camera.")
            FliSdk_V2.Exit(camera)
        print("Setting mode full.")
        FliSdk_V2.SetMode(camera, FliSdk_V2.Mode.Full)
        print("Updating...")
        camera_err_flag = FliSdk_V2.Update(camera)
        if not camera_err_flag:
            print("Error while updating SDK.")
            FliSdk_V2.Exit(camera)
           
        # connecting to DM
       
        dm = bmc.BmcDm() # init DM object
        dm_err_flag  = dm.open_dm(DM_serial_number) # open DM

        dm_sim_mode = False # temporary thing for remote testing camera - sometimes DM is off and no one there to turn on 
        if dm_err_flag :
            print('Error initializing DM')
            print( 'PUTTING IN SIMULATION MODE - ZWFS WILL NOT HAVE FULL FUNCTIONALITY ')
            #raise Exception(dm.error_string(dm_err_flag))
            dm_sim_mode = True
       
        # ========== CAMERA & DM
        self.camera = camera
        if not dm_sim_mode:
            self.dm = dm
            self.dm.DM_serial_number = DM_serial_number
        else: 
            self.dm = {} # place holder 
        self.dm_number_of_actuators = 140
        
        # ========== dictionary to hold reduction products
        self.reduction_dict = {'bias':[], 'dark':[], 'flat':[], 'bad_pixel_mask':[]}


        # ========== DM shapes
        shapes_dict = {}
        if os.path.isdir(DMshapes_path):
            filenames = glob.glob(DMshapes_path+'*.csv')
            if filenames == []:
                print('no valid files in path. Empty dictionary appended to ZWFS.dm_shapes')
            else:
                for file in filenames:
                
                    try:
                        shape_name = file.split('/')[-1].split('.csv')[0]
                        shape = pd.read_csv(file, header=None)[0].values
                        if len(shape)==self.dm_number_of_actuators:
                            shapes_dict[shape_name] = np.array( (shape) )
                            # >>>>>>>>>>>>> DEFINE HERE THE FLAT DM MAP <<<<<<<<<<<
                            if DM_serial_number in file:
                                shapes_dict['flat_dm'] =  np.array( (shape) )
                            # >>>>>>>>>>>>> <<<<<<<<<<<
                        else: 
                            print(f'file: {file}\n has {len(shape)} values which does not match the number of actuators on the DM ({self.dm_number_of_actuators})')

                    except:
                        print(f'falided to read and/or append shape corresponding to file: {file}')
        else:
            print('DMshapes path does not exist. Empty dictionary appended to ZWFS.dm_shapes')

        self.dm_shapes = shapes_dict
        
        #  we can aggregate the output image over a sub window WxW where W = pixelation_factor.
        # we use this sometimes when oversample pupil, so can use this to get to expected number of pixels with some careful choices
        self.pixelation_factor = None 
        

        # this is placed as an attribute here since from 1 image we will have 4 pupils 
        # which will control 4 different DMs.
        # So we need to define where to find each pupil for any given image taken. 
        if not hasattr(pupil_crop_region, '__len__'):
            #pupil_crop_region == None:
            pupil_crop_region = [None, None, None, None] # no cropping of the image
        elif hasattr(pupil_crop_region, '__len__'):
            if len( pupil_crop_region ) == 4:
                #if all(isinstance(x, int) for x in pupil_crop_region):
                self.pupil_crop_region = pupil_crop_region  # [row1 , row2, col1 , col2]
                #else:
                #    self.pupil_crop_region = [None, None, None, None]
                #    print('pupil_crop_region has INVALID entries, it needs to be integers')
            else:
                self.pupil_crop_region = [None, None, None, None]
                print('pupil_crop_region has INVALID length, it needs to have length = 4')
        else:
            self.pupil_crop_region = [None, None, None, None]
            print('pupil_crop_region has INVALID type. Needs to be list of integers of length = 4')

        # define our image coordinates based on cropping and pixelisation
        self._update_image_coordinates( )

        # define our region where we can robustly expect to uniquely find the ZWFS pupil. 

        # ========== initiate info on reference regions that controllers will need to work.
        """
        # this can be updated later after DM, pupil centering, for example using: 
        report = pupil_control.analyse_pupil_openloop( zwfs, debug = True, return_report = True) 
        as input to zwfs.update_reference_regions_in_img( report ).
        This has to be updated before any controller can be applied!
        """

        #boolean filters for reference regions 
        self.pupil_pixel_filter = [np.nan]
        self.secondary_pixel_filter = [np.nan]
        self.outside_pixel_filter = [np.nan]
        self.refpeak_pixel_filter = [np.nan]

        # pixels in reference regions  
        self.pupil_pixels = [np.nan]  
        self.secondary_pixels = [np.nan]
        self.outside_pixels = [np.nan]
        self.refpeak_pixels = [np.nan]

        # bad pixels 
        self.bad_pixel_filter = [np.nan]  
        self.bad_pixels = [np.nan]  

        # reference centers 
        self.pupil_center_ref_pixels = (np.nan, np.nan)
        self.dm_center_ref_pixels = (np.nan, np.nan)
        self.secondary_center_ref_pixels =  (np.nan, np.nan)

        # DM coordinates in pixel space relative to self.row_coords, self.col_coords
        self.dm_col_coordinates_in_pixels = [np.nan]
        self.dm_row_coordinates_in_pixels = [np.nan]

        # Note this is duplicated in phase_control object... need to decide where best to keep it! 
        # both pupil and phase controllers need this reference - so maybe best here?
        self.N0   =  None #2D array of intensity when FPM OUT 
        self.I0   =  None #2D array of intensity when FPM IN 


        # ========== CONTROLLERS
        self.phase_controllers = [] # initiate with no controllers
        self.pupil_controllers = [] # initiate with no controllers

        # ========== STATES
        """
        Notes:
        - Any process that takes ZWFS object as input is required to update ZWFS states
        - before using any controller, the ZWFS states should be compared with controller
        configuration file to check for consistency (i.e. can controller be applied in the current state?)
        """
        self.states = {}
       
        self.states['simulation_mode'] = 0 # 1 if we have in simulation mode
       
        self.states['telescopes'] = ['AT1'] #
        self.states['phase_ctrl_state'] = 0 # 0 open loop, 1 closed loop
        self.states['pupil_ctrl_state'] = 0 # 0 open loop, 1 closed loop
        self.states['source'] = 0 # 0 for no source, 1 for internal calibration source, 2 for stellar source
        self.states['sky'] = 0 # 1 if we are on a sky (background), 0 otherwise
        self.states['fpm'] = 0 # 0 for focal plance mask out, positive number for in depending on which one)
        self.states['busy'] = 0 # 1 if the ZWFS is doing something
        # etc 
       



    def _update_image_coordinates(self): 
        # define coordinates in our image based on cropping and pixelisation factors 
        
        #define pixel coordinates in this cropped region  
        r1, r2, c1, c2 = self.pupil_crop_region
        if (np.all(self.pupil_crop_region != [None, None, None, None])) & (self.pixelation_factor == None) :
            self.row_coords = np.linspace( (r1 - r2)/2 , (r2 - r1)/2, r2-r1)  #rows
            self.col_coords = np.linspace( (c1 - c2)/2 , (c2 - c1)/2, c2-c1)  #columns
        elif (np.all(self.pupil_crop_region != [None, None, None, None])) & (self.pixelation_factor != None) :
            pf = self.pixelation_factor
            self.row_coords = np.linspace( 1/pf * (r1 - r2)/2 , 1/pf * (r2 - r1)/2, int(1/pf * (r2-r1) ) )  #rows
            self.col_coords = np.linspace( 1/pf * (c1 - c2)/2 , 1/pf * (c2 - c1)/2, int(1/pf * (c2-c1) ) )  #columns

        elif np.all((self.pupil_crop_region == [None, None, None, None])) & (self.pixelation_factor == None) :# full image 
            r1 = 0; c1 = 0
            c2, r2 = FliSdk_V2.GetCurrentImageDimension(self.camera)
            self.row_coords = np.linspace( (r1 - r2)/2 , (r2 - r1)/2, r2-r1)  #rows
            self.col_coords = np.linspace( (c1 - c2)/2 , (c2 - c1)/2, c2-c1)  #columns

        elif np.all((self.pupil_crop_region == [None, None, None, None])) & (self.pixelation_factor != None) :
            r1 = 0; c1 = 0
            c2, r2 = FliSdk_V2.GetCurrentImageDimension(self.camera)
            pf = self.pixelation_factor
            self.row_coords = np.linspace( 1/pf * (r1 - r2)/2 , 1/pf * (r2 - r1)/2, int(1/pf * (r2-r1) ) )  #rows
            self.col_coords = np.linspace( 1/pf * (c1 - c2)/2 , 1/pf * (c2 - c1)/2, int(1/pf *(c2-c1) ) )  #columns




    def send_cmd(self, cmd):
       
        self.dm_err_flag = self.dm.send_data(cmd)
       
       
   
    def propagate_states(self, simulation_mode = False):
       
        if not self.states['simulation_mode']:
           
            for state, value in self.states.items():
                try:
                    print('check the systems state relative to current ZWFS state and \
                      rectify (e.g. send command to some motor) if any discrepency')
                except:
                    print('raise an error or implement some workaround if the requested state cannot be realised')

    def build_manual_dark( self ):
        fps = self.get_camera_fps()
        dark_list = []
        for _ in range(1000):
            time.sleep(1/fps)
            dark_list.append( self.get_image(apply_manual_reduction  = False) )
        dark = np.median(dark_list ,axis = 0).astype(int)

        if len( self.reduction_dict['bias'] ) > 0:
            dark -= self.reduction_dict['bias'][0]

        self.reduction_dict['dark'].append( dark )


    def get_bad_pixel_indicies( self, no_frames = 1000, std_threshold = 100 , flatten=False):
        # To get bad pixels we just take a bunch of images and look at pixel variance 
        self.enable_frame_tag( True )
        time.sleep(0.5)
        #zwfs.get_image_in_another_region([0,1,0,4])
        i=0
        dark_list = []
        while len( dark_list ) < no_frames: # poll 1000 individual images
            full_img = self.get_image_in_another_region() # we can also specify region (#zwfs.get_image_in_another_region([0,1,0,4]))
            current_frame_number = full_img[0][0] #previous_frame_number
            if i==0:
                previous_frame_number = current_frame_number
            if current_frame_number > previous_frame_number:
                if current_frame_number == 65535:
                    previous_frame_number = -1 #// catch overflow case for int16 where current=0, previous = 65535
                else:
                    previous_frame_number = current_frame_number 
                    dark_list.append( self.get_image( apply_manual_reduction  = False) )
            i+=1
        dark_std = np.std( dark_list ,axis=0)
        # define our bad pixels where std > 100 or zero variance
        #if not flatten:
        bad_pixels = np.where( (dark_std > std_threshold) + (dark_std == 0 ))
        #else:  # flatten is useful for when we filter regions by flattened pixel indicies
        bad_pixels_flat = np.where( (dark_std.reshape(-1) > std_threshold) + (dark_std.reshape(-1) == 0 ))

        #self.bad_pixels = bad_pixels_flat

        if not flatten:
            return( bad_pixels )
        else:
            return( bad_pixels_flat )


    def build_bad_pixel_mask( self, bad_pixels , set_bad_pixels_to = 0):
        """
        bad_pixels = tuple of array of row and col indicies of bad pixels.
        Can create this simply by bad_pixels = np.where( <condition on image> )
        gets a current image to generate bad_pixel_mask shape
        - Note this also updates zwfs.bad_pixel_filter  and zwfs.bad_pixels
           which can be used to filterout bad pixels in the controlled pupil region 
        """
        i = self.get_image(apply_manual_reduction = False )
        bad_pixel_mask = np.ones( i.shape )
        for ibad,jbad in list(zip(bad_pixels[0], bad_pixels[1])):
            bad_pixel_mask[ibad,jbad] = set_bad_pixels_to

        self.reduction_dict['bad_pixel_mask'].append( bad_pixel_mask )

        badpixel_bool_array = np.zeros(i.shape , dtype=bool)
        for ibad,jbad in list(zip(bad_pixels[0], bad_pixels[1])):
            badpixel_bool_array[ibad,jbad] = True
        
        self.bad_pixel_filter = badpixel_bool_array.reshape(-1)
        self.bad_pixels = np.where( self.bad_pixel_filter )[0]


    def get_image(self, apply_manual_reduction  = True ):

        # I do not check if the camera is running. Users should check this 
        # gets the last image in the buffer
        if not apply_manual_reduction:
            img = FliSdk_V2.GetRawImageAsNumpyArray( self.camera , -1)
            cropped_img = img[self.pupil_crop_region[0]:self.pupil_crop_region[1],self.pupil_crop_region[2]: self.pupil_crop_region[3]].astype(int)  # make sure int and not uint16 which overflows easily     
        else :
            img = FliSdk_V2.GetRawImageAsNumpyArray( self.camera , -1)
            cropped_img = img[self.pupil_crop_region[0]:self.pupil_crop_region[1],self.pupil_crop_region[2]: self.pupil_crop_region[3]].astype(int)  # make sure 

            if len( self.reduction_dict['bias'] ) > 0:
                cropped_img -= self.reduction_dict['bias'][0] # take the most recent bias. bias must be set in same cropping state 

            if len( self.reduction_dict['dark'] ) > 0:
                cropped_img -= self.reduction_dict['dark'][0] # take the most recent dark. Dark must be set in same cropping state 

            if len( self.reduction_dict['flat'] ) > 0:
                cropped_img /= np.array( self.reduction_dict['flat'][0] , dtype = type( cropped_img[0][0]) ) # take the most recent flat. flat must be set in same cropping state 

            if len( self.reduction_dict['bad_pixel_mask'] ) > 0:
                # enforce the same type for mask
                cropped_img *= np.array( self.reduction_dict['bad_pixel_mask'][0] , dtype = type( cropped_img[0][0]) ) # bad pixel mask must be set in same cropping state 


        if type( self.pixelation_factor ) == int : 
            cropped_img = util.block_sum(ar=cropped_img, fact = self.pixelation_factor)
        elif self.pixelation_factor != None:
            raise TypeError('ZWFS.pixelation_factor has to be of type None or int')
        return(cropped_img)    


    def get_some_frames(self, number_of_frames = 100, apply_manual_reduction=True, timeout_limit = 20000 ):
        """
        poll sequential frames (no repeats) and store in list  
        """
        ref_img_list = []
        i=0
        timeout_counter = 0 
        timeout_flag = 0
        while (len( ref_img_list  ) < number_of_frames) and not timeout_flag: # poll  individual images
            if timeout_counter > timeout_limit: # we have done timeout_limit iterations without a frame update
                timeout_flag = 1 
                raise TypeError('timeout! timeout_counter > 10000')

            full_img = self.get_image_in_another_region() # we can also specify region (#zwfs.get_image_in_another_region([0,1,0,4]))
            current_frame_number = full_img[0][0] #previous_frame_number
            if i==0:
                previous_frame_number = current_frame_number
            if current_frame_number > previous_frame_number:
                timeout_counter = 0 # reset timeout counter
                if current_frame_number == 65535:
                    previous_frame_number = -1 #// catch overflow case for int16 where current=0, previous = 65535
                else:
                    previous_frame_number = current_frame_number 
                    ref_img_list.append( self.get_image( apply_manual_reduction  = apply_manual_reduction) )
            i+=1
            timeout_counter += 1
            
        return( ref_img_list )  


    def estimate_noise_covariance( self, number_of_frames = 1000, where = 'pupil' ):
        
        img_list = self.get_some_frames( number_of_frames )

        # looking at covariance of pixel noise 
        #img_list  = np.array( img_list  )
        if where == 'pupil':
            img_list_filtered = np.array( [d.reshape(-1)[self.pupil_pixel_filter] for d in img_list] )

        elif where == 'whole_image':
            img_list_filtered = np.array( [d.reshape(-1) for d in img_list] )


        cov_matrix = np.cov( img_list_filtered ,ddof=1, rowvar = False ) # rowvar = False => rows are samples, cols variables 
        return( cov_matrix )

    def get_processed_image(self):
        FliSdk_V2.GetProcessedImage(self.camera, -1)

    def send_fli_cmd(self, cmd ):
        camera_err_flag = FliSdk_V2.FliSerialCamera.SendCommand(self.camera, cmd)
        if not camera_err_flag:
            print(f"Error with command {cmd}")

    def get_image_in_another_region(self, crop_region=[0,-1,0,-1]):
        
        # I do not check if the camera is running. Users should check this 
        # gets the last image in the buffer
        img = FliSdk_V2.GetRawImageAsNumpyArray( self.camera , -1)
        cropped_img = img[crop_region[0]:crop_region[1],crop_region[2]: crop_region[3]].astype(int)  # make sure int and not uint16 which overflows easily     
        
        #if type( self.pixelation_factor ) == int : 
        #    cropped_img = util.block_sum(ar=cropped_img, fact = self.pixelation_factor)
        #elif self.pixelation_factor != None:
        #    raise TypeError('ZWFS.pixelation_factor has to be of type None or int')
        return( cropped_img )    


    def build_bias(self, number_frames=256):
        #nb = 256 # number of frames for building bias 
        # cred 3 
        #FliSdk_V2.FliSerialCamera.SendCommand(self.camera, f"buildnuc bias {number_frames}")
        # Cred 2 
        FliSdk_V2.FliSerialCamera.SendCommand(self.camera, "exec buildbias")

    def build_flat(self ):
        #nb = 256 # number of frames for building bias 
        # cred 3 
        #FliSdk_V2.FliSerialCamera.SendCommand(self.camera, f"exec buildflat")
        # Cred 2 
        FliSdk_V2.FliSerialCamera.SendCommand(self.camera, "exec buildflat")


    def flat_on(self):
        FliSdk_V2.FliSerialCamera.SendCommand(self.camera, "set flat on")

    def flat_off(self):
        FliSdk_V2.FliSerialCamera.SendCommand(self.camera, "set flat off")

    def bias_on(self):
        FliSdk_V2.FliSerialCamera.SendCommand(self.camera, "set bias on")

    def bias_off(self):
        FliSdk_V2.FliSerialCamera.SendCommand(self.camera, "set bias off")


    def start_camera(self):
        FliSdk_V2.Start(self.camera)

    def stop_camera(self):
        FliSdk_V2.Stop(self.camera)

    def get_camera_dit(self):
        camera_err_flag, DIT = FliSdk_V2.FliSerialCamera.SendCommand(self.camera, "tint raw")
        return( DIT ) 

    def get_camera_fps(self):
        camera_err_flag, fps = FliSdk_V2.FliSerialCamera.GetFps(self.camera)
        return( fps ) 

    def get_dit_limits(self):
               
        camera_err_flag, minDIT = FliSdk_V2.FliSerialCamera.SendCommand(self.camera, "mintint raw")
        self.camera_err_flag = camera_err_flag
       
        camera_err_flag, maxDIT = FliSdk_V2.FliSerialCamera.SendCommand(self.camera, "maxtint raw")
        self.camera_err_flag = camera_err_flag
       
        return(minDIT, maxDIT)   

    def set_camera_dit(self, DIT):
        # set detector integration time (DIT). input in seconds
        minDit, maxDit = self.get_dit_limits()
        if (DIT >= float(minDit)) & (DIT <= float(maxDit)):
            FliSdk_V2.FliSerialCamera.SendCommand(self.camera, "set tint " + str(float(DIT)))
        else:
            print(f"requested DIT {1e3*DIT}ms outside DIT limits {(1e3*float(minDit),1e3*float(maxDit))}ms.\n Cannot change DIT to this value")
    
    def set_camera_fps(self, fps):
        FliSdk_V2.FliSerialCamera.SetFps(self.camera, fps)

    def set_camera_cropping(self, r1, r2, c1, c2 ): 
        FliSdk_V2.FliSerialCamera.SendCommand(self.camera, "set cropping off")
        # cropped columns must be multiple of 32 - multiple of 32 minus 1
        #e.g. fli.FliSerialCamera.SendCommand(camera, "set cropping columns 64-287")
        FliSdk_V2.FliSerialCamera.SendCommand(self.camera, f"set cropping columns {c1}-{c2}")
        # cropped rows must be multiple of 4 - multiple of 4 minus 1
        #e.g. fli.FliSerialCamera.SendCommand(camera, "set cropping rows 120-299")
        FliSdk_V2.FliSerialCamera.SendCommand(self.camera, f"set cropping rows {r1}-{r2}")
        FliSdk_V2.FliSerialCamera.SendCommand(self.camera, "set cropping on")
    


    def deactive_cropping(self):
        FliSdk_V2.FliSerialCamera.SendCommand(self.camera, "set cropping off")

    def shutdown(self):
        FliSdk_V2.FliSerialCamera.SendCommand(self.camera, "shutdown")

    def exit_camera(self):
        FliSdk_V2.Stop(self.camera)
        FliSdk_V2.Exit(self.camera)

    def exit_dm(self):
        self.dm.close_dm()

    def enable_frame_tag(self, tag = True):
        if tag:
            FliSdk_V2.FliSerialCamera.SendCommand(self.camera, "set imagetags on")
        else:
            FliSdk_V2.FliSerialCamera.SendCommand(self.camera, "set imagetags off")

    def set_sensitivity(self, gain_string):
        """
        gain_string must be "low", "medium" or "high"
        """
        # cred 3
        #FliSdk_V2.FliSerialCamera.SendCommand(self.camera, f"set sensitivity {gain_string}")
        # cred 2 
        FliSdk_V2.FliSerialCamera.SendCommand(self.camera, f"set sensibility {gain_string}")
        
    def restore_default_settings(self): 
        FliSdk_V2.FliSerialCamera.SendCommand(self.camera, "restorefactory")

    def update_pupil_pixel_filter(self): 
        # updates self.pupil_pixel_filter, self.pupil_pixels, 
        # Need to decide if we also have to update central reference points here? - I dont think so
        return(None)
 
    def update_reference_regions_in_img(self, pupil_report ):
       
        self.pupil_pixel_filter = pupil_report['pupil_pixel_filter']
        self.secondary_pixel_filter = pupil_report['secondary_pupil_pixel_filter']
        self.outside_pixel_filter = pupil_report['outside_pupil_pixel_filter']
        self.refpeak_pixel_filter = pupil_report['reference_field_peak_filter']

        self.pupil_pixels = pupil_report['pupil_pixels']  
        self.secondary_pixels = pupil_report['secondary_pupil_pixels']
        self.outside_pixels = pupil_report['outside_pupil_pixels']
        self.refpeak_pixels = pupil_report['reference_field_peak_pixels']

        self.pupil_center_ref_pixels = pupil_report['pupil_center_ref_pixels']
        self.dm_center_ref_pixels = pupil_report['dm_center_ref_pixels']
        self.secondary_center_ref_pixels =  pupil_report['secondary_center_ref_pixels']

        # DM coordinates in pixel space relative to self.row_coords, self.col_coords
        self.dm_col_coordinates_in_pixels = pupil_report['dm_x_coords_in_pixels']
        self.dm_row_coordinates_in_pixels = pupil_report['dm_y_coords_in_pixels']

        # Note this is duplicated in phase_control object... need to decide where best to keep it! 
        self.N0   =  pupil_report['N0'] #2D array of intensity when FPM OUT 
        self.I0   =  pupil_report['I0'] #2D array of intensity when FPM IN 


    
    def plot_classified_pupil_region(self):
        # to check pupil regions are defined ok. 
        # If you want to check
        fig,ax = plt.subplots(1,5,figsize=(20,4))
        # note with ZWFS I0, N0 are not pixel filtered 
        ax[0].imshow( self.pupil_pixel_filter.reshape(self.I0.shape)) # cp_x2-cp_x1, cp_y2-cp_y1) )
        ax[1].imshow( self.outside_pixel_filter.reshape(self.I0.shape))#( cp_x2-cp_x1, cp_y2-cp_y1) )
        ax[2].imshow( self.secondary_pixel_filter.reshape(self.I0.shape))#( cp_x2-cp_x1, cp_y2-cp_y1) )
        ax[3].imshow( self.I0 )
        ax[4].imshow( self.N0 )
        
        for axx,l in zip(ax, ['inside pupil','outside pupil','secondary','I0','N0']):
            axx.set_title(l)
        plt.show()
    
    def write_reco_fits( self, phase_controller, ctrl_label, save_path, save_label=None):
        """
        phase_controller object from phase_control module
        ctrl_label is a string indicating the label used 
        when calibrating the phase_controller. see phase_control 
        module for details. 

        save_label can be used to add a user description to the file

        """
        # timestamp
        tstamp = datetime.datetime.now().strftime("%d-%m-%YT%H.%M.%S")

        # == info about set up 
        info_fits = fits.PrimaryHDU( [] ) #[pupil_classification_file,WFS_response_file] )

        info_fits.header.set('EXTNAME','INFO')
        #info_fits.header.set('IM_construction_method',f'{reconstruction_method}')
        #info_fits.header.set('DM poke amplitude[normalized]',f'{poke_amps[amp_idx]}')
        #info_fits.header.set('what is?','names of input files for building IM')
        
        # INFO - camera info
        camera_info_dict = util.get_camera_info(self.camera)
        for k,v in camera_info_dict.items():
            info_fits.header.set(k,v)
        #info_fits.header.set('#images per DM command', number_images_recorded_per_cmd )
        #info_fits.header.set('take_median_of_images', take_median_of_images )
        
        # NOTE!! cropping_corners define cropping post camera read-out
        # they are different to the camera cropping 
        # IMPORTANT: if not NONE, pixel filters (e.g. pupil_pixel) are defined 
        # relative to this crop region! 
        info_fits.header.set('cropping_corners_r1', self.pupil_crop_region[0] )
        info_fits.header.set('cropping_corners_r2', self.pupil_crop_region[1] )
        info_fits.header.set('cropping_corners_c1', self.pupil_crop_region[2] )
        info_fits.header.set('cropping_corners_c2', self.pupil_crop_region[3] )

        # INFO - control basis info
        info_fits.header.set('control_basis', phase_controller.config['basis'] )
        info_fits.header.set('number_of_controlled_modes', phase_controller.config['number_of_controlled_modes'] )
        info_fits.header.set('dm_control_diameter', phase_controller.config['dm_control_diameter'] )
        #info_fits.header.set('dm_control_center', phase_controller.config['dm_control_center'] )

        info_fits.header.set('CM_build_method', 'FILL ME' ) # how did we build the CM 
        info_fits.header.set('poke_amplitude', phase_controller.ctrl_parameters[ctrl_label]['poke_amp'] ) # how did we build the CM 
        # push, pull, push-pull ? 

        # INTERACTION MATRIX 
        IM_fits = fits.PrimaryHDU( phase_controller.ctrl_parameters[ctrl_label]["IM"]  )  # unfiltered interaction matrix  
        IM_fits.header.set('what is?','unfiltered interaction matrix (IM)')
        IM_fits.header.set('EXTNAME','IM')      

        # CONTROL MATRIX ( I should define CM = M2C @ I2M )
        #CM = phase_controller.ctrl_parameters[ctrl_label]["CM"]
        #M2C = phase_controller.config['M2C']
        CM_fits = fits.PrimaryHDU( phase_controller.ctrl_parameters[ctrl_label]["CM"]   )  # filtered control matrix 
        CM_fits.header.set('what is?','filtered control matrix (CM)')
        CM_fits.header.set('EXTNAME','CM')

        I2M_fits = fits.PrimaryHDU( phase_controller.ctrl_parameters[ctrl_label]["I2M"]   )  # filtered control matrix 
        I2M_fits.header.set('what is?','intensity to modes')
        I2M_fits.header.set('EXTNAME','I2M')
        
        # REFERENCE INTENSITY WITH FPM OUT (N0)
        # normalized and filtered with defined pupil (1D)
        #N0_filt_fits = fits.PrimaryHDU( phase_controller.ctrl_parameters[ctrl_label]["ref_pupil_FPM_out"]  )
        #N0_filt_fits.header.set('what is?','pupil filtered FPM OUT')
        #N0_filt_fits.header.set('EXTNAME','N0_filt') 

        # un-normalized, unfiltered (2D)
        N0_fits = fits.PrimaryHDU( phase_controller.ctrl_parameters[ctrl_label]["ref_pupil_FPM_out"]  )
        N0_fits.header.set('what is?','FPM OUT')
        N0_fits.header.set('EXTNAME','N0')        
        
        # REFERENCE INTENSITY WITH FPM IN (I0)
        # normalized and filtered with defined pupil (1D)
        #I0_filt_fits = fits.PrimaryHDU( phase_controller.ctrl_parameters[ctrl_label]["ref_pupil_FPM_in"]  )
        #I0_filt_fits.header.set('what is?','pupil filtered FPM IN')
        #I0_filt_fits.header.set('EXTNAME','I0_filt')       

        # un-normalized, unfiltered (2D)
        I0_fits = fits.PrimaryHDU( phase_controller.ctrl_parameters[ctrl_label]["ref_pupil_FPM_in"]  ) 
        I0_fits.header.set('what is?','FPM IN')
        I0_fits.header.set('EXTNAME','I0')        

        # MODE TO DM COMMAND MATRIX (normalized <C|C> = 1)
        M2C_fits = fits.PrimaryHDU( phase_controller.config['M2C']  ) # mode to commands
        M2C_fits.header.set('what is?','mode to dm cmd matrix')
        M2C_fits.header.set('EXTNAME','M2C')


        # NOTE: if not cropping_corners!=NONE, pixel filters (e.g. pupil_pixel) are defined 
        # relative to this crop region! which is cropped post frame read-out.

        # pixel inside pupil
        pupil_fits = fits.PrimaryHDU( phase_controller.ctrl_parameters[ctrl_label]['pupil_pixels']  )
        pupil_fits.header.set('what is?','pixels_inside_pupil')
        pupil_fits.header.set('EXTNAME','pupil_pixels')

        # secondary 
        secondary_fits = fits.PrimaryHDU( phase_controller.ctrl_parameters[ctrl_label]['secondary_pupil_pixels']  )
        secondary_fits.header.set('what is?','pixels_inside_secondary obstruction')
        secondary_fits.header.set('EXTNAME','secondary_pixels')

        #pixel outside pupil and not in secondary obstruction
        outside_fits = fits.PrimaryHDU( phase_controller.ctrl_parameters[ctrl_label]['outside_pupil_pixels'] )
        outside_fits.header.set('what is?','pixels_outside_pupil')
        outside_fits.header.set('EXTNAME','outside_pixels')

        #DM center pixel during CM calibration
        dm_pixel_center_fits = fits.PrimaryHDU( phase_controller.ctrl_parameters[ctrl_label]['dm_center_ref_pixels']  )
        dm_pixel_center_fits .header.set('what is?','dm_center_reference_pixels')
        dm_pixel_center_fits .header.set('EXTNAME','dm_center_ref')

        # TO DO... Depends on modal basis used 
        RTT_fits = fits.PrimaryHDU( np.zeros( phase_controller.ctrl_parameters[ctrl_label]['R_TT'].shape) )
        RTT_fits.header.set('what is?','tip-tilt reconstructor')
        RTT_fits.header.set('EXTNAME','R_TT')

        RHO_fits = fits.PrimaryHDU(  np.zeros(phase_controller.ctrl_parameters[ctrl_label]['R_HO'].shape) )
        RHO_fits.header.set('what is?','higher-oder reconstructor')
        RHO_fits.header.set('EXTNAME','R_HO')

        fits_list = [info_fits, IM_fits, CM_fits, I2M_fits, M2C_fits, N0_fits, I0_fits,\
        pupil_fits, secondary_fits, outside_fits, dm_pixel_center_fits,\
        RTT_fits,RHO_fits]


        reconstructor_fits = fits.HDUList( [] )
        for f in fits_list:
            reconstructor_fits.append( f )

        if save_label!=None:
            reconstructor_fits.writeto( save_path + f'RECONSTRUCTORS_{save_label}_DIT-{round(float(info_fits.header["camera_tint"]),6)}_gain_{info_fits.header["camera_gain"]}_{tstamp}.fits',overwrite=True )  
        else:
            reconstructor_fits.writeto( save_path + f'RECONSTRUCTORS_DIT-{round(float(info_fits.header["camera_tint"]),6)}_gain_{info_fits.header["camera_gain"]}_{tstamp}.fits',overwrite=True )  