import numpy as np
import time
import datetime
import sys
from pathlib import Path
import os 
from astropy.io import fits
import json
import numpy as np
import matplotlib.pyplot as plt


sys.path.insert(1, '/opt/FirstLightImaging/FliSdk/Python/demo/')
import FliSdk_V2
import FliCredOne
import FliCredTwo
import FliCredThree

from PyQt5.QtWidgets import QApplication, QMainWindow, QLabel, QPushButton, QVBoxLayout, QWidget, QLineEdit, QHBoxLayout, QTextEdit, QFileDialog, QSlider
from PyQt5.QtCore import QTimer, Qt
from PyQt5.QtGui import QPixmap, QImage
from astropy.io import fits

#list of vailable commands for the 
#send_fli_cmd() method based on C-RED One User Manual_20170116v0.2
cred1_command_dict = {
    "all raw": "Display, colon-separated, camera parameters",
    "powers": "Get all camera powers",
    "powers raw": "raw printing",
    "powers getter": "Get getter power",
    "powers getter raw": "raw printing",
    "powers pulsetube": "Get pulsetube power",
    "powers pulsetube raw": "raw printing",
    "temperatures": "Get all camera temperatures",
    "temperatures raw": "raw printing",
    "temperatures motherboard": "Get mother board temperature",
    "temperatures motherboard raw": "raw printing",
    "temperatures frontend": "Get front end temperature",
    "temperatures frontend raw": "raw printing",
    "temperatures powerboard": "Get power board temperature",
    "temperatures powerboard raw": "raw printing",
    "temperatures water": "Get water temperature",
    "temperatures water raw": "raw printing",
    "temperatures ptmcu": "Get pulsetube MCU temperature",
    "temperatures ptmcu raw": "raw printing",
    "temperatures cryostat diode": "Get cryostat temperature from diode",
    "temperatures cryostat diode raw": "raw printing",
    "temperatures cryostat ptcontroller": "Get cryostat temperature from pulsetube controller",
    "temperatures cryostat ptcontroller raw": "raw printing",
    "temperatures cryostat setpoint": "Get cryostat temperature setpoint",
    "temperatures cryostat setpoint raw": "raw printing",
    "fps": "Get frame per second",
    "fps raw": "raw printing",
    "maxfps": "Get the max frame per second regarding current camera configuration",
    "maxfps raw": "raw printing",
    "peltiermaxcurrent": "Get peltiermaxcurrent",
    "peltiermaxcurrent raw": "raw printing",
    "ptready": "Get pulsetube ready information",
    "ptready raw": "raw printing",
    "pressure": "Get cryostat pressure",
    "pressure raw": "raw printing",
    "gain": "Get gain",
    "gain raw": "raw printing",
    "bias": "Get bias correction status",
    "bias raw": "raw printing",
    "flat": "Get flat correction status",
    "flat raw": "raw printing",
    "imagetags": "Get tags in image status",
    "imagetags raw": "raw printing",
    "led": "Get LED status",
    "led raw": "raw printing",
    "sendfile bias <bias image file size> <file MD5>": "Interpreter waits for bias image binary bytes; timeout restarts interpreter.",
    "sendfile flat <flat image file size> <file MD5>": "Interpreter waits for flat image binary bytes.",
    "getflat <url>": "Retrieve flat image from URL.",
    "getbias <url>": "Retrieve bias image from URL.",
    "gettestpattern <url>": "Retrieve test pattern images tar.gz file from URL for testpattern mode.",
    "testpattern": "Get testpattern mode status.",
    "testpattern raw": "raw printing",
    "events": "Camera events sending status",
    "events raw": "raw printing",
    "extsynchro": "Get external synchro usage status",
    "extsynchro raw": "raw printing",
    "rawimages": "Get raw images (no embedded computation) status",
    "rawimages raw": "raw printing",
    "getter nbregeneration": "Get getter regeneration count",
    "getter nbregeneration raw": "raw printing",
    "getter regremainingtime": "Get time remaining for getter regeneration",
    "getter regremainingtime raw": "raw printing",
    "cooling": "Get cooling status",
    "cooling raw": "raw printing",
    "standby": "Get standby mode status",
    "standby raw": "raw printing",
    "mode": "Get readout mode",
    "mode raw": "raw printing",
    "resetwidth": "Get reset width",
    "resetwidth raw": "raw printing",
    "nbreadworeset": "Get read count without reset",
    "nbreadworeset raw": "raw printing",
    "cropping": "Get cropping status (active/inactive)",
    "cropping raw": "raw printing",
    "cropping columns": "Get cropping columns config",
    "cropping columns raw": "raw printing",
    "cropping rows": "Get cropping rows config",
    "cropping rows raw": "raw printing",
    "aduoffset": "Get ADU offset",
    "aduoffset raw": "raw printing",
    "version": "Get all product versions",
    "version raw": "raw printing",
    "version firmware": "Get firmware version",
    "version firmware raw": "raw printing",
    "version firmware detailed": "Get detailed firmware version",
    "version firmware detailed raw": "raw printing",
    "version firmware build": "Get firmware build date",
    "version firmware build raw": "raw printing",
    "version fpga": "Get FPGA version",
    "version fpga raw": "raw printing",
    "version hardware": "Get hardware version",
    "version hardware raw": "raw printing",
    "status": (
        "Get camera status. Possible statuses:\n"
        "- starting: Just after power on\n"
        "- configuring: Reading configuration\n"
        "- poorvacuum: Vacuum between 10-3 and 10-4 during startup\n"
        "- faultyvacuum: Vacuum above 10-3\n"
        "- vacuumrege: Getter regeneration\n"
        "- ready: Ready to be cooled\n"
        "- isbeingcooled: Being cooled\n"
        "- standby: Cooled, sensor off\n"
        "- operational: Cooled, taking valid images\n"
        "- presave: Previous usage error occurred"
    ),
    "status raw": "raw printing",
    "status detailed": "Get last status change reason",
    "status detailed raw": "raw printing",
    "continue": "Resume camera if previously in error/poor vacuum state.",
    "save": "Save current settings; cooling/gain not saved.",
    "save raw": "raw printing",
    "ipaddress": "Display camera IP settings",
    "cameratype": "Display camera information",
    "exec upgradefirmware <url>": "Upgrade firmware from URL",
    "exec buildbias": "Build the bias image",
    "exec buildbias raw": "raw printing",
    "exec buildflat": "Build the flat image",
    "exec buildflat raw": "raw printing",
    "exec redovacuum": "Start vacuum regeneration",
    "set testpattern on": "Enable testpattern mode (loop of 32 images).",
    "set testpattern on raw": "raw printing",
    "set testpattern off": "Disable testpattern mode",
    "set testpattern off raw": "raw printing",
    "set fps <fpsValue>": "Set the frame rate",
    "set fps <fpsValue> raw": "raw printing",
    "set gain <gainValue>": "Set the gain",
    "set gain <gainValue> raw": "raw printing",
    "set bias on": "Enable bias correction",
    "set bias on raw": "raw printing",
    "set bias off": "Disable bias correction",
    "set bias off raw": "raw printing",
    "set flat on": "Enable flat correction",
    "set flat on raw": "raw printing",
    "set flat off": "Disable flat correction",
    "set flat off raw": "raw printing",
    "set imagetags on": "Enable tags in image",
    "set imagetags on raw": "raw printing",
    "set imagetags off": "Disable tags in image",
    "set imagetags off raw": "raw printing",
    "set led on": "Turn on LED; blinks purple if operational.",
    "set led on raw": "raw printing",
    "set led off": "Turn off LED",
    "set led off raw": "raw printing",
    "set events on": "Enable camera event sending (error messages)",
    "set events on raw": "raw printing",
    "set events off": "Disable camera event sending",
    "set events off raw": "raw printing",
    "set extsynchro on": "Enable external synchronization",
    "set extsynchro on raw": "raw printing",
    "set extsynchro off": "Disable external synchronization",
    "set extsynchro off raw": "raw printing",
    "set rawimages on": "Enable embedded computation on images",
    "set rawimages on raw": "raw printing",
    "set rawimages off": "Disable embedded computation",
    "set rawimages off raw": "raw printing",
    "set cooling on": "Enable cooling",
    "set cooling on raw": "raw printing",
    "set cooling off": "Disable cooling",
    "set cooling off raw": "raw printing",
    "set standby on": "Enable standby mode (cools camera, sensor off)",
    "set standby on raw": "raw printing",
    "set standby off": "Disable standby mode",
    "set standby off raw": "raw printing",
    "set mode globalreset": "Set global reset mode (legacy compatibility)",
    "set mode globalresetsingle": "Set global reset mode (single frame)",
    "set mode globalresetcds": "Set global reset correlated double sampling",
    "set mode globalresetbursts": "Set global reset multiple non-destructive readout mode",
    "set mode rollingresetsingle": "Set rolling reset (single frame)",
    "set mode rollingresetcds": "Set rolling reset correlated double sampling (compatibility)",
    "set mode rollingresetnro": "Set rolling reset multiple non-destructive readout",
    "set resetwidth <resetwidthValue>": "Set reset width",
    "set resetwidth <resetwidthValue> raw": "raw printing",
    "set nbreadworeset <nbreadworesetValue>": "Set read count without reset",
    "set nbreadworeset <nbreadworesetValue> raw": "raw printing",
    "set cropping on": "Enable cropping",
    "set cropping on raw": "raw printing",
    "set cropping off": "Disable cropping",
    "set cropping off raw": "raw printing",
    "set cropping columns <columnsValue>": "Set cropping columns selection; format: e.g., '1,3-9'.",
    "set cropping columns <columnsValue> raw": "raw printing",
    "set cropping rows <rowsValue>": "Set cropping rows selection; format: e.g., '1,3,9'.",
    "set cropping rows <rowsValue> raw": "raw printing",
    "set aduoffset <aduoffsetValue>": "Set ADU offset",
    "set aduoffset <aduoffsetValue> raw": "raw printing",
}

# to do..
cred2_command_dict = {}
# to do..
cred3_command_dict = {}

class fli( ):

    def __init__(self, cameraIndex=0 , roi=[None, None, None, None], config_file_path = None):
        self.camera = FliSdk_V2.Init() # init camera object

        if config_file_path is None:
            # default
            config_file_path = "config_files"
             # get project root in way that also works in interactive shell (cannot use __file__)
            project_root = Path.cwd()
            while not (project_root / ".git").is_dir() and project_root != project_root.parent:
                project_root = project_root.parent
            self.config_file_path  = os.path.join( project_root,  config_file_path )
        else:
            self.config_file_path = config_file_path 
        #self.dark = [] 
        #self.bias = []
        #self.flat = []
        self.reduction_dict = {'bias':[], 'dark':[],'flat':[],'bad_pixel_mask':[]}
        #self.bad_pixel_mask = []
        self.pupil_crop_region = roi # region of interest where we crop (post readout)

        listOfGrabbers = FliSdk_V2.DetectGrabbers(self.camera)
        listOfCameras = FliSdk_V2.DetectCameras(self.camera)
        # print some info and exit if nothing detected
        if len(listOfGrabbers) == 0:
            print("No grabber detected, exit.")
            FliSdk_V2.Exit(self.camera)
        if len(listOfCameras) == 0:
            print("No camera detected, exit.")
            FliSdk_V2.Exit(self.camera)
        for i,s in enumerate(listOfCameras):
            print("- index:" + str(i) + " -> " + s)

        #cameraIndex = int( input('input index corresponding to the camera you want to use') )
        
        print(f'--->using cameraIndex={cameraIndex}')
        # set the camera
        camera_err_flag = FliSdk_V2.SetCamera(self.camera, listOfCameras[cameraIndex])
        if not camera_err_flag:
            print("Error while setting camera.")
            FliSdk_V2.Exit(self.camera)
        print("Setting mode full.")
        FliSdk_V2.SetMode(self.camera, FliSdk_V2.Mode.Full)
        print("Updating...")
        camera_err_flag = FliSdk_V2.Update(self.camera)
        if not camera_err_flag:
            print("Error while updating SDK.")
            FliSdk_V2.Exit(self.camera)

        # Dynamically inherit based on camera type
        if FliSdk_V2.IsCredOne(self.camera):
            self.__class__ = type("FliCredOneWrapper", (self.__class__, FliCredOne.FliCredOne), {})
            print("Inherited from FliCredOne")
            self.command_dict = cred1_command_dict
        elif FliSdk_V2.IsCredTwo(self.camera):
            self.__class__ = type("FliCredTwoWrapper", (self.__class__, FliCredTwo.FliCredTwo), {})
            print("Inherited from FliCredTwo")
            self.command_dict = cred2_command_dict
        elif FliSdk_V2.IsCredThree(self.camera):
            self.__class__ = type("FliCredThreeWrapper", (self.__class__, FliCredThree.FliCredThree), {})
            print("Inherited from FliCredThree")
            self.command_dict = cred3_command_dict
        else:
            print("No compatible camera type detected.")
            FliSdk_V2.Exit(self.camera)
            
    # send FLI command (based on firmware version)
    def send_fli_cmd(self, cmd ):
        val = FliSdk_V2.FliSerialCamera.SendCommand(self.camera, cmd)
        #if not val:
        #    print(f"Error with command {cmd}")
        return val 
    

    def print_camera_commands(self):
        """Prints all available commands and their descriptions in a readable format."""
        print('Available Camera Commands with "send_fli_cmd()" method:')
        print("=" * 30)
        for command, description in self.command_dict.items():
            print(f"{command}: {description}")
        print("=" * 30)


    def configure_camera( self, config_file , sleep_time = 0.2):
        """
        config_file must be json and follow convention
        that the cameras firmware CLI accepts the command
        > "set {k} {v}"
        where k is the config file key and v is the value
        """
        with open( config_file, "r") as file:
            camera_config = json.load(file)  # Parses the JSON content into a Python dictionary

        # check if in standby 
        if self.send_fli_cmd( 'standby raw' )[1] == 'on':
            try :
                self.send_fli_cmd( f"set standby on" )
            except:
                raise UserWarning( "---\ncamera in standby mode and the fli command 'set standby off' failed\n " )
        
        for k, v in camera_config.items():
            time.sleep( sleep_time )

            # for some reason set stanby mode timesout
            # if setting to the same state - so we manually check
            # before sending the command
            if 'standby' in k:
                if v != self.send_fli_cmd( 'standby raw' )[1]:
                    ok , _  = self.send_fli_cmd( f"set {k} {v}")
                    if not ok :
                        print( f"FAILED FOR set {k} {v}")


            ok , _  = self.send_fli_cmd( f"set {k} {v}")
            if not ok :
                print( f"FAILED FOR set {k} {v}")

        
    # basic wrapper functions
    def start_camera(self):
        ok = FliSdk_V2.Start(self.camera)
        return ok 
    def stop_camera(self):
        ok = FliSdk_V2.Stop(self.camera)
        return ok
    
    def exit_camera(self):
        FliSdk_V2.Exit(self.camera)

    def get_last_raw_image_in_buffer(self):
        img = FliSdk_V2.GetRawImageAsNumpyArray(self.camera, -1)
        return img 
    

    def get_camera_config(self):
        # config_dict = {
        #     'mode':self.send_fli_cmd('mode raw' )[1], 
        #     'fps': self.send_fli_cmd('fps raw' )[1],
        #     'gain': self.send_fli_cmd('gain raw' )[1],
        #     "cropping_state": self.send_fli_cmd('cropping raw' )[1],
        #     "reset_width":self.send_fli_cmd('resetwidth raw' )[1],
        #     "aduoffset":self.send_fli_cmd( 'aduoffset raw' )[1],
        #     "resetwidth":self.send_fli_cmd( "resetwidth raw")[1]
        # } 

        # read in default_cred1_config

         
        # open the default config file to get the keys 
        with open(os.path.join( self.config_file_path , "default_cred1_config.json"), "r") as file:
            default_cred1_config = json.load(file)  # Parses the JSON content into a Python dictionary

        config_dict = {}
        for k, v in default_cred1_config.items():
            config_dict[k] = self.send_fli_cmd( f"{k} raw" )[1].strip() # reads the state
        return( config_dict )
     

    # some custom functions

    def build_manual_dark( self , no_frames = 100 ):
        
        # full frame variables here were used in previous rtc. 
        # maybe redundant now. 
        #fps = float( self.send_fli_cmd( "fps")[1] )
        #dark_fullframe_list = []
        
        #dark_list = []
        #for _ in range(no_frames):
        #    time.sleep(1/fps)
        #    dark_list.append( self.get_image(apply_manual_reduction  = False) )
        #    #dark_fullframe_list.append( self.get_image_in_another_region() ) 
        print('...getting frames')
        dark_list = self.get_some_frames(number_of_frames = no_frames, apply_manual_reduction=False, timeout_limit = 20000 )
        print('...aggregating frames')
        dark = np.median(dark_list ,axis = 0).astype(int)
        # dark_fullframe = np.median( dark_fullframe_list , axis=0).astype(int)

        if len( self.reduction_dict['bias'] ) > 0:
            print('...applying bias')
            dark -= self.reduction_dict['bias'][0]

        #if len( self.reduction_dict['bias_fullframe']) > 0 :
        #    dark_fullframe -= self.reduction_dict['bias_fullframe'][0]
        print('...appending dark')
        self.reduction_dict['dark'].append( dark )
        #self.reduction_dict['dark_fullframe'].append( dark_fullframe )



    def get_bad_pixel_indicies( self, no_frames = 100, std_threshold = 100 , flatten=False):
        # To get bad pixels we just take a bunch of images and look at pixel variance 
        #self.enable_frame_tag( True )
        time.sleep(0.5)
        #zwfs.get_image_in_another_region([0,1,0,4])
        
        dark_list = self.get_some_frames( number_of_frames = no_frames , apply_manual_reduction  = False  ) #[]
        #i=0
        # while len( dark_list ) < no_frames: # poll 1000 individual images
        #     full_img = self.get_image_in_another_region() # we can also specify region (#zwfs.get_image_in_another_region([0,1,0,4]))
        #     current_frame_number = full_img[0][0] #previous_frame_number
        #     if i==0:
        #         previous_frame_number = current_frame_number
        #     if current_frame_number > previous_frame_number:
        #         if current_frame_number == 65535:
        #             previous_frame_number = -1 #// catch overflow case for int16 where current=0, previous = 65535
        #         else:
        #             previous_frame_number = current_frame_number 
        #             dark_list.append( self.get_image( apply_manual_reduction  = False) )
        #     i+=1
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


    def get_image(self, apply_manual_reduction  = True, which_index = -1 ):

        # I do not check if the camera is running. Users should check this 
        # gets the last image in the buffer
        if not apply_manual_reduction:
            img = FliSdk_V2.GetRawImageAsNumpyArray( self.camera , -1)
            cropped_img = img[self.pupil_crop_region[0]:self.pupil_crop_region[1],self.pupil_crop_region[2]: self.pupil_crop_region[3]].astype(int)  # make sure int and not uint16 which overflows easily     
        else :
            img = FliSdk_V2.GetRawImageAsNumpyArray( self.camera , -1)
            cropped_img = img[self.pupil_crop_region[0]:self.pupil_crop_region[1],self.pupil_crop_region[2]: self.pupil_crop_region[3]].astype(int)  # make sure 

            if len( self.reduction_dict['bias'] ) > 0:
                cropped_img -= self.reduction_dict['bias'][which_index] # take the most recent bias. bias must be set in same cropping state 

            if len( self.reduction_dict['dark'] ) > 0:
                cropped_img -= self.reduction_dict['dark'][which_index] # take the most recent dark. Dark must be set in same cropping state 

            if len( self.reduction_dict['flat'] ) > 0:
                cropped_img /= np.array( self.reduction_dict['flat'][which_index] , dtype = type( cropped_img[0][0]) ) # take the most recent flat. flat must be set in same cropping state 

            if len( self.reduction_dict['bad_pixel_mask'] ) > 0:
                # enforce the same type for mask
                cropped_img *= np.array( self.reduction_dict['bad_pixel_mask'][which_index] , dtype = type( cropped_img[0][0]) ) # bad pixel mask must be set in same cropping state 

        return(cropped_img)    

    def get_image_in_another_region(self, crop_region=[None,None,None,None]):
        # useful if we want to look outside of the region of interest 
        # defined by self.pupil_crop_region

        img = FliSdk_V2.GetRawImageAsNumpyArray( self.camera , -1)
        cropped_img = img[crop_region[0]:crop_region[1],crop_region[2]: crop_region[3]].astype(int)  # make sure int and not uint16 which overflows easily     
        
        #if type( self.pixelation_factor ) == int : 
        #    cropped_img = util.block_sum(ar=cropped_img, fact = self.pixelation_factor)
        #elif self.pixelation_factor != None:
        #    raise TypeError('ZWFS.pixelation_factor has to be of type None or int')
        return( cropped_img )    
    

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

            full_img = self.get_image_in_another_region() # empty argunment for full frame
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


    def save_fits( self , fname ,  number_of_frames=10, apply_manual_reduction=True ):

        hdulist = fits.HDUList([])

        frames = self.get_some_frames( number_of_frames=number_of_frames, apply_manual_reduction=apply_manual_reduction,timeout_limit=20000)
        
        # Convert list to numpy array for FITS compatibility
        data_array = np.array(frames, dtype=float)  # Ensure it is a float array or any appropriate type

        # Create a new ImageHDU with the data
        hdu = fits.ImageHDU( np.array(frames) )

        # Set the EXTNAME header to the variable name
        hdu.header['EXTNAME'] = 'FRAMES'
        #hdu.header['config'] = config_file_name

        config_tmp = self.get_camera_config()
        for k, v in config_tmp.items():
            hdu.header[k] = v
        # Append the HDU to the HDU list
        hdulist.append(hdu)

        # append reduction info
        for k, v in self.reduction_dict.items():
            if len(v) > 0 :
                hdu = fits.ImageHDU( v[-1] )
                hdu.header['EXTNAME'] = k
                hdulist.append(hdu)
            else: # we just append empty list to show that its empty!
                hdu = fits.ImageHDU( v )
                hdu.header['EXTNAME'] = k
                hdulist.append(hdu)

        hdulist.writeto(fname, overwrite=True)


    # ensures we exit safely and set gain to unity
    def __del__(self):
        # Cleanup when object is deleted
        if hasattr(self, 'camera') and self.camera is not None:
            self.send_fli_cmd( "set gain 1" )
            FliSdk_V2.Exit(self.camera)
            print("Camera SDK exited cleanly.")




if __name__ == "__main__":
    
    # example to get series of darks in different modes
    # and save as fits 
    # camera operating modes are 
    #    - single read (set mode globalresetsingle)
    #    - correlated double sampling (set mode globalresetcds)
    #    - multiple non-destructive reads (set mode globalresetbursts)
    #    - rolling versions of these modes (set mode rollingresetsingle)
    # see section 7. Camera Operating Modes from C-RED 1 user manual
    
    data_path = '/home/heimdallr/Downloads/'
    tstamp = datetime.datetime.now().strftime("%d-%m-%YT%H.%M.%S")
    roi = [None, None, None, None] # No region of interest
    aduoffset = 100 # to avoid overflow with negative
    # init camera
    c = fli(cameraIndex=0, roi=[100,200,100,200])

    # print the camera commands available in send_fli_cmd method
    c.print_camera_commands()

    # set up 
    config_file_name = os.path.join( c.config_file_path , "default_cred1_config.json")
    c.configure_camera( config_file_name )
    c.send_fli_cmd( "set aduoffset 100")
    #FliSdk_V2.Update(c.camera)

    # start
    ok = c.start_camera()

    print( c.send_fli_cmd( "status" ) )
    print("GetImageReceivedRate:", FliSdk_V2.GetImageReceivedRate(c.camera) )

    # c.build_manual_dark()
    #bp = c.get_bad_pixel_indicies( no_frames = 100, std_threshold = 100 , flatten=False)
    # c.a
    #f = c.get_some_frames( number_of_frames=10, apply_manual_reduction=True)
    #c.save_fits('/home/heimdallr/Downloads/test_imgs.fits', number_of_frames=10, apply_manual_reduction=True )

    dark_dict = {}
    gain_grid = np.linspace(1, 100, 5)
    fps_grid = np.logspace(2, 3.5, 5)
    number_of_frames = 500
    for cnt, gain in enumerate( gain_grid ):
        
        print( f"{cnt / len( gain_grid )}% complete" )

        c.send_fli_cmd( f"set gain {gain}" )
        time.sleep(0.2)

        dark_dict[gain] = {}

        for fps in fps_grid:
            
            c.send_fli_cmd( f"set fps {fps}" )
            time.sleep(0.2)

            dark_dict[gain][fps] =  c.get_some_frames(number_of_frames=number_of_frames,\
                                                        apply_manual_reduction=False, timeout_limit = 20000)  
    ok = c.start_camera()


    #################
    # Plots
    #################

    # Initialize lists to store results for plotting
    mean_pixel_values = {}
    std_pixel_values = {}

    # Extract mean and standard deviation for each gain and fps setup
    for gain, fps_dict in dark_dict.items():
        mean_pixel_values[gain] = []
        std_pixel_values[gain] = []
        
        for fps, frames in fps_dict.items():
            # Compute mean pixel value across all frames and store
            mean_pixel = np.mean([np.mean(frame) for frame in frames])
            mean_pixel_values[gain].append((fps, mean_pixel))
            
            # Compute standard deviation across all frames and store
            std_pixel = np.mean([np.std(frame) for frame in frames])
            std_pixel_values[gain].append((fps, std_pixel))

    # plot 0: check an image 
    plt.figure()
    gain = gain_grid[-1]
    fps = fps_grid[0]
    plt.imshow( np.mean( dark_dict[gain][fps] ,axis=0 ) )
    plt.colorbar()
    plt.savefig('delme.png')

    # Plot 1: Mean pixel value vs. frame rate for different gain settings
    plt.figure(figsize=(10, 6))
    for gain, values in mean_pixel_values.items():
        fps_values, mean_values = zip(*values)
        plt.plot(fps_values, mean_values, marker='o', label=f'Gain {gain}')
    plt.xscale('log')
    plt.xlabel('Frame Rate (fps)')
    plt.ylabel('Mean Pixel Value')
    plt.title('Mean Pixel Value vs Frame Rate for Different Gains')
    plt.legend()
    plt.savefig('delme.png')

    # Plot 2: Standard deviation of pixel values vs. frame rate for different gain settings
    plt.figure(figsize=(10, 6))
    for gain, values in std_pixel_values.items():
        fps_values, std_values = zip(*values)
        plt.plot(fps_values, std_values, marker='o', label=f'Gain {gain}')
    plt.xscale('log')
    plt.xlabel('Frame Rate (fps)')
    plt.ylabel('Standard Deviation of Pixel Value')
    plt.title('Pixel Value Standard Deviation vs Frame Rate for Different Gains')
    plt.legend()
    plt.savefig('delme.png')

    #################
    # Save
    #################

    for gain, nested_dict in dark_dict.items():
        print( gain )

        for fps, frames in nested_dict.items():

            # do one at a time (seems to die if I append too much!)
            hdulist = fits.HDUList([])
            # Convert list to numpy array for FITS compatibility
            data_array = np.array(frames, dtype=float)  # Ensure it is a float array or any appropriate type

            # Create a new ImageHDU with the data
            hdu = fits.ImageHDU( np.array(frames) )

            # Set the EXTNAME header to the variable name
            hdu.header['EXTNAME'] = f'FPS-{round(fps,1)}_GAIN-{round(gain,1)}'
            hdu.header['config'] = config_file_name

            config_tmp = c.get_camera_config()
            for k, v in config_tmp.items():
                hdu.header[k] = v
            # Append the HDU to the HDU list
            hdulist.append(hdu)

            # 
            hdulist.writeto(data_path + f'dark_FPS-{round(fps,1)}__GAIN-{round(gain,1)}_{tstamp}.fits', overwrite=True)



