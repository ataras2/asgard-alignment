import numpy as np
import time
import datetime
import sys
from pathlib import Path
import re
import os 
from astropy.io import fits
import json
import numpy as np
import matplotlib.pyplot as plt
from xaosim.shmlib import shm
import zmq

#from PyQt5.QtWidgets import QApplication, QMainWindow, QLabel, QPushButton, QVBoxLayout, QWidget, QLineEdit, QHBoxLayout, QTextEdit, QFileDialog, QSlider
#from PyQt5.QtCore import QTimer, Qt
#from PyQt5.QtGui import QPixmap, QImage




"""
from xaosim.shmlib import shm
[15/2/2025, 4:25:24 PM] Mike Ireland: Frantz sets self.mySHM to shm(FILENAME)
[15/2/2025, 4:25:32 PM] Mike Ireland: Then self.mySHM.mtdata["size"] is the image size.
[15/2/2025, 4:26:08 PM] Mike Ireland: self.mySHM.get_counter() is the counter, that increments when there is a new frame.
[15/2/2025, 4:26:20 PM] Mike Ireland: self.mySHM.get_data() is the actual data for the frame.
[15/2/2025, 4:28:23 PM] Mike Ireland: To communicate with the camera, you need to run “asgard_cam_server —socket” rather than just “ascard_cam_server”, then starting the camera is ‘fetct’ and direct camera commands are sent over zmq as e.g. ‘cli [“set gain 10”]’
[15/2/2025, 4:45:33 PM] Mike Ireland: For the camera interface, I ran “ZMQ_control_client.py” in an xterm on the (so far only) NoMachine client. The problem is that when communicating via ZMQ, “commander” fails if an invalid json string is given, e.g. no square brackets. The simplistic python interface also can’t handle the degrees symbol. So lots to do!
[15/2/2025, 4:46:44 PM] Mike Ireland: No rush on your side at all. But I think I’ll head to bed shortly (get closer to Australian time zone)
"""


"""
FLI_Cameras Module

This module provides an wrapped interface for controlling First Light Imaging (FLI) cameras,
The module supports multiple FLI camera types, including C-RED One, Two, and Three, and
offers methods for camera configuration, command execution, and image acquisition.

Example Usage:
```python
from FLI_Cameras import fli

# Initialize and configure the camera
camera = fli(cameraIndex=0)
camera.configure_camera("path/to/config.json")
camera.start_camera()

# Capture an image with reductions applied
image = camera.get_image(apply_manual_reduction=True)

# Save frames to a FITS file
camera.save_fits("output.fits", number_of_frames=10)
"""


### MDS
mds_host = "192.168.100.2"# France, Nice: "172.16.8.6"
mds_port = 5555 
timeout = 5000
context = zmq.Context()

context.socket(zmq.REQ)
mds_socket = context.socket(zmq.REQ)
mds_socket.setsockopt(zmq.RCVTIMEO, timeout)
mds_socket.connect( f"tcp://{mds_host}:{mds_port}")

#mds_state_dict = {"message_history": [], "socket": socket}



### CAMERA PORT 
cam_host = "tcp://192.168.100.2" # France, Nice:"tcp://172.16.8.6" #"tcp://mimir" #doesnt seem to work with mimir?
cam_port = 6667
context = zmq.Context()
cam_socket = context.socket(zmq.REQ)
cam_socket.connect(f"{cam_host}:{cam_port}") #cam_socket.connect(f"tcp://localhost:{cam_port}")

cmd_sz = 10 # finite size command with blanks filled


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


def extract_value(s):
    """
    when returning msgs from C-red 1 server they follow a certain format. 
    This function extracts the important bits of the striung
    specifically extracts and returns the substring between the first double quote (")
    and the literal '\\r\\n' sequence from the input string `s`, with surrounding
    whitespace removed.
    
    Parameters:
        s (str): The input string, e.g., '"  1739.356\\r\\nfli-cli>"'
        
    Returns:
        str or None: The extracted substring (with whitespace stripped) if found,
                     otherwise None.
    """

    pattern = r'^"\s*(.*?)\\r\\n'
    match = re.search(pattern, s)
    if match:
        return match.group(1).strip()
    
    return None



def get_bad_pixels( dark_frames, std_threshold = 20, mean_threshold=6):
    
    mean_frame = np.mean(dark_frames, axis=0)
    std_frame = np.std(dark_frames, axis=0)

    global_mean = np.mean(mean_frame)
    global_std = np.std(mean_frame)
    bad_pixel_map = (np.abs(mean_frame - global_mean) > mean_threshold * global_std) | (std_frame > std_threshold * np.median(std_frame))

    bad_pixels = np.where( bad_pixel_map )

    
    bad_pixel_mask = np.zeros( np.array(dark_frames[-1]).shape ).astype(bool)
    for ibad,jbad in list(zip(bad_pixels[0], bad_pixels[1])):
        bad_pixel_mask[ibad,jbad] = True

    return bad_pixels, bad_pixel_mask 




def detect_resets(data, threshold=None, axis=(1, 2), min_gap=10, k=15.0):
    """
    Detect reset points in burst-mode NDRO data.

    Parameters
    ----------
    data : ndarray
        3D array of shape (T, H, W), full NDRO sequence.
    threshold : float or None
        If specified, uses this absolute threshold for frame-to-frame mean jumps.
        If None, sets threshold = k * MAD of diff(mean_intensity).
    axis : tuple of int
        Axes to average over (typically spatial axes).
    min_gap : int
        Minimum number of frames between resets.
    k : float
        If threshold is None, use k × MAD as automatic threshold.

    Returns
    -------
    reset_indices : list of int
        Frame indices immediately before each reset (i.e., index of last frame before jump).
    """
    # 1. Mean intensity over spatial dimensions
    y = np.mean(data, axis=axis)  # shape (T,)
    dy = np.abs(np.diff(y))       # shape (T-1,)

    if threshold is None:
        # Use median absolute deviation (MAD) for robust threshold
        mad = np.median(np.abs(dy - np.median(dy)))
        threshold = k * mad
        print(f"[INFO] Auto-calculated threshold = {threshold:.3f} (k={k}, MAD={mad:.3f})")

    # 2. Initial reset candidates
    idx_raw = np.where(dy > threshold)[0]

    # 3. Enforce minimum spacing between resets
    idx_clean = []
    for i in idx_raw:
        if not idx_clean or (i - idx_clean[-1] >= min_gap):
            idx_clean.append(i)

    return idx_clean



def segment_ndro_stream(data, threshold=15.0):
    """
    Segment NDRO burst-mode stream into individual burst cubes.

    Parameters
    ----------
    data : ndarray
        3D array of shape (T, H, W).
    threshold : float
        Threshold for reset detection.

    Returns
    -------
    burst_list : list of ndarray
        Each item is an ndarray of shape (N_burst, H, W).
    """
    reset_idx = detect_resets(data, threshold=threshold)
    if not reset_idx:
        return [data]  # no reset detected

    starts = [0] + [i + 1 for i in reset_idx]
    stops = reset_idx + [data.shape[0]]

    burst_list = [data[start:stop] for start, stop in zip(starts, stops)]
    return burst_list


class fli( ):
    ## cred1 server now has crop_mode which crops (as of 13/9/25) only in y 
    # the server hasnt got a method to query the crop mode yet.
    # here we dont deal with it, just make an attribute that its cropped with the difference  , we just leave it. only place we care is when we update
    # the subframes for the cred1 

    def __init__(self, shm_target = "/dev/shm/cred1.im.shm" , roi=[None, None, None, None], config_file_path = None, quick_startup=False):
        #self.camera = FliSdk_V2.Init() # init camera object
        self.shm_loc = shm_target
        self.mySHM = shm(self.shm_loc, nosem=False)
        self.semid = 3
        self.mds_port = mds_port
        self.cam_port = cam_port

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

        print("Reading in current camera configuration\m")
        if quick_startup:
            # this was implemented just for ESO demo to make things faster - should not generally be used
            with open(os.path.join( self.config_file_path , "default_cred1_config.json"), "r") as file:
                default_cred1_config = json.load(file)  # Parses the JSON content into a Python dictionary
            config_dict = {} #to populate
            for k, v in default_cred1_config.items():
                if "fps" in k:
                    config_dict[k] = extract_value( self.send_fli_cmd( f"{k} raw" ) ) # reads the state
                elif "gain" in k:
                    config_dict[k] = extract_value( self.send_fli_cmd( f"{k} raw" ) ) # reads the state
                elif "mode" in k:
                    config_dict[k] = extract_value( self.send_fli_cmd( f"{k} raw" ) ) # reads the state
                else:
                    config_dict[k] = "place_holder" # watch out! 
            self.config = config_dict
        else:
            self.config = self.get_camera_config()
        
        #self.dark = [] 
        #self.bias = []
        #self.flat = []
        self.reduction_dict = {'bias':[], 'dark':[],'flat':[],'bad_pixel_mask':[]}
        #self.bad_pixel_mask = []
        
        # try get size to infer crop mode
        test_img = self.mySHM.get_data() # typically for full frame this should be 200 frames x 320x256
        if len(test_img.shape) < 3:
            raise RuntimeError(f"shm.get_data() method returns array that is not a data cube (frames x pix_x x pix_y) ")
        print(f"shm.get_data() returns array of shape {test_img.shape} ")
        
        if test_img[0].shape == (256,320):
            print('Camera in crop mode')
            self.crop_mode = False 
            self.y_offset = 0
           
        else:
            #_ = input("camera not in cropped mode. It must be in cropped mode for Baldr calibration, do you wish to continue") 
            print('Camera NOT in crop mode')
            self.crop_mode = True 
            self.y_offset = int( (256 - test_img[0].shape[0])  ) 

        # either way we reference the roi to the read in (potentially already cropped frame)        
        self.pupil_crop_region = roi 

        # if roi != [None, None, None,None]:
        #     # then we reference the roi to the cropped frame
        #     #print(f'inferred y-offset of {self.y_offset} pixels')

        #     self.pupil_crop_region = roi #[roi[0]-self.y_offset, roi[1]-self.y_offset, roi[2], roi[3]] # region of interest where we crop (post readout)
        # else:
        #     print("no roi input, so no crop offset applied") 
        #     self.pupil_crop_region = roi #[None,None,None,None]
        try:
            self.shm_shape = self.mySHM.get_data().shape
        except:
            print('failed to get shm shape. Set to ()')
            self.shm_shape = () 


        # # Dynamically inherit based on camera type
        if 1: # FliSdk_V2.IsCredOne(self.camera):
            #self.__class__ = type("FliCredOneWrapper", (self.__class__, FliCredOne.FliCredOne), {})
            #print("Inherited from FliCredOne")
            self.command_dict = cred1_command_dict

        #listOfGrabbers = FliSdk_V2.DetectGrabbers(self.camera)
        #listOfCameras = FliSdk_V2.DetectCameras(self.camera)
        # print some info and exit if nothing detected
        
        # if len(listOfGrabbers) == 0:
        #     print("No grabber detected, exit.")
        #     FliSdk_V2.Exit(self.camera)
        # if len(listOfCameras) == 0:
        #     print("No camera detected, exit.")
        #     FliSdk_V2.Exit(self.camera)
        # for i,s in enumerate(listOfCameras):
        #     print("- index:" + str(i) + " -> " + s)

        # #cameraIndex = int( input('input index corresponding to the camera you want to use') )
        
        # print(f'--->using cameraIndex={cameraIndex}')
        # # set the camera
        # camera_err_flag = FliSdk_V2.SetCamera(self.camera, listOfCameras[cameraIndex])
        # if not camera_err_flag:
        #     print("Error while setting camera.")
        #     FliSdk_V2.Exit(self.camera)
        # print("Setting mode full.")
        # FliSdk_V2.SetMode(self.camera, FliSdk_V2.Mode.Full)
        # print("Updating...")
        # camera_err_flag = FliSdk_V2.Update(self.camera)
        # if not camera_err_flag:
        #     print("Error while updating SDK.")
        #     FliSdk_V2.Exit(self.camera)



        # elif FliSdk_V2.IsCredTwo(self.camera):
        #     self.__class__ = type("FliCredTwoWrapper", (self.__class__, FliCredTwo.FliCredTwo), {})
        #     print("Inherited from FliCredTwo")
        #     self.command_dict = cred2_command_dict
        # elif FliSdk_V2.IsCredThree(self.camera):
        #     self.__class__ = type("FliCredThreeWrapper", (self.__class__, FliCredThree.FliCredThree), {})
        #     print("Inherited from FliCredThree")
        #     self.command_dict = cred3_command_dict
        # else:
        #     print("No compatible camera type detected.")
        #     FliSdk_V2.Exit(self.camera)
            
    # send FLI command (based on firmware version)
    def send_fli_cmd( self, cmd_raw ):

        cmd = f'cli "{cmd_raw}"'
        #cmd_sz = 10  # finite size command with blanks filled
        out_cmd = cmd + (cmd_sz - 1 - len(cmd)) * " " # fill the blanks
        cam_socket.send_string(out_cmd)
        
        #  Get the reply.
        resp = cam_socket.recv().decode("ascii")
        print(f"== Reply: [{resp}]")
        #val = FliSdk_V2.FliSerialCamera.SendCommand(self.camera, cmd)
        #if not val:
        #    print(f"Error with command {cmd}")

        # we update the config dict without asking the camera just to make things quicker
        # safer option would be to re-query camera each time - but this is slow! 
        if "Result:OK" in resp:
            if "set fps" in cmd_raw:
                self.config["fps"] = float( cmd_raw.split("fps ")[-1] )
            
            if "set gain" in cmd_raw:
                self.config["gain"] = float( cmd_raw.split("gain ")[-1] )
            

        return resp 
    

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
        # if self.send_fli_cmd( 'standby raw' )[1] == 'on':
        #     try :
        #         self.send_fli_cmd( f"set standby on" )
        #     except:
        #         raise UserWarning( "---\ncamera in standby mode and the fli command 'set standby off' failed\n " )
        
        for k, v in camera_config.items():

            time.sleep( sleep_time )

            self.send_fli_cmd( f"set {k} {v}")

            # for some reason set stanby mode timesout
            # if setting to the same state - so we manually check
            # before sending the command
            # if 'standby' in k:
            #     if v != self.send_fli_cmd( 'standby raw' )[1]:
            #         ok , _  = self.send_fli_cmd( f"set {k} {v}")
            #         if not ok :
            #             print( f"FAILED FOR set {k} {v}")

            # ok , _  = self.send_fli_cmd( f"set {k} {v}")
            # if not ok :
            #     print( f"FAILED FOR set {k} {v}")

        
    # # basic wrapper functions
    # def start_camera(self):
    #     # not really startig camera , but just the thread to pwrite to shm
    #     #fetch []
    #     #ok = FliSdk_V2.Start(self.camera)
    #     #return ok 

    #     cmd = "fetch []"
    #     #cmd_sz = 10  # finite size command with blanks filled
    #     out_cmd = cmd + (cmd_sz - 1 - len(cmd)) * " " # fill the blanks
    #     cam_socket.send_string(out_cmd)
        
    #     #  Get the reply.
    #     resp = cam_socket.recv()#.decode("ascii")
    #     print(f"== Reply: [{resp}]")
    #     #val = FliSdk_V2.FliSerialCamera.SendCommand(self.camera, cmd)
    #     #if not val:
    #     #    print(f"Error with command {cmd}")
    #     return resp 

    # def stop_camera(self):
    #     #ok = FliSdk_V2.Stop(self.camera)
    #     #return ok
    
    # def exit_camera(self):
    #     #FliSdk_V2.Exit(self.camera)

    def get_camera_config(self):
        config_dict = {} 
        # open the default config file to get the keys 
        "/home/asg/Progs/repos/asgard-alignment/config_files/default_cred1_config.json"
        with open(os.path.join( self.config_file_path , "default_cred1_config.json"), "r") as file:
            default_cred1_config = json.load(file)  # Parses the JSON content into a Python dictionary
            for k, v in default_cred1_config.items():
                config_dict[k] = extract_value( self.send_fli_cmd( f"{k} raw" ) ) # reads the state
        return( config_dict )
     

    # some custom functions
    def build_manual_bias( self , no_frames = 100 , sleeptime = 2, save_file_name = None, **kwargs):

        if "cds" in self.config["mode"]:
            maxfps = float( extract_value( self.send_fli_cmd("maxfps raw") ) ) #1739 #Hz

            priorfps = self.config["fps"] # this config should update everytime set fps cmd is sent 

            res = self.send_fli_cmd(f"set fps {maxfps}")

            print(f"response for setting fps = {maxfps}:{res}")

        
        ### here 
        # message = "off SBB"
        # mds_socket.send_string(message)
        # response = mds_socket.recv_string()#.decode("ascii")
        # print( response )
        
        time.sleep(sleeptime)
        
        if "cds" in self.config["mode"]:
            print('...getting frames')
            bias_list = self.get_some_frames(number_of_frames = no_frames, apply_manual_reduction=False, timeout_limit = 20000 )
            print('...aggregating frames')
            bias = np.mean(bias_list ,axis = 0).astype(int)
            self.reduction_dict['bias'].append( np.array( bias)  ) #ADU

        elif "bursts" in self.config["mode"]:


            print('...getting frames')
            bias_list = self.get_some_frames(number_of_frames = no_frames, apply_manual_reduction=False, timeout_limit = 20000 )
            print('...aggregating frames')
            # Segment data into burst blocks
            bursts = segment_ndro_stream(np.array(bias_list), threshold=15)  # auto threshold
            # For each burst, fit a line y = Bt + C; take slope as dark (ADU/s)
            fps = float(self.config["fps"])
            t_unit = 1.0 / fps
            slopes = []
            intercepts = []
            assert len( bursts ) > 1

            for burst in bursts[1:]: # we ignore the first index because we need to always fit where we have a fresh reset! 
                
                t = np.arange(len(burst)) * t_unit
                #y = np.mean(burst, axis=(1,2))
                #slopes_frame = np.zeros_like( burst[0] )
                intercept_frame = np.zeros_like( burst[0] )
                for i in range(burst.shape[1]):
                    for j in range(burst.shape[2]):

                        # if len(t) < 4:
                        #     continue  # too short to reliably fit
                        slope, intercept  = np.polyfit(t, burst[:,i,j], deg=1)
                        #slopes_frame[i,j] = slope.copy()    
                        intercept_frame[i,j] = intercept.copy()

                #slopes.append(slopes_frame)  # [ADU/s] for each pixel 
                intercepts.append(intercept_frame )
            
            #self.reduction_dict['dark'].append( np.median(slopes, axis=0) / self.config["gain"] )  # [ADU/s/gain] f
            # we get bias for free here! 
            self.reduction_dict['bias'].append( np.median(intercepts, axis=0).astype(int)  )  # [ADU] 

            if len(slopes) == 0:
                raise RuntimeError("No valid bursts found for NDRO dark estimation.")


        print("returning to previous fps")
        self.send_fli_cmd(f"set fps {priorfps}")

        print("turning BB source back on")
        ### here 
        # message = "on SBB"
        # mds_socket.send_string(message)
        # response = mds_socket.recv_string()#.decode("ascii")
        # print( response )
        time.sleep(2)

        print("Done.")


    def build_manual_dark( self , no_frames = 100 , sleeptime = 3, build_bad_pixel_mask = False , save_file_name = None, **kwargs):
        """
        gets a dark in units of ADU / s 
        """
        # try turn off source 
        #my_controllino.turn_off("SBB")
        ### here 
        message = "off SBB"
        mds_socket.send_string(message)
        response = mds_socket.recv_string()#.decode("ascii")
        print( response )
        
        print(f'turning off source and waiting {sleeptime}s')
        time.sleep(sleeptime) # wait a bit to settle
       


        # full frame variables here were used in previous rtc. 
        # maybe redundant now. 
        #fps = float( self.send_fli_cmd( "fps")[1] )
        #dark_fullframe_list = []
        
        #dark_list = []
        #for _ in range(no_frames):
        #    time.sleep(1/fps)
        #    dark_list.append( self.get_image(apply_manual_reduction  = False) )
        #    #dark_fullframe_list.append( self.get_image_in_another_region() ) 
        
        ## check the FPS and make sure consistent with current config file
        fps = extract_value( self.send_fli_cmd( f"fps raw" ) )
        # update if necessary.
        # This is important since darks are normalize ADU/s!
        if fps != self.config["fps"]:
            print("updating fps:{fps}")
            self.config["fps"] = fps 

        print('...getting frames')
        dark_list = self.get_some_frames(number_of_frames = no_frames, apply_manual_reduction=False, timeout_limit = 20000 )
        print('...aggregating frames')
        dark = np.mean(dark_list ,axis = 0).astype(int)
        # dark_fullframe = np.median( dark_fullframe_list , axis=0).astype(int)
        if self.pupil_crop_region == [None, None, None, None]:
            # then we ignore frame tags! 
            print('removing frame tags from dark')
            dark[0, 0:5] = np.mean(  np.array(dark)[1:,1:] )
            #print(dark[0, 0:5])


        if 1 : #"cds" in self.config["mode"]:
            if len( self.reduction_dict['bias'] ) > 0:
                print('...applying bias')
                dark -= self.reduction_dict['bias'][-1]

            #if len( self.reduction_dict['bias_fullframe']) > 0 :
            #    dark_fullframe -= self.reduction_dict['bias_fullframe'][0]
            print(f'...appending dark in units ADU/s calculated with current fps = {self.config["fps"]}')
            self.reduction_dict['dark'].append( (dark * float( self.config["fps"] ) / float( self.config["gain"] )).astype(int)  ) # ADU / s / gain
            #self.reduction_dict['dark_fullframe'].append( dark_fullframe )

        elif 0: #"bursts" in self.config["mode"]:

            # Segment data into burst blocks
            bursts = segment_ndro_stream(np.array(dark_list), threshold=15)  #  15 works well , if None auto threshold

            # For each burst, fit a line y = Bt + C; take slope as dark (ADU/s)
            fps = float(self.config["fps"])
            t_unit = 1.0 / fps
            slopes = []
            intercepts = []
            assert len( bursts ) > 1

            for burst in bursts[1:]: # we ignore the first index because we need to always fit where we have a fresh reset! 
                
                t = np.arange(len(burst)) * t_unit
                #y = np.mean(burst, axis=(1,2))
                slopes_frame = np.zeros_like( burst[0] )
                intercept_frame = np.zeros_like( burst[0] )
                for i in range(burst.shape[1]):
                    for j in range(burst.shape[2]):

                        # if len(t) < 4:
                        #     continue  # too short to reliably fit
                        slope, intercept  = np.polyfit(t, burst[:,i,j], deg=1)
                        slopes_frame[i,j] = slope.copy()    
                        intercept_frame[i,j] = intercept.copy()

                slopes.append(slopes_frame)  # [ADU/s] for each pixel 
                intercepts.append(intercept_frame )
            
            self.reduction_dict['dark'].append( (np.median(slopes, axis=0) / float( self.config["gain"] )).astype(int) )  # [ADU/s/gain] f
            # we get bias for free here! 
            self.reduction_dict['bias'].append( np.median(intercepts, axis=0).astype(int)  )  # [ADU] 

            if len(slopes) == 0:
                raise RuntimeError("No valid bursts found for NDRO dark estimation.")

        else:
            raise UserWarning("invalid mode. needs to be globalresetcds or globalresetbursts. check self.config")
        
        time.sleep(2)
        # try turn source back on 
        #my_controllino.turn_on("SBB")
        print("turning BB source back on")
        ### here 
        message = "on SBB"
        mds_socket.send_string(message)
        response = mds_socket.recv_string()#.decode("ascii")
        print( response )
        time.sleep(2)

        if build_bad_pixel_mask :
            print("building bad pixel mask on the darks")
            std_threshold = kwargs.get('std_threshold', 20)
            mean_threshold = kwargs.get('mean_threshold', 6)

            bad_pixels, bad_pixel_mask  = get_bad_pixels( dark_list, std_threshold = std_threshold, mean_threshold=mean_threshold)

            # take conjugate to mask mask true at pixels we want to keep
            self.reduction_dict['bad_pixel_mask'].append( ~bad_pixel_mask ) 

        print("Done.")

        if save_file_name is not None:
            
            #f"/home/asg/Progs/repos/asgard-alignment/calibration/cal_data/darks/dark_{}.fits"
            
            # Create PrimaryHDU using FRAMES
            primary_hdu = fits.PrimaryHDU(dark_list)
            primary_hdu.header['EXTNAME'] = 'DARK_FRAMES'  # This is not strictly necessary for PrimaryHDU

            # Append camera configuration to the primary header
            config_tmp = self.get_camera_config()
            for k, v in config_tmp.items():
                primary_hdu.header[k] = v

            # Create HDUList and add the primary HDU
            hdulist = fits.HDUList([primary_hdu])

            hdulist.writeto(save_file_name, overwrite=True)



    def get_bad_pixels( self, no_frames = 1000, std_threshold = 20, mean_threshold=6): #, flatten=False):
        # To get bad pixels we just take a bunch of images and look at pixel variance 
        #self.enable_frame_tag( True )
        time.sleep(0.5)
        #zwfs.get_image_in_another_region([0,1,0,4])
        
        message = "off SBB"
        mds_socket.send_string(message)
        response = mds_socket.recv_string()#.decode("ascii")
        print( response )
        time.sleep(5)
       
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

        ## Identify bad pixels
        mean_frame = np.mean(dark_list, axis=0)
        std_frame = np.std(dark_list, axis=0)

        global_mean = np.mean(mean_frame)
        global_std = np.std(mean_frame)
        bad_pixel_map = (np.abs(mean_frame - global_mean) > mean_threshold * global_std) | (std_frame > std_threshold * np.median(std_frame))

        message = "on SBB"
        mds_socket.send_string(message)
        response = mds_socket.recv_string()#.decode("ascii")
        print( response )
        time.sleep(2)

        #bad_pixels = np.where( bad_pixel_map )

        self.reduction_dict["bad_pixel_mask"].append( bad_pixel_map.astype(int) ) # save as int so can write to FITS files! 
        
        return bad_pixel_map 

    #self.bad_pixel_filter = badpixel_bool_array.reshape(-1)
    #self.bad_pixels = np.where( self.bad_pixel_filter )[0]

    #if not flatten:
    #    bad_pixels = np.where( bad_pixel_map )
    #else: 
    #    bad_pixels_flat = np.where( bad_pixel_map.reshape(-1) ) 

    ## OlD WAY 
    # dark_std = np.std( dark_list ,axis=0)
    # # define our bad pixels where std > 100 or zero variance
    # #if not flatten:
    # bad_pixels = np.where( (dark_std > std_threshold) + (dark_std == 0 ))
    # #else:  # flatten is useful for when we filter regions by flattened pixel indicies
    # bad_pixels_flat = np.where( (dark_std.reshape(-1) > std_threshold) + (dark_std.reshape(-1) == 0 ))

    # if not flatten:
    #     return bad_pixels 
    # else:
    #     return bad_pixels_flat 


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


    def get_data(self, apply_manual_reduction=False, which_index=-1):
        """ 
        # legacy function
        gets most recent 100 frames in buffer. 
        this is to be compatiple with origin SHM raw code
        other methods below are legacy and allow compatibility with previously implemented
        (SDK) camera instances of this class. updated to work with SHM
        
        apply_manual_reduction=True reduces image using self.reduction_dict
        which_index indicates which index in reduction_dict lists to use. Default (-1) is the most recent
        """
        #self.mySHM.catch_up_with_sem(self.semid)
        img = self.mySHM.get_latest_data(self.semid)

        if not apply_manual_reduction:
            #img = FliSdk_V2.GetRawImageAsNumpyArray( self.camera , -1)
            #img = self.mySHM.get_data() #FliSdk_V2.GetProcessedImageGrayscale16bNumpyArray(self.camera, -1)
            cropped_img = img[:,self.pupil_crop_region[0]:self.pupil_crop_region[1],self.pupil_crop_region[2]: self.pupil_crop_region[3]].astype(int)  # make sure int and not uint16 which overflows easily     
        else :
            #img = FliSdk_V2.GetRawImageAsNumpyArray( self.camera , -1)
            #img = self.mySHM.get_data() #FliSdk_V2.GetProcessedImageGrayscale16bNumpyArray(self.camera, -1)
            cropped_img = img[:, self.pupil_crop_region[0]:self.pupil_crop_region[1],self.pupil_crop_region[2]: self.pupil_crop_region[3]].astype(int)  # make sure 

            if len( self.reduction_dict['bias'] ) > 0:
                cropped_img -= self.reduction_dict['bias'][which_index] # take the most recent bias. bias must be set in same cropping state 

            if len( self.reduction_dict['dark'] ) > 0:
                # Darks are now adu/s so divide by fps
                cropped_img = cropped_img - np.array( float(self.config["gain"]) / float(self.config["fps"]) * self.reduction_dict['dark'][which_index], dtype = type( cropped_img[0][0][0]) ) # take the most recent dark. Dark must be set in same cr

            if len( self.reduction_dict['flat'] ) > 0:
                # build this with pupil filter and set outside to mean pupil (ADU/s)
                cropped_img /= np.array( self.reduction_dict['flat'][which_index] , dtype = type( cropped_img[0][0][0]) ) # take the most recent flat. flat must be set in same cropping state 

            if len( self.reduction_dict['bad_pixel_mask'] ) > 0:
                # enforce the same type for mask
                #cropped_img *= np.array( self.reduction_dict['bad_pixel_mask'][which_index] , dtype = type( cropped_img[0][0]) ) # bad pixel mask must be set in same cropping state 
                # Just set to zero for now!
                #cropped_img[:,~self.reduction_dict['bad_pixel_mask'][which_index]] = 0
                if len(np.array(cropped_img).shape)==3:
                    cropped_img[:, ~self.reduction_dict['bad_pixel_mask'][which_index].astype(bool)] = 0 #, dtype = type( cropped_img[0][0]) ) # bad pixel mask must be set in same cropping state 
                elif len(np.array(cropped_img).shape)==2:
                    cropped_img[~self.reduction_dict['bad_pixel_mask'][which_index].astype(bool)] = 0 #, dtype = type( cropped_img[0][0]) ) # bad pixel mask must be set in same cropping state 
        return(cropped_img)    


    def get_last_raw_image_in_buffer(self):
        
        img = self.mySHM.get_latest_data_slice(self.semid) # typically its a buchch of 100 frames so get last one 

        #img = FliSdk_V2.GetRawImageAsNumpyArray(self.camera, -1)
        return img 
    

    def get_image(self, apply_manual_reduction  = True, which_index = -1 ):

        # I do not check if the camera is running. Users should check this 
        # gets the last image in the buffer
        if not apply_manual_reduction:
            #img = FliSdk_V2.GetRawImageAsNumpyArray( self.camera , -1)
            img = self.mySHM.get_latest_data_slice(self.semid)  #FliSdk_V2.GetProcessedImageGrayscale16bNumpyArray(self.camera, -1)
            cropped_img = img[self.pupil_crop_region[0]:self.pupil_crop_region[1],self.pupil_crop_region[2]: self.pupil_crop_region[3]].astype(int)  # make sure int and not uint16 which overflows easily     
        else :
            #img = FliSdk_V2.GetRawImageAsNumpyArray( self.camera , -1)
            img = self.mySHM.get_latest_data_slice(self.semid) #self.mySHM.get_data()[-1] #FliSdk_V2.GetProcessedImageGrayscale16bNumpyArray(self.camera, -1)
            cropped_img = img[self.pupil_crop_region[0]:self.pupil_crop_region[1],self.pupil_crop_region[2]: self.pupil_crop_region[3]].astype(int)  # make sure 

            if len( self.reduction_dict['bias'] ) > 0:
                cropped_img -= np.array(self.reduction_dict['bias'][which_index], dtype = type( cropped_img[0][0]) ) # take the most recent bias. bias must be set in same cropping state 

            if len( self.reduction_dict['dark'] ) > 0:
                # darks are ADU / s so adjust 
                cropped_img -= np.array( float(self.config["gain"]) / float(self.config["fps"]) * self.reduction_dict['dark'][which_index], dtype = type( cropped_img[0][0]) ) # take the most recent dark. Dark must be set in same cropping state 

            if len( self.reduction_dict['flat'] ) > 0:
                # flat are ADU / s. build to set outside pupil to mean interior 
                cropped_img /= np.array( float(self.config["gain"]) / float(self.config["fps"]) *  self.reduction_dict['flat'][which_index] , dtype = type( cropped_img[0][0]) ) # take the most recent flat. flat must be set in same cropping state 

            if len( self.reduction_dict['bad_pixel_mask'] ) > 0:
                # enforce the same type for mask
                cropped_img *= np.array( self.reduction_dict['bad_pixel_mask'][which_index] , dtype = type( cropped_img[0][0]) ) # bad pixel mask must be set in same cropping state 

        return(cropped_img)    

    def get_image_in_another_region(self, crop_region=[None,None,None,None]):
        # useful if we want to look outside of the region of interest 
        # defined by self.pupil_crop_region

        #img = FliSdk_V2.GetRawImageAsNumpyArray( self.camera , -1)
        self.mySHM.catch_up_with_sem(self.semid)
        img = self.mySHM.get_latest_data_slice(self.semid) #FliSdk_V2.GetProcessedImageGrayscale16bNumpyArray(self.camera, -1)
        cropped_img = img[crop_region[0]:crop_region[1],crop_region[2]: crop_region[3]].astype(int)  # make sure int and not uint16 which overflows easily     
        
        #if type( self.pixelation_factor ) == int : 
        #    cropped_img = util.block_sum(ar=cropped_img, fact = self.pixelation_factor)
        #elif self.pixelation_factor != None:
        #    raise TypeError('ZWFS.pixelation_factor has to be of type None or int')
        return( cropped_img )    
    

    def get_some_frames(self, number_of_frames = 10, apply_manual_reduction=True, timeout_limit = 20000 ):
        """
        poll sequential frames and store in list  
        used for calibration and not real-time applications 
        """
        frames = []
        cnt = [] # secondary check that we dont skip frames 
        self.mySHM.catch_up_with_sem(self.semid)
        # do this as fast as possible (manual reduction is done after)
        while len(frames) < number_of_frames:

            fullframe = self.mySHM.get_latest_data_slice( self.semid )
            frames.append( fullframe[self.pupil_crop_region[0]:self.pupil_crop_region[1],self.pupil_crop_region[2]: self.pupil_crop_region[3]] )
            cnt.append( fullframe[0][0] )

        # delete this later but keep for now to test behaviour!         
        
        if np.max(np.diff(cnt)) == 1 :
            None
        elif np.max(np.diff(cnt)) > 1 :
            print(f"some skipped frames. Max frames skipped = {np.max(np.diff(cnt))}")
        
        else:
            print("something strange is happending ")

        if apply_manual_reduction:
            which_index = -1 # use most recent reduction frames in self.reduction_dict
            frames = np.array( frames).astype(int)
            red_frames = []
            for cropped_img in frames:
                if len( self.reduction_dict['bias'] ) > 0:
                    cropped_img -= np.array(self.reduction_dict['bias'][which_index], dtype = type( cropped_img[0][0]) ) # take the most recent bias. bias must be set in same cropping state 

                if len( self.reduction_dict['dark'] ) > 0:
                    # darks are in ADU/s 
                    cropped_img -= np.array( float(self.config["gain"]) / float(self.config["fps"]) * self.reduction_dict['dark'][which_index], dtype = type( cropped_img[0][0]) ) # take the most recent dark. Dark must be set in same cropping state 

                if len( self.reduction_dict['flat'] ) > 0:
                    # darks are in ADU/s 
                    cropped_img /= np.array( float(self.config["gain"]) / float(self.config["fps"]) * self.reduction_dict['flat'][which_index] , dtype = type( cropped_img[0][0]) ) # take the most recent flat. flat must be set in same cropping state 

                if len( self.reduction_dict['bad_pixel_mask'] ) > 0:
                    # enforce the same type for mask
                    cropped_img *= np.array( self.reduction_dict['bad_pixel_mask'][which_index] , dtype = type( cropped_img[0][0]) ) # bad pixel mask must be set in same cropping state 
                red_frames.append( cropped_img )
        
            return np.array( red_frames ) 
        
        else:
            return np.array( frames ).astype(int)
        


    def save_fits( self , fname ,  number_of_frames=100, apply_manual_reduction=True ):

        #hdulist = fits.HDUList([])

        frames = self.get_some_frames( number_of_frames=number_of_frames, apply_manual_reduction=apply_manual_reduction,timeout_limit=20000)
        
        # Convert list to numpy array for FITS compatibility
        data_array = np.array(frames, dtype=float)  # Ensure it is a float array or any appropriate type

        # # Create a new ImageHDU with the data
        # hdu = fits.ImageHDU( data_array )

        # # Set the EXTNAME header to the variable name
        # hdu.header['EXTNAME'] = 'FRAMES'
        # #hdu.header['config'] = config_file_name

        # config_tmp = self.get_camera_config()
        # for k, v in config_tmp.items():
        #     hdu.header[k] = v
        
        # Create PrimaryHDU using FRAMES
        primary_hdu = fits.PrimaryHDU(data_array)
        primary_hdu.header['EXTNAME'] = 'FRAMES'  # This is not strictly necessary for PrimaryHDU

        # Append camera configuration to the primary header
        config_tmp = self.get_camera_config()
        for k, v in config_tmp.items():
            primary_hdu.header[k] = v

        # Create HDUList and add the primary HDU
        hdulist = fits.HDUList([primary_hdu])

        # append reduction info
        for k, v in self.reduction_dict.items():
            if len(v) > 0 :
                if np.array( v[-1] ).dtype == 'bool':
                    hdu = fits.ImageHDU(  np.array( v[-1] ).astype(int) )
                else:
                    hdu = fits.ImageHDU( v[-1] )
                hdu.header['EXTNAME'] = k
                hdulist.append(hdu)
            else: # we just append empty list to show that its empty!
                hdu = fits.ImageHDU( v )
                hdu.header['EXTNAME'] = k
                hdulist.append(hdu)

        hdulist.writeto(fname, overwrite=True)


    def close(self, erase_file=False):
        #print('setting gain = 1 before closing')
        #self.send_fli_cmd( "set gain 1" )
        self.mySHM.close( erase_file = erase_file)
        print(f'closed camera SHM that used target {self.shm_loc}') 

    # ensures we exit safely and set gain to unity
    def __del__(self):
        # Cleanup when object is deleted
        if hasattr(self, 'camera') and self.camera is not None:
            self.send_fli_cmd( "set gain 1" )
            
            self.close( erase = False)

            #FliSdk_V2.Exit(self.camera)
            print("Camera SHM exited cleanly.")




if __name__ == "__main__":
    
    # example to get series of darks in different modes
    # and save as fits 
    # camera operating modes are 
    #    - single read (set mode globalresetsingle)
    #    - correlated double sampling (set mode globalresetcds)
    #    - multiple non-destructive reads (set mode globalresetbursts)
    #    - rolling versions of these modes (set mode rollingresetsingle)
    # see section 7. Camera Operating Modes from C-RED 1 user manual
    
    #!!!!!!!!! turn off sources first!!!!!!!!


    data_path = '/home/asg/Videos/cred1_dark_analysis/'
    if not os.path.exists( data_path ):
        os.makedirs( "data_path")

    tstamp = datetime.datetime.now().strftime("%d-%m-%YT%H.%M.%S")
    roi = [None, None, None, None] # No region of interest
    #aduoffset = 1000 # to avoid overflow with negative
    # init camera
    c = fli(roi=roi) #cameraIndex=0, roi=[100,200,100,200])

    cam_config = c.get_camera_config()
    # print the camera commands available in send_fli_cmd method
    c.print_camera_commands()

    # set up 
    # config_file_name = os.path.join( c.config_file_path , "default_cred1_config.json")
    # c.configure_camera( config_file_name )
    # c.send_fli_cmd( f"set aduoffset {aduoffset}")
    #FliSdk_V2.Update(c.camera)

    # start
    print( c.send_fli_cmd( "status" ) )

    # c.build_manual_dark()
    #bp = c.get_bad_pixel_indicies( no_frames = 100, std_threshold = 100 , flatten=False)
    # c.a
    #f = c.get_some_frames( number_of_frames=10, apply_manual_reduction=True)
    #c.save_fits('/home/heimdallr/Downloads/test_imgs.fits', number_of_frames=10, apply_manual_reduction=True )

    dark_dict = {}
    gain_grid = np.arange(1, 6).astype(int)
    fps_grid = [100, 200, 500, 1000, 1700] #[25, 50, 100, 200, 500, 1000, 1700]
    number_of_frames = 500
    for cnt, gain in enumerate( gain_grid ):
        
        print( f"{cnt / len( gain_grid )}% complete" )

        c.send_fli_cmd( f"set gain {gain}" )
        time.sleep(3)

        dark_dict[gain] = {}

        for fps in fps_grid:
            
            c.send_fli_cmd( f"set fps {fps}" )
            time.sleep(3)

            dark_dict[gain][fps] =  c.get_some_frames(number_of_frames=number_of_frames,\
                                                        apply_manual_reduction=False, timeout_limit = 20000)  
    


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
    plt.axhline( float(cam_config["aduoffset"]), ls=":", color='k', label="ADU offset")
    for gain, values in mean_pixel_values.items():
        fps_values, mean_values = zip(*values)
        plt.plot(fps_values, mean_values, marker='o', label=f'Gain {gain}')
    plt.xscale('log')
    plt.xlabel('Frame Rate [frames per second]', fontsize=15)
    plt.ylabel('Mean Pixel Value', fontsize=15)
    plt.title('Mean Pixel Value vs Frame Rate for Different Gains')
    plt.gca().tick_params(labelsize=15)
    plt.legend()
    plt.savefig( data_path + 'dark_mean_vs_fps-gain.png')

    # Plot 2: Standard deviation of pixel values vs. frame rate for different gain settings
    plt.figure(figsize=(10, 6))
    for gain, values in std_pixel_values.items():
        fps_values, std_values = zip(*values)
        plt.plot(fps_values, std_values, marker='o', label=f'Gain {gain}')
    plt.xscale('log')
    plt.xlabel('Frame Rate [frames per second]', fontsize=15)
    plt.ylabel('Standard Deviation of Pixel Value', fontsize=15)
    plt.title('Pixel Value Standard Deviation vs Frame Rate for Different Gains')
    plt.legend()
    plt.gca().tick_params(labelsize=15)
    plt.savefig(data_path + 'std_vs_fps-gain.png')

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
            #hdu.header['config'] = config_file_name

            # config_tmp = c.get_camera_config()
            # for k, v in config_tmp.items():
            #     hdu.header[k] = v
            # # Append the HDU to the HDU list
            hdulist.append(hdu)

            # 
            hdulist.writeto(data_path + f'dark_FPS-{round(fps,1)}__GAIN-{round(gain,1)}_{tstamp}.fits', overwrite=True)



# if len(self.shm_shape) == 3:

#     if self.shm_shape[0] >= number_of_frames :
#         ref_img_list = list(self.mySHM.get_data())

#     elif self.shm_shape[0] < number_of_frames :
#         ref_img_list = list(self.mySHM.get_data())
#         while (len( ref_img_list  ) < number_of_frames) : #and (not timeout_flag):
#             ref_img_list = ref_img_list + list(self.mySHM.get_data())

#     ref_img_list = list(np.array( ref_img_list)[ : number_of_frames, self.pupil_crop_region[0]:self.pupil_crop_region[1],self.pupil_crop_region[2]: self.pupil_crop_region[3]] )

# if len(self.shm_shape) == 2: # then probably just the most recent (polling last)
#     ref_img_list = []
#     i=0
#     timeout_counter = 0 
#     timeout_flag = 0
#     while (len( ref_img_list  ) < number_of_frames) and not timeout_flag: # poll  individual images
#         if timeout_counter > timeout_limit: # we have done timeout_limit iterations without a frame update
#             timeout_flag = 1 
#             raise TypeError('timeout! timeout_counter > 10000')

#         full_img = self.get_image_in_another_region() # empty argunment for full frame
#         current_frame_number = full_img[0][0] #previous_frame_number
#         if i==0:
#             previous_frame_number = current_frame_number
#         if current_frame_number != previous_frame_number:
#             timeout_counter = 0 # reset timeout counter
#             # if current_frame_number == 65535:
#             #     previous_frame_number = -1 #// catch overflow case for int16 where current=0, previous = 65535
#             # else:
#             previous_frame_number = current_frame_number 
#             ref_img_list.append( self.get_image( apply_manual_reduction  = apply_manual_reduction) )
#         i+=1
#         timeout_counter += 1
#     else:
#         raise UserWarning('nothing mettt here')

