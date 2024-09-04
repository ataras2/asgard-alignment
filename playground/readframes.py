import numpy as np
import sys  
from astropy.io import fits 
sys.path.insert(1, '/opt/FirstLightImaging/FliSdk/Python/demo/')
import FliSdk_V2


def get_some_frames(camera, number_of_frames = 100, timeout_limit = 20000 , to_correct_overflow=True):
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

        img = FliSdk_V2.GetRawImageAsNumpyArray( camera , -1).astype(np.int32) # we can also specify region (#zwfs.get_image_in_another_region([0,1,0,4]))
        current_frame_number = img[0][0] #previous_frame_number
        if i==0:
            previous_frame_number = current_frame_number
        if current_frame_number > previous_frame_number:
            timeout_counter = 0 # reset timeout counter
            if current_frame_number == 65535:
                previous_frame_number = -1 #// catch overflow case for int16 where current=0, previous = 65535
            else:
                previous_frame_number = current_frame_number 
                if to_correct_overflow:
                    ref_img_list.append( correct_overflow( img ) )
                else:
                    ref_img_list.append(  img  )
        i+=1
        timeout_counter += 1
        
    return( ref_img_list )  

def correct_overflow(unsigned_array):
    """
    correct overflow in unsigned array
    """
    # check for overflow
    overflow = np.where(unsigned_array > 2**15)
    overflow[0,0] = False # first frame is for counting frame number
    # correct overflow
    unsigned_array[overflow] = unsigned_array[overflow] - 2**16
    
    return unsigned_array


if __name__ == "__main__":

    cameraIndex = 0
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

    # set frametags on 
    FliSdk_V2.FliSerialCamera.SendCommand(camera, "set imagetags on")
    
    # start the camera
    FliSdk_V2.Start(camera)

    # get some frames 
    frames = get_some_frames(camera, number_of_frames = 5, timeout_limit = 20000, to_correct_overflow=True )

    # basic info for headers 
    ok, tint = FliSdk_V2.FliSerialCamera.SendCommand(camera, "tint")
    ok, fps = FliSdk_V2.FliSerialCamera.SendCommand(camera, "fps")

    # write to fits file
    frame_fits = fits.PrimaryHDU( frames )
    frame_fits.header.set('tint',tint.split(': ')[-1])
    frame_fits.header.set('fps',fps.split(': ')[-1])

    savepath = '/home/heimdallr/Documents/asgard-alignment/tmp/'
    print(f'--->saving to {savepath}frame.fits')
    frame_fits.writeto(savepath + 'frame.fits', overwrite=True)

