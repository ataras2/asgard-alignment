from xaosim.shmlib import shm
import time
import numpy as np
from io import StringIO
import sys
from asgard_alignment import FLI_Cameras as FLI


def run_camera_test():
    """Function to test camera functionality and return results as a list of messages."""
    
    # Redirect standard output
    # old_stdout = sys.stdout
    # sys.stdout = mystdout = StringIO()
        
    ## Testing if cam server / semaphores can keep up with frame rate
    c = FLI.fli( "/dev/shm/cred1.im.shm", roi = [None,None,None,None])

    # for safety
    print('setting camera gain to 1 for safety')
    c.send_fli_cmd( f"set gain 1" )

    print("============= TESTS ===============")
    print(f"CRED ONE object looking at shared memory address {c.shm_loc}")
    print(f"CRED ONE object sending camera commands via ZMQ to port:{c.cam_port}")
    print(f"CRED ONE object sending MDS commands via ZMQ to port:{c.mds_port}")
    print(f"CRED ONE object using semaphore ID:{c.semid}")

    print("test sending a basic command to camera server")
    # test communication to camera server 
    try:
        res = c.send_fli_cmd("fps")
        if "frames per second" in res.lower():
            comm_good = True
        else:
            comm_good = False
    except:
        comm_good = False

    print("++++check if imagetags and/or cropping is enabled")
    if comm_good:
        # check if image tagging (frame counter) is active
        res = c.send_fli_cmd( "imagetag")
        if "image tags state: on" in res.lower():
            image_tags = True
        else:
            image_tags = False
        # check if cropping is on 
        res = c.send_fli_cmd( "cropping")
        if "cropping off" in res.lower():
            cropping = False
        else:
            cropping = True

    if (not cropping) and (image_tags):
        print("++++scan different frame rates and check for skipped frames")
        cannot_count_frames = False

        skipping_list = []
        frame_std_list = []  
        fps_grid = [100, 500, 1000, 1500]          
        for fps in fps_grid:
            c.send_fli_cmd (f"set fps {fps}")

            time.sleep(5)

            c.mySHM.catch_up_with_sem(c.semid)
            frames = c.mySHM.get_latest_data(c.semid)

            # variance in frames?
            std_frames = np.std( frames )

            # skipped frames? 
            med_skipped_frames = np.median(  np.diff([aa[0][0] for aa in frames])  )

            print("fps = ", fps, f"std of frames = {std_frames}", f", median frames skipped {med_skipped_frames}" )
        
            skipping_list.append( med_skipped_frames )
            frame_std_list.append( std_frames )
        
        size_frames = frames.shape

    else:
        cannot_count_frames = True


    trouble_skip = np.where( np.array(skipping_list) != 1.0 )[0]
    trouble_std = np.where( np.array(frame_std_list) == 0 )[0]

    output = [ ]
    if not comm_good:
        output.append( "camera class cannot communicate with camera server. Check server port")

    if cannot_count_frames:
        if not image_tags:
            output.append( "image tags is off. Cannot count frames. Try 'set imagetags on'" )
        if cropping:
            output.append( "Frame cropping is on. Cannot reliably count frames. Try 'set cropping off'" )
            

    if not cannot_count_frames:
        if len(trouble_skip) == 0:
            output.append("No Frames skipped on CRED 1")
        else:
            output.append( "CRED 1 Frames were skipped. Try restarting camera server" ) 

        if len(trouble_std) == 0:
            output.append( "CRED 1 Frames seem to be updating fine")
        else:
            output.append( "CRED 1 Frames seem to not be updating. Try 'fetch' on camera server")

    # print results
    print("============= RESULTS ===============")
    for mes in output:
        print( mes )

    c.close( erase_file=False )

    # Restore stdout and capture output
    # sys.stdout = old_stdout
    # script_output = mystdout.getvalue()
    print("FINAL EXIT") 
    #sys.exit(0)  # Success exit code

    #return output



if __name__ == "__main__":
    run_camera_test()