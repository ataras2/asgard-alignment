import numpy as np
import time
from scipy.ndimage import center_of_mass
import matplotlib.pyplot as plt 
import scipy.interpolate as interp

def compute_image_difference(img1, img2):
    # normalize both images first
    img1 = img1.copy() /np.sum(img1)
    img2 = img2.copy() /np.sum(img2)
    return np.sum(np.abs(img1 - img2))

def calculate_movement_directions(image):
    """
    Calculate the direction to move the phase mask to improve symmetry.
    
    Parameters:
    - image: 2D numpy array representing the image.
    
    Returns:
    - Tuple of (dx, dy) indicating the direction to move the phase mask.
    """
    y_center, x_center = np.array(image.shape) // 2

    # Extract the four quadrants
    q1 = image[:y_center, :x_center]  # Top-left
    q2 = np.flip(image[y_center:, :x_center], axis=0)  # Bottom-left (flipped)
    q3 = np.flip(image[:y_center, x_center:], axis=1)  # Top-right (flipped)
    q4 = np.flip(image[y_center:, x_center:], axis=(0, 1))  # Bottom-right (flipped)

    # Calculate the differences
    diff1 = np.sum(np.abs(q1 - q4))
    diff2 = np.sum(np.abs(q2 - q3))

    # Determine movement directions based on differences
    dx = (np.sum(np.abs(q3 - q1)) - np.sum(np.abs(q2 - q4))) / (np.sum(np.abs(q3 + q1)) + np.sum(np.abs(q2 + q4)))
    dy = (np.sum(np.abs(q2 - q1)) - np.sum(np.abs(q3 - q4))) / (np.sum(np.abs(q2 + q1)) + np.sum(np.abs(q3 + q4)))

    # Normalize to unit length
    magnitude = np.sqrt(dx**2 + dy**2)
    if magnitude > 0:
        dx /= magnitude
        dy /= magnitude

    return dx, dy

def is_symmetric(image, threshold=0.1):
    """
    Check if the image is symmetric and calculate the direction to move for better symmetry.
    
    Parameters:
    - image: 2D numpy array representing the image.
    - threshold: float, maximum allowable difference for symmetry to be considered acceptable.
    
    Returns:
    - Tuple of (is_symmetric, (dx, dy)) indicating whether the image is symmetric and the direction to move.
    """
    y_center, x_center = np.array(image.shape) // 2

    # Extract the four quadrants
    q1 = image[:y_center, :x_center]  # Top-left
    q2 = np.flip(image[y_center:, :x_center], axis=0)  # Bottom-left (flipped)
    q3 = np.flip(image[:y_center, x_center:], axis=1)  # Top-right (flipped)
    q4 = np.flip(image[y_center:, x_center:], axis=(0, 1))  # Bottom-right (flipped)

    # Calculate the differences
    diff1 = np.sum(np.abs(q1 - q4))
    diff2 = np.sum(np.abs(q2 - q3))
    
    # Determine if the image is symmetric
    symmetric = diff1 <= threshold and diff2 <= threshold

    # Calculate the direction to move if not symmetric
    if not symmetric:
        dx, dy = calculate_movement_directions(image)
    else:
        dx, dy = 0, 0

    return symmetric, (dx, dy)

def spiral_search_and_center(zwfs, phasemask, phasemask_name, search_radius, dr, dtheta, reference_img, fine_tune_threshold=3, savefigName=None, usr_input=True):

    phasemask.move_to_mask( phasemask_name ) # move to phasemask   
    initial_pos = phasemask.phase_positions[phasemask_name] # set initial position  


    x, y = initial_pos 
    angle = 0
    radius = 0
    plot_cnt = 0 # so we don't plot every iteration 
    
    diff_list = [] # to track our metrics 
    x_pos_list = [] 
    y_pos_list = []
    sleep_time = 0.7 #s
    while radius < search_radius:
        x_pos = x + radius * np.cos(angle)
        y_pos = y + radius * np.sin(angle)

        phasemask.move_absolute([x_pos, y_pos])
        time.sleep( sleep_time)  # wait for the phase mask to move and settle
        img = np.mean(zwfs.get_some_frames(number_of_frames = 10, apply_manual_reduction = True ) , axis=0 )
    
        initial_img = img.copy() # take a copy of original image 

        diff = compute_image_difference(img, reference_img)
        diff_list.append( diff )
        x_pos_list.append( x_pos )
        y_pos_list.append( y_pos )
        print(f'img diff = {diff}, fine_tune_threshold={fine_tune_threshold}')

        # Update for next spiral step
        angle += dtheta
        radius += dr


        #print( radius )
        #_ = input('next')
        if savefigName != None: 
            if np.mod( plot_cnt , 5) == 0:

                norm = plt.Normalize(0 , fine_tune_threshold)

                fig,ax = plt.subplots( 1,3 ,figsize=(20,7))
                ax[0].set_title( f'image\nphasemask={phasemask_name}' )
                ax[1].set_title( f'search positions\nx:{phasemask.motors["x"]}\ny:{phasemask.motors["y"]}' )
                ax[2].set_title( 'search metric' )

                ax[0].imshow( img )
                ax[1].plot( [x_pos,y_pos] , 'x', color='r', label='current pos')
                ax[1].plot( [initial_pos[0],initial_pos[1]] , 'o', color='k', label='current pos')
                tmp_diff_list = np.array(diff_list)
                tmp_diff_list[tmp_diff_list < 1e-5 ] = 0.1 # very small values got to finite value (errors whern 0!)
                # s= np.exp( 400 * np.array(tmp_diff_list) / fine_tune_threshold )
                ax[1].scatter( x_pos_list, y_pos_list , s = 10   ,\
                 marker='o', c=diff_list, cmap='viridis', norm=norm)
                ax[1].set_xlim( [initial_pos[0] - search_radius,  initial_pos[0] + search_radius] )
                ax[1].set_ylim( [initial_pos[1] - search_radius,  initial_pos[1] + search_radius] )
                ax[1].legend() 
                ax[2].plot( diff_list )
                ax[2].set_xlim( [0, search_radius/dr] )

                ax[0].axis('off')
                ax[1].set_ylabel( 'y pos (um)' )
                ax[1].set_xlabel( 'x pos (um)' )
                ax[2].set_ylabel( r'$\Sigma|img - img_{off}|$' )
                ax[2].set_xlabel( 'iteration' )
                plt.savefig( savefigName)
                plt.close()
            plot_cnt += 1

    best_pos = [ x_pos_list[np.argmax( diff_list )], y_pos_list[np.argmax( diff_list )] ]

    if usr_input :
        move2best = int(input( f'Spiral search complete. move to recommended best position = {best_pos}? enter 1 or 0'))
    else:
        move2best = True

    if move2best :
        phasemask.move_absolute( best_pos )
    else :
        print('moving back to initial position')
        phasemask.move_absolute( initial_pos )

    #phasemask.move_absolute( phasemask.phase_positions[phasemask_name]  )
    time.sleep(0.5)
    if savefigName != None: 
        img =  np.mean(zwfs.get_some_frames(number_of_frames = 10, apply_manual_reduction = True ) , axis=0 )
        plt.figure();plt.imshow( img ) ;plt.savefig( savefigName )
        plt.close()
    if usr_input :
        do_fine_adjustment = int(input('ready for fine adjustment? enter 1 or 0') )
    else:
        do_fine_adjustment = False

    if do_fine_adjustment:
        # do fine adjustments 
        fine_adj_imgs = []
        for i in range(5):
            img = np.mean(zwfs.get_some_frames(number_of_frames = 20, apply_manual_reduction = True ) , axis=0 )
            fine_adj_imgs.append( img )
            #dr = dr/2 # half movements each time  
            dx, dy = calculate_movement_directions(img) # dx, dy are normalized to radius 1
            phasemask.move_relative( [dr * dx, dr * dy] ) 

            if savefigName != 0 :
                fig,ax = plt.subplots( 1,2 ,figsize=(14,7))
                ax[0].imshow( fine_adj_imgs[0] )
                ax[1].imshow( fine_adj_imgs[-1] )
                ax[0].set_title('origin of fine adjustment')
                ax[1].set_title('current image of fine adjustment')
                plt.savefig( savefigName )
                plt.close()
            
            # fig,ax = plt.subplots( len(fine_adj_imgs))
            # for img,axx in zip(fine_adj_imgs,ax.reshape(-1)):
            #     axx.imshow( img )
            # plt.savefig( savefigName )

    if usr_input:
        manual_alignment  = int(input('enter manual alignment mode? enter 1 or 0') )
        if manual_alignment:
            move_relative_and_get_image( zwfs, phasemask , savefigName=savefigName  )
            
    if not usr_input: # we by default save the final image
        tstamp = datetime.datetime.now().strftime("%d-%m-%YT%H.%M.%S")
        final_img =  np.mean(zwfs.get_some_frames(number_of_frames = 10, apply_manual_reduction = True ) , axis=0 )
        fig,ax = plt.subplots( 1,2 ,figsize=(5,10))
        ax[0].imshow( initial_img )
        ax[1].imshow( final_img )
        ax[0].set_title(f'initial ({phasemask_name})')
        ax[1].set_title(f'final ({phasemask_name})')
        plt.savefig( f'tmp/phasemask_alignment_SYD_{phasemask_name}_{tstamp}.png' ,dpi=150  )
        plt.close()

    if usr_input:
        save_pos = int(input('save position? enter 1 or 0') )
    else:
        save_pos = True
    
    if save_pos :
        phasemask.update_mask_position( phasemask_name )

    return phasemask.get_position()
   
def move_relative_and_get_image( zwfs, phasemask , savefigName= None ):
    print( f'input savefigName = {savefigName} <- this is where output images will be saved.\nNo plots created if savefigName = None')
    exit = 0
    while not exit:
        input_str = input('enter "e" to exit, else input relative movement in um: x,y')
        if input_str == 'e':
            exit = 1
        else:
            try:
                xy = input_str.split(',') 
                x = float( xy[0]  )
                y = float( xy[1] )
                phasemask.move_relative( [x,y] )
                time.sleep( 0.5 )
                img = np.mean(zwfs.get_some_frames(number_of_frames = 20, apply_manual_reduction = True ) , axis=0 )
                if savefigName != None:
                    plt.figure()
                    plt.imshow( img )
                    plt.savefig( savefigName )
            except:
                print('incorrect input. Try input "1,1" as an example, or "e" to exit')
                
        



if __name__=="__main__":

    print( ' THIS TAKES SEVERAL MINUTES TO RUN. WHAT WE ARE DOING IS: \n\
     - connect to motors, DM and camera. Set up detector with darks, bad pixel mask etc.\n \
     - iterate through all phase masks on beam 3 (in Sydney) and update phasemask positions and save them.\n \
     - This should only serve as example. must be called from asgard_alignment folder\n \
    DEVELOPED IN SYDNEY WITH MOTORS ONLY ON BEAM 3 - UPDATE ACCORDINGLY \n \
    DOING AUTOMATED SEARCH OVER LIMITED RADIUS, IF RESULTS ARE POOR - ADJUST SEARCH RADIUS / GRID.\
    ')
    import numpy as np
    import glob 
    from astropy.io import fits
    import time
    import os 
    import matplotlib.pyplot as plt 
    import importlib
    import sys
    import datetime
    sys.path.append('pyBaldr/' )  
    sys.path.insert(1, '/opt/FirstLightImaging/FliSdk/Python/demo/')
    sys.path.insert(1,'/opt/Boston Micromachines/lib/Python3/site-packages/')

    from pyBaldr import utilities as util
    from pyBaldr import ZWFS


    import bmc
    import FliSdk_V2
    from zaber_motion.ascii import Connection
    from asgard_alignment.ZaberMotor import BaldrPhaseMask, LAC10AT4A,  BifrostDichroic, SourceSelection

    trouble_shooting_dict = {
        #format:
        'short error key' :
        {
            'error string' : 'longer string descibing error',
            'fix': 'how to fix it '
        },
        'SerialPortBusyException' : 
        {
            'error string':"SerialPortBusyException: SerialPortBusyException: Cannot open serial port: Port is likely already opened by another application.",
            'fix':"You can check if any processes are using the serial port with the following command: lsof /dev/*name* (e.g. name=ttyUSB0).\nIf you found a process using the port from the previous step, you can terminate it with: sudo kill -9 <PID> "
        }

        }


    # ====== hardware variables
    beam = 3 # in sydney only using beam 3.. need to update this code when we have other beams working with phasemask etc.
    phasemask_name = 'J3' # initial one , we iterate through them 
    phasemask_OUT_offset = [1000,1000]  # relative offset (um) to take phasemask out of beam
    BFO_pos = 4000 # um (absolute position of detector imgaging lens) 
    dichroic_name = "J"
    source_name = 'SBB'
    DM_serial_number = '17DW019#122' # Syd = '17DW019#122', ANU = '17DW019#053'



    def print_current_state():
        print(f'source motor: \n   {source_selection.device}')
        print(f'    -available sources: {source_selection.sources}')
        print(f'    -current position: {source_selection.current_position}')
        for d in dichroics:
            print(f'dichroic motor:\n   {d.device}')
            print(f'    -available dichroic positions: {d.dichroics}' )
            print(f'    -current position: {d.current_dichroic}')
        print('availabel phasemask positions: ', )
        print(f' phasemask motors: \n   {phasemask.motors}')
        print(f'    -available positions:')
        for l, p in phasemask.phase_positions.items():
            print(f'   {l, p}')
        print(f'    -current position: {phasemask.get_position()}um')
        print(f'focus motor:\n   {focus_motor}')
        print(f'    -current position: {focus_motor.get_position()}um')


    def exit_all():
        # close things 
        try:
            con.close() #"192.168.1.111"
        except:
            print('no "con" to close')
        try:
            connection.close() # "/dev/ttyUSB0"
        except:
            print('no "connection" to close')
        try:
            zwfs.exit_dm() # DM 
        except:
            print('no DM to close')
        try:
            zwfs.exit_camera() #camera
        except:
            print('no camera to close')


    # timestamp
    tstamp = datetime.datetime.now().strftime("%d-%m-%YT%H.%M.%S")

    fig_path = 'tmp/' #'/home/baldr/Documents/baldr/ANU_demo_scripts/BALDR/figures/' 
    data_path = 'tmp/' #'/home/baldr/Documents/baldr/ANU_demo_scripts/BALDR/data/' 

    # ======  set up source 

    # start with source out !

    # ======  set up dichroic 

    # do manually (COM3 communication issue)

    #  ConnectionFailedException: ConnectionFailedException: Cannot open serial port: no such file or directory

    connection =  Connection.open_serial_port("/dev/ttyUSB0")
    connection.enable_alerts()

    device_list = connection.detect_devices()
    print("Found {} devices".format(len(device_list)))

    dichroics = []
    source_selection = None
    for dev in device_list:
        if dev.name == "X-LSM150A-SE03":
            dichroics.append(BifrostDichroic(dev))
        elif dev.name == "X-LHM100A-SE03":
            source_selection = SourceSelection(dev)
    print(f"Found {len(dichroics)} dichroics")
    if source_selection is not None:
        print("Found source selection")

    for dichroic in dichroics:
        dichroic.set_dichroic("J")

    while dichroics[0].get_dichroic() != "J":
        pass

    # ====== set up phasemask
    con = Connection.open_tcp("192.168.1.111")
    print("Found {} devices".format(len(con.detect_devices())))
    x_axis = con.get_device(1).get_axis(1)
    y_axis = con.get_device(1).get_axis(3)

    # get most recent positions file
    maskpos_files = glob.glob( f"phase_positions_beam_{beam}*.json")
    latest_maskpos_file = max(maskpos_files, key=os.path.getctime)
    phasemask = BaldrPhaseMask(
        LAC10AT4A(x_axis), LAC10AT4A(y_axis), latest_maskpos_file 
    )
    """ 
    # e.g: to update position and write to file 
    phasemask.move_absolute( [3346, 1205])
    phasemask.update_mask_position( 'J3' )
    phasemask.write_current_mask_positions() 
    """

    # ====== set up focus 
    focus_axis = con.get_device(1).get_axis(2)
    focus_motor = LAC10AT4A(focus_axis)


    # print out motors we have 

    print_current_state()
    # ====== Set up and calibrate 

    debug = True # plot some intermediate results 

    # take out source to calibate 
    source_selection.set_source(  'none' )
    time.sleep(1)
    focus_motor.move_absolute( BFO_pos )
    time.sleep(1)
    phasemask.move_to_mask(phasemask_name) 
    time.sleep(1)
    dichroic.set_dichroic("J")
    time.sleep(1)


    pupil_crop_region = [204,268,125, 187] #[None, None, None, None] #[0, 192, 0, 192] 

    #init our ZWFS (object that interacts with camera and DM) (old path = home/baldr/Documents/baldr/ANU_demo_scripts/BALDR/)
    zwfs = ZWFS.ZWFS(DM_serial_number=DM_serial_number, cameraIndex=0, DMshapes_path = 'DMShapes/', pupil_crop_region=pupil_crop_region ) 

    # the sydney BMC multi-3.5 calibrated flat seems shit! Try with just a 

    zwfs.set_camera_dit( 0.001 );time.sleep(0.2)
    zwfs.set_camera_fps( 200 );time.sleep(0.2)
    zwfs.set_sensitivity('high');time.sleep(0.2)
    zwfs.enable_frame_tag(tag = True);time.sleep(0.2)
    zwfs.bias_off();time.sleep(0.2)
    zwfs.flat_off();time.sleep(0.2)

    zwfs.dm_shapes['flat_dm'] = 0.5 * np.ones(140)

    zwfs.start_camera()

    # !!!! TAKE OUT SOURCE !!!! 
    # at sydney move 01 X-LSM150A-SE03 to 133.07mm
    zwfs.build_manual_dark()

    # get our bad pixels 
    bad_pixels = zwfs.get_bad_pixel_indicies( no_frames = 1000, std_threshold = 50 , flatten=False)

    # update zwfs bad pixel mask and flattened pixel values 
    zwfs.build_bad_pixel_mask( bad_pixels , set_bad_pixels_to = 0 )

    # move source back in 
    source_selection.set_source(  source_name )
    time.sleep(2)

    # quick check that dark subtraction works and we have signal
    I0 = zwfs.get_image( apply_manual_reduction  = True)
    plt.figure(); plt.title('test image \nwith dark subtraction \nand bad pixel mask'); plt.imshow( I0 ); plt.colorbar()
    plt.savefig( fig_path + 'delme.png')
    plt.close()

    print_current_state()

    # spiral parameters for search
    phasemask_diameter = 50 # um <- have to ensure grid is at this resolution 
    search_radius = 100  # search radius for spiral search (um)
    dtheta = np.pi / 20  # angular increment (rad) 
    iterations_per_circle = 2*np.pi / dtheta
    dr = phasemask_diameter / iterations_per_circle # cover 1 phasemask diameter per circle

    # Start spiral search and fine centering
    fine_tune_threshold=3
    savefigName = fig_path + 'delme.png'


    for phasemask_name in phasemask.phase_positions.keys():

        initial_pos= phasemask.phase_positions[phasemask_name] # starting position of phase mask

        # move off phase mask, its good to make sure zwfs object has dark, bad pixel map etc first to see better
        #phasemask.move_absolute( initial_pos ) 
        phasemask.move_relative( [1000,1000] )  # 1mm in each axis
        time.sleep(1.2)
        reference_img =  np.mean(zwfs.get_some_frames(number_of_frames = 10, apply_manual_reduction = True ) , axis=0 ) # Capture reference image when misaligned
        phasemask.move_absolute( initial_pos )  # move back to initial_position 
        time.sleep(1.2)

        centered_position = spiral_search_and_center(
            zwfs, phasemask, phasemask_name, search_radius, dr, dtheta, reference_img, \
            fine_tune_threshold=fine_tune_threshold, savefigName=None, usr_input=False \
        )

    # write them all in a timestamped json file 
    phasemask.write_current_mask_positions() 

    exit_all()



#--------- IGNORE BELOW
# Some old stuff converting x,y spiral search to polar, interpolating and plotting metric vs radii vs angular bins
# this is useful in noisy environments with fine radial search to make sure metric has global optimum and isnt noise

# # Assuming you have these arrays:
# # positions: list of [x, y] positions
# # diffs: corresponding diff values for each [x, y] position

# def convert_to_polar(x, y, center=(0, 0)):
#     """ Convert (x, y) to polar coordinates (radius, angle) relative to a center. """
#     x_centered, y_centered = x - center[0], y - center[1]
#     radius = np.sqrt(x_centered**2 + y_centered**2)
#     angle = np.arctan2(y_centered, x_centered)
#     return radius, angle

# def bin_by_angle(radius, angle, diff_list, nbins):
#     """ Bin diffs by angle into nbins. """
#     bins = np.linspace(-np.pi, np.pi, nbins + 1)
#     digitized = np.digitize(angle, bins)

#     # Adjust the binning so that values in the last bin are included
#     digitized = np.clip(digitized, 1, nbins)
    
#     binned_data = {i: [] for i in range(1, nbins + 1)}

#     for r, a, d, bin_id in zip(radius, angle, diff_list, digitized):
#         binned_data[bin_id].append((r, d))
    
#     return binned_data
# import numpy as np
# import scipy.interpolate as interp

# def interpolate_max_in_bins(binned_data):
#     """ Interpolate diff vs radius and find max in each angle bin. """
#     max_radii = []
#     max_angles = []
#     interpolated_data = {}

#     for bin_id, data in binned_data.items():
#         if len(data) < 2:  # Ensure at least 2 points for interpolation
#             continue

#         data = sorted(data, key=lambda x: x[0])  # Sort by radius
#         radii, diffs = zip(*data)
        
#         if len(radii) > 3:  # Default spline degree is 3
#             spline = interp.UnivariateSpline(radii, diffs, s=0)
#             fine_radii = np.linspace(min(radii), max(radii), 1000)
#             fine_diffs = spline(fine_radii)
#             max_idx = np.argmax(fine_diffs)
#             max_radii.append(fine_radii[max_idx])
#             max_angles.append(np.mean([2 * np.pi / len(binned_data) * (bin_id - 1)]))
#             interpolated_data[bin_id] = (fine_radii, fine_diffs)
#         else:
#             max_radii.append(max(radii, key=lambda r: diffs[radii.index(r)]))
#             max_angles.append(np.mean([2 * np.pi / len(binned_data) * (bin_id - 1)]))
#             interpolated_data[bin_id] = (radii, diffs)  # No fine interpolation for sparse data

#     return max_radii, max_angles, interpolated_data


# def polar_to_cartesian(radius, angle, center=(0, 0)):
#     """ Convert polar coordinates (radius, angle) to cartesian (x, y). """
#     x = center[0] + radius * np.cos(angle)
#     y = center[1] + radius * np.sin(angle)
#     return x, y

# 