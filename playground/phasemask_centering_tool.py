import numpy as np
import time
from scipy.ndimage import center_of_mass
import matplotlib.pyplot as plt 

def compute_image_difference(img1, img2):
    # normalize both images first
    img1 = img1.copy() /np.sum(img1)
    img2 = img2.copy() /np.sum(img2)
    return np.sum(np.abs(img1 - img2))

def is_symmetric(image, threshold=0.1):
    center = center_of_mass(image)
    y_center, x_center = np.array(image.shape) // 2
    
    # Check symmetry by comparing opposite quadrants
    q1 = image[:y_center, :x_center]
    q2 = np.flip(image[y_center:, :x_center], axis=0)
    q3 = np.flip(image[:y_center, x_center:], axis=1)
    q4 = np.flip(image[y_center:, x_center:], axis=(0, 1))

    # Calculate the differences
    diff1 = np.sum(np.abs(q1 - q4))
    diff2 = np.sum(np.abs(q2 - q3))
    
    print(f'in is_summetric loop :\n  (diff1 + diff2) = {(diff1 + diff2)}, threshold={threshold}')
    return (diff1 + diff2) < threshold

def spiral_search_and_center(zwfs, phasemask, initial_pos, search_radius, dr, dtheta, reference_img, fine_tune_threshold=3, plot=True):
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
        img = zwfs.get_image()

        diff = compute_image_difference(img, reference_img)
        diff_list.append( diff )
        x_pos_list.append( x_pos )
        y_pos_list.append( y_pos )
        print(f'img diff = {diff}, fine_tune_threshold={fine_tune_threshold}')

        """ # ----> should do this as a second stage after spiral search, do smarter centering here 
        if diff > fine_tune_threshold:
            # Fine-tuning phase mask position
            step_size = dr / 2  # start with a smaller step size
            while step_size > 0.1:  # convergence criterion
                phasemask.move_absolute([x_pos, y_pos])
                time.sleep( sleep_time)
                img = zwfs.get_image(apply_manual_reduction=True)
                
                if is_symmetric(img):
                    print(f"Phase mask centered at: ({x_pos}, {y_pos})")
                    return (x_pos, y_pos)
                
                # Adjust position
                x_pos += step_size * np.random.choice([-1, 1])
                y_pos += step_size * np.random.choice([-1, 1])
                step_size *= 0.5
        """    

        # Update for next spiral step
        angle += dtheta
        radius += dr

        if plot: 
            fig_path = '/home/heimdallr/Documents/asgard-alignment/tmp/'
            if np.mod( plot_cnt , 5) == 0:

                norm = plt.Normalize(0 , fine_tune_threshold)

                fig,ax = plt.subplots( 1,3 ,figsize=(20,6))
                ax[0].set_title( 'image' )
                ax[1].set_title( f'search positions\nx:{phasemask.motors["x"]}\ny:{phasemask.motors["y"]}' )
                ax[2].set_title( 'search metric' )

                ax[0].imshow( img )
                ax[1].plot( [x_pos,y_pos] , 'x', color='r', label='current pos')
                ax[1].plot( [initial_pos[0],initial_pos[1]] , 'o', color='k', label='current pos')
                tmp_diff_list = np.array(diff_list)
                tmp_diff_list[tmp_diff_list < 1e-5 ] = 0.1 # very small values got to finite value (errors whern 0!)
                tmp_diff_list[tmp_diff_list < 1e-5 ] = 0.1
                ax[1].scatter( x_pos_list, y_pos_list , s =  np.exp( 400 * np.array(tmp_diff_list) / fine_tune_threshold )  ,\
                 marker='o', c=diff_list, cmap='viridis', norm=norm)
                ax[1].set_xlim( [initial_pos[0] - search_radius,  initial_pos[0] + search_radius] )
                ax[1].set_ylim( [initial_pos[1] - search_radius,  initial_pos[1] + search_radius] )
                ax[1].legend() 
                ax[2].plot( diff_list )
                ax[2].set_xlim( [0, search_radius/dr] )

                ax[0].axis('off')
                ax[1].set_ylabel( 'y pos' )
                ax[1].set_xlabel( 'x pos' )
                ax[2].set_ylabel( r'$\Sigma|img - img_off|$' )
                ax[2].set_xlabel( 'iteration' )
                plt.savefig( fig_path + 'delme.png')
                plt.cla()
            plot_cnt += 1
    print("Spiral search complete, centering failed.")
    return None


# Initial conditions
if __name__=="main":
    # This should only serve as example . must be called from asgard_alignment folder

    import sys 
    sys.path.append('pyBaldr/' )  
    from pyBaldr import ZWFS
    from zaber_motion.ascii import Connection
    from asgard_alignment.ZaberMotor import BaldrPhaseMask, LAC10AT4A


    # set up phase mask 
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

    # set up camera etc 
    pupil_crop_region = [204,268,125, 187] #[None, None, None, None] #[0, 192, 0, 192] 

    DM_serial_number = '17DW019#122' # Syd = '17DW019#122', ANU = '17DW019#053'

    #init our ZWFS (object that interacts with camera and DM) (old path = home/baldr/Documents/baldr/ANU_demo_scripts/BALDR/)
    zwfs = ZWFS.ZWFS(DM_serial_number=DM_serial_number, cameraIndex=0, DMshapes_path = 'DMShapes/', pupil_crop_region=pupil_crop_region ) 

    zwfs.start_camera()

    zwfs.build_manual_dark()
    # get our bad pixels 
    bad_pixels = zwfs.get_bad_pixel_indicies( no_frames = 1000, std_threshold = 50 , flatten=False)
    # update zwfs bad pixel mask and flattened pixel values 
    zwfs.build_bad_pixel_mask( bad_pixels , set_bad_pixels_to = 0 )


    initial_position = phasemask.phase_positions["J2"] # starting position of phase mask
    phasemask_diameter = 50 # um <- have to ensure grid is at this resolution 
    search_radius = 100  # search radius for spiral search (um)
    delta_theta = np.pi / 10  # angular increment (rad) 
    iterations_per_circle = 2*np.pi / delta_theta
    delta_radius = phasemask_diameter / iterations_per_circle # cover 1 phasemask diameter per circle

    # move off phase mask, its good to make sure zwfs object has dark, bad pixel map etc first to see better
    phasemask.move_absolute( initial_position ) 
    phasemask.move_relative( [1000,1000] )  # 1mm in each axis
    time.sleep(0.5)
    reference_image = zwfs.get_image( apply_manual_reduction=True)  # Capture reference image when misaligned
    phasemask.move_absolute( initial_position )  # move back to initial_position 
    time.sleep(0.5)
    # Start spiral search and fine centering
    centered_position = spiral_search_and_center(
        zwfs, phasemask, initial_position, search_radius, delta_radius, delta_theta, reference_image, fine_tune_threshold=3, plot=True
    )

    if centered_position:
        print(f"Centered position: {centered_position}")
    else:
        print("Failed to center the phase mask.")
