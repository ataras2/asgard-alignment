import time 
import numpy as np 
import matplotlib.pyplot as plt
from scipy import signal
import itertools
import aotools
from matplotlib import colors

from . import utilities as util 
#from . import hardware 

fig_path = 'data/' #'/home/baldr/Documents/baldr/ANU_demo_scripts/BALDR/figures/' 

class pupil_controller_1():
    """
    measures and controls the ZWFS pupil
    """
   
    def __init__(self, config_file = None):
       
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
            self.config['source'] = 1
            self.config['pupil_control_motor'] = 'XXX'
            
            self.ctrl_parameters = {} # empty dictionary cause we have none
            

    def measure_dm_center_offset(self, zwfs, debug=True): 
        
        zwfs.states['busy'] = 1

        # r1,r2,c1,c2 = zwfs.pupil_crop_region <- this cropping gets done automatically in zwfs.get_image()

        # make sure phase mask is in 

        # check if camera is running

        amp = 0.15
        #push DM corner actuators & get image 
        zwfs.send_cmd(zwfs.dm_shapes['flat_dm'] + amp * zwfs.dm_shapes['four_torres'] ) 
        time.sleep(0.003)
        img_push = zwfs.get_image().astype(int) # get_image returns uint16 which cannot be negative
        #pull DM corner actuators & get image 
        zwfs.send_cmd(zwfs.dm_shapes['flat_dm'] - amp * zwfs.dm_shapes['four_torres'] ) 
        time.sleep(0.003)
        img_pull = zwfs.get_image().astype(int) # get_image returns uint16 which cannot be negative
        
        delta_img = abs( img_push - img_pull ) # [r1:r2,c1:c2] #zwfs.get_image() automatically crops here

        #define symetric coordinates in image (i.e. origin is at image center)  
        y = np.arange( -delta_img.shape[0]//2, delta_img.shape[0]//2) # y is row
        x = np.arange( -delta_img.shape[1]//2, delta_img.shape[1]//2) # x is col
        #x,y position weights
        w_x = np.sum( delta_img, axis = 0 ) #sum along columns 
        w_y = np.sum( delta_img, axis = 1 ) #sum along rows 
        #x,y errors
        e_x = np.mean( x * w_x ) / np.mean(w_x) 
        e_y = np.mean( y * w_y ) / np.mean(w_y) 

        if debug:
            plt.figure()
            #plt.pcolormesh(y, x, delta_img ) 
            plt.imshow( delta_img , extent=[y[0],y[-1],x[0],x[-1]]) # extent is [horizontal min, horizontal max, vertical min, vertical max]
            plt.xlabel('x pixels',fontsize=15)
            plt.ylabel('y pixels',fontsize=15)
            plt.gca().tick_params(labelsize=15)
            plt.arrow( 0, 0, e_x, e_y, color='w', head_width=1.2 ,width=0.3, label='error vector')
            plt.legend(fontsize=15) 
            plt.tight_layout()
            #plt.savefig(figure_path + 'process_1.2_center_source_DM.png',dpi=300) 
            plt.show()

        return( e_x, e_y )      


    def set_pupil_reference_pixels(self ):
        print('to do')

    def set_pupil_filter( self ):
        print('to do')  

        

def analyse_pupil_openloop( zwfs, debug = True, return_report = True, symmetric_pupil=True):
    """
    
    DM_torre_actuator_seperation is hard coded and should ALWAYS match the real seperation on  zwfs.dm_shapes['four_torres_2'] 
    """

 
    report = {} # initialize empty dictionary for report

    #rows, columns to crop
    r1, r2, c1, c2 = zwfs.pupil_crop_region

    # check if camera is running
    # Shape for finding DM center in pixel space and DM coordinates in pixel space.
    calibration_shape = 'four_torres_2' # this pokes four inner corners of the DM 
   
    # putting the calibration command in 2D
    torres_shape_2D = util.get_DM_command_in_2D(zwfs.dm_shapes[ calibration_shape ])
    # get how many actuator seperations between the poking corners
    DM_torre_actuator_seperation = np.diff( np.where( np.nansum( torres_shape_2D, axis=0))[0] )[0]

    #================================
    #============= Measure pupil center 
    
    # make sure phase mask is OUT !!!! 
    #hardware.set_phasemask( phasemask = 'out' ) # no motors to implement this on yet, so does nothing 


    zwfs.send_cmd(zwfs.dm_shapes['flat_dm'])


    print( '2*TIP MODE TO MOVE PHASE MASK OUT OF BEAM')

    fourier_basis = util.construct_command_basis( basis='fourier', number_of_modes = 5, Nx_act_DM = 12, Nx_act_basis = 12, act_offset=(0,0), without_piston=True)

    zwfs.dm.send_data(  zwfs.dm_shapes['flat_dm'] + 2 * fourier_basis[:,0] )
    time.sleep( 0.1 )
    
    # simple idea: we are clearly going to have 2 modes in distribution of pixel intensity, one centered around the mean pupil intensity where illuminated, and another centered around the "out of pupil" region which will be detector noise / scattered light. np.histogram in default setup automatically calculates bins that incoorporate the range and resolution of data. Take the median frequency it returns (which is an intensity value) and set this as the pupil intensity threshold filter. This should be roughly between the two distributions.

    X, Y = np.meshgrid( zwfs.col_coords, zwfs.row_coords )

    imglist = []
    for _ in range(10):
       imglist.append( zwfs.get_image().astype(int) )  #[r1: r2, c1: c2] <- we made zwfs automatically crop 
       time.sleep(0.1)
    img = np.median( imglist, axis=0)

    N0 = img # set this as our reference FPM OUT image for now 

    ## General Assymetric pupil
    density, intensity_bins  = np.histogram( img.reshape(-1) )
    intensity_threshold =  np.median( intensity_bins ) - 1.*np.std(intensity_bins) #np.median( intensity_edges ) 

    pupil_filter = img.reshape(-1) > intensity_threshold
    
    # for now just essentially copy pupil_filter with higher threshold
    # TO REVIEW 
    pupil_filter_tight = img.reshape(-1) > 1.1 * intensity_threshold
    

    x_pupil_center, y_pupil_center = np.mean(X.reshape(-1)[pupil_filter]), np.mean(Y.reshape(-1)[pupil_filter])

    #plt.figure(); plt.imshow( pupil_filter.reshape(img.shape) ); plt.show()


    if symmetric_pupil: # we impose symmetry constraints  
        ## symmetric pupil
        collapsed_pupil_x = np.sum(img, axis=0)
        collapsed_pupil_y = np.sum(img, axis=1)
        #plt.figure(); plt.plot( collapsed_pupil_x); plt.show()
        density, intensity_bins  = np.histogram( list(collapsed_pupil_x) + list(collapsed_pupil_y)  )
        intensity_threshold =  np.median( intensity_bins ) - 1*np.std(intensity_bins) 
        diam_filter_x = collapsed_pupil_x > intensity_threshold
        diam_filter_y = collapsed_pupil_y > intensity_threshold

        pupil_diam_x = np.sum( diam_filter_x )
        pupil_diam_y = np.sum( diam_filter_y )
        pupil_diam = 0.5*(pupil_diam_x+pupil_diam_y) 

        #  warning! - circle_centre=(x_pupil_center, y_pupil_center) definitions of x,y may be inverted...CHECK THIS.
        inside_pup = util.circle(radius=pupil_diam//2, size=img.shape, circle_centre=(x_pupil_center, y_pupil_center), origin='middle')

        
        # do a tight one to really only capture inner regions and not edges 
        inside_pup_tight = util.circle(radius=int( 0.5*pupil_diam//2), size=img.shape, circle_centre=(x_pupil_center, y_pupil_center), origin='middle')

        #debug
        #fig,ax = plt.subplots(1,2)
        #ax[0].imshow( inside_pup.reshape(img.shape) ); ax[1].imshow( img); plt.show()
        #plt.figure();plt.imshow( inside_pup.reshape(img.shape), alpha=0.5); plt.imshow( img,alpha=0.5); plt.show()

        # do a tight one to really only capture inner regions and not edges 
        pupil_filter_tight = ( inside_pup_tight > 0 ).reshape(-1)

        pupil_filter  = ( inside_pup > 0 ).reshape(-1)

        if hasattr(zwfs, 'bad_pixel_filter'): 
            # bad_pixel_filter is 1 if bad pixel, zero otherwise 
            pupil_filter *= ~zwfs.bad_pixel_filter
            pupil_filter_tight *= ~zwfs.bad_pixel_filter

    
    pupil_pixels =  np.where( pupil_filter )[0]

    #================================
    #============= Add to report card 
    report['pupil_pixel_filter'] = pupil_filter # within the cropped img frame based on zwfs.pupil_crop_region
    report['pupil_pixel_filter_tight'] = pupil_filter_tight # within the cropped img frame based on zwfs.pupil_crop_region

    report['pupil_pixels'] = pupil_pixels # within the cropped img frame based on zwfs.pupil_crop_region
    report['pupil_center_ref_pixels'] = ( x_pupil_center, y_pupil_center ) # within the cropped img frame based on zwfs.pupil_crop_region

    if debug: 
        fig,ax = plt.subplots(2,1,figsize=(5,10))
        ax[0].pcolormesh( zwfs.col_coords, zwfs.row_coords,   img) 
        ax[0].set_title('measured pupil')
        ax[1].pcolormesh( zwfs.col_coords, zwfs.row_coords,   pupil_filter.reshape(img.shape) ) 
        ax[1].set_title('derived pupil filter')
        for axx in ax.reshape(-1):
            axx.axvline(x_pupil_center,color='r',label='measured center')
            axx.axhline(y_pupil_center,color='r')
        #plt.savefig(fig_path + '1.1_centers_FPM-OFF_internal_source.png',bbox_inches='tight', dpi=300)
        plt.legend() 

    #================================
    #============= Measure DM center 

    # make sure phase mask is IN !!!! 
    #hardware.set_phasemask( phasemask = 'posX' ) # no motors to implement this on yet, so does nothing 
    print( 'FLATTEN DM TO MOVE PHASE MASK BACK IN BEAM')
    zwfs.send_cmd(zwfs.dm_shapes['flat_dm'])
    time.sleep( 0.1 )

    #_ = input('MANUALLY MOVE PHASE MASK INTO BEAM, PRESS ENTER TO BEGIN' )
    #util.watch_camera(zwfs, frames_to_watch = 50, time_between_frames=0.05) 

    # now we get a reference I0 intensity 
    imglist = []
    for _ in range(10):
       imglist.append( zwfs.get_image().astype(int) )  #[r1: r2, c1: c2] <- we made zwfs automatically crop 
       time.sleep(0.1)
    img = np.median( imglist, axis=0)
    I0 = img # set this as reference FPM IN intensity 


    #now add shape on DM to infer centers
    amp = 0.1
    delta_img_list = [] # hold our images, which we will take median of 
    for _ in range(10): # get median of 10 images 
        #push DM corner actuators & get image 
        zwfs.send_cmd(zwfs.dm_shapes['flat_dm'] + amp * zwfs.dm_shapes[calibration_shape] ) 
        time.sleep(0.003)
        img_push = zwfs.get_image().astype(int) # get_image returns uint16 which cannot be negative
        #pull DM corner actuators & get image 
        zwfs.send_cmd(zwfs.dm_shapes['flat_dm'] - amp * zwfs.dm_shapes[calibration_shape] ) 
        time.sleep(0.003)
        img_pull = zwfs.get_image().astype(int) # get_image returns uint16 which cannot be negative
        
        delta_img_list.append( abs( img_push - img_pull ) ) # zwfs.get_image automatiicaly crops so this is DEFINED in the crop region

    zwfs.send_cmd(zwfs.dm_shapes['flat_dm']) # flat DM 
    delta_img = np.median( delta_img_list, axis = 0 ) #get median of our modulation images 


    # define our cropped regions coordinates ( maybe this can be done with initiation of crop region attribute - move this to zwfs object since it is ALWAYS asscoiated with images taken from this object)
    y = zwfs.row_coords  #rows
    x = zwfs.col_coords  #columns

    #image quadrants coordinate indicies (row1, row2, col1, col2)
    q_11 = 0,  delta_img.shape[0]//2,  0,  delta_img.shape[1]//2
    q_12 = 0,  delta_img.shape[0]//2, delta_img.shape[1]//2,  None
    q_21 = delta_img.shape[0]//2, None, delta_img.shape[1]//2,  None
    q_22 = delta_img.shape[0]//2,  None,  0, delta_img.shape[1]//2


    ep = [] # to hold x,y of each quadrants peak pixel which will be our line end points
    for q in [q_11, q_12, q_21, q_22]: #[Q11, Q12, Q21, Q22]:
        xq = x[ q[2]:q[3] ] # cols
        yq = y[ q[0]:q[1] ] # rows
        d = delta_img[  q[0]:q[1], q[2]:q[3] ]
        x_peak = xq[ np.argmax( np.sum( d  , axis = 0) ) ] # sum along y (row) axis and find x where peak
        y_peak = yq[ np.argmax( np.sum( d  , axis = 1) ) ] # sum along x (col) axis and find y where peak
        #plt.figure()
        #plt.imshow( d )

        ep.append( (x_peak, y_peak) )  #(col, row)
    
    # define our line end points 
    line1 = (ep[0], ep[2]) # top left, bottom right 
    line2 = (ep[1], ep[3]) # top right, bottom left

    # find intersection to get centerpoint of DM in the defined cropped region coordinates 
    x_dm_center, y_dm_center = util.line_intersection(line1, line2)

    # absolute difference in pixel space between the median of (top left col - top right col, bottom left col - bottom right col, top left row - bottom left row, bottom left row - bottom right row )
    med_pixel_distance_between_torres = np.median( [ abs(ep[0][0] - ep[1][0]), abs(ep[3][0] - ep[1][0]), abs(ep[0][1] - ep[1][1]), abs(ep[3][1] - ep[1][1]) ] )

    dx_dm =  med_pixel_distance_between_torres / DM_torre_actuator_seperation # how many pixels on camera per DM actuator 

    # DM coordinates in pixel space (Note DM is 12x12 grid which is hard coded here)
    x_dm_coord = np.linspace(x_dm_center - 6* dx_dm , x_dm_center + 6 * dx_dm, 12 )
    y_dm_coord = np.linspace(y_dm_center - 6 * dx_dm , x_dm_center + 6 * dx_dm, 12 )


    #print('CENTERS=', x_dm_center, y_dm_center) 

    if debug:
        # DM center calculation
        plt.figure()
        #plt.pcolormesh( x, y, delta_img)
        plt.imshow( delta_img, extent = [x[0],x[-1],y[0],y[-1]] )
        plt.colorbar(label='[adu]')
        xx1 = [ep[0] for ep in line1]
        yy1 = [ep[1] for ep in line1]
        xx2 = [ep[0] for ep in line2]
        yy2 = [ep[1] for ep in line2] 
        plt.plot( xx1, yy1 ,linestyle = '-',color='r',lw=3 ) 
        plt.plot( xx2, yy2 ,linestyle = '-',color='r',lw=3 )  
        plt.xlabel('x [pixels]',fontsize=15)
        plt.ylabel('y [pixels]',fontsize=15)
        plt.gca().tick_params(labelsize=15) 
        plt.tight_layout()
        #plt.savefig(fig_path + 'process_1.3_analyse_pupil_DM_center.png',bbox_inches='tight', dpi=300)

        # DM imprint in pixel space 
        plt.figure() 
        plt.title('DM torre imprint in pixel space - check overlap!')
        plt.imshow( delta_img, extent = [x[0],x[-1],y[0],y[-1]] )
        plt.imshow( torres_shape_2D, extent = [x_dm_coord[0],x_dm_coord[-1],y_dm_coord[0],y_dm_coord[-1]] , alpha =0.5)
        plt.xlabel('x [pixels]',fontsize=15)
        plt.ylabel('y [pixels]',fontsize=15)
        plt.gca().tick_params(labelsize=15) 
        plt.tight_layout()
        plt.show() 



    #================================
    #============= Add to report card 

    report['dm_center_ref_pixels'] = x_dm_center, y_dm_center
    
    report['dm_x_coords_in_pixels'] = x_dm_coord
    report['dm_y_coords_in_pixels'] = y_dm_coord

    report['N0'] = N0 
    report['I0'] = I0 
    #================================
    #============= Get secondary obstruction and outside pupil filters


    #===  secondary obstruction pupil filter
    # use the tight pupil filter so we don't capture outside edges in our filtering
    _, pupil_intensity_bins  = np.histogram( N0.reshape(-1)[pupil_filter_tight] )
    intensity_threshold =  np.median( pupil_intensity_bins ) - 1.*np.std( pupil_intensity_bins ) #np.median( intensity_edges ) 

    secondary_pupil_filter_subspace = N0.reshape(-1)[pupil_filter_tight] < intensity_threshold
    #we have to insert this back into main image grid now (since we filtered on the inside pupil subspace)
    secondary_pupil_filter = np.zeros(N0.shape).astype(bool)
    secondary_pupil_filter.reshape(-1)[pupil_filter_tight] = secondary_pupil_filter_subspace #Note: this is still be 2D array 



    x_sec_pupil_center, y_sec_pupil_center = np.mean(X.reshape(-1)[secondary_pupil_filter.reshape(-1)]), np.mean(Y.reshape(-1)[secondary_pupil_filter.reshape(-1)])

    if symmetric_pupil: 
        # we force secondary_pupil_filter to be circle 
        secondary_diam = np.sum( 0 < np.sum(secondary_pupil_filter,axis=0) )  

        secondary_pupil_filter = util.circle(radius=secondary_diam//2, size=N0.shape, circle_centre=(x_sec_pupil_center, y_sec_pupil_center), origin='middle').astype(bool)


    secondary_pupil_filter = secondary_pupil_filter.reshape(-1) #make sure its flattened

    secondary_pupil_pixels =  np.where( secondary_pupil_filter )[0]

    #plt.figure();plt.imshow( N0 );plt.imshow( secondary_pupil_filter.reshape(N0.shape), alpha=0.2, label='secondary_obstruction_filter');plt.legend();  plt.savefig(fig_path + 'process_1.3_secondary_filter.png',bbox_inches='tight', dpi=300); plt.show()


    #===  outside filter
    outside_pupil_filter  = ~pupil_filter 
    outside_pupil_pixels =  np.where( outside_pupil_filter )[0]

    #plt.figure();plt.imshow( outside_pupil_filter.reshape(N0.shape), alpha=0.5); plt.imshow( N0,alpha=0.5); plt.show()

    #===  reference field peak  
    #get indicies where I0-N0 is maximum - this corresponds to peak of M^2 = |psi_r|^2 reference field. This can be used for fitting 'b' along with outside pupil intensities. Also a health check that this matches the center of the found secondary obstruction.. If not this could indicate mis-alignment of the focal plane phase mask or bad aberrations in the system.    
    i_sec, j_sec = np.unravel_index( np.argmax( abs(I0-N0).reshape(-1) ), I0.shape )
 
    # careful here, we don't know if we should take indicies above or below peak center etc..
    # we should take 3x3 grid and 2D convolve with ones to see where peak is and take those points 
    subgrid = abs(I0 - N0)[ i_sec - 1 : i_sec + 2, j_sec - 1 : j_sec + 2 ] # isec,jsec at center of subgrid (index = 1,1) 

    conv = signal.convolve2d(subgrid, np.ones([2,2]), mode='valid')
    di, dj = np.unravel_index( np.argmax(conv), conv.shape )
    si_rows = [i_sec + di, i_sec + di - 1]
    si_cols = [j_sec + dj , j_sec + dj - 1 ] 

    peak_ref_field_indicies = list(itertools.product(si_rows, si_cols))

    ref_field_peak_pupil_filter = np.zeros(N0.shape).astype(bool)
    for i,j in peak_ref_field_indicies :
        ref_field_peak_pupil_filter[i,j] = True

    ref_field_peak_pupil_filter = ref_field_peak_pupil_filter.reshape(-1) 

    ref_field_peak_pupil_pixels =  np.where( ref_field_peak_pupil_filter )[0]

    #plt.figure();plt.imshow( ref_field_peak_pupil_filter.reshape(N0.shape), alpha=0.5); plt.imshow( N0,alpha=0.5); plt.show()

    #================================
    #============= Add to report card 

    #================================
    #============= Add to report card 
    report['secondary_pupil_pixel_filter'] = secondary_pupil_filter # within the cropped img frame based on zwfs.pupil_crop_region
    report['secondary_pupil_pixels'] = secondary_pupil_pixels # within the cropped img frame based on zwfs.pupil_crop_region
    report['secondary_center_ref_pixels'] = ( x_sec_pupil_center, y_sec_pupil_center  ) # within the cropped img frame based on zwfs.pupil_crop_region

    report['outside_pupil_pixel_filter'] = outside_pupil_filter # within the cropped img frame based on zwfs.pupil_crop_region
    report['outside_pupil_pixels'] = outside_pupil_pixels # within the cropped img frame based on zwfs.pupil_crop_region

    # this is important for fitting b parameter as long as this reference point lies within secondary obstruction 
    report['reference_field_peak_filter'] = ref_field_peak_pupil_filter
    
    report['reference_field_peak_pixels'] = ref_field_peak_pupil_pixels

    if debug:

        cmap = colors.ListedColormap(['black', 'green','blue','yellow'])

        norm = colors.BoundaryNorm([0.5,1.5,2.5,3.5], cmap.N)

        region_highlight = np.zeros( N0.shape )
        region_highlight[pupil_filter.reshape(N0.shape)]=1
        region_highlight[secondary_pupil_filter.reshape(N0.shape)]=2
        region_highlight[outside_pupil_filter.reshape(N0.shape)]=0
        region_highlight[ref_field_peak_pupil_filter.reshape(N0.shape)]=3

        fig,ax = plt.subplots(figsize=(8,8))
        #plt.axis('off')
        cax = ax.imshow( region_highlight ,cmap=cmap)
        cbar=fig.colorbar(cax, ticks=[0, 1, 2, 3])
        cbar.ax.set_yticklabels(['outside pupil', 'inside pupil', 'secondary obstruction', 'peak of the reference field'])
        #plt.savefig(fig_path + 'process_1.3_pupil_region_classification.png',bbox_inches='tight', dpi=300)
        plt.show()
    #================================
    #============= Now some basic quality checks 

    if (np.sum( report['pupil_pixel_filter'] ) > 0) & (np.sum( report['pupil_pixel_filter'] ) < 1e20) : # TO DO put reasonable values here (how many pixels do we expect the pupil to cover? -> this will be mode dependent if we are in 12x12 or 6x6. 
        report['got_expected_illum_pixels'] = 1
    else:
        report['got_expected_illum_pixels'] = 0 # check for vignetting etc. 

    if abs(x_dm_center - x_pupil_center) < 50 : #pixels - TO DO put reasonable values here 
        report['dm_center_pix_x=pupil_center_x'] = 1
    else:
        report['dm_center_pix_x=pupil_center_x'] = 0

    if abs(y_dm_center - y_pupil_center) < 50 : #pixels - TO DO put reasonable values here 
        report['dm_center_pix_y=pupil_center_y'] = 1
    else:
        report['dm_center_pix_y=pupil_center_y'] = 0

    #etc we can do other quality control tests 

    report['pupil_quality_flag'] = report['got_expected_illum_pixels'] & report['dm_center_pix_x=pupil_center_x'] & report['dm_center_pix_y=pupil_center_y']
    
    return( report ) 


    #================================
    #============= Measure P2C <-- this should be in phase controller ..



