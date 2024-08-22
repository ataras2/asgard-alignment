import os
import numpy as np 
import matplotlib.pyplot as plt 
import pyzelda.utils.zernike as zernike
from mpl_toolkits.axes_grid1 import make_axes_locatable
import datetime
import time 
from astropy.io import fits 
from scipy import ndimage
from scipy import signal
from scipy.interpolate import RegularGridInterpolator
from scipy.optimize import curve_fit
import itertools
import corner

import sys 
sys.path.insert(1, '/opt/FirstLightImaging/FliSdk/Python/demo/')
sys.path.insert(1,'/opt/Boston Micromachines/lib/Python3/site-packages/')
import FliSdk_V2 
import FliCredThree
import FliCredTwo
import FliCredOne

import bmc
# ============== UTILITY FUNCTIONS


data_path = '/home/baldr/Documents/baldr/ANU_demo_scripts/BALDR/data/' 


def construct_command_basis( basis='Zernike', number_of_modes = 20, Nx_act_DM = 12, Nx_act_basis = 12, act_offset=(0,0), without_piston=True):
    """
    returns a change of basis matrix M2C to go from modes to DM commands, where columns are the DM command for a given modal basis. e.g. M2C @ [0,1,0,...] would return the DM command for tip on a Zernike basis. Modes are normalized on command space such that <M>=0, <M|M>=1. Therefore these should be added to a flat DM reference if being applied.    

    basis = string of basis to use
    number_of_modes = int, number of modes to create
    Nx_act_DM = int, number of actuators across DM diameter
    Nx_act_basis = int, number of actuators across the active basis diameter
    act_offset = tuple, (actuator row offset, actuator column offset) to offset the basis on DM (i.e. we can have a non-centered basis)
    IM_covariance = None or an interaction matrix from command to measurement space. This only needs to be provided if you want KL modes, for this the number of modes is infered by the shape of the IM matrix. 
     
    """

   
    # shorter notations
    #Nx_act = DM.num_actuators_width() # number of actuators across diameter of DM.
    #Nx_act_basis = actuators_across_diam
    c = act_offset
    # DM BMC-3.5 is 12x12 missing corners so 140 actuators , we note down corner indicies of flattened 12x12 array.
    corner_indices = [0, Nx_act_DM-1, Nx_act_DM * (Nx_act_DM-1), -1]

    bmcdm_basis_list = []
    # to deal with
    if basis == 'Zernike':
        if without_piston:
            number_of_modes += 1 # we add one more mode since we dont include piston 

        raw_basis = zernike.zernike_basis(nterms=number_of_modes, npix=Nx_act_basis )
        for i,B in enumerate(raw_basis):
            # normalize <B|B>=1, <B>=0 (so it is an offset from flat DM shape)
            Bnorm = np.sqrt( 1/np.nansum( B**2 ) ) * B
            # pad with zeros to fit DM square shape and shift pixels as required to center
            # we also shift the basis center with respect to DM if required
            if np.mod( Nx_act_basis, 2) == 0:
                pad_width = (Nx_act_DM - B.shape[0] )//2
                padded_B = shift( np.pad( Bnorm , pad_width , constant_values=(np.nan,)) , c[0], c[1])
            else:
                pad_width = (Nx_act_DM - B.shape[0] )//2 + 1
                padded_B = shift( np.pad( Bnorm , pad_width , constant_values=(np.nan,)) , c[0], c[1])[:-1,:-1]  # we take off end due to odd numebr

            flat_B = padded_B.reshape(-1) # flatten basis so we can put it in the accepted DM command format
            np.nan_to_num(flat_B,0 ) # convert nan -> 0
            flat_B[corner_indices] = np.nan # convert DM corners to nan (so lenght flat_B = 140 which corresponds to BMC-3.5 DM)

            # now append our basis function removing corners (nan values)
            bmcdm_basis_list.append( flat_B[np.isfinite(flat_B)] )

        # our mode 2 command matrix
        if without_piston:
            M2C = np.array( bmcdm_basis_list )[1:].T #remove piston mode
        else:
            M2C = np.array( bmcdm_basis_list ).T # take transpose to make columns the modes in command space.


    elif basis == 'KL':         
        if without_piston:
            number_of_modes += 1 # we add one more mode since we dont include piston 

        raw_basis = zernike.zernike_basis(nterms=number_of_modes, npix=Nx_act_basis )
        b0 = np.array( [np.nan_to_num(b) for b in raw_basis] )
        cov0 = np.cov( b0.reshape(len(b0),-1) )
        U , S, UT = np.linalg.svd( cov0 )
        KL_raw_basis = ( b0.T @ U ).T # KL modes that diagonalize Zernike covariance matrix 
        for i,B in enumerate(KL_raw_basis):
            # normalize <B|B>=1, <B>=0 (so it is an offset from flat DM shape)
            Bnorm = np.sqrt( 1/np.nansum( B**2 ) ) * B
            # pad with zeros to fit DM square shape and shift pixels as required to center
            # we also shift the basis center with respect to DM if required
            if np.mod( Nx_act_basis, 2) == 0:
                pad_width = (Nx_act_DM - B.shape[0] )//2
                padded_B = shift( np.pad( Bnorm , pad_width , constant_values=(np.nan,)) , c[0], c[1])
            else:
                pad_width = (Nx_act_DM - B.shape[0] )//2 + 1
                padded_B = shift( np.pad( Bnorm , pad_width , constant_values=(np.nan,)) , c[0], c[1])[:-1,:-1]  # we take off end due to odd numebr

            flat_B = padded_B.reshape(-1) # flatten basis so we can put it in the accepted DM command format
            np.nan_to_num(flat_B,0 ) # convert nan -> 0
            flat_B[corner_indices] = np.nan # convert DM corners to nan (so lenght flat_B = 140 which corresponds to BMC-3.5 DM)

            # now append our basis function removing corners (nan values)
            bmcdm_basis_list.append( flat_B[np.isfinite(flat_B)] )

        # our mode 2 command matrix
        if without_piston:
            M2C = np.array( bmcdm_basis_list )[1:].T #remove piston mode
        else:
            M2C = np.array( bmcdm_basis_list ).T # take transpose to make columns the modes in command space.


    #elif basis == 'Sensor_Eigenmodes': this is done specifically in a phase_control.py function - as it needs a interaction matrix covariance first 

             
    elif basis == 'fourier':
        # NOT TESTED YET ON REAL DM!! 
        if without_piston:
            number_of_modes += 1 # we add one more mode since we dont include piston 

        # NOTE BECAUSE WE HAVE N,M DIMENSIONS WE NEED TO ROUND UP TO SQUARE NUMBER THE MIGHT NOT = EXACTLY number_of_modes
        control_basis_dict  = develop_Fourier_basis( int(np.ceil(number_of_modes**0.5)), int(np.ceil(number_of_modes**0.5)) ,P = 2 * Nx_act_DM, Nx = Nx_act_DM, Ny = Nx_act_DM )
        raw_basis = np.array(list( control_basis_dict.values() ) ) #[:number_of_modes]
        
        bmcdm_basis_list = []
        for i,B in enumerate(raw_basis):
            B = B.reshape(-1)
            B[corner_indices] = np.nan
            bmcdm_basis_list.append( B[np.isfinite(B)] )
        # flatten & normalize each basis cmd 
        if without_piston:
            control_basis = [np.sqrt( 1/np.nansum( cb**2 ) ) * cb.reshape(-1) for cb in bmcdm_basis_list[1:]] #remove piston mode
        else:
            control_basis = [np.sqrt( 1/np.nansum( cb**2 ) ) * cb.reshape(-1) for cb in bmcdm_basis_list]# take transpose to make columns the modes in command space.
        M2C = np.array( control_basis ).T 

    elif basis == 'Zonal': 
        #hardcoded for BMC multi3.5 DM (140 actuators)
        M2C = np.eye( 140 ) # we just consider this over all actuators (so defaults to 140 modes) 
        # we filter zonal basis in the eigenvectors of the control matrix. 
    
    return(M2C)


def fourier_vector(n, m, P = 2*12, Nx = 12, Ny = 12):
    """
    OR we can do it with complex exponetial, in-quadrature is real part, out of quadrature is imaginary 
    Normalized <Bx|Bx>=1 , <By|By>=1

    Parameters
    ----------
    n : TYPE
        DESCRIPTION.
    m : TYPE
        DESCRIPTION.
    P : TYPE, optional
        DESCRIPTION. The default is 2*12.
    Nx : TYPE, optional
        DESCRIPTION. The default is 12.
    Ny : TYPE, optional
        DESCRIPTION. The default is 12.

    Returns
    -------
    None.

    """
    x = np.linspace(-6,6,Nx)
    y = np.linspace(-6,6,Ny)
    X,Y = np.meshgrid(x,y)
    
    
    Bx = np.exp( 1J * 2 * np.pi * n/P * X )
    if np.sum( abs(Bx) ):
        Bx *= 1/np.sum( abs(Bx)**2 )**0.5

    By = np.exp( 1J * 2 * np.pi * m/P * Y )
    if np.sum( abs(By) ):
        By *= 1/np.sum( abs(By)**2 )**0.5
    
    return( Bx, By )

def develop_Fourier_basis( n,m ,P = 2*12, Nx = 12, Ny = 12):
    """
    

    Parameters
    ----------
    n : TYPE int
        DESCRIPTION. what order in x (column) dimension do we create Fourier basis for?
    m : TYPE int
        DESCRIPTION. what order in y (row) dimension do we create Fourier basis for?

    Returns
    -------
    basis_dict - a dictionary indexed by mode order tuple (n,m) with corresponding 2D Fourier basis
    
    
    # what is logical indexing? 
    basis naturally forms 2 NxN squares, one square corresponding to odd components (sin) in x,y other to even (cos)

    for each axis dimension cnt, with even numbers corresponding to even functions (cos), odd numbers to odd functions (sin)
    therefore to recover cos or sin order we simply divide by 2 and round (//2)

    we do not count piston  
    e.g. indexing for x dimension:
    0 = np.real(F_basis_x[0] )  
    1 = np.imag(F_basis_x[0] )  
    2 = np.iamg(F_basis_x[1] ) 
    3 = np.real(F_basis_x[1] ) 

    therefore for example index (3,2)
    B_(3,2) = np.real(F_basis_x[1] ) * np.imag(F_basis_y[1] )
    first index corresponds to variation across columns (x), 
    second index corresponds to variation across rows (y)

    """
    basis_dict = {}

    for x_idx in range(0,n):
        for y_idx in range(0,m):            
            #
            x_order = x_idx//2
            y_order = y_idx//2
            
            if not ((x_idx==0) | (y_idx==0)): # otherwise we get lots of pistons 
                Bx, By = fourier_vector(x_order, y_order, P , Nx , Ny )
                    
                if not np.mod(x_idx,2): #odd number take imaginary (odd) part
                    Bx_q = np.imag( Bx )
    
                else: # even number take real (even) part
                    Bx_q = np.real( Bx )
    
                    
                if not np.mod(y_idx,2): #odd number take imaginary (odd) part
                    By_q = np.imag( By )
                    
                else: # even number take real (even) part
                    By_q = np.real( By )
            
                #if x_idx > 1:
                mode_tmp = Bx_q * By_q - np.mean(Bx_q * By_q)
                if np.sum( mode_tmp**2):
                    mode_tmp *= 1/np.sum( mode_tmp**2)**0.5 #normalized <M|M>=1
                basis_dict[(x_idx-1,y_idx-1)] =  mode_tmp


    return(basis_dict)


def spiral_search_TT_coefficients( dr, dtheta, aoi_tp, aoi_tt, num_points, r0=0, theta0=0):
    """
    generate tip (tp) / tilt (tt) coefficients for a spiral search covering
    "num_points" samples with angular increments dtheta, radial increments dr
    aoi_tp, aoi_tilt are the anlge of incidence on the DM for tip and tilt. 

    ALL angular units should be input as radians.
    """

    coefficients = []
    theta = theta0 # initial angle
    radius = r0 # initial radius
    
    for _ in range(num_points):
        a_tp = radius * np.cos(theta) * np.cos( aoi_tp )
        a_Tt = radius * np.sin(theta) * np.cos( aoi_tt )
        coefficients.append((a_tp, a_Tt))
        
        # Increment radius and angle
        radius += dr
        theta += dtheta
    
    return coefficients


def get_DM_command_in_2D(cmd,Nx_act=12):
    # function so we can easily plot the DM shape (since DM grid is not perfectly square raw cmds can not be plotted in 2D immediately )
    #puts nan values in cmd positions that don't correspond to actuator on a square grid until cmd length is square number (12x12 for BMC multi-2.5 DM) so can be reshaped to 2D array to see what the command looks like on the DM.
    corner_indices = [0, Nx_act-1, Nx_act * (Nx_act-1), Nx_act*Nx_act]
    cmd_in_2D = list(cmd.copy())
    for i in corner_indices:
        cmd_in_2D.insert(i,np.nan)
    return( np.array(cmd_in_2D).reshape(Nx_act,Nx_act) )



def circle(radius, size, circle_centre=(0, 0), origin="middle"):
    
    """
    Adopted from AO tools with edit that we can use size as a tuple of row_size, col_size to include rectangles 
    
    Create a 2-D array: elements equal 1 within a circle and 0 outside.

    The default centre of the coordinate system is in the middle of the array:
    circle_centre=(0,0), origin="middle"
    This means:
    if size is odd  : the centre is in the middle of the central pixel
    if size is even : centre is in the corner where the central 4 pixels meet

    origin = "corner" is used e.g. by psfAnalysis:radialAvg()

    Examples: ::

        circle(1,5) circle(0,5) circle(2,5) circle(0,4) circle(0.8,4) circle(2,4)
          00000       00000       00100       0000        0000          0110
          00100       00000       01110       0000        0110          1111
          01110       00100       11111       0000        0110          1111
          00100       00000       01110       0000        0000          0110
          00000       00000       00100

        circle(1,5,(0.5,0.5))   circle(1,4,(0.5,0.5))
           .-->+
           |  00000               0000
           |  00000               0010
          +V  00110               0111
              00110               0010
              00000

    Parameters:
        radius (float)       : radius of the circle
        size (int)           : tuple of row  and column size of the 2-D array in which the circle lies
        circle_centre (tuple): coords of the centre of the circle
        origin (str)  : where is the origin of the coordinate system
                               in which circle_centre is given;
                               allowed values: {"middle", "corner"}

    Returns:
        ndarray (float64) : the circle array
    """
	
    size_row , size_col = size
    # (2) Generate the output array:
    C = np.zeros((size_row, size_col))

    # (3.a) Generate the 1-D coordinates of the pixel's centres:
    # coords = np.linspace(-size/2.,size/2.,size) # Wrong!!:
    # size = 5: coords = array([-2.5 , -1.25,  0.  ,  1.25,  2.5 ])
    # size = 6: coords = array([-3. , -1.8, -0.6,  0.6,  1.8,  3. ])
    # (2015 Mar 30; delete this comment after Dec 2015 at the latest.)

    # Before 2015 Apr 7 (delete 2015 Dec at the latest):
    # coords = np.arange(-size/2.+0.5, size/2.-0.4, 1.0)
    # size = 5: coords = array([-2., -1.,  0.,  1.,  2.])
    # size = 6: coords = array([-2.5, -1.5, -0.5,  0.5,  1.5,  2.5])

    coords_r = np.arange(0.5, size_row, 1.0)
    coords_c = np.arange(0.5, size_col, 1.0)
    # size = 5: coords = [ 0.5  1.5  2.5  3.5  4.5]
    # size = 6: coords = [ 0.5  1.5  2.5  3.5  4.5  5.5]

    # (3.b) Just an internal sanity check:
    if len(coords_r) != size_row:
        print('opps')

    # (3.c) Generate the 2-D coordinates of the pixel's centres:
    x, y = np.meshgrid(coords_c, coords_r)

    # (3.d) Move the circle origin to the middle of the grid, if required:
    if origin == "middle":
        x -= size_col / 2.
        y -= size_row / 2.

    # (3.e) Move the circle centre to the alternative position, if provided:
    x -= circle_centre[0]
    y -= circle_centre[1]

    # (4) Calculate the output:
    # if distance(pixel's centre, circle_centre) <= radius:
    #     output = 1
    # else:
    #     output = 0
    mask = x * x + y * y <= radius * radius
    C[mask] = 1

    # (5) Return:
    return C


def shift(xs, n, m, fill_value=np.nan):
    # shifts a 2D array xs by n rows, m columns and fills the new region with fill_value

    e = xs.copy()
    if n!=0:
        if n >= 0:
            e[:n,:] = fill_value
            e[n:,:] =  e[:-n,:]
        else:
            e[n:,:] = fill_value
            e[:n,:] =  e[-n:,:]
   
       
    if m!=0:
        if m >= 0:
            e[:,:m] = fill_value
            e[:,m:] =  e[:,:-m]
        else:
            e[:,m:] = fill_value
            e[:,:m] =  e[:,-m:]
    return e


def line_intersection(line1, line2):
    """
    find intersection of lines given by their endpoints, 
       line1 = (A,B)
       line2 = (C,D)
       where A=[x1_1, y1_1], B=[x1_2,y1_2], are end points of line1 
             C=[x2_1, y2_1], D=[x2_2, y2_2], are end points of line2
        
    """
 
    xdiff = (line1[0][0] - line1[1][0], line2[0][0] - line2[1][0])
    ydiff = (line1[0][1] - line1[1][1], line2[0][1] - line2[1][1])

    def det(a, b):
        return a[0] * b[1] - a[1] * b[0]

    div = det(xdiff, ydiff)
    if div == 0:
       raise Exception('lines do not intersect')

    d = (det(*line1), det(*line2))
    x = det(d, xdiff) / div
    y = det(d, ydiff) / div
    return( x, y )


def move_fpm( tel, pos = 0):
    # in real life this will command a motor - but for now i do this manually
    """
    # SETS FOCAL PLANE MASK - TO IMPLIMENT ONCE WE HAVE FPM MOTORS
    if pos == 0:    
        print( 'move FPM to position X')
    elif pos == 1:  
        print( 'move FPM to position Y') #etc
    """
    return(None)


def watch_camera(zwfs, frames_to_watch = 10, time_between_frames=0.01,cropping_corners=None) :
  
    print( f'{frames_to_watch} frames to watch with ~{time_between_frames}s wait between frames = ~{5*time_between_frames*frames_to_watch}s watch time' )

    #t0= datetime.datetime.now() 
    plt.figure(figsize=(15,15))
    plt.ion() # turn on interactive mode 
    #FliSdk_V2.Start(camera)     
    seconds_passed = 0
    if type(cropping_corners)==list: 
        x1,x2,y1,y2 = cropping_corners #[row min, row max, col min, col max]

    for i in range(int(frames_to_watch)): 
        
        a = zwfs.get_image()
        if type(cropping_corners)==list: 
            plt.imshow(a[x1:x2,y1:y2])
        else: 
            plt.imshow(a)
        plt.pause( time_between_frames )
        #time.sleep( time_between_frames )
        plt.clf() 
    """
    while seconds_passed < seconds_to_watch:
        a=FliSdk_V2.GetRawImageAsNumpyArray(camera,-1)
        plt.imshow(a)
        plt.pause( time_between_frames )
        time.sleep( time_between_frames )
        plt.clf() 
        t1 = datetime.datetime.now() 
        seconds_passed = (t1 - t0).seconds"""

    #FliSdk_V2.Stop(camera) 
    plt.ioff()# turn off interactive mode 
    plt.close()







def create_phase_screen_cmd_for_DM(scrn,  scaling_factor=0.1, drop_indicies = None, plot_cmd=False):
    """
    aggregate a scrn (aotools.infinitephasescreen object) onto a DM command space. phase screen is normalized by
    between +-0.5 and then scaled by scaling_factor. Final DM command values should
    always be between -0.5,0.5 (this should be added to a flat reference so flat reference + phase screen should always be bounded between 0-1). phase screens are usually a NxN matrix, while DM is MxM with some missing pixels (e.g. 
    corners). drop_indicies is a list of indicies in the flat MxM DM array that should not be included in the command space. 
    """

    #print('----------\ncheck phase screen size is multiple of DM\n--------')
    
    Nx_act = 12 #number of actuators across DM diameter
    
    scrn_array = ( scrn.scrn - np.min(scrn.scrn) ) / (np.max(scrn.scrn) - np.min(scrn.scrn)) - 0.5 # normalize phase screen between -0.5 - 0.5 
    
    size_factor = int(scrn_array.shape[0] / Nx_act) # how much bigger phase screen is to DM shape in x axis. Note this should be an integer!!
    
    # reshape screen so that axis 1,3 correspond to values that should be aggregated 
    scrn_to_aggregate = scrn_array.reshape(scrn_array.shape[0]//size_factor, size_factor, scrn_array.shape[1]//size_factor, size_factor)
    
    # now aggreagate and apply the scaling factor 
    scrn_on_DM = scaling_factor * np.mean( scrn_to_aggregate, axis=(1,3) ).reshape(-1) 

    #If DM is missing corners etc we set these to nan and drop them before sending the DM command vector
    #dm_cmd =  scrn_on_DM.to_list()
    if drop_indicies != None:
        for i in drop_indicies:
            scrn_on_DM[i]=np.nan
             
    if plot_cmd: #can be used as a check that the command looks right!
        fig,ax = plt.subplots(1,2,figsize=(12,6))
        im0 = ax[0].imshow( scrn_on_DM.reshape([Nx_act,Nx_act]) )
        ax[0].set_title('DM command (averaging offset)')
        im1 = ax[1].imshow(scrn.scrn)
        ax[1].set_title('original phase screen')
        plt.colorbar(im0, ax=ax[0])
        plt.colorbar(im1, ax=ax[1]) 
        plt.show() 

    dm_cmd =  list( scrn_on_DM[np.isfinite(scrn_on_DM)] ) #drop non-finite values which should be nan values created from drop_indicies array
    return(dm_cmd) 






def block_sum(ar, fact): # sums over subwindows of a 2D array
    # ar is the 2D array, fact is the factor to reduce it by 
    # obviously  fact should be factor of ar.shape 
    assert isinstance(fact, int), type(fact)
    sx, sy = ar.shape
    X, Y = np.ogrid[0:sx, 0:sy]
    regions = sy//fact * (X//fact) + Y//fact
    res = ndimage.sum(ar, labels=regions, index=np.arange(regions.max() + 1))
    res.shape = (sx//fact, sy//fact)
    return res




# ========== PLOTTING STANDARDS 
def nice_heatmap_subplots( im_list , xlabel_list, ylabel_list, title_list,cbar_label_list, fontsize=15, cbar_orientation = 'bottom', axis_off=True, vlims=None, savefig=None):

    n = len(im_list)
    fs = fontsize
    fig = plt.figure(figsize=(5*n, 5))

    for a in range(n) :
        ax1 = fig.add_subplot(int(f'1{n}{a+1}'))
        ax1.set_title(title_list[a] ,fontsize=fs)

        if vlims!=None:
            im1 = ax1.imshow(  im_list[a] , vmin = vlims[a][0], vmax = vlims[a][1])
        else:
            im1 = ax1.imshow(  im_list[a] )
        ax1.set_title( title_list[a] ,fontsize=fs)
        ax1.set_xlabel( xlabel_list[a] ,fontsize=fs) 
        ax1.set_ylabel( ylabel_list[a] ,fontsize=fs) 
        ax1.tick_params( labelsize=fs ) 

        if axis_off:
            ax1.axis('off')
        divider = make_axes_locatable(ax1)
        if cbar_orientation == 'bottom':
            cax = divider.append_axes('bottom', size='5%', pad=0.05)
            cbar = fig.colorbar( im1, cax=cax, orientation='horizontal')
                
        elif cbar_orientation == 'top':
            cax = divider.append_axes('top', size='5%', pad=0.05)
            cbar = fig.colorbar( im1, cax=cax, orientation='horizontal')
                
        else: # we put it on the right 
            cax = divider.append_axes('right', size='5%', pad=0.05)
            cbar = fig.colorbar( im1, cax=cax, orientation='vertical')  
        
   
        cbar.set_label( cbar_label_list[a], rotation=0,fontsize=fs)
        cbar.ax.tick_params(labelsize=fs)
    if savefig!=None:
        plt.savefig( savefig , bbox_inches='tight', dpi=300) 

    #plt.show() 

def nice_DM_plot( data, savefig=None ): #for a 140 actuator BMC 3.5 DM
    fig,ax = plt.subplots(1,1)
    if len( np.array(data).shape ) == 1: 
        ax.imshow( get_DM_command_in_2D(data) )
    else: 
        ax.imshow( data )
    ax.set_title('poorly registered actuators')
    ax.grid(True, which='minor',axis='both', linestyle='-', color='k', lw=2 )
    ax.set_xticks( np.arange(12) - 0.5 , minor=True)
    ax.set_yticks( np.arange(12) - 0.5 , minor=True)
    if savefig!=None:
        plt.savefig( savefig , bbox_inches='tight', dpi=300) 




def project_matrix( CM , projection_vector_list ):
    """
    create two new matrices CM_TT, and CM_HO from CM, 
    where CM_TT projects any "signal" onto the column space of vectors in 
    projection_vector_list vectors, 
    and CM_HO which projects any "signal" to the null space of CM_TT that is within CM.
    
    Typical use is to seperate control matrix to tip/tilt reconstructor (CM_TT) and
    higher order reconstructor (CM_HO)

    Note vectors in projection_vector_list  must be 
    
    Parameters
    ----------
    CM : TYPE matrix
        DESCRIPTION. 
    projection_vector_list : list of vectors that are in col space of CM
        DESCRIPTION.

    Returns
    -------
    CM_TT , CM_HO

    """
    
    # Create the matrix T from projection_vector_list (e.g. tip and tilt vectors )
    projection_vector_list
    T = np.column_stack( projection_vector_list )  # T is Mx2
    
    # Calculate the projection matrix P
    #P = T @ np.linalg.inv(T.T @ T) @ T.T  # P is MxM <- also works like this 
    # Compute SVD of T (this way is more numerically stable)
    U, S, Vt = np.linalg.svd(T, full_matrices=False)
    
    #  Compute the projection matrix P using SVD
    P = U @ U.T  # U @ U.T gives the projection matrix onto the column space of T
    
    #  Compute CM_TT (projection onto tip and tilt space)
    CM_TT = P @ CM  # CM_TT is MxN
    
    #  Compute the null space projection matrix and CM_HO
    I = np.eye(T.shape[0])  # Identity matrix of size MxM
    CM_HO = (I - P) @ CM  # CM_HO is MxN

    return( CM_TT , CM_HO )

# ==========

def test_controller_in_cmd_space( zwfs, phase_controller, Vw =None , D=None , AO_lag=None  ):
    #outputs fits file with telemetry  
    # applies open loop for X iterations, Closed loop for Y iterations
    # ability to ramp number of controlled modes , how to deal with gains? gain = 1/number controlled modes for Zernike basis

    #open loop 
    for i in range(X) : 

        for i in range( 3 ) :
            raw_img_list.append( zwfs.get_image() ) # @D, remember for control_phase method this needs to be flattened and filtered for pupil region

        raw_img.append( np.median( raw_img_list, axis = 0 ) ) 

        # ALSO MANUALLY GET PSF IMAGE 
        
        err_img.append(  phase_controller.get_img_err( raw_img[-1].reshape(-1)[zwfs.pupil_pixels]  ) )

        reco_modes.append( phase_controller.control_phase( err_img[-1]  , controller_name = ctrl_method_label) )

        reco_dm_cmds.append( phase_controller.config['M2C'] @ mode_reco[-1] )

        delta_cmd.append( Ki * delta_cmd[-1] - Kp * reco_dm_cmds[-1] )

        # propagate_phase_screen one step 

        # don't yet apply the correction cmd while in open loop
        zwfs.dm.send_data( zwfs.dm_shapes['flat_dm'] + dist[-1] ) 

        time.sleep(0.1)


    #close loop 
    for i in range(X) : 

        for i in range( 3 ) :
            raw_img_list.append( zwfs.get_image() ) # @D, remember for control_phase method this needs to be flattened and filtered for pupil region

        raw_img.append( np.median( raw_img_list, axis = 0 ) ) 

        err_img.append(  phase_controller.get_img_err( raw_img[-1].reshape(-1)[zwfs.pupil_pixels]  ) )

        reco_modes.append( phase_controller.control_phase( err_img[-1]  , controller_name = ctrl_method_label) )

        reco_dm_cmds.append( phase_controller.config['M2C'] @ mode_reco[-1] )

        delta_cmd.append( Ki * delta_cmd[-1] - Kp * reco_dm_cmds[-1] )

        # propagate_phase_screen one step 

        # then apply the correction 
        zwfs.dm.send_data( zwfs.dm_shapes['flat_dm'] + delta_cmd[-1] + dist[-1] )

        time.sleep(0.1)




def apply_sequence_to_DM_and_record_images(zwfs, DM_command_sequence, number_images_recorded_per_cmd = 1, take_median_of_images=False, save_dm_cmds = True, calibration_dict=None, additional_header_labels=None, sleeptime_between_commands=0.01, cropping_corners=None, save_fits = None):
    """
    

    Parameters
    ----------

    zwfs is a zwfs object from the ZWFS class that must have zwfs.dm and zwfs.camera attributes:
        dm : TYPE
            DESCRIPTION. DM Object from BMC SDK. Initialized  (this is zwfs.camera object) 
        camera : TYPE camera objection from FLI SDK. (this is zwfs.camera object)
            DESCRIPTION. Camera context from FLISDK. Initialized from context = FliSdk_V2.Init(), 

    DM_command_sequence : TYPE list 
        DESCRIPTION. Nc x Na matrix where Nc is number of commands to send in sequence (rows)
        Na is number actuators on DM (columns).   
    number_images_recorded_per_cmd : TYPE, optional
        DESCRIPTION. The default is 1. puting a value >= 0 means no images are recorded.
    take_median_of_images: TYPE, optional
        DESCRIPTION. The default is False. if True we take the median image of number_images_recorded_per_cmd such that there is only one image per command (that is the aggregated image)
    calibration_dict: TYPE, optional
        DESCRIPTION. The default is None meaning saved images don't get flat fielded. 
        if flat fielding is required a dictionary must be supplied that contains 
        a bias, dark and flat frame under keys 'bias', 'dark', and 'flat' respectively
    additional_header_labels : TYPE, optional
        DESCRIPTION. The default is None which means no additional header is appended to fits file 
        otherwise a tuple (header, value) or list of tuples [(header_0, value_0)...] can be used. 
        If list, each item in list will be added as a header. 
    cropping_corners: TYPE, optional
        DESCRIPTION. list of length 4 holding [row min, row max, col min, col max] to crop raw data frames.
    save_fits : TYPE, optional
        DESCRIPTION. The default is None which means images are not saved, 
        if a string is provided images will be saved with that name in the current directory
    sleeptime_between_commands : TYPE, optional float
        DESCRIPTION. time to sleep between sending DM commands and recording images in seconds. default is 0.01s
    Returns
    -------
    fits file with images corresponding to each DM command in sequence
    first extension is images
    second extension is DM commands

    """

    
    should_we_record_images = True
    try: # foce to integer
        number_images_recorded_per_cmd = int(number_images_recorded_per_cmd)
        if number_images_recorded_per_cmd <= 0:
            should_we_record_images = False
    except:
        raise TypeError('cannot convert "number_images_recorded_per_cmd" to a integer. Check input type')
    
    image_list = [] #init list to hold images

    # NOTE THE CAMERA SHOULD ALREADY BE STARTED BEFORE BEGINNING - No checking here yet
    for cmd_indx, cmd in enumerate(DM_command_sequence):
        print(f'executing cmd_indx {cmd_indx} / {len(DM_command_sequence)}')
        # wait a sec        
        time.sleep(sleeptime_between_commands)
        # ok, now apply command
        zwfs.dm.send_data(cmd)
        # wait a sec        
        time.sleep(sleeptime_between_commands)

        if should_we_record_images: 
            if take_median_of_images:
                ims_tmp = [np.median([zwfs.get_image() for _ in range(number_images_recorded_per_cmd)] , axis=0)] #keep as list so it is the same type as when take_median_of_images=False
            else:
                ims_tmp = zwfs.get_image() #get_raw_images(camera, number_images_recorded_per_cmd, cropping_corners) 
            image_list.append( ims_tmp )

    
    #FliSdk_V2.Stop(camera) # stop camera
    
    # init fits files if necessary
    if should_we_record_images: 
        #cmd2pix_registration
        data = fits.HDUList([]) #init main fits file to append things to
        
        # Camera data
        cam_fits = fits.PrimaryHDU( image_list )
        
        cam_fits.header.set('EXTNAME', 'SEQUENCE_IMGS' )
        #camera headers
        camera_info_dict = get_camera_info(zwfs.camera)
        for k,v in camera_info_dict.items():
            cam_fits.header.set(k,v)
        cam_fits.header.set('#images per DM command', number_images_recorded_per_cmd )
        cam_fits.header.set('take_median_of_images', take_median_of_images )
        
        cam_fits.header.set('cropping_corners_r1', zwfs.pupil_crop_region[0] )
        cam_fits.header.set('cropping_corners_r2', zwfs.pupil_crop_region[1] )
        cam_fits.header.set('cropping_corners_c1', zwfs.pupil_crop_region[2] )
        cam_fits.header.set('cropping_corners_c2', zwfs.pupil_crop_region[3] )

        #if user specifies additional headers using additional_header_labels
        if (additional_header_labels!=None): 
            if type(additional_header_labels)==list:
                for i,h in enumerate(additional_header_labels):
                    cam_fits.header.set(h[0],h[1])
            else:
                cam_fits.header.set(additional_header_labels[0],additional_header_labels[1])

        # add camera data to main fits
        data.append(cam_fits)
        
        if save_dm_cmds:
            # put commands in fits format
            dm_fits = fits.PrimaryHDU( DM_command_sequence )
            #DM headers 
            dm_fits.header.set('timestamp', str(datetime.datetime.now()) )
            dm_fits.header.set('EXTNAME', 'DM_CMD_SEQUENCE' )
            #dm_fits.header.set('DM', DM.... )
            #dm_fits.header.set('#actuators', DM.... )

            # append to the data
            data.append(dm_fits)
        
        if save_fits!=None:
            if type(save_fits)==str:
                data.writeto(save_fits)
            else:
                raise TypeError('save_images needs to be either None or a string indicating where to save file')
            
            
        return(data)
    
    else:
        return(None)



def get_camera_info(camera):
    if FliSdk_V2.IsCredOne(camera):
        cred = FliCredOne.FliCredOne() #cred1 object 

    elif FliSdk_V2.IsCredTwo(camera):
        cred = FliCredTwo.FliCredTwo() #cred3 object 
    
    elif FliSdk_V2.IsCredThree(camera):
        cred = FliCredThree.FliCredThree() #cred3 object 
    camera_info_dict = {} 
    
    # cropping rows 
    _, cropping_rows = FliSdk_V2.FliSerialCamera.SendCommand(camera, "cropping rows")
    _, cropping_columns =  FliSdk_V2.FliSerialCamera.SendCommand(camera, "cropping columns")

    # query camera settings 
    fps_res, fps_response = FliSdk_V2.FliSerialCamera.GetFps(camera)
    tint_res, tint_response = FliSdk_V2.FliSerialCamera.SendCommand(camera, "tint raw")
  
    # gain
    #gain = cred.GetConversionGain(camera)[1]
    _, gain = FliSdk_V2.FliSerialCamera.SendCommand(camera, "sensibility") #NOTE OLD FIRMWARE, NEWER USES sensitivity

    #camera headers
    camera_info_dict['timestamp'] = str(datetime.datetime.now()) 
    camera_info_dict['camera'] = FliSdk_V2.GetCurrentCameraName(camera) 
    camera_info_dict['camera_fps'] = fps_response
    camera_info_dict['camera_tint'] = tint_response
    camera_info_dict['camera_gain'] = gain.split(':')[-1]
    camera_info_dict['cropping_rows'] = cropping_rows
    camera_info_dict['cropping_columns'] = cropping_columns

    return(camera_info_dict)

    
def scan_detector_framerates(zwfs, frame_rates, number_images_recorded_per_cmd = 50, cropping_corners=None, save_fits = None): 
    """
    iterate through different camera frame rates and record a series of images for each
    this can be used for building darks or flats.

    Parameters
    ----------
    zwfs : TYPE zwfs object holding camera object from FLI SDK.
        DESCRIPTION. Camera context from FLISDK. Initialized from context = FliSdk_V2.Init() 
    frame_rates : TYPE list like 
        DESCRIPTION. array holding different frame rates to iterate through
    number_images_recorded_per_cmd : TYPE, optional
        DESCRIPTION. The default is 50. puting a value >= 0 means no images are recorded.
    save_fits : TYPE, optional
        DESCRIPTION. The default is None which means images are not saved, 
        if a string is provided images will be saved with that name in the current directory

    Raises
    ------
    TypeError
        DESCRIPTION.

    Returns
    -------
    fits file with each extension corresponding to a different camera frame rate 

    """
    
    #zwfs.set_camera_fps(600) 
    zwfs.start_camera() # start camera

    data = fits.HDUList([]) 
    for fps in frame_rates:
        
        zwfs.set_camera_fps(fps) # set max dit (tint=None) for given fps
	
        time.sleep(1) # wait 1 second
        #tmp_fits = fits.PrimaryHDU( [FliSdk_V2.GetProcessedImageGrayscale16bNumpyArray(camera,-1)  for i in range(number_images_recorded_per_cmd)] )
        img_list = []
        for _ in range(number_images_recorded_per_cmd):
            img_list.append( zwfs.get_image() )
            time.sleep(0.01)

        tmp_fits = fits.PrimaryHDU(  img_list )

        
        camera_info_dict = get_camera_info(zwfs.camera)
        for k,v in camera_info_dict.items():
            tmp_fits.header.set(k,v)     

        data.append( tmp_fits )

    zwfs.stop_camera()  # stop camera
    
    if save_fits!=None:
        if type(save_fits)==str:
            data.writeto(save_fits)
        else:
            raise TypeError('save_images needs to be either None or a string indicating where to save file')
        
    return(data)




def GET_BDR_RECON_DATA_INTERNAL(zwfs, number_amp_samples = 18, amp_max = 0.2, number_images_recorded_per_cmd = 2, save_fits = None) :

    flat_dm_cmd = zwfs.dm_shapes['flat_dm']

    modal_basis = np.eye(len(flat_dm_cmd))
    cp_x1,cp_x2,cp_y1,cp_y2 = zwfs.pupil_crop_region

    ramp_values = np.linspace(-amp_max, amp_max, number_amp_samples)

    # ======== reference image with FPM OUT

    zwfs.dm.send_data(flat_dm_cmd) 
    _ = input('MANUALLY MOVE PHASE MASK OUT OF BEAM, PRESS ENTER TO BEGIN' )
    watch_camera(zwfs, frames_to_watch = 70, time_between_frames=0.05)

    N0_list = []
    for _ in range(number_images_recorded_per_cmd):
        N0_list.append( zwfs.get_image( ) ) #REFERENCE INTENSITY WITH FPM IN
    N0 = np.median( N0_list, axis = 0 ) 

    #make_fits

    # ======== reference image with FPM IN
    _ = input('MANUALLY MOVE PHASE MASK BACK IN, PRESS ENTER TO BEGIN' )
    watch_camera(zwfs, frames_to_watch = 70, time_between_frames=0.05)

    I0_list = []
    for _ in range(number_images_recorded_per_cmd):
        I0_list.append( zwfs.get_image(  ) ) #REFERENCE INTENSITY WITH FPM IN
    I0 = np.median( I0_list, axis = 0 ) 

    # ======== BIAS FRAME
    _ = input('COVER THE DETECTOR FOR A BIAS FRAME' )


    watch_camera(zwfs, frames_to_watch = 50, time_between_frames=0.05)

    BIAS_list = []
    for _ in range(100):
        time.sleep(0.05)
        BIAS_list.append( zwfs.get_image(  ) ) #REFERENCE INTENSITY WITH FPM IN
    #I0 = np.median( I0_list, axis = 0 ) 

    #====== make references fits files
    I0_fits = fits.PrimaryHDU( I0 )
    N0_fits = fits.PrimaryHDU( N0 )
    BIAS_fits = fits.PrimaryHDU( BIAS_list )
    I0_fits.header.set('EXTNAME','FPM_IN')
    N0_fits.header.set('EXTNAME','FPM_OUT')
    BIAS_fits.header.set('EXTNAME','BIAS')

    flat_DM_fits = fits.PrimaryHDU( flat_dm_cmd )
    flat_DM_fits.header.set('EXTNAME','FLAT_DM_CMD')

    _ = input('PRESS ENTER WHEN READY TO BUILD IM' )

    #make_fits

    # ======== RAMPING ACTUATORS  
    # --- creating sequence of dm commands
    _DM_command_sequence = [list(flat_dm_cmd + amp * modal_basis) for amp in ramp_values ]  
    # add in flat dm command at beginning of sequence and reshape so that cmd sequence is
    # [0, a0*b0,.. aN*b0, a0*b1,...,aN*b1, ..., a0*bM,...,aN*bM]
    DM_command_sequence = [flat_dm_cmd] + list( np.array(_DM_command_sequence).reshape(number_amp_samples*modal_basis.shape[0],modal_basis.shape[1] ) )

    # --- additional labels to append to fits file to keep information about the sequence applied 
    additional_labels = [('cp_x1',cp_x1),('cp_x2',cp_x2),('cp_y1',cp_y1),('cp_y2',cp_y2),('in-poke max amp', np.max(ramp_values)),('out-poke max amp', np.min(ramp_values)),('#ramp steps',number_amp_samples), ('seq0','flatdm'), ('reshape',f'{number_amp_samples}-{modal_basis.shape[0]}-{modal_basis.shape[1]}'),('Nmodes_poked',len(modal_basis)),('Nact',140)]

    # --- poke DM in and out and record data. Extension 0 corresponds to images, extension 1 corresponds to DM commands
    raw_recon_data = apply_sequence_to_DM_and_record_images(zwfs, DM_command_sequence, number_images_recorded_per_cmd = number_images_recorded_per_cmd, take_median_of_images=True, save_dm_cmds = True, calibration_dict=None, additional_header_labels = additional_labels,sleeptime_between_commands=0.03, cropping_corners=None,  save_fits = None ) # None

    zwfs.dm.send_data(flat_dm_cmd) 

    # append FPM IN and OUT references (note FPM in reference is also first entry in recon_data so we can compare if we want!) 
    raw_recon_data.append( I0_fits ) 
    raw_recon_data.append( N0_fits ) 
    raw_recon_data.append( BIAS_fits )
    raw_recon_data.append( flat_DM_fits )

    if save_fits != None:  
        if type(save_fits)==str:
            raw_recon_data.writeto(save_fits)
        else:
            raise TypeError('save_images needs to be either None or a string indicating where to save file')


    return(raw_recon_data) 


def Ic_model_constrained(x, A, B, F, mu):
    penalty = 0
    if (F < 0) or (mu < 0): # F and mu can be correlated so constrain the quadrants 
        penalty = 1e3
    I = A + 2 * B * np.cos(F * x + mu) + penalty
    return I 

def Ic_model_constrained_3param(x, A,  F, mu): 
    #penalty = 0
    #if (F < 0) or (mu < 0): # F and mu can be correlated so constrain the quadrants 
    #    penalty = 1e3
    # 
    I = A**2 + 2 * A * np.cos(F * x + mu) + penalty
    return I 


# should this be free standing or a method? ZWFS? controller? - output a report / fits file
def PROCESS_BDR_RECON_DATA_INTERNAL(recon_data, bad_pixels = ([],[]), active_dm_actuator_filter=None, debug=True, savefits=None) :
    """
    # calibration of our ZWFS: 
    # this will fit M0, b0, mu, F which can be appended to a phase_controller,
    # f and/or mu can be be provided (because may haave been previously characterise) in which case only b0/M0 is fitted.   
    # also P2C_1x1, P2C_3x3 matrix which corresponds to 1x1 and 3x3 pixels around where peak influence for an actuator was found. 
    # Also pupil regions to filter 
       - active pupil pixels 
       - secondary obstruction pixels
       - outside pupil pixels 

    # plot fits, histograms of values (corner plot?), also image highlighting the regions (coloring pixels) 
    # estimate center of DM in pixels and center of pupil in pixels (using previous method)   
    # note A^2 can be measured with FPM out, M can be sampled with FPM in where A^2 = 0. 
    
    # e.g. to generate bad pixel tuple in correct format
    # np.where( (np.std( poke_imgs ,axis = (0,1)) > 100) + (np.std( poke_imgs ,axis = (0,1)) == 0 ) )
    """    


    # ========================== !! 0 !! =====================
    # -- prelims of reading in and labelling data 

    # poke values used in linear ramp
    No_ramps = int(recon_data['SEQUENCE_IMGS'].header['#ramp steps'])
    max_ramp = float( recon_data['SEQUENCE_IMGS'].header['in-poke max amp'] )
    min_ramp = float( recon_data['SEQUENCE_IMGS'].header['out-poke max amp'] ) 
    ramp_values = np.linspace( min_ramp, max_ramp, No_ramps)

    flat_dm_cmd = recon_data['FLAT_DM_CMD'].data

    Nmodes_poked = int(recon_data[0].header['HIERARCH Nmodes_poked']) # can also see recon_data[0].header['RESHAPE']

    Nact =  int(recon_data[0].header['HIERARCH Nact'])  

    N0 = recon_data['FPM_OUT'].data
    #P = np.sqrt( pupil ) # 
    I0 = recon_data['FPM_IN'].data

    # the first image is another reference I0 with FPM IN and flat DM
    poke_imgs = recon_data['SEQUENCE_IMGS'].data[1:].reshape(No_ramps, 140, I0.shape[0], I0.shape[1])
    #poke_imgs = poke_imgs[1:].reshape(No_ramps, 140, I0.shape[0], I0.shape[1])

    plt.figure( ) 
    plt.imshow( np.std( poke_imgs ,axis = (0,1)) )
    plt.colorbar(label = 'std pixels') 
    
    recomended_bad_pixels = np.where( (np.std( poke_imgs ,axis = (0,1)) > 100) + (np.std( poke_imgs ,axis = (0,1)) == 0 ))
    print('recommended bad pixels (high or zero std) at :',recomended_bad_pixels )
    
    if len(bad_pixels[0]) > 0:
        
        bad_pixel_mask = np.ones(I0.shape)
        for ibad,jbad in list(zip(bad_pixels[0], bad_pixels[1])):
            bad_pixel_mask[ibad,jbad] = 0
            
        I0 *= bad_pixel_mask
        N0 *= bad_pixel_mask
        poke_imgs  = poke_imgs * bad_pixel_mask

    a0 = len(ramp_values)//2 - 2 # which poke value (index) do we want to consider for finding region of influence. Pick a value near the center of the ramp (ramp values are from negative to positive) where we are in a linear regime.
    
    if hasattr(active_dm_actuator_filter,'__len__'):
        #is it some form of boolean type?
        bool_test = ( (all(isinstance(x,np.bool_) for x in active_dm_actuator_filter)) or (all(isinstance(x,bool) for x in active_dm_actuator_filter) ) )
        #is it the right length (corresponds to number of DM actuators?) 
        len_test = (len( active_dm_actuator_filter ) == Nact)

        if len_test & bool_test:
            dm_pupil_filt = np.array(active_dm_actuator_filter ) # force to numpy array
        else:
            raise TypeError('active_dm_actuator_filter needs to be list like with boolean entries (numpy or naitive) with length = 140 (corresponding to number of actuators on DM')


    elif active_dm_actuator_filter==None:
        # ========================== !! 1 !! =====================
        #  == Then we let the user define the region of influence on DM where we will fit our models (by defining a threshold for I(epsilon)_max -I_0). This is important because the quality of the fits can go to shit around pupil/DM edges, we only need some good samples around the center to reconstruct what we need, setting this threshold is semi automated here  

        fig,ax= plt.subplots( 4, 4, figsize=(10,10))
        num_pixels = []
        candidate_thresholds = np.linspace(4 * np.std(abs(poke_imgs[a0,:,:,:] - I0)),np.max(abs(poke_imgs[a0,:,:,:] - I0)),16)
        for axx, thresh in zip(ax.reshape(-1),candidate_thresholds):
    
            dm_pupil_filt = thresh < np.array( [np.max( abs( poke_imgs[a0][act] - I0) ) for act in range(140)] ) 
            axx.imshow( get_DM_command_in_2D( dm_pupil_filt ) ) 
            axx.set_title('threshold = {}'.format(round( thresh )),fontsize=12) 
            axx.axis('off')
            num_pixels.append(sum(dm_pupil_filt)) 
            # we could use this to automate threshold decision.. look for where 
            # d num_pixels/ d threshold ~ 0.. np.argmin( abs( np.diff( num_pixels ) )[:10])
        plt.show()

        recommended_threshold = candidate_thresholds[np.argmin( abs( np.diff( num_pixels ) )[2:11]) + 1 ]
        print( f'\n\nrecommended threshold ~ {round(recommended_threshold)} \n(check this makes sense with the graph by checking the colored area is stable around changes in threshold about this value)\n\n')

        pupil_filt_threshold = float(input('input threshold of peak differences'))

        ### <---- THIS FILTER DETERMINES WHERE WE FIT THE MODELS (ONLY FIT WHERE DM HAS GOOD INFLUENCE!)
        dm_pupil_filt = pupil_filt_threshold < np.array( [np.max( abs( poke_imgs[a0][act] - I0) ) for act in range(Nact)] ) 

        if debug:
           plt.figure()
           plt.imshow( get_DM_command_in_2D( dm_pupil_filt ) )
           plt.title('influence region on DM where we will fit intensity models per actuator')
           plt.show()


      
    # ========================== !! 2 !! =====================
    # ======== P2C 

    Sw_x, Sw_y = 3,3 #+- pixels taken around region of peak influence. PICK ODD NUMBERS SO WELL CENTERED!   
    act_img_mask_1x1 = {} #pixel with peak sensitivity to the actuator
    act_img_mask_3x3 = {} # 3x3 region around pixel with peak sensitivity to the actuator
    poor_registration_list = np.zeros(Nact).astype(bool) # list of actuators in control region that have poor registration 


    # I should probably include a threshold filter here - that no registration is made if delta < threshold
    # threshold set to 5 sigma above background (seems to work - but can be tweaked) 
    registration_threshold = 5 * np.mean(np.std(abs(poke_imgs- I0),axis=(0,1)) )
    # how to best deal with actuators that have poor registration ?
    for act_idx in range(Nact):
        delta =  poke_imgs[a0][act_idx] - I0

        mask_3x3 = np.zeros( I0.shape )
        mask_1x1 = np.zeros( I0.shape )
        if dm_pupil_filt[act_idx]: #  if we decided actuator has strong influence on ZWFS image, we 
            peak_delta = np.max( abs(delta) ) 
            if peak_delta > registration_threshold:

                i,j = np.unravel_index( np.argmax( abs(delta) ), I0.shape )
          
                mask_3x3[i-Sw_x-1: i+Sw_x, j-Sw_y-1:j+Sw_y] = 1 # keep centered, 
                mask_1x1[i,j] = 1 
                #mask *= 1/np.sum(mask[i-Sw_x-1: i+Sw_x, j-Sw_y-1:j+Sw_y]) #normalize by #pixels in window 
                act_img_mask_3x3[act_idx] = mask_3x3
                act_img_mask_1x1[act_idx] = mask_1x1
            else:
                poor_registration_list[act_idx] = True
                act_img_mask_3x3[act_idx] = mask_3x3 
                act_img_mask_1x1[act_idx] = mask_1x1 
        else :
            act_img_mask_3x3[act_idx] = mask_3x3 
            act_img_mask_1x1[act_idx] = mask_1x1 
            #act_flag[act_idx] = 0 
    if debug:
        plt.title('masked regions of influence per actuator')
        plt.imshow( np.sum( list(act_img_mask_3x3.values()), axis = 0 ) )
        plt.show()

    # turn our dictionary to a big pixel to command matrix 
    P2C_1x1 = np.array([list(act_img_mask_1x1[act_idx].reshape(-1)) for act_idx in range(Nact)])
    P2C_3x3 = np.array([list(act_img_mask_3x3[act_idx].reshape(-1)) for act_idx in range(Nact)])

    # we can look at filtering a particular actuator in image P2C_3x3[i].reshape(zwfs.get_image().shape)
    # we can also interpolate signals in 3x3 grid to if DM actuator not perfectly registered to center of pixel. 
    


    if debug: 
        im_list = [(I0 - N0) / N0 ]
        xlabel_list = ['x [pixels]']
        ylabel_list = ['y [pixels]']
        title_list = ['']
        cbar_label_list = [r'$\frac{|\psi_C|^2 - |\psi_A|^2}{|\psi_A|^2}$']
        savefig = None# fig_path + f'pupil_FPM_IN-OUT_readout_mode-FULL_t{tstamp}.png'

        nice_heatmap_subplots( im_list , xlabel_list, ylabel_list, title_list, cbar_label_list, fontsize=15, axis_off=True, cbar_orientation = 'bottom', savefig=savefig)

        # check the active region 
        fig,ax = plt.subplots(1,1)
        ax.imshow( get_DM_command_in_2D( dm_pupil_filt  ) )
        ax.set_title('active DM region')
        ax.grid(True, which='minor',axis='both', linestyle='-', color='k' ,lw=3)
        ax.set_xticks( np.arange(12) - 0.5 , minor=True)
        ax.set_yticks( np.arange(12) - 0.5 , minor=True)
        #plt.savefig( fig_path + f'active_DM_region_{tstamp}.png' , bbox_inches='tight', dpi=300) 
        # check the well registered DM region : 
        fig,ax = plt.subplots(1,1)
        ax.imshow( get_DM_command_in_2D( np.sum( P2C_1x1, axis=1 )))
        ax.set_title('well registered actuators')
        ax.grid(True, which='minor',axis='both', linestyle='-', color='k',lw=2 )
        ax.set_xticks( np.arange(12) - 0.5 , minor=True)
        ax.set_yticks( np.arange(12) - 0.5 , minor=True)
  
        #plt.savefig( fig_path + f'poorly_registered_actuators_{tstamp}.png' , bbox_inches='tight', dpi=300) 

        # check poorly registered actuators: 
        fig,ax = plt.subplots(1,1)
        ax.imshow( get_DM_command_in_2D(poor_registration_list) )
        ax.set_title('poorly registered actuators')
        ax.grid(True, which='minor',axis='both', linestyle='-', color='k', lw=2 )
        ax.set_xticks( np.arange(12) - 0.5 , minor=True)
        ax.set_yticks( np.arange(12) - 0.5 , minor=True)

        #plt.savefig( fig_path + f'poorly_registered_actuators_{tstamp}.png', bbox_inches='tight', dpi=300)

        plt.show() 


# ========================== !! 3 !! =====================
    # ======== FITTING 
    # what do we fit ( I - N0 ) / N0 


    param_dict = {}
    cov_dict = {}
    fit_residuals = []
    nofit_list = []
    if debug:
        Nrows = np.ceil( sum( dm_pupil_filt )**0.5).astype(int)
        fig,ax = plt.subplots(Nrows,Nrows,figsize=(20,20))
        axx = ax.reshape(-1)
        for aaa in axx:
            aaa.axis('off')
        j=0 #axx index

    #mean_filtered_pupil = 1/(3*3)  * np.mean( P2C_3x3 @ N0.reshape(-1) )

    for act_idx in range(len(flat_dm_cmd)): 

        #Note that if we use the P2C_3x3 we need to normalize it 1/(3*3) * P2C_3x3
        if dm_pupil_filt[act_idx] * ( ~poor_registration_list)[act_idx]:

            # -- we do this with matrix multiplication using  mask_matrix
            #P_i = np.sum( act_img_mask[act_idx] * pupil ) #Flat DM with FPM OUT 
            #P_i = mean_filtered_pupil.copy() # just consider mean pupil! 
        
            I_i = np.array( [P2C_1x1[act_idx] @ poke_imgs[i][act_idx].reshape(-1) for i in  range(len(ramp_values))] ) #np.array( [np.sum( act_img_mask[act_idx] * poke_imgs[i][act_idx] ) for i in range(len(ramp_values))] ) #spatially filtered sum of intensities per actuator cmds 
            I_0 = P2C_1x1[act_idx] @ I0.reshape(-1) #np.sum( act_img_mask[act_idx] * I0 ) # Flat DM with FPM IN  
            N_0 = P2C_1x1[act_idx] @ N0.reshape(-1) #np.sum( act_img_mask[act_idx] * N0 )
            # ================================
            #   THIS IS OUR MODEL!S=A+B*cos(F*x + mu)  
            #S = (I_i - I_0) / P_i # signal to fit!
            S = I_i  # A 
            #S = (I_i - N_0) / N_0 # signal to fit! <- should I take mean of total pupil?? 
            # THEORETICALLY THIS SIGNAL IS: S = |M0|^2/|A|^2 + M0/|A| * cos(F.c + mu)  
            # ================================

            #re-label and filter to capture best linear range 
            x_data = ramp_values[2:-2].copy()
            y_data = S[2:-2].copy()

            initial_guess = [np.mean(S), (np.max(S)-np.min(S))/2,  15, 2.4]
            #initial_guess = [7, 2, 15, 2.4] #[0.5, 0.5, 15, 2.4]  #A_opt, B_opt, F_opt, mu_opt  ( S = A+B*cos(F*x + mu) )

            try:
                # FIT 
                popt, pcov = curve_fit(Ic_model_constrained, x_data, y_data, p0=initial_guess)

                # Extract the optimized parameters explictly to measure residuals
                A_opt, B_opt, F_opt, mu_opt = popt

                # STORE FITS 
                param_dict[act_idx] = popt
                cov_dict[act_idx] = pcov 
                # also record fit residuals 
                fit_residuals.append( S - Ic_model_constrained(ramp_values, A_opt, B_opt, F_opt, mu_opt) )


                if debug: 

                    axx[j].plot( ramp_values, Ic_model_constrained(ramp_values, A_opt, B_opt, F_opt, mu_opt) ,label=f'fit (act{act_idx})') 
                    axx[j].plot( ramp_values, S ,label=f'measured (act{act_idx})' )
                    #axx[j].set_xlabel( 'normalized DM command')
                    #axx[j].set_ylabel( 'normalized Intensity')
                    axx[j].legend(fontsize=6)
                    #axx[j].set_title(act_idx,fontsize=5)
                    #ins = axx[j].inset_axes([0.15,0.15,0.25,0.25])
                    #ins.imshow(poke_imgs[3][act_idx] )
                    #axx[j].axis('off')
                    j+=1
            except:
                print(f'\n!!!!!!!!!!!!\nfit failed for actuator {act_idx}\n!!!!!!!!!!!!\nanalyse plot to try understand why')
                nofit_list.append( act_idx ) 
                fig1, ax1 = plt.subplots(1,1)
                ax1.plot( ramp_values, S )
                ax1.set_title('could not fit this!') 
                 
       

    if debug:
        #plt.savefig( fig_path + f'cosine_interation_model_fits_ALL_{tstamp}.png' , bbox_inches='tight', dpi=300) 
        plt.show() 

    if debug:
        """ used to buff things out (adding new 0 normal noise variance to samples) 
        Qlst,Wlst,Flst,mulst = [],[],[],[]
        Q_est =  np.array(list( param_dict.values() ))[:, 0]
        W_est = np.array(list( param_dict.values() ))[:, 1] 
        F_est = np.array(list( param_dict.values() ))[:, 2]
        mu_est = np.array(list( param_dict.values() ))[:, 3] 
        for q,w,f,u in param_dict.values():
            Qlst.append( list( q + 0.01*np.mean(Q_est)*np.random.randn(100 ) ) )
            Wlst.append( list( w + 0.01*np.mean(W_est)*np.random.randn(100 ) ) )
            Flst.append( list( f + 0.01*np.mean(F_est)*np.random.randn(100 ) ) )
            mulst.append( list( u + 0.01*np.mean(mu_est)*np.random.randn(100 ) ) )

        #buffcorners = np.array( [list( param_dict.values() ) for _ in range(10)]).reshape(-1,4)
        buffcorners = np.array([np.array(Qlst).ravel(),np.array(Wlst).ravel(), np.array(Flst).ravel(),np.array(mulst).ravel()]).T
        corner.corner( buffcorners , quantiles=[0.16,0.5,0.84], show_titles=True, labels = ['Q [adu]', 'W [adu/cos(rad)]', 'F [rad/cmd]', r'$\mu$ [rad]'] ) 
        """
        corner.corner( np.array(list( param_dict.values() )), quantiles=[0.16,0.5,0.84], show_titles=True, labels = ['Q', 'W', 'F', r'$\mu$'] , range = [(0,2*np.mean(y_data)),(0, 10*(np.max(y_data)-np.min(y_data)) ) , (5,20), (0,6) ] ) #, range = [(2*np.min(S), 102*np.max(S)), (0, 2*(np.max(S) - np.min(S)) ), (5, 20), (-3,3)] ) #['Q [adu]', 'W [adu/cos(rad)]', 'F [rad/cmd]', r'$\mu$ [rad]']
        #plt.savefig( fig_path + f'fitted_Ic_model_parameters_corner_plot_{tstamp}.png', bbox_inches='tight', dpi=300)
        plt.show()
        
    output_fits = fits.HDUList( [] )

    # reference images 
    N0_fits = fits.PrimaryHDU( N0 )
    N0_fits.header.set('EXTNAME','FPM OUT REF')
    N0_fits.header.set('WHAT IS','ref int. with FPM out')

    I0_fits = fits.PrimaryHDU( I0 )
    I0_fits.header.set('EXTNAME','FPM IN REF')
    I0_fits.header.set('WHAT IS','ref int. with FPM in')

    # output fits files 
    P2C_fits = fits.PrimaryHDU( np.array([P2C_1x1, P2C_3x3]) )
    P2C_fits.header.set('EXTNAME','P2C')
    P2C_fits.header.set('WHAT IS','pixel to DM actuator register')
    P2C_fits.header.set('index 0','P2C_1x1') 
    P2C_fits.header.set('index 1','P2C_3x3')    
    
    #fitted parameters
    param_fits = fits.PrimaryHDU( np.array(list( param_dict.values() )) )
    param_fits.header.set('EXTNAME','FITTED_PARAMS')
    param_fits.header.set('COL0','Q [adu]')
    param_fits.header.set('COL1','W [adu/cos(rad)]')
    param_fits.header.set('COL2','F [rad/cmd]')
    param_fits.header.set('COL4','mu [rad]')
    
    if len(nofit_list)!=0:
        for i, act_idx in enumerate(nofit_list):
            param_fits.header.set(f'{i}_fit_fail_act', act_idx)
        
    #covariances
    cov_fits = fits.PrimaryHDU( np.array(list(cov_dict.values())) )
    cov_fits.header.set('EXTNAME','FIT_COV')
    # residuals 
    res_fits = fits.PrimaryHDU( np.array(fit_residuals) )
    res_fits.header.set('EXTNAME','FIT_RESIDUALS')

    #DM regions 
    dm_fit_regions = fits.PrimaryHDU( np.array( [dm_pupil_filt, dm_pupil_filt*(~poor_registration_list), poor_registration_list] ).astype(int) )
    dm_fit_regions.header.set('EXTNAME','DM_REGISTRATION_REGIONS')
    dm_fit_regions.header.set('registration_threshold',registration_threshold)
    dm_fit_regions.header.set('index 0 ','active_DM_region')   
    dm_fit_regions.header.set('index 1 ','well registered actuators') 
    dm_fit_regions.header.set('index 2 ','poor registered actuators') 
 
    for f in [N0_fits, I0_fits, P2C_fits, param_fits, cov_fits,res_fits, dm_fit_regions ]:
        output_fits.append( f ) 

    if savefits!=None:
           
        output_fits.writeto( savefits )  #data_path + 'ZWFS_internal_calibration.fits'

    return( output_fits ) 



def _compute_weighted_frequency(vector):
    # Step 1: Compute the Fourier Transform of the vector
    fft_values = np.fft.fft(vector)
    
    # Step 2: Compute the Power Spectral Density (PSD)
    psd = np.abs(fft_values)**2
    
    # Step 3: Compute the frequencies associated with the PSD
    freqs = np.fft.fftfreq(len(vector))
    
    # Only consider the positive frequencies
    positive_freq_indices = np.where(freqs >= 0)
    freqs = freqs[positive_freq_indices]
    psd = psd[positive_freq_indices]
    
    # Step 4: Compute the weighted frequency
    weighted_freq = np.sum(freqs * psd) / np.sum(psd)
    
    return weighted_freq

def sort_vectors_by_weighted_frequency(vectors):
    """
    sort list of vectors based on the power spectral density weighted 
    frequency of the vectors ordered from vectors having most power at 
    lower frequencies to vectors having most power at higher frequencies. 
    
    Useful for sorting DM basis cmds that may seem un-ordered
    """
    
    # Step 5: Compute the weighted frequency for each vector
    weighted_frequencies = [_compute_weighted_frequency(v) for v in vectors]
    
    # Step 6: Sort the vectors based on the weighted frequency
    sorted_vectors = [v for _, v in sorted(zip(weighted_frequencies, vectors))]
    
    return sorted_vectors



def twoD_Gaussian(xy, amplitude, xo, yo, sigma_x, sigma_y, theta, offset):
    x, y = xy
    xo = float(xo)
    yo = float(yo)    
    a = (np.cos(theta)**2)/(2*sigma_x**2) + (np.sin(theta)**2)/(2*sigma_y**2)
    b = -(np.sin(2*theta))/(4*sigma_x**2) + (np.sin(2*theta))/(4*sigma_y**2)
    c = (np.sin(theta)**2)/(2*sigma_x**2) + (np.cos(theta)**2)/(2*sigma_y**2)
    g = offset + amplitude*np.exp( - (a*((x-xo)**2) + 2*b*(x-xo)*(y-yo)
                            + c*((y-yo)**2)))
    return g.ravel()


def fit_b_pixel_space(I0, theta, image_filter , debug=True): 
    # fit b parameter from the reference fields I0 (FPM IN), N0 (FPM OUT) which should be 2D arrays, theta is scalar estimate of the FPM phase shift 
    # we can use N0 to remove bias/bkg by subtraction 

    x = np.linspace(-I0.shape[0]//2 , I0.shape[0]//2 ,I0.shape[0])
    y = np.linspace(-I0.shape[1]//2 , I0.shape[1]//2 ,I0.shape[1])
    X, Y = np.meshgrid( x, y)
    X_f=X.reshape(-1)[image_filter]
    Y_f=Y.reshape(-1)[image_filter]

    # we normalize by average of I0 over entire image 
    I0_mean = np.mean( I0 ) # we take the median FPM OUT signal inside the pupil (appart from center pixels) 
    data = (I0.reshape(-1)[image_filter]/I0_mean).reshape(-1) #((I0-N0)/N0).reshape(-1)[image_filter] #this is M^2/|A|^2
    initial_guess = (np.nanmax(data),np.mean(x),np.mean(y),np.std(x),np.std(y), 0, 0) 

    # fit it 
    popt, pcov = curve_fit(twoD_Gaussian, (X_f, Y_f), data, p0=initial_guess)

    data_fitted = twoD_Gaussian((X, Y), *popt)  #/ np.mean(I0.reshape(-1)[image_filter])

    # fitted b in pixel space
    bfit = data_fitted.reshape(X.shape)**0.5 / (2*(1-np.cos(theta)))**0.5

    if debug:
        im_list = [I0 * image_filter.reshape(I0.shape), bfit]
        xlabel_list = ['','']
        ylabel_list = ['','']
        title_list = ['','']
        cbar_label_list = ['filtered I0 [adu]', 'fitted b'] 
        savefig = None #fig_path + 'b_fit_internal_cal.png'

        nice_heatmap_subplots( im_list , xlabel_list, ylabel_list, title_list,cbar_label_list, fontsize=15, cbar_orientation = 'bottom', axis_off=True, savefig=savefig)

        plt.show()

    return(bfit) 


def put_b_in_cmd_space( b, zwfs , debug = True):
    # b is usually fitted in WFS pixel space - we need to map this back to the DM command space. 

    # create interpolation function zwfs.row_coords ,  zwfs.col_coords 
    b_interpolator = RegularGridInterpolator((zwfs.row_coords, zwfs.col_coords ), b)
    # interpolate onto zwfs.dm_row_coordinates_in_pixels, zwfs.dm_col_coordinates_in_pixels (this is 12x12 DM grid in pixel space). 
    X, Y = np.meshgrid(zwfs.dm_col_coordinates_in_pixels, zwfs.dm_row_coordinates_in_pixels ) 
    pts = np.vstack([X.ravel(), Y.ravel()]).T 
    b_on_dm_in_pixel_space = b_interpolator( pts ) 

    #plt.figure(); plt.imshow( b_on_dm_in_pixel_space.reshape(12,12) );plt.colorbar(); plt.show()
    
    # drop the corners and flatten - this is our b in command space 
    Nx_act_DM = 12 
    corner_indices = [0, Nx_act_DM-1, Nx_act_DM * (Nx_act_DM-1), -1]
    # put corners to nan on flattened array 
    b_on_dm_in_pixel_space.reshape(-1)[corner_indices] = np.nan
    #drop nan values so we go frp, 144 - 140 length array - OUR b in DM CMD SPACE :) 
    b_in_dm_cmd_space = b_on_dm_in_pixel_space[ np.isfinite(b_on_dm_in_pixel_space) ] 

    if debug:
        plt.figure(); 
        plt.title('fitted b in DM cmd space')
        plt.imshow( get_DM_command_in_2D( b_in_dm_cmd_space ));plt.colorbar(); 
        plt.show()
   
    return( b_in_dm_cmd_space ) 



"""
    # get an estimate for things that should not ideally have spatial variability
    _, _, F_est, mu_est = np.median( list( param_dict.values() ) ,axis = 0)

    # from mu measure theta - our estimated phase shift  
    #   (using result: mu = tan^-1(sin(theta)/(cos(theta)-1) = 0.5*(theta - pi) ) 
    theta_est  = 2*mu_est + np.pi
        
    # THEORETICALLY THIS SIGNAL: S = (I - N0)/N0 = |M0|^2/|A|^2 + 2*M0/|A| * cos(F.c + mu) 
    # Q =  |M0|^2/|A|^2 , W = |M0|/|A|. 
    # therefore M0 = np.sqrt( Q * |A|^2 ) => W = Q
        

    # things that can have spatial variability
    Q_est =  np.array(list( param_dict.values() ))[:, 0]
    W_est = np.array(list( param_dict.values() ))[:, 1] 


    # ========== GET A and B from these ... check against measured values within pupil obstruction and fit gaussian for b within pupil.   

    A_est = np.sqrt( Q_est + np.sqrt( Q_est**2 + W_est**2 ) ) / np.sqrt(2) 

    M_est = W_est /  (np.sqrt(2) * np.sqrt( Q_est + np.sqrt( Q_est**2 - W_est**2 ) )  ) 
    
    b_est = M_est / (2*(1-np.cos(theta_est)))**0.5

    bsamples = np.nan * np.zeros( dm_pupil_filt.shape)
    bsamples[dm_pupil_filt] = b_est

    plt.figure()
    plt.imshow( get_DM_command_in_2D( bsamples )); plt.colorbar()
    plt.show()
"""






