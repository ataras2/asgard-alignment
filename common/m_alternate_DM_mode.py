import sys
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import argparse
import os
import time
from scipy.ndimage import distance_transform_edt
from scipy.spatial import distance
from math import factorial

#import bmc
sys.path.insert(1,'/opt/Boston Micromachines/lib/Python3/site-packages/')
import bmc
# Dynamically add the path to pyBaldr based on the location of this script
# script_dir = os.path.dirname(os.path.realpath(__file__))
# pyBaldr_path = os.path.join(script_dir, '../')
# sys.path.append(pyBaldr_path)
# from pyBaldr import utilities as util

def get_DM_command_in_2D(cmd,Nx_act=12):
    # function so we can easily plot the DM shape (since DM grid is not perfectly square raw cmds can not be plotted in 2D immediately )
    #puts nan values in cmd positions that don't correspond to actuator on a square grid until cmd length is square number (12x12 for BMC multi-2.5 DM) so can be reshaped to 2D array to see what the command looks like on the DM.
    corner_indices = [0, Nx_act-1, Nx_act * (Nx_act-1), Nx_act*Nx_act]
    cmd_in_2D = list(cmd.copy())
    for i in corner_indices:
        cmd_in_2D.insert(i,np.nan)
    return( np.array(cmd_in_2D).reshape(Nx_act,Nx_act) )



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

def construct_command_basis( basis='Zernike_pinned_edges', number_of_modes = 20, Nx_act_DM = 12, Nx_act_basis = 12, act_offset=(0,0), without_piston=True):
    """
    returns a change of basis matrix M2C to go from modes to DM commands, where columns are the DM command for a given modal basis. e.g. M2C @ [0,1,0,...] would return the DM command for tip on a Zernike basis. Modes are normalized on command space such that <M>=0, <M|M>=1. Therefore these should be added to a flat DM reference if being applied.    

    basis = string of basis to use
    number_of_modes = int, number of modes to create
    Nx_act_DM = int, number of actuators across DM diameter
    Nx_act_basis = int, number of actuators across the active basis diameter
    act_offset = tuple, (actuator row offset, actuator column offset) to offset the basis on DM (i.e. we can have a non-centered basis)

    """

    
   
    # shorter notations
    #Nx_act = DM.num_actuators_width() # number of actuators across diameter of DM.
    #Nx_act_basis = actuators_across_diam
    c = act_offset
    # DM BMC-3.5 is 12x12 missing corners so 140 actuators , we note down corner indicies of flattened 12x12 array.
    corner_indices = [0, Nx_act_DM-1, Nx_act_DM * (Nx_act_DM-1), -1]

    

    if basis == 'Hadamard':
        # BMC multi-3.5 DM is 12x12 with missing corners - not multiple of 2 so cannot apply standard method of construction
        # first 
        # Define a known 12x12 Hadamard matrix
        H_12 = np.array([
            [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
            [1, -1, 1, -1, 1, -1, 1, -1, 1, -1, 1, -1],
            [1, 1, -1, -1, 1, 1, -1, -1, 1, 1, -1, -1],
            [1, -1, -1, 1, 1, -1, -1, 1, 1, -1, -1, 1],
            [1, 1, 1, 1, -1, -1, -1, -1, 1, 1, 1, 1],
            [1, -1, 1, -1, -1, 1, -1, 1, 1, -1, 1, -1],
            [1, 1, -1, -1, -1, -1, 1, 1, 1, 1, -1, -1],
            [1, -1, -1, 1, -1, 1, 1, -1, 1, -1, -1, 1],
            [1, 1, 1, 1, 1, 1, -1, -1, -1, -1, -1, -1],
            [1, -1, 1, -1, 1, -1, -1, 1, -1, 1, -1, 1],
            [1, 1, -1, -1, 1, 1, 1, 1, -1, -1, -1, -1],
            [1, -1, -1, 1, -1, 1, -1, 1, -1, 1, -1, 1]
        ])

        # Generate a 144x144 Hadamard matrix using Kronecker product
        H_144 = np.kron(H_12, H_12)

        # Define the mask to remove corners without affecting the 2D structure
        corner_mask = np.ones((12, 12), dtype=bool)
        corner_mask[0, 0] = False  # top-left corner
        corner_mask[0, -1] = False  # top-right corner
        corner_mask[-1, 0] = False  # bottom-left corner
        corner_mask[-1, -1] = False  # bottom-right corner

        H_140 = []
        for h in H_144:
            H_140.append( h.reshape(12,12)[corner_mask] )
            
        if without_piston:
            M2C = np.array( H_140 )[1:].T #remove piston mode
        else:
            M2C = np.array( H_140 ).T # take transpose to make columns the modes in command space.


    # to deal with
    elif basis == 'Zernike':
        bmcdm_basis_list = []
        if without_piston:
            number_of_modes += 1 # we add one more mode since we dont include piston 

        raw_basis = zernike_basis(nterms=number_of_modes, npix=Nx_act_basis )
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


    elif basis == 'Zernike_pinned_edges':
        """
        designed for BMC multi 3.5 DM, define zernike basis on 10x10 central grid and 
        interpolate outside of this grid, pinning the value of perimeter actuators to the
        inner perimeter value. 
        """
        nact_len = 12 # must be 12
        # alway construct with piston cause we use as a filter, we delete piston later if specified by user
        b0 = construct_command_basis( basis='Zernike', number_of_modes = number_of_modes, Nx_act_DM = nact_len, Nx_act_basis = nact_len, act_offset=(0,0), without_piston=False)

        # put values outside pupil to nan 
        btmp = np.array( [get_DM_command_in_2D( bb ) for bb in b0.T])

        # interpolate
        nan_mask = btmp[0] #get_DM_command_in_2D( b0.T[0] != 0 )
        nan_mask[nan_mask==0] = np.nan

        #nan_mask = np.isnan(nan_mask)
        nearest_index = distance_transform_edt(np.isnan(nan_mask), return_distances=False, return_indices=True)

        # Use the indices to replace NaNs with the nearest non-NaN values

        with_corners = np.array( [ (nan_mask * bb)[tuple(nearest_index)] for bb in btmp[:]] ).T
        #filled_data = get_DM_command_in_2D( new_flat )[tuple(nearest_index)]


        # Define the indices of the corners to be removed
        corners = [(0, 0), (0, nact_len-1), (nact_len-1, 0), (nact_len-1, nact_len-1)]
        # Convert 2D corner indices to 1D
        corner_indices = [i * 12 + j for i, j in corners]

        # Flatten the array
        bmcdm_basis_list = []
        for w in with_corners.T:
            flattened_array = w.flatten()
            filtered_array = np.delete(flattened_array, corner_indices)

            bmcdm_basis_list.append( filtered_array )

        # piston was taken care of in construction of original zernike basis  b0 = construct_command_basis(
        if without_piston:
            control_basis = [np.sqrt( 1/np.nansum( cb**2 ) ) * cb.reshape(-1) for cb in bmcdm_basis_list[1:]]
        else:
            control_basis = [np.sqrt( 1/np.nansum( cb**2 ) ) * cb.reshape(-1) for cb in bmcdm_basis_list[:]]
        M2C = np.array( control_basis ).T

             
    elif basis == 'fourier':
        # NOT TESTED YET ON REAL DM!! 
        if without_piston:
            number_of_modes += 1 # we add one more mode since we dont include piston 

        # NOTE BECAUSE WE HAVE N,M DIMENSIONS WE NEED TO ROUND UP TO SQUARE NUMBER THE MIGHT NOT = EXACTLY number_of_modes
        n = round( number_of_modes**0.5 ) + 1 # number of modes = (n-1)*(m-1) , n=m => (n-1)**2 
        control_basis_dict  = develop_Fourier_basis( n, n ,P = 2 * Nx_act_DM, Nx = Nx_act_DM, Ny = Nx_act_DM )
        
        # create raw basis as ordered list from our dictionary
        raw_basis = []
        for i in range( n-1 ):
            for j in np.arange( i , n-1 ):
                if i==j:
                    raw_basis.append( control_basis_dict[i,i] )
                else:
                    raw_basis.append( control_basis_dict[i,j] ) # get either side of diagonal 
                    raw_basis.append( control_basis_dict[j,i] )
                    
        
        bmcdm_basis_list = []
        for i,B in enumerate(raw_basis):
            B = B.reshape(-1)
            B[corner_indices] = np.nan
            bmcdm_basis_list.append( B[np.isfinite(B)] )
        # flatten & normalize each basis cmd 
        # <M|M> = 1
        if without_piston:
            control_basis = [np.sqrt( 1/np.nansum( cb**2 ) ) * cb.reshape(-1) for cb in bmcdm_basis_list[1:]] #remove piston mode
        else:
            control_basis = [np.sqrt( 1/np.nansum( cb**2 ) ) * cb.reshape(-1) for cb in bmcdm_basis_list]# take transpose to make columns the modes in command space.
        M2C = np.array( control_basis ).T 

    elif basis == 'fourier_pinned_edges':
        """
        designed for BMC multi 3.5 DM, define zernike basis on 10x10 central grid and 
        interpolate outside of this grid, pinning the value of perimeter actuators to the
        inner perimeter value. 
        """
        n = round( number_of_modes**0.5 ) + 1 # number of modes = (n-1)*(m-1) , n=m => (n-1)**2 
        actlen_tmp = 10 # must be 10 for this option! we then calculate perimeter values here! 
        control_basis_dict  = develop_Fourier_basis( n, n ,P = 2 * actlen_tmp, Nx = actlen_tmp, Ny = actlen_tmp )
                
        # create raw basis as ordered list from our dictionary
        raw_basis = []
        for i in range( n-1 ):
            for j in np.arange( i , n-1 ):
                if i==j:
                    raw_basis.append( control_basis_dict[i,i] )
                else:
                    raw_basis.append( control_basis_dict[i,j] ) # get either side of diagonal 
                    raw_basis.append( control_basis_dict[j,i] )
                    
        # pin_outer_actuators_to_inner requires 10x10 input!!! creates 12x12 with missing corner pinning outer actuators 
        bmcdm_basis_list = np.array( [pin_outer_actuators_to_inner_diameter(bb.reshape(-1)) for bb in np.array( raw_basis)] )

        # <M|M> = 1
        if without_piston:
            control_basis = [np.sqrt( 1/np.nansum( cb**2 ) ) * cb.reshape(-1) for cb in bmcdm_basis_list[1:]] #remove piston mode
        else:
            control_basis = [np.sqrt( 1/np.nansum( cb**2 ) ) * cb.reshape(-1) for cb in bmcdm_basis_list]# take transpose to make columns the modes in command space.
        
        M2C = np.array( control_basis ).T 


    elif basis == 'Zonal': 
        #hardcoded for BMC multi3.5 DM (140 actuators)
        M2C = np.eye( 140 ) # we just consider this over all actuators (so defaults to 140 modes) 
        # we filter zonal basis in the eigenvectors of the control matrix. 
    
    elif basis == 'Zonal_pinned_edges':
        # pin edges of actuator
        b = np.eye(100) #
        bmcdm_basis_list = np.array( [pin_outer_actuators_to_inner_diameter(bb) for bb in b.T] )
        # <M|M> = 1
        control_basis = np.array( [np.sqrt( 1/np.nansum( cb**2 ) ) * cb.reshape(-1) for cb in bmcdm_basis_list] )

        M2C = np.array( control_basis ).T

    else:
        raise TypeError( ' input basis name invalid. Try: "Zonal", "Zonal_pinned_edges", "Zernike", "Zernike_pinned_edges", "fourier", "fourier_pinned_edges"  etc ')
    
    
    return(M2C)


def get_tip_tilt_vectors( dm_model='bmc_multi3.5' ,nact_len=12):
    tip = np.array([[n for n in np.linspace(-1,1,nact_len)] for _ in range(nact_len)])
    tilt = tip.T
    if dm_model == 'bmc_multi3.5':
        # Define the indices of the corners to be removed
        corners = [(0, 0), (0, nact_len-1), (nact_len-1, 0), (nact_len-1, nact_len-1)]
        # Convert 2D corner indices to 1D
        corner_indices = [i * 12 + j for i, j in corners]

        # remove corners
        tip_tilt_list = []
        for i,B in enumerate([tip,tilt]):
            B = B.reshape(-1)
            B[corner_indices] = np.nan
            tip_tilt_list.append( B[np.isfinite(B)] )
        
        tip_tilt = np.array( [np.sqrt( 1/np.nansum( cb**2 ) ) * cb.reshape(-1) for cb in tip_tilt_list] ).T

    else:
        tip_tilt = np.array( [np.sqrt( 1/np.nansum( cb**2 ) ) * cb.reshape(-1) for cb in [tip.reshape(-1),tilt.reshape(-1)]] ).T

    return( tip_tilt ) 


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






def noll_indices(j):
    """
    copied from taken from pyzelda (2024) 
    # https://github.com/avigan/pyZELDA/blob/master/pyzelda/utils/zernike.py
    
    Convert from 1-D to 2-D indexing for Zernikes or Hexikes.

    Parameters
    ----------
    j : int
        Zernike function ordinate, following the convention of Noll et al. JOSA 1976.
        Starts at 1.

    """

    if j < 1:
        raise ValueError("Zernike index j must be a positive integer.")

    # from i, compute m and n
    # I'm not sure if there is an easier/cleaner algorithm or not.
    # This seems semi-complicated to me...

    # figure out which row of the triangle we're in (easy):
    n = int(np.ceil((-1 + np.sqrt(1 + 8 * j)) / 2) - 1)
    if n == 0:
        m = 0
    else:
        nprev = (n + 1) * (n + 2) / 2  # figure out which entry in the row (harder)
        # The rule is that the even Z obtain even indices j, the odd Z odd indices j.
        # Within a given n, lower values of m obtain lower j.

        resid = int(j - nprev - 1)

        if _is_odd(j):
            sign = -1
        else:
            sign = 1

        if _is_odd(n):
            row_m = [1, 1]
        else:
            row_m = [0]

        for i in range(int(np.floor(n / 2.))):
            row_m.append(row_m[-1] + 2)
            row_m.append(row_m[-1])

        m = row_m[resid] * sign

    return n, m

def _is_odd(integer):
    """Helper for testing if an integer is odd by bitwise & with 1."""
    return integer & 1

def zernike_basis(nterms=15, npix=512, rho=None, theta=None, **kwargs):
    """
    copied from taken from pyzelda (2024) 
    # https://github.com/avigan/pyZELDA/blob/master/pyzelda/utils/zernike.py
    
    Return a cube of Zernike terms from 1 to N each as a 2D array
    showing the value at each point. (Regions outside the unit circle on which
    the Zernike is defined are initialized to zero.)

    Parameters
    -----------
    nterms : int, optional
        Number of Zernike terms to return, starting from piston.
        (e.g. ``nterms=1`` would return only the Zernike piston term.)
        Default is 15.
    npix: int
        Desired pixel diameter for circular pupil. Only used if `rho`
        and `theta` are not provided.
    rho, theta : array_like
        Image plane coordinates. `rho` should be 0 at the origin
        and 1.0 at the edge of the circular pupil. `theta` should be
        the angle in radians.

    Other parameters are passed through to `poppy.zernike.zernike`
    and are documented there.
    """
    if rho is not None and theta is not None:
        # both are required, but validated in zernike1
        shape = rho.shape
        use_polar = True
    elif (theta is None and rho is not None) or (theta is not None and rho is None):
        raise ValueError("If you provide either the `theta` or `rho` input array, you must "
                             "provide both of them.")

    else:
        shape = (npix, npix)
        use_polar = False

    zern_output = np.zeros((nterms,) + shape)

    if use_polar:
        for j in range(nterms):
            zern_output[j] = zernike1(j + 1, rho=rho, theta=theta, **kwargs)
    else:
        for j in range(nterms):
            zern_output[j] = zernike1(j + 1, npix=npix, **kwargs)
    return zern_output



def zernike1(j, **kwargs):
    """ 
    copied from taken from pyzelda (2024) 
    # https://github.com/avigan/pyZELDA/blob/master/pyzelda/utils/zernike.py
    
    
    Return the Zernike polynomial Z_j for pupil points {r,theta}.

    For this function the desired Zernike is specified by a single index j.
    See zernike for an equivalent function in which the polynomials are
    ordered by two parameters m and n.

    Note that there are multiple contradictory conventions for labeling Zernikes
    with one single index. We follow that of Noll et al. JOSA 1976.

    Parameters
    ----------
    j : int
        Zernike function ordinate, following the convention of
        Noll et al. JOSA 1976

    Additional arguments are defined as in `poppy.zernike.zernike`.

    Returns
    -------
    zern : 2D numpy array
        Z_j evaluated at each (rho, theta)
    """
    n, m = noll_indices(j)
    return zernike(n, m, **kwargs)


def R(n, m, rho):
    """
    copied from taken from pyzelda (2024) 
    # https://github.com/avigan/pyZELDA/blob/master/pyzelda/utils/zernike.py
    
    Compute R[n, m], the Zernike radial polynomial

    Parameters
    ----------
    n, m : int
        Zernike function degree
    rho : array
        Image plane radial coordinates. `rho` should be 1 at the desired pixel radius of the
        unit circle
    """

    m = int(np.abs(m))
    n = int(np.abs(n))
    output = np.zeros(rho.shape)
    if _is_odd(n - m):
        return 0
    else:
        for k in range(int((n - m) / 2) + 1):
            coef = ((-1) ** k * factorial(round( n - k)) /
                    (factorial(k) * factorial( round( (n + m) / 2. - k) ) * factorial( round( (n - m) / 2. - k))) )
            output += coef * rho ** (n - 2 * k)
        return output


def zernike(n, m, npix=100, rho=None, theta=None, outside=np.nan,
            noll_normalize=True):
    """
    copied from taken from pyzelda (2024) 
    # https://github.com/avigan/pyZELDA/blob/master/pyzelda/utils/zernike.py
    
    Return the Zernike polynomial Z[m,n] for a given pupil.

    For this function the desired Zernike is specified by 2 indices m and n.
    See zernike1 for an equivalent function in which the polynomials are
    ordered by a single index.

    You may specify the pupil in one of two ways:
     zernike(n, m, npix)       where npix specifies a pupil diameter in pixels.
                               The returned pupil will be a circular aperture
                               with this diameter, embedded in a square array
                               of size npix*npix.
     zernike(n, m, rho=r, theta=theta)    Which explicitly provides the desired pupil coordinates
                               as arrays r and theta. These need not be regular or contiguous.

    The expressions for the Zernike terms follow the normalization convention
    of Noll et al. JOSA 1976 unless the `noll_normalize` argument is False.

    Parameters
    ----------
    n, m : int
        Zernike function degree
    npix: int
        Desired diameter for circular pupil. Only used if `rho` and
        `theta` are not provided.
    rho, theta : array_like
        Image plane coordinates. `rho` should be 0 at the origin
        and 1.0 at the edge of the circular pupil. `theta` should be
        the angle in radians.
    outside : float
        Value for pixels outside the circular aperture (rho > 1).
        Default is `np.nan`, but you may also find it useful for this to
        be 0.0 sometimes.
    noll_normalize : bool
        As defined in Noll et al. JOSA 1976, the Zernike definition is
        modified such that the integral of Z[n, m] * Z[n, m] over the
        unit disk is pi exactly. To omit the normalization constant,
        set this to False. Default is True.

    Returns
    -------
    zern : 2D numpy array
        Z(m,n) evaluated at each (rho, theta)
    """
    if not n >= m:
        raise ValueError("Zernike index m must be >= index n")
    if (n - m) % 2 != 0:
        print("Radial polynomial is zero for these inputs: m={}, n={} "
                  "(are you sure you wanted this Zernike?)".format(m, n))

    if theta is None and rho is None:
        x = (np.arange(npix, dtype=np.float64) - (npix - 1) / 2.) / ((npix - 1) / 2.)
        y = x
        xx, yy = np.meshgrid(x, y)

        rho = np.sqrt(xx ** 2 + yy ** 2)
        theta = np.arctan2(yy, xx)
    elif (theta is None and rho is not None) or (theta is not None and rho is None):
        raise ValueError("If you provide either the `theta` or `rho` input array, you must "
                             "provide both of them.")

    if not np.all(rho.shape == theta.shape):
        raise ValueError('The rho and theta arrays do not have consistent shape.')

    aperture = np.ones(rho.shape)
    aperture[np.where(rho > 1)] = 0.0  # this is the aperture mask

    if m == 0:
        if n == 0:
            zernike_result = aperture
        else:
            norm_coeff = np.sqrt(n + 1) if noll_normalize else 1
            zernike_result = norm_coeff * R(n, m, rho) * aperture
    elif m > 0:
        norm_coeff = np.sqrt(2) * np.sqrt(n + 1) if noll_normalize else 1
        zernike_result = norm_coeff * R(n, m, rho) * np.cos(np.abs(m) * theta) * aperture
    else:
        norm_coeff = np.sqrt(2) * np.sqrt(n + 1) if noll_normalize else 1
        zernike_result = norm_coeff * R(n, m, rho) * np.sin(np.abs(m) * theta) * aperture

    zernike_result[np.where(rho > 1)] = outside
    return zernike_result





def pin_outer_actuators_to_inner_diameter(inner_command):
    """
    input a basis defined on 10x10 grid and this will convert it to a
    12x12 grid without corners (BMC multi3.5 DM geometry) with the outer
    perimeter actuators pinned to the inner perimeter value
    """
    if len(inner_command) != 100:
        raise ValueError("Input command must be of length 100")

    inner_command = np.array(inner_command).reshape(10, 10)
    
    # Initialize a 12x12 grid with zeros
    command_140 = np.zeros((12, 12))

    # Map the inner 10x10 command to the corresponding position in the 12x12 grid
    command_140[1:11, 1:11] = inner_command

    # Set the perimeter actuators equal to the inner adjacent values
    # Top and bottom rows
    command_140[0, 1:11] = command_140[1, 1:11]
    command_140[11, 1:11] = command_140[10, 1:11]

    # Left and right columns
    command_140[1:11, 0] = command_140[1:11, 1]
    command_140[1:11, 11] = command_140[1:11, 10]

    # Corners (set these to zero since they are missing actuators)
    corners = [(0, 0), (0, 11), (11, 0), (11, 11)]
    for corner in corners:
        command_140[corner] = 0

    # Flatten the 12x12 grid to get the final 140-length command
    command_140_flat = command_140.flatten()

    # Remove the corner actuators (i.e., elements 0, 11, 132, 143)
    indices_to_remove = [0, 11, 132, 143]
    command_140_flat = np.delete(command_140_flat, indices_to_remove)

    return command_140_flat.tolist()

def pin_to_nearest_registered_with_missing_corners(dm_shape, missing_corners, registered_indices):
    """
    Pins non-registered actuators to the closest registered actuator, excluding missing corners.

    Parameters:
    - dm_shape: Tuple (rows, cols) representing the DM grid, e.g., (12, 12).
    - missing_corners: List of indices (in the flattened array) of missing corners.
    - registered_indices: 1D array of indices corresponding to actuators with registered values.

    Returns:
    - basis: 2D array (dm_shape[0] * dm_shape[1] - len(missing_corners), len(registered_indices))
             where each non-registered actuator is pinned to its closest registered actuator.
    """
    # Create the full DM grid with flattened indices
    flattened_size = dm_shape[0] * dm_shape[1]
    
    # Generate 2D coordinates for each point on the grid
    grid_coords = np.array(np.unravel_index(np.arange(flattened_size), dm_shape)).T
    
    # Remove missing corners from the grid and flatten the remaining actuators
    valid_indices = np.setdiff1d(np.arange(flattened_size), missing_corners)
    valid_coords = grid_coords[valid_indices]

    # Extract coordinates of the registered actuators
    registered_coords = grid_coords[registered_indices]
    
    # Initialize the basis matrix for valid actuators
    basis = np.zeros((len(valid_indices), len(registered_indices)))
    
    # For each valid actuator, find the closest registered actuator
    for idx, valid_idx in enumerate(valid_indices):
        if valid_idx in registered_indices:
            # If the actuator is registered, set its basis vector to be identity
            basis[idx, registered_indices == valid_idx] = 1.0
        else:
            # If the actuator is not registered, pin it to the nearest registered actuator
            distances = distance.cdist([grid_coords[valid_idx]], registered_coords)
            nearest_idx = np.argmin(distances)
            # Pin to the nearest registered actuator
            basis[idx, nearest_idx] = 1.0
    
    #<m|m>=1
    basis_norm = np.array( [b/np.sum(b**2)**0.5 for b in basis.T] ).T
    
    
    return basis_norm



def apply_oscillating_mode(beam,  basis_name,  mode, speed, strength, duration, plot_shape=False, DMshapes_path = 'DMShapes/'):
    """Apply oscillating mode to the DM at a specified speed and strength."""
    ## >>>> ADAM - YOU WILL NEED TO UPDATE THIS DICTIONARY WITH THE CORRECT SERIAL NUMBERS FOR YOUR DMs <<<<<< 
    # best put all in a standard json file and load it in
    DM_serial_number_dict = {'1':'17DW019#122', '2': '17DW019#122', '3': '17DW019#122', '4':'17DW019#122'}  

    SIMULATION = False
    # Initialize deformable mirror
    if SIMULATION:
        dm = {}#bmc.BmcDm()
        dm_err_flag = 0#dm.open_dm(DM_serial_number_dict[beam])
    else:
        dm = bmc.BmcDm()
        dm_err_flag = dm.open_dm(DM_serial_number_dict[beam])
        
    flatdm = pd.read_csv(DMshapes_path + '{}_FLAT_MAP_COMMANDS.csv'.format(DM_serial_number_dict[beam]), header=None)[0].values
    
    # Construct the basis of modes
    # options are: ['Hadamard', "Zonal", "Zonal_pinned_edges", "Zernike", "Zernike_pinned_edges", "fourier", "fourier_pinned_edges"]
    basis = construct_command_basis(basis=basis_name, number_of_modes=int(mode)+10, Nx_act_DM=12, Nx_act_basis=12, act_offset=(0,0), without_piston=True)
    
    if dm_err_flag != 0:
        print(f"Error opening DM: {dm_err_flag}")
        sys.exit(1)

    # Ensure mode is within valid range
    if mode >= basis.shape[1] or mode < 0:
        print(f"Mode '{mode}' out of range. Please choose an integer between 0 and {basis.shape[1]-1}.")
        sys.exit(1)

    selected_mode = basis.T[mode]

    print(f"Oscillating mode {mode} from basis '{basis_name}' at {speed} Hz with strength {strength} on beam {beam} for {duration} seconds.")
    
    # Start the oscillation loop
    plt.ion()
    start_time = time.time()
    while (time.time() - start_time) < duration:
        current_time = time.time() - start_time
        dm_command = flatdm + strength * selected_mode * np.sin(2 * np.pi * speed * current_time)
        if not SIMULATION:
            dm.send_data(dm_command)
        
        if plot_shape:
            plt.clf()
            plt.imshow(get_DM_command_in_2D(dm_command), vmax = 0.5 + 1.2 * strength, vmin = 0.5 - 1.2 * strength)
            plt.colorbar(label='DM command')
            plt.pause(0.01)
            #plt.show()
        time.sleep(1.0 / (speed * 10))  # Control loop timing for smooth oscillation - designed for slow oscillations! 


if __name__ == "__main__":
    # Set up argument parser
    parser = argparse.ArgumentParser(description="Oscillate a mode on the DM")
    parser.add_argument("-beam", type=str, required=True, help="Beam identifier")
    parser.add_argument("-basis", type=str, required=True, help="Basis name for the DM modes. Options are [Hadamard, Zonal, Zonal_pinned_edges, Zernike, Zernike_pinned_edges, fourier, fourier_pinned_edges]")
    parser.add_argument("-mode", type=int, required=True, help="Mode index to apply from the basis")
    parser.add_argument("-speed", type=float, required=True, help="Speed of oscillation in Hz")
    parser.add_argument("-strength", type=float, required=True, help="Strength of the oscillation")
    parser.add_argument("-duration", type=float, required=True, help="Duration of oscillation in seconds")
    parser.add_argument("-plot_shape", type=bool, required=False, default=False, help="Do you want to plot the shape during oscillation?")
    parser.add_argument("-DMshapes_path", type=str, required=False,default='DMShapes/',  help="Where do we find the DM shape files?")

    # Parse arguments
    args = parser.parse_args()

    # Run the main function
    apply_oscillating_mode(args.beam, args.basis, args.mode, args.speed, args.strength,\
        args.duration,  args.plot_shape , args.DMshapes_path )



# DM_serial_number_dict = {'1':'17DW019#122', '2': '17DW019#122', '3': '17DW019#122', '4':'17DW019#122'}  # Adjust serial number as needed
# DMshapes_path = 'DMShapes/'

# b = construct_command_basis( basis='fourier_pinned_edges', number_of_modes = 40, Nx_act_DM = 12, Nx_act_basis = 12, act_offset=(0,0), without_piston=True)

# plt.figure()
# plt.imshow( get_DM_command_in_2D( b.T[0] ) )
# plt.show()

#python playground/m_alternate_DM_mode.py -beam 1 -mode 5 -speed 1.0 -strength 0.1 -duration 10 -basis 'fourier_pinned_edges'

#apply_oscillating_mode(beam='1', mode=3, speed=2, strength=0.1, duration=10, basis_name='fourier', plot_shape=False, DMshapes_path = 'DMShapes/')