import numpy as np
import matplotlib.pyplot as plt
from pyBaldr import utilities as util
from xaosim.shmlib import shm


# dictionary with depths referenced for beam 2 (1-5 goes big to small)
phasemask_parameters = {  
                        "J5": {"depth":0.474 ,  "diameter":32},
                        "J4": {"depth":0.474 ,  "diameter":36}, 
                        "J3": {"depth":0.474 ,  "diameter":44}, 
                        "J2": {"depth":0.474 ,  "diameter":54},
                        "J1": {"depth":0.474 ,  "diameter":65},
                        "H1": {"depth":0.654 ,  "diameter":68},  
                        "H2": {"depth":0.654 ,  "diameter":53}, 
                        "H3": {"depth":0.654 ,  "diameter":44}, 
                        "H4": {"depth":0.654 ,  "diameter":37},
                        "H5": {"depth":0.654 ,  "diameter":31}
                        }

"""
email from Mike 5/12/24 ("dichroic curves")
optically you have 1380-1820nm (50% points) optically, 
and including the atmosphere it is ~1420-1820nm. 

coldstop is has diameter 2.145 mm
baldr beams (30mm collimating lens) fratio 21.2 at focal plane focused by 254mm OAP
is xmm with 200mm imaging lens
2.145e-3 / ( 2 * 200 / (254 / 21.2 * 30 / 254 ) * 1.56e-6 )
wvl = 1.56um
coldstop_diam  ~ 4.8 lambda/D 
"""


T = 1900 #K lab thermal source temperature 
lambda_cut_on, lambda_cut_off =  1.38, 1.82 # um
wvl = util.find_central_wavelength(lambda_cut_on, lambda_cut_off, T) # central wavelength of Nice setup
mask = "J5"
F_number = 21.2
coldstop_diam = 4.8
mask_diam = 1.22 * F_number * wvl / phasemask_parameters[mask]['diameter']
eta = 0.647/4.82 #~= 1.1/8.2 (i.e. UTs) # ratio of secondary obstruction (UTs)
P, Ic = util.get_theoretical_reference_pupils( wavelength = wvl ,
                                              F_number = F_number , 
                                              mask_diam = mask_diam, 
                                              coldstop_diam=coldstop_diam,
                                              eta = eta, 
                                              diameter_in_angular_units = True, 
                                              get_individual_terms=False, 
                                              phaseshift = util.get_phasemask_phaseshift(wvl=wvl, depth = phasemask_parameters[mask]['depth'], dot_material='N_1405') , 
                                              padding_factor = 6, 
                                              debug= False, 
                                              analytic_solution = False )

############################################
## Plot theoretical intensities on fine grid 
imgs = [P, Ic]
titles=['Clear Pupil', 'ZWFS Pupil']
cbars = ['Intensity', 'Intensity']
xlabel_list, ylabel_list = ['',''], ['','']
util.nice_heatmap_subplots(im_list=imgs ,
                            xlabel_list=xlabel_list, 
                            ylabel_list=ylabel_list, 
                            title_list=titles, 
                            cbar_label_list=cbars, 
                            fontsize=15, 
                            cbar_orientation = 'bottom', 
                            axis_off=True, 
                            vlims=None, 
                            savefig='delme.png')
plt.show()

############################################
## Plot theoretical intensities on CRED1 Detector (12 pixel diameter)
# we can use a clear pupil measurement to interpolate this onto 
# the measured pupil pixels.

# Original grid dimensions from the theoretical pupil
M, N = Ic.shape

m, n = 36, 36  # New grid dimensions (width, height in pixels)
# To center the pupil, set the center at half of the grid size.
x_c, y_c = int(m/2), int(n/2)
# For a 12-pixel diameter pupil, the new pupil radius should be 6 pixels.
new_radius = 6

# Interpolate the theoretical intensity onto the new grid.
detector_intensity = util.interpolate_pupil_to_measurement(P, Ic, M, N, m, n, x_c, y_c, new_radius)

# Plot the interpolated theoretical pupil intensity.
imgs = [detector_intensity]
titles=[ 'Detected\nZWFS Pupil']
cbars = ['Intensity']
xlabel_list, ylabel_list = [''], ['']
util.nice_heatmap_subplots(im_list=imgs ,
                            title_list=titles,
                            xlabel_list=xlabel_list, 
                            ylabel_list=ylabel_list, 
                            cbar_label_list=cbars, 
                            fontsize=15, 
                            cbar_orientation = 'bottom', 
                            axis_off=True, 
                            vlims=None, 
                            savefig='delme2.png')
plt.show()




################
# Plot all of them 

fig, axes = plt.subplots(2, 5, figsize=(15, 6))
axes = axes.flatten()  # Flatten to iterate easily

# Loop over each phasemask and generate synthetic intensity data
for i, (mask, params) in enumerate(phasemask_parameters.items()):
    mask_diam = 1.22 * F_number * wvl / params['diameter']  # Compute mask diameter
    phase_shift = util.get_phasemask_phaseshift(wvl=wvl, depth = phasemask_parameters[mask]['depth'], dot_material='N_1405') , 
    
    P, Ic = util.get_theoretical_reference_pupils( wavelength = wvl ,
                                                F_number = F_number , 
                                                mask_diam = mask_diam, 
                                                coldstop_diam=coldstop_diam,
                                                eta = eta, 
                                                diameter_in_angular_units = True, 
                                                get_individual_terms=False, 
                                                phaseshift = util.get_phasemask_phaseshift(wvl=wvl, depth = phasemask_parameters[mask]['depth'], dot_material='N_1405') , 
                                                padding_factor = 6, 
                                                debug= False, 
                                                analytic_solution = False )

    detector_intensity = util.interpolate_pupil_to_measurement(P, Ic, M, N, m, n, x_c, y_c, new_radius)

    # Plot the results
    im = axes[i].imshow(detector_intensity, cmap='inferno')
    axes[i].set_title(mask, fontsize=20)
    axes[i].axis('off')

# Adjust layout and add colorbar
fig.subplots_adjust(right=0.85)
cbar_ax = fig.add_axes([0.88, 0.15, 0.02, 0.7])
fig.colorbar(im, cax=cbar_ax, label='Intensity')

# Show the final figure
plt.suptitle("ZWFS Theoretical Intensities on CRED1 Detector", fontsize=14)
plt.show()




##### END 













#################### Explore mis-aligned coldstop 


def get_theoretical_reference_pupils_misalgined_coldstop( wavelength = 1.65e-6 ,F_number = 21.2, mask_diam = 1.2, coldstop_diam=4.5, coldstop_misalign=5, eta=0, diameter_in_angular_units = True, get_individual_terms=False, phaseshift = np.pi/2 , padding_factor = 4, debug= True, analytic_solution = True ) :
    """
    get theoretical reference pupil intensities of ZWFS with / without phasemask 
    

    Parameters
    ----------
    wavelength : TYPE, optional
        DESCRIPTION. input wavelength The default is 1.65e-6.
    F_number : TYPE, optional
        DESCRIPTION. The default is 21.2.
    mask_diam : phase dot diameter. TYPE, optional
            if diameter_in_angular_units=True than this has diffraction limit units ( 1.22 * f * lambda/D )
            if  diameter_in_angular_units=False than this has physical units (m) determined by F_number and wavelength
        DESCRIPTION. The default is 1.2.
    coldstop_diam : diameter in lambda / D of focal plane coldstop
    eta : ratio of secondary obstruction radius (r_2/r_1), where r2 is secondary, r1 is primary. 0 meams no secondary obstruction
    diameter_in_angular_units : TYPE, optional
        DESCRIPTION. The default is True.
    get_individual_terms : Type optional
        DESCRIPTION : if false (default) with jsut return intensity, otherwise return P^2, abs(M)^2 , phi + mu
    phaseshift : TYPE, optional
        DESCRIPTION. phase phase shift imparted on input field (radians). The default is np.pi/2.
    padding_factor : pad to change the resolution in image plane. TYPE, optional
        DESCRIPTION. The default is 4.
    debug : TYPE, optional
        DESCRIPTION. Do we want to plot some things? The default is True.
    analytic_solution: TYPE, optional
        DESCRIPTION. use analytic formula or calculate numerically? The default is True.
    Returns
    -------
    Ic, reference pupil intensity with phasemask in 
    P, reference pupil intensity with phasemask out 

    """
    pupil_radius = 1  # Pupil radius in meters

    # Define the grid in the pupil plane
    N = 2**9+1  # for parity (to not introduce tilt) works better ODD!  # Number of grid points (assumed to be square)
    L_pupil = 2 * pupil_radius  # Pupil plane size (physical dimension)
    dx_pupil = L_pupil / N  # Sampling interval in the pupil plane
    x_pupil = np.linspace(-L_pupil/2, L_pupil/2, N)   # Pupil plane coordinates
    y_pupil = np.linspace(-L_pupil/2, L_pupil/2, N) 
    X_pupil, Y_pupil = np.meshgrid(x_pupil, y_pupil)
    

    # Define a circular pupil function
    pupil = (np.sqrt(X_pupil**2 + Y_pupil**2) > eta*pupil_radius) & (np.sqrt(X_pupil**2 + Y_pupil**2) <= pupil_radius)

    # Zero padding to increase resolution
    # Increase the array size by padding (e.g., 4x original size)
    N_padded = N * padding_factor
    if (N % 2) != (N_padded % 2):  
        N_padded += 1  # Adjust to maintain parity
        
    pupil_padded = np.zeros((N_padded, N_padded))
    #start_idx = (N_padded - N) // 2
    #pupil_padded[start_idx:start_idx+N, start_idx:start_idx+N] = pupil

    start_idx_x = (N_padded - N) // 2
    start_idx_y = (N_padded - N) // 2  # Explicitly ensure symmetry

    pupil_padded[start_idx_y:start_idx_y+N, start_idx_x:start_idx_x+N] = pupil

    # Perform the Fourier transform on the padded array (normalizing for the FFT)
    pupil_ft = np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(pupil_padded)))
    
    # Compute the Airy disk scaling factor (1.22 * λ * F)
    airy_scale = 1.22 * wavelength * F_number

    # Image plane sampling interval (adjusted for padding)
    L_image = wavelength * F_number / dx_pupil  # Total size in the image plane
    dx_image_padded = L_image / N_padded  # Sampling interval in the image plane with padding
    

    if diameter_in_angular_units:
        x_image_padded = np.linspace(-L_image/2, L_image/2, N_padded) / airy_scale  # Image plane coordinates in Airy units
        y_image_padded = np.linspace(-L_image/2, L_image/2, N_padded) / airy_scale
    else:
        x_image_padded = np.linspace(-L_image/2, L_image/2, N_padded)  # Image plane coordinates in Airy units
        y_image_padded = np.linspace(-L_image/2, L_image/2, N_padded) 
        
    X_image_padded, Y_image_padded = np.meshgrid(x_image_padded, y_image_padded)

    if diameter_in_angular_units:
        mask = np.sqrt(X_image_padded**2 + Y_image_padded**2) <= mask_diam / 4
    else: 
        mask = np.sqrt(X_image_padded**2 + Y_image_padded**2) <= mask_diam / 4
        
    if coldstop_diam is not None:
        coldmask = np.sqrt((X_image_padded-coldstop_misalign[0])**2 + (Y_image_padded-coldstop_misalign[1])**2) <= coldstop_diam / 4
    else:
        coldmask = np.ones(X_image_padded.shape)

    pupil_ft = np.fft.fft2(np.fft.ifftshift(pupil_padded))  # Remove outer fftshift
    pupil_ft = np.fft.fftshift(pupil_ft)  # Shift only once at the end

    psi_B = coldmask * pupil_ft
                            
    b = np.fft.fftshift( np.fft.ifft2( mask * psi_B ) ) 

    
    if debug: 
        
        psf = np.abs(pupil_ft)**2  # Get the PSF by taking the square of the absolute value
        psf /= np.max(psf)  # Normalize PSF intensity
        
        if diameter_in_angular_units:
            zoom_range = 3  # Number of Airy disk radii to zoom in on
        else:
            zoom_range = 3 * airy_scale 
            
        extent = (-zoom_range, zoom_range, -zoom_range, zoom_range)

        fig,ax = plt.subplots(1,1)
        ax.imshow(psf, extent=(x_image_padded.min(), x_image_padded.max(), y_image_padded.min(), y_image_padded.max()), cmap='gray')
        ax.contour(X_image_padded, Y_image_padded, mask, levels=[0.5], colors='red', linewidths=2, label='phasemask')
        #ax[1].imshow( mask, extent=(x_image_padded.min(), x_image_padded.max(), y_image_padded.min(), y_image_padded.max()), cmap='gray')
        #for axx in ax.reshape(-1):
        #    axx.set_xlim(-zoom_range, zoom_range)
        #    axx.set_ylim(-zoom_range, zoom_range)
        ax.set_xlim(-zoom_range, zoom_range)
        ax.set_ylim(-zoom_range, zoom_range)
        ax.set_title( 'PSF' )
        ax.legend() 
        #ax[1].set_title('phasemask')


    
    # if considering complex b 
    # beta = np.angle(b) # complex argunment of b 
    # M = b * (np.exp(1J*theta)-1)**0.5
    
    # relabelling
    theta = phaseshift # rad , 
    P = pupil_padded.copy() 
    
    if analytic_solution :
        
        M = abs( b ) * np.sqrt((np.cos(theta)-1)**2 + np.sin(theta)**2)
        mu = np.angle((np.exp(1J*theta)-1) ) # np.arctan( np.sin(theta)/(np.cos(theta)-1) ) #
        
        phi = np.zeros( P.shape ) # added aberrations 
        
        # out formula ----------
        #if measured_pupil!=None:
        #    P = measured_pupil / np.mean( P[P > np.mean(P)] ) # normalize by average value in Pupil
        
        Ic = ( P**2 + abs(M)**2 + 2* P* abs(M) * np.cos(phi + mu) ) #+ beta)
        if not get_individual_terms:
            return( P, Ic )
        else:
            return( P, abs(M) , phi+mu )
    else:
        
        # phasemask filter 
        
        T_on = 1
        T_off = 1
        H = T_off*(1 + (T_on/T_off * np.exp(1j * theta) - 1) * mask  ) 
        
        Ic = abs( np.fft.fftshift( np.fft.ifft2( H * psi_B ) ) ) **2 
    
        return( P, Ic )






P, Ic = get_theoretical_reference_pupils_misalgined_coldstop( wavelength = wvl ,
                                              F_number = F_number , 
                                              mask_diam = mask_diam, 
                                              coldstop_diam=coldstop_diam,
                                              coldstop_misalign=[1,0],
                                              eta = eta, 
                                              diameter_in_angular_units = True, 
                                              get_individual_terms=False, 
                                              phaseshift = util.get_phasemask_phaseshift(wvl=wvl, depth = phasemask_parameters[mask]['depth'], dot_material='N_1405') , 
                                              padding_factor = 6, 
                                              debug= False, 
                                              analytic_solution = False )


## Plot theoretical intensities on fine grid 
imgs = [P, Ic]
titles=['Clear Pupil', 'ZWFS Pupil\n'+r'Cold stop decentered by 1$\lambda/D$ (330um)']
cbars = ['Intensity', 'Intensity']
util.nice_heatmap_subplots(im_list=imgs , title_list=titles, cbar_label_list=cbars, fontsize=15, cbar_orientation = 'bottom', axis_off=True, vlims=None, savefig=None)
plt.show()



# Original grid dimensions from the theoretical pupil
M, N = Ic.shape

m, n = 36, 36  # New grid dimensions (width, height in pixels)
# To center the pupil, set the center at half of the grid size.
x_c, y_c = int(m/2), int(n/2)
# For a 12-pixel diameter pupil, the new pupil radius should be 6 pixels.
new_radius = 6

# Interpolate the theoretical intensity onto the new grid.
detector_intensity = util.interpolate_pupil_to_measurement(P, Ic, M, N, m, n, x_c, y_c, new_radius)

# Plot the interpolated theoretical pupil intensity.
imgs = [detector_intensity]
titles=[ 'Detected\nZWFS Pupil']
cbars = ['Intensity']
util.nice_heatmap_subplots(im_list=imgs , title_list=titles, cbar_label_list=cbars, fontsize=15, cbar_orientation = 'bottom', axis_off=True, vlims=None, savefig=None)
plt.show()


