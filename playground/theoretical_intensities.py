import numpy as np
import matplotlib.pyplot as plt
from pyBaldr import utilities as util



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
I think the photon flux-weighted central wavelength is 
also the central wavelength of 1620nm."""

T = 1900 #K lab thermal source temperature 
lambda_cut_on, lambda_cut_off =  1.38, 1.82 # um
wvl = util.find_central_wavelength(lambda_cut_on, lambda_cut_off, T) # central wavelength of Nice setup
mask = "H4"
F_number = 21.2
mask_diam = 1.22 * F_number * wvl / phasemask_parameters[mask]['diameter']
eta = 1/8 # ratio of secondary obstruction (UTs)
P, Ic = util.get_theoretical_reference_pupils( wavelength = wvl ,
                                              F_number = F_number , 
                                              mask_diam = mask_diam, 
                                              eta = eta, 
                                              diameter_in_angular_units = True, 
                                              get_individual_terms=False, 
                                              phaseshift = np.pi/2 , 
                                              padding_factor = 4, 
                                              debug= False, 
                                              analytic_solution = True )

############################################
## Plot theoretical intensities on fine grid 
imgs = [P, Ic]
titles=['Clear Pupil', 'ZWFS Pupil']
cbars = ['Intensity', 'Intensity']
xlabel_list, ylabel_list = ['',''], ['','']
util.nice_heatmap_subplots(im_list=imgs , xlabel_list=xlabel_list, ylabel_list=ylabel_list, title_list=titles, cbar_label_list=cbars, fontsize=15, cbar_orientation = 'bottom', axis_off=True, vlims=None, savefig=None)
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
util.nice_heatmap_subplots(im_list=imgs , title_list=titles,xlabel_list=xlabel_list, ylabel_list=ylabel_list, cbar_label_list=cbars, fontsize=15, cbar_orientation = 'bottom', axis_off=True, vlims=None, savefig=None)
plt.show()
