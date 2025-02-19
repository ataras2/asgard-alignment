import sys
import os
import numpy as np
import fnmatch
import matplotlib.pyplot as plt
import json
import pandas as pd
import datetime
from astropy.io import fits
from scipy.ndimage import label, find_objects, median_filter
from scipy.stats import linregress
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from fpdf import FPDF
import argparse
#sys.path.insert(1, "/Users/bencb/Documents/ASGARD/BaldrApp" )
script_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(script_dir)
import common.DM_registration as DM_registration
from pyBaldr import utilities as util

# to use plotting when remote sometimes X11 forwarding is bogus.. so use this: 
import matplotlib 
matplotlib.use('Agg')

# default paths from one file
with open( "config_files/file_paths.json") as f:
    default_path_dict = json.load(f)

default_data_calibration_path =  default_path_dict["baldr_calibration_data"] # default path to look for most recent calibration files. HARD CODED 
ramp_file_pattern = 'calibration_Zonal*.fits' # pattern to ffind the most recent pokeramp calibration file 
kolmogorov_file_pattern = 'kolmogorov_calibration*.fits'# pattern to ffind the most recent kolmogorov calibration file (if specified)

#ipython calibration/baldr_calibration.py /home/heimdallr/data/stability_tests/calibration_27-11-2024T17.31.28.fits --beam 2

def get_row_col(actuator_index):
    # Function to get row and column for a given actuator index (for plotting)
    rows, cols = 12, 12

    # Missing corners
    missing_corners = [(0, 0), (0, 11), (11, 0), (11, 11)]

    # Create a flattened index map for valid positions
    valid_positions = [(r, c) for r in range(rows) for c in range(cols) if (r, c) not in missing_corners]

    if actuator_index < 0 or actuator_index >= len(valid_positions):
        raise ValueError(f"Invalid actuator index: {actuator_index}")
    return valid_positions[actuator_index]


def DM_actuator_mosaic_plot( xx, yy , filter_crosscoupling = False ):
    """
    xx and yy are 2D array like, rows are samples, columns are actuators.
    plots x vs y for each actuator in a mosaic plot with the same grid layout 
    as the BMC multi-3.5 DM.   
    """
    fig, axes = plt.subplots(12, 12, figsize=(10, 10), sharex=True, sharey=True)
    fig.tight_layout(pad=2.0) 
    for axx in axes.reshape(-1):
        axx.axis('off')

    for act in range(140):

        # Select data for the current actuator and label data
        x = xx[:, act]
        y = yy[:, act]

        if filter_crosscoupling:
            filt = x != 0
        else:
            filt = None
            

        ax = axes[get_row_col(act)]
        # Data points
        ax.plot(x[filt], y[filt], '.', label='Data')
        # Plot setup
        ax.set_xlabel([]) 
        ax.set_ylabel([]) 
        ax.set_title(f'act#{act+1}')
        plt.grid(True)

    return( fig, axes )




def plot_eigenmodes( IM, M2C, save_path = None ):

    #tstamp = datetime.datetime.now().strftime("%d-%m-%YT%H.%M.%S")

    U,S,Vt = np.linalg.svd( IM, full_matrices=True)

    #singular values
    plt.figure() 
    plt.semilogy(S) #/np.max(S))
    #plt.axvline( np.pi * (10/2)**2, color='k', ls=':', label='number of actuators in pupil')
    plt.legend() 
    plt.xlabel('mode index')
    plt.ylabel('singular values')
    if save_path is not None:
        plt.savefig(save_path +  f'singularvalues.png', bbox_inches='tight', dpi=200)
    plt.show()

    # THE IMAGE MODES 
    #M2C_0 = np.eye(140)
    n_row = round( np.sqrt( M2C.shape[0]) ) - 1
    fig,ax = plt.subplots(n_row  ,n_row ,figsize=(15,15))
    plt.subplots_adjust(hspace=0.1,wspace=0.1)
    for i,axx in enumerate(ax.reshape(-1)):
        # we filtered circle on grid, so need to put back in grid
        #tmp =  zwfs_ns.pupil_regions.pupil_filt.copy()
        #vtgrid = np.zeros(I0.shape)
        #vtgrid[tmp] = Vt[i]
        r1,r2,c1,c2 = 10,-10,10,-10
        axx.imshow( Vt[i].reshape( poke_imgs_cropped.shape[2], poke_imgs_cropped.shape[3] ) ) #cp_x2-cp_x1,cp_y2-cp_y1) )
        #axx.set_title(f'\n\n\nmode {i}, S={round(S[i]/np.max(S),3)}',fontsize=5)
        #
        axx.text( 10,10, f'{i}',color='w',fontsize=4)
        axx.text( 10,20, f'S={round(S[i]/np.max(S),3)}',color='w',fontsize=4)
        axx.axis('off')
        #plt.legend(ax=axx)
    plt.tight_layout()
    if save_path is not None:
        plt.savefig(save_path + f'det_eignmodes.png',bbox_inches='tight',dpi=200)
    plt.show()


    # NOTE: if not zonal (modal) i might need M2C to get this to dm space 
    # if zonal M2C is just identity matrix. 
    fig,ax = plt.subplots(n_row, n_row, figsize=(15,15))
    plt.subplots_adjust(hspace=0.1,wspace=0.1)
    for i,axx in enumerate(ax.reshape(-1)):
        axx.imshow( util.get_DM_command_in_2D( M2C @ U.T[i] ) )
        #axx.set_title(f'mode {i}, S={round(S[i]/np.max(S),3)}')
        axx.text( 1,2,f'{i}',color='w',fontsize=6)
        axx.text( 1,3,f'S={round(S[i]/np.max(S),3)}',color='w',fontsize=6)
        axx.axis('off')
        #plt.legend(ax=axx)

    plt.tight_layout()
    if save_path is not None:
        plt.savefig(save_path +  f'dm_eignmodes.png',bbox_inches='tight',dpi=200)
    plt.show()



def get_most_recent_file(directory, pattern):
    most_recent_file = None
    most_recent_time = None

    # Walk through the directory and its subdirectories
    for root, _, files in os.walk(directory):
        for filename in files:
            if fnmatch.fnmatch(filename, pattern):  # Match the file pattern
                filepath = os.path.join(root, filename)
                file_time = os.path.getmtime(filepath)  # Get last modified time
                
                if most_recent_time is None or file_time > most_recent_time:
                    most_recent_time = file_time
                    most_recent_file = filepath

    if most_recent_file:
        print(f"Most recent file: {most_recent_file}")
        print(f"Last modified time: {datetime.datetime.fromtimestamp(most_recent_time)}")
    else:
        print("No files matched the given pattern.")

    return most_recent_file


def parse_arguments():
    # Define the argument parser
    parser = argparse.ArgumentParser(description="Parse command-line arguments for calibration script.")
    
    # Obligatory argument
    parser.add_argument(
        'ramp_file',
        type=str,
        help="Path to the ramp file (obligatory). If you want to use the most recent one input 'recent'"
    )
    
    # Optional arguments with defaults
    parser.add_argument(
        '--kol_file',
        type=str,
        default=None,
        help="Path to the fits file with Kolmogorov phasescreen on DM. Optional. Default: %(default)s. If you want to use the most recent one input 'recent'"
    )
    parser.add_argument(
        '--beam',
        type=int,
        default=2,
        help="Beam number. Default: %(default)s"
    )
    parser.add_argument(
        '--write_report',
        type=bool,
        default=True,
        help="Write a PDF report that provides analytics on calibration results. Default: %(default)s"
    )

    parser.add_argument(
        '--a',
        type=int,
        default=2,
        help="Amplitude index for calculating +/- around flat when calibrating DM/detector transform. Default: %(default)s"
    )
    parser.add_argument(
        '--signal_method',
        type=str,
        default='I-I0/N0',
        help="Signal method. Default: %(default)s"
    )
    parser.add_argument(
        '--control_method',
        type=str,
        default='zonal_linear',
        help="Control method. Default: %(default)s"
    )

    parser.add_argument(
        '--output_config_filename',
        type=str,
        default=f'baldr_transform_dict_beam{2}_{tstamp}.json',
        help="Output configuration filename. Default: %(default)s"
    )
    parser.add_argument(
        '--output_report_dir',
        type=str,
        default=f'/home/asg/Progs/repos/asgard-alignment/calibration/reports/{tstamp_rough}/',
        help="Output directory for calibration reports. Default: %(default)s"
    )
    parser.add_argument(
        '--fig_path',
        type=str,
        default=f'/home/asg/Progs/repos/asgard-alignment/calibration/reports/{tstamp_rough}/figures/',
        help="Path for saving figures. Default: %(default)s"
    )
    
    return parser.parse_args()


# hard coded - this could become a json file
# baldr_pupil_regions = {
#     4:(31, 85, 208, 256), 
#     3:(120, 174, 160, 214), 
#     2:(211, 270, 151, 210), 
#     1:(271, 320, 149, 206)
#     }

baldr_pupils_path = default_path_dict['baldr_pupil_crop'] #"config_files/baldr_pupils_coords.json"

with open(baldr_pupils_path, "r") as json_file:
    baldr_pupil_regions = json.load(json_file)

plt.ion() 

tstamp_rough = datetime.datetime.now().strftime("%d-%m-%Y")
tstamp = datetime.datetime.now().strftime("%d-%m-%YT%H.%M.%S")


args = parse_arguments()

# Redefine variables based on parsed arguments
ramp_file = args.ramp_file
kol_file = args.kol_file
write_report = args.write_report
beam = args.beam
a = args.a
signal_method = args.signal_method
control_method = args.control_method
output_config_filename = args.output_config_filename
output_report_dir = args.output_report_dir
fig_path = args.fig_path

if ramp_file == 'recent':
    print( f'using the most recent poke ramp calibration fits file from {default_data_calibration_path}.')

    ramp_file = get_most_recent_file(default_data_calibration_path, ramp_file_pattern)


if kol_file == 'recent':
    print( f'using the most recent kolmogorov calibration fits file from {default_data_calibration_path}.')

    kol_file = get_most_recent_file(default_data_calibration_path, kolmogorov_file_pattern)


os.makedirs(output_report_dir, exist_ok=True)

os.makedirs(fig_path, exist_ok=True)

    
"""
# input parameters (these should be available as input from command line )
ramp_file = #"/Users/bencb/Documents/ASGARD/Nice_trip_nov24/calibration_25-11-2024T17.16.09.fits"

kol_file = #"/Users/bencb/Documents/ASGARD/Nice_trip_nov24/kolmogorov_calibration_25-11-2024T17.16.09.fits"

beam = 2 # have hard coded region per beam, for now a local dictionary - but this could be eventually a json file 

a = 2 # amplitude index for calculating +/- around flat when calibrating DM / detector transform

signal_method = 'I-I0/N0'

control_method = 'zonal_linear'

plot=True

output_config_filename = f'baldr_transform_dict_beam{beam}_{tstamp}.json'

output_report_dir = f'/home/asg/Progs/repos/asgard-alignment/calibration/{tstamp_rough}/'

os.makedirs(output_report_dir, exist_ok=True)

fig_path = output_report_dir + 'figures/'

os.makedirs(fig_path, exist_ok=True)
"""
#===========================
# PDF Initialization
class PDFReport(FPDF):
    def header(self):
        self.set_font("Arial", "B", 12)
        self.cell(0, 10, "Baldr Model Calibration Report", border=0, ln=1, align="C")

    def footer(self):
        self.set_y(-15)
        self.set_font("Arial", "I", 10)
        self.cell(0, 10, f"Page {self.page_no()}", align="C")


if write_report:
    pdf = PDFReport()
    pdf.set_auto_page_break(auto=True, margin=15)

#===========================

#ramp_file = '/home/heimdallr/data/baldr_calibration/08-12-2024/calibration_Zonal_08-12-2024T08.47.26.fits'

recon_data = fits.open( ramp_file )

No_ramps = int(recon_data['SEQUENCE_IMGS'].header['#ramp steps'])
max_ramp = float( recon_data['SEQUENCE_IMGS'].header['in-poke max amp'] )
min_ramp = float( recon_data['SEQUENCE_IMGS'].header['out-poke max amp'] ) 
ramp_values = np.linspace( min_ramp, max_ramp, No_ramps)

flat_dm_cmd = recon_data['FLAT_DM_CMD'].data

Nmodes_poked = int(recon_data[0].header['HIERARCH Nmodes_poked']) # can also see recon_data[0].header['RESHAPE']

Nact =  int(recon_data[0].header['HIERARCH Nact'])  

N0 = np.mean( recon_data['FPM_OUT'].data, axis = 0) 
#P = np.sqrt( pupil ) # 
I0 = np.mean( recon_data['FPM_IN'].data, axis = 0) 

poke_imgs = recon_data['SEQUENCE_IMGS'].data[1:].reshape(No_ramps, 140, I0.shape[0], I0.shape[1])

x_start, x_end , y_start, y_end= baldr_pupil_regions[str(beam)]

# average over axis 1 which is number of frames taken per iteration 
poke_imgs_cropped = poke_imgs[:,:, x_start:x_end, y_start:y_end] #np.mean( recon_data['SEQUENCE_IMGS'].data[:,:, y_start:y_end, x_start:x_end] , axis=1)


## Identify bad pixels
fits_extensions = [hdu.name for hdu in recon_data]
if 'DARK' in fits_extensions:
    print('using DARK to create bad pixel map')
    img4badpixels = recon_data['DARK'].data[ :,x_start:x_end, y_start:y_end]
    mean_frame = np.mean(img4badpixels, axis=0)
    std_frame = np.std(img4badpixels, axis=0)
else:
    print('no DARK found. Using illuminated images to create bad pixel map')
    mean_frame = np.mean(poke_imgs_cropped, axis=(0, 1))
    std_frame = np.std(poke_imgs_cropped, axis=(0, 1))

global_mean = np.mean(mean_frame)
global_std = np.std(mean_frame)
bad_pixel_map = (np.abs(mean_frame - global_mean) > 6 * global_std) | (std_frame > 20 * np.median(std_frame))

# save bad pixel map to PDF
plt.figure() ; plt.imshow( bad_pixel_map ) ;plt.colorbar() ; plt.savefig(fig_path + 'bad_pixel_map.png')
plt.close('all')

def interpolate_bad_pixels(image, bad_pixel_map):
    filtered_image = image.copy()
    filtered_image[bad_pixel_map] = median_filter(image, size=3)[bad_pixel_map]
    return filtered_image

def process_poke_images(poke_images, bad_pixel_map):
    """
    Apply bad pixel interpolation to all frames and pokes.
    """
    num_ramps, num_acts, height, width = poke_images.shape
    filtered_images = np.zeros_like(poke_images)
    
    for r in range(num_ramps):
        for a in range(num_acts):
            filtered_images[r, a] = interpolate_bad_pixels(poke_images[r, a], bad_pixel_map)
    return filtered_images


poke_imgs_cropped = process_poke_images(poke_imgs_cropped, bad_pixel_map)
#plt.figure() ; plt.imshow( poke_imgs_cropped[0,0]-poke_imgs_cropped[1,0] ) ;plt.colorbar() ; plt.savefig('delme.png')





if write_report:
    util.nice_heatmap_subplots( im_list = [np.log10( I0[ x_start : x_end, y_start:y_end]), np.log10( N0[x_start : x_end,y_start:y_end] )] , 
                            xlabel_list=['x [pixels]','x [pixels]'], 
                            ylabel_list=['y [pixels]','y [pixels]'], 
                            title_list=['Phasemask in\n\n\n\n', 'phasemask out\n\n\n\n'], 
                            cbar_label_list=['ADU','ADU'], 
                            fontsize=15, 
                            cbar_orientation = 'bottom', 
                            axis_off=True, 
                            #vlims=[[0, 0.5*np.max(I0)], [0, 0.8*np.max(I0)]], 
                            savefig=fig_path + f'reference_intensities_beam{beam}.png')

plt.close('all')
### ADD table with camera settings 




# Convert header to a list of key-value pairs
header_data = [{"Key": k, "Value": v} for k, v in recon_data[0].header.items()]
df = pd.DataFrame(header_data)

if write_report:
    col_widths = [90, 90]  # Adjust to your content

    # Add Table Header
    pdf.add_page()
    pdf.set_font("Arial", style="B", size=12)
    for col_name, col_width in zip(df.columns, col_widths):
        pdf.cell(col_width, 10, col_name, border=1, align="C")
    pdf.ln()

    # Add Table Rows
    pdf.set_font("Arial", size=10)
    for _, row in df.iterrows():
        for col, col_width in zip(row, col_widths):
            pdf.cell(col_width, 10, str(col), border=1, align="C")
        pdf.ln()


    ### add reference intensities
    pdf.add_page()
    pdf.set_font("Arial", size=12)
    pdf.cell(0, 10, f"beam {beam} reference intensity with phasemask IN and OUT of the beam", ln=True)
    pdf.image(fig_path + f'reference_intensities_beam{beam}.png', x=10, y=30, w=190)




# Bad pixels in pdf
pdf.add_page()
pdf.set_font("Arial", size=12)
pdf.cell(0, 10,  'bad pixel map (interpolates bad pixels)', ln=True)
pdf.image(fig_path + 'bad_pixel_map.png', x=10, y=30, w=190)




## SVD 

amp_idx = (poke_imgs.shape[0]//2 - a , poke_imgs.shape[0]//2 + a)

IM = poke_imgs_cropped[ amp_idx[0] , : , :,:].reshape( 140, -1)  \
     - poke_imgs_cropped[ amp_idx[1] , : , :,:].reshape( 140, -1)

M2C = np.eye(140)

if write_report:
    plot_eigenmodes( IM, M2C, save_path = fig_path  )

    plt.close('all')

    pdf.add_page()
    pdf.set_font("Arial", size=12)
    pdf.cell(0, 10, f"EIGENMODES FROM ZONAL PUSH-PULL ON DM", ln=True)

    pdf.add_page()
    pdf.set_font("Arial", size=12)
    pdf.cell(0, 10, f"Singularvalues", ln=True)
    pdf.image(fig_path + f'singularvalues.png', x=10, y=30, w=190)

    pdf.add_page()
    pdf.set_font("Arial", size=12)
    pdf.cell(0, 10, f"Detector Eigenmodes", ln=True)
    pdf.image(fig_path + f'det_eignmodes.png', x=10, y=30, w=190)

    pdf.add_page()
    pdf.set_font("Arial", size=12)
    pdf.cell(0, 10, f"DM Eigenmodes", ln=True)
    pdf.image(fig_path + f'dm_eignmodes.png', x=10, y=30, w=190)



############# -- maybe this part can just be input (the region to look for the pupil)
# if pupil_regions == None:
#     pupil_regions = percentile_based_detect_pupils(N0, percentile=99, min_group_size=100, buffer=20, plot=True)

# pupils = []
# cropped_pupils = []
# for region in pupil_regions:
#     x_start, x_end, y_start, y_end = region
#     # Crop the pupil region
#     cropped_image = poke_imgs[:, :, y_start:y_end, x_start:x_end]
#     # Calculate the center column index of the region
#     center_col = (x_start + x_end) // 2
#     cropped_pupils.append((center_col, cropped_image))

# # Sort the cropped pupils by their center column index
# cropped_pupils.sort(key=lambda x: x[0])

# sorted_pupil_images = [pupil[1] for pupil in cropped_pupils]

# b2i = {'4':0, '3':1, '2':2, '1':3} #beam to index (assumes beam 4 has bottom)
##################


# get BMC DM actuator indicies for four inner corners 
# (used to calibrated affine transform between DM actuators and detector pixels)
dm_4_corners = DM_registration.get_inner_square_indices(outer_size=12, inner_offset=4) # flattened index of the DM actuator 

#for beam in range( 4 ):
img_4_corners = []
for actuator_number in dm_4_corners:
    # for each actuator poke we get the corresponding (differenced) image
    # and append it to img_4_corners
    
    amp_idx = (poke_imgs.shape[0]//2 - a , poke_imgs.shape[0]//2 + a)

    delta_img_raw = abs( poke_imgs_cropped[amp_idx[0]][actuator_number] - poke_imgs_cropped[amp_idx[1]][actuator_number] ) 
    
    delta_img = (delta_img_raw - np.mean( delta_img_raw)) / np.std( delta_img_raw )

    img_4_corners.append( delta_img ) 
    

transform_dict = DM_registration.calibrate_transform_between_DM_and_image( dm_4_corners, img_4_corners , debug=True, fig_path = fig_path  )


# Create a matrix for bi-linear interpolation
# (BCB: Tested against scipy map_coordinate method)
x_target = np.array( [x for x,_ in transform_dict['actuator_coord_list_pixel_space']] )
y_target = np.array( [y for _,y in transform_dict['actuator_coord_list_pixel_space']] )
x_grid = np.arange(poke_imgs_cropped[0][0].shape[0])
y_grid = np.arange(poke_imgs_cropped[0][0].shape[1])
pix2dm = DM_registration.construct_bilinear_interpolation_matrix(image_shape=poke_imgs_cropped[0][0].shape, 
                                        x_grid=x_grid, 
                                        y_grid=y_grid, 
                                        x_target=x_target,
                                        y_target=y_target)

# e.g. interpolate cmd_like_array = pix2dm @ img.flatten() 
# Note this is only valid for local frame defined by baldr_pupil_regions

if write_report:
    pdf.add_page()
    pdf.set_font("Arial", size=12)

    pdf.multi_cell(0, 10, "Calibrating Affine transform between DM actuators and camera pixels. Poke DM inner-corners, interpolate the ZWFS intensity onto a finner grid, fit Gaussians to each actuators response, and register coordinates of the fitted central peak. Find the intersection between the fitted peaks of the corners in pixel space. From this solve for the transform between actuator and pixel space" )
    #. 1. Poke DM corners, interpolation ZWFS intensity onto finner grid, fit Gaussian, and register coordinates of the central peak", ln=True)
    pdf.image(fig_path + f'DM_corner_poke_in_DM_space.png', x=10, y=60, w=170)

    # for i, a in enumerate(transform_dict[ 'corner_fit_results'].keys()):
    #     DM_registration.plot_fit_results(transform_dict[ 'corner_fit_results'][a], savefig = fig_path + f'corner_poke_fit_act{a}.png')
    #     pdf.add_page()
    #     pdf.set_font("Arial", size=15)
    #     pdf.cell(0, 10 , f"fit from corner actuator {a}", ln=True)
    #     pdf.image(fig_path + f'corner_poke_fit_act{a}.png', x=10, y=30, w=190)

    # Add a new page for all corner actuator plots
    pdf.add_page()
    pdf.set_font("Arial", size=12)
    pdf.cell(0, 10, "Fitting ZWFS pixel response to DM inner-corner pokes", ln=True)
    # Add a new page for all corner actuator plots

    # Vertical layout (4 rows)
    y_positions = [30, 90, 150, 210]  # Vertical positions for each row
    plot_width = 170  # Full-width plots
    plot_height = 40  # Adjust the height to fit four plots

    # Iterate over the actuators
    for i, a in enumerate(transform_dict['corner_fit_results'].keys()):
        # Generate and save the plot for each corner actuator
        DM_registration.plot_fit_results(
            transform_dict['corner_fit_results'][a], 
            savefig=fig_path + f'corner_poke_fit_act{a}.png'
        )

        # Position the plot
        y = y_positions[i]

        # Embed the plot image
        pdf.image(fig_path + f'corner_poke_fit_act{a}.png', x=10, y=y, w=plot_width, h=plot_height)

        # Add a caption below each plot
        pdf.set_xy(10, y + plot_height + 2)  # Move below the image
        pdf.set_font("Arial", size=10)
        pdf.cell(plot_width, 10, f"Actuator {a}", ln=False, align="C")


    # add a table 



        

    # Add this plot to PDF
    pdf.add_page()
    #pdf.multi_cell(0, 10, f"2. ")
    pdf.set_font("Arial", size=12)
    pdf.cell(0, 30, f"beam {beam} DM registration in pixelspace", ln=True)
    pdf.image(fig_path + "DM_registration_in_pixel_space.png", x=10, y=40, w=190)




# save and add it to pdf 
cal_frames = poke_imgs_cropped.reshape( -1,  x_end - x_start, y_end - y_start ) #poke_imgs_cropped.reshape( -1, y_end - y_start, x_end - x_start )

I0 = np.mean( recon_data['FPM_IN'].data, axis = 0) 
N0 = np.mean( recon_data['FPM_OUT'].data, axis = 0) 

#see with matrix method 
interpolated_i =  np.array( [DM_registration.interpolate_pixel_intensities(image = i, pixel_coords = transform_dict['actuator_coord_list_pixel_space']) for i in cal_frames] )

interpolated_I0 = DM_registration.interpolate_pixel_intensities(image = I0, pixel_coords = transform_dict['actuator_coord_list_pixel_space'])
interpolated_N0 = DM_registration.interpolate_pixel_intensities(image = N0, pixel_coords = transform_dict['actuator_coord_list_pixel_space'])

dm_cmd = recon_data['DM_CMD_SEQUENCE'].data[1:]

if write_report:
    # plot the response 
    fig, axes = DM_actuator_mosaic_plot( xx = dm_cmd , yy = interpolated_i , filter_crosscoupling = True )

    fig.savefig(fig_path + "wfs_response_single_act.png" )

    pdf.add_page()
    pdf.set_font("Arial", size=12)
    pdf.multi_cell(0, 10, f"DM Mosaic Plot for beam {beam} WFS response on the registered (interpolated) pixel from a linear ramp of each DM actuator.")
    pdf.image(fig_path + "wfs_response_single_act.png", x=10, y=50, w=190)



# interpolated_I0 = DM_registration.interpolate_pixel_intensities(image = I0, pixel_coords = transform_dict['actuator_coord_list_pixel_space'])
# interpolated_N0 = DM_registration.interpolate_pixel_intensities(image = N0, pixel_coords = transform_dict['actuator_coord_list_pixel_space'])


if kol_file: # then we fit a different Kolmogorov disturbance on 
    
    kol_data = fits.open( kol_file )

    # first few frames are usually reference intensities. We note up to what index we measure these. 
    I0_index = int( kol_data['SEQUENCE_IMGS'].header['I0_indicies'].split('-')[-1] ) 

    # users can take a few frames per iteration so we average over this axis
    cal_frames = np.mean( kol_data['SEQUENCE_IMGS'].data[I0_index:,:, y_start:y_end, x_start:x_end], axis = 1) 

    I0 = np.mean( kol_data['FPM_IN'].data, axis = 0) 
    N0 = np.mean( kol_data['FPM_OUT'].data, axis = 0) 

    interpolated_i =  np.array( [DM_registration.interpolate_pixel_intensities(image = i, pixel_coords = transform_dict['actuator_coord_list_pixel_space']) for i in cal_frames] )

    interpolated_I0 = DM_registration.interpolate_pixel_intensities(image = I0, pixel_coords = transform_dict['actuator_coord_list_pixel_space'])
    interpolated_N0 = DM_registration.interpolate_pixel_intensities(image = N0, pixel_coords = transform_dict['actuator_coord_list_pixel_space'])

    dm_cmd = np.array( kol_data['DM_CMD_SEQUENCE'].data[I0_index:,:]  )



fit_results=[]

if control_method == 'zonal_linear':

    if write_report:

        # Function to get row and column for a given actuator index (for plotting)
        def _get_row_col(actuator_index):

            rows, cols = 12, 12

            # Missing corners
            missing_corners = [(0, 0), (0, 11), (11, 0), (11, 11)]

            # Create a flattened index map for valid positions
            valid_positions = [(r, c) for r in range(rows) for c in range(cols) if (r, c) not in missing_corners]

            if actuator_index < 0 or actuator_index >= len(valid_positions):
                raise ValueError(f"Invalid actuator index: {actuator_index}")
            return valid_positions[actuator_index]


        fig, axes = plt.subplots(12, 12, figsize=(10, 10), sharex=True, sharey=True)
        fig.tight_layout(pad=2.0) 
        for axx in axes.reshape(-1):
            axx.axis('off')
    
    # Loop through all actuators
    for act in range(140):

        # Select data for the current actuator and label data
        x = interpolated_i[:, act] # NORMALIZATION
        y = dm_cmd[:, act]
        
        # Standard Linear Regression
        slope_std, intercept_std, _, _, std_err_std = linregress(x, y)
        
        # Prior parameters for Bayesian regression
        alpha = 1.0  # Precision of prior distribution
        beta_values = 1.0 / np.var(dm_cmd, axis=0)  # Precision of observation noise (per actuator)

        # Bayesian Linear Regression
        X = np.vstack([x, np.ones_like(x)]).T
        beta = beta_values[act]
        S_0_inv = alpha * np.eye(X.shape[1])
        S_N_inv = S_0_inv + beta * X.T @ X
        S_N = np.linalg.inv(S_N_inv)
        m_N = beta * S_N @ X.T @ y
        slope_bayes, intercept_bayes = m_N
        uncertainties = np.sqrt(np.diag(S_N))  # Posterior standard deviation
        
        # Example data
        y_true = dm_cmd[:, act]  # True DM commands
        y_pred = slope_std * interpolated_i[:, act] + intercept_std  # Standard fit predictions

        # Metrics
        r2 = r2_score(y_true, y_pred)
        mse = mean_squared_error(y_true, y_pred)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_true, y_pred)

        # Store results
        fit_results.append({
            "actuator": act,
            "slope_standard": slope_std,
            "intercept_standard": intercept_std,
            "slope_bayesian": slope_bayes,
            "intercept_bayesian": intercept_bayes,
            "slope_uncertainty": uncertainties[0],
            "intercept_uncertainty": uncertainties[1],
            "std_err_standard": std_err_std,
            "r2":r2,
            'mse':mse,
            'rmse':rmse,
            'mae':mae
        })


        if write_report:
            # New input data for prediction
            x_new = np.linspace(x.min(), x.max(), len(x))
            X_new = np.vstack([x_new, np.ones_like(x_new)]).T

            # Posterior predictive mean and variance
            mu_pred = X_new @ m_N
            var_pred = 1 / beta + np.sum(X_new @ S_N * X_new, axis=1)
            std_pred = np.sqrt(var_pred)


            ax = axes[_get_row_col(act)]
            ax.axis('off')
            # Plot the data and both regression results
            #plt.figure(figsize=(10, 6))

            # Data points
            ax.plot(x, y, '.', label='Data')

            # Standard Linear Regression
            #plt.plot(x_fit, y_fit_standard, 'g--', label=f'Standard Fit: y={slope:.2e}x + {intercept:.2e}')

            # Bayesian Linear Regression
            #plt.plot(x_fit, y_fit_bayesian, 'r-', label=f'Bayesian Fit: y={m_N[0]:.2e}x + {m_N[1]:.2e}')
            #plt.fill_between(x_fit, y_fit_bayesian - 1.96 * y_fit_std, y_fit_bayesian + 1.96 * y_fit_std,
            #                color='red', alpha=0.2, label='Bayesian 95% CI')
            ax.plot(x_new, mu_pred, 'r-', label='Bayesian Prediction')
            ax.fill_between(x_new, mu_pred - 1.96 * std_pred, mu_pred + 1.96 * std_pred, color='red', alpha=0.2, label='95% CI')
            # Plot setup
            ax.set_xlabel([]) #'Interpolated ZWFS Signal')
            ax.set_ylabel([]) #f'DM Actuator {act} Command')
            ax.set_title(f'act#{act+1}')
            #plt.legend()
            plt.grid(True)
        

    if write_report:
        plt.tight_layout()
        plt.savefig( fig_path + 'fit_mosaic.png')
        plt.show() 

        pdf.add_page()
        pdf.set_font("Arial", size=12)
        pdf.multi_cell(0, 10, "DM Mosaic Plot for linear fits of a rolling Kolmogorov phasescreen on the DM. y = DM cmds, x = ZWFS signal")
        pdf.image(fig_path + 'fit_mosaic.png', x=10, y=30, w=190)


        # plt.figure() 
        # plt.imshow( util.get_DM_command_in_2D( [r['mse'] for r in fit_results] ) )
        # plt.colorbar(label='linear fit MSE per DM actuator')
        # plt.show()

        plt.figure() 
        plt.imshow( util.get_DM_command_in_2D( [r['r2'] for r in fit_results] ) )
        plt.colorbar(label=r'$R^2$ per DM actuator')
        plt.savefig( fig_path + 'fit_R2.png')
        plt.show()

        pdf.add_page()
        pdf.set_font("Arial", size=12)
        pdf.multi_cell(0, 10, "R2 between Kolmogorov actuator commands and ZWFS response in the registered (interpolated) pixel")
        pdf.image(fig_path + 'fit_R2.png', x=10, y=30, w=190)


        # plt.figure() 
        # plt.imshow( util.get_DM_command_in_2D( [r['r2'] for r in fit_results] ) > 0.5 )
        # plt.colorbar(label=r'$R^2$ per DM actuator')
        # plt.show()



    # Strehl Model 


else:
    raise UserWarning("control_model not valid")



if write_report:
    pdf_output_path = fig_path + f"calibration_report_{tstamp}.pdf"
    pdf.output(pdf_output_path)


final_dict = {"beam":beam,
              "signal_method":signal_method,
              "control_method":control_method,
              "dm_registration": transform_dict, 
              "pix2dm_interp_matrix": pix2dm,
              "control_model":fit_results, 
              "pupil_regions":baldr_pupil_regions[str(beam)],
              "interpolated_I0":interpolated_I0,
              "interpolated_N0":interpolated_N0,
              "I0":I0,
              "N0":N0}


# write to json
with open(output_config_filename, 'w') as f:
    json.dump( util.convert_to_serializable( final_dict ) , f)




# """
# global config file for my system called Baldr. 4 telescopes with 4 corresponding beams, each with :
# Sub pupil pixel coordinates to crop image from camera for each telescope 
# Matrix to interpolate image to registered actuator position in pixelspace 
# Boolean matrix with active actuators 
# Vectors for actuator linear model 
# """