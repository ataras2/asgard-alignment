

# script_dir = os.path.dirname(os.path.abspath(__file__))
# sys.path.append(script_dir)
import numpy as np
import time 
import zmq
import glob
import sys
import os 
import toml
import json
import argparse
import matplotlib.pyplot as plt
from astropy.io import fits
import datetime
from scipy.stats import pearsonr
from scipy.ndimage import median_filter
from xaosim.shmlib import shm
import asgard_alignment.controllino as co # for turning on / off source \
from asgard_alignment.DM_shm_ctrl import dmclass
import common.DM_basis_functions as dmbases
import common.phasemask_centering_tool as pct
import common.phasescreens as ps 
import pyBaldr.utilities as util 


try:
    from asgard_alignment import controllino as co
    myco = co.Controllino('172.16.8.200')
    controllino_available = True
    print('controllino connected')
    
except:
    print('WARNING Controllino cannot connect. WILL NOT MOVE SOURCE OUT FOR DARK')
    controllino_available = False 



"""
convention to apply flat command on channel 0!
convention to apply calibration shapes on channel 1!
convention to apply DM modes on channel 2!
dmclass.set_data() applies on channel 2!
"""


### using SHM camera structure
def move_relative_and_get_image(cam, beam, baldr_pupils, phasemask, savefigName=None, use_multideviceserver=True):
    print(
        f"input savefigName = {savefigName} <- this is where output images will be saved.\nNo plots created if savefigName = None"
    )
    r1,r2,c1,c2 = baldr_pupils[f"{beam}"]
    exit = 0
    while not exit:
        input_str = input('enter "e" to exit, else input relative movement in um: x,y')
        if input_str == "e":
            exit = 1
        else:
            try:
                xy = input_str.split(",")
                x = float(xy[0])
                y = float(xy[1])

                if use_multideviceserver:
                    #message = f"fpm_moveabs phasemask{beam} {[x,y]}"
                    #phasemask.send_string(message)
                    message = f"moverel BMX{beam} {x}"
                    phasemask.send_string(message)
                    response = phasemask.recv_string()
                    print(response)

                    message = f"moverel BMY{beam} {y}"
                    phasemask.send_string(message)
                    response = phasemask.recv_string()
                    print(response)

                else:
                    phasemask.move_relative([x, y])

                time.sleep(3)
                img = np.mean(
                    cam.get_data(),
                    axis=0,
                )[r1:r2,c1:c2]
                if savefigName != None:
                    plt.figure()
                    plt.imshow( np.log10( img ) )
                    plt.colorbar()
                    plt.savefig(savefigName)
            except:
                print('incorrect input. Try input "1,1" as an example, or "e" to exit')

    plt.close()


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



def get_bad_pixel_indicies( imgs, std_threshold = 20, mean_threshold=6):
    # To get bad pixels we just take a bunch of images and look at pixel variance and mean

    ## Identify bad pixels
    mean_frame = np.mean(imgs, axis=0)
    std_frame = np.std(imgs, axis=0)

    global_mean = np.mean(mean_frame)
    global_std = np.std(mean_frame)
    bad_pixel_map = (np.abs(mean_frame - global_mean) > mean_threshold * global_std) | (std_frame > std_threshold * np.median(std_frame))

    return bad_pixel_map


def interpolate_bad_pixels(img, bad_pixel_map):
    filtered_image = img.copy()
    filtered_image[bad_pixel_map] = median_filter(img, size=3)[bad_pixel_map]
    return filtered_image



def send_and_get_response(message):
    # st.write(f"Sending message to server: {message}")
    state_dict["message_history"].append(
        f":blue[Sending message to server: ] {message}\n"
    )
    state_dict["socket"].send_string(message)
    response = state_dict["socket"].recv_string()
    if "NACK" in response or "not connected" in response:
        colour = "red"
    else:
        colour = "green"
    # st.markdown(f":{colour}[Received response from server: ] {response}")
    state_dict["message_history"].append(
        f":{colour}[Received response from server: ] {response}\n"
    )

    return response.strip()


def get_motor_states_as_list_of_dicts( ): 

    motor_names = ["SDLA", "SDL12", "SDL34", "SSS", "BFO"]
    motor_names_no_beams = [
                "HFO",
                "HTPP",
                "HTPI",
                "HTTP",
                "HTTI",
                "BDS",
                "BTT",
                "BTP",
                "BMX",
                "BMY",
            ]


    for motor in motor_names_no_beams:
        for beam_number in range(1, 5):
            motor_names.append(f"{motor}{beam_number}")

    states = []
    for name in motor_names:
        message = f"read {name}"
        res = send_and_get_response(message)

        if "NACK" in res:
            is_connected = False
        else:
            is_connected = True

        state = {
            "name": name,
            "is_connected": is_connected,
        }
        if is_connected:
            state["position"] = float(res)

        states.append(state)

    return states


def save_motor_states_as_hdu(motor_states):
    """
    Create an HDU for motor states as a binary table.

    Parameters:
    - motor_states (list of dict): List of motor state dictionaries.

    Returns:
    - fits.BinTableHDU: The binary table HDU containing motor states.
    """
    # Prepare columns for the FITS binary table
    motor_names = [state["name"] for state in motor_states]
    is_connected = [state["is_connected"] for state in motor_states]
    positions = [state.get("position", np.nan) for state in motor_states]  # Use NaN for missing positions

    col1 = fits.Column(name="MotorName", format="20A", array=np.array(motor_names))  # ASCII strings
    col2 = fits.Column(name="IsConnected", format="L", array=np.array(is_connected))  # Logical (boolean)
    col3 = fits.Column(name="Position", format="E", array=np.array(positions, dtype=np.float32))  # Float32

    # Create the binary table HDU
    cols = fits.ColDefs([col1, col2, col3])
    return fits.BinTableHDU.from_columns(cols, name="MotorStates")


def recursive_update(orig, new):
    """
    Recursively update dictionary 'orig' with 'new' without overwriting sub-dictionaries.
    """
    for key, value in new.items():
        if (key in orig and isinstance(orig[key], dict) 
            and isinstance(value, dict)):
            recursive_update(orig[key], value)
        else:
            orig[key] = value
    return orig


def process_signal( i, I0, N0):
    # i is intensity, I0 reference intensity (zwfs in), N0 clear pupil (zwfs out)
    return ( i - I0 ) / N0 




# setting up socket to ZMQ communication to multi device server
parser = argparse.ArgumentParser(description="Baldr Pupil Fit Configuration.")

default_toml = os.path.join( "config_files", "baldr_config_#.toml") #os.path.dirname(os.path.abspath(__file__)), "..", "config_files", "baldr_config.toml")

# setting up socket to ZMQ communication to multi device server
parser.add_argument("--host", type=str, default="localhost", help="Server host")
parser.add_argument("--port", type=int, default=5555, help="Server port")
parser.add_argument(
    "--timeout", type=int, default=5000, help="Response timeout in milliseconds"
)

# Camera shared memory path
parser.add_argument(
    "--global_camera_shm",
    type=str,
    default="/dev/shm/cred1.im.shm",
    help="Camera shared memory path. Default: /dev/shm/cred1.im.shm"
)

# TOML file path; default is relative to the current file's directory.
parser.add_argument(
    "--toml_file",
    type=str,
    default=default_toml,
    help="TOML file to write/edit. Default: ../config_files/baldr_config.toml (relative to script)"
)

# Beam ids: provided as a comma-separated string and converted to a list of ints.
parser.add_argument(
    "--beam_id",
    type=lambda s: [int(item) for item in s.split(",")],
    default=[2],
    help="Comma-separated beam IDs to apply. Default: 1,2,3,4"
)

parser.add_argument(
    "--phasemask",
    type=str,
    default=None,
    help="select which phasemask we're calibrating the model for. Default: None - in which case the user is prompted to enter the phasemask"
)

parser.add_argument(
    "--reconstructor_model",
    type=str,
    default="zonal_linear",
    help="Reconstructor model to use. Default is 'zonal_linear'."
)

parser.add_argument(
    "--model_metric_threshold",
    type=float,
    default=None,
    help="Model metric threshold (DM units). Default is None."
)

parser.add_argument(
    "--act_filter_method",
    type=str,
    default="act_rad-4",
    help="Actuator filter method. Default is 'act_rad-4'."
)

parser.add_argument(
    "--number_of_rolls",
    type=int,
    default=1000,
    help="Number of rolls. Default is 500."
)

parser.add_argument(
    "--scaling_factor",
    type=float,
    default=0.2,
    help="Scaling factor. Default is 0.2."
)


#--act_filter_method pearson_R --model_metric_threshold 0.65

args=parser.parse_args()


reconstructor_model = args.reconstructor_model
model_metric_threshold = args.model_metric_threshold
act_filter_method = args.act_filter_method 
number_of_rolls = args.number_of_rolls
scaling_factor = args.scaling_factor

## Input configuration 

pupil_mask = {}
exterior_mask = {}
dm_flat_offsets = {}
dm_flat_offsets = {}
I2M_dict = {}
for beam_id in args.beam_id:

    # read in TOML as dictionary for config 
    with open(args.toml_file.replace('#',f'{beam_id}'), "r") as f:
        config_dict = toml.load(f)

        # Baldr pupils from global frame 
        baldr_pupils = config_dict['baldr_pupils']
        # boolean filter of the registered (clear) pupil
        # for each local (cropped) baldr pupil frame 
        pupil_mask[beam_id] = config_dict[f'beam{beam_id}']["pupil_mask"] 
        # boolean filter for outside pupil where we see scattered
        # light with phase mask in (basic Strehl proxy)
        exterior_mask[beam_id] =  config_dict[f'beam{beam_id}']["pupil_mask"] 
        # flat offset for DMs (mainly focus)
        dm_flat_offsets[beam_id] = config_dict[f'beam{beam_id}']["DM_flat_offset"] 
        # intensities interpolation matrix to registered DM actuator (from local pupil frame)
        I2M_dict[beam_id] = config_dict[f'beam{beam_id}']['I2M']



# set up ZMQ to communicate with motors 
context = zmq.Context()
context.socket(zmq.REQ)
socket = context.socket(zmq.REQ)
socket.setsockopt(zmq.RCVTIMEO, args.timeout)
server_address = f"tcp://{args.host}:{args.port}"
socket.connect(server_address)
state_dict = {"message_history": [], "socket": socket}

#####
##### --- manually set up camera settings before hand
#####
print( 'You should manually set up camera settings before hand')

# Set up global camera frame SHM 
c = shm(args.global_camera_shm)

# set up DM SHMs 
dm_shm_dict = {}
for beam_id in args.beam_id:
    dm_shm_dict[beam_id] = dmclass( beam_id=beam_id )
    # zero all channels
    dm_shm_dict[beam_id].zero_all()
    # activate flat (does this on channel 1)
    dm_shm_dict[beam_id].activate_flat()
    # apply dm flat offset (does this on channel 2)
    #dm_shm_dict[beam_id].set_data( np.array( dm_flat_offsets[beam_id] ) )


# set up phasemask 
# # This could be moved to function in pahsemask_centering_tools.py (input beam_id, phasemask, baldr_pupils, state_dict['socket'])
for beam_id in args.beam_id:
    valid_reference_position_files = glob.glob(
        f"/home/asg/Progs/repos/asgard-alignment/config_files/phasemask_positions/beam{args.beam_id}/*json"
        )


    with open(max(valid_reference_position_files, key=os.path.getmtime), "r") as file:
        print(f"using most recent phasemask position file for beam {beam_id}:\n({file}")
        phasemask_positions = json.load(file)


    valid_input_phasemasks = [f"H{i}" for i in range(1,6)] + [f"J{i}" for i in range(1,6)] + ["undefined"]
    if (args.phasemask not in valid_input_phasemasks) and (args.phasemask is not None)  :
        raise UserWarning(f"invalid phasemask position. Try one of the following: {valid_input_phasemasks}")

    if args.phasemask is None:
        userInput = 1
        ui = input("\ninput mask label. Try H1-5, J1-5 or undefined. This will be input to TOML config file.\n")
        invalid = ui not in valid_input_phasemasks
        while invalid:
            ui = input(f"\nInvalid entry. input mask label. Try 'undefined' for example. Valid options are: {valid_input_phasemasks}\n")
            invalid = ui not in valid_input_phasemasks
        
        # Define phasemask label to write in TOML 
        args.phasemask = ui 


    ui = input("\nEnter 1 to continue in current position or 0 to move to specific position\n")
    invalid = ui not in ['0', '1']
    while invalid :
        ui = input("\nInvalid entry. Press 1 to continue in current position or 0 to move to specific position\n")
        invalid = ui not in ['0', '1']

    if ui == '1':
        print(f"\nwe continue in current position. registering it as phasemask { args.phasemask}\n")
    elif ui =='0':
        if args.phasemask == 'undefined':
            print( '\nundefined mask position. we just continue in current position\n')
        else:
            Xpos0 = phasemask_positions[args.phasemask][0]
            Ypos0 = phasemask_positions[args.phasemask][1]
        
            print(f"\nmoving to mask {args.phasemask} on beam {beam_id} at X,Y = {round(Xpos0)}um, {round(Ypos0)}um\n")

            # move to position 
            message = f"moveabs BMX{beam_id} {Xpos0}"
            send_and_get_response(message)
            time.sleep(3)
            message = f"moveabs BMY{beam_id} {Ypos0}"
            send_and_get_response(message)

            ui = input("\nEnter 1 to adjust mask position or 0 to continue\n")
            invalid = ui not in ['0', '1']
            while invalid :
                ui = input("\nInvalid entry. Press 1 to continue in current position or 0 to move to specific position\n")
                invalid = ui not in ['0', '1']

            if ui == '1':
                move_relative_and_get_image(cam=c, beam=beam_id, baldr_pupils = baldr_pupils, phasemask=state_dict["socket"],  savefigName='delme.png', use_multideviceserver=True )






# Get Darks 
if controllino_available:
    
    myco.turn_off("SBB")
    time.sleep(10)
    
    dark_raw = c.get_data()

    myco.turn_on("SBB")
    time.sleep(10)

    bad_pixel_mask = get_bad_pixel_indicies( dark_raw, std_threshold = 20, mean_threshold=6)
else:
    dark_raw = c.get_data()

    bad_pixel_mask = get_bad_pixel_indicies( dark_raw, std_threshold = 20, mean_threshold=6)



# check phasemask alignment 
beam = int( input( "\ndo you want to check the phasemasks for a beam. Enter beam number (1,2,3,4) or 0 to continue\n") )
while beam :
    print( 'we save images as delme.png in asgard-alignment project - open it!')
    img = np.sum( c.get_data()  , axis = 0 ) 
    r1,r2,c1,c2 = baldr_pupils[str(beam)]
    #print( r1,r2,c1,c2  )
    plt.figure(); plt.imshow( np.log10( img[r1:r2,c1:c2] ) ) ; plt.colorbar(); plt.savefig('delme.png')

    # time.sleep(5)

    # manual centering 
    move_relative_and_get_image(cam=c, beam=beam, baldr_pupils = baldr_pupils, phasemask=state_dict["socket"],  savefigName='delme.png', use_multideviceserver=True, roi=[r1,r2,c1,c2 ])

    beam = int( input( "do you want to check the phasemask alignment for a particular beam. Enter beam number (1,2,3,4) or 0 to continue") )


# Get reference pupils (later this can just be a SHM address)
zwfs_pupils = {}
clear_pupils = {}
rel_offset = 200.0 #um phasemask offset for clear pupil

# ZWFS Pupil
img = np.mean( c.get_data() ,axis=0) 
for beam_id in args.beam_id:
    r1,r2,c1,c2 = baldr_pupils[f"{beam_id}"]
    #cropped_img = interpolate_bad_pixels(img[r1:r2, c1:c2], bad_pixel_mask[r1:r2, c1:c2])
    cropped_img = img[r1:r2, c1:c2] #/np.mean(img[r1:r2, c1:c2][pupil_masks[bb]])
    zwfs_pupils[beam_id] = cropped_img

    message = f"moverel BMX{beam_id} {rel_offset}"
    res = send_and_get_response(message)
    print(res) 
    time.sleep( 1 )
    message = f"moverel BMY{beam_id} {rel_offset}"
    res = send_and_get_response(message)
    print(res) 
    time.sleep(10)


#Clear Pupil
img = np.mean( c.get_data() ,axis=0) 
for beam_id in args.beam_id:
    r1,r2,c1,c2 = baldr_pupils[f"{beam_id}"]
    cropped_img = img[r1:r2, c1:c2]
    clear_pupils[beam_id] = cropped_img

    message = f"moverel BMX{beam_id} {-rel_offset}"
    res = send_and_get_response(message)
    print(res) 
    time.sleep(1)
    message = f"moverel BMY{beam_id} {-rel_offset}"
    res = send_and_get_response(message)
    print(res) 
    time.sleep(10)


## check them 
dark = {}
for beam_id in args.beam_id:
    r1,r2,c1,c2 = baldr_pupils[f"{beam_id}"]


    # checks 
    cbar_label_list = ['[adu]','[adu]', '[adu]']
    title_list = ['DARK', 'CLEAR PUPIL', 'ZWFS PUPIL']
    xlabel_list = ['','','']
    ylabel_list = ['','','']

    dark[beam_id] = np.mean( dark_raw[:,r1:r2,c1:c2], axis=0)

    im_list = [dark[beam_id], clear_pupils[beam_id], zwfs_pupils[beam_id] ]
    util.nice_heatmap_subplots( im_list , 
                                cbar_label_list = cbar_label_list,
                                title_list=title_list,
                                xlabel_list=xlabel_list,
                                ylabel_list=ylabel_list,
                                savefig='delme.png' )

    plt.show()
    


#prepare phasescreen to put on DM 
D = 1.8
act_per_it = 0.5 # how many actuators does the screen pass per iteration 
V = 10 / act_per_it  / D #m/s (10 actuators across pupil on DM)
#scaling_factor = 0.2
I0_indicies = 10 # how many reference pupils do we get?
scrn = ps.PhaseScreenVonKarman(nx_size= int( 12 / act_per_it ) , pixel_scale= D / 12, r0=0.1, L0=12)
corner_indicies = [0, 11, 11 * 12, -1] # DM corner indidices
#number_of_rolls = 500

DM_command_sequence = [np.zeros([12,12]) for _ in range(I0_indicies)]
for i in range(number_of_rolls):
    scrn.add_row()
    # bin phase screen onto DM space 
    dm_scrn = util.create_phase_screen_cmd_for_DM(scrn,  scaling_factor=scaling_factor, drop_indicies = [0, 11, 11 * 12, -1] , plot_cmd=False)
    # update DM command 
    #plt.figure(i)
    #plt.imshow(  util.get_DM_command_in_2D(dm_scrn) )
    #plt.colorbar()

    # put in SHM format 140 1D cmd -> 144 square 
    twoDized = np.nan_to_num( util.get_DM_command_in_2D(dm_scrn), 0 )
    DM_command_sequence.append( twoDized )


# --- additional labels to append to fits file to keep information about the sequence applied
additional_header_labels = [
    ("number_of_rolls", number_of_rolls),
    ('I0_indicies','0-10'),
    ('act_per_it',act_per_it),
    ('D',D),
    ('V',V),
    ('scaling_factor', scaling_factor),
    ("Nact", 140),
    ('fps', 200),
    ('gain', 1)
]


tstamp = datetime.datetime.now().strftime("%d-%m-%YT%H.%M.%S")
tstamp_rough =  datetime.datetime.now().strftime("%d-%m-%Y")


sleeptime_between_commands = 3 
print(f'GOING SLOW FOR SHM LAG {sleeptime_between_commands}s DELAY')
image_list = {b:[] for b in args.beam_id}
for cmd_indx, cmd in enumerate(DM_command_sequence):
    print(f"executing cmd_indx {cmd_indx} / {len(DM_command_sequence)}")
    # wait a sec

    # ok, now apply command
    for beam_id in args.beam_id:
        #without dmflat offset 
        dm_shm_dict[beam_id].set_data( cmd ) #dm_flat_offsets[beam_id] + cmd)

    # wait a sec
    time.sleep(sleeptime_between_commands)

    # get the image
    ims_tmp = np.mean(
            c.get_data() , axis = 0
        )
    
    # get the pupil cropped images
    for beam_id in args.beam_id:
        r1,r2,c1,c2 = baldr_pupils[f"{beam_id}"]
        # add additional nested list, since some times we may make many measurements per iteration 
        image_list[beam_id].append([ims_tmp[r1:r2, c1:c2]])


# init fits files if necessary
# should_we_record_images = True
# take_mean_of_images = True
# save_dm_cmds = True



# show the fits !! 


# ====== make references fits files

save_fits = {}
for beam_id in args.beam_id:
    save_fits[beam_id] = "~/Downloads/" + f"kolmogorov_calibration_{beam_id}.fits" # _{tstamp}.fits"    
    r1,r2,c1,c2 = baldr_pupils[f"{beam_id}"]


    data = fits.HDUList([])  # init main fits file to append things to

    cam_fits = fits.PrimaryHDU( image_list[beam_id] )
    dm_fits = fits.PrimaryHDU(DM_command_sequence)

    flat_DMoffset_fits = fits.PrimaryHDU(dm_flat_offsets[beam_id])
    flat_DM_fits = fits.PrimaryHDU(dm_shm_dict[beam_id].shms[0].get_data())
    I2M_fits = fits.PrimaryHDU([I2M_dict[beam_id]])

    I0_fits = fits.PrimaryHDU(clear_pupils[beam_id])
    N0_fits = fits.PrimaryHDU(clear_pupils[beam_id])
    DARK_fits = fits.PrimaryHDU([d[r1:r2,c1:c2] for d in dark_raw])
    
    
    # headers
    cam_fits.header.set('EXTNAME', "SEQUENCE_IMGS")
    if additional_header_labels != None:
        if type(additional_header_labels) == list:
            for i, h in enumerate(additional_header_labels):
                cam_fits.header.set(h[0], h[1])
        else:
            cam_fits.header.set(additional_header_labels[0], additional_header_labels[1])


    dm_fits.header.set("timestamp", str(datetime.datetime.now()))
    dm_fits.header.set("EXTNAME", "DM_CMD_SEQUENCE")


    flat_DMoffset_fits.header.set("EXTNAME", "FLAT_DM_OFFSET_CMD")
    flat_DM_fits.header.set("EXTNAME", "FLAT_DM_CMD")
    I2M_fits.header.set("EXTNAME", "I2M")

    I0_fits.header.set("EXTNAME", "FPM_IN")
    N0_fits.header.set("EXTNAME", "FPM_OUT")
    DARK_fits.header.set("EXTNAME", "DARK")


    # motor states 
    motor_states = get_motor_states_as_list_of_dicts()
    bintab_fits = save_motor_states_as_hdu( motor_states )


    # append to the data
    data.append(cam_fits)
    data.append(dm_fits)
    data.append(flat_DM_fits)
    data.append(flat_DMoffset_fits)
    data.append(I2M_fits)
    data.append(I0_fits)
    data.append(N0_fits)
    data.append(DARK_fits)
    data.append(bintab_fits )




    if type(save_fits[beam_id]) == str:
        print(f'saving {save_fits[beam_id]}')
        data.writeto(save_fits[beam_id], overwrite=True)
    else:
        raise TypeError(
            "save_images needs to be either None or a string indicating where to save file"
        )
        



for beam_id in args.beam_id:

    # zero all channels
    dm_shm_dict[beam_id].zero_all()
    # activate flat (does this on channel 1)
    dm_shm_dict[beam_id].activate_flat()
    # apply dm flat offset (does this on channel 2)
    dm_shm_dict[beam_id].set_data( np.array( dm_flat_offsets[beam_id] ) )






for beam_id in args.beam_id:

    d = fits.open( save_fits[beam_id] ) 

    # what index do we start rolling phasescreen
    iStart = int(d["SEQUENCE_IMGS"].header['HIERARCH I0_indicies'].split('-')[-1])
    I2M = d["I2M"].data[0]
    imgs = d["SEQUENCE_IMGS"].data[iStart:,0,:,:]
    cmds = d["DM_CMD_SEQUENCE"].data[iStart:]
    I0 =   d["FPM_IN"].data
    I0dm = I2M @ I0.reshape(-1)
    N0 =   d["FPM_OUT"].data
    N0dm = I2M @ N0.reshape(-1)

    Nx_act_DM = 12 
    corner_indices = [0, Nx_act_DM-1, Nx_act_DM * (Nx_act_DM-1), -1]

    cmds1D = []
    for arr in cmds:
        flat_arr = arr.flatten()  # Flatten the 2D array
        remaining_values = np.delete(flat_arr, corner_indices)  # Remove specified indices
        cmds1D.append(remaining_values.tolist())  # Convert back to list


    idm = [I2M @ i.reshape(-1) for i in imgs]

    # look at the correlation between the DM command and the interpolated intensity (Pearson R) 
    R_list = []
    for act in range(140):
        R_list.append( pearsonr([a[act] for a in idm ], [a[act] for a in cmds1D]).statistic )


    plt.figure() 
    plt.imshow( util.get_DM_command_in_2D( R_list ) )
    plt.colorbar(label='Pearson R') 
    plt.title( 'Pearson R between DM command and \ninterpolated intensity onto DM actuator space')
    #plt.savefig('delme.png') # fig_path + f'pearson_r_dmcmd_dmIntensity_dm_interactuator_coupling-{zwfs_ns.dm.actuator_coupling_factor}.png')
    plt.show()  
    plt.close() 

    # xtmp=np.
    # actuator_filter = np.zeros([12,12]).astype(bool)


    X = process_signal( idm ,  I0dm,  N0dm ) 
    Y = np.array(cmds1D)

    if reconstructor_model == 'zonal_linear':
        coe, interc, res = [],[], []
        for a in range(len(X.T)):
            M, c = np.polyfit( X.T[a], Y.T[a], 1 )
            coe.append( M )
            interc.append( c )

            res.append( Y.T[a] -  M * X.T[a] + c  )

    else:
        raise UserWarning('not a valid model. Try zonal_linear')
        
    # simple check on center actuator 
    plt.plot( X.T[65], Y.T[65] , '.' ); plt.plot( X.T[65], coe[65] * X.T[65] + interc[65], ls='-') ; plt.savefig('delme.png') ; plt.show()

    fig, ax = DM_actuator_mosaic_plot( X, Y , filter_crosscoupling = False )
    plt.savefig('delme2.png')

    # we filter a little tighter (4 actuator radius) because edge effects are bad 
    if act_filter_method == 'pearson_R' :
        act_filt = ( np.array( R_list ) > model_metric_threshold ) 
    elif act_filter_method == 'residuals' :
        act_filt = ( np.std(res,axis=1) > model_metric_threshold ) 
    elif 'act_rad' in act_filter_method:
        # pattern 'act_rad-<actuator radius>'
        rad = int(act_filter_method.split('-')[-1])
        act_filt = util.get_circle_DM_command(radius= rad, Nx_act=12).astype(bool) 
    else:
        print('ALERT: Using set DM actuator radius = 4 actuators for control region')
        act_filt = util.get_circle_DM_command(radius= 5, Nx_act=12).astype(bool)
    
    plt.close('all')

    plt.figure()
    plt.imshow( util.get_DM_command_in_2D( np.std(res,axis=1) ) ) ;
    plt.colorbar(); 
    plt.show()
    plt.title(f'residuals {reconstructor_model}')
    plt.savefig('delme1.png')

    #res_cov =  np.cov( res ) 
    M2C = [coe, interc]

    # write to tomlo 


    ### Add to toml 
    new_data = {
        f"beam{beam_id}": 
            {f"{args.phasemask}": {
                "reconstructor_model":{
                    "reconstructor_model":reconstructor_model,
                    "model_metric_threshold":model_metric_threshold,
                    "act_filter_method":act_filter_method,
                    "linear_model_dm_actuator_filter": act_filt.tolist() ,
                    "linear_model_coes": np.array(coe).tolist(),
                    "linear_model_interc" : np.array(interc).tolist(),
                    "linear_model_train_residuals" : np.array(res).tolist(),
                    "M2C" : np.array(M2C).tolist(),
                    "I0" :np.array(I0).tolist() ,
                    "N0" :np.array(N0).tolist() 
                }
            }
        }
    }

    # with open("/Users/bencb/Documents/test.toml", "w") as f:
    #     toml.dump(new_data, f)

    # Check if file exists; if so, load and update.
    if os.path.exists(args.toml_file.replace('#',f'{beam_id}')):
        try:
            current_data = toml.load(args.toml_file.replace('#',f'{beam_id}'))
        except Exception as e:
            raise UserWarning(f"Error loading TOML file: {e}")
            #current_data = {}
    else:
        raise UserWarning(f"Error loading TOML file:")
        #current_data = {}

    # Update current data with new_data (beam specific)
    current_data = recursive_update(current_data, new_data)
 
 
    # Write the updated data back to the TOML file.
    with open(args.toml_file.replace('#',f'{beam_id}'), "w") as f:
        toml.dump(current_data, f)



## Make this a seperate file 

# ##### END 

#### Get some noise estimates 
# """
# Distribution of signal (idm(t) - I0dm) / N0dm 
# Heatmap of std projected on DM actuators for beam 2, fitted coefficients m , and therefore Model noise cmd = |m| * sigma """


# b = 2 
# r1,r2,c1,c2 = baldr_pupils[f"{b}"]
# I0 =  np.array( zwfs_pupils[b] )
# N0 =  np.array( np.mean( clear_pupils[b], axis=0) ) # fix later

# i = []
# for _ in range(100):
#     i.append( np.mean( c.get_data() ,axis=0 )[r1:r2,c1:c2] )


# i0dm = I2M_dict[b] @ I0.reshape(-1)
# n0dm = I2M_dict[b] @ N0.reshape(-1)
# idm = np.array( [I2M_dict[b] @ ii.reshape(-1) for ii in i] )

# imgs = np.std( ( idm - np.mean(idm,axis=0) ) / n0dm, axis = 0 )

# xlabel_list=['']
# ylabel_list=['']
# title_list = ['']
# cbar_label_list = ['std (I-I0)/N0']
# util.nice_heatmap_subplots( [util.get_DM_command_in_2D( imgs ) ], xlabel_list=xlabel_list, ylabel_list=ylabel_list , title_list=title_list, cbar_label_list=cbar_label_list, savefig='delme.png')


# play 

# d = fits.open( "~/Downloads/" + f"kolmogorov_calibration_{beam_id}.fits" )

# I2M = d["I2M"].data[0]
# imgs = d["SEQUENCE_IMGS"].data[:,0,:,:]
# cmds = d["DM_CMD_SEQUENCE"].data 
# I0 =  np.mean( d["FPM_IN"].data, axis = 0)  
# N0 =  np.mean( d["FPM_OUT"].data ,axis=0)
# Nx_act_DM = 12 
# corner_indices = [0, Nx_act_DM-1, Nx_act_DM * (Nx_act_DM-1), -1]

# cmds1D = []
# for arr in cmds:
#     flat_arr = arr.flatten()  # Flatten the 2D array
#     remaining_values = np.delete(flat_arr, corner_indices)  # Remove specified indices
#     cmds1D.append(remaining_values.tolist())  # Convert back to list


# idm = [I2M @ (( i.reshape(-1) - I0.reshape(-1) ) / N0.reshape(-1)) for i in imgs]

# plt.figure(); 
# plt.plot( [c[65] for c in cmds1D], [i[65] for i in idm], '.' ); 
# plt.xlabel('dm cmd (act 65)')
# plt.ylabel('interpolated intensity (act 65)')
# plt.title('Kolmogorov phase screen on DM')
# plt.savefig('delme.png')
# #plt.imshow(util.get_DM_command_in_2D( I2M_dict[beam_id] @ image_list[beam_id][30][0].reshape(-1)) ) ;plt.savefig('delme.png')


# #display_images_as_movie(image_lists=[imgs, cmds], plot_titles=None, cbar_labels=None,save_path="output_movie.mp4", fps=5)
# #display_images_as_movie(image_lists=[imgs, cmds], plot_titles=None, cbar_labels=None, save_path="users/bencb/Downloads/output_movie.mp4", fps=5)





# def display_images_with_slider(image_lists, plot_titles=None, cbar_labels=None):
#     """
#     Displays multiple images or 1D plots from a list of lists with a slider to control the shared index.
    
#     Parameters:
#     - image_lists: list of lists where each inner list contains either 2D arrays (images) or 1D arrays (scalars).
#                    The inner lists must all have the same length.
#     - plot_titles: list of strings, one for each subplot. Default is None (no titles).
#     - cbar_labels: list of strings, one for each colorbar. Default is None (no labels).
#     """
#     import math
#     # Check that all inner lists have the same length
#     assert all(len(lst) == len(image_lists[0]) for lst in image_lists), "All inner lists must have the same length."
    
#     # Number of rows and columns based on the number of plots
#     num_plots = len(image_lists)
#     ncols = math.ceil(math.sqrt(num_plots))  # Number of columns for grid
#     nrows = math.ceil(num_plots / ncols)     # Number of rows for grid
    
#     num_frames = len(image_lists[0])

#     # Create figure and axes
#     fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(5 * ncols, 5 * nrows))
#     plt.subplots_adjust(bottom=0.2)

#     # Flatten axes array for easier iteration
#     axes = axes.flatten() if num_plots > 1 else [axes]

#     # Store the display objects for each plot (either imshow or line plot)
#     img_displays = []
#     line_displays = []
    
#     # Get max/min values for 1D arrays to set static axis limits
#     max_values = [max(lst) if not isinstance(lst[0], np.ndarray) else None for lst in image_lists]
#     min_values = [min(lst) if not isinstance(lst[0], np.ndarray) else None for lst in image_lists]

#     for i, ax in enumerate(axes[:num_plots]):  # Only iterate over the number of plots
#         # Check if the first item in the list is a 2D array (an image) or a scalar
#         if isinstance(image_lists[i][0], np.ndarray) and image_lists[i][0].ndim == 2:
#             # Use imshow for 2D data (images)
#             img_display = ax.imshow(image_lists[i][0], cmap='viridis')
#             img_displays.append(img_display)
#             line_displays.append(None)  # Placeholder for line plots
            
#             # Add colorbar if it's an image
#             cbar = fig.colorbar(img_display, ax=ax)
#             if cbar_labels and i < len(cbar_labels) and cbar_labels[i] is not None:
#                 cbar.set_label(cbar_labels[i])

#         else:
#             # Plot the list of scalar values up to the initial index
#             line_display, = ax.plot(np.arange(len(image_lists[i])), image_lists[i], color='b')
#             line_display.set_data(np.arange(1), image_lists[i][:1])  # Start with only the first value
#             ax.set_xlim(0, len(image_lists[i]))  # Set x-axis to full length of the data
#             ax.set_ylim(min_values[i], max_values[i])  # Set y-axis to cover the full range
#             line_displays.append(line_display)
#             img_displays.append(None)  # Placeholder for image plots

#         # Set plot title if provided
#         if plot_titles and i < len(plot_titles) and plot_titles[i] is not None:
#             ax.set_title(plot_titles[i])

#     # Remove any unused axes
#     for ax in axes[num_plots:]:
#         ax.remove()

#     # Slider for selecting the frame index
#     ax_slider = plt.axes([0.2, 0.05, 0.65, 0.03], facecolor='lightgoldenrodyellow')
#     frame_slider = Slider(ax_slider, 'Frame', 0, num_frames - 1, valinit=0, valstep=1)

#     # Update function for the slider
#     def update(val):
#         index = int(frame_slider.val)  # Get the selected index from the slider
#         for i, (img_display, line_display) in enumerate(zip(img_displays, line_displays)):
#             if img_display is not None:
#                 # Update the image data for 2D data
#                 img_display.set_data(image_lists[i][index])
#             if line_display is not None:
#                 # Update the line plot for scalar values (plot up to the selected index)
#                 line_display.set_data(np.arange(index), image_lists[i][:index])
#         fig.canvas.draw_idle()  # Redraw the figure

#     # Connect the slider to the update function
#     frame_slider.on_changed(update)

#     plt.show()




# def display_images_as_movie(image_lists, plot_titles=None, cbar_labels=None, save_path="output_movie.mp4", fps=5):
#     """
#     Creates an animation from multiple images or 1D plots from a list of lists and saves it as a movie.
    
#     Parameters:
#     - image_lists: list of lists where each inner list contains either 2D arrays (images) or 1D arrays (scalars).
#                    The inner lists must all have the same length.
#     - plot_titles: list of strings, one for each subplot. Default is None (no titles).
#     - cbar_labels: list of strings, one for each colorbar. Default is None (no labels).
#     - save_path: path where the output movie will be saved.
#     - fps: frames per second for the output movie.
#     """
#     import math 
#     # Check that all inner lists have the same length
#     assert all(len(lst) == len(image_lists[0]) for lst in image_lists), "All inner lists must have the same length."
    
#     # Number of rows and columns based on the number of plots
#     num_plots = len(image_lists)
#     ncols = math.ceil(math.sqrt(num_plots))  # Number of columns for grid
#     nrows = math.ceil(num_plots / ncols)     # Number of rows for grid
    
#     num_frames = len(image_lists[0])

#     # Create figure and axes
#     fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(5 * ncols, 5 * nrows))
#     plt.subplots_adjust(bottom=0.2)

#     # Flatten axes array for easier iteration
#     axes = axes.flatten() if num_plots > 1 else [axes]

#     # Store the display objects for each plot (either imshow or line plot)
#     img_displays = []
#     line_displays = []
    
#     # Get max/min values for 1D arrays to set static axis limits
#     max_values = [max(lst) if not isinstance(lst[0], np.ndarray) else None for lst in image_lists]
#     min_values = [min(lst) if not isinstance(lst[0], np.ndarray) else None for lst in image_lists]

#     for i, ax in enumerate(axes[:num_plots]):  # Only iterate over the number of plots
#         # Check if the first item in the list is a 2D array (an image) or a scalar
#         if isinstance(image_lists[i][0], np.ndarray) and image_lists[i][0].ndim == 2:
#             # Use imshow for 2D data (images)
#             img_display = ax.imshow(image_lists[i][0], cmap='viridis')
#             img_displays.append(img_display)
#             line_displays.append(None)  # Placeholder for line plots
            
#             # Add colorbar if it's an image
#             cbar = fig.colorbar(img_display, ax=ax)
#             if cbar_labels and i < len(cbar_labels) and cbar_labels[i] is not None:
#                 cbar.set_label(cbar_labels[i])

#         else:
#             # Plot the list of scalar values up to the initial index
#             line_display, = ax.plot(np.arange(len(image_lists[i])), image_lists[i], color='b')
#             line_display.set_data(np.arange(1), image_lists[i][:1])  # Start with only the first value
#             ax.set_xlim(0, len(image_lists[i]))  # Set x-axis to full length of the data
#             ax.set_ylim(min_values[i], max_values[i])  # Set y-axis to cover the full range
#             line_displays.append(line_display)
#             img_displays.append(None)  # Placeholder for image plots

#         # Set plot title if provided
#         if plot_titles and i < len(plot_titles) and plot_titles[i] is not None:
#             ax.set_title(plot_titles[i])

#     # Remove any unused axes
#     for ax in axes[num_plots:]:
#         ax.remove()

#     # Function to update the frames
#     def update_frame(frame_idx):
#         for i, (img_display, line_display) in enumerate(zip(img_displays, line_displays)):
#             if img_display is not None:
#                 # Update the image data for 2D data
#                 img_display.set_data(image_lists[i][frame_idx])
#             if line_display is not None:
#                 # Update the line plot for scalar values (plot up to the current index)
#                 line_display.set_data(np.arange(frame_idx), image_lists[i][:frame_idx])
#         return img_displays + line_displays

#     # Create the animation
#     ani = animation.FuncAnimation(fig, update_frame, frames=num_frames, blit=False, repeat=False)

#     # Save the animation as a movie file
#     ani.save(save_path, fps=fps, writer='ffmpeg')

#     plt.show()



