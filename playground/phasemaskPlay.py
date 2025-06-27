
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
from scipy.ndimage import median_filter
from xaosim.shmlib import shm
import asgard_alignment.controllino as co # for turning on / off source \
from asgard_alignment.DM_shm_ctrl import dmclass
import common.DM_registration as DM_registration
import common.DM_basis_functions as dmbases
import common.phasemask_centering_tool as pct
from pyBaldr import utilities as util
from asgard_alignment import FLI_Cameras as FLI

try:
    from asgard_alignment import controllino as co
    myco = co.Controllino('172.16.8.200')
    controllino_available = True
    print('controllino connected')
    
except:
    print('WARNING Controllino cannot connect. WILL NOT MOVE SOURCE OUT FOR DARK')
    controllino_available = False 

#################################


### using SHM camera structure
def move_relative_and_get_image(cam, beam, baldr_pupils, phasemask, savefigName=None, use_multideviceserver=True,roi=[None,None,None,None]):
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

                time.sleep(0.5)
                img = np.mean(
                    cam.get_data(),
                    axis=0,
                )[r1:r2,c1:c2]
                if savefigName != None:
                    plt.figure()
                    plt.imshow( np.log10( img[roi[0]:roi[1],roi[2]:roi[3]] ) )
                    plt.colorbar()
                    plt.savefig(savefigName)
            except:
                print('incorrect input. Try input "1,1" as an example, or "e" to exit')

    plt.close("all")



def spiral_square_search_and_save_images(
    cam,
    beam,
    baldr_pupils,
    phasemask,
    starting_point,
    step_size,
    search_radius,
    sleep_time=1,
    use_multideviceserver=True,
):
    """
    Perform a spiral square search pattern to find the best position for the phase mask.
    if use_multideviceserver is True, the function will use ZMQ protocol to communicate with the
    MultiDeviceServer to move the phase mask. In this case phasemask should be the socket for the ZMQ protocol.
    Otherwise, it will move the phase mask directly and phasemask shold be the BaldrPhaseMask object.
    """
    r1,r2,c1,c2 = baldr_pupils[f"{beam}"]
    spiral_pattern = pct.square_spiral_scan(starting_point, step_size, search_radius)

    x_points, y_points = zip(*spiral_pattern)
    img_dict = {}

    for i, (x_pos, y_pos) in enumerate(zip(x_points, y_points)):
        print("at ", x_pos, y_pos)
        print(f"{100 * i/len(x_points)}% complete")

        # motor limit safety checks!
        if x_pos <= 0:
            print('x_pos < 0. set x_pos = 1')
            x_pos = 1
        if x_pos >= 10000:
            print('x_pos > 10000. set x_pos = 9999')
            x_pos = 9999
        if y_pos <= 0:
            print('y_pos < 0. set y_pos = 1')
            y_pos = 1
        if y_pos >= 10000:
            print('y_pos > 10000. set y_pos = 9999')
            y_pos = 9999

        if use_multideviceserver:
            #message = f"fpm_moveabs phasemask{beam} {[x_pos, y_pos]}"
            message = f"moveabs BMX{beam} {x_pos}"
            phasemask.send_string(message)
            response = phasemask.recv_string()
            print(response)

            message = f"moveabs BMY{beam} {y_pos}"
            phasemask.send_string(message)
            response = phasemask.recv_string()
            print(response)
        else:
            phasemask.move_absolute([x_pos, y_pos])

        time.sleep(sleep_time)  # wait for the phase mask to move and settle
        img = np.mean(
            cam.get_data(),
            axis=0,
        )[r1:r2,c1:c2]

        img_dict[(x_pos, y_pos)] = img

    return img_dict


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



parser = argparse.ArgumentParser(description="Baldr Pupil Fit Configuration.")

default_toml = os.path.join( "config_files", "baldr_config.toml") #os.path.dirname(os.path.abspath(__file__)), "..", "config_files", "baldr_config.toml")

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
    default=[2], #, 2, 3, 4],
    help="Comma-separated beam IDs to apply. Default: 1,2,3,4"
)

# Plot: default is True, with an option to disable.
parser.add_argument(
    "--plot", 
    dest="plot",
    action="store_true",
    default=True,
    help="Enable plotting (default: True)"
)


args=parser.parse_args()






# inputs 
number_of_pokes = 100 
poke_amplitude = 0.02
dm_4_corners = DM_registration.get_inner_square_indices(outer_size=12, inner_offset=4) # flattened index of the DM actuator 
dm_turbulence = False # roll phasescreen on DM?
#all_dm_shms_list = [args.dm1_shm, args.dm2_shm, args.dm3_shm, args.dm4_shm]

assert hasattr(args.beam_id , "__len__")
assert len(args.beam_id) <= 4
assert max(args.beam_id) <= 4
assert min(args.beam_id) >= 1 


# set up subpupils and pixel mask
with open(args.toml_file ) as file:
    pupildata = toml.load(file)
    # Extract the "baldr_pupils" section
    baldr_pupils = pupildata.get("baldr_pupils", {})

    # the registered pupil mask for each beam (in the local frame)
    pupil_masks={}
    for beam_id in args.beam_id:
        pupil_masks[beam_id] = pupildata.get(f"beam{beam_id}", {}).get("pupil_mask", {}).get("mask", None) #pupildata.get(f"beam{beam_id}.pupil_mask.mask")
        if pupil_masks[beam_id] is None:
            raise UserWarning(f"pupil mask returned none in toml file. check for beam{beam_id}.pupil_mask.mask in the file:{args.toml_file}")


# global camera image shm 
roi = [None for _ in range(4)]
c = FLI.fli(roi=roi) # #shm(args.global_camera_shm)

# DMs
dm_shm_dict = {}
for beam_id in args.beam_id:
    dm_shm_dict[beam_id] = dmclass( beam_id=beam_id )
    # zero all channels
    dm_shm_dict[beam_id].zero_all()
    # activate flat 
    dm_shm_dict[beam_id].activate_flat()


c.send_fli_cmd("set fps 100")
c.send_fli_cmd("set gain 5")


# try get dark and build bad pixel mask 
# if controllino_available:
    
#     myco.turn_off("SBB")
#     time.sleep(2)
    
#     dark_raw = c.get_data()

#     myco.turn_on("SBB")
#     time.sleep(2)

#     bad_pixel_mask = get_bad_pixel_indicies( dark_raw, std_threshold = 20, mean_threshold=6)
# else:
#     dark_raw = c.get_data()

#     bad_pixel_mask = get_bad_pixel_indicies( dark_raw, std_threshold = 20, mean_threshold=6)


context = zmq.Context()

context.socket(zmq.REQ)

socket = context.socket(zmq.REQ)

socket.setsockopt(zmq.RCVTIMEO, args.timeout)

server_address = f"tcp://{args.host}:{args.port}"

socket.connect(server_address)

state_dict = {"message_history": [], "socket": socket}


phasemask_name = 'J4'
beam_id = 1

# # get all available files 
valid_reference_position_files = glob.glob(
    f"/home/asg/Progs/repos/asgard-alignment/config_files/phasemask_positions/beam{beam_id}/*json"
    )

# read in the most recent and make initial posiition the most recent one for given mask 
with open(max(valid_reference_position_files, key=os.path.getmtime)
, "r") as file:
    start_position_dict = json.load(file)

    Xpos0 = start_position_dict[phasemask_name][0]
    Ypos0 = start_position_dict[phasemask_name][1]

    print(f'starting at {phasemask_name} from {file}\npos={Xpos0, Ypos0}')

# # current position 
# message = f"read BMX{beam_id}"
# Xpos = float(send_and_get_response(message))

# message = f"read BMY{beam_id}"
# Ypos = float(send_and_get_response(message))

# print(Xpos, Ypos)
#X~8390, Y1~1400, Y2 ~2400


Xpos0, Ypos0 =  2759.9878124999987, 4010.0011874999987
search_dict = spiral_square_search_and_save_images(
    cam=c,
    beam=beam_id,
    baldr_pupils=baldr_pupils,
    phasemask=state_dict["socket"],
    starting_point=[Xpos0, Ypos0], #[Xpos0, Ypos0],
    step_size=20,
    search_radius=300,
    sleep_time=1,
    use_multideviceserver=True,
)

# for small grid searches this is good for simple manual check
pct.plot_image_grid(search_dict, savepath='delme.png')

# larger searches best to use cluster analuysis 
image_list = np.array( list(search_dict.values() ) ) 
res = pct.cluster_analysis_on_searched_images(images= image_list,
                                          detect_circle_function=pct.detect_circle, 
                                          n_clusters=6, 
                                          plot_clusters=False)


positions = [eval(str(key)) for key in search_dict.keys()]
x_positions, y_positions = zip(*positions)
plot_cluster_heatmap( x_positions,  y_positions ,  res['clusters'] ) 
plt.savefig('delme.png')




# Current position BMX1: 8408.003249999996 um
# Current position BMY1: 2434.9948124999987 um

#Current position BMX1: 8388.000749999996 um
#Current position BMY1: 1424.9876249999993 um

#Current position BMX1: 8367.998249999997 um
#Current position BMY1: 424.9816874999998 um

posss = {1:[8367.9,424.9 ],2:[8388.0, 1424.9], 3:[8408.0,2434.9]}
allposs = pct.complete_collinear_points(known_points=posss, separation=1000, tolerance=20)



# check and manually move to best 
message = f"moveabs BMX{beam_id} 1090.0"
send_and_get_response(message)
time.sleep(2)

message = f"moveabs BMY{beam_id} 2060.0"
send_and_get_response(message)


move_relative_and_get_image(cam=c, beam=2,baldr_pupils=baldr_pupils, phasemask=state_dict["socket"], savefigName='delme.png', use_multideviceserver=True)


amp=0.03
imlist = [] 
amps = np.linspace( -0.1, 0.1, 36)
for amp in amps:
    print(amp)
    zbasis = dmbases.zer_bank(1, 10 )
    bb=2
    dm_shm_dict[bb].set_data( amp * zbasis[3] )
    #dm_shm_dict[bb].shms[2].set_data(amp * zbasis[3])
    time.sleep(1)
    img = np.mean( c.get_data() ,axis = 0 ) 
    r1,r2,c1,c2 = baldr_pupils[f"{bb}"]
    imlist.append( img[r1:r2, c1:c2] )
    print( np.mean( img[r1:r2, c1:c2][strehl_mask[beam_id]]))

fig,ax = plt.subplots( 5,5, figsize=(15,15))
for i, a, axx in zip( imlist, amps, ax.reshape(-1)):
    axx.imshow(i ) 
    axx.set_title( f'amp={round(a,3)}')
plt.savefig('delme1.png') 

### Try get 



amp = 0.02
dm_shm_dict[beam_id].set_data( amp * zbasis[3] )
#dm_shm_dict[bb].shms[2].set_data(amp * zbasis[3])
time.sleep(1)
img = np.mean( c.get_data() ,axis = 0 ) 
r1,r2,c1,c2 = baldr_pupils[f"{beam_id}"]




# mask in 
zwfs_pupils = {}
img = np.mean( c.get_data() ,axis=0) 
for bb in [beam_id]:
    r1,r2,c1,c2 = baldr_pupils[f"{bb}"]
    #cropped_img = interpolate_bad_pixels(img[r1:r2, c1:c2], bad_pixel_mask[r1:r2, c1:c2])
    cropped_img = img[r1:r2, c1:c2] #/np.mean(img[r1:r2, c1:c2][pupil_masks[bb]])
    zwfs_pupils[bb] = cropped_img


# get initial clear pupil
rel_offset = 500.0
clear_pupils = {}

message = f"moverel BMX{beam_id} {rel_offset}"
res = send_and_get_response(message)
print(res) 

message = f"moverel BMY{beam_id} {rel_offset}"
res = send_and_get_response(message)
print(res) 

time.sleep(5)

img = np.mean( c.get_data() ,axis=0) 
for bb in [beam_id]:
    r1,r2,c1,c2 = baldr_pupils[f"{bb}"]
    #cropped_img = interpolate_bad_pixels(img[r1:r2, c1:c2], bad_pixel_mask[r1:r2, c1:c2])
    cropped_img = img[r1:r2, c1:c2] #/np.mean(img[r1:r2, c1:c2][pupil_masks[bb]])
    clear_pupils[bb] = cropped_img
#plt.figure();plt.imshow(clear_pupils[bb]);plt.savefig('delme1.png')

time.sleep(5)

message = f"moverel BMX{beam_id} {-rel_offset}"
res = send_and_get_response(message)
print(res) 

message = f"moverel BMY{beam_id} {-rel_offset}"
res = send_and_get_response(message)
print(res) 


norm = np.mean(clear_pupils[beam_id][pupil_masks[beam_id]])
imgs = [zwfs_pupils[beam_id] / norm , clear_pupils[beam_id] / norm, (zwfs_pupils[beam_id] - clear_pupils[beam_id])/norm]
titles = ['I0', 'N0', r'$\Delta$']
cbars = ['adu', 'adu', r'$\Delta$adu']
xlabel_list, ylabel_list = ['','',''], ['','','']
util.nice_heatmap_subplots(im_list=imgs , title_list=titles,xlabel_list=xlabel_list, ylabel_list=ylabel_list, cbar_label_list=cbars, fontsize=15, cbar_orientation = 'bottom', axis_off=True, vlims=None, savefig='delme.png')


## get strehl mask 

deltaI = {}
strehl_mask = {} 

deltaI[beam_id] = (zwfs_pupils[beam_id] - clear_pupils[beam_id])/norm 
# filter out central pupil 
deltaI[beam_id][pupil_masks[beam_id]] = 0
strehl_mask[beam_id] = deltaI[beam_id] > 2 * np.std(deltaI[beam_id] )
plt.figure(); plt.imshow(deltaI[beam_id]); plt.colorbar() ; plt.savefig('delme.png')
plt.figure(); plt.imshow(strehl_mask[beam_id]); plt.colorbar() ; plt.savefig('delme.png')


## ok again scan focus and see where max Strehl?


amp=0.03
Strehllist = [] 

amps = np.linspace( -0.1, 0.1, 40)
for amp in amps:
    print(amp)
    zbasis = dmbases.zer_bank(1, 10 )
    bb=2
    dm_shm_dict[bb].set_data( amp * zbasis[3] )
    #dm_shm_dict[bb].shms[2].set_data(amp * zbasis[3])
    time.sleep(1)
    img = np.mean( c.get_data() ,axis = 0 ) 
    r1,r2,c1,c2 = baldr_pupils[f"{bb}"]
    Strehllist.append( np.mean( img[r1:r2, c1:c2][strehl_mask[beam_id]]))
    print( Strehllist[-1] )


plt.figure(); plt.plot( amps, Strehllist)
plt.xlabel('focus amp')
plt.ylabel('Mean Exterior Pixels') 
plt.savefig('delme.png')



### FINE 
imlist = [] 
slist = []
amps = np.linspace( 0.01, 0.03, 25)
for amp in amps:
    print(amp)
    zbasis = dmbases.zer_bank(1, 10 )
    bb=2
    dm_shm_dict[bb].set_data( amp * zbasis[3] )
    #dm_shm_dict[bb].shms[2].set_data(amp * zbasis[3])
    time.sleep(2)
    img = np.mean( c.get_data() ,axis = 0 ) 
    r1,r2,c1,c2 = baldr_pupils[f"{bb}"]
    imlist.append( img[r1:r2, c1:c2] )
    slist.append( np.mean( img[r1:r2, c1:c2][strehl_mask[beam_id]]) )
    print( np.mean( img[r1:r2, c1:c2][strehl_mask[beam_id]]))



plt.figure(); plt.plot( amps, slist)
plt.xlabel('focus amp')
plt.ylabel('Mean Exterior Pixels') 
plt.savefig('delme.png')

fig,ax = plt.subplots( 5,5, figsize=(15,15))
for i, a, axx in zip( imlist, amps, ax.reshape(-1)):
    axx.imshow(i ) 
    axx.set_title( f'amp={round(a,3)}')
plt.savefig('delme1.png') 



best_amp = amps[ np.argmax( slist ) ]

dm_shm_dict[beam_id].set_data( best_amp * zbasis[3] )

ab = 0.05 * zbasis[4] + best_amp * zbasis[3] 

dm_shm_dict[beam_id].set_data(  best_amp * zbasis[3] )

time.sleep(5 )
img = np.mean( c.get_data() ,axis = 0 ) 
r1,r2,c1,c2 = baldr_pupils[f"{bb}"]
I0 =  img[r1:r2, c1:c2] 

mode_i = np.arange(1,5)
imgs = []
aber = []
for i in mode_i:
    print(i) 
    ab = 0.05 * zbasis[i] 
    aber.append(ab)
    dm_shm_dict[beam_id].set_data( ab + best_amp * zbasis[3]  )
    time.sleep(10)
    img = np.mean( c.get_data() ,axis = 0 ) 
    r1,r2,c1,c2 = baldr_pupils[f"{bb}"]
    imgs.append( img[r1:r2, c1:c2] )


fig,ax = plt.subplots( len(aber), 2 , figsize=(4,12) )
for i in range(len(aber)):
    ax[i,0].imshow( aber[i] )
    ax[i,1].imshow( imgs[i] - I0 )
plt.savefig('delme.png')


### Add to toml 
new_data = {
    f"beam{beam_id}": {
        "strehl_mask": strehl_mask[beam_id] ,
        "DM_flat_offset": (best_amp * zbasis[3]).tolist()
    }
}

# Check if file exists; if so, load and update.
if os.path.exists(args.toml_file):
    try:
        current_data = toml.load(args.toml_file)
    except Exception as e:
        raise UserWarning(f"Error loading TOML file: {e}")
        #current_data = {}
else:
    raise UserWarning(f"Error loading TOML file: {e}")
    #current_data = {}

# Update current data with new_data (beam specific)
current_data.update(new_data)

# Write the updated data back to the TOML file.
with open(args.toml_file, "w") as f:
    toml.dump(current_data, f)


