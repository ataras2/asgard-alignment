
#!/usr/bin/env python
import zmq
import numpy as np
import toml  # Make sure to install via `pip install toml` if needed
import argparse
import os
import json
import time
import matplotlib.pyplot as plt
from scipy.interpolate import RegularGridInterpolator
from pyBaldr import utilities as util
from asgard_alignment import FLI_Cameras as FLI
from asgard_alignment.DM_shm_ctrl import dmclass
from common import phasemask_centering_tool as pct

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



parser = argparse.ArgumentParser(description="Controller for fine phasemask alignment using BMX/BMY motors")


######## HARD CODED 
hc_fps = 500
hc_gain = 15
default_toml = os.path.join("/usr/local/etc/baldr/", "baldr_config_#.toml") 

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
    help="TOML file pattern (replace # with beam_id) to write/edit. Default: ../config_files/baldr_config.toml (relative to script)"
)

# Beam ids: provided as a comma-separated string and converted to a list of ints.
parser.add_argument(
    "--beam_id",
    type=lambda s: [int(item) for item in s.split(",")],
    default=[2], # 1, 2, 3, 4],
    help="Comma-separated beam IDs to apply. Default: 1,2,3,4"
)

parser.add_argument("--fig_path", 
                    type=str, 
                    default=None, 
                    help="path/to/output/image/ for the saved figures")


parser.add_argument(
    "--phasemask_gain",
    type=float,
    default=2500 , #1e6 * 0.5,
    help="gain for phasemask feedback control (um/(ADU/s)). Dont be shy, this should be a high number, and we check and limit movement . Default: %(defult)s"
)

parser.add_argument(
    "--dither_amp",
    type=float,
    default=6,
    help="Ampitude (um) of phasemask motor dither amp for gradient descent. Default: %(defult)s"
)
parser.add_argument(
    "--sleeptime",
    type=float,
    default=200/500,
    help="sleeptime between phase mask movements. Default: %(defult)s"
)


parser.add_argument(
    "--method",
    type=str,
    default="gradient_descent",
    help="method for fine alignment. gradient_descent or brute_scan. Default: %(defult)s"
)


# parser.add_argument(
#     "--phasemask",
#     type=str,
#     default="H3",
#     help="Comma-separated beam IDs to apply. Default: 1,2,3,4"
# )

# parser.add_argument(
#     "--poke_amp",
#     type=float,
#     default=10,
#     help="amplitude (micros) to move phasemask motors for building interaction matrix"
# )


# parser.add_argument(
#     "--inverse_method",
#     type=str,
#     default="pinv",
#     help="Method used for inverting interaction matrix to build control (intensity-mode) matrix I2M"
# )

parser.add_argument("--host", type=str, default="172.16.8.6", help="Server host")#"localhost"
parser.add_argument("--port", type=int, default=5555, help="Server port")
parser.add_argument(
    "--timeout", type=int, default=5000, help="Response timeout in milliseconds"
)



#########################


args=parser.parse_args()


# set up commands to move motors phasemask
context = zmq.Context()
context.socket(zmq.REQ)
socket = context.socket(zmq.REQ)
socket.setsockopt(zmq.RCVTIMEO, args.timeout)
server_address = f"tcp://{args.host}:{args.port}"
socket.connect(server_address)
state_dict = {"message_history": [], "socket": socket}


pupil_masks = {}
exterior_filter = {}
sec_filter = {}
for beam_id in args.beam_id:

    # read in TOML as dictionary for config 
    with open(args.toml_file.replace('#',f'{beam_id}'), "r") as f:
        config_dict = toml.load(f)
        # Baldr pupils from global frame 
        baldr_pupils = config_dict['baldr_pupils'] 
        # exterior strehl pixels
        exterior_filter[beam_id] = config_dict[f"beam{beam_id}"]["pupil_mask"]["exterior"]
        # secondary obstruction pixels
        sec_filter[beam_id] = config_dict[f"beam{beam_id}"]["pupil_mask"]["secondary"]


c_dict = {}
for beam_id in args.beam_id:
    r1,r2,c1,c2 = baldr_pupils[f'{beam_id}']
    c_dict[beam_id] = FLI.fli(args.global_camera_shm, roi = [r1,r2,c1,c2])


## get initial gain and fps
fps0 = FLI.extract_value( c_dict[beam_id].send_fli_cmd( "fps raw" ) ) 
gain0 = FLI.extract_value( c_dict[beam_id].send_fli_cmd( "gain raw" ) ) 

#####################
#Hard coded frame rate and gain 
#####################

# only set once (sin)
c_dict[args.beam_id[0]].send_fli_cmd(f"set fps {hc_fps}")
time.sleep(1)
c_dict[args.beam_id[0]].send_fli_cmd(f"set gain {hc_gain}")
time.sleep(1)

# can probably do it without reduction ... test later
# for beam_id in args.beam_id:
#     c_dict[beam_id].build_manual_bias(number_of_frames=500)
#     c_dict[beam_id].build_manual_dark(number_of_frames=500, 
#                                       apply_manual_reduction=True,
#                                       build_bad_pixel_mask=True, 
#                                       sleeptime = 3,
#                                       kwargs={'std_threshold':10, 'mean_threshold':6} )
  


# img = np.mean( c_dict[beam_id].get_some_frames(number_of_frames=100, apply_manual_reduction=True) , axis=0)
# title_list = ['bias','dark','img']
# im_list = [c_dict[beam_id].reduction_dict['bias'][-1], c_dict[beam_id].reduction_dict['dark'][-1], img]
# util.nice_heatmap_subplots( im_list, savefig='delme.png')


# set up DM SHMs 
print( 'setting up DMs')
dm_shm_dict = {}
for beam_id in args.beam_id:
    dm_shm_dict[beam_id] = dmclass( beam_id=beam_id )
    # zero all channels
    dm_shm_dict[beam_id].zero_all()
    
    # activate flat (does this on channel 1)
    #dm_shm_dict[beam_id].activate_flat()

    # apply dm flat + calibrated offset (does this on channel 1)
    dm_shm_dict[beam_id].activate_calibrated_flat()
    


initial_pos = {}
for beam_id in args.beam_id:
    message = f"read BMX{beam_id}"
    initial_Xpos = float(send_and_get_response(message))

    message = f"read BMY{beam_id}"
    initial_Ypos = float(send_and_get_response(message))
    
    # definition is [X,Y]
    initial_pos[beam_id] = [initial_Xpos, initial_Ypos]



# import concurrent.futures

# def brute_scan(beam_id):

#     result1 = pct.spiral_square_search_and_save_images(
#         cam=c_dict[beam_id],
#         beam=beam_id,
#         phasemask=state_dict["socket"],
#         starting_point=initial_pos[beam_id],
#         step_size=5,
#         search_radius=40,
#         sleep_time=0.1,
#         use_multideviceserver=True,
#     )

#     ib = np.argmax( [v[np.array(sec_filter[beam_id]).astype(bool)][4] for _,v in result1.items()])

#     Xb, Yb = list(  result1.keys() )[ib]

#     # go finer
#     result2 = pct.spiral_square_search_and_save_images(
#             cam=c_dict[beam_id],
#             beam=beam_id,
#             phasemask=state_dict["socket"],
#             starting_point=[Xb, Yb],
#             step_size=2,
#             search_radius=5,
#             sleep_time=0.1,
#             use_multideviceserver=True,
#         )

#     ib = np.argmax( [v[np.array(sec_filter[beam_id]).astype(bool)][4] for _,v in result2.items()])

#     Xb, Yb = list(  result2.keys() )[ib]

#     print(f"moving to best position at {Xb, Yb} for beam {beam_id}")
#     for m, p in zip(["BMX","BMY"], [Xb,Yb]):
#         time.sleep(1)
#         message = f"moveabs {m}{beam_id} {p}"
#         res = send_and_get_response(message)
#         print(res)

if args.method.lower() == "brute_scan":
    # Use ThreadPoolExecutor to run functions in parallel
    # with concurrent.futures.ThreadPoolExecutor(max_workers=len(args.beam_id)) as executor:
    #     # Map each input to the function and execute in parallel
    #     results = list(executor.map(brute_scan, args.beam_id))

    for beam_id in args.beam_id:
        result1 = pct.spiral_square_search_and_save_images(
            cam=c_dict[beam_id],
            beam=beam_id,
            phasemask=state_dict["socket"],
            starting_point=initial_pos[beam_id],
            step_size=5,
            search_radius=40,
            sleep_time=0.1,
            use_multideviceserver=True,
        )

        ib = np.argmax( [v[np.array(sec_filter[beam_id]).astype(bool)][4] for _,v in result1.items()])

        Xb, Yb = list(  result1.keys() )[ib]

        # go finer
        result2 = pct.spiral_square_search_and_save_images(
                cam=c_dict[beam_id],
                beam=beam_id,
                phasemask=state_dict["socket"],
                starting_point=[Xb, Yb],
                step_size=2,
                search_radius=5,
                sleep_time=0.1,
                use_multideviceserver=True,
            )

        ib = np.argmax( [v[np.array(sec_filter[beam_id]).astype(bool)][4] for _,v in result2.items()])

        Xb, Yb = list(  result2.keys() )[ib]

        print(f"moving to best position at {Xb, Yb} for beam {beam_id}")
        for m, p in zip(["BMX","BMY"], [Xb,Yb]):
            time.sleep(1)
            message = f"moveabs {m}{beam_id} {p}"
            res = send_and_get_response(message)
            print(res)

####################################################################
### FINE SPIRAL SEARCH 

# if args.method.lower() == "brute_scan":
#     beams_img_dict = {}
#     for beam_id in args.beam_id:

#         beams_img_dict[beam_id]= pct.spiral_square_search_and_save_images(
#             cam=c_dict[beam_id],
#             beam=beam_id,
#             phasemask=state_dict["socket"],
#             starting_point=initial_pos[beam_id],
#             step_size=5,
#             search_radius=40,
#             sleep_time=0.1,
#             use_multideviceserver=True,
#         )

#         ib = np.argmax( [v[np.array(sec_filter[beam_id]).astype(bool)][4] for _,v in beams_img_dict[beam_id].items()])

#         Xb, Yb = list(  beams_img_dict[beam_id].keys() )[ib]

#         # go finer
#         for beam_id in args.beam_id:
#             beams_img_dict[beam_id]= pct.spiral_square_search_and_save_images(
#                 cam=c_dict[beam_id],
#                 beam=beam_id,
#                 phasemask=state_dict["socket"],
#                 starting_point=[Xb, Yb],
#                 step_size=2,
#                 search_radius=5,
#                 sleep_time=0.1,
#                 use_multideviceserver=True,
#             )

#         ib = np.argmax( [v[np.array(sec_filter[beam_id]).astype(bool)][4] for _,v in beams_img_dict[beam_id].items()])

#         Xb, Yb = list(  beams_img_dict[beam_id].keys() )[ib]

#         print(f"moving to best position at {Xb, Yb} for beam {beam_id}")
#         for m, p in zip(["BMX","BMY"], [Xb,Yb]):
#             time.sleep(1)
#             message = f"moveabs {m}{beam_id} {p}"
#             res = send_and_get_response(message)
#             print(res)

####################################################################
### GRADIENT DESENT WITH DITHERING TECHNIQUE FOR GRADIENT ESTIMATION

elif args.method.lower() == "gradient_descent":
    for beam_id in args.beam_id:
        close = True 
        prev_pos = initial_pos[beam_id].copy()
        dz = args.dither_amp # dither amplitude
        cnt = 0 # limit number of iterations 
        max_iterations = 20 
        dx_prev = 0
        dy_prev = 0
        near_flag_passed = False  # flag to change dither amplitude if we are getting close! 

        # err_x = 0 
        # err_y = 0

        telem = {"i":[], "signal":[], "dx":[],"dy":[]}

        while close: 

            cnt += 1 
            dxtmp = []
            dytmp = []
            signaltmp2 = []
            for _ in range(2): # median of 2 iterations - can be noisy
                signaltmp = []

                #for _ in range(4): # could get the mean after a few iterations 
                for j, m in enumerate( ["BMX","BMX","BMY","BMY"] ):

                    sign = (-1)**j # direction of movement (+/-)
                    
                    z0 = prev_pos[int(j > 1)] # if j < 1 than we look at x (index = 0), else y (index = 1)

                    message = f"moveabs {m}{beam_id} { z0 + sign * dz }"
                    print(message)
                    res = send_and_get_response(message)
                    print(res)
                    time.sleep(args.sleeptime)
                    # ADU/s! 
                    # i = float( c_dict[beam_id].config["fps"]) * np.mean(  c_dict[beam_id].get_some_frames( 
                    #                     number_of_frames = 100, 
                    #                     apply_manual_reduction = True ),
                    #                     axis = 0)

                    i = float( c_dict[beam_id].config["fps"]) * np.mean(  c_dict[beam_id].get_data( 
                                        apply_manual_reduction = True ),
                                        axis = 0)

                    # i = float( c_dict[beam_id].config["fps"]) * c_dict[beam_id].get_image( 
                    #                     apply_manual_reduction = True )
                    #signaltmp.append(  np.mean( i[ np.array(exterior_filter).astype(bool) ] ) / np.sum( i ) ) # [+x, -x, +y, -y] 
                    signaltmp.append(  i[ np.array(sec_filter[beam_id]).astype(bool) ][4] / np.sum( i ) ) # [+x, -x, +y, -y] 

                
                dxtmp.append( args.phasemask_gain * ( signaltmp[0] - signaltmp[1] ) )  # seems to work , may change with different cropping sizes, 
                dytmp.append( args.phasemask_gain * ( signaltmp[2] - signaltmp[3] ) )
                signaltmp2.append( signaltmp )
            
            signal = np.median( signaltmp, axis=0) # 4 signals (up,down,left,right)
            dx = np.median( dxtmp )
            dy = np.median( dytmp )

            # err_x += dx 
            # err_y += dy 

            if dx**2 + dy**2 > 0 :
                rel_change = ( np.sqrt( dx**2 + dy**2 ) - np.sqrt( dx_prev**2 + dy_prev**2 ) ) / np.sqrt( dx**2 + dy**2 ) 
            else:
                rel_change = 0 

            if (abs(dx) > 70) or (abs(dy) > 70):
                close = False
                print(f'error signals dx, dy = {round(dx)}, {round(dy)}um seem too high.. check gains and limits.')
                convergence = False 

            if (dx**2 + dy**2)**0.5 < 1  and not near_flag_passed: #(np.sqrt( dx**2 + dy**2 ) < 0.35) and not near_flag_passed:
                # 2 was roughly the measured std in the signal * default gain 
                near_flag_passed = True
                dz *= 1.3
                args.phasemask_gain *= 0.5 # reduce the gain
                print('getting close, going to increase dither amplitude')

            
            # if rel_change < 0.03 : #np.sqrt( dx**2 + dy**2 ) < 0.15:
            #     close = False
            #     message = f"moveabs BMX{beam_id} { prev_pos[0] }"
            #     res = send_and_get_response(message)
            #     message = f"moveabs BMY{beam_id} { prev_pos[1] }"
            #     res = send_and_get_response(message)
            #     print("relative movements seem to have converged. Stopping here")
            #     convergence = True

            if (dx**2 + dy**2)**0.5 < 1.5 : #np.sqrt( dx**2 + dy**2 ) < 0.15:
                close = False
                message = f"moveabs BMX{beam_id} { prev_pos[0] }"
                res = send_and_get_response(message)
                message = f"moveabs BMY{beam_id} { prev_pos[1] }"
                res = send_and_get_response(message)
                print("relative movements seem to have converged. Stopping here")
                convergence = True

            if cnt > max_iterations:
                close = False
                print(f"passed maximum of {max_iterations} iterations. Stopping here")
                convergence = False 


            dx_prev = dx
            dy_prev = dy

            prev_pos = prev_pos +  np.array( [dx, dy] ) #np.array( [err_x, err_y] ) # #[dx, dy] )

            print( f"moved {dx}, {dy}")

            telem["i"].append( i )
            telem["signal"].append( signal )
            telem["dx"].append( dx )
            telem["dy"].append( dy )

        print( f"\n==============\n  CONVERGENCE: {convergence}" )

else:
    raise UserWarning("invalid method input. Try --method brute_scan for example")


#### SAVING JSON DATA
# with open("/home/asg/Videos/finealign_phasemask_telemetry.json", "w") as f:
#     json.dump(telem, f, indent=4, default=lambda o: o.tolist() if isinstance(o, np.ndarray) else o)


print("returning back to prior camera settings")
c_dict[beam_id].send_fli_cmd(f"set fps {fps0}")
time.sleep(1)
c_dict[beam_id].send_fli_cmd(f"set gain {gain0}")
time.sleep(1)

print("closing camera and DM SHM objects")

for beam_id in args.beam_id:
    c_dict[beam_id].close(erase_file=False)
    dm_shm_dict[beam_id].close(erase_file=False)


# print('plotting')

# try: 
#     from matplotlib.offsetbox import OffsetImage, AnnotationBbox
#     # Compute cumulative positions
#     cumulative_dx = np.cumsum(telem["dx"])
#     cumulative_dy = np.cumsum(telem["dy"])

#     # Create scatter plot of cumulative dx vs cumulative dy
#     plt.figure(figsize=(6, 6))
#     plt.scatter(cumulative_dx, cumulative_dy, c='blue', marker='o')
#     plt.xlabel("Relative phase mask X position [microns]")
#     plt.ylabel("Relative phase mask Y position [microns]")
#     #plt.title("Scatter Plot of Cumulative dx vs dy")
#     plt.grid(True)
#     plt.savefig('delme.png') #,bbox_='tight')




#     fig, axes = plt.subplots(nrows=2, ncols=1, figsize=(12, 10), sharex=True)
#     cost_fn = telem["signal"] #1/np.mean( telem["signal"],axis=1)
#     axes[0].plot(cost_fn , marker='o', color='k')
#     axes[0].set_ylabel(r"Cost Function [unitless]",fontsize=15)
#     axes[0].grid(True)
#     axes[0].tick_params(labelsize=15)

#     axes[1].plot( cumulative_dx, ls=':', color='k',label=r"X")
#     axes[1].plot(cumulative_dy, ls='--', color='k',label=r"Y")
#     axes[1].set_ylabel("Phase mask motor\n"+r"relative offset [$\mu$m]",fontsize=15)
#     axes[1].tick_params(labelsize=15)
#     #axes[1].set_title("Cumulative X Position")
#     axes[1].legend(fontsize=15)
#     axes[1].grid(True)

#     zoom_val = 1.8 

#     # Image at the beginning: telem["i"][0]
#     img0 = telem["i"][0]
#     ab0 = AnnotationBbox(OffsetImage(img0, zoom=zoom_val),
#                         (1, cost_fn[1]),
#                         frameon=False)
#     axes[0].add_artist(ab0)

#     # Image at the end: telem["i"][-1]
#     img_last = telem["i"][-1]
#     ab_last = AnnotationBbox(OffsetImage(img_last, zoom=zoom_val),
#                             (len(cost_fn)-2, cost_fn[1]),
#                             frameon=False)
#     axes[0].add_artist(ab_last)

#     plt.xlabel("Iteration",fontsize=15)


#     if args.fig_path is None:
#         savepath="delme{beam_id}.png"
#     else: # we save with default name at fig path 
#         savepath=args.fig_path + f'phasemask_auto_center_beam{beam_id}'
#     print(f"saving figure {savepath}")

#     plt.savefig(savepath, bbox_inches='tight')


#     # ### NOISE ANALYSIS 
#     # Sx=[]
#     # Sy=[]
#     # for ss in telem["signal"]:
#     #     Sx.append( ( ss[0] - ss[1] )  ) # ADU /s
#     #     Sy.append( ( ss[2] - ss[3] ) ) # ADU /s


#     #     dx = args.phasemask_gain * ( ss[0] - ss[1] )  # seems to work , may change with different cropping sizes, 
#     #     dy = args.phasemask_gain * ( ss[2] - ss[3] )

#     # # Convert to numpy arrays (optional, but useful for computing stats)
#     # Sx = np.array(Sx)
#     # Sy = np.array(Sy)

#     # # Compute mean and standard deviation for each distribution
#     # mean_Sx = np.mean(Sx)
#     # std_Sx = np.std(Sx)
#     # mean_Sy = np.mean(Sy)
#     # std_Sy = np.std(Sy)
#     # print( f"sx, mean = {mean_Sx}, std={std_Sx}")
#     # print( f"sy, mean = {mean_Sy}, std={std_Sy}")
#     # # sx, mean = 0.000560768241116661, std=0.0012595275496219676
#     # # sy, mean = 0.0003143405690655363, std=0.00425961632425167

#     # # real system was off +25 in x, +15 in y , so good gain per um
#     # # gain_x = 25 / mean_Sx = 50000.0 (um / iter)

#     # # issues is the noise on this is very large! 
#     # # Create a figure with two rows (subplots)
#     # fig, axes = plt.subplots(nrows=1, ncols=1, figsize=(10, 5))

#     # # Histogram for Sx
#     # axes.hist(Sx, bins=np.linspace(-5*std_Sx, 5*std_Sx,20), alpha=0.5, label="X")
#     # axes.hist(Sy, bins=np.linspace(-5*std_Sy, 5*std_Sy,20), alpha=0.5, label="Y")
#     # axes.axvline(mean_Sx, color='red', linestyle='--', linewidth=2,
#     #             label=f'X Mean={mean_Sx:.2e} ADU/s, X Std={std_Sx:.2e} ADU/s')
#     # axes.axvline(mean_Sy, color='red', linestyle='--', linewidth=2,
#     #             label=f'Y Mean={mean_Sy:.2e} ADU/s, Y Std={std_Sy:.2e} ADU/s')
#     # axes.legend()
#     # #axe].set_title("Histogram of Sx (ADU/s)")
#     # axes.set_xlabel("Strehl pixel signal (ADU/s)",fontsize=15)
#     # axes.set_ylabel("Frequency",fontsize=15)
#     # axes.tick_params(labelsize=15)
#     # plt.savefig("delme.jpeg",bbox_inches='tight')



# except Exception as e:
#     print(f"failed to make plots : {e}")


















## END 
### ### TESTING ALGORITHMS PRIOR TO FINALIZATION 




# def interpolate_onto_set_grid(original_array, original_size):
#     # Define the new 24x24 grid
#     ## HARDCODED 
#     new_size = 24

#     x = np.linspace(-1, 1, original_size)
#     y = np.linspace(-1, 1, original_size)

#     interp_func = RegularGridInterpolator((x, y), original_array)


#     x_new = np.linspace(-1, 1, new_size)
#     y_new = np.linspace(-1, 1, new_size)
#     X_new, Y_new = np.meshgrid(x_new, y_new, indexing='ij')
#     new_points = np.array([X_new.flatten(), Y_new.flatten()]).T

#     # Interpolate the original array onto the new grid
#     interpolated_array = interp_func(new_points).reshape(new_size, new_size)

#     return interpolated_array

# def send_and_get_response(message):
#     # st.write(f"Sending message to server: {message}")
#     state_dict["message_history"].append(
#         f":blue[Sending message to server: ] {message}\n"
#     )
#     state_dict["socket"].send_string(message)
#     response = state_dict["socket"].recv_string()
#     if "NACK" in response or "not connected" in response:
#         colour = "red"
#     else:
#         colour = "green"
#     # st.markdown(f":{colour}[Received response from server: ] {response}")
#     state_dict["message_history"].append(
#         f":{colour}[Received response from server: ] {response}\n"
#     )
#     return response.strip()




# def get_cropped_inner_pupil( mask ): 
#     r1, r2, c1, c2 = np.array( util.crop_to_square( mask) ) + 1 # annoying offset of 1 to center 
#     r1 += 1
#     r2 -= 1
#     c1 += 1
#     c2 -= 1 # filter edges
#     return r1,r2,c1,c2 

# # Get reference pupils (later this can just be a SHM address)
# #Clear Pupil
# N0_norm_crop = {}
# fine_crop_coords = {}
# clear_pupils = {}
# rel_offset = 200.0 #um phasemask offset for clear pupil
# print( 'Moving FPM out to get clear pupils')
# for beam_id in args.beam_id:
#     message = f"moverel BMX{beam_id} {rel_offset}"
#     res = send_and_get_response(message)
#     print(res) 
#     time.sleep(2)

#     print( 'gettin clear pupils')

#     N0s = c_dict[beam_id].get_some_frames(number_of_frames = 1000, apply_manual_reduction=True) 

#     clear_pupils[beam_id] = N0s

#     # move back (so we have time buffer while calculating b)
#     print( 'Moving FPM back in beam.')
#     message = f"moverel BMX{beam_id} {-rel_offset}"
#     res = send_and_get_response(message)
#     print(res) 
#     time.sleep(2)


#     # Now procees/fit the pupil  (ADU/S)!!
#     N0 = float(c_dict[beam_id].config['fps']) *  np.mean( clear_pupils[beam_id] , axis=0) 

#     center_x, center_y, a, b, theta, pupil_mask = util.detect_pupil(N0, sigma=2, threshold=0.5, plot=False, savepath=None)

#     r1, r2, c1, c2 = get_cropped_inner_pupil( pupil_mask )

#     N0_mean = np.mean( N0[pupil_mask] )

#     N0_norm = N0.copy()
#     N0_norm[~pupil_mask] = N0_mean

#     #N0_norm_crop[beam_id] = N0_norm[r1:r2, c1:c2].copy()

#     assert r2-r1 == c2 - c1

#     fine_crop_coords[beam_id] = [r1,r2,c1,c2]

#     N0_norm_crop[beam_id] = interpolate_onto_set_grid(original_array= N0_norm[r1:r2, c1:c2].copy(),
#                                                        original_size= r2-r1)


# #img = np.mean( c_dict[beam_id].get_some_frames(number_of_frames=100, apply_manual_reduction=True) , axis=0)
# title_list = ["N0_interp","N0"]
# im_list = [N0_norm_crop[beam_id], np.mean( clear_pupils[beam_id] , axis=0)]
# util.nice_heatmap_subplots( im_list, savefig='delme.png')



# input("center well to register initial position for building phasemask IM. Press enter when ready")

# initial_pos = {}
# for beam_id in args.beam_id:
#     message = f"read BMX{beam_id}"
#     initial_Xpos = float(send_and_get_response(message))

#     message = f"read BMY{beam_id}"
#     initial_Ypos = float(send_and_get_response(message))
    
#     # definition is [X,Y]
#     initial_pos[beam_id] = [initial_Xpos, initial_Ypos]



# I0_ref = float(c_dict[beam_id].config['fps']) * np.mean( 
#             c_dict[beam_id].get_some_frames( 
#                 number_of_frames = 100, 
#                 apply_manual_reduction = True ),
#                 axis = 0) # ADU/s !   


# #np.sum( I0_ref[exterior_filter[beam_id]])

# pupil_edge_filter = util.filter_exterior_annulus(pupil_mask, inner_radius=7, outer_radius=100) # to limit pupil edge pixels
# pupil_limit_filter = ~util.filter_exterior_annulus(pupil_mask, inner_radius=10, outer_radius=100) # to limit far out pixel
# exterior_filter = ( abs( I0_ref  - N0 ) > 0.12 * np.mean( N0[pupil_mask] ) ) * pupil_edge_filter * pupil_limit_filter

# util.nice_heatmap_subplots( [exterior_filter], cbar_label_list = ["ON/OFF"] , savefig='delme.png')

# ### VERIFICATION 
# exterior_sig = []
# for dx in np.linspace( -50,50,20):
#     print( dx )
#     message = f"moveabs BMX{beam_id} {initial_pos[beam_id][0] + dx}"
#     res = send_and_get_response(message)
#     time.sleep(1)
#     i  = float(c_dict[beam_id].config['fps']) * np.mean( 
#                     c_dict[beam_id].get_some_frames( 
#                         number_of_frames = 20, 
#                         apply_manual_reduction = True ),
#                         axis = 0) # ADU/s ! 
    
#     exterior_sig.append( np.mean( i[exterior_filter]) / np.sum( i ) )

# plt.figure(figsize=(8,5)); 
# plt.plot( np.linspace( -50,50,20) - 10,  np.array( exterior_sig )/ np.sum( exterior_filter )  ,color='k')
# plt.ylabel(r'filtered exterior pixels mean [ADU/s]',fontsize=15)
# plt.xlabel(r'phase mask offset [$\mu$m]',fontsize=15)
# plt.gca().tick_params(labelsize=15)
# plt.savefig('delme.png', bbox_inches = 'tight') 




# # #img = np.mean( c_dict[beam_id].get_some_frames(number_of_frames=100, apply_manual_reduction=True) , axis=0)
# # title_list = ["N0_interp","N0"]
# # im_list = [N0_norm_crop[beam_id], np.mean( clear_pupils[beam_id] , axis=0), exterior_filter]
# # util.nice_heatmap_subplots( im_list, savefig='delme.png')


# # exterior_filter = util.filter_exterior_annulus(pupil_mask, inner_radius=7, outer_radius=10)
# # im_list = [N0_norm_crop[beam_id], np.mean( clear_pupils[beam_id] , axis=0),exterior_filter, abs(I0_ref - N0 ), abs(I0_ref - N0 ) > 0.2 * np.mean( N0) ]
# # util.nice_heatmap_subplots( im_list, savefig='delme.png')


# imgs_to_mean = 100 # for each poke we average this number of frames
# IM_dict = {}
# I2M_dict = {}
# method = "push-pull"
# for beam_id in args.beam_id:
#     IM = []
#     Iplus_all = []
#     Iminus_all = []
#     r1, r2, c1, c2 = fine_crop_coords[beam_id]


#     if method == "zero_ref" :

#         r1,r2,c1,c2 =  fine_crop_coords[beam_id] 

#         I0 = float(c_dict[beam_id].config['fps']) * np.mean( 
#                     c_dict[beam_id].get_some_frames( 
#                         number_of_frames = imgs_to_mean, 
#                         apply_manual_reduction = True ),
#                         axis = 0) # ADU/s !   

#         I0_crop = I0[r1:r2,c1:c2]
#         I0_interp = interpolate_onto_set_grid(original_array= I0_crop,
#                                                         original_size= r2-r1 ) / N0_norm_crop[beam_id]
        

        
#         for i,m in enumerate(["BMX","BMY"]):
#             sign = 1
#             message = f"moveabs {m}{beam_id} {initial_pos[beam_id][i] + sign * args.poke_amp }"
#             res = send_and_get_response(message)

#             I_plus = float(c_dict[beam_id].config['fps']) * np.mean( 
#                     c_dict[beam_id].get_some_frames( 
#                         number_of_frames = imgs_to_mean, 
#                         apply_manual_reduction = True ),
#                         axis = 0) # ADU/s ! 
            

#             I_plus_interp = interpolate_onto_set_grid(original_array= I_plus[r1:r2,c1:c2],
#                                                         original_size= r2-r1 ) / N0_norm_crop[beam_id]

#             errsig = (I_plus_interp - I0_interp).reshape(-1) / args.poke_amp

#             IM.append( list(  errsig.reshape(-1) ) ) 

#     elif method == "push-pull":
#         for i,m in enumerate(["BMX","BMY"]):
#             print(f'executing for motor{m}')
#             I_plus_list = []
#             I_minus_list = []
#             for sign in [(-1)**n for n in range(10)]: #[-1,1]:

#                 message = f"moveabs {m}{beam_id} {initial_pos[beam_id][i] + sign * args.poke_amp/2 }"
#                 res = send_and_get_response(message)
                
#                 time.sleep( 1 )

#                 img = float(c_dict[beam_id].config['fps']) * np.mean( 
#                     c_dict[beam_id].get_some_frames( 
#                         number_of_frames = imgs_to_mean, 
#                         apply_manual_reduction = True ),
#                         axis = 0) # ADU/s ! 
                
#                 img_norm = img[r1:r2, c1:c2] #/ N0_norm[r1:r2, c1:c2] # ADU/s

#                 if sign > 0:
#                     I_plus_list.append( list(  img_norm ) )
#                 if sign < 0:
#                     I_minus_list.append( list( img_norm ) )

#             r1,r2,c1,c2 =  fine_crop_coords[beam_id] 
#             I_plus = np.mean( I_plus_list, axis = 0) # ADU/s

#             I_plus_interp = interpolate_onto_set_grid(original_array= I_plus,
#                                                         original_size= r2-r1 ) / N0_norm_crop[beam_id]

#             I_minus = np.mean( I_minus_list, axis = 0) # ADU/s
#             I_minus_interp = interpolate_onto_set_grid(original_array= I_minus,
#                                                         original_size= r2-r1 ) / N0_norm_crop[beam_id]
            
#             # Try minimize dependancies, if I2A not calibrated or DM mask then the above fails.. keep simple. We can deal with this in post processing
#             errsig = (I_plus_interp - I_minus_interp).reshape(-1)  / args.poke_amp

#             # reenter pokeamp norm
#             #Iplus_all.append( I_plus_list ) # ADU/S
#             #Iminus_all.append( I_minus_list ) # ADU/s

#             IM.append( list(  errsig.reshape(-1) ) ) 

#     # final dick
#     IM_dict[beam_id] = IM 

#     I2M_dict[beam_id] = np.linalg.pinv( IM )


# #### LETS LOOK AT THE MODES 
# util.nice_heatmap_subplots( [np.array(IM[0]).reshape(24,24),np.array(IM[1]).reshape(24,24)],savefig='delme.png')


# ### LETS TEST IT 

# #center_x, center_y, a, b, theta, pupil_mask = util.detect_pupil(N0, sigma=2, threshold=0.5, plot=False, savepath=None)

# ### TRY GRADIENT DESENT 

# initial_pos = {}
# for beam_id in args.beam_id:
#     message = f"read BMX{beam_id}"
#     initial_Xpos = float(send_and_get_response(message))

#     message = f"read BMY{beam_id}"
#     initial_Ypos = float(send_and_get_response(message))
    
#     # definition is [X,Y]
#     initial_pos[beam_id] = [initial_Xpos, initial_Ypos]




# close = True 
# prev_pos = initial_pos[beam_id]
# dz = 8

# err_x = 0 
# err_y = 0
# near_flag_passed = False 
# signal = []

# telem = {"i":[], "signal":[], "dx":[],"dy":[]}
# while close: 

#     signal = []

#     for j, m in enumerate( ["BMX","BMX","BMY","BMY"] ):

        
#         sign = (-1)**j
         
#         z0 = prev_pos[int(j > 1)] # if j < 2 than we look at x (index = 0), else y (index = 1)

        
#         message = f"moveabs {m}{beam_id} { z0 + sign * dz }"
#         print(message)
#         res = send_and_get_response(message)
#         print(res)
#         time.sleep(0.1)
#         i = np.mean(  c_dict[beam_id].get_some_frames( 
#                             number_of_frames = 100, 
#                             apply_manual_reduction = True ),
#                             axis = 0)

#         signal.append(  np.mean( i[exterior_filter ] ) / np.sum( i ) ) # [+x, -x, +y, -y] 


#     dx = 15000*( signal[0] - signal[1] ) 
#     dy = 15000*( signal[2] - signal[3] )

#     if (np.sqrt( dx**2 + dy**2 ) < 0.4) and not near_flag_passed:
#         near_flag_passed = True
#         dz *= 1.5
#         print('getting close, going to increase dither amplitude')

    
#     if np.sqrt( dx**2 + dy**2 ) < 0.15:
#         close = False
#         message = f"moveabs BMX{beam_id} { prev_pos[0] }"
#         res = send_and_get_response(message)
#         message = f"moveabs BMY{beam_id} { prev_pos[1] }"
#         res = send_and_get_response(message)
#         print("relative movements seem to have converged. Stopping here")
#     #err_x += dx 
#     #err_y += dy

#     prev_pos = prev_pos + np.array( [dx, dy] ) # [err_x, err_y] ) #[dx, dy] )

#     print( f"moved {dx}, {dy}")

#     telem["i"].append( i )
#     telem["signal"].append( signal )
#     telem["dx"].append( dx )
#     telem["dy"].append( dy )

# telem()

# with open("/home/asg/Videos/finealign_phasemask_telemetry.json", "w") as f:
#     json.dump(telem, f, indent=4, default=lambda o: o.tolist() if isinstance(o, np.ndarray) else o)

# # Compute cumulative positions
# cumulative_dx = np.cumsum(telem["dx"])
# cumulative_dy = np.cumsum(telem["dy"])

# # Create scatter plot of cumulative dx vs cumulative dy
# plt.figure(figsize=(6, 6))
# plt.scatter(cumulative_dx, cumulative_dy, c='blue', marker='o')
# plt.xlabel("Relative phase mask X position [microns]")
# plt.ylabel("Relative phase mask Y position [microns]")
# #plt.title("Scatter Plot of Cumulative dx vs dy")
# plt.grid(True)
# plt.savefig('delme.png') #,bbox_='tight')


# # Create 3 subplots in a single column with shared x-axis
# from matplotlib.offsetbox import OffsetImage, AnnotationBbox

# fig, axes = plt.subplots(nrows=2, ncols=1, figsize=(12, 10), sharex=True)


# cost_fn = 1/np.mean( telem["signal"],axis=1)

# axes[0].plot(cost_fn , marker='o', color='k')
# axes[0].set_ylabel(r"Cost Function [unitless]",fontsize=15)
# #axes[0].set_title("Signal Time Series")
# #axes[0].grid(True)
# axes[0].tick_params(labelsize=15)

# # Middle subplot: Cumulative X position vs. Iterations
# axes[1].plot( cumulative_dx, ls=':', color='k',label=r"X")
# axes[1].plot(cumulative_dy, ls='--', color='k',label=r"Y")
# axes[1].set_ylabel("Phase mask motor\n"+r"relative offset [$\mu$m]",fontsize=15)
# axes[1].tick_params(labelsize=15)
# #axes[1].set_title("Cumulative X Position")
# axes[1].legend(fontsize=15)
# #axes[1].grid(True)

# # Add images near iteration 0 and at the last iteration in axes[1]
# zoom_val = 1.8 

# # Image at the beginning: telem["i"][0]
# img0 = telem["i"][0]
# ab0 = AnnotationBbox(OffsetImage(img0, zoom=zoom_val),
#                      (1, cost_fn[1]),
#                      frameon=False)
# axes[0].add_artist(ab0)

# # Image at the end: telem["i"][-1]
# img_last = I0_ref #telem["i"][-1]
# ab_last = AnnotationBbox(OffsetImage(img_last, zoom=zoom_val),
#                          (len(cost_fn)-2, cost_fn[1]),
#                          frameon=False)
# axes[0].add_artist(ab_last)

# plt.xlabel("Iteration",fontsize=15)
# plt.savefig('delme.jpeg', bbox_inches='tight')


# ####
# ### IM control matrix tests
# r1, r2, c1, c2 = get_cropped_inner_pupil( pupil_mask )
# fps_gain = 0.03
# err_x = 0
# err_y = 0


#     # ADU/s
#     i = float(c_dict[beam_id].config['fps']) * np.mean( 
#                     c_dict[beam_id].get_some_frames( 
#                         number_of_frames = imgs_to_mean, 
#                         apply_manual_reduction = True ),
#                         axis = 0)

    
#     signal  = np.sum( i[exterior_filter ] ) 

#     i_n = interpolate_onto_set_grid(original_array= i[r1:r2, c1:c2].copy(),
#                                                     original_size= r2-r1)
#     s = i_n / N0_norm_crop[beam_id]

#     dx, dy = I2M_dict[beam_id].T @ s.reshape(-1)
#     print(dx,dy)

#     err_x += dx
#     err_y += dy

#     if (dx**2 + dy**2)**0.5 < 3:
#         print('done')
#         close = False

#     for m, dw in zip(["BMX", "BMY"], [err_x, err_y]):
#         message = f"moverel {m}{beam_id} { -fps_gain * dw}"
#         res = send_and_get_response(message)
#         print(res)
#         time.sleep(0.2)





# # # to apply it 
# def apply( i , pupil_mask, N0 , I2M, set_grid_size): 

#     # Input intensity i must be adu/s on reduced image. 

#     #center_x, center_y, a, b, theta, pupil_mask = util.detect_pupil(N0, sigma=2, threshold=0.5, plot=False, savepath=None)
#     r1, r2, c1, c2 = get_cropped_inner_pupil( pupil_mask )

#     i_norm = i[r1:r2, c1:c2].copy() / N0 
    
#     i_n = interpolate_onto_set_grid(original_array= i[r1:r2, c1:c2].copy(),
#                                                     original_size= r2-r1, 
#                                                     new_size = set_grid_size )

#     dx, dy = I2M @ i_n.reshape(-1)

#     return dx, dy


# # ############## PREVIOUS METHOD LOOKING AT PUPIL SYMMETRY - WASNT CRASH HOT 
# # try:
# #     from asgard_alignment import controllino as co
# #     myco = co.Controllino('172.16.8.200')
# #     controllino_available = True
# #     print('controllino connected')
    
# # except:
# #     print('WARNING Controllino cannot connect. WILL NOT MOVE SOURCE OUT FOR DARK')
# #     controllino_available = False 
# # """
# # idea it to be able to align phasemask position 
# # in a mode independent way with significant focus offsets
# # using image symmetry across registered pupil as objective 

# # TO DO : tweak zero point with clear ppupil quadrants . 
# # # check error sign 
# # """


# # def send_and_get_response(message):
# #     # st.write(f"Sending message to server: {message}")
# #     state_dict["message_history"].append(
# #         f":blue[Sending message to server: ] {message}\n"
# #     )
# #     state_dict["socket"].send_string(message)
# #     response = state_dict["socket"].recv_string()
# #     if "NACK" in response or "not connected" in response:
# #         colour = "red"
# #     else:
# #         colour = "green"
# #     # st.markdown(f":{colour}[Received response from server: ] {response}")
# #     state_dict["message_history"].append(
# #         f":{colour}[Received response from server: ] {response}\n"
# #     )

# #     return response.strip()




# # def split_into_quadrants(image, pupil_mask):
# #     """
# #     Split the image into four quadrants using the active pupil mask.

# #     Parameters:
# #         image (ndarray): Input image.
# #         pupil_mask (ndarray): Boolean array representing the active pupil.

# #     Returns:
# #         dict: Dictionary of quadrants (top-left, top-right, bottom-left, bottom-right).
# #     """
# #     y, x = np.indices(image.shape)
# #     cx, cy = np.mean(np.where(pupil_mask), axis=1).astype(int)

# #     # Create boolean masks for each quadrant
# #     top_left_mask = (y < cy) & (x < cx) & pupil_mask
# #     top_right_mask = (y < cy) & (x >= cx) & pupil_mask
# #     bottom_left_mask = (y >= cy) & (x < cx) & pupil_mask
# #     bottom_right_mask = (y >= cy) & (x >= cx) & pupil_mask

# #     quadrants = {
# #         "top_left": image[top_left_mask],
# #         "top_right": image[top_right_mask],
# #         "bottom_left": image[bottom_left_mask],
# #         "bottom_right": image[bottom_right_mask],
# #     }

# #     return quadrants

# # def weighted_photometric_difference(quadrants):
# #     """
# #     Calculate the weighted photometric difference between quadrants.

# #     Parameters:
# #         quadrants (dict): Dictionary of quadrants.

# #     Returns:
# #         tuple: (x_error, y_error) error vectors.
# #     """
# #     top = np.mean(quadrants["top_left"]) + np.sum(quadrants["top_right"])
# #     bottom = np.mean(quadrants["bottom_left"]) + np.sum(quadrants["bottom_right"])

# #     left = np.mean(quadrants["top_left"]) + np.sum(quadrants["bottom_left"])
# #     right = np.mean(quadrants["top_right"]) + np.sum(quadrants["bottom_right"])

# #     y_error = top - bottom
# #     x_error = left - right

# #     return x_error, y_error


# # def get_bad_pixel_indicies( imgs, std_threshold = 20, mean_threshold=6):
# #     # To get bad pixels we just take a bunch of images and look at pixel variance and mean

# #     ## Identify bad pixels
# #     mean_frame = np.mean(imgs, axis=0)
# #     std_frame = np.std(imgs, axis=0)

# #     global_mean = np.mean(mean_frame)
# #     global_std = np.std(mean_frame)
# #     bad_pixel_map = (np.abs(mean_frame - global_mean) > mean_threshold * global_std) | (std_frame > std_threshold * np.median(std_frame))

# #     return bad_pixel_map


# # def interpolate_bad_pixels(img, bad_pixel_map):
# #     filtered_image = img.copy()
# #     filtered_image[bad_pixel_map] = median_filter(img, size=3)[bad_pixel_map]
# #     return filtered_image



# # # def plot_telemetry(telemetry, savepath=None):
# # #     """
# # #     Plots the phasemask centering telemetry for each beam.
    
# # #     Parameters:
# # #         telemetry (dict): A dictionary where keys are beam IDs and values are dictionaries
# # #                           with keys:
# # #                               "phasmask_Xpos" - list of X positions,
# # #                               "phasmask_Ypos" - list of Y positions,
# # #                               "phasmask_Xerr" - list of X errors,
# # #                               "phasmask_Yerr" - list of Y errors.
# # #     """
# # #     for beam_id, data in telemetry.items():
# # #         # Determine the number of iterations
# # #         num_iterations = len(data["phasmask_Xpos"])
# # #         iterations = np.arange(1, num_iterations + 1)
        
# # #         # Create a figure with two subplots: one for positions and one for errors
# # #         fig, axs = plt.subplots(1, 2, figsize=(12, 5))
# # #         fig.suptitle(f"Telemetry for Beam {beam_id}", fontsize=14)
        
# # #         # Plot phasemask positions
# # #         axs[0].plot(iterations, data["phasmask_Xpos"], marker='o', label="X Position")
# # #         axs[0].plot(iterations, data["phasmask_Ypos"], marker='s', label="Y Position")
# # #         axs[0].set_xlabel("Iteration")
# # #         axs[0].set_ylabel("Position (um)")
# # #         axs[0].set_title("Phasemask Positions")
# # #         axs[0].legend()
# # #         axs[0].grid(True)
        
# # #         # Plot phasemask errors
# # #         axs[1].plot(iterations, data["phasmask_Xerr"], marker='o', label="X Error")
# # #         axs[1].plot(iterations, data["phasmask_Yerr"], marker='s', label="Y Error")
# # #         axs[1].set_xlabel("Iteration")
# # #         axs[1].set_ylabel("Error (um)")
# # #         axs[1].set_title("Phasemask Errors")
# # #         axs[1].legend()
# # #         axs[1].grid(True)
        
# # #         plt.tight_layout(rect=[0, 0, 1, 0.95])
# # #         if savepath is not None:
# # #             plt.savefig(savepath)
# # #         plt.show()


# # def plot_telemetry(telemetry,savepath='delme.png'):
# #     """
# #     For each beam, produce scatter plots of:
# #       - Phasemask positions: X vs. Y
# #       - Phasemask errors: X error vs. Y error

# #     Parameters:
# #         telemetry (dict): Dictionary with beam IDs as keys. Each beam's value is a
# #                           dictionary with keys:
# #                               "phasmask_Xpos" : list of X positions,
# #                               "phasmask_Ypos" : list of Y positions,
# #                               "phasmask_Xerr" : list of X errors,
# #                               "phasmask_Yerr" : list of Y errors.
# #     """
# #     for beam_id, data in telemetry.items():
# #         # Create a figure with two subplots side-by-side.
# #         fig, axs = plt.subplots(1, 2, figsize=(12, 5))
# #         fig.suptitle(f"Telemetry Scatter Plots for Beam {beam_id}", fontsize=14)
        
# #         # Scatter plot for phasemask positions.
# #         axs[0].scatter(data["phasmask_Xpos"], data["phasmask_Ypos"], 
# #                        color='blue', marker='o', s=50)
# #         axs[0].set_xlabel("Phasemask X Position")
# #         axs[0].set_ylabel("Phasemask Y Position")
# #         axs[0].set_title("Positions")
# #         axs[0].grid(True)
        
# #         # Scatter plot for phasemask errors.
# #         axs[1].scatter(data["phasmask_Xerr"], data["phasmask_Yerr"],
# #                        color='red', marker='s', s=50)
# #         axs[1].set_xlabel("Phasemask X Error")
# #         axs[1].set_ylabel("Phasemask Y Error")
# #         axs[1].set_title("Errors")
# #         axs[1].grid(True)
        
# #         plt.tight_layout(rect=[0, 0, 1, 0.92])
# #         if savepath is not None:
# #             plt.savefig(savepath)
# #         plt.show()


# # def image_slideshow(telemetry, beam_id):

# #     # Interactive plot with slider
# #     positions = [(x,y) for x,y in zip(telemetry[beam_id]["phasmask_Xpos"],telemetry[beam_id]["phasmask_Ypos"])]
# #     images = telemetry[beam_id]["img"]
# #     fig, ax = plt.subplots(1, 2, figsize=(12, 6))
# #     plt.subplots_adjust(bottom=0.2)

# #     # Initialize plots
# #     image_plot = ax[0].imshow(images[0], cmap='hot')
# #     ax[0].set_title("Image")
# #     position_plot, = ax[1].plot(positions[:, 0], positions[:, 1], marker='o', linestyle='-', color='blue', alpha=0.5)
# #     current_position, = ax[1].plot(positions[0, 0], positions[0, 1], marker='o', color='red')
# #     ax[1].set_xlim(positions[:, 0].min() - 5, positions[:, 0].max() + 5)
# #     ax[1].set_ylim(positions[:, 1].min() - 5, positions[:, 1].max() + 5)
# #     ax[1].set_title("Phasemask Center History")
# #     ax[1].set_xlabel("x position")
# #     ax[1].set_ylabel("y position")
# #     ax[1].grid()

# #     # Slider setup
# #     ax_slider = plt.axes([0.2, 0.05, 0.65, 0.03])
# #     slider = Slider(ax_slider, "Iteration", 0, len(images) - 1, valinit=0, valstep=1)

# #     # Update function for slider
# #     def update(val):
# #         idx = int(slider.val)
# #         image_plot.set_data(images[idx])
# #         current_position.set_data([positions[idx, 0]], [positions[idx, 1]])
# #         fig.canvas.draw_idle()

# #     slider.on_changed(update)
# #     plt.show()


# # parser = argparse.ArgumentParser(description="Baldr phase mask fine x-y adjustment")

# # parser.add_argument("--host", type=str, default="localhost", help="Server host")
# # parser.add_argument("--port", type=int, default=5555, help="Server port")
# # parser.add_argument(
# #     "--timeout", type=int, default=5000, help="Response timeout in milliseconds"
# # )
# # # Camera shared memory path
# # parser.add_argument(
# #     "--global_camera_shm",
# #     type=str,
# #     default="/dev/shm/cred1.im.shm",
# #     help="Camera shared memory path. Default: /dev/shm/cred1.im.shm"
# # )

# # # TOML file path; default is relative to the current file's directory.
# # default_toml = os.path.join("config_files", "baldr_config.toml") #os.path.dirname(os.path.abspath(__file__)), "..", "config_files", "baldr_config.toml")
# # parser.add_argument(
# #     "--toml_file",
# #     type=str,
# #     default=default_toml,
# #     help="TOML file to write/edit. Default: ../config_files/baldr_config.toml (relative to script)"
# # )

# # # Beam ids: provided as a comma-separated string and converted to a list of ints.
# # parser.add_argument(
# #     "--beam_id",
# #     type=lambda s: [int(item) for item in s.split(",")],
# #     default=[2], #1, 2, 3, 4],
# #     help="Comma-separated list of beam IDs. Default: 1,2,3,4"
# # )

# # parser.add_argument(
# #     "--max_iterations",
# #     type=int,
# #     default=10,
# #     help="maximum number of iterations allowed in centering. Default = 10"
# # )

# # parser.add_argument(
# #     "--gain",
# #     type=int,
# #     default=0.1,
# #     help="gain to be applied for centering beam. Default = 0.1 "
# # )

# # parser.add_argument(
# #     "--tol",
# #     type=int,
# #     default=0.1,
# #     help="tolerence for convergence of centering algorithm. Default = 0.1 "
# # )

# # # Plot: default is True, with an option to disable.
# # parser.add_argument(
# #     "--plot", 
# #     dest="plot",
# #     action="store_true",
# #     default=True,
# #     help="Enable plotting (default: True)"
# # )


# # args = parser.parse_args()

# # # set up commands to move motors phasemask
# # context = zmq.Context()
# # context.socket(zmq.REQ)
# # socket = context.socket(zmq.REQ)
# # socket.setsockopt(zmq.RCVTIMEO, args.timeout)
# # server_address = f"tcp://{args.host}:{args.port}"
# # socket.connect(server_address)
# # state_dict = {"message_history": [], "socket": socket}

# # # phasemask specific commands
# # # message = f"fpm_movetomask phasemask{args.beam} {args.phasemask_name}"
# # # res = send_and_get_response(message)
# # # print(res)
# # phasemask_center = {}
# # for beam_id in args.beam_id:
# #     message = f"read BMX{beam_id}"
# #     Xpos = float( send_and_get_response(message) )

# #     message = f"read BMY{beam_id}"
# #     Ypos = float( send_and_get_response(message) )

# #     print(f'starting from current positiom X={Xpos}, Y={Ypos}um on beam {beam_id}')
# #     phasemask_center[beam_id] = [Xpos, Ypos]

# # # beam 2 initial pos
# # # In [56]: Xpos, Ypos
# # # Out[56]: (6054.994874999997, 3589.9963124999986)

# # # #example to move x-y of each beam's phasemask 
# # # for beam_id in args.beam_id:
# # #     message = f"moveabs BMX{beam_id} {Xpos}"
# # #     res = send_and_get_response(message)
# # #     print(res) 

# # #     message = f"moveabs BMY{beam_id} {Ypos}"
# # #     res = send_and_get_response(message)
# # #     print(res) 

# # # to manually adjust
# # # yi=20.0
# # # message = f"moverel BMY{beam_id} {yi}"
# # # res = send_and_get_response(message)
# # # print(res) 
# # # time.sleep(0.5)
# # # img = np.mean( c.get_data() ,axis=0) 
# # # for beam_id in args.beam_id:
# # #     r1,r2,c1,c2 = baldr_pupils[f"{beam_id}"]
# # #     cropped_img = interpolate_bad_pixels(img[r1:r2, c1:c2], bad_pixel_mask[r1:r2, c1:c2])
# # #     cropped_img *= 1/np.mean(cropped_img[pupil_masks[beam_id]])
# # #     clear_pupils[beam_id] = cropped_img
# # # plt.figure();plt.imshow(cropped_img);plt.savefig('delme1.png')

# # # set up commands to move DM 
# # assert hasattr(args.beam_id , "__len__")
# # assert len(args.beam_id) <= 4
# # assert max(args.beam_id) <= 4
# # assert min(args.beam_id) >= 1 

# # dm_shm_dict = {}
# # for beam_id in args.beam_id:
# #     dm_shm_dict[beam_id] = dmclass( beam_id=beam_id )
# #     # zero all channels
# #     dm_shm_dict[beam_id].zero_all()
# #     # activate flat 
# #     dm_shm_dict[beam_id].activate_flat()

# # # set up camera 
# # c = shm(args.global_camera_shm)

# # # set up subpupils and pixel mask
# # with open(args.toml_file ) as file:
# #     pupildata = toml.load(file)
# #     # Extract the "baldr_pupils" section
# #     baldr_pupils = pupildata.get("baldr_pupils", {})

# #     # the registered pupil mask for each beam (in the local frame)
# #     pupil_masks={}
# #     for beam_id in args.beam_id:
# #         pupil_masks[beam_id] = pupildata.get(f"beam{beam_id}", {}).get("pupil_mask", {}).get("mask", None) #pupildata.get(f"beam{beam_id}.pupil_mask.mask")
# #         if pupil_masks[beam_id] is None:
# #             raise UserWarning(f"pupil mask returned none in toml file. check for beam{beam_id}.pupil_mask.mask in the file:{args.toml_file}")


# # # dark and badpixel mask on global frame
# # if controllino_available:

# #     myco.turn_off("SBB")
# #     time.sleep(2)
    
# #     dark_raw = c.get_data()

# #     myco.turn_on("SBB")
# #     time.sleep(2)

# #     bad_pixel_mask = get_bad_pixel_indicies( dark_raw, std_threshold = 20, mean_threshold=6)
# # else:
# #     dark_raw = c.get_data()

# #     bad_pixel_mask = get_bad_pixel_indicies( dark_raw, std_threshold = 20, mean_threshold=6)


# # # get initial clear pupil
# # rel_offset = 500.0
# # clear_pupils = {}

# # message = f"moverel BMX{beam_id} {rel_offset}"
# # res = send_and_get_response(message)
# # print(res) 

# # message = f"moverel BMY{beam_id} {rel_offset}"
# # res = send_and_get_response(message)
# # print(res) 

# # time.sleep(1)

# # img = np.mean( c.get_data() ,axis=0) 
# # for beam_id in args.beam_id:
# #     r1,r2,c1,c2 = baldr_pupils[f"{beam_id}"]
# #     cropped_img = interpolate_bad_pixels(img[r1:r2, c1:c2], bad_pixel_mask[r1:r2, c1:c2])
# #     cropped_img *= 1/np.mean(cropped_img[pupil_masks[beam_id]])
# #     clear_pupils[beam_id] = cropped_img
# # plt.figure();plt.imshow(cropped_img);plt.savefig('delme1.png')

# # message = f"moverel BMX{beam_id} {-rel_offset}"
# # res = send_and_get_response(message)
# # print(res) 

# # message = f"moverel BMY{beam_id} {-rel_offset}"
# # res = send_and_get_response(message)
# # print(res) 

# # time.sleep(1)

# # # get initial image
# # img = np.mean( c.get_data() ,axis=0) #  full image 
# # initial_images = {}
# # for beam_id in args.beam_id:
# #     r1,r2,c1,c2 = baldr_pupils[f"{beam_id}"]
# #     cropped_img = interpolate_bad_pixels(img[r1:r2, c1:c2], bad_pixel_mask[r1:r2, c1:c2])
# #     initial_images[beam_id] = cropped_img
# # plt.figure();plt.imshow(initial_images[beam_id]);plt.savefig('delme1.png')


# # # begin centering algorithm, tracking telemetry
# # telemetry={b:{"img":[],"phasmask_Xpos":[],"phasmask_Ypos":[],"phasmask_Xerr":[], "phasmask_Yerr":[]} for b in args.beam_id }

# # complete_flag={b:False for b in args.beam_id}

# # for iteration in range(args.max_iterations):
# #     time.sleep(0.5)
# #     # get image 
# #     img = np.mean(c.get_data(), axis=0) # full image 

# #     for beam_id in args.beam_id:
# #         if not complete_flag[beam_id]:
# #             r1,r2,c1,c2 = baldr_pupils[f"{beam_id}"]
# #             cropped_img = interpolate_bad_pixels(img[r1:r2, c1:c2], bad_pixel_mask[r1:r2, c1:c2])
# #             # normalize by the mean within defined pupil mask
# #             cropped_img *= 1/np.mean(cropped_img[pupil_masks[beam_id]])

# #             # normed 
# #             normed_img = cropped_img / clear_pupils[beam_id] 
# #             # will need some clip and filtering 
        
# #             quadrants = split_into_quadrants(normed_img , pupil_masks[beam_id]) #cropped_img, pupil_masks[beam_id])
# #             #print( [len(v) for _,v in quadrants] )
# #             x_error, y_error = weighted_photometric_difference(quadrants)

# #             # Update phasemask center
# #             phasemask_center[beam_id][0] -= args.gain * x_error / np.sum(pupil_masks[beam_id])
# #             phasemask_center[beam_id][1] -= args.gain * y_error / np.sum(pupil_masks[beam_id])

# #             telemetry[beam_id]["img"].append( cropped_img )
# #             telemetry[beam_id]["phasmask_Xpos"].append( phasemask_center[beam_id][0] )
# #             telemetry[beam_id]["phasmask_Ypos"].append( phasemask_center[beam_id][1] )
# #             telemetry[beam_id]["phasmask_Xerr"].append( x_error )
# #             telemetry[beam_id]["phasmask_Yerr"].append( y_error )

# #             # Move 
# #             message = f"moveabs BMX{beam_id} {phasemask_center[beam_id][0]}"
# #             ok =  send_and_get_response(message) 
# #             print(ok)
# #             message = f"moveabs BMY{beam_id} {phasemask_center[beam_id][1]}"
# #             ok = send_and_get_response(message) 
# #             print(ok)
            
# #             # Check for convergence
# #             metric = np.sqrt(x_error**2 + y_error**2)
# #             if metric < args.tol:
# #                 print(f"Beam {beam_id} converged in {iteration + 1} iterations.")
# #                 complete_flag[beam_id] = True

            

# #             # taking slow for initial testing
# #             # if iteration > 0:
# #             #     #plot_telemetry(telemetry, savepath='delme.png')
# #             #     plt.figure();plt.imshow(cropped_img);plt.savefig('delme2.png')
# #             #     print("saving telemetry plot in rproject root delme.png to review.")
# #             #     input('continue?')

# # # some diagnostic plots  
# # plot_telemetry(telemetry, savepath='delme.png')

# # # get final image after convergence 
# # img = np.mean( c.get_data() ,axis=0) #  full image 
# # final_images = {}
# # for beam_id in args.beam_id:
# #     r1,r2,c1,c2 = baldr_pupils[f"{beam_id}"]
# #     cropped_img = interpolate_bad_pixels(img[r1:r2, c1:c2], bad_pixel_mask[r1:r2, c1:c2])
# #     final_images[beam_id] = cropped_img
# # plt.figure();plt.imshow(cropped_img);plt.savefig('delme2.png')


# # ii=0
# # plt.figure();plt.imshow( telemetry[beam_id]["img"][ii]);plt.savefig('delme.png')

# # # slideshow of images for a beam
# # #image_slideshow(telemetry, beam_id)