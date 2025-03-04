
import time 
import numpy as np 
import matplotlib.pyplot as plt
import asgard_alignment.controllino as co # for turning on / off source \
from asgard_alignment.DM_shm_ctrl import dmclass
import common.DM_basis_functions as dmbases
from pyBaldr import utilities as util
from xaosim.shmlib import shm

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

    plt.close()


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

    plt.close()



def setup(beam_ids, global_camera_shm, toml_file) :

    NNN = 10 # number of time get_data() called / appended

    print( 'setting up controllino and MDS ZMQ communication')

    controllino_port = '172.16.8.200'

    myco = co.Controllino(controllino_port)


    print( 'Reading in configurations') 

    I2A_dict = {}
    for beam_id in beam_ids:

        # read in TOML as dictionary for config 
        with open(toml_file.replace('#',f'{beam_id}'), "r") as f:
            config_dict = toml.load(f)
            # Baldr pupils from global frame 
            baldr_pupils = config_dict['baldr_pupils']
            I2A_dict[beam_id] = config_dict[f'beam{beam_id}']['I2A']


    # Set up global camera frame SHM 
    print('Setting up camera. You should manually set up camera settings before hand')
    c = shm(global_camera_shm)

    # set up DM SHMs 
    print( 'setting up DMs')
    dm_shm_dict = {}
    for beam_id in beam_ids:
        dm_shm_dict[beam_id] = dmclass( beam_id=beam_id )
        # zero all channels
        dm_shm_dict[beam_id].zero_all()
        # activate flat (does this on channel 1)
        dm_shm_dict[beam_id].activate_flat()
        # apply dm flat offset (does this on channel 2)
        #dm_shm_dict[beam_id].set_data( np.array( dm_flat_offsets[beam_id] ) )
    


    # Get Darks
    print( 'getting Darks')
    myco.turn_off("SBB")
    time.sleep(15)
    darks = []
    for _ in range(NNN):
        darks.append(  c.get_data() )

    darks = np.array( darks ).reshape(-1, darks[0].shape[1], darks[0].shape[2])

    myco.turn_on("SBB")
    time.sleep(10)

    dark_dict = {}
    for beam_id in beam_ids:
        r1,r2,c1,c2 = baldr_pupils[f"{beam_id}"]
        cropped_imgs = [nn[r1:r2,c1:c2] for nn in darks]
        dark_dict[beam_id] = cropped_imgs


    # Get reference pupils (later this can just be a SHM address)
    zwfs_pupils = {}
    clear_pupils = {}
    rel_offset = 200.0 #um phasemask offset for clear pupil
    print( 'Moving FPM out to get clear pupils')
    for beam_id in beam_ids:
        message = f"moverel BMX{beam_id} {rel_offset}"
        res = send_and_get_response(message)
        print(res) 
        time.sleep( 1 )
        message = f"moverel BMY{beam_id} {rel_offset}"
        res = send_and_get_response(message)
        print(res) 
        time.sleep(10)


    #Clear Pupil
    print( 'gettin clear pupils')
    N0s = []
    for _ in range(NNN):
         N0s.append(  c.get_data() )
    N0s = np.array(  N0s ).reshape(-1,  N0s[0].shape[1],  N0s[0].shape[2])


    for beam_id in beam_ids:
        r1,r2,c1,c2 = baldr_pupils[f"{beam_id}"]
        cropped_imgs = [nn[r1:r2,c1:c2] for nn in N0s]
        clear_pupils[beam_id] = cropped_imgs

        # move back 
        print( 'Moving FPM back in beam.')
        message = f"moverel BMX{beam_id} {-rel_offset}"
        res = send_and_get_response(message)
        print(res) 
        time.sleep(1)
        message = f"moverel BMY{beam_id} {-rel_offset}"
        res = send_and_get_response(message)
        print(res) 
        time.sleep(10)


    # check the alignment is still ok 
    beam = int( input( "\ndo you want to check the phasemasks for a beam. Enter beam number (1,2,3,4) or 0 to continue\n") )
    while beam :
        save_tmp = 'delme.png'
        print(f'open {save_tmp } to see generated images after each iteration')
        
        move_relative_and_get_image(cam=c, beam=beam, baldr_pupils = baldr_pupils, phasemask=state_dict["socket"],  savefigName = save_tmp, use_multideviceserver=True )
        
        beam = int( input( "\ndo you want to check the phasemasks for a beam. Enter beam number (1,2,3,4) or 0 to continue\n") )
    

    # ZWFS Pupil
    print( 'Getting ZWFS pupils')
    I0s = []
    for _ in range(NNN):
        I0s.append(  c.get_data() )
    I0s = np.array(  I0s ).reshape(-1,  I0s[0].shape[1],  I0s[0].shape[2])

    for beam_id in beam_ids:
        r1,r2,c1,c2 = baldr_pupils[f"{beam_id}"]
        #cropped_img = interpolate_bad_pixels(img[r1:r2, c1:c2], bad_pixel_mask[r1:r2, c1:c2])
        cropped_img = [nn[r1:r2,c1:c2] for nn in I0s] #/np.mean(img[r1:r2, c1:c2][pupil_masks[bb]])
        zwfs_pupils[beam_id] = cropped_img

    return c, dm_shm_dict, dark_dict, zwfs_pupils, clear_pupils, baldr_pupils, I2A_dict



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


