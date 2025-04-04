# read in config 

# I2M , I2A , Strehl , secondary mask
import numpy as np 
import toml
import argparse
import zmq
import time
import toml
import os 
import matplotlib.pyplot as plt
import glob

from astropy.io import fits

from asgard_alignment import FLI_Cameras as FLI
import pyBaldr.utilities as util 


MDS_port = 5555
MDS_host = 'localhost'
context = zmq.Context()
context.socket(zmq.REQ)
socket = context.socket(zmq.REQ)
socket.setsockopt(zmq.RCVTIMEO, 5000)
server_address = f"tcp://{MDS_host}:{MDS_port}"
socket.connect(server_address)
state_dict = {"message_history": [], "socket": socket}


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



# Camera 
c = FLI.fli(args.global_camera_shm, roi = baldr_pupils[f'{beam_id}'])

resp = send_and_get_response("off SBB")
print(resp)