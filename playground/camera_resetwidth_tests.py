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
from xaosim.shmlib import shm
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



#reset width from 8 to 64 clock periods.

# {'fps': 3000.0,
#  'gain': 5.0,
#  'testpattern': 'off',
#  'bias': 'off',
#  'flat': 'off',
#  'imagetags': 'on',
#  'led': 'on',
#  'events': 'on',
#  'extsynchro': 'off',
#  'rawimages': 'off',
#  'cooling': 'on',
#  'mode': 'globalresetcds',
#  'resetwidth': '20',
#  'nbreadworeset': '2',
#  'cropping': 'off:1-10:1-256',
#  'cropping columns': '1-10',
#  'cropping rows': '1-256',
#  'aduoffset': '1000'}

# in cds '"Number of read without reset: 2\\r\\nfli-cli>"' ,"Reset witdth: 20\\r\\nfli-cli>"'
# Camera 
cc = shm("/dev/shm/cred1.im.shm", nosem=False)
c = FLI.fli("/dev/shm/cred1.im.shm", roi = [None,None,None,None])

resp = send_and_get_response("on SBB")
print(resp)

resp = c.send_fli_cmd("set gain 1")
time.sleep(1)
resp = c.send_fli_cmd("set fps 3000")
time.sleep(1)
resp = c.send_fli_cmd("set mode globalresetburst")
time.sleep(1)
resp = c.send_fli_cmd("set rawimages on")
time.sleep(1)
resp = c.send_fli_cmd("set nbreadworeset 200")
time.sleep(1)
resp = c.send_fli_cmd("set imagetag on")

c.send_fli_cmd("set resetwidth 5")


frames = cc.get_data(False, True).mean(0)

frames = cc.get_data(False, True)

frames  = cc.get_latest_data()

c.mySHM.catch_up_with_sem(c.semid)
frames = c.mySHM.get_latest_data()

cnt = [f[0][0] for f in frames]
i = [f[218][199] for f in frames]

plt.figure(); plt.plot( [np.mod( f[0][0], 100 ) for f in frames], np.mean( frames, axis=(1,2)) ); plt.savefig('delme.png')

plt.figure() 

# for width in widths:
#     resp = c.send_and_get_response(f"set resetwidth {width}")
    
#     frames = c.mySHM.get_latest_data()



imgs = []
fcnt = 0
cnt = 0
while cnt < 30:
    print( cnt )
    if cc.get_counter()!=fcnt: #i[0][0]!=fcnt:
        i = cc.get_latest_data_slice() #c.mySHM.get_latest_data_slice()
        fcnt = cc.get_counter() #i[0][0]
        imgs.append(i)
        cnt += 1

cnt = [f[0][0] for f in imgs]
plt.figure(); plt.plot(  np.mean( np.array(imgs)[:, 182+10:236-10, 190+10:244-10], axis=(1,2)) ); plt.savefig('delme.png')
plt.figure(); plt.plot(  np.mean( frames[:, 182+10:236-10, 190+10:244-10], axis=(1,2)) ); plt.savefig('delme.png')


cnt = [f[0][0] for f in frames]
plt.figure(); plt.plot( cnt[:50], np.mean( frames[:50, 182+10:236-10, 190+10:244-10], axis=(1,2)),'-o' ); 
plt.xlabel('frame count')
plt.ylabel("mean ADU on illum. pupil")
plt.savefig('delme.png')