## sometime from legacy i still use global frame with the registered sub pupuil 
# need to check how cred1 crops the subframe is consistent otherwise we will be off by pixels

import pyBaldr.utilities as util 
import pyzelda.ztools as ztools
import datetime
from xaosim.shmlib import shm
from asgard_alignment import FLI_Cameras as FLI
import matplotlib.pyplot as plt
import numpy as np

beam = 1
toml_file = os.path.join("/usr/local/etc/baldr/", "baldr_config_#.toml") 


# ensure camera is crop_mode 1, split_mode 1
c_sub  = FLI.fli(f"/dev/shm/baldr{beam}.im.shm", roi = [None,None,None,None])
sub_img = c_sub.mySHM.get_data() 

_ = input("change crop_mode 0")

# change camera crop_mode 0

#read in cropped pupil from config file 
with open(toml_file.replace('#',f'{beam}'), "r") as f:
    config_dict = toml.load(f)
    # Baldr pupils from global frame 
    baldr_pupils = config_dict['baldr_pupils']

r1,r2,c1,c2 = baldr_pupils[f'{beam}']

c_full  = FLI.fli("/dev/shm/cred1.im.shm", roi = [None,None,None,None])
full_img = np.mean( c_full.get_data() ,axis=0) # full frame get_data returns cube instead of frame 
my_crop = full_img[r1:r2,c1:c2]

delta = my_crop - sub_img

util.nice_heatmap_subplots([sub_img, my_crop, delta], title_list =["subframe","crop from global","delta"])
plt.show()
