import numpy as np
from asgard_alignment.DM_shm_ctrl import dmclass
import pyBaldr.utilities as util 

#flat_offsets = {}
#for beam_id in [1, 2, 3, 4]:
#    dm = dmclass(beam_id)

    #flat_offset_cmd = np.loadtxt(dm.select_flat_cmd_offset("/home/asg/Progs/repos/asgard-alignment/DMShapes/"))

    #flat_offsets[beam_id] = flat_offset_cmd


#im_list = [util.get_DM_command_in_2D( flat_offsets[beam_id] ) for beam_id in [1, 2, 3, 4]]
#titles = [f"Beam {x}" for x in [1,2,3,4]]
#cbars = ["DM Units [0-1]" for _ in im_list]

#util.nice_heatmap_subplots( im_list=im_list ,title_list=titles, cbar_label_list=cbars,savefig='delme.png')


# saving heimdallr flats 
from asgard_alignment.DM_shm_ctrl import dmclass
import pyBaldr.utilities as util 


flat_offsets = {}
for beam_id in [1, 2, 3, 4]:
    dm = dmclass(beam_id)
    for i in [0,1,2]:
        print( f"mean channel {i} data = {np.mean(dm.shms[i].get_data())}")
    #offset = dm.shms[2].get_data()

    #fname = f"/home/asg/Progs/repos/asgard-alignment/DMShapes/heim_flat_beam_{beam_id}.txt"
    #np.savetxt( fname, offset ,fmt="%.7f")

    #flat_offsets[beam_id] = flat_offset_cmd
