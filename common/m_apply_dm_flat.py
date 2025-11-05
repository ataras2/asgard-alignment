
import matplotlib.pyplot as plt
import argparse
import numpy as np
from xaosim.shmlib import shm
from asgard_alignment.DM_shm_ctrl import dmclass


def apply_flat( beam_list ):
    print( 'setting up DMs')
    dm_shm_dict = {}
    for beam in beam_list:
        dm_shm_dict[beam] = dmclass( beam_id=beam, main_chn=3 ) # we poke on ch3 so we can close TT on chn 2 with rtc when building IM 
        # zero all channels
        dm_shm_dict[beam].zero_all()
        
        if args.DM_flat.lower() == 'factory':
            # activate flat (does this on channel 1)
            dm_shm_dict[beam].activate_flat()
        elif args.DM_flat.lower() == 'baldr':
            # apply dm flat + calibrated offset (does this on channel 1)
            dm_shm_dict[beam].activate_calibrated_flat()

        elif args.DM_flat.lower() == 'heim':
            # not implemented in dmclass .. to do, so do it manually
            # heim flat is relative to factory flat so they are added 
            wdirtmp = "/home/asg/Progs/repos/asgard-alignment/DMShapes/"  # os.path.dirname(__file__)
            flat_cmd = dm_shm_dict[beam].cmd_2_map2D(
                np.loadtxt(
                    dm_shm_dict[beam].select_flat_cmd(
                        wdirtmp
                    )
                )
            )
            flat_cmd_offset = np.loadtxt(
                wdirtmp + f"heim_flat_beam_{beam}.txt"
            )
            dm_shm_dict[beam].shms[0].set_data(
                flat_cmd + flat_cmd_offset
            )
            
            dm_shm_dict[beam].shm0.post_sems(1)

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="flat dm standard")

    parser.add_argument(
        "--beam_id",
        type=lambda s: [int(item) for item in s.split(",")],
        default=[1,2,3,4], # 1, 2, 3, 4],
        help="Comma-separated beam IDs to apply. Default: 1,2,3,4"
    )


    parser.add_argument(
        "--DM_flat",
        type=str,
        default="baldr",
        help="What flat do we use on the DM during the calibration. either 'baldr','heim' or 'factory'. Default: %(default)s"
    )

    args=parser.parse_args()

    apply_flat( args.beam_id )

    print("done")