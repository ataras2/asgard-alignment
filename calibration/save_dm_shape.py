# saving dm shape for a particular beam and DM shared memory channel, option to save them for baldr or heimdallr flats (in correct format) 
import numpy as np 
import argparse
import subprocess
from pathlib import Path
import datetime
from asgard_alignment.DM_shm_ctrl import dmclass
import pyBaldr.utilities as util 


def get_git_root() -> Path:
    return Path(
        subprocess.check_output(["git", "rev-parse", "--show-toplevel"], text=True).strip()
    )


parser = argparse.ArgumentParser(description="DM shared memory shape saving options.")

# Beam ids: provided as a comma-separated string and converted to a list of ints.
parser.add_argument(
    "--beam_id",
    type=lambda s: [int(item) for item in s.split(",")],
    default=[1],
    help="Comma-separated beam IDs to apply. Default: 1,2,3,4"
)

# TOML file path; default is relative to the current file's directory.
parser.add_argument(
    "--shm_channel",
    type=int,
    default=2,
    help="which channel (int) from the DM shared memory to save. Typically channel 0 is the flat, channel 1 is alignment shape (e.g. cross), channel 2 is baldr, channel 3 is heimdallr. Default: %(default)s"
)

parser.add_argument(
    "--filename",
    type=str,
    default= get_git_root() / "DMShapes" / "saved_dm_shape.txt",
    help="path/to/file to save data to. Default: %(default)s"
)
  
parser.add_argument(
    "--flat_calibration",
    type=str,
    default=None,
    help="is this saving a flat calibration? if not enter None, otherweise input 'heim' or 'baldr' to save in the correct format. If not None this overwrites the filename option. Default: %(default)s"
)


args = parser.parse_args()

tstamp = datetime.datetime.now().strftime("%d-%m-%YT%H.%M.%S")
#tstamp_rough = datetime.datetime.now().strftime("%d-%m-%Y")

for beam_id in args.beam_id:
    
    dm = dmclass(beam_id)
    # # just for simulation mode if we want calibrated baldr flat to make DM zero'd
    # dm = dmclass(beam_id)
    # dm.activate_flat()
    # offset = -dm.shms[0].get_data()  # Wait for the semaphore to be set
    offset = dm.shms[args.shm_channel].get_data()
    
    # for i in range(4):
    #     dm.shms[i].set_data(0 * offset)
    # dm.shm0.post_sems(1)
    #dm.shms[args.shm_channel].set_data(0 * offset)

    
    #fname = f"/home/asg/Progs/repos/asgard-alignment/DMShapes/heim_flat_beam_{beam_id}.txt"
    if args.flat_calibration is None:
        np.savetxt( args.filename.replace(".txt",f"{beam_id}.txt"), offset ,fmt="%.7f")
    else:
        # if flat_calibration is heim or baldr, save in the correct format
        if args.flat_calibration.lower() == "heim":
            args.filename = get_git_root() / "DMShapes" / f"heim_flat_beam_{beam_id}.txt" 
            np.savetxt(args.filename, offset, fmt="%.7f")
        elif args.flat_calibration.lower() == "baldr":
            args.filename = get_git_root() / "DMShapes" / f"BEAM{beam_id}_FLAT_MAP_OFFSETS_{tstamp}.txt"
            np.savetxt(args.filename, util.convert_12x12_to_140(offset), fmt="%.7f") # convert to 140x1 format for baldr (rtc convention for BMC DMs)
        else:
            raise ValueError("flat_calibration must be 'heim', 'baldr', or None")
    #flat_offsets[beam_id] = flat_offset_cmd

    print(f"DM shape from beam {beam_id}, shm channel = {args.shm_channel} saved successfully as:\n   {args.filename}")
    
    dm.close(erase_file=False)  # Close the shared memory without erasing it