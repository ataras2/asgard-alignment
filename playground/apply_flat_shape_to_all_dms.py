import sys
import pandas as pd
import numpy as np
sys.path.insert(1,'/opt/Boston Micromachines/lib/Python3/site-packages/')
import bmc
import atexit
import os

# Path to DM shapes
DMshapes_path = 'DMShapes/'

# Dictionary with DM serial numbers
DM_serial_number_dict = {
    "1": "17DW019#113",
    "2": "17DW019#053",
    "3": "17DW019#093",
    "4": "17DW019#122"
}

# Dictionary to store DM objects, accessible after running
dm_objects = {}

# Function to close all DMs on exit
def close_all_dms():
    for dm_id, dm in dm_objects.items():
        try:
            dm.close_dm()
            print(f"Closed connection to DM{dm_id}")
        except Exception as e:
            print(f"Failed to close DM{dm_id}: {e}")

# Register the function to close all DMs at exit
atexit.register(close_all_dms)

# Function to apply the flat map to a given DM
def apply_flat_to_dm(dm_id, dm_serial):
    dm = bmc.BmcDm()
    dm_err_flag = dm.open_dm(dm_serial)
    if dm_err_flag != 0:
        print(f"Error opening DM{dm_id} with serial {dm_serial}: {dm_err_flag}")
        return False

    # Load the flat map for this DM
    flat_map_file = os.path.join(DMshapes_path, f"{dm_serial}_FLAT_MAP_COMMANDS.csv")
    flat_map = pd.read_csv(flat_map_file, header=None)[0].values

    # Send the flat map to the DM
    dm.send_data(flat_map)
    print(f"\n\n===\nFlat map applied to DM{dm_id} (Serial: {dm_serial})\n===\n")

    # Store the DM object in the global dictionary
    dm_objects[dm_id] = dm
    return True

def main():
    # Apply the flat map to each DM
    for dm_id, dm_serial in DM_serial_number_dict.items():
        print(f"Applying flat map to DM{dm_id} (Serial: {dm_serial})...")
        success = apply_flat_to_dm(dm_id, dm_serial)
        if not success:
            print(f"Failed to apply flat map to DM{dm_id}")

if __name__ == "__main__":
    main()
