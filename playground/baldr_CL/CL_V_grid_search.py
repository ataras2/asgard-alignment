#!/usr/bin/env python3
import subprocess

# List of ki values to test.
V_values = [0.2, 0.5, 1, 2, 5, 10]

# Common command parts.
# We assume you want to run: "python playground/baldr_CL/CL.py ..."
base_command = ["python", "playground/baldr_CL/CL.py"]
common_args = [
    "--number_of_iterations", "5000",
    '--number_of_turb_iterations',"2000",
    "--r0", "0.3",
    "--ki", "0.2",
    "--folder_pth", "/home/asg/Videos/CL_V_scan_1kHz_gain15_zonal/"
]

# Iterate over each ki value and run the command.
for V in V_values:
    # Construct full command.
    cmd = base_command + common_args + ["--V", str(V)]
    
    print("Running command:", " ".join(cmd))
    # Run the command and check for errors.
    subprocess.run(cmd, check=True)