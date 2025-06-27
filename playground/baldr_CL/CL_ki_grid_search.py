#!/usr/bin/env python3
import subprocess

# List of ki values to test.
ki_values = [0.2, 0.0, 0.1, 0.3, 0.5, 0.6, 0.7]

# Common command parts.
# We assume you want to run: "python playground/baldr_CL/CL.py ..."
base_command = ["python", "playground/baldr_CL/CL.py"]
common_args = [
    "--number_of_iterations", "5000",
    '--number_of_turb_iterations',"400",
    "--r0", "0.2",
    "--V", "0.4",
    "--folder_pth", "/home/asg/Videos/CL_ki_scan_1kHz_gain15_zonal_automode/"
]

# Iterate over each ki value and run the command.
for ki in ki_values:
    # Construct full command.
    cmd = base_command + common_args + ["--ki", str(ki)]
    
    print("Running command:", " ".join(cmd))
    # Run the command and check for errors.
    subprocess.run(cmd, check=True)