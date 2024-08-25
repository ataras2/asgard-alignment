import os
import json
import matplotlib.pyplot as plt
from datetime import datetime

# Directory containing the JSON files
directory = "/home/heimdallr/Documents/asgard-alignment"

# File matching pattern
file_pattern = "phase_positions_beam_3_"

# Dictionary to store data for each label
data = {}

# Loop through the files in the directory
for filename in os.listdir(directory):
    if filename.startswith(file_pattern) and filename.endswith(".json"):
        # Extract the timestamp from the filename
        timestamp_str = filename[len(file_pattern):-5]
        timestamp = datetime.strptime(timestamp_str, "%d-%m-%YT%H.%M.%S")
        
        # Read the JSON file
        filepath = os.path.join(directory, filename)
        with open(filepath, 'r') as f:
            positions = json.load(f)
        
        # Store the data
        for label, (x_pos, y_pos) in positions.items():
            if label not in data:
                data[label] = {'time': [], 'x': [], 'y': []}
            data[label]['time'].append(timestamp)
            data[label]['x'].append(x_pos)
            data[label]['y'].append(y_pos)

# Plotting the x and y positions vs time for each label
for label in data:
    plt.figure(figsize=(14, 6))
    
    # Plot x position vs time
    plt.subplot(1, 2, 1)
    plt.plot(data[label]['time'], data[label]['x'],'o')
    plt.xlabel('Time')
    plt.ylabel('X Position')
    plt.title(f'X Position vs Time for {label}')
    
    # Plot y position vs time
    plt.subplot(1, 2, 2)
    plt.plot(data[label]['time'], data[label]['y'],'o')
    plt.xlabel('Time')
    plt.ylabel('Y Position')
    plt.title(f'Y Position vs Time for {label}')
    
    # Adjust layout and show plot
    plt.tight_layout()
    plt.savefig(directory+f"/tmp/phasemask_drifts_{label}.png")
    
