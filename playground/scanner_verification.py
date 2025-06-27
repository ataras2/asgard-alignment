import numpy as np
import matplotlib.pyplot as plt
import common.phasemask_centering_tool as pct  # Assuming the cross_scan function is here

# Set the parameters for the test
starting_point = (0, 0)  # Center of the cross scan (origin)
dx = 0.1  # Step size in the X-direction (spacing between points)
dy = 0.1  # Step size in the Y-direction (spacing between points)
X_amp = 2.0  # Half-length of the cross in the X-direction (Amplitude)
Y_amp = 2.0  # Half-length of the cross in the Y-direction (Amplitude)
angles = [0, 45, 90, 135]  # Angles to rotate the cross scan

# Create a function to plot the cross scan results at different angles
# def plot_cross_scan_results(starting_point, dx, dy, X_amp, Y_amp, angles):
    

# Create subplots for each angle
fig, axes = plt.subplots(len(angles), 1, figsize=(8, 6 * len(angles)))

# If there's only one plot, make sure axes is a list
if len(angles) == 1:
    axes = [axes]  # Ensure axes is always a list

# Loop through all angles and plot the cross scan
for i, angle in enumerate(angles):
    # Get the cross scan points
    scan_points = pct.cross_scan(starting_point, dx, dy, X_amp, Y_amp, angle)
    
    # Unzip the list of points into separate x and y lists
    line_x, line_y = zip(*scan_points)
    
    # Plot the points for this angle
    axes[i].plot(line_x, line_y, 'bo-', label=f'Cross Scan (angle={angle}°)')
    
    # Set the plot title and labels
    axes[i].set_title(f"Cross Scan Pattern for angle = {angle}°")
    axes[i].set_xlabel('X')
    axes[i].set_ylabel('Y')
    axes[i].set_aspect('equal', 'box')  # Keep the aspect ratio equal
    axes[i].legend()
    
plt.tight_layout()  # Adjust layout so plots don't overlap
plt.show()
