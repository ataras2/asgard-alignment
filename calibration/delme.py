import os
import json
from astropy.io import fits
import numpy as np

def process_fits_files(base_directory, output_json):
    """
    Iterates through subdirectories, processes FITS files, and writes the average data to a JSON file.

    Parameters:
        base_directory (str): The base directory containing subdirectories with FITS files.
        output_json (str): The output JSON file to save the results.

    Returns:
        None
    """
    # Initialize the dictionary to store averaged images
    averaged_data = {}

    # Walk through the directory tree
    for root, _, files in os.walk(base_directory):
        for file in files:
            # Process only FITS files
            if file.endswith(".fits"):
                file_path = os.path.join(root, file)

                try:
                    # Open the FITS file
                    with fits.open(file_path) as hdul:
                        # Extract and average the 'FRAMES' extension
                        if 'FRAMES' in hdul:
                            img = np.mean(hdul['FRAMES'].data, axis=0)

                            # Extract the timestamp from the file name
                            timestamp = file.split('_')[-1].replace('.fits', '')

                            # Store the averaged image in the dictionary
                            averaged_data[timestamp] = img.tolist()  # Convert NumPy array to list for JSON serialization
                        else:
                            print(f"'FRAMES' extension not found in {file_path}")
                except Exception as e:
                    print(f"Failed to process {file_path}: {e}")

    # Write the dictionary to a JSON file
    with open(output_json, 'w') as json_file:
        json.dump(averaged_data, json_file, indent=4)
    print(f"Processed data saved to {output_json}")


base_directory = "/home/heimdallr/data/stability_analysis/24-12-2024/pupils"
output_json = base_directory + "/averaged_fits_data.json"
process_fits_files(base_directory, output_json)
