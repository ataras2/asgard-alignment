Scripts For Alignment
======================

If you are unsure about any of the commands, you can always run `python <script> --help` to get more information, where `<script>` is the name of the script you want to run.

All scripts are run from the root directory of the repository:

.. code-block:: bash
    
    cd Documents/asgard-alignment


Running the MDS
^^^^^^^^^^^^^^^^
This script runs the MultiDeviceServer (MDS) which handles communication with various devices.

**Arguments:**

- `--config`: Path to the configuration file (required).
- `--host`: Host address (default: "localhost").
- `--port`: Port number (default: 5555).

**Example Usage:**

.. code-block:: bash

    python asgard_alignment/MultiDeviceServer.py --config motor_info.json 

Running the Engineering GUI
^^^^^^^^^^^^^^^^^^^^^^^^^^^^
This script runs the Streamlit-based GUI for engineering purposes.

**Arguments:**

- `--host`: Server host (default: "localhost").
- `--port`: Server port (default: 5555).
- `--timeout`: Response timeout in milliseconds (default: 5000).

**Example Usage:**

.. code-block:: bash

    python -m streamlit run common/m_engineering_GUI.py --host localhost --port 5555 --timeout 5000

Analysing fringe data
^^^^^^^^^^^^^^^^^^^^^


This script analyses fringe data using Fourier transforms.

**Arguments:**

- `--savepath`: Path to the image data (required).
- `--pswidth`: Width of the power spectrum (default: 24).
- `--xcrop`: Number of pixels to crop from the x-axis (default: 512).
- `--ycrop`: Number of pixels to crop from the y-axis (default: 512).

**Example Usage:**

.. code-block:: bash

    python common/m_fourier_fringes_from_imgs.py --savepath data/<DATE>/sol_12_run<X> 

.. code-block:: bash

    python common/m_fourier_fringes_from_imgs.py --savepath data/<DATE>/heim_13run1 

Solarstein fringe search
^^^^^^^^^^^^^^^^^^^^^^^^
This script controls the motor position and captures images using a camera for Solarstein.

**Arguments:**

- `--savepath`: Path to save images and data (required).
- `--axis`: Axis number (required).
- `--start`: Start position in micrometers (required).
- `--end`: End position in micrometers (required).
- `--step`: Step size in micrometers (required).
- `--n_imgs`: Number of images to capture at each position (default: 3).
- `--host`: Server host (default: "localhost").
- `--port`: Server port (default: 5555).
- `--timeout`: Response timeout in milliseconds (default: 5000).

**Example Usage:**

.. code-block:: bash

    python solarstein/m_step_and_save.py --savepath ./data --axis SDL12 --start 3000 --end 7000 --step 10 

Heimdallr fringe search
^^^^^^^^^^^^^^^^^^^^^^^
In Heimdallr, the main script to run is `m_step_and_save_newport.py`.


This script is used to control the motor position and capture images using a camera. The script communicates with a server via ZeroMQ to move the motor to specified positions and captures images at each position.

**Arguments:**

- `--path`: Path to save images and data.
- `--beam`: Beam number to move (choices: 1, 2, 3, 4).
- `--host`: Server host (default: "localhost").
- `--port`: Server port (default: 5555).
- `--timeout`: Response timeout in milliseconds (default: 5000).
- `--start`: Start position in mm (default: 6).
- `--stop`: End position in mm (default: 10).
- `--step_size`: Step size in mm (default: 0.010).
- `--n_imgs`: Number of images to average per position (default: 3).

**Example Usage:**

A minimal usage of the args:

.. code-block:: bash

    python heimdallr/m_step_and_save_newport.py --path ./data --beam 2

A typical usage of most args:

.. code-block:: bash

    python heimdallr/m_step_and_save_newport.py --path ./data --beam 2 --start 6 --stop 10 --step_size 0.010

A usage of all args:

.. code-block:: bash

    python heimdallr/m_step_and_save_newport.py --path ./data --beam 2 --host 192.168.1.1 --port 5555 --timeout 5000 --start 6 --stop 10 --step_size 0.010 --n_imgs 3

Running the Strehl Ratio GUI
^^^^^^^^^^^^^^^^^^^^^^^^^^^^
This script runs the Strehl Ratio GUI for various optical setups.

**Arguments:**

- `--focal_length`: Focal length of the lens in meters (required).
- `--beam_diameter`: Diameter of the beam in meters (required).
- `--wavelength`: Wavelength of the laser in meters (default: 635e-9).
- `--pixel_scale`: Pixel scale of the camera in meters (default: 3.45e-6).
- `--width_to_spot_size_ratio`: The ratio of the width of the region of interest to the spot size (default: 2.0).
- `--method`: The method to use for finding the maximum value, one of naive, smoothed, gauss_diff (default: gauss_diff).

**Example Usage:**

To run the Strehl ratio GUI for OAP1:

.. code-block:: bash

    python playground/spin_SR_gui.py --focal_length 681e-3 --beam_diameter 18e-3 --wavelength 635e-9 --pixel_scale 3.45e-6 --width_to_spot_size_ratio 3.0 --method gauss_diff

To run the Strehl ratio GUI for the spherical mirror:

.. code-block:: bash

    python playground/spin_SR_gui.py --focal_length 2.0 --beam_diameter 12e-3 --wavelength 635e-9 --pixel_scale 3.45e-6 --width_to_spot_size_ratio 3.0 --method gauss_diff

To run the Strehl ratio GUI for Baldr OAP:

.. code-block:: bash

    python playground/spin_SR_gui.py --focal_length 254e-3 --beam_diameter 12e-3 --wavelength 535e-9 --pixel_scale 3.45e-6 --width_to_spot_size_ratio 3.0 --method gauss_diff

To run it in SIMULATION MODE:

.. code-block:: bash

    python /playground/spin_SR_gui.py --focal_length 254e-3 --beam_diameter 12e-3 --wavelength 635e-9 --pixel_scale 3.45e-6 --width_to_spot_size_ratio 3.0 --method gauss_diff --simulation True --sim_fname data/lab_imgs/beam_4_f400_laser_top_level_nd3.png

Poke Ramp (interaction matricies of varying amplitudess)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^
This script pokes each actuator on the DMs over a specified range of values and records images using the CRED ONE camera. The default configuration uses the globalresetcds mode and settings from a default_cred1_config.json file. Users can modify the FPS, gain, and modal basis for DM operations. The script outputs a fits files with the respective images , DM commands and the system states. 

**Arguments:**

- `--host`: Server hostname or IP address for ZeroMQ communication (default: localhost).
- `--port`: Port number for ZeroMQ communication (default: 5555).
- `--timeout`: Response timeout in milliseconds (default: 5000).
- `--dm_config_path`: Path to the DM configuration JSON file (default: /home/heimdallr/Documents/asgard-alignment/config_files/dm_serial_numbers.json).
- `--DMshapes_path`: Path to the directory containing DM shapes (default: /home/heimdallr/Documents/asgard-alignment/DMShapes/).
- `--data_path`: Directory to store calibration data FITS files (default: /home/heimdallr/data/baldr_calibration/<timestamp>/).
- `--number_images_recorded_per_cmd`: Number of images recorded per DM command, typically averaged (default: 5).
- `--number_amp_samples`: Number of amplitude steps to apply to DM actuators (default: 18).
- `--amp_max`: Maximum DM amplitude to apply, normalized to 0–1. The script ramps between +/- of this value (default: 18).
- `--basis_name`: Name of the modal basis to use for DM operations. Options include Zonal, Zonal_pinned_edges, Hadamard, Zernike, fourier, etc. (default: Zonal).
- `--number_of_modesv: Number of modes to include in the modal basis (default: 140).
- `--cam_fps`: Frames per second for the camera (default: 50).
- `--cam_gain`: Camera gain setting (default: 1).


**Example Usage:**
To poke actuators using the Zonal basis with default FPS and gain:

.. code-block:: bash

    python calibration/poke_dm_actuators.py --host localhost --port 5555 --timeout 5000 \
    --dm_config_path /home/heimdallr/Documents/asgard-alignment/config_files/dm_serial_numbers.json \
    --DMshapes_path /home/heimdallr/Documents/asgard-alignment/DMShapes/ \
    --data_path /home/heimdallr/data/baldr_calibration/01-12-2024/ \
    --number_images_recorded_per_cmd 5 \
    --number_amp_samples 18 \
    --amp_max 0.1 \
    --basis_name Zonal \
    --number_of_modes 140 \
    --cam_fps 50 \
    --cam_gain 1


Applying Kolmogorov Phase Screens on the Deformable Mirror (DM)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
This script applies Kolmogorov phase screens across multiple DMs (four by default) and records images using the CRED ONE camera. The camera operates in the globalresetcds mode with a default configuration from default_cred1_config.json. Users can modify FPS, gain, and scaling factors for phase screens as required.

**Arguments:**

- `--host`: Server hostname or IP address for ZeroMQ communication (default: localhost).
- `--portv: Port number for ZeroMQ communication (default: 5555).
- `--timeout`: Response timeout in milliseconds (default: 5000).
- `--dm_config_path`: Path to the DM configuration JSON file (default: /home/heimdallr/Documents/asgard-alignment/config_files/dm_serial_numbers.json).
- `--DMshapes_path`: Path to the directory containing DM shapes (default: /home/heimdallr/Documents/asgard-alignment/DMShapes/).
- `--data_path`: Directory to store phase screen calibration FITS files (default: /home/heimdallr/data/baldr_calibration/<timestamp>/).
- `--number_of_rolls`: Number of iterations (rolls) of the Kolmogorov phase screen applied to the DM (default: 1000).
- `--scaling_factor`: Scaling factor for the amplitude of the phase screen applied to the DM. Keep this value low to avoid saturation (default: 0.05).
- `--number_images_recorded_per_cmd`: Number of images recorded for each DM command, typically averaged (default: 5).
- `--cam_fps`: Frames per second for the camera (default: 50).
- `--cam_gain`: Camera gain setting (default: 1).

**Example Usage:**

To apply Kolmogorov screens with default parameters:

.. code-block:: bash

    python calibration/apply_kolmogorov_screens.py --host localhost --port 5555 --timeout 5000 \
    --dm_config_path /home/heimdallr/Documents/asgard-alignment/config_files/dm_serial_numbers.json \
    --DMshapes_path /home/heimdallr/Documents/asgard-alignment/DMShapes/ \
    --data_path /home/heimdallr/data/baldr_calibration/01-12-2024/ \
    --number_of_rolls 1000 \
    --scaling_factor 0.05 \
    --number_images_recorded_per_cmd 5 \
    --cam_fps 50 \
    --cam_gain 1


Baldr Calibration Script 
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

This script performs basic calibration for the Baldr wavefront sensing and control system. It generates analytics and visualizations for diagnostics and creates a PDF report summarizing the results. The calibration process includes affine transform calibration between the DM actuators and camera pixels, eigenmode analysis, and fitting of ZWFS responses. Also records motor states for stability analysis. The default output is a json file that can be used to initialise the Baldr RTC. The calibration report is optional.

**Arguments:**
- `--ramp_file`: Path to the ramp FITS file (obligatory).
- `--kol_file`: Path to the Kolmogorov phase screen FITS file. Optional (default: None).
- `--beam`: Beam number for calibration (default: 2).
- `--write_report`: Boolean to enable/disable writing the PDF report (default: True).
- `--a`: Amplitude index for calculating +/- around flat during DM/detector transform calibration (default: 2).
- `--signal_method`: Method to compute the ZWFS signal, e.g., I-I0/N0 (default: 'I-I0/N0').
- `--control_method`: Control method for DM operations, e.g., zonal_linear (default: 'zonal_linear').
- `--output_config_filename`: Output JSON filename for saving calibration results (default: baldr_transform_dict_beam2_<timestamp>.json).
- `--output_report_dir`: Directory to save PDF reports (default: /home/heimdallr/Documents/asgard-alignment/calibration/reports/<timestamp>/).
- `--fig_path`: Directory to save generated figures (default: /home/heimdallr/Documents/asgard-alignment/calibration/reports/<timestamp>/figures/).

**Example Usage:**

To calibrate with a pokeramp file and generate a report:

.. code-block:: bash

    python baldr_calibration.py \
    /path/to/ramp_file.fits \
    --beam 2 
    --write_report True 
    --fig_path /custom/figures/dir

To include Kolmogorov phase screen analysis:

.. code-block:: bash

    python baldr_calibration.py \
    /path/to/ramp_file.fits \
    --kol_file /path/to/kolmogorov_file.fits \
    --beam 2 \
    --write_report True \


