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


