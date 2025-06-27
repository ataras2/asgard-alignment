
# Documentation

Some documentation can be found at [https://asgard-alignment.readthedocs.io/en/latest/](https://asgard-alignment.readthedocs.io/en/latest/). In particular the `Script For Alignment` section is useful for running the scripts in this repository.

# Installed scripts and settings

The key scripts are:

* *mds* : the Multi-device server. It relies on a single settings file, which is usually "motor_info_full_system.json".
* *eng_gui* : the Engineering GUI. This requires no settings files. However, it saves and restores motor states.

# Useful commands

If you are unsure about any of the commands, you can always run `python <script> --help` to get more information, where `<script>` is the name of the script you want to run. 

All scripts are run from the root directory of the repository:
```bash
cd Documents/asgard-alignment
```


## Running the MDS directly from the repository
```bash
python asgard_alignment/MultiDeviceServer.py -c <config file name>
```

Typically we use config file "motor_info_full_system.json", so this is
```bash
python asgard_alignment/MultiDeviceServer.py -c motor_info_full_system.json
```


## Running the engineering GUI directly from the repository
```bash
python -m streamlit run common/m_engineering_GUI.py
```

## Running solarstein fringe search

To run the fringe data taking (with SpinView and Zaber launcher closed):
```bash
python solarstein/m_step_and_save.py --savepath data/<DATE>/sol_12_run<X> --bs_num 7 --start 3000 --end 8000 --step 5
```

```bash
python common/m_fourier_fringes_from_imgs.py --savepath data/<DATE>/sol_12_runX
```

## Running Heimdallr fringe search

To run the fringe data taking (with SpinView and Zaber launcher closed):
```bash
python heimdallr/m_step_and_save_newport.py --path data/<DATE>/heim_13run1 --beam 1 --start 8 --stop 12 --step 0.01
```
Then run the Fourier analysis:
```bash
python common/m_fourier_fringes_from_imgs.py --savepath data/<DATE>/heim_13run1
```
Noting that usually the true fringe position is ~20um before the peak due a delay in the camera.


## Strehl ratio GUI

To run the Strehl ratio GUI for OAP1:
```bash
python playground/spin_SR_gui.py --focal_length 681e-3 --beam_diameter 18e-3 --wavelength 635e-9 --pixel_scale 3.45e-6 --width_to_spot_size_ratio 3.0 --method gauss_diff
```


To run the Strehl ratio GUI for the spherical mirror:
```bash
python playground/spin_SR_gui.py --focal_length 2.0 --beam_diameter 12e-3 --wavelength 635e-9 --pixel_scale 3.45e-6 --width_to_spot_size_ratio 3.0 --method gauss_diff
```


To run the Strehl ratio GUI for Baldr OAP:
```bash
python playground/spin_SR_gui.py --focal_length 254e-3 --beam_diameter 12e-3 --wavelength 535e-9 --pixel_scale 3.45e-6 --width_to_spot_size_ratio 3.0 --method gauss_diff
```
