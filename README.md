

# Useful commands

If you are unsure about any of the commands, you can always run `python <script> --help` to get more information, where `<script>` is the name of the script you want to run.



## Running the MDS
```bash
python asgard-alignment/MultiDeviceServer.py -c <config file name>
```


## Running the engineering GUI
```bash
python -m streamlit run common/m_engineering_GUI.py
```

## Running solarstein aligment

To run the fringe data taking (with SpinView and Zaber launcher closed):
```bash
conda activate spinview
cd Documents/asgard-alignment
python solarstein/m_step_and_save.py --savepath data/<DATE>/sol_12_run<X> --bs_num 7 --start 3000 --end 8000 --step 5
```

```bash
python common/m_fourier_fringes_from_imgs.py --savepath data/<DATE>/sol_12_runX
```
