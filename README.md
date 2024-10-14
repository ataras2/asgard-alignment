

# Useful commands

If you are unsure about any of the commands, you can always run `python <script> --help` to get more information, where `<script>` is the name of the script you want to run.

## Running solarstein aligment

To run the fringe data taking:
```bash
python solarstein/m_step_and_save.py --savepath data/<DATE>/sol_12_run<X> --bs_num 7 --start 3000 --end 8000 --step 5
```

```bash
python solarstein/m_fourier_fringes_from_imgs.py --savepath data/<DATE>/sol_12_runX
```