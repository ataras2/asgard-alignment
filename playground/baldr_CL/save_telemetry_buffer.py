import argparse
import numpy as np
import toml
import time
import os
import threading
from collections import deque
from datetime import datetime
import copy
from astropy.io import fits

from asgard_alignment import FLI_Cameras as FLI
from asgard_alignment.DM_shm_ctrl import dmclass

def load_config(toml_path, beam_id):
    with open(toml_path.replace('#', str(beam_id)), "r") as f:
        return toml.load(f)

def init_camera(shm_path, roi):
    return FLI.fli(shm_path, roi=roi, quick_startup=True)

def compute_signals(img, I2A, I0dm, N0dm, bias_dm, dark_dm, gain, fps):
    img_dm = (I2A @ img.reshape(-1)) - (dark_dm / fps * gain) - bias_dm
    signal = (img_dm - I0dm) / N0dm
    return signal, img_dm

def save_telemetry(telemetry, static_data, save_dir):
    os.makedirs(save_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filepath = os.path.join(save_dir, f"baldr_telem_{timestamp}.fits")

    hdul = fits.HDUList()
    hdul.append(fits.PrimaryHDU())

   

    telemetry_copy = copy.deepcopy(telemetry)  # Deep copy to prevent mutation during write

    for key, val in telemetry_copy.items():
        try:
            data = np.array(val)
            hdul.append(fits.ImageHDU(data=data, name=key))
        except Exception as e:
            print(f"Error saving key '{key}': {e}")

            
    # for key, val in telemetry.items():
    #     data = np.array(val)
    #     hdul.append(fits.ImageHDU(data=data, name=key))

    for key, val in static_data.items():
        data = np.array(val)
        hdul.append(fits.ImageHDU(data=data, name=f"STATIC_{key}"))

    hdul.writeto(filepath, overwrite=True)
    print(f"Saved telemetry to {filepath}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--beam_id", type=int, default=1)
    parser.add_argument("--toml_file", type=str, default=os.path.join("/usr/local/etc/baldr/", "baldr_config_#.toml"))
    parser.add_argument("--global_camera_shm", type=str, default="/dev/shm/cred1.im.shm")
    args = parser.parse_args()

    beam_id = args.beam_id
    config = load_config(args.toml_file, beam_id)

    pupil_mask = np.array(config.get(f"beam{beam_id}", {}).get("pupil_mask", {}).get("mask", None) ).astype(bool)   # matrix bool
    I2A = np.array(config[f'beam{beam_id}']['I2A'])
    I2M_LO = np.array(config[f'beam{beam_id}']['H3']['ctrl_model']['I2M_LO'])
    I2M_HO = np.array(config[f'beam{beam_id}']['H3']['ctrl_model']['I2M_HO'])
    I0 = np.array(config[f'beam{beam_id}']['H3']['ctrl_model']['I0'])
    N0 = np.array(config[f'beam{beam_id}']['H3']['ctrl_model']['N0'])
    bias = np.array(config[f'beam{beam_id}']['H3']['ctrl_model']['bias'])
    dark = np.array(config[f'beam{beam_id}']['H3']['ctrl_model']['dark'])
    gain = float(config[f'beam{beam_id}']['H3']['ctrl_model']['camera_config']['gain'])
    fps = float(config[f'beam{beam_id}']['H3']['ctrl_model']['camera_config']['fps'])
    inside_edge_filt = np.array(config.get(f"beam{beam_id}", {}).get(f"{args.phasemask}", {}).get("ctrl_model", None).get("inner_pupil_filt", None) )#.astype(bool)

    I0dm = gain / fps * (I2A @ I0)
    N0dm = gain / fps * (I2A @ N0)
    bias_dm = I2A @ bias
    dark_dm = I2A @ dark

    secondary_mask = np.array(config[f'beam{beam_id}']['pupil_mask']['secondary']).astype(bool)
    bias_sec = bias[secondary_mask.reshape(-1)][4]
    dark_sec = dark[secondary_mask.reshape(-1)][4]
    exterior_mask = np.array(config[f'beam{beam_id}']['pupil_mask']['exterior']).astype(bool)

    cam = init_camera(args.global_camera_shm, config['baldr_pupils'][str(beam_id)])
    dm = dmclass(beam_id=beam_id)

    dmtight_mask = I2A @ np.array([int(a) for a in inside_edge_filt])

    buf_len = 5000
    telemetry = {
        "e_TT": [],
        "e_HO": [],
        "i_s": [],
        "i_exterior": [],
        "img": [],
        "i_dm": [],
        "dm_ch1": [],
        "dm_ch2": [],
        "dm_ch3": [],
    }

    static_data = {
        "I2A": I2A,
        "I2M_LO": I2M_LO,
        "I2M_HO": I2M_HO,
        "I0dm": I0dm,
        "N0dm": N0dm,
        "pupil_mask":pupil_mask,
        "inside_edge_filt":inside_edge_filt,
        "dmtight_mask_dm":dmtight_mask,
        "bias_dm": bias_dm,
        "dark_dm": dark_dm
    }

    def user_input_thread():
        while True:
            folder = input("Enter folder name (relative to /home/asg/data/baldr_last_day/) to save telemetry: ")
            save_path = os.path.join("/home/asg/data/baldr_last_day", folder)
            save_telemetry(telemetry, static_data, save_path)

    threading.Thread(target=user_input_thread, daemon=True).start()

    while True:
        t0 = time.time()
        img = cam.get_image(apply_manual_reduction=False)
        signal, i_dm = compute_signals(img, I2A, I0dm, N0dm, bias_dm, dark_dm, gain, fps)
        e_tt = I2M_LO @ signal
        e_ho = I2M_HO @ signal

        i_s = img[secondary_mask][4] - bias_sec - (1 / fps * dark_sec)
        i_ext = np.mean(img[exterior_mask])

        telemetry["e_TT"].append(e_tt)
        telemetry["e_HO"].append(e_ho)
        telemetry["i_s"].append(i_s)
        telemetry["i_exterior"].append(i_ext)
        telemetry["img"].append(img)
        telemetry["i_dm"].append(i_dm)
        telemetry["dm_ch1"].append(dm.shms[1].get_data().copy())
        telemetry["dm_ch2"].append(dm.shms[2].get_data().copy())
        telemetry["dm_ch3"].append(dm.shms[3].get_data().copy())

        if len(telemetry["e_TT"]) > buf_len:
            for k in telemetry:
                telemetry[k].pop(0)

        time.sleep(0.005)  # Approx. 1 kHz

if __name__ == "__main__":
    print("Script started")
    main()


"""

# check 
import numpy as np
from astropy.io import fits
import matplotlib.pyplot as plt
import os

# === CONFIGURATION ===
fits_path = os.path.expanduser("~/data/baldr_last_day/turb_100nmRMS_lock_beam1/baldr_telem_20250424_001858.fits")

# === LOAD FITS ===
hdul = fits.open(fits_path)
print("FITS file opened. Available extensions:")
for hdu in hdul:
    print(f"- {hdu.name}")

# === EXTRACT DATA ===
e_HO = hdul['e_HO'].data  # Shape: (N_frames, N_modes)
dm_ch2 = hdul['dm_ch2'].data  # Shape: (N_frames, H, W)
dm_ch3 = hdul['dm_ch3'].data  # Same shape

#tight_dm_mask = hdul['dmtight_mask_dm'].data

# === PLOT: Modal Errors ===
plt.figure(figsize=(10, 4))
# for i in range(min(5, e_HO.shape[1])):  # Plot up to 5 modes
#     plt.plot(e_HO[:, i], label=f'Mode {i}')

plt.plot(e_HO)
plt.title('High-Order Modal Errors (e_HO)')
plt.xlabel('Frame')
plt.ylabel('Error')
plt.legend()
plt.grid(True)
plt.savefig('delme.png')
plt.close()

# === PLOT: DM Residual (Ch2 - Ch3) for last frame ===
residual = dm_ch2 + dm_ch3
plt.figure(figsize=(5, 4))
plt.plot(residual.reshape(-1,144))
plt.ylabel('residual dm units')
plt.xlabel('sample')
plt.tight_layout()
plt.savefig('delme.png')

plt.close()

"""