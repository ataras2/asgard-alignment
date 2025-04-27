import argparse
import numpy as np
import toml
import time
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from collections import deque

from asgard_alignment import FLI_Cameras as FLI

def load_config(toml_path, beam_id):
    with open(toml_path.replace('#', str(beam_id)), "r") as f:
        return toml.load(f)

def init_camera(shm_path, roi):
    return FLI.fli(shm_path, roi=roi, quick_startup=True)

def compute_error(img, I2A, I0dm, N0dm, bias_dm, dark_dm, gain, fps, I2M_LO, I2M_HO):
    img_dm = (I2A @ img.reshape(-1)) - (dark_dm / fps * gain) - bias_dm
    signal = (img_dm - I0dm) / N0dm
    return I2M_LO @ signal, I2M_HO @ signal, img

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--beam_id", type=int, default=1)
    parser.add_argument("--toml_file", type=str, default="/home/asg/Progs/repos/asgard-alignment/config_files/baldr_config_#.toml")
    parser.add_argument("--global_camera_shm", type=str, default="/dev/shm/cred1.im.shm")
    args = parser.parse_args()

    beam_id = args.beam_id
    config = load_config(args.toml_file, beam_id)

    I2A = np.array(config[f'beam{beam_id}']['I2A'])
    I2M_LO = np.array(config[f'beam{beam_id}']['H3']['ctrl_model']['I2M_LO'])
    I2M_HO = np.array(config[f'beam{beam_id}']['H3']['ctrl_model']['I2M_HO'])

    I0 = np.array(config[f'beam{beam_id}']['H3']['ctrl_model']['I0'])
    N0 = np.array(config[f'beam{beam_id}']['H3']['ctrl_model']['N0'])
    bias = np.array(config[f'beam{beam_id}']['H3']['ctrl_model']['bias'])
    dark = np.array(config[f'beam{beam_id}']['H3']['ctrl_model']['dark'])

    gain = float(config[f'beam{beam_id}']['H3']['ctrl_model']['camera_config']['gain'])
    fps = float(config[f'beam{beam_id}']['H3']['ctrl_model']['camera_config']['fps'])

    I0dm = gain / fps * (I2A @ I0)
    N0dm = gain / fps * (I2A @ N0)
    bias_dm = I2A @ bias
    dark_dm = I2A @ dark

    secondary_mask = np.array(config[f'beam{beam_id}']['pupil_mask']['secondary']).astype(bool)
    bias_sec = bias[secondary_mask.reshape(-1)][4]
    dark_sec = dark[secondary_mask.reshape(-1)][4]
    exterior_mask = np.array(config[f'beam{beam_id}']['pupil_mask']['exterior']).astype(bool)

    cam = init_camera(args.global_camera_shm, config['baldr_pupils'][str(beam_id)])

    buf_len = 500
    e_lo_bufs = [deque(maxlen=buf_len) for _ in range(I2M_LO.shape[0])]
    e_ho_bufs = [deque(maxlen=buf_len) for _ in range(I2M_HO.shape[0])]
    i_s_buf = deque(maxlen=buf_len)
    i_ext_buf = deque(maxlen=buf_len)

    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 8))
    lines_lo = [ax1.plot([], [], label=f'e_TT[{i}]')[0] for i in range(I2M_LO.shape[0])]
    lines_ho = [ax2.plot([], [], label=f'e_HO[{i}]')[0] for i in range(I2M_HO.shape[0])]
    ax1.set_title("e_TT (Tip/Tilt)")
    ax2.set_title("e_HO (High Order)")

    ax3.set_title("i_s (secondary) and mean(exterior)")
    ax3.set_xlabel("Frame")

    ax1.set_xlim(0, buf_len)
    ax2.set_xlim(0, buf_len)
    ax3.set_xlim(0, buf_len)

    ax1.grid(True)
    ax2.grid(True)
    ax3.grid(True)

    ax3_right = ax3.twinx()
    line_is, = ax3.plot([], [], label='i_s', color='tab:blue')
    line_ext, = ax3_right.plot([], [], label='exterior mean', color='tab:orange')

    def init():
        for line in lines_lo + lines_ho:
            line.set_data([], [])
        line_is.set_data([], [])
        line_ext.set_data([], [])
        return lines_lo + lines_ho + [line_is, line_ext]

    def update(frame):
        img = cam.get_image(apply_manual_reduction=False)
        e_lo, e_ho, img = compute_error(img, I2A, I0dm, N0dm, bias_dm, dark_dm, gain, fps, I2M_LO, I2M_HO)

        for i, val in enumerate(e_lo):
            e_lo_bufs[i].append(val)
            x = np.arange(len(e_lo_bufs[i]))
            lines_lo[i].set_data(x, list(e_lo_bufs[i]))

        for i, val in enumerate(e_ho):
            e_ho_bufs[i].append(val)
            x = np.arange(len(e_ho_bufs[i]))
            lines_ho[i].set_data(x, list(e_ho_bufs[i]))

        i_s = img[secondary_mask][4] - bias_sec - (1 / fps * dark_sec)
        i_ext = np.mean(img[exterior_mask])
        i_s_buf.append(i_s)
        i_ext_buf.append(i_ext)

        x_s = np.arange(len(i_s_buf))
        line_is.set_data(x_s, list(i_s_buf))
        line_ext.set_data(x_s, list(i_ext_buf))

        ax1.relim()
        ax1.autoscale_view()
        ax2.relim()
        ax2.autoscale_view()
        ax3.relim()
        ax3.autoscale_view()
        ax3_right.relim()
        ax3_right.autoscale_view()

        return lines_lo + lines_ho + [line_is, line_ext]

    ani = FuncAnimation(fig, update, init_func=init, interval=10, blit=False)
    fig.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()
# import argparse
# import numpy as np
# import toml
# import time
# import matplotlib.pyplot as plt
# from matplotlib.animation import FuncAnimation
# from collections import deque

# from asgard_alignment import FLI_Cameras as FLI

# def load_config(toml_path, beam_id):
#     with open(toml_path.replace('#', str(beam_id)), "r") as f:
#         return toml.load(f)

# def init_camera(shm_path, roi):
#     return FLI.fli(shm_path, roi=roi, quick_startup=True)

# def compute_error(img, I2A, I0dm, N0dm, bias_dm, dark_dm, gain, fps, I2M_LO, I2M_HO):
#     img_dm = (I2A @ img.reshape(-1)) - (dark_dm / fps * gain) - bias_dm
#     signal = (img_dm - I0dm) / N0dm
#     return I2M_LO @ signal, I2M_HO @ signal, img

# def main():
#     parser = argparse.ArgumentParser()
#     parser.add_argument("--beam_id", type=int, default=1)
#     parser.add_argument("--toml_file", type=str, default="/home/asg/Progs/repos/asgard-alignment/config_files/baldr_config_#.toml")
#     parser.add_argument("--global_camera_shm", type=str, default="/dev/shm/cred1.im.shm")
#     args = parser.parse_args()

#     beam_id = args.beam_id
#     config = load_config(args.toml_file, beam_id)

#     I2A = np.array(config[f'beam{beam_id}']['I2A'])
#     I2M_LO = np.array(config[f'beam{beam_id}']['H3']['ctrl_model']['I2M_LO'])
#     I2M_HO = np.array(config[f'beam{beam_id}']['H3']['ctrl_model']['I2M_HO'])

#     I0 = np.array(config[f'beam{beam_id}']['H3']['ctrl_model']['I0'])
#     N0 = np.array(config[f'beam{beam_id}']['H3']['ctrl_model']['N0'])
#     bias = np.array(config[f'beam{beam_id}']['H3']['ctrl_model']['bias'])
#     dark = np.array(config[f'beam{beam_id}']['H3']['ctrl_model']['dark'])

#     gain = float(config[f'beam{beam_id}']['H3']['ctrl_model']['camera_config']['gain'])
#     fps = float(config[f'beam{beam_id}']['H3']['ctrl_model']['camera_config']['fps'])

#     I0dm = gain / fps * (I2A @ I0)
#     N0dm = gain / fps * (I2A @ N0)
#     bias_dm = I2A @ bias
#     dark_dm = I2A @ dark

#     secondary_mask = np.array(config[f'beam{beam_id}']['pupil_mask']['secondary']).astype(bool)
#     bias_sec = bias[secondary_mask.reshape(-1)][4]
#     dark_sec = dark[secondary_mask.reshape(-1)][4]
#     exterior_mask = np.array(config[f'beam{beam_id}']['pupil_mask']['exterior']).astype(bool)

#     cam = init_camera(args.global_camera_shm, config['baldr_pupils'][str(beam_id)])

#     buf_len = 500
#     e_lo_bufs = [deque(maxlen=buf_len) for _ in range(I2M_LO.shape[0])]
#     e_ho_bufs = [deque(maxlen=buf_len) for _ in range(I2M_HO.shape[0])]
#     i_s_buf = deque(maxlen=buf_len)
#     i_ext_buf = deque(maxlen=buf_len)

#     fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 8))
#     lines_lo = [ax1.plot([], [], label=f'e_TT[{i}]')[0] for i in range(I2M_LO.shape[0])]
#     lines_ho = [ax2.plot([], [], label=f'e_HO[{i}]')[0] for i in range(I2M_HO.shape[0])]
#     line_is, = ax3.plot([], [], label='i_s')
#     line_ext, = ax3.plot([], [], label='mean exterior')

#     ax1.set_title("e_TT (Tip/Tilt)")
#     ax2.set_title("e_HO (High Order)")
#     ax3.set_title("i_s (secondary) and mean(exterior)")

#     for ax in (ax1, ax2, ax3):
#         ax.set_xlim(0, buf_len)
#         ax.grid(True)
#     ax3.legend(loc='upper right')

#     def init():
#         for line in lines_lo + lines_ho:
#             line.set_data([], [])
#         line_is.set_data([], [])
#         line_ext.set_data([], [])
#         return lines_lo + lines_ho + [line_is, line_ext]

#     def update(frame):
#         img = cam.get_image(apply_manual_reduction=False)
#         e_lo, e_ho, img = compute_error(img, I2A, I0dm, N0dm, bias_dm, dark_dm, gain, fps, I2M_LO, I2M_HO)

#         for i, val in enumerate(e_lo):
#             e_lo_bufs[i].append(val)
#             x = np.arange(len(e_lo_bufs[i]))
#             lines_lo[i].set_data(x, list(e_lo_bufs[i]))

#         for i, val in enumerate(e_ho):
#             e_ho_bufs[i].append(val)
#             x = np.arange(len(e_ho_bufs[i]))
#             lines_ho[i].set_data(x, list(e_ho_bufs[i]))

#         i_s = img[secondary_mask][4] - bias_sec - (1 / fps * dark_sec)
#         i_ext = np.mean(img[exterior_mask])
#         i_s_buf.append(i_s)
#         i_ext_buf.append(i_ext)

#         x_s = np.arange(len(i_s_buf))
#         line_is.set_data(x_s, list(i_s_buf))
#         line_ext.set_data(x_s, list(i_ext_buf))

#         for ax in (ax1, ax2, ax3):
#             ax.relim()
#             ax.autoscale_view()

#         return lines_lo + lines_ho + [line_is, line_ext]

#     ani = FuncAnimation(fig, update, init_func=init, interval=10, blit=False)
#     plt.tight_layout()
#     plt.show()


# if __name__ == "__main__":
#     main()

    