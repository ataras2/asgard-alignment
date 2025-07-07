# %%
from xaosim.shmlib import shm
import numpy as np
import matplotlib.pyplot as plt
import time
import zmq
import os
from tqdm import tqdm

moving_beam = 1
reference_beam = 3

stepper_range = 3  # will do +/- this number
n_stepper_values = 6

dl_range = 100e-3  # in mm
n_dl_values = 9


cur_datetime = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())

# savepth = None
base = ["data", "stepper_vs_delay_line"]
fname = f"stepper_vs_linbo_{moving_beam}{reference_beam}_{cur_datetime}.npz"
savepth = os.path.join(*base, fname)

assert reference_beam != moving_beam

# %%
### SHM/processing functions


def get_ps(ps_stream):
    full_stack = ps_stream.get_data() ** 2
    half_width = full_stack.shape[1] // 2
    stacked_ps = np.array([full_stack[:, :half_width], full_stack[:, half_width:]])
    return stacked_ps


def make_mask(img, centre, radius):
    y, x = np.ogrid[
        -centre[0] : img.shape[0] - centre[0], -centre[1] : img.shape[1] - centre[1]
    ]
    mask = x**2 + y**2 < radius**2
    return mask


def ps_to_v2(stacked_ps, mask):
    peak = np.max(stacked_ps * mask, axis=(-1, -2))
    bias = np.array([np.median(stacked_ps[i][mask]) for i in range(2)])
    dc_term = 1e4**2
    v2 = (peak - bias) / dc_term
    return v2


def capture_and_process_data(ps_stream, mask, n_samp=5):
    v2s = np.zeros((n_samp, 2))
    for i in range(n_samp):
        ps = get_ps(ps_stream)
        v2 = ps_to_v2(ps, mask)
        v2s[i] = v2

    return v2s


### MDS functions


def open_mds_connection():
    context = zmq.Context()
    socket = context.socket(zmq.REQ)
    socket.setsockopt(zmq.RCVTIMEO, 10000)
    server_address = "tcp://192.168.100.2:5555"
    socket.connect(server_address)
    return socket


def send_and_get_response(socket, message):
    print(f"sending: {message}")
    socket.send_string(message)
    response = socket.recv_string()
    return response.strip()


def read_stepper(socket, beam):
    return int(send_and_get_response(socket, f"read HPOL{beam}"))


def read_dl(socket, beam):
    res = send_and_get_response(socket, f"read HFO{beam}")
    print(res)
    return float(res)


def mv_dl(socket, beam, pos, blocking=True):
    send_and_get_response(socket, f"moveabs HFO{beam} {pos}")
    if blocking:
        cur_pos = read_dl(socket, beam)
        while not np.isclose(pos, cur_pos, atol=0.5):
            time.sleep(0.2)
            cur_pos = read_dl(socket, beam)


def mv_stepper(socket, beam, pos, blocking=True):
    send_and_get_response(socket, f"moveabs HPOL{beam} {float(pos)}")

    if blocking:
        cur_pos = read_stepper(socket, beam)
        while not np.isclose(pos, cur_pos, atol=0.5e-3):
            time.sleep(0.2)
            cur_pos = read_stepper(socket, beam)


# %%
ps_stream = shm("/dev/shm/hei_ps.im.shm")
mds = open_mds_connection()

res = send_and_get_response(mds, f"read HFO{moving_beam}")
print(res)

ps = get_ps(ps_stream)
print(ps.shape)

shp = ps[0].shape
mask = np.logical_not(make_mask(ps[0], np.array(shp) // 2, radius=4))


# plt.subplot(121)
# plt.imshow(mask)
# plt.subplot(122)
# ps_stream.catch_up_with_sem(semid)
# ps = get_ps(ps_stream)
# plt.imshow(np.log10(ps[0]))
# plt.show()

# %%
stepper_start = read_stepper(mds, moving_beam)
stepper_vals = np.arange(
    stepper_start - stepper_range,
    stepper_start + stepper_range,
    (2 * stepper_range) // n_stepper_values,
)
print(stepper_vals)

dl_start = read_dl(mds, moving_beam)
dl_vals = np.arange(
    dl_start - dl_range, dl_start + dl_range, (2 * dl_range) / n_dl_values
)
print(dl_vals)

input("enter to continue")

# %%
n_samples = 50
stats = np.zeros([len(stepper_vals), len(dl_vals), n_samples, 2])

for i, step in tqdm(enumerate(stepper_vals)):
    mv_stepper(mds, moving_beam, step)
    for j, dlp in tqdm(enumerate(dl_vals)):
        mv_dl(mds, moving_beam, dlp, blocking=True)
        time.sleep(0.5)
        v2s = capture_and_process_data(ps_stream, mask, n_samples)
        stats[i, j] = v2s

# reset back to og positions
mv_stepper(mds, moving_beam, stepper_start)
mv_dl(mds, moving_beam, dl_start)

# %%
np.savez(
    savepth,
    stats=stats,
    stepper_vals=stepper_vals,
    dl_vals=dl_vals,
    moving_beam=moving_beam,
    reference_beam=reference_beam,
    ps=ps,
    mask=mask,
)
