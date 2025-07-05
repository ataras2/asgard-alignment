from xaosim.shmlib import shm
import numpy as np
import matplotlib.pyplot as plt
import time
import zmq

moving_beam = 1
reference_beam = 3




assert reference_beam != moving_beam

### SHM/processing functions

def get_ps(ps_stream):
    full_stack =  ps_stream.get_data()**2
    half_width = full_stack.shape[1] // 2
    stacked_ps = np.array([full_stack[:,:half_width],full_stack[:,half_width:]])
    return stacked_ps

def make_mask(img, centre, radius):
    y,x = np.ogrid[
        -centre[0] : img.shape[0] - centre[0], -centre[1] : img.shape[1] - centre[1]
    ]
    mask = x**2 + y**2 < radius **2
    return mask

def ps_to_v2(stacked_ps, mask):
    peak = np.max(stacked_ps*mask, axis=(-1,-2))
    bias = np.array([np.median(stacked_ps[i][mask]) for i in range(2)])
    dc_term = 1e4
    v2 = (peak-bias)/dc_term
    return v2

def capture_and_process_data(ps_stream, mask, n_samp=5):
    v2s = np.zeros(n_samp, 2)
    for i in range(n_samp):
        ps = get_ps(ps_stream)
        v2 = ps_to_v2(ps, mask)
        v2s[i] = v2
    
    return v2s

### MDS functions

def open_mds_connection():
    context = zmq.Context()
    socket = context.socket(zmq.REQ)
    socket.setsockopt(
            zmq.RCVTIMEO, 10000
        )
    server_address = (
            f"tcp://192.168.100.2:5555"
        )
    socket.connect(server_address)
    return socket


def send_and_get_response(socket,message):
    print(f"sending: {message}")
    socket.send_string(message)
    response = socket.recv_string()
    return response.strip()

def read_stepper(socket, beam):
    return send_and_get_response(socket, f"read HPOL{beam}")

def read_delay_line(socket, beam):
    return send_and_get_response(socket, f"read HFO{beam}")

def mv_delay_line(socket,beam, pos, blocking=True):
    send_and_get_response(socket, f"moveabs HFO{beam} {pos}")
    if blocking:
        cur_pos = read_delay_line(beam)
        while not np.isclose(pos, cur_pos, atol=1.0):
            time.sleep(0.5)
            cur_pos = read_delay_line(beam)

def mv_stepper(socket, beam, pos, blocking=True):
    send_and_get_response(socket, f"moveabs HPOL{beam} {pos}")

    if blocking:
        cur_pos = read_stepper(beam)
        while not np.isclose(pos, cur_pos, atol=0.5):
            time.sleep(0.5)
            cur_pos = read_stepper(beam)


if __name__ == "__main__":
    ps_stream = shm('/dev/shm/hei_ps.im.shm')
    mds = open_mds_connection()

    res = send_and_get_response(mds, f"read HFO{moving_beam}")
    print(res)

    ps = get_ps(ps_stream)
    print(ps.shape)
    
    shp = ps[0].shape
    mask = make_mask(ps[0], np.array(shp)//2, radius = 3)
    # plt.imshow(mask)
    # plt.show()


    stepper_start = read_stepper(mds, moving_beam)
