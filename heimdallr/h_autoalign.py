"""
Autoalign the heimdallr beams using the c red one data stream and the MDS
Uses only K1, since it is brighter

The overall structure is:
1. shutter all beams off except 1, pause for a short time
2. find the centre of the blob in the image
3. save the pixel offsets
4. repeat for all beams
5. unshutter all beams
6. move them using the offsets + moveimage like calculation
"""

from xaosim.shmlib import shm
import numpy as np
import matplotlib.pyplot as plt
import time
import asgard_alignment.Engineering as asgE
import zmq
import os
from tqdm import tqdm
import scipy.ndimage as ndi


target_pixels = (28, 50)  # target pixels for the blob centre (K1)
if target_pixels[0] is None:
    raise NotImplementedError()


class HeimdallrAA:
    def __init__(self, shutter_pause_time=0.5, n_frames=5):
        self.mds = self._open_mds_connection()

        self.stream = self._open_stream_connection()

        self._shutter_pause_time = (
            shutter_pause_time  # seconds to pause after shuttering
        )
        self._n_frames = n_frames  # number of frames to average for each beam

        self.row_bnds = (0, 128)

    # MDS interface
    def _open_mds_connection(self):
        context = zmq.Context()
        socket = context.socket(zmq.REQ)
        socket.setsockopt(zmq.RCVTIMEO, 10000)
        server_address = "tcp://192.168.100.2:5555"
        socket.connect(server_address)
        return socket

    def _send_and_get_response(self, message):
        print("sending", message)
        self.mds.send_string(message)
        response = self.mds.recv_string()
        print("response", response)
        return response.strip()

    # stream interface
    def _open_stream_connection(self):
        stream_path = "/dev/shm/cred1.im.shm"
        if not os.path.exists(stream_path):
            raise FileNotFoundError(f"Stream file {stream_path} does not exist.")
        return shm(stream_path)

    # processing
    def _find_blob_centre(self, frame):
        # use ndi median filter and then a gaussian filter on cropped image
        cropped_frame = frame[self.row_bnds[0] : self.row_bnds[1], :]

        filtered_frame = ndi.median_filter(cropped_frame, size=3)
        filtered_frame = ndi.gaussian_filter(filtered_frame, sigma=2)

        max_loc_cropped = np.unravel_index(
            np.argmax(filtered_frame), filtered_frame.shape
        )
        max_loc_cropped = np.array(max_loc_cropped)
        max_loc = max_loc_cropped + np.array([self.row_bnds[0], 0])

        return max_loc

    def _get_frame(self):
        full_frame = self.stream.get_data().mean(0)
        return full_frame

    def _get_and_process_blob(self):
        full_frame = self._get_frame()
        blob_centre = self._find_blob_centre(full_frame)
        return blob_centre

    def autoalign_parallel(self):
        # 1. shutter all beams off except 1, pause for a short time
        # 2. find the centre of the blob in the image
        # 3. save the pixel offsets
        # 4. repeat for all beams
        pixel_offsets = {}

        msg = f"h_shut close 2"
        self._send_and_get_response(msg)
        msg = f"h_shut close 3"
        self._send_and_get_response(msg)
        msg = f"h_shut close 4"
        self._send_and_get_response(msg)
        msg = f"h_shut open 1"
        self._send_and_get_response(msg)
        time.sleep(self._shutter_pause_time)

        for target_beam in range(1, 5):
            print(f"doing beam {target_beam}")
            if target_beam > 1:
                msg = f"h_shut close {target_beam-1}"
                self._send_and_get_response(msg)
                msg = f"h_shut open {target_beam}"
                self._send_and_get_response(msg)
                time.sleep(self._shutter_pause_time)

            blob_centre = self._get_and_process_blob()

            # calculate the pixel offsets from the target pixels
            # pixel_offsets[target_beam] = np.array(target_pixels) - np.array(blob_centre)
            pixel_offsets[target_beam] = np.array(
                [
                    target_pixels[1] - blob_centre[0],
                    target_pixels[0] - blob_centre[1]
                ]
            )

        # 5. unshutter all beams
        msg = "h_shut open 1,2,3,4"
        self._send_and_get_response(msg)
        time.sleep(self._shutter_pause_time)

        # 6. move them using the offsets + moveimage like calculation
        # key here is to parallelise
        uv_commands = {}
        for beam, offset in pixel_offsets.items():
            uv_cmd = asgE.move_img_calc("c_red_one_focus", beam, offset)
            uv_commands[beam] = uv_cmd

        # send commands
        axis_list = ["HTPP", "HTTP", "HTPI", "HTTI"]
        axes = [
            [axis + str(beam_number) for axis in axis_list]
            for beam_number in range(1, 5)
        ]

        print(pixel_offsets)
        for beam, uv_cmd in uv_commands.items():
            cmd = f"moverel {axes[beam-1][0]} {uv_cmd[0]}"
            self._send_and_get_response(cmd)
            cmd = f"moverel {axes[beam-1][2]} {uv_cmd[2]}"
            self._send_and_get_response(cmd)

        time.sleep(0.4)

        for beam, uv_cmd in uv_commands.items():
            cmd = f"moverel {axes[beam-1][1]} {uv_cmd[1]}"
            self._send_and_get_response(cmd)
            cmd = f"moverel {axes[beam-1][3]} {uv_cmd[3]}"
            self._send_and_get_response(cmd)


if __name__ == "__main__":
    shutter_pause_time = 2.5  # seconds to pause after shuttering
    n_frames = 5  # number of frames to average for each beam
    heimdallr_aa = HeimdallrAA(shutter_pause_time=shutter_pause_time, n_frames=n_frames)
    heimdallr_aa.autoalign_parallel()
    print("Autoalignment completed.")
