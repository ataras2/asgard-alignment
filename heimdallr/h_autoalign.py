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


target_pixels = (None, None)  # target pixels for the blob centre (K1)
if target_pixels[0] is None:
    raise NotImplementedError()


class HeimdallrAA:
    def __init__(self):
        self.mds = self._open_mds_connection()

        self._shutter_pause_time = 0.5  # seconds to pause after shuttering
        self._n_frames = 5  # number of frames to average for each beam

    # MDS interface
    def _open_mds_connection(self):
        context = zmq.Context()
        socket = context.socket(zmq.REQ)
        socket.setsockopt(zmq.RCVTIMEO, 10000)
        server_address = "tcp://192.168.100.2:5555"
        socket.connect(server_address)
        return socket

    def _send_and_get_response(self, message):
        self.mds.send_string(message)
        response = self.mds.recv_string()
        return response.strip()

    # processing
    def _find_blob_centre(self, frame):
        # use ndi median filter and then a gaussian filter on cropped image
        pass

    def _get_frame(self):
        pass

    def _get_and_process_blob(self):
        pass

    def autoalign_parallel(self):
        # 1. shutter all beams off except 1, pause for a short time
        # 2. find the centre of the blob in the image
        # 3. save the pixel offsets
        # 4. repeat for all beams
        pixel_offsets = {}

        for target_beam in range(1, 5):
            beam_to_close = [i for i in range(1, 5) if i != target_beam]
            msg = f"h_shut close {''.join(map(str, beam_to_close))}"
            self._send_and_get_response(msg)
            msg = f"h_shut open {target_beam}"
            self._send_and_get_response(msg)
            time.sleep(self._shutter_pause_time)

            blob_centre = self._get_and_process_blob()

            # calculate the pixel offsets from the target pixels
            pixel_offsets[target_beam] = np.array(target_pixels) - np.array(blob_centre)

        # 5. unshutter all beams
        msg = "h_shut open 1234"
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

        for beam, uv_cmd in uv_commands.items():
            pass
        # self.devices[axes[0]].move_relative(uv_commands[0][0])
        # self.devices[axes[2]].move_relative(uv_commands[2][0])
        # time.sleep(0.5)
        # self.devices[axes[1]].move_relative(uv_commands[1][0])
        # self.devices[axes[3]].move_relative(uv_commands[3][0])
