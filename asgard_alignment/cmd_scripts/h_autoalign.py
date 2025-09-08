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

from scipy.optimize import curve_fit
import argparse


class HeimdallrAA:
    def __init__(self, shutter_pause_time, band, flux_threshold, savepth):
        # Set target_pixels and col_bnds based on band
        if band.upper() == "K1":
            self.target_pixels = (27, 49)
            self.col_bnds = (0, 160)
        elif band.upper() == "K2":
            self.target_pixels = (289, 51)
            self.col_bnds = (160, 320)
        else:
            raise ValueError("Unknown band: choose 'K1' or 'K2'.")

        self.flux_threshold = flux_threshold

        self.mds = self._open_mds_connection()

        self.stream = self._open_stream_connection()

        self._shutter_pause_time = (
            shutter_pause_time  # seconds to pause after shuttering
        )

        if savepth.lower() == "none":
            self.savepth = None
        else:
            self.savepth = savepth

        self.row_bnds = (0, 128)
        # self.col_bnds is set above

    def get_init_motors_state(self):
        motors = {}
        for beam in range(1, 5):
            axis_list = ["HTPP", "HTTP", "HTPI", "HTTI"]
            axes = [axis + str(beam) for axis in axis_list]
            motors[beam] = {}
            for axis in axes:
                cmd = f"read {axis}"
                resp = self._send_and_get_response(cmd)
                motors[beam][axis] = float(resp)
        return motors

    def set_complete_state(self, motors):
        for beam in range(1, 5):
            for axis, pos in motors[beam].items():
                cmd = f"moveabs {axis} {pos}"
                self._send_and_get_response(cmd)

    # MDS interface
    def _open_mds_connection(self):
        context = zmq.Context()
        socket = context.socket(zmq.REQ)
        socket.setsockopt(zmq.RCVTIMEO, 10000)
        server_address = "tcp://192.168.100.2:5555"
        socket.connect(server_address)
        return socket

    def _send_and_get_response(self, message):
        # print("sending", message)
        self.mds.send_string(message)
        response = self.mds.recv_string()
        # print("response", response)
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
        cropped_frame = frame[
            self.row_bnds[0] : self.row_bnds[1], self.col_bnds[0] : self.col_bnds[1]
        ]

        filtered_frame = ndi.median_filter(cropped_frame, size=3)
        filtered_frame = ndi.gaussian_filter(filtered_frame, sigma=2)

        max_loc_cropped = np.unravel_index(
            np.argmax(filtered_frame), filtered_frame.shape
        )
        max_loc_cropped = np.array(max_loc_cropped)
        max_loc = max_loc_cropped + np.array([self.row_bnds[0], self.col_bnds[0]])

        return max_loc

    def _get_frame(self):
        full_frame = self.stream.get_data().mean(0)
        return full_frame

    def _get_and_process_blob(self):
        full_frame = self._get_frame() - 1000.0
        blob_centre = self._find_blob_centre(full_frame)
        return blob_centre

    def _get_blob_with_flux(self, radius):
        full_frame = self._get_frame() - 1000.0
        blob_centre = self._find_blob_centre(full_frame)
        mask = self.circle_mask(full_frame.shape, blob_centre, radius)
        blob_flux = full_frame[mask].sum()
        return blob_centre, blob_flux

    @staticmethod
    def circle_mask(shape, center, radius):
        y, x = np.ogrid[: shape[0], : shape[1]]
        dist_from_center = np.sqrt((x - center[1]) ** 2 + (y - center[0]) ** 2)
        mask = dist_from_center <= radius
        return mask

    def open_all_shutters(self):
        msg = f"h_shut open 1,2,3,4"
        self._send_and_get_response(msg)

    def autoalign_coarse_parallel(self):
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

        flux_radius = 6

        for target_beam in range(1, 5):
            print(f"doing beam {target_beam}")
            if target_beam > 1:
                msg = f"h_shut close {target_beam-1}"
                self._send_and_get_response(msg)
                msg = f"h_shut open {target_beam}"
                self._send_and_get_response(msg)
                time.sleep(self._shutter_pause_time)

            blob_centre, flux = self._get_blob_with_flux(flux_radius)

            if flux < self.flux_threshold:
                print(
                    f"Beam {target_beam} has low flux: {flux}. setting delta to zero."
                )
                pixel_offsets[target_beam] = np.array([0, 0])
            else:
                print(f"Beam {target_beam} flux: {flux:.2f}, centre: {blob_centre}")
                # calculate the pixel offsets from the target pixels
                pixel_offsets[target_beam] = np.array(
                    [
                        self.target_pixels[1] - blob_centre[0],
                        self.target_pixels[0] - blob_centre[1],
                    ]
                )

        # 5. unshutter all beams
        self.open_all_shutters()
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

    # helper methods for pupil alignment fitting
    @staticmethod
    def fit_func(x, m, a, b, c):
        return -m * np.abs(x - a) - m * np.abs(x - b) + c

    @staticmethod
    def get_peak_flux_loc(pos, fluxes):
        fluxes /= np.max(fluxes)  # normalise
        params, _ = curve_fit(HeimdallrAA.fit_func, pos, fluxes, p0=[1, -0.05, 0.05, 1])

        return (params[1] + params[2]) / 2

    def autoalign_pupil(self, beam):
        mv_time = 2.5

        unused_beams = [b for b in range(1, 5) if b != beam]

        msg = f"h_shut close {','.join(map(str, unused_beams))}"
        self._send_and_get_response(msg)
        msg = f"h_shut open {beam}"
        self._send_and_get_response(msg)
        time.sleep(self._shutter_pause_time)

        # 2. rough align beam 3
        blob_centre = self._get_and_process_blob()

        pix_offset = np.array(
            [
                self.target_pixels[1] - blob_centre[0],
                self.target_pixels[0] - blob_centre[1],
            ]
        )
        uv_cmd = asgE.move_img_calc("c_red_one_focus", beam, pix_offset)

        axis_list = ["HTPP", "HTTP", "HTPI", "HTTI"]
        axes = [axis + str(beam) for axis in axis_list]

        cmd = f"moverel {axes[0]} {uv_cmd[0]}"
        self._send_and_get_response(cmd)
        cmd = f"moverel {axes[2]} {uv_cmd[2]}"
        self._send_and_get_response(cmd)
        time.sleep(0.5)
        cmd = f"moverel {axes[1]} {uv_cmd[1]}"
        self._send_and_get_response(cmd)
        cmd = f"moverel {axes[3]} {uv_cmd[3]}"
        self._send_and_get_response(cmd)
        time.sleep(0.5)

        # 3. move pupil to optimize flux
        pup_offset = 0.2  # mm
        n_samp = 9
        flux_beam_radius = 6  # pixels

        measurement_locs_x = np.linspace(-pup_offset, pup_offset, n_samp)
        relative_measurement_locs = np.array(
            [
                measurement_locs_x[i] - measurement_locs_x[i - 1]
                for i in range(1, n_samp)
            ]
        )
        relative_measurement_locs = np.concatenate(
            [[-pup_offset], relative_measurement_locs]
        )

        fluxes_x = []
        for delta in relative_measurement_locs:
            cmd = f"mv_pup c_red_one_focus {beam} {delta} {0.0}"
            self._send_and_get_response(cmd)
            print("sent", cmd)
            time.sleep(mv_time)
            centre, flux = self._get_blob_with_flux(flux_beam_radius)
            fluxes_x.append(flux)

            # check if centre is close to the edge of the frame
            for bound in self.col_bnds:
                if np.abs(centre[1] - bound) < 10:
                    print(
                        f"Warning: Beam {beam} centroid close to edge of frame ({centre[1]} pixels). Alignment may be unreliable."
                    )

        cur_pupil_pos = measurement_locs_x[-1]

        fluxes_x = np.array(fluxes_x)
        if np.max(fluxes_x) < self.flux_threshold:
            print(
                f"Beam {beam} has low flux: {np.max(fluxes_x)}. Setting pupil offset back to original."
            )
            del_needed = -cur_pupil_pos
            cmd = f"mv_pup c_red_one_focus {beam} {del_needed} {0.0}"
            self._send_and_get_response(cmd)
            return

        optimal_offset_x = self.get_peak_flux_loc(measurement_locs_x, fluxes_x)

        del_needed = optimal_offset_x - cur_pupil_pos

        print(
            f"Optimal pupil offset for beam {beam} x: {optimal_offset_x:.4f} mm ({del_needed:.4f} mm from current position)"
        )
        cmd = f"mv_pup c_red_one_focus {beam} {del_needed} {0.0}"
        self._send_and_get_response(cmd)

        time.sleep(2)

        fluxes_y = []
        for delta in relative_measurement_locs:
            cmd = f"mv_pup c_red_one_focus {beam} {0.0} {delta}"
            self._send_and_get_response(cmd)
            print("sent", cmd)
            time.sleep(2.5)
            centre, flux = self._get_blob_with_flux(flux_beam_radius)
            fluxes_y.append(flux)

            if np.abs(centre[1] - bound) < 10:
                print(
                    f"Warning: Beam {beam} centroid close to edge of frame ({centre[1]} pixels). Alignment may be unreliable."
                )

        fluxes_y = np.array(fluxes_y)

        optimal_offset_y = self.get_peak_flux_loc(measurement_locs_x, fluxes_y)
        cur_pupil_pos = measurement_locs_x[-1]

        del_needed = optimal_offset_y - cur_pupil_pos

        print(
            f"Optimal pupil offset for beam {beam} y: {optimal_offset_y:.4f} mm ({del_needed:.4f} mm from current position)"
        )
        cmd = f"mv_pup c_red_one_focus {beam} {0.0} {del_needed}"
        self._send_and_get_response(cmd)

        # now do x axis again to refine
        time.sleep(2)

        fluxes_x2 = []
        for delta in relative_measurement_locs:
            cmd = f"mv_pup c_red_one_focus {beam} {delta} {0.0}"
            self._send_and_get_response(cmd)
            print("sent", cmd)
            time.sleep(mv_time)
            centre, flux = self._get_blob_with_flux(flux_beam_radius)
            fluxes_x2.append(flux)

            if np.abs(centre[1] - bound) < 10:
                print(
                    f"Warning: Beam {beam} centroid close to edge of frame ({centre[1]} pixels). Alignment may be unreliable."
                )

        fluxes_x2 = np.array(fluxes_x2)

        optimal_offset_x2 = self.get_peak_flux_loc(measurement_locs_x, fluxes_x2)
        cur_pupil_pos = measurement_locs_x[-1]
        del_needed = optimal_offset_x2 - cur_pupil_pos
        print(
            f"Refined optimal pupil offset for beam {beam} x: {optimal_offset_x2:.4f} mm ({del_needed:.4f} mm from current position)"
        )

        cmd = f"mv_pup c_red_one_focus {beam} {del_needed} {0.0}"
        self._send_and_get_response(cmd)
        time.sleep(2)

        # saving
        if self.savepth is not None:
            np.savez(
                os.path.join(self.savepth, f"heimdallr_pupil_beam{beam}.npz"),
                meas_locs_x=measurement_locs_x,
                meas_locs_y=measurement_locs_x,
                fluxes_x=fluxes_x,
                fluxes_y=fluxes_y,
                optimal_offset_x=optimal_offset_x,
                optimal_offset_y=optimal_offset_y,
                measurement_locs_x2=measurement_locs_x,
                fluxes_x2=fluxes_x2,
                optimal_offset_x2=optimal_offset_x2,
            )

    def autoalign_pupil_all(self, plot):
        # just like autoalign_3_pupil but for all beams
        datas = {}
        for beam in range(1, 5):
            datas[beam] = self.autoalign_pupil(beam)
            # open all shutters
            self.open_all_shutters()

        if plot:
            self.plot_autoalign_pupil_all(datas)

    @staticmethod
    def plot_autoalign_pupil_all(data_all):

        n_beams = 4
        fig, axs = plt.subplots(2, 2, figsize=(10, 8), sharex=True, sharey=True)

        for beam in range(1, n_beams + 1):
            data = data_all[beam]
            positions = [data[f"meas_locs_{i}"] for i in ["x", "y", "x"]]
            fluxes = [data[f"fluxes_{i}"] for i in ["x", "y", "x2"]]
            optimal_offsets = [data[f"optimal_offset_{i}"] for i in ["x", "y", "x2"]]
            ax = axs.flat[beam - 1]
            ax.plot(positions[0], fluxes[0], "o-", label="x", color="C0")
            ax.plot(positions[1], fluxes[1], "o-", label="y", color="C1")
            ax.plot(positions[2], fluxes[2], "o-", label="x2", color="C2")
            for i in range(3):
                ax.axvline(optimal_offsets[i], color=f"C{i}", ls="--")
            ax.set_title(f"Beam {beam}")
            if beam in [3, 4]:
                ax.set_xlabel("Position")
            if beam in [1, 3]:
                ax.set_ylabel("Flux")
            if beam == 1:
                ax.legend()
            ax.grid()

        plt.show()

    def autoalign_full(self):
        pass


def main():

    parser = argparse.ArgumentParser(description="Autoalign Heimdallr beams.")
    parser.add_argument(
        "--shutter_pause_time",
        type=float,
        default=2.5,
        help="Seconds to pause after shuttering (default: 2.5)",
    )
    parser.add_argument(
        "-a",
        "--align",
        type=str,
        required=True,
        choices=[
            "cp",
            "coarseparallel",
            "ia",
            "imageall",
            "p3",
            "pupil3",
            "pa",
            "pupilall",
        ],
        help="Alignment method: 'ia'/'imageall', 'pa'/'pupilall' or 'p3'/'pupil3'",
    )
    parser.add_argument(
        "-b",
        "--band",
        type=str,
        default="K2",
        choices=["K1", "K2"],
        help="Band to use: 'K1' or 'K2' (default: K2)",
    )
    parser.add_argument(
        "-s",
        "--save_path",
        type=str,
        default="None",
        help="Path to save the alignment results (default: current directory)",
    )

    parser.add_argument(
        "-p",
        "--plot",
        type=bool,
        default=False,
        help="if results should be plotted to the screen when done (only valid for pa)",
    )
    args = parser.parse_args()

    heimdallr_aa = HeimdallrAA(
        shutter_pause_time=args.shutter_pause_time,
        band=args.band,
        flux_threshold=200.0,
        savepth=args.save_path,
    )

    init_vals = heimdallr_aa.get_init_motors_state()

    try:
        if args.align in ["cp", "coarseparallel"]:
            print("New command name, rerun with -a ia or -a imageall instead")
        elif args.align in ["ia", "imageall"]:
            heimdallr_aa.autoalign_coarse_parallel()
        elif args.align in ["p3", "pupil3"]:
            heimdallr_aa.autoalign_pupil(3, plot=args.plot)
            # open all shutters
            heimdallr_aa.open_all_shutters()
        elif args.align in ["pa", "pupilall"]:
            heimdallr_aa.autoalign_pupil_all()
        else:
            raise ValueError("Unknown alignment method.")

        print("Autoalignment completed.")
    except Exception as e:
        print(f"Error during alignment: {e}")
        print("Restoring initial motor positions.")
        heimdallr_aa.set_complete_state(init_vals)
        print("Initial motor positions restored.")


if __name__ == "__main__":
    main()
