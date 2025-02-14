import tkinter as tk
import tkinter.ttk as ttk
from PIL import Image, ImageTk
import asgard_alignment.Cameras
import time
import math
import numpy as np


def mock_wfs_signal(img):
    n_senses = 6
    # generate random in range -300 to 300
    # signals = np.random.randint(-300, 300, n_senses)
    signals = np.linspace(-300, 300, n_senses)

    return signals


class CameraApp:
    def __init__(self, root, camera, fps=15):
        self.root = root
        self.camera = camera
        self.camera.start_stream()
        self.fps = fps

        self.label = tk.Label(root)
        self.label.pack()

        control_frame = tk.Frame(root)
        control_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        self.gain_label = tk.Label(control_frame, text="Gain:")
        self.gain_label.grid(row=0, column=0, sticky=tk.W)
        self.gain_entry = tk.Entry(control_frame)
        self.gain_entry.grid(row=0, column=1, sticky=tk.W)
        self.gain_button = tk.Button(
            control_frame, text="Set Gain", command=self.set_gain
        )
        self.gain_button.grid(row=0, column=2, sticky=tk.W)

        self.exposure_label = tk.Label(control_frame, text="Exposure Time:")
        self.exposure_label.grid(row=1, column=0, sticky=tk.W)
        self.exposure_entry = tk.Entry(control_frame)
        self.exposure_entry.grid(row=1, column=1, sticky=tk.W)
        self.exposure_button = tk.Button(
            control_frame, text="Set Exposure", command=self.set_exposure
        )
        self.exposure_button.grid(row=1, column=2, sticky=tk.W)

        self.info_label = tk.Label(control_frame, text="")
        self.info_label.grid(row=2, column=0, columnspan=3, sticky=tk.W)

        self.save_frame_label = tk.Label(control_frame, text="Filename:")
        self.save_frame_label.grid(row=3, column=0, sticky=tk.W)
        self.save_frame_entry = tk.Entry(control_frame)
        self.save_frame_entry.grid(row=3, column=1, sticky=tk.W)

        self.num_frames_label = tk.Label(control_frame, text="Number of Frames:")
        self.num_frames_label.grid(row=4, column=0, sticky=tk.W)
        self.num_frames_entry = tk.Entry(control_frame)
        self.num_frames_entry.grid(row=4, column=1, sticky=tk.W)

        self.save_button = tk.Button(
            control_frame, text="Save Frames", command=self.save_frames
        )
        self.save_button.grid(row=5, column=0, columnspan=3, sticky=tk.W)

        stats_frame = tk.Frame(root)
        stats_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)

        self.percentile_labels = ["0th", "1st", "5th", "95th", "99th", "100th"]
        self.percentile_bars = []
        self.percentile_values = []

        for i, label in enumerate(self.percentile_labels):
            row_frame = tk.Frame(stats_frame)
            row_frame.pack(fill=tk.X)

            label_widget = tk.Label(row_frame, text=f"{label:>5}%:")
            label_widget.pack(side=tk.LEFT)

            bar = ttk.Progressbar(
                row_frame, orient="horizontal", length=200, mode="determinate"
            )
            bar.pack(side=tk.LEFT, padx=5)
            self.percentile_bars.append(bar)

            value_label = tk.Label(row_frame, text="0")
            value_label.pack(side=tk.LEFT)
            self.percentile_values.append(value_label)

        separator = ttk.Separator(stats_frame, orient="horizontal")
        separator.pack(fill=tk.X, pady=10)

        self.extra_bars = []
        for i in range(6):
            row_frame = tk.Frame(stats_frame)
            row_frame.pack(fill=tk.X)

            bar = ttk.Progressbar(
                row_frame,
                orient="horizontal",
                length=200,
                mode="determinate",
                maximum=600,
            )
            bar.pack(side=tk.LEFT, padx=5)
            self.extra_bars.append(bar)

            value_label = tk.Label(row_frame, text="0")
            value_label.pack(side=tk.LEFT)

        self.gain_min = math.floor(self.camera.get_node_min("Gain") * 100) / 100
        self.gain_max = math.floor(self.camera.get_node_max("Gain") * 100) / 100
        self.exposure_min = int(self.camera.get_node_min("ExposureTime"))
        self.exposure_max = int(self.camera.get_node_max("ExposureTime"))
        self.update_info()

        self.update_frame()

    def update_frame(self):
        img = self.camera.get_frame()
        img = Image.fromarray(img)
        imgtk = ImageTk.PhotoImage(image=img)
        self.label.imgtk = imgtk
        self.label.configure(image=imgtk)

        img_array = np.array(img)
        percentiles = np.percentile(img_array, [0, 1, 5, 95, 99, 100])

        for i, percentile in enumerate(percentiles):
            self.percentile_bars[i]["value"] = percentile
            self.percentile_values[i].config(text=f"{percentile:.2f}")

        wfs_signals = mock_wfs_signal(img_array)
        for i, signal in enumerate(wfs_signals):
            self.extra_bars[i]["value"] = signal + 300  # Normalize to 0-600 range

        self.root.after(int(1000 / self.fps), self.update_frame)  # Update based on FPS

    def set_gain(self):
        try:
            gain_value = float(self.gain_entry.get())
            if self.gain_min <= gain_value <= self.gain_max:
                self.camera["Gain"] = gain_value
                self.update_info()
            else:
                print(
                    f"Gain value out of range. Must be between {self.gain_min} and {self.gain_max}."
                )
        except ValueError:
            print("Invalid gain value. Please enter a number.")

    def set_exposure(self):
        try:
            exposure_value = float(self.exposure_entry.get())
            if self.exposure_min <= exposure_value <= self.exposure_max:
                self.camera["ExposureTime"] = exposure_value
                self.update_info()
            else:
                print(
                    f"Exposure time out of range. Must be between {self.exposure_min} and {self.exposure_max}."
                )
        except ValueError:
            print("Invalid exposure time. Please enter a number.")

    def save_frames(self):
        filename = self.save_frame_entry.get()
        try:
            num_frames = int(self.num_frames_entry.get())
            if num_frames <= 0:
                raise ValueError("Number of frames must be positive.")
        except ValueError:
            print("Invalid number of frames. Please enter a positive integer.")
            return

        frames = []
        for _ in range(num_frames):
            frames.append(self.camera.get_frame())

        gain = self.camera["Gain"]
        exposure_time = self.camera["ExposureTime"]

        np.savez(
            filename, images=np.array(frames), gain=gain, exposure_time=exposure_time
        )
        print(f"Saved {num_frames} frames to {filename}.npz")

    def update_info(self):
        current_gain = math.floor(self.camera["Gain"] * 100) / 100
        current_exposure = int(self.camera["ExposureTime"])
        self.info_label.config(
            text=f"Current Gain: {current_gain}, Min: {self.gain_min}, Max: {self.gain_max}\n"
            f"Current Exposure: {current_exposure}, Min: {self.exposure_min}, Max: {self.exposure_max}"
        )

    def __del__(self):
        self.camera.stop_stream()


if __name__ == "__main__":
    cam = asgard_alignment.Cameras.PointGrey(None, "14432631")

    root = tk.Tk()
    app = CameraApp(root, cam, fps=15)

    root.mainloop()
