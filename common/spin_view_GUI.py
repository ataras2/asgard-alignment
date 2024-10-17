import cv2
import numpy as np
import PySpin
import tkinter as tk
from PIL import Image, ImageTk

n_coadds = 5


class CameraStream:
    def __init__(self):
        self.system = PySpin.System.GetInstance()
        self.cam_list = self.system.GetCameras()
        self.cam = self.cam_list.GetByIndex(0)
        self.cam.Init()
        self.cam.AcquisitionMode.SetValue(PySpin.AcquisitionMode_Continuous)
        self.cam.BeginAcquisition()

    def get_frame(self):

        for i in range(n_coadds):
            image_result = self.cam.GetNextImage()
            frame = np.array(image_result.GetData(), dtype="uint8").reshape(
                (image_result.GetHeight(), image_result.GetWidth())
            )
            image_result.Release()
            if i == 0:
                frame_sum = frame.astype(np.float32)
            else:
                frame_sum += frame

        frame = (frame_sum / n_coadds).astype(np.uint8)
        return frame

    def release(self):
        self.cam.EndAcquisition()
        self.cam.DeInit()
        self.cam_list.Clear()
        self.system.ReleaseInstance()


class App:
    def __init__(self, root):
        self.root = root
        self.root.title("Live Camera Stream with Drawing Tools")

        # Set fullscreen
        # self.root.attributes("-fullscreen", True)
        # set to still be windowed, but maximised
        self.root.state("zoomed")

        self.camera = CameraStream()

        self.canvas_ref = tk.Canvas(root)
        self.canvas_ref.grid(row=0, column=0, sticky="nsew")

        self.canvas_live = tk.Canvas(root)
        self.canvas_live.grid(row=0, column=1, sticky="nsew")

        self.canvas_diff = tk.Canvas(root)
        self.canvas_diff.grid(row=0, column=2, sticky="nsew")

        self.diff_info_label = tk.Label(root, text="")
        self.diff_info_label.grid(row=1, column=2, sticky="ew")

        self.btn_frame = tk.Frame(root)
        self.btn_frame.grid(row=2, column=0, columnspan=3, sticky="ew")

        self.btn_capture = tk.Button(
            self.btn_frame, text="Capture Reference", command=self.capture_reference
        )
        self.btn_capture.pack(side=tk.LEFT)

        self.btn_clear = tk.Button(
            self.btn_frame, text="Clear", command=self.clear_drawing
        )
        self.btn_clear.pack(side=tk.LEFT)

        self.reference_frame = None
        self.ref_resized = None
        self.points = []

        frame = self.camera.get_frame()
        frame = cv2.cvtColor(frame, cv2.COLOR_BAYER_BG2BGR)
        # Resize the frame to have a width of 512 pixels
        self.resize_width = 512
        self.resize_height = int(frame.shape[0] * (self.resize_width / frame.shape[1]))

        self.canvas_live.bind("<Button-1>", self.on_canvas_click)
        self.root.bind("<Configure>", self.on_resize)

        self.update_stream()

    def capture_reference(self):
        self.reference_frame = self.camera.get_frame()
        # should display the reference frame in the left canvas, resize it to fit
        self.ref_resized = cv2.resize(
            self.reference_frame,
            (self.resize_width, self.resize_height),
        )
        frame = cv2.cvtColor(self.ref_resized, cv2.COLOR_BAYER_BG2BGR)
        self.display_image(frame, self.canvas_ref)

    def update_stream(self):
        frame = self.camera.get_frame()
        # Resize the frame to have a width of 512 pixels
        frame = cv2.resize(frame, (self.resize_width, self.resize_height))

        disp_frame = cv2.cvtColor(frame, cv2.COLOR_BAYER_BG2BGR)

        for i, point in enumerate(self.points):
            cv2.circle(disp_frame, point, 2, (0, 255, 0), -1)

        points = np.array(self.points)
        for i in range(0, len(points) // 2):
            cv2.circle(
                disp_frame,
                ((points[2 * i] + points[2 * i + 1]) / 2).round().astype(int),
                2,
                (0, 0, 255),
                -1,
            )
        self.display_image(disp_frame, self.canvas_live)

        if self.reference_frame is not None:
            # diff_frame = cv2.absdiff(ref_resized, frame)
            diff_frame = np.float32(frame) - np.float32(self.ref_resized)
            # diff_frame = cv2.normalize(diff_frame, None, 0, 255, cv2.NORM_MINMAX)

            min_val, max_val, _, _ = cv2.minMaxLoc(diff_frame)
            self.diff_info_label.config(text=f"Min: {min_val}, Max: {max_val}")

            # normalise it such that -30 is mapped to 0 and 30 is mapped to 255
            cut = 20
            diff_frame[diff_frame > cut] = cut
            diff_frame[diff_frame < 0] = 0

            diff_frame = cv2.normalize(
                diff_frame, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U
            )

            # apply viridis colourmap
            diff_frame = cv2.applyColorMap(diff_frame, cv2.COLORMAP_VIRIDIS)

            for i, point in enumerate(self.points):
                cv2.circle(diff_frame, point, 2, (0, 255, 0), -1)

            points = np.array(self.points)
            for i in range(0, len(points) // 2):
                cv2.circle(
                    diff_frame,
                    ((points[2 * i] + points[2 * i + 1]) / 2).round().astype(int),
                    2,
                    (0, 0, 255),
                    -1,
                )
            self.display_image(diff_frame, self.canvas_diff)

        self.root.after(10, self.update_stream)

    def display_image(self, frame, canvas):
        canvas.delete("all")
        im = ImageTk.PhotoImage(
            image=Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        )
        canvas.create_image(0, 0, anchor=tk.NW, image=im)
        canvas.image = im
        canvas.config(width=frame.shape[1], height=frame.shape[0])

    def on_canvas_click(self, event):
        self.points.append((event.x, event.y))

    def on_resize(self, event):
        self.canvas_live.config(
            width=event.width // 3, height=event.height - self.btn_frame.winfo_height()
        )
        self.canvas_ref.config(
            width=event.width // 3, height=event.height - self.btn_frame.winfo_height()
        )
        self.canvas_diff.config(
            width=event.width // 3, height=event.height - self.btn_frame.winfo_height()
        )

    def clear_drawing(self):
        self.points = []

    def on_close(self):
        self.camera.release()
        self.root.destroy()


if __name__ == "__main__":
    root = tk.Tk()
    root.grid_rowconfigure(0, weight=1)
    root.grid_columnconfigure(0, weight=1)
    root.grid_columnconfigure(1, weight=1)
    root.grid_columnconfigure(2, weight=1)
    app = App(root)
    root.protocol("WM_DELETE_WINDOW", app.on_close)
    root.mainloop()
