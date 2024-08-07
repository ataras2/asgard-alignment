import cv2
import numpy as np
import PySpin
import tkinter as tk
from PIL import Image, ImageTk


wvl = 0.635e-6  # laser wavelength
D = 12e-3  # mm
# f = 254e-3 #mm
f = 400e-3  # mm
# N = f/D #
pixel_scale = 3.45e-6  # on point grey


def compute_strehl(img):
    xc, yc = np.unravel_index(np.argmax(img), img.shape)

    ext = 35
    img_psf_region = img[xc - ext : xc + ext, yc - ext : yc + ext]

    pointgrey_grid_x_um = np.linspace(
        -pixel_scale * img_psf_region.shape[0] / 2,
        pixel_scale * img_psf_region.shape[0] / 2,
        img_psf_region.shape[0],
    )
    pointgrey_grid_y_um = np.linspace(
        -pixel_scale * img_psf_region.shape[0] / 2,
        pixel_scale * img_psf_region.shape[1] / 2,
        img_psf_region.shape[1],
    )

    x, y = np.meshgrid(pointgrey_grid_x_um, pointgrey_grid_y_um)

    theta_x = x / f  # np.linspace(-3*wvl/D,3*wvl/D,1000)
    theta_y = y / f  # np.linspace(-3*wvl/D,3*wvl/D,1000)

    theta_r = (theta_x**2 + theta_y**2) ** 0.5

    airy_2D = (
        2
        * scipy.special.jv(1, 2 * np.pi / wvl * (D / 2) * np.sin(theta_r))
        / (2 * np.pi / wvl * (D / 2) * np.sin(theta_r))
    ) ** 2
    airy_2D *= 1 / np.sum(airy_2D)

    return np.max(reduced_psf_meas / np.sum(reduced_psf_meas)) / np.max(airy_2D)


class CameraStream:
    def __init__(self):
        self.system = PySpin.System.GetInstance()
        self.cam_list = self.system.GetCameras()
        self.cam = self.cam_list.GetByIndex(0)
        self.cam.Init()
        self.cam.AcquisitionMode.SetValue(PySpin.AcquisitionMode_Continuous)
        self.cam.BeginAcquisition()

    def get_frame(self):
        image_result = self.cam.GetNextImage()
        frame = np.array(image_result.GetData(), dtype="uint8").reshape(
            (image_result.GetHeight(), image_result.GetWidth())
        )
        image_result.Release()
        # Convert the frame to BGR color space
        frame = cv2.cvtColor(frame, cv2.COLOR_BAYER_BG2BGR)
        # Calculate total flux
        total_flux = np.sum(frame)
        # Put total flux on the frame
        position = (10, 30)  # Position at the top left corner
        font_scale = 0.7
        color = (255, 255, 255)  # White color
        thickness = 2
        cv2.putText(
            frame,
            f"Total Flux: {total_flux}",
            position,
            cv2.FONT_HERSHEY_SIMPLEX,
            font_scale,
            color,
            thickness,
        )

        return cv2.cvtColor(frame, cv2.COLOR_BAYER_BG2BGR)

    def release(self):
        self.cam.EndAcquisition()
        self.cam.DeInit()
        self.cam_list.Clear()
        self.system.ReleaseInstance()


class App:
    def __init__(self, root):
        self.root = root
        self.root.title("Live Camera Stream with Drawing Tools")

        self.camera = CameraStream()

        self.canvas = tk.Canvas(root)
        self.canvas.pack(fill=tk.BOTH, expand=True)

        self.btn_clear = tk.Button(root, text="Clear", command=self.clear_drawing)
        self.btn_clear.pack(side=tk.LEFT)

        self.points = []

        self.canvas.bind("<Button-1>", self.on_canvas_click)
        self.root.bind("<Configure>", self.on_resize)

        self.update_stream()

    def update_stream(self):
        frame = self.camera.get_frame()
        frame = cv2.resize(frame, (frame.shape[1] // 2, frame.shape[0] // 2))

        for i, point in enumerate(self.points):
            cv2.circle(frame, point, 2, (0, 255, 0), -1)
            # cv2.circle(frame, point, 5, (0, 255, 0), -1)

        points = np.array(self.points)
        for i in range(0, len(points) // 2):
            # draw the centre of each pair of points
            cv2.circle(
                frame,
                ((points[2 * i] + points[2 * i + 1]) / 2).round().astype(int),
                2,
                (0, 0, 255),
                -1,
            )

        self.display_image(frame)
        self.root.after(10, self.update_stream)

    def display_image(self, frame):
        self.canvas.delete("all")
        self.im = ImageTk.PhotoImage(
            image=Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        )
        self.canvas.create_image(0, 0, anchor=tk.NW, image=self.im)
        self.canvas.config(width=frame.shape[1], height=frame.shape[0])

    def on_canvas_click(self, event):
        self.points.append((event.x, event.y))

    def on_resize(self, event):
        self.canvas.config(width=event.width, height=event.height)

    def clear_drawing(self):
        self.points = []

    def on_close(self):
        self.camera.release()
        self.root.destroy()


if __name__ == "__main__":
    root = tk.Tk()
    app = App(root)
    root.protocol("WM_DELETE_WINDOW", app.on_close)
    root.mainloop()
