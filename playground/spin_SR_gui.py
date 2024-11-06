import cv2
import numpy as np
import PySpin
import tkinter as tk
from tkinter import ttk
import scipy
from PIL import Image, ImageTk
import scipy.ndimage
from scipy.interpolate import RegularGridInterpolator
import argparse


parser = argparse.ArgumentParser(description="Strehl Ratio GUI")
# input arguments, mandatory: focal length, beam diameter
parser.add_argument(
    "--focal_length",
    type=float,
    default=254e-3,
    help="Focal length of the lens in meters",
)
parser.add_argument(
    "--beam_diameter",
    type=float,
    default=12e-3,
    help="Diameter of the beam in meters",
)
# optional: wavelength, pixel scale, spot size factor, method
parser.add_argument(
    "--wavelength",
    type=float,
    default=0.635e-6,
    help="Wavelength of the laser in meters",
)
parser.add_argument(
    "--pixel_scale",
    type=float,
    default=3.45e-6,
    help="Pixel scale of the camera in meters",
)
parser.add_argument(
    "--width_to_spot_size_ratio",
    type=float,
    default=2.0,
    help="The ratio of the width of the region of interest to the spot size",
)
parser.add_argument(
    "--method",
    type=str,
    default="gauss_diff",
    help="The method to use for finding the maximum value, one of naive, smoothed, gauss_diff",
    choices=["naive", "smoothed", "gauss_diff"],
)

args = parser.parse_args()


def naive_max_find(img):
    max_loc = np.unravel_index(np.argmax(img), img.shape)
    return max_loc


def max_find_smoothed(img, scale):
    img_smooth = scipy.ndimage.gaussian_filter(img, scale)
    max_loc = np.unravel_index(np.argmax(img_smooth), img_smooth.shape)
    return max_loc


def max_find_gauss_diff(img, scale):
    img_smooth = scipy.ndimage.gaussian_filter(img.astype(float), scale / 2**0.25)
    img_smooth2 = scipy.ndimage.gaussian_filter(img.astype(float), scale * 2**0.25)
    img_diff = img_smooth - img_smooth2
    max_loc = np.unravel_index(np.argmax(img_diff), img_diff.shape)
    return max_loc


# set the parameters
wvl = args.wavelength
D = args.beam_diameter
f = args.focal_length
pixel_scale = args.pixel_scale
spot_size = 2.44 * wvl / D * f / pixel_scale
width = int(args.width_to_spot_size_ratio * spot_size)

if args.method == "naive":
    max_find_method = naive_max_find
elif args.method == "smoothed":
    max_find_method = lambda img: max_find_smoothed(img, spot_size)
elif args.method == "gauss_diff":
    max_find_method = lambda img: max_find_gauss_diff(img, spot_size)


pointgrey_grid_x_um = np.linspace(
    -pixel_scale * width,
    pixel_scale * width,
    width * 2,
)
pointgrey_grid_y_um = np.linspace(
    -pixel_scale * width,
    pixel_scale * width,
    width * 2,
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

# normalise and uint8
display_airy_2D = (airy_2D / np.max(airy_2D) * 255).astype(np.uint8)


def compute_strehl(img):

    max_loc = max_find_method(img)

    xc, yc = max_loc
    ext = width
    img_psf_region = img[xc - ext : xc + ext, yc - ext : yc + ext]
    # convert to float
    img_psf_region = img_psf_region.astype(float)

    # subtract background
    # img_psf_region -= np.median(img)
    bkg_width = ext
    xx, yy = np.meshgrid(np.arange(img.shape[0]), np.arange(img.shape[1]))
    mask = np.logical_and(
        ext**2 < (xx - xc) ** 2 + (yy - yc) ** 2,
        (xx - xc) ** 2 + (yy - yc) ** 2 < (ext + bkg_width) ** 2,
    )
    mask = mask.T

    ad = np.abs(img[mask] - np.median(img[mask]))
    mad = np.median(ad)
    if np.isclose(mad, 0):
        mad = 1 / 6.0
    print(f"mad: {mad}")

    # filter at 5 sigma
    valid_bkg = img[mask][ad <= 6 * mad]

    print(f"bkg mean: {np.mean(valid_bkg)}")

    img_psf_region -= np.mean(valid_bkg)

    return np.max(img_psf_region) / np.sum(img_psf_region) / np.max(airy_2D)


class CameraStream:
    def __init__(self):
        self.system = PySpin.System.GetInstance()
        self.cam_list = self.system.GetCameras()
        self.cam = self.cam_list.GetByIndex(0)

        self.cam.Init()
        self.cam.UserSetSelector.SetValue(PySpin.UserSetSelector_Default)
        self.cam.UserSetLoad()
        self.cam.AcquisitionMode.SetValue(PySpin.AcquisitionMode_Continuous)
        self.cam.ExposureAuto.SetValue(PySpin.ExposureAuto_Off)
        self.cam.ExposureMode.SetValue(
            PySpin.ExposureMode_Timed
        )  # Timed or TriggerWidth (must comment out trigger parameters other that Line)

        self.cam.GainAuto.SetValue(PySpin.GainAuto_Off)
        self.cam.Gain.SetValue(0.00)
        self.cam.BeginAcquisition()

    def get_frame(self):
        image_result = self.cam.GetNextImage()
        frame = np.array(image_result.GetData(), dtype="uint8").reshape(
            (image_result.GetHeight(), image_result.GetWidth())
        )
        image_result.Release()

        return frame

    def set_exposure_time(self, new_time):
        self.cam.ExposureTime.SetValue(new_time)

    def get_exposure_time(self):
        try:
            return self.cam.ExposureTime.GetValue()
        except:
            return -1

    def release(self):
        self.cam.EndAcquisition()
        self.cam.DeInit()
        self.cam_list.Clear()
        self.system.ReleaseInstance()


class App:
    def __init__(self, root):
        self.root = root
        self.root.title("Strehl Ratio GUI")

        # Live Camera Stream
        self.camera_label = tk.Label(root)
        self.camera_label.grid(row=0, column=0)
        self.cap = CameraStream()

        # Zoomed-in Window
        self.zoomed_label = tk.Label(root)
        self.zoomed_label.grid(row=0, column=1)

        # Static Ideal Image
        self.ideal_label = tk.Label(root)
        self.ideal_label.grid(row=0, column=2)

        # Strehl Ratio Label
        self.strehl_label = tk.Label(root, text="Strehl Ratio (0 to 1)")
        self.strehl_label.grid(row=1, column=0, columnspan=2)

        # Strehl Ratio Value
        self.strehl_value_label = tk.Label(root, text="0.0")
        self.strehl_value_label.grid(row=1, column=2)

        # Strehl Ratio Bar
        self.strehl_bar = ttk.Progressbar(
            root, orient="horizontal", length=200, mode="determinate", maximum=1
        )
        self.strehl_bar.grid(row=2, column=0, columnspan=2)

        # Max Value Label
        self.max_value_label = tk.Label(root, text="Max Value (0 to 255)")
        self.max_value_label.grid(row=3, column=0, columnspan=2)

        # Max Value Value
        self.max_value_value_label = tk.Label(root, text="0")
        self.max_value_value_label.grid(row=3, column=2)

        # Max Value Bar
        self.max_value_bar = ttk.Progressbar(
            root, orient="horizontal", length=200, mode="determinate", maximum=255
        )
        self.max_value_bar.grid(row=4, column=0, columnspan=2)

        # Exposure Time Label
        self.exposure_label = tk.Label(root, text="Exposure Time:")
        self.exposure_label.grid(row=2, column=0)

        # Exposure Time Entry
        self.exposure_entry = tk.Entry(root)
        self.exposure_entry.grid(row=2, column=1)
        self.exposure_entry.bind("<Return>", self.update_exposure_time)

        # Current Exposure Time Label
        self.current_exposure_label = tk.Label(
            root, text=f"Current Exposure Time: {-2}"
        )
        self.current_exposure_label.grid(row=2, column=2)

        self.update()

    def update_exposure_time(self, event):
        try:
            new_exposure_time = float(self.exposure_entry.get())
            self.cap.set_exposure_time(new_exposure_time)
            self.current_exposure_label.config(
                text=f"Current Exposure Time: {self.cap.get_exposure_time()}"
            )
        except ValueError:
            print("Invalid exposure time entered")

    def update(self):
        scaling_factor = 0.2  # Adjust this factor to make the image smaller or larger
        frame = self.cap.get_frame()

        disp_img = cv2.resize(frame, (0, 0), fx=scaling_factor, fy=scaling_factor)
        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(disp_img)
        # convert to bgr
        disp_img = cv2.cvtColor(disp_img, cv2.COLOR_BGR2RGB)

        gray = frame

        # save the image to a file
        cv2.imwrite("hard_case2.png", gray)

        # print(gray.shape)
        # img_smooth = scipy.ndimage.gaussian_filter(gray, 10)
        # max_loc = np.unravel_index(np.argmax(img_smooth), img_smooth.shape)
        max_loc = max_find_method(gray)

        max_val = gray[max_loc[0], max_loc[1]]
        # swap around
        max_loc = (max_loc[1], max_loc[0])

        # Zoomed-in region
        zoomed_region = gray[
            max(0, max_loc[1] - width) : min(gray.shape[0], max_loc[1] + width),
            max(0, max_loc[0] - width) : min(gray.shape[1], max_loc[0] + width),
        ]
        try:
            strehl_ratio = compute_strehl(frame)
        except:
            strehl_ratio = -1.0
        self.strehl_bar["value"] = strehl_ratio
        self.strehl_value_label.config(text=f"{strehl_ratio:.2f}")

        zoomed_image = cv2.applyColorMap(zoomed_region, cv2.COLORMAP_VIRIDIS)[
            :, :, ::-1
        ]
        zoomed_image = Image.fromarray(zoomed_image)
        zoomed_image = zoomed_image.resize(
            (disp_img.shape[1], disp_img.shape[1]), Image.NEAREST
        )
        zoomed_image = ImageTk.PhotoImage(image=zoomed_image)

        # Update GUI elements
        self.zoomed_label.config(image=zoomed_image)
        self.zoomed_label.image = zoomed_image

        self.max_value_bar["value"] = max_val
        self.max_value_value_label.config(text=f"{max_val}")

        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        disp_img = cv2.applyColorMap(disp_img, cv2.COLORMAP_VIRIDIS)

        cv2.rectangle(
            disp_img,
            (
                int((max_loc[0] - width) * scaling_factor),
                int((max_loc[1] - width) * scaling_factor),
            ),
            (
                int((max_loc[0] + width) * scaling_factor),
                int((max_loc[1] + width) * scaling_factor),
            ),
            (255, 255, 255),
            2,
        )
        disp_img = disp_img[:, :, ::-1]

        img = ImageTk.PhotoImage(image=Image.fromarray(disp_img))
        self.camera_label.config(image=img)
        self.camera_label.image = img

        # Resize the airy disc image to match the frame size
        airy_image = Image.fromarray(
            cv2.applyColorMap(display_airy_2D, cv2.COLORMAP_VIRIDIS)[:, :, ::-1]
        )
        airy_image = airy_image.resize(
            (disp_img.shape[1], disp_img.shape[1]), Image.NEAREST
        )
        airy_image = ImageTk.PhotoImage(image=airy_image)
        self.ideal_label.config(image=airy_image)
        self.ideal_label.image = airy_image

        self.root.after(10, self.update)

    def on_closing(self):
        self.cap.release()
        self.root.destroy()


if __name__ == "__main__":
    root = tk.Tk()
    app = App(root)
    root.protocol("WM_DELETE_WINDOW", app.on_closing)
    root.mainloop()
