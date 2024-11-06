import numpy as np
import scipy.special
import scipy.ndimage
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
    # default="gauss_diff",
    # default="naive",
    default="smoothed",
    help="The method to use for finding the maximum value, one of naive, smoothed, gauss_diff",
    choices=["naive", "smoothed", "gauss_diff"],
)

args = parser.parse_args()


def naive_max_find(img):
    max_loc = np.unravel_index(np.argmax(img), img.shape)
    return max_loc


def max_find_smoothed(img, scale, downsample=2):
    img = img[::downsample, ::downsample]
    scale /= downsample

    img_smooth = scipy.ndimage.gaussian_filter(img, scale)
    max_loc = np.unravel_index(np.argmax(img_smooth), img_smooth.shape)

    max_loc = np.array(max_loc) * downsample
    return max_loc


def max_find_gauss_diff(img, scale, downsample=2):
    # downsample the image
    img = img[::downsample, ::downsample]
    scale /= downsample

    # inp_img = np.fft.fft2(img.astype(float))
    # img_smooth = np.fft.ifft2(scipy.ndimage.fourier_gaussian(inp_img, scale / 2**0.25))
    # img_smooth2 = np.fft.ifft2(scipy.ndimage.fourier_gaussian(inp_img, scale * 2**0.25))

    img_smooth = scipy.ndimage.gaussian_filter(img.astype(float), scale / 2**0.25)
    img_smooth2 = scipy.ndimage.gaussian_filter(img.astype(float), scale * 2**0.25)
    img_diff = img_smooth - img_smooth2
    max_loc = np.unravel_index(np.argmax(img_diff), img_diff.shape)

    # upsample the location
    max_loc = np.array(max_loc) * downsample
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

airy_max = np.max(airy_2D)

# normalise and uint8
display_airy_2D = (airy_2D / np.max(airy_2D) * 255).astype(np.uint8)

img_shape = (1536, 2048)

xx, yy = np.meshgrid(np.arange(img_shape[0]), np.arange(img_shape[1]))


def compute_strehl(img):

    max_loc = max_find_method(img)

    xc, yc = max_loc
    ext = width
    img_psf_region = img[xc - ext : xc + ext, yc - ext : yc + ext]
    # convert to float
    img_psf_region = img_psf_region.astype(float)

    # subtract background
    # img_psf_region -= np.median(img)

    # bkg_width = ext
    # dist = np.hypot(xx - xc, yy - yc)
    # mask = np.logical_and(
    #     ext < dist,
    #     dist < (ext + bkg_width),
    # )
    # mask = mask.T

    bkg_width = 2 * ext
    mask = np.zeros_like(xx.T, dtype=bool)
    mask[xc - bkg_width : xc + bkg_width, yc - bkg_width : yc + bkg_width] = True
    mask[xc - ext : xc + ext, yc - ext : yc + ext] = False

    masked_img = img[mask]

    ad = np.abs(masked_img - np.median(masked_img))
    mad = np.median(ad)
    if np.isclose(mad, 0):
        mad = 1 / 6.0
    # print(f"mad: {mad}")

    # filter at 5 sigma
    valid_bkg = masked_img[ad <= 6 * mad]

    # print(f"bkg mean: {np.mean(valid_bkg)}")

    img_psf_region -= np.mean(valid_bkg)

    return np.max(img_psf_region) / np.sum(img_psf_region) / airy_max


if __name__ == "__main__":
    from PIL import Image
    from skimage.util import random_noise

    pth = "data/lab_imgs/beam_4_f400_laser_top_level_nd3.png"
    img = np.array(Image.open(pth))

    print(img.shape)

    noisy_img = random_noise(img, mode="s&p", amount=0.04)
    noisy_img = (noisy_img * 255).astype(np.uint8)

    print(f"Strehl is {compute_strehl(img):.3f}")
    print(f"Strehl (noisy img) is {compute_strehl(noisy_img):.3f}")

    import timeit

    execution_time = timeit.timeit(lambda: compute_strehl(img), number=10) / 10

    print(f"compute_strehl(img) took {execution_time} seconds")
    print(f"Equivelant FPS: {1/execution_time}Hz")
