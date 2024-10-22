# %%
import numpy as np
import matplotlib.pyplot as plt
import os
import scipy

pth = "../data/241016/"


fnames = [
    "B3_DM_focus.raw",
    "B3_DM_focus_try2.raw",
]

fname = fnames[0]
# OAP1
D = 18e-3  # mm
f = 681.56e-3  # mm
ext = 128


# fnames = [
#     "B3_initial_focus.raw",
#     "B3_initial_focus_nomask.raw",
#     "B3_initial_focus_nomask_anu2.raw",
#     "B3_initial_focus_nomask_anu3.raw",
# ]
# # init focus (i.e. after spherical)
# D = 12e-3  # mm
# f = 2  # mm
# ext = 150

# for fname in fnames:
img = np.fromfile(os.path.join(pth, fname), dtype=np.uint8)
img = img.reshape(-1, 2048)
# %%
plt.figure()
plt.imshow(img)
plt.show()

# %%

max_loc = np.unravel_index(np.argmax(img), img.shape)
import scipy.ndimage

# max_loc = np.array(
#     scipy.ndimage.measurements.center_of_mass(img > np.percentile(img, 99)),
#     dtype=int,
# )

img_psf_region = img[
    max_loc[0] - ext : max_loc[0] + ext, max_loc[1] - ext : max_loc[1] + ext
]
# convert to float
img_psf_region = img_psf_region.astype(float)

# subtract background
# img_psf_region -= np.median(img)
psf_rect_mask = np.ones_like(img)
# mask around the maximum pixel in img
psf_rect_mask[
    max_loc[0] - ext : max_loc[0] + ext, max_loc[1] - ext : max_loc[1] + ext
] = 0

background = np.median(img)

img_psf_region -= background

plt.figure()
plt.imshow(img_psf_region)

# %%

wvl = 0.635e-6  # laser wavelength
pixel_scale = 3.45e-6  # on point grey

width = ext

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
    max_loc = np.unravel_index(np.argmax(img), img.shape)
    # max_loc = np.array(
    #     scipy.ndimage.measurements.center_of_mass(img > np.percentile(img, 99)),
    #     dtype=int,
    # )

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

    return np.max(img_psf_region) / np.sum(img_psf_region) / np.max(airy_2D), mask


# %%


strehl, mask = compute_strehl(img)
print(f"{fname} Strehl: {strehl:.3f}")

plt.savefig(os.path.join(pth, f"{fname}_psf.png"))

plt.figure()
plt.subplot(121)
plt.imshow(img)
plt.colorbar()
plt.subplot(122)
plt.imshow(img * mask)
plt.colorbar()
