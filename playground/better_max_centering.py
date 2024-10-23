# %%

import numpy as np
import matplotlib.pyplot as plt
import PIL
import scipy.ndimage
import os

# %%
# pth = "../../../Desktop/18oct_OAP1"
# fname = "beam_2_in_focus.png"


pth = ""
fname = "hard_case2.png"

print(os.path.exists(f"{pth}{fname}"))

img = np.array(PIL.Image.open(f"{pth}{fname}"))

print(f"img.shape: {img.shape}, max: {np.max(img)}")

# add a random bad pixel
img[300, 300] = 255
# %%

maxloc = np.unravel_index(np.argmax(img), img.shape)

plt.figure()
plt.imshow(img)
plt.plot(maxloc[1], maxloc[0], "rx")
plt.show()

# %%

extent = 150
img_psf_region = img[
    maxloc[0] - extent : maxloc[0] + extent, maxloc[1] - extent : maxloc[1] + extent
]

# %%
plt.figure()
plt.imshow(img_psf_region)
plt.show()


# %%

# now to do the clever max finding
import scipy.ndimage

max_loc = np.array(
    scipy.ndimage.measurements.center_of_mass(img > np.percentile(img, 99)), dtype=int
)
print(f"max_loc: {max_loc}")

img_psf_region = img[
    max_loc[0] - extent : max_loc[0] + extent, max_loc[1] - extent : max_loc[1] + extent
]

plt.figure()
plt.imshow(img_psf_region)
plt.colorbar()
plt.show()


# %%
# attempt 2

# convolve with a gaussian
import scipy.ndimage
from scipy.ndimage import gaussian_filter

img_smooth = gaussian_filter(img, 10)

max_loc = np.unravel_index(np.argmax(img_smooth), img_smooth.shape)
print(f"max_loc: {max_loc}")

img_psf_region = img[
    max_loc[0] - extent : max_loc[0] + extent, max_loc[1] - extent : max_loc[1] + extent
]

# now find the max within this window

plt.figure()
plt.imshow(img_psf_region)
plt.colorbar()
plt.show()


# %%

# even more complciated - difference of gaussians

# init_smoothing_factor = 70/2**(1/4)
# img_smooth = gaussian_filter(img.astype(float), init_smoothing_factor)
# img_smooth2 = gaussian_filter(img.astype(float), init_smoothing_factor*np.sqrt(2))
# img_diff = img_smooth - img_smooth2
# max_loc = np.unravel_index(np.argmax(img_smooth_diff), img_smooth_diff.shape)

scale = 74.8
img_smoothed_1 = scipy.ndimage.gaussian_filter(img.astype, scale / 2**0.25)
img_smoothed_2 = scipy.ndimage.gaussian_filter(img, scale * 2**0.25)
img_diff = img_smoothed_1 - img_smoothed_2
max_loc = np.unravel_index(np.argmax(np.abs(img_diff)), img_diff.shape)


print(f"max_loc: {max_loc}")


plt.figure()
plt.imshow(img_diff)
plt.plot(max_loc[1], max_loc[0], "rx")
plt.colorbar()
plt.show()

plt.figure()
plt.hist(img_diff.flatten(), bins=20)
plt.yscale("log")
plt.show()

img_psf_region = img[
    max_loc[0] - extent : max_loc[0] + extent, max_loc[1] - extent : max_loc[1] + extent
]

plt.figure()
plt.imshow(img_psf_region)
plt.colorbar()
plt.show()
