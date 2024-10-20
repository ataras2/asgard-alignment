# %%

import numpy as np
import matplotlib.pyplot as plt
import PIL

# %%
# pth = "../../../Desktop/18oct_OAP1"
# fname = "beam_2_in_focus.png"


pth = "../"
fname = "hard_case.png"

img = np.array(PIL.Image.open(f"{pth}/{fname}"))

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

img_smooth = gaussian_filter(img, 10).astype(float)
img_smooth2 = gaussian_filter(img, 10*np.sqrt(2)).astype(float)

img_smooth_diff = img_smooth - img_smooth2

max_loc = np.unravel_index(np.argmax(img_smooth_diff), img_smooth_diff.shape)

print(f"max_loc: {max_loc}")


plt.figure()
plt.imshow(img_smooth_diff)
plt.plot(max_loc[1], max_loc[0], "rx")
plt.colorbar()
plt.show()

plt.figure()
plt.hist(img_smooth_diff.flatten(), bins=20)
plt.yscale("log")
plt.show()

img_psf_region = img[
    max_loc[0] - extent : max_loc[0] + extent, max_loc[1] - extent : max_loc[1] + extent
]

plt.figure()
plt.imshow(img_psf_region)
plt.colorbar()
plt.show()