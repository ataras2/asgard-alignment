# %%
import os
import numpy as np
from astropy.io import fits
import matplotlib.pyplot as plt
import scipy.ndimage as ndi
# %%

pth = os.path.join("..", "data", "cube_13_08_38_hei_k1_ufan_off.fits")


# Read the FITS file
with fits.open(pth) as hdul:
    data = hdul[0].data  # This gets the primary HDU data
    header = hdul[0].header  # This gets the header if needed

# %%
img = data.mean(0)
plt.imshow(img)

# %%
bl_centres = np.array(
    [
        [10.5, 5.5],
    ]
)


fft = np.fft.fft2(np.fft.fftshift(img))
ps = np.abs(fft) ** 2
plt.subplot(121)
plt.imshow(np.log(np.abs(fft)))
plt.subplot(122)
plt.imshow(np.angle(fft), cmap="twilight")

# add bl centre crosses to the first subplot
plt.subplot(121)
for bl_centre in bl_centres:
    plt.plot(
        bl_centre[0],
        bl_centre[1],
        "rx",
    )


# %%
ps_radius = 2.6

def soft_circle_mask(shape, center, radius):
    radius += 1 / 2
    y, x = np.ogrid[: shape[0], : shape[1]]
    dist_from_center = (x - center[1]) ** 2 + (y - center[0]) ** 2
    mask = radius - np.sqrt((dist_from_center))
    mask = np.clip(mask, 0, 1)  # Ensure values are between 0 and 1
    return mask


mask = soft_circle_mask(img.shape, bl_centres[0,::-1], ps_radius)
plt.imshow(np.log10(ps))
plt.contour(mask, levels=[0.5], colors='r', linewidths=2)


# %%
beam_inverse = np.fft.fftshift(np.fft.ifft2(np.fft.ifftshift(mask * fft)))
beam = np.abs(beam_inverse)**2
plt.imshow(beam)

# show CoM
com = ndi.center_of_mass(beam)
plt.plot(com[1], com[0], "rx")