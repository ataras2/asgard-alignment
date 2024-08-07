import numpy as np
import matplotlib.pyplot as plt
import scipy.ndimage as nd
import os

plt.ion()

pth = "data/mid_july/heimdallr_34_run0_white_light"

# ims = np.load(os.path.join(pth, "img_stack.npy")).astype(float)
# positions = np.load(os.path.join(pth, "positions.npy"))

# data = np.load(os.path.join(pth, "img_stack.npz"))
data = np.load(os.path.join(pth, "img_stack.npz"))

pswidth = 24

# Crop image to this size before FFTs. It should be a product of small prime numbers, e.g.
# a multiple of 2.
ycrop = 448
xcrop = 512
# ------------ Finish user modified parameters --------------
ims = data["img_stack"]
positions = data["positions"]
imshape = ims.shape
nsets = imshape[0]
ims_per_set = imshape[1]

# crop
# ims = ims[:, :, 460:770, 850:1250]

print(ims.shape)

imshape = ims.shape


ims = ims.reshape((nsets * ims_per_set, imshape[2], imshape[3]))[:, :ycrop, :xcrop]

print(ims.shape)
# crop:
# ims = ims[:, 600:900, 150:400]


max_pwr = np.zeros((nsets * ims_per_set))

im_av = np.zeros_like(ims)
print()
for i in range(nsets * ims_per_set):
    print(f"Processing image {i} of {nsets * ims_per_set}\r", end="")
    imps = np.abs(np.fft.rfft2(ims[i])) ** 2
    imps[:pswidth, :pswidth] = 0
    imps[-pswidth:, :pswidth] = 0
    imps_smoothed = nd.gaussian_filter(imps, 5)
    max_pwr[i] = np.max(imps_smoothed)

max_pwr = np.reshape(max_pwr, (nsets, ims_per_set))


plt.figure(1)
plt.plot(positions, np.sum(max_pwr, axis=1))

import pickle

pickle.dump(
    plt.gcf(), open("heimdallr_34_run0_white_light.pickle", "wb")
)  # This is for Python 3 - py2 may need `file` instead of `open`

print("max power at: {:d}".format(np.argmax(max_pwr.flatten())))

plt.figure(2)
plt.imshow(ims[np.argmax(max_pwr)])


plt.show()


bias = np.median(max_pwr)
mad = np.median(np.abs(max_pwr - bias))
snr = (np.max(max_pwr) - bias) / (1.4826 * mad)

print(f"Maximum signal to noise ratio: {snr}")
