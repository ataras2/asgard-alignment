import numpy as np
import matplotlib.pyplot as plt
import os

pth = "data/may/heimdallr_34_run1"
# ims = np.load(os.path.join(pth, "img_stack.npy")).astype(float)
# positions = np.load(os.path.join(pth, "positions.npy"))
data = np.load(os.path.join(pth, "img_stack.npz"))
ims = data["img_stack"]
positions = data["positions"]
imshape = ims.shape
nsets = imshape[0]
ims_per_set = imshape[1]

print(ims.shape)


ims = ims.reshape((nsets * ims_per_set, imshape[2], imshape[3]))

print(ims.shape)
# crop:
# ims = ims[:, 600:900, 150:400]

mnsq = np.zeros((nsets * ims_per_set))

im_av = np.zeros_like(ims)
print()
for i in range(nsets * ims_per_set):
    print(f"Processing image {i} of {nsets * ims_per_set}\r", end="")
    im_av[i] = np.mean(
        ims[np.maximum(i - 20, 0) : np.minimum(i + 20, nsets * ims_per_set)], axis=0
    )
    mnsq[i] = np.sum((ims[i] - im_av[i]) ** 2)

mnsq = np.reshape(mnsq, (nsets, ims_per_set))


plt.clf()
plt.plot(positions, np.sum(mnsq, axis=1))
plt.show()

import pdb

pdb.set_trace()
