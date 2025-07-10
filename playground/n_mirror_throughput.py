# %%
import numpy as np
import matplotlib.pyplot as plt


def circle_mask(shape, center, radius):
    y, x = np.ogrid[: shape[0], : shape[1]]
    dist_from_center = np.sqrt((x - center[1]) ** 2 + (y - center[0]) ** 2)
    mask = dist_from_center <= radius
    return mask


# %%
w = 256
true_offset = 8
hole = circle_mask((w, w), (w // 2+true_offset, w // 2), w // 4)

blob_radius = 0.95 * w // 4

offsets = np.linspace(-0.3 * blob_radius, 0.3 * blob_radius, 7)
flux = np.zeros(len(offsets))

# plt.imshow(hole)

for i, offset in enumerate(offsets):
    blob = circle_mask((w, w), (w // 2 + offset, w // 2 + 1.5), blob_radius)
    # plt.imshow(blob, alpha=0.5)
    flux[i] = np.sum(blob & hole)

flux /= np.max(flux)  # normalise

noise = np.random.normal(0, 0.005, len(flux))
# flux += noise  # add some noise

offsets/= 100
true_offset /= 100

# curve fit - this is of the form
# y = -m|x-a| -m|x-b| + c
from scipy.optimize import curve_fit
import time
def fit_func(x, m, a, b, c):
    return -m * np.abs(x - a) - m * np.abs(x - b) + c
start = time.time()
n_loops = 1000
for _ in range(n_loops):
    params, _ = curve_fit(fit_func, offsets, flux, p0=[1, -0.05, 0.05, 1])
print(f"Curve fit took {(time.time() - start)/n_loops:.5f} seconds")

# estimate is (b + a) / 2
estimate_offset = (params[1] + params[2]) / 2

x_fit = np.linspace(offsets.min(), offsets.max(), 100)
y_fit = fit_func(x_fit, *params)
plt.plot(x_fit, y_fit, color='red', label='Fitted Curve')

plt.plot(offsets, flux, marker="o")
plt.axvline(true_offset, color='green', linestyle='--', label='True Offset')
plt.axvline(estimate_offset, color='orange', linestyle='--', label='Estimated Offset')

# %%
import asgard_alignment.Engineering as asgE

asgE.move_pup_calc("c_red_one_focus", 3, [0, 0.2/7])