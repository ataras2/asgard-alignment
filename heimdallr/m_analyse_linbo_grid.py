# %%

import numpy as np
import matplotlib.pyplot as plt
import os

# %%

base = ["data", "stepper_vs_delay_line"]
fname = f"stepper_vs_linbo_13_2025-07-06 22:04:13.npz"
savepth = os.path.join(*base, fname)

data = np.load(savepth)

print(list(data.keys()))

v2s = data["stats"]
step_vals = data["stepper_vals"]
dl_vals = data["dl_vals"]

print(v2s.shape)

plt.imshow(v2s.mean(-2)[..., 0])
plt.colorbar()
plt.show()
