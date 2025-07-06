# %%

import numpy as np
import matplotlib.pyplot as plt
import os

# %%

base = ["data", "stepper_vs_delay_line"]
fname = f"stepper_vs_linbo_2025-07-05 21:17:13.npz"
savepth = os.path.join(*base, fname)

data = np.load(savepth)

print(list(data.keys()))

v2s = data["stats"]
step_vals = data["stepper_vals"]
dl_vals = data["dl_vals"]

print(v2s.shape)