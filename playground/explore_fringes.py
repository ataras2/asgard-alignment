# %%

import numpy as np
import matplotlib.pyplot as plt
import os

#%%

pth = "../data/Oct22/heim_13"

# Load image data
data = np.load(os.path.join(pth, "img_stack.npz"))


img = data["img_stack"][0, 0]