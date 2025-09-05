# %%
import numpy as np
import matplotlib.pyplot as plt


# %%

pth = "../data/pupil_moves"
fname = f"heimdallr_pupil_beam{1}.npz"

data = np.load(f"{pth}/{fname}")
list(data.keys())
# %%
positons = [data[f"meas_locs_{i}"] for i in ["x", "y"]]
fluxes = [data[f"fluxes_{i}"] for i in ["x", "y"]]
optimal_offsets = [data[f"optimal_offset_{i}"] for i in ["x", "y"]]
# %%
plt.figure()
plt.plot(positons[0], fluxes[0], "o-", label="x", color="C0")
plt.plot(positons[1], fluxes[1], "o-", label="y", color="C1")
for i in range(2):
    plt.axvline(optimal_offsets[i], color=f"C{i}", ls="--")
plt.xlabel("Position")
plt.ylabel("Flux")
plt.legend()
plt.show()

# %%
# repeat for all beams in subplots (2,2 grid)

n_beams = 4
fig, axs = plt.subplots(2, 2, figsize=(10, 8), sharex=True, sharey=True)

for beam in range(1, n_beams + 1):
    data = np.load(f"{pth}/heimdallr_pupil_beam{beam}.npz")
    positons = [data[f"meas_locs_{i}"] for i in ["x", "y"]]
    fluxes = [data[f"fluxes_{i}"] for i in ["x", "y"]]
    optimal_offsets = [data[f"optimal_offset_{i}"] for i in ["x", "y"]]
    ax = axs.flat[beam - 1]
    ax.plot(positons[0], fluxes[0], "o-", label="x", color="C0")
    ax.plot(positons[1], fluxes[1], "o-", label="y", color="C1")
    for i in range(2):
        ax.axvline(optimal_offsets[i], color=f"C{i}", ls="--")
    ax.set_title(f"Beam {beam}")
    if beam in [3, 4]:
        ax.set_xlabel("Position")
    if beam in [1, 3]:
        ax.set_ylabel("Flux")
    if beam == 1:
        ax.legend()
    ax.grid()

# %%
