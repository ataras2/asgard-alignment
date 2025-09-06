# %%
import numpy as np
import matplotlib.pyplot as plt


# %%

pth = "../data/pupil_moves/a1"
fname = f"heimdallr_pupil_beam{1}.npz"

data = np.load(f"{pth}/{fname}")
list(data.keys())
# %%
positions = [data[f"meas_locs_{i}"] for i in ["x", "y","x"]]
fluxes = [data[f"fluxes_{i}"] for i in ["x", "y","x2"]]
optimal_offsets = [data[f"optimal_offset_{i}"] for i in ["x", "y","x2"]]
# %%
plt.figure()
plt.plot(positions[0], fluxes[0], "o-", label="x", color="C0")
plt.plot(positions[1], fluxes[1], "o-", label="y", color="C1")
plt.plot(positions[2], fluxes[2], "o-", label="x2", color="C2")
for i in range(3):
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
    positions = [data[f"meas_locs_{i}"] for i in ["x", "y","x"]]
    fluxes = [data[f"fluxes_{i}"] for i in ["x", "y","x2"]]
    optimal_offsets = [data[f"optimal_offset_{i}"] for i in ["x", "y","x2"]]
    ax = axs.flat[beam - 1]
    ax.plot(positions[0], fluxes[0], "o-", label="x", color="C0")
    ax.plot(positions[1], fluxes[1], "o-", label="y", color="C1")
    ax.plot(positions[2], fluxes[2], "o-", label="x2", color="C2")
    for i in range(3):
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
