# %%
import numpy as np
import matplotlib.pyplot as plt

# %%
fps = 500
signal_freq = 100
signal_2_freq = 5.4

t = np.arange(0, 20, 1 / fps)

signal = 100 * np.sin(2 * np.pi * signal_freq * t) + 20 * np.sin(
    2 * np.pi * signal_2_freq * t
)

# add a little noise
signal += np.random.normal(0, 0.1, len(t))

plt.plot(t, signal)
# %%
ps = np.abs(np.fft.rfft(signal)) ** 2
freqs = np.fft.rfftfreq(len(signal), 1 / fps)

plt.plot(freqs, ps)
# plt.plot(freqs, np.sqrt(ps) / len(signal) * np.sqrt(2))

plt.yscale("log")

print(f"peak value is at {freqs[np.argmax(ps)]} Hz, value: {np.max(ps)}")
plt.grid()
# %%
# convert power to an rms at that frequency
rms = np.sqrt(np.max(ps)) / len(signal) * np.sqrt(2)
print(f"rms at {freqs[np.argmax(ps)]} Hz is {rms}")
# %%
