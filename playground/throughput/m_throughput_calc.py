import numpy as np
import matplotlib.pyplot as plt


# source properties
def spectral_power(wavel):
    if wavel < 1000:
        return np.exp(-((wavel - 1000) ** 2) / (2 * 300**2))
    else:
        # line between (1000,1) and (2600,0)
        return 1 - (wavel - 1000) / 1600


wavels = np.linspace(200, 2600, 1000)

powers = np.vectorize(spectral_power)(wavels)

plt.plot(wavels, powers)
# grid on
plt.grid()


