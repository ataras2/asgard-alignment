# %%
import numpy as np
import matplotlib.pyplot as plt
import astropy.units as u


# %%
def nglass(l, glass="sio2"):
    """Refractive index of fused silica and other glasses. Note that C is
    in microns^{-2}

    Parameters
    ----------
    l: wavelength
    """
    l = l.to(u.micron).value  # convert to microns
    try:
        nl = len(l)
    except:
        l = [l]
        nl = 1
    l = np.array(l)
    if glass == "sio2":
        B = np.array([0.696166300, 0.407942600, 0.897479400])
        C = np.array([4.67914826e-3, 1.35120631e-2, 97.9340025])
    elif glass.lower() == "linbo3_ne":
        B = np.array([2.9804, 0.5981, 8.9543])
        C = np.array([0.02047, 0.0666, 416.08])
    elif glass.lower() == "linbo3_no":
        B = np.array([2.6734, 1.2290, 12.614])
        C = np.array([0.01764, 0.05914, 474.60])
    else:
        print("ERROR: Unknown glass {0:s}".format(glass))
        raise UserWarning
    n = np.ones(nl)
    for i in range(len(B)):
        n += B[i] * l**2 / (l**2 - C[i])
    return np.sqrt(n)


# %%
wavels = np.linspace(2.0, 2.5, 1000) * u.um  # in microns

plt.figure(figsize=(8, 6))
plt.plot(wavels, nglass(wavels, "linbo3_ne"), label="LinBO3 Ne")
plt.plot(wavels, nglass(wavels, "linbo3_no"), label="LinBO3 No")
plt.legend()
plt.xlabel("Wavelength (microns)")
plt.ylabel("Refractive Index")
plt.title("Refractive Index of LinBO3")
plt.grid()
plt.show()


# %%
# how much does the opd change as a function of wavelength and angle of the glass plate?
def opd(wavelength, theta, thickness, glass):
    """
    Calculate the optical path difference (OPD) for a glass plate at a given wavelength and angle.

    Parameters
    ----------
    wavelength : float
        Wavelength of light in microns.
    theta : float
        Angle of incidence in degrees.
    thickness : float
        Thickness of the glass plate in microns.
    glass : str
        Type of glass (e.g., "linbo3_ne", "linbo3_no").

    Returns
    -------
    float
        Optical path difference in microns.
    """
    n = nglass(wavelength, glass)  # refractive index

    r = np.arcsin(np.sin(theta) / n)
    t = thickness
    a = t / np.cos(r)

    dely = a * np.sin(theta - r)

    opd = n * a + dely * np.tan(theta) - n * t - t * (1 / np.cos(theta) - 1)

    return opd


theta_vals = np.linspace(0, 20, 100) * u.deg  # angle of incidence in degrees
wavelength = 2.2 * u.um  # wavelength in microns
thickness = 3.0 * u.mm  # thickness of the glass plate in microns

opd_vals_no = np.array(
    [opd(wavelength, theta, thickness, "linbo3_no") for theta in theta_vals]
)

plt.plot(theta_vals, opd_vals_no, label="LinBO3 n_o")
opd_vals_ne = np.array(
    [opd(wavelength, theta, thickness, "linbo3_ne") for theta in theta_vals]
)
plt.plot(theta_vals, opd_vals_ne, label="LinBO3 n_e")

plt.show()


# %%
from scipy.optimize import fsolve


def pol_opd(wavelength, theta, thickness, pol):
    """
    Calculate the optical path difference (OPD) for a glass plate at a given wavelength and angle.

    Parameters
    ----------
    wavelength : float
        Wavelength of light in microns.
    theta : float
        Angle of incidence in degrees.
    thickness : float
        Thickness of the glass plate in microns.
    glass : str
        Type of glass (e.g., "linbo3_ne", "linbo3").

    Returns
    -------
    float
        Optical path difference in microns.
    """
    assert pol in ["H", "V"], "Polarization must be 'H' or 'V'"
    n_o = nglass(wavelength, "linbo3_no")  # refractive index
    n_e = nglass(wavelength, "linbo3_ne")  # refractive index

    if pol == "V":
        n = n_o
        r = np.arcsin(np.sin(theta) / n)
    elif pol == "H":
        # numerical solve needed
        def fn(r):
            return np.sin(theta) - np.sqrt(
                n_e**2 * np.cos(r) ** 2 + n_o**2 * np.sin(r) ** 2
            ) * np.sin(r)

        r = fsolve(fn, np.arcsin(np.sin(theta.to(u.rad).value) / n_o))[0] * u.rad
        n = np.sqrt(n_e**2 * np.cos(r) ** 2 + n_o**2 * np.sin(r) ** 2)

    t = thickness
    a = t / np.cos(r)

    dely = a * np.sin(theta - r)

    opd = n * a + dely * np.tan(theta) - n * t - t * (1 / np.cos(theta) - 1)

    return opd


theta_vals = np.linspace(0, 20, 100) * u.deg  # angle of incidence in degrees
wavelength = 2.2 * u.um  # wavelength in microns
thickness = 3.0 * u.mm  # thickness of the glass plate in microns

opd_vals_v = (
    np.array([pol_opd(wavelength, theta, thickness, "V") for theta in theta_vals])
    * pol_opd(wavelength, 0 * u.deg, thickness, "V").unit
)
opd_vals_h = (
    np.array([pol_opd(wavelength, theta, thickness, "H") for theta in theta_vals])
    * pol_opd(wavelength, 0 * u.deg, thickness, "H").unit
)

plt.plot(theta_vals, opd_vals_v, label="LinBO3 V")
plt.plot(theta_vals, opd_vals_h, label="LinBO3 H")
plt.xlabel(f"Angle of Incidence ({theta_vals.unit})")
plt.ylabel(f"Optical Path Difference ({opd_vals_v.unit})")
plt.legend()

plt.figure()
opd_diff = opd_vals_v - opd_vals_h
opd_diff_rad = opd_diff.to("um")
plt.plot(theta_vals, opd_diff_rad)
plt.xlabel(f"Angle of Incidence ({theta_vals.unit})")
plt.ylabel(f"OPD Difference between pols ({opd_diff_rad.unit})")


del_lambda = 200 * u.nm
coherence_length = wavelength**2 / del_lambda
print(f"Coherence length: {coherence_length.to('um'):.2f}")
