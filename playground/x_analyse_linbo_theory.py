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


# %%
theta_vals = np.linspace(0, 40, 100) * u.deg  # angle of incidence in degrees
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

# %%
# how well does opd_vals_v look like a parabola?
# use np.polyfit to fit a parabola to the data
coeffs = np.polyfit(theta_vals.to(u.deg).value, opd_vals_v.to(u.um).value, 2)
pred = np.polyval(coeffs, theta_vals.to(u.deg).value) * u.um
plt.figure()
plt.plot(theta_vals, opd_vals_v.to("um"), label="LinBO3 V")
plt.plot(theta_vals, pred.to("um"), label="Parabola fit")
plt.xlabel(f"Angle of Incidence ({theta_vals.unit})")
plt.ylabel(f"Optical Path Difference ({opd_vals_v.unit})")
plt.legend()
plt.show()

plt.figure()
residuals = opd_vals_v.reshape(-1) - pred
plt.plot(theta_vals, residuals.to("um"))


# %%
# now the differnce between pols as a function of wavelength
theta_vals = np.linspace(0, 20, 100) * u.deg  # angle of incidence in degrees
wavels = np.linspace(2.0, 2.4, 5) * u.um  # wavelength in microns

opd_diff_vals = np.zeros((len(wavels), len(theta_vals))) * u.um
for i, wavelength in enumerate(wavels):
    opd_vals_v = (
        np.array([pol_opd(wavelength, theta, thickness, "V") for theta in theta_vals])
        * pol_opd(wavelength, 0 * u.deg, thickness, "V").unit
    )
    opd_vals_h = (
        np.array([pol_opd(wavelength, theta, thickness, "H") for theta in theta_vals])
        * pol_opd(wavelength, 0 * u.deg, thickness, "H").unit
    )
    opd_diff_vals[i] = (opd_vals_v - opd_vals_h).reshape(-1)

plt.figure(figsize=(10, 6))
for i, wavelength in enumerate(wavels):
    plt.plot(
        theta_vals,
        opd_diff_vals[i].to("um"),
        label=f"{wavelength.to(u.um):.2f}",
    )
plt.xlabel(f"Angle of Incidence ({theta_vals.unit})")
plt.ylabel(f"OPD Difference ({opd_diff_vals.unit})")
plt.legend()

plt.figure()
plt.imshow(
    opd_diff_vals.to("um").value,
    extent=[
        theta_vals[0].to(u.deg).value,
        theta_vals[-1].to(u.deg).value,
        wavels[0].to(u.um).value,
        wavels[-1].to(u.um).value,
    ],
    aspect="auto",
    origin="lower",
)
plt.colorbar(label="OPD Difference (um)")
plt.contour(
    theta_vals.to(u.deg).value,
    wavels.to(u.um).value,
    opd_diff_vals.to("um").value,
    levels=np.linspace(
        opd_diff_vals.to("um").value.min(), opd_diff_vals.to("um").value.max(), 10
    ),
    colors="black",
    linewidths=0.5,
    linestyles="dashed",
)
plt.xlabel("Angle of Incidence (deg)")
plt.ylabel("Wavelength (um)")
plt.title("OPD Difference between Polarizations")


# %%
def vis_given_opd(opd, wavel):
    return 0.5 * np.abs(np.exp(1j * 2 * np.pi * opd / wavel) + 1)


theta_vals = np.linspace(0, 40, 100) * u.deg  # angle of incidence in degrees

opd_offset = -2.5 * u.um
wavelength = 2.2 * u.um  # wavelength in microns
# wavels = [2.0, 2.2, 2.4] * u.um  # wavelength in microns
wavels = np.linspace(2.0, 2.4, 10) * u.um  # wavelength in microns

vis_totals = np.zeros(len(theta_vals))

for wavelength in wavels:
    opd_vals_h = (
        np.array([pol_opd(wavelength, theta, thickness, "H") for theta in theta_vals])
    ) * pol_opd(wavelength, 0 * u.deg, thickness, "H").unit
    opd_vals_v = (
        np.array([pol_opd(wavelength, theta, thickness, "V") for theta in theta_vals])
    ) * pol_opd(wavelength, 0 * u.deg, thickness, "V").unit

    opd_diff = opd_vals_v - opd_vals_h + opd_offset
    vis_vals = vis_given_opd(opd_diff, wavelength)

    plt.plot(theta_vals, vis_vals, label=f"{wavelength.to(u.um):.2f}")
    vis_totals += vis_vals[:,0]
    
plt.plot(theta_vals, vis_totals/len(wavels), 'k')

plt.xlabel(f"Angle of Incidence ({theta_vals.unit})")
plt.ylabel("Visibility")
plt.title(f"Vis for OPD offset of {opd_offset.to(u.um):.2f}")
plt.legend()

# %%
