"""
four part series to illustrate cold stop mis alignment and aliasing issues

0) visualize input phase aberration
1) image plane cartoon of cold stop with speckles
2) fourier (pupil plane) analsis of the effect of the filter with speckle at its cut off with/without misalignment
3) just a simple image of output after cold stop w/wo misalignment
4) visual illustration of the ZWFS images , decompose to even and odd modes 

"""

stop_misalignment = 0.5


#%%
################################################################
################################################################
################ Section 0 -  aberration input 
################################################################
################################################################
import numpy as np

def fourier_phase_cpd(N=1025, cycles_per_diam=2.0, theta=0.0, amp_rad=0.30):
    """
    Phase aberration phi(x,y) = amp_rad * cos(pi*k * (x cosθ + y sinθ)),
    with k = cycles_per_diam (cycles / pupil diameter).
    Coordinates are on [-1,1] so the pupil diameter is 2 in these units.

    Parameters
    ----------
    N : int
        Grid size (use odd so the pupil is perfectly centered).
    cycles_per_diam : float
        Spatial frequency k in cycles per pupil *diameter*.
        e.g. k=2.0 puts the two speckles right at the edge of a 4 λ/D cold stop.
    theta : float
        Orientation [rad], 0 = along +x, +π/2 = along +y.
    amp_rad : float
        Target RMS phase (radians) *inside the pupil* after normalization.

    Returns
    -------
    phi : (N,N) float
        Phase map [rad].
    P   : (N,N) bool
        Circular pupil mask (unit radius).
    """
    # pupil on [-1,1]^2 (unit radius ⇒ diameter = 2)
    x = np.linspace(-1, 1, N)
    X, Y = np.meshgrid(x, x, indexing='xy')
    P = (X**2 + Y**2) <= 1.0

    # sinusoid argument: π * k * (x cosθ + y sinθ)
    # (Because length across the pupil is 2, so cycles/diam → multiply x by k/2 ⇒ π k x)
    s = np.cos(np.pi * cycles_per_diam * (X*np.cos(theta) + Y*np.sin(theta)))

    # zero-mean inside pupil, unit-RMS, then scale to amp_rad
    s -= s[P].mean()
    s /= np.sqrt((s[P]**2).mean() + 1e-30)

    phi = amp_rad * s * P
    return phi, P

phi, P = fourier_phase_cpd(N=513, cycles_per_diam=2.0, theta=0.0, amp_rad=0.30)
plt.imshow( phi ); plt.axis('off')
plt.show()


#%%
################################################################
################################################################
################ Section 1 -  image plane cartoon of cold stop with speckles
################################################################
################################################################


import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from matplotlib import rcParams

def draw_coldstop_cartoon(Rcs=2.0,                   # stop radius in cycles/diam (D_cs=4 ⇒ Rcs=2)
                          k0=None,                   # speckle radius; default = on the stop edge
                          core_radius=0.18,          # on-axis PSF core radius (cartoon)
                          speckle_radius=0.18,       # speckle “size” (cartoon)
                          misalign=(0.0, 0.0),       # (dx,dy) shift in cycles/diam
                          ax=None, title="", annotate=True):
    """
    Simple image-plane cartoon in cycles/pupil-diameter units.
    Draws: dashed stop, central PSF core, two speckle circles at ±k0, optional misalignment.
    """
    fs = 15 # fontsize 
    if ax is None:
        fig, ax = plt.subplots(figsize=(4,4))
    ax.set_aspect('equal')

    if k0 is None:
        k0 = Rcs  # put speckles on the edge by default

    # Misalignment vector (Δρ) applies to the stop only (typical lab case); set to (0,0) for aligned
    dx, dy = misalign

    # Cold stop
    ax.add_patch(Circle((dx, dy), radius=Rcs, fill=False, ls='--', lw=2.0, ec='k', alpha=0.8, label="Cold stop"))

    # Central PSF core (cartoon)
    ax.add_patch(Circle((0.0, 0.0), radius=core_radius, color='#9999ff', alpha=0.9, label="PSF core"))

    # Speckles at ±k0 along x (rotate if you prefer a different azimuth)
    ax.add_patch(Circle((+k0, 0.0), radius=speckle_radius, color='#66cc66', alpha=0.95, label="+ speckle"))
    ax.add_patch(Circle((-k0, 0.0), radius=speckle_radius, color='#66cc66', alpha=0.95, label="− speckle"))

    # Axes, limits, labels
    m = 1.25*Rcs
    ax.set_xlim(-m, m); ax.set_ylim(-m, m)
    ax.set_xlabel(r"$\xi_x$  [cycles / pupil diam]",fontsize=fs)
    ax.set_ylabel(r"$\xi_y$  [cycles / pupil diam]",fontsize=fs)
    ax.grid(alpha=0.2)
    ax.tick_params(labelsize=fs)
    #if title:
    #    #ax.set_title(title,fontsize=fs)

    if annotate:
        ax.text(0.03, 0.95, rf"$D_{{\rm cs}}={2*Rcs:.1f}\,\lambda/D$",
                transform=ax.transAxes, ha='left', va='top',fontsize=fs)
        ax.text(0.03, 0.88, rf"$|\xi_0|=k_0={k0:.2f}$", transform=ax.transAxes,
                ha='left', va='top',fontsize=fs)
        if (dx,dy)!=(0.0,0.0):
            ax.text(0.03, 0.81, rf"$\Delta\rho=({dx:.2f},{dy:.2f})$", transform=ax.transAxes,
                    ha='left', va='top',fontsize=fs)

    return ax

# --------- Example usage: two-panel cartoon ---------
Dcs = 4.0
Rcs = Dcs/2.0                 # 2 cycles/diam
k0  = Rcs                     # speckles exactly on the stop edge
core_r = 0.20                 # purely illustrative
speck_r = 0.20

fig, axs = plt.subplots(2, 1, figsize=(6.2, 12), constrained_layout=True)

# (a) Aligned stop
draw_coldstop_cartoon(Rcs=Rcs, k0=k0,
                      core_radius=core_r, speckle_radius=speck_r,
                      misalign=(0.0, 0.0),
                      ax=axs[0], title="Aligned cold stop", annotate=True)

# (b) Misaligned stop (e.g. Δρ = 0.3 λ/D along +x)
draw_coldstop_cartoon(Rcs=Rcs, k0=k0,
                      core_radius=core_r, speckle_radius=speck_r,
                      misalign=(stop_misalignment, 0.00),
                      ax=axs[1], title=rf"Misaligned stop ($\Delta\rho={stop_misalignment}\,\lambda/D$)", annotate=True)

plt.show()



#%% 
################################################################
################################################################
################ Section 2 -  fourier (pupil plane) analsis of the effect of the filter with speckle at its cut off with/without misalignment
################################################################
################################################################
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.colors import Normalize
from scipy.special import j1  # Bessel J1

# ---------- helpers ----------
def colorline(ax, x, y, phase, cmap='hsv', lw=2.5, zorder=3):
    norm = Normalize(vmin=0.0, vmax=2*np.pi)
    pts = np.array([x, y]).T.reshape(-1,1,2)
    segs = np.concatenate([pts[:-1], pts[1:]], axis=1)
    lc = LineCollection(segs, cmap=cmap, norm=norm, linewidth=lw, zorder=zorder)
    lc.set_array((phase % (2*np.pi)))
    ax.add_collection(lc)

def airy_filter_1d(k, Rcs, d_rho=0.0):
    x = 2*np.pi*Rcs*np.abs(k) + 1e-20
    Hmag = np.abs(j1(x) / x)
    Hmag /= (Hmag.max() + 1e-20)
    return Hmag * np.exp(1j * 2*np.pi * d_rho * k)

def gaussian_component(k, k0, fwhm, phase):
    sigma = fwhm / (2*np.sqrt(2*np.log(2)))   # FWHM→σ
    return np.exp(-(k-k0)**2/(2*sigma**2)) * np.exp(1j*phase)

def draw_spike_markers(ax, k, S, k0, lw=3):
    for sgn in (+1, -1):
        i = np.argmin(np.abs(k - sgn*k0))
        amp = np.abs(S[i]); phs = np.angle(S[i])
        col = plt.get_cmap('hsv')((phs % (2*np.pi)) / (2*np.pi))
        ax.vlines(k[i], 0, amp, color=col, linewidth=lw, zorder=5)

# ---------- parameters ----------
Dcs   = 4.0                 # cold-stop diameter [λ/D]
Rcs   = Dcs/2.0             # cutoff radius [cycles/diam]
phi0  = np.pi/2             # intrinsic phase of the sinusoid

# Airy zeros (J1): 3.831706, 7.015586
z1, z2 = 3.831706, 7.015586
k_null1 = z1/(2*np.pi*Rcs)  # ≈ 0.305
k_null2 = z2/(2*np.pi*Rcs)

# Frequency axis (limit to first two nulls)
kmin, kmax = -1.1*k_null2, 1.1*k_null2
k = np.linspace(kmin, kmax, 4001)

# Probe just inside the first null, finite width (narrow)
k0   = 0.98 * k_null1
fwhm = 0.10

# Two components (colored separately in the input panels)
S_plus  = gaussian_component(k, +k0, fwhm, +phi0)
S_minus = gaussian_component(k, -k0, fwhm, +phi0)
# Sum is what physically multiplies the filter
S = S_plus + S_minus
# Normalize the sum for display
S /= (np.max(np.abs(S)) + 1e-20)

# Filters
H_al = airy_filter_1d(k, Rcs, d_rho=0.0)  # aligned
H_ms = airy_filter_1d(k, Rcs, d_rho=stop_misalignment)  # misaligned (≈ 1 λ/D ramp)

# Products
P_al = S * H_al
P_ms = S * H_ms

# ---------- plotting ----------
fig, axs = plt.subplots(2, 2, figsize=(12.0, 6.2), sharex=True, sharey=True, constrained_layout=True)
fs = 15 # fontsize
def panel(axL, axR, H, titleL, titleR, lw_line=2.5):
    
    axL.set_title(titleL, fontsize =fs)

    # Filter magnitude with its own phase coloring
    colorline(axL, k, np.abs(H), np.angle(H), lw=lw_line)

    # Input spectrum: color the two finite spikes by their *own* constant phases
    colorline(axL, k, np.abs(S_plus),  +np.zeros_like(k)+(+phi0), lw=lw_line-0.5, zorder=4)
    colorline(axL, k, np.abs(S_minus), +np.zeros_like(k)+(+phi0), lw=lw_line-0.5, zorder=4)

    # Guides: bold at ±k0; faint at 1st/2nd nulls
    axL.axvline(+k0, color='k', lw=2.0); axL.axvline(-k0, color='k', lw=2.0)
    #for kk, a in [(+k_null1,0.5), (-k_null1,0.5), (+k_null2,0.25), (-k_null2,0.25)]:
    #    axL.axvline(kk, color='k', lw=1, ls=':', alpha=a)

    # Markers at ±k0 using *sum* (what the sensor sees)
    draw_spike_markers(axL, k, S, k0, lw=3)

    axL.set_ylabel("Amplitude",fontsize=fs); axL.grid(True, alpha=0.25)

    # Right: product (sum × filter), phase-colored
    axR.set_title(titleR, fontsize =fs)
    colorline(axR, k, np.abs(S*H), np.angle(S*H), lw=lw_line)
    draw_spike_markers(axR, k, S*H, k0, lw=3)
    axR.axvline(+k0, color='k', lw=2.0); axR.axvline(-k0, color='k', lw=2.0)
    for kk, a in [(+k_null1,0.5), (-k_null1,0.5), (+k_null2,0.25), (-k_null2,0.25)]:
        axR.axvline(kk, color='k', lw=1, ls=':', alpha=a)
    axR.grid(True, alpha=0.25)

# Row 1: aligned
panel(axs[0,0], axs[0,1], H_al,
      "Aberration at spatial frequency (S) + \ncold-stop filter $|\\widehat K|$",
      "Product $S\\cdot\\widehat K$ (aligned cold-stop)")

# Row 2: misaligned (phase ramp -> rotation)
panel(axs[1,0], axs[1,1], H_ms,
      f"Aberration at spatial frequency (S) + \nmisaligned cold-stop filter ($\\Delta\\rho={stop_misalignment}\\,\\lambda/D$)",
      "Product $S\\cdot\\widehat K$ (misaligned cold-stop)")

for r in range(2):
    for c in range(2):
        axs[r,c].set_xlim(kmin, kmax)
        axs[r,c].set_ylim(0, 1.12)
        if r > 0:
            axs[r,c].set_xlabel(r"Spatial frequency $k$ [cycles / pupil diam]", fontsize=fs)
        axs[r,c].tick_params(labelsize=15)
# Colorbar on the side (not covering the plots)
sm = plt.cm.ScalarMappable(cmap='hsv', norm=plt.Normalize(0, 2*np.pi))
sm.set_array([])
cbar = fig.colorbar(sm, ax=axs, location='right', fraction=0.035, pad=0.02)
cbar.ax.tick_params(labelsize=fs)
cbar.set_label("Phase [rad]",fontsize=fs)

plt.show()


#%%

################################################################
################################################################
################ Section 3 -  just a simple image of output after cold stop w/wo misalignment
################################################################
################################################################

mask = 'H3'

phasemask_parameters = {
    "J5": {"depth":0.474 ,  "diameter":32},
    "J4": {"depth":0.474 ,  "diameter":36}, 
    "J3": {"depth":0.474 ,  "diameter":44}, 
    "J2": {"depth":0.474 ,  "diameter":54},
    "J1": {"depth":0.474 ,  "diameter":65},
    "H1": {"depth":0.654 ,  "diameter":68},  
    "H2": {"depth":0.654 ,  "diameter":53}, 
    "H3": {"depth":0.654 ,  "diameter":44}, 
    "H4": {"depth":0.654 ,  "diameter":37},
    "H5": {"depth":0.654 ,  "diameter":31}
}

def zwfs_intensity(phi, coldstop_diam_lamOverD, padding=6, phaseshift=None,coldstop_misalign=[0,0]):
    P_, Ic_ = util.get_theoretical_reference_pupils_with_aber(
        wavelength = wvl,
        F_number = F_number,
        mask_diam = mask_diam,
        coldstop_diam = coldstop_diam_lamOverD,  # DIAMETER in λ/D
        coldstop_misalign = coldstop_misalign,
        eta = eta,
        phi = phi,
        diameter_in_angular_units = True,
        get_individual_terms = False,
        phaseshift = phaseshift if phaseshift is not None 
                         else util.get_phasemask_phaseshift(wvl=wvl, depth=phasemask_parameters[mask]['depth'], dot_material='N_1405'),
        padding_factor = 6,
        debug = False,
        analytic_solution = False
    )
    return P_, Ic_



P0_0, I0_0 = zwfs_intensity(phi=phi,
                        coldstop_diam_lamOverD=1e3,
                        coldstop_misalign=[0.0,0],
                        padding=6)

P0, I0= zwfs_intensity(phi=phi,
                        coldstop_diam_lamOverD=4,
                        coldstop_misalign=[0.0,0],
                        padding=6)



P0_mis, I0_mis = zwfs_intensity(phi=phi,
                        coldstop_diam_lamOverD=4,
                        coldstop_misalign=[0.5,0],
                        padding=6)

n = I0.shape[0]
csa = 5

plt.figure()
plt.imshow( I0_0[n//2 - n//csa:n//2 + n//csa, n//2 - n//csa:n//2 + n//csa] )
plt.axis('off')
plt.show()

plt.figure()
plt.imshow( I0[n//2 - n//csa:n//2 + n//csa, n//2 - n//csa:n//2 + n//csa] )
plt.axis('off')
plt.show()

plt.figure()
plt.imshow( I0_mis[n//2 - n//csa:n//2 + n//csa, n//2 - n//csa:n//2 + n//csa] )
plt.axis('off')
plt.show()

#%%
################################################################
################################################################
################ Section 4 -  (didnt use) visual illustration of the ZWFS images , decompose to even and odd modes 
################################################################
################################################################

import numpy as np
import matplotlib.pyplot as plt
from pyBaldr import utilities as util
from common import DM_basis_functions as dmbasis

# ===============================
# Global: cold-stop misalignment
# ===============================
stop_misalignment = 0.5   # λ/D (applied along +x)

# ===============================
# Helpers
# ===============================
def unit_rms(mode, pupil):
    m = np.nan_to_num(mode) * pupil
    rms = np.sqrt(np.mean(m[pupil>0]**2) + 1e-30)
    return m / rms

def even_odd_centrosym(A):
    """Even/odd under 180° rotation (centro-symmetry)."""
    Arot = np.rot90(A, 2)
    even = 0.5*(A + Arot)
    odd  = 0.5*(A - Arot)
    return even, odd

def zwfs_intensity(phi, wvl, F_number, mask_diam, eta, phaseshift, Dcs,
                   misalign=(0.0,0.0), padding=6):
    """Wrapper around util.get_theoretical_reference_pupils_with_aber."""
    P_amp, I = util.get_theoretical_reference_pupils_with_aber(
        wavelength = wvl,
        F_number = F_number,
        mask_diam = mask_diam,
        coldstop_diam = Dcs,                     # diameter in λ/D
        coldstop_misalign = misalign,            # [dx,dy] in λ/D
        eta = eta,
        phi = phi,
        diameter_in_angular_units = True,
        get_individual_terms = False,
        phaseshift = phaseshift,
        padding_factor = padding,
        debug = False,
        analytic_solution = False
    )
    return P_amp, I

def make_aberration_from_zernike(N, nterms, idx, amp_rad=0.3):
    """Build a Zernike, normalize to 1 rad RMS inside pupil, then scale to amp_rad."""
    x = np.linspace(-1, 1, N)
    X, Y = np.meshgrid(x, x, indexing='xy')
    P = (X**2 + Y**2) <= 1
    Zbasis = dmbasis.zernike_basis(nterms=nterms, npix=N, rho=None, theta=None)
    Zbasis = [np.nan_to_num(z) for z in Zbasis]
    Z = unit_rms(Zbasis[idx] * P, P)
    return amp_rad * Z, P

# ------- detector binning + pretty zoom -------
def _pupil_center_outer_radius(P_amp):
    """Estimate pupil center and outer radius on fine grid."""
    Y, X = np.indices(P_amp.shape)
    w = (np.abs(P_amp) > 0).astype(float)
    tot = w.sum() + 1e-20
    cx = (X*w).sum()/tot
    cy = (Y*w).sum()/tot
    R = np.sqrt((X-cx)**2 + (Y-cy)**2)
    r_outer = R[w>0].max()
    return cx, cy, r_outer

def bin_to_detector_mean(I_fine, P_amp, M):
    """
    Flux-preserving area-average so the pupil spans exactly M pixels across the diameter.
    Returns (M x M).
    """
    cx, cy, r = _pupil_center_outer_radius(P_amp)
    Df = int(np.round(2*r))
    x0 = int(np.round(cx - Df/2)); x1 = x0 + Df
    y0 = int(np.round(cy - Df/2)); y1 = y0 + Df
    I_crop = I_fine[y0:y1, x0:x1]
    ny, nx = I_crop.shape
    if ny == 0 or nx == 0:  # guard
        return np.zeros((M, M), float)
    if (ny % M == 0) and (nx % M == 0):
        ky = ny // M; kx = nx // M
        blocks = I_crop.reshape(M, ky, M, kx).swapaxes(1,2)  # (M,M,ky,kx)
        return blocks.mean(axis=(2,3))
    # general mapping if not divisible
    jj, ii = np.indices((ny, nx))
    ix = np.floor(ii * (M/nx)).astype(int)
    iy = np.floor(jj * (M/ny)).astype(int)
    ok = (ix>=0)&(ix<M)&(iy>=0)&(iy<M)
    I_sum = np.zeros((M, M), float); N_hit = np.zeros((M, M), float)
    np.add.at(I_sum, (iy[ok], ix[ok]), I_crop[ok])
    np.add.at(N_hit, (iy[ok], ix[ok]), 1.0)
    out = np.zeros_like(I_sum); nz = N_hit>0
    out[nz] = I_sum[nz]/N_hit[nz]
    return out

def upsample_and_crop(I_det, crop_padding_px=1, upsample=12):
    """
    For display: nearest-neighbor upsample (so detector pixels look crisp),
    then crop tightly around the pupil with a small padding.
    """
    M = I_det.shape[0]
    # nearest-neighbor upsample
    I_big = I_det.repeat(upsample, axis=0).repeat(upsample, axis=1)
    # crop around central (M x M) footprint with padding
    cx = cy = (M*upsample)//2
    half = (M*upsample)//2 + crop_padding_px*upsample
    y0 = max(0, cy - half); y1 = min(I_big.shape[0], cy + half)
    x0 = max(0, cx - half); x1 = min(I_big.shape[1], cx + half)
    return I_big[y0:y1, x0:x1]

# ===============================
# Compute once; plot twice
# ===============================
def compute_zwfs_cases(
        wvl, F_number, mask_diam, eta,
        phasemask_parameters, mask_key="J5",
        Dcs=4.0, zernike_index=3, amp_rad=0.30,
        N=2**9+1
    ):
    # phase shift for this mask
    depth = phasemask_parameters[mask_key]['depth']
    phaseshift = util.get_phasemask_phaseshift(wvl=wvl, depth=depth, dot_material='N_1405')

    # aberration
    phi, Pmask = make_aberration_from_zernike(N, nterms=80, idx=zernike_index, amp_rad=amp_rad)

    # aligned
    P0_al, I0_al = zwfs_intensity(phi=np.zeros_like(Pmask), wvl=wvl, F_number=F_number,
                                  mask_diam=mask_diam, eta=eta, phaseshift=phaseshift,
                                  Dcs=Dcs, misalign=(0.0, 0.0))
    _,     I_al  = zwfs_intensity(phi=phi, wvl=wvl, F_number=F_number,
                                  mask_diam=mask_diam, eta=eta, phaseshift=phaseshift,
                                  Dcs=Dcs, misalign=(0.0, 0.0))
    dI_al = I_al - I0_al
    even_al, odd_al = even_odd_centrosym(dI_al)

    # misaligned
    dx = float(stop_misalignment); dy = 0.0
    P0_ms, I0_ms = zwfs_intensity(phi=np.zeros_like(Pmask), wvl=wvl, F_number=F_number,
                                  mask_diam=mask_diam, eta=eta, phaseshift=phaseshift,
                                  Dcs=Dcs, misalign=(dx, dy))
    _,     I_ms  = zwfs_intensity(phi=phi, wvl=wvl, F_number=F_number,
                                  mask_diam=mask_diam, eta=eta, phaseshift=phaseshift,
                                  Dcs=Dcs, misalign=(dx, dy))
    dI_ms = I_ms - I0_ms
    even_ms, odd_ms = even_odd_centrosym(dI_ms)

    return dict(
        phi=phi,
        P0_al=P0_al, I0_al=I0_al, I_al=I_al, dI_al=dI_al, even_al=even_al, odd_al=odd_al,
        P0_ms=P0_ms, I0_ms=I0_ms, I_ms=I_ms, dI_ms=dI_ms, even_ms=even_ms, odd_ms=odd_ms,
        phaseshift=phaseshift
    )

# ===============================
# Plot A: fine-grid 2×6 storyboard (per-column horizontal colorbars)
# ===============================
def plot_fine_storyboard(results, Dcs, mask_key, phasemask_parameters,
                         zernike_index, amp_rad, cmap='viridis'):
    phi = results['phi']
    I0_al, I_al, dI_al, even_al, odd_al = results['I0_al'], results['I_al'], results['dI_al'], results['even_al'], results['odd_al']
    I0_ms, I_ms, dI_ms, even_ms, odd_ms = results['I0_ms'], results['I_ms'], results['dI_ms'], results['even_ms'], results['odd_ms']

    col_titles = [
        r"Input phase $\phi$ [rad]",
        r"Ref. pupil $I_0$",
        r"ZWFS pupil $I$",
        r"$\Delta I = I - I_0$",
        r"centro-even($\Delta I$)",
        r"centro-odd($\Delta I$)"
    ]
    row0 = [phi, I0_al, I_al, dI_al, even_al, odd_al]
    row1 = [phi, I0_ms, I_ms, dI_ms, even_ms, odd_ms]

    # shared vlims per column across rows
    vlims = []
    for c in range(6):
        a, b = row0[c], row1[c]
        vmin = np.nanmin([np.nanmin(a), np.nanmin(b)])
        vmax = np.nanmax([np.nanmax(a), np.nanmax(b)])
        if c in (0,3,4,5):  # signed
            m = max(abs(vmin), abs(vmax)); vlims.append((-m, m))
        else:
            vlims.append((vmin, vmax))

    fig = plt.figure(figsize=(18, 7.0))
    gs = fig.add_gridspec(2, 6, left=0.04, right=0.98, top=0.92, bottom=0.20, wspace=0.06, hspace=0.12)

    axs = np.empty((2,6), dtype=object)
    ims = [[None]*6 for _ in range(2)]
    for r in range(2):
        for c in range(6):
            ax = fig.add_subplot(gs[r, c])
            axs[r, c] = ax
            im = ax.imshow([row0, row1][r][c], origin='lower', cmap=cmap,
                           vmin=vlims[c][0], vmax=vlims[c][1])
            ims[r][c] = im
            ax.set_title(col_titles[c], fontsize=11)
            ax.set_xticks([]); ax.set_yticks([])

    # annotate rows
    axs[0, 0].text(0.02, 0.96, "Aligned stop",
                   transform=axs[0,0].transAxes, ha='left', va='top',
                   color='w', bbox=dict(fc='k', alpha=0.35, lw=0))
    axs[1, 0].text(0.02, 0.96, rf"Misaligned stop  ($\Delta\rho={stop_misalignment:.2f}\,\lambda/D$)",
                   transform=axs[1,0].transAxes, ha='left', va='top',
                   color='w', bbox=dict(fc='k', alpha=0.35, lw=0))

    # per-column horizontal colorbars (under bottom row)
    for c in range(6):
        bbox = axs[1, c].get_position()
        cbar_h = 0.018; pad = 0.010
        cax = fig.add_axes([bbox.x0, bbox.y0 - pad - cbar_h, bbox.width, cbar_h])
        cb = fig.colorbar(ims[1][c], cax=cax, orientation='horizontal')
        cb.ax.tick_params(labelsize=9)

    fig.suptitle(
        rf"ZWFS pupil (fine grid) with cold stop $D_{{\rm cs}}={Dcs:.1f}\,\lambda/D$  •  "
        rf"Zernike index {zernike_index}  •  amplitude {amp_rad:.2f} rad RMS  •  "
        rf"mask {mask_key} (depth {phasemask_parameters[mask_key]['depth']:.3f})",
        fontsize=13, y=0.99
    )
    plt.show()

def plot_detector_storyboard(results, Dcs, mask_key, phasemask_parameters,
                             zernike_index, amp_rad,
                             Mdet=6, det_upsample=18, det_crop_pad=1, cmap='viridis'):
    # --- unpack (fine-grid data present here) ---
    phi = results['phi']
    P0_al, I0_al, I_al = results['P0_al'], results['I0_al'], results['I_al']
    P0_ms, I0_ms, I_ms = results['P0_ms'], results['I0_ms'], results['I_ms']

    # --- detector binning (use proper pupil for geometric crop) ---
    I0_al_det = bin_to_detector_mean(I0_al, P0_al, Mdet)
    I_al_det  = bin_to_detector_mean(I_al,  P0_al, Mdet)
    dI_al_det = I_al_det - I0_al_det
    even_al_det, odd_al_det = even_odd_centrosym(dI_al_det)

    I0_ms_det = bin_to_detector_mean(I0_ms, P0_ms, Mdet)
    I_ms_det  = bin_to_detector_mean(I_ms,  P0_ms, Mdet)
    dI_ms_det = I_ms_det - I0_ms_det
    even_ms_det, odd_ms_det = even_odd_centrosym(dI_ms_det)

    # --- upsample & crop detector images so the pupil fills the panel ---
    def make_disp(I0_det, I_det, dI_det, ev_det, od_det):
        I0 = upsample_and_crop(I0_det, crop_padding_px=det_crop_pad, upsample=det_upsample)
        I  = upsample_and_crop(I_det,   crop_padding_px=det_crop_pad, upsample=det_upsample)
        dI = upsample_and_crop(dI_det,  crop_padding_px=det_crop_pad, upsample=det_upsample)
        EV = upsample_and_crop(ev_det,  crop_padding_px=det_crop_pad, upsample=det_upsample)
        OD = upsample_and_crop(od_det,  crop_padding_px=det_crop_pad, upsample=det_upsample)
        return I0, I, dI, EV, OD

    I0_al_disp, I_al_disp, dI_al_disp, ev_al_disp, od_al_disp = make_disp(
        I0_al_det, I_al_det, dI_al_det, even_al_det, odd_al_det)
    I0_ms_disp, I_ms_disp, dI_ms_disp, ev_ms_disp, od_ms_disp = make_disp(
        I0_ms_det, I_ms_det, dI_ms_det, even_ms_det, odd_ms_det)

    # --- assemble rows (col0 is full-res phi, same for both rows) ---
    det_titles = [
        r"$\phi$ [rad] (full-res)",
        r"$I_0^{\rm det}$",
        r"$I^{\rm det}$",
        r"$\Delta I^{\rm det}$",
        r"centro-even($\Delta I^{\rm det}$)",
        r"centro-odd($\Delta I^{\rm det}$)"
    ]
    row0 = [phi, I0_al_disp, I_al_disp, dI_al_disp, ev_al_disp, od_al_disp]
    row1 = [phi, I0_ms_disp, I_ms_disp, dI_ms_disp, ev_ms_disp, od_ms_disp]

    # --- shared vlims per column (phase & ΔI symmetric) ---
    vlims = []
    # col 0: phi (signed, full-res shared)
    m0 = np.nanmax(np.abs(phi))
    vlims.append((-m0, m0))
    # col 1: I0
    v1min = min(np.nanmin(row0[1]), np.nanmin(row1[1])); v1max = max(np.nanmax(row0[1]), np.nanmax(row1[1]))
    vlims.append((v1min, v1max))
    # col 2: I
    v2min = min(np.nanmin(row0[2]), np.nanmin(row1[2])); v2max = max(np.nanmax(row0[2]), np.nanmax(row1[2]))
    vlims.append((v2min, v2max))
    # col 3: ΔI (signed)
    m3 = max(np.nanmax(np.abs(row0[3])), np.nanmax(np.abs(row1[3])))
    vlims.append((-m3, m3))
    # col 4: even (signed)
    m4 = max(np.nanmax(np.abs(row0[4])), np.nanmax(np.abs(row1[4])))
    vlims.append((-m4, m4))
    # col 5: odd (signed)
    m5 = max(np.nanmax(np.abs(row0[5])), np.nanmax(np.abs(row1[5])))
    vlims.append((-m5, m5))

    # --- figure layout: 2×6, horizontal per-column cbars ---
    fig = plt.figure(figsize=(18.0, 8.2))
    gs = fig.add_gridspec(2, 6, left=0.05, right=0.98, top=0.92, bottom=0.22, wspace=0.06, hspace=0.16)

    axs = np.empty((2,6), dtype=object)
    ims = [[None]*6 for _ in range(2)]

    for r, row in enumerate([row0, row1]):
        for c in range(6):
            ax = fig.add_subplot(gs[r, c]); axs[r, c] = ax
            im = ax.imshow(row[c], origin='lower', cmap=cmap, vmin=vlims[c][0], vmax=vlims[c][1])
            ims[r][c] = im
            if r == 0:
                # add Mdet only to I0 title
                ttl = det_titles[c] + (f"  (M={Mdet})" if c==1 else "")
                ax.set_title(ttl, fontsize=11)
            ax.set_xticks([]); ax.set_yticks([])

    # annotate rows
    fig.text(0.02, 0.565, "Aligned stop", fontsize=11, ha='left', va='center',
             bbox=dict(fc='k', ec='none', alpha=0.25, pad=3, boxstyle="round"))
    fig.text(0.02, 0.305, rf"Misaligned stop  ($\Delta\rho={stop_misalignment:.2f}\,\lambda/D$)", fontsize=11, ha='left', va='center',
             bbox=dict(fc='k', ec='none', alpha=0.25, pad=3, boxstyle="round"))

    # per-column horizontal colorbars (under bottom row)
    for c in range(6):
        bbox = axs[1, c].get_position()
        cbar_h = 0.020; pad = 0.012
        cax = fig.add_axes([bbox.x0, bbox.y0 - pad - cbar_h, bbox.width, cbar_h])
        cb = fig.colorbar(ims[1][c], cax=cax, orientation='horizontal')
        cb.ax.tick_params(labelsize=9)
        if c == 0:
            cb.set_label("Phase [rad]", fontsize=10)
        if c == 3:
            cb.set_label(r"$\Delta I^{\rm det}$ scale", fontsize=10)
        if c == 4:
            cb.set_label("centro-even scale", fontsize=10)
        if c == 5:
            cb.set_label("centro-odd scale", fontsize=10)

    fig.suptitle(
        rf"ZWFS detector view (M={Mdet} px / pupil diam) + full-res input phase, "
        rf"cold stop $D_{{\rm cs}}={Dcs:.1f}\,\lambda/D$  •  "
        rf"Zernike index {zernike_index}  •  amplitude {amp_rad:.2f} rad RMS  •  "
        rf"mask {mask_key} (depth {phasemask_parameters[mask_key]['depth']:.3f})",
        fontsize=13, y=0.98
    )
    plt.show()
# ===============================
# Example call (fill in instrument numbers)
# ===============================
# Lab/instrument numbers you shared earlier
T = 1900  # K
lambda_cut_on, lambda_cut_off = 1.38, 1.82  # μm
wvl = util.find_central_wavelength(lambda_cut_on, lambda_cut_off, T)

mask_key = "J5"
F_number = 21.2
phasemask_parameters = {
    "J5": {"depth":0.474 ,  "diameter":32},
    "J4": {"depth":0.474 ,  "diameter":36}, 
    "J3": {"depth":0.474 ,  "diameter":44}, 
    "J2": {"depth":0.474 ,  "diameter":54},
    "J1": {"depth":0.474 ,  "diameter":65},
    "H1": {"depth":0.654 ,  "diameter":68},  
    "H2": {"depth":0.654 ,  "diameter":53}, 
    "H3": {"depth":0.654 ,  "diameter":44}, 
    "H4": {"depth":0.654 ,  "diameter":37},
    "H5": {"depth":0.654 ,  "diameter":31}
}
mask_diam = 1.22 * F_number * wvl / phasemask_parameters[mask_key]['diameter']
eta = 138 / 1800
Dcs = 4.0  # λ/D

# Compute once
res = compute_zwfs_cases(
    wvl, F_number, mask_diam, eta,
    phasemask_parameters, mask_key=mask_key,
    Dcs=Dcs, zernike_index=3, amp_rad=0.30, N=2**9+1
)

# A) Fine-grid storyboard
plot_fine_storyboard(res, Dcs, mask_key, phasemask_parameters,
                     zernike_index=3, amp_rad=0.30, cmap='viridis')

# B) Detector storyboard @ Mdet=6 (separate figure)
plot_detector_storyboard(res, Dcs, mask_key, phasemask_parameters,
                         zernike_index=3, amp_rad=0.30,
                         Mdet=6, det_upsample=18, det_crop_pad=1.2, cmap='viridis')


