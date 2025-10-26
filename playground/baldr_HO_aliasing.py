import numpy as np
import matplotlib.pyplot as plt
from pyBaldr import utilities as util
from common import DM_basis_functions as dmbasis
from xaosim.shmlib import shm



"""
What happens to a high order aberration that is aliased into the control range in 
the case of it being fully or half transmitted through the cold stop? Does the 
simulation framework deal with this properly (e.g. subsampling pixels adequately, 
and showing that you do by changing the subsampling and seeing the results change).
"""
# dictionary with depths referenced for beam 2 (1-5 goes big to small)
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

"""
email from Mike 5/12/24 ("dichroic curves")
optically you have 1380-1820nm (50% points) optically, 
and including the atmosphere it is ~1420-1820nm. 

"""


T = 1900 #K lab thermal source temperature 
lambda_cut_on, lambda_cut_off =  1.38, 1.82 # um
wvl = util.find_central_wavelength(lambda_cut_on, lambda_cut_off, T) # central wavelength of Nice setup
mask = "J5"
F_number = 21.2
coldstop_diam = 4.04 #according to calc in thesis 8.07 lmabda/D bright, 4.04 lambda/D faint
mask_diam = 1.22 * F_number * wvl / phasemask_parameters[mask]['diameter']
eta = 138 / 1800 #0.647/4.82 #~= 1.1/8.2 (i.e. UTs) # ratio of secondary obstruction (UTs), secondary obstruction ATs 138 mm / 1800mm, (https://www.eso.org/sci/facilities/paranal/telescopes/vlti/subsystems/at/technic.html)


###################
 
N = 2**9 + 1 # number of pixels

x = np.linspace(-1, 1, N)
X, Y = np.meshgrid(x, x, indexing='xy')
P = (X**2 + Y**2) <= 1  # circular pupil mask
basis = dmbasis.zernike_basis(nterms=120, npix=N, rho=None, theta=None) #dmbasis.develop_Fourier_basis( n=10, m=10 ,P = 120, Nx =N, Ny = N) #for k in basis:basis[k] *= P
basis = [np.nan_to_num(b) for b in basis]

# =========================
# Aliasing demo helpers
# =========================

def unit_rms(arr, mask):
    arr = arr.copy()
    arr -= arr[mask].mean()
    rms = np.sqrt((arr[mask]**2).mean()) + 1e-12
    return arr / rms

def make_fourier_probe(X, Y, P, k_cyc_per_diam=2.0, theta_deg=0.0):
    """
    Cosine Fourier phase probe on the *pupil* (cycles per pupil diameter).
    """
    # coords are in [-1,1] across diameter ⇒ 2 units across diameter
    # "k_cyc_per_diam" cycles across diameter ⇒ k cycles / 2 units ⇒ spatial freq per unit = k/2
    k = k_cyc_per_diam / 2.0
    th = np.deg2rad(theta_deg)
    KX, KY = k*np.cos(th), k*np.sin(th)
    phi = np.cos(2*np.pi*(KX*X + KY*Y)) * P
    return unit_rms(phi, P)

def pick_low_order_zernikes(basis, P, noll_indices=(2,3,4,5,6)):
    """
    Pick TT/Defocus/Astig/etc by Noll indices (1-based).
    Default grabs: Z2 (X-tilt), Z3 (Y-tilt), Z4 (defocus), Z5/Z6 (astigs).
    """
    los = []
    for j in noll_indices:
        z = np.nan_to_num(basis[j-1]) * P
        los.append(unit_rms(z, P))
    return los

def zwfs_intensity(phi, coldstop_diam_lamOverD, padding=6, phaseshift=None):
    """
    Wrapper to call your util forward model. Returns (PupilAmplitude, Intensity) on the fine grid.
    """
    P_, Ic_ = util.get_theoretical_reference_pupils_with_aber(
        wavelength = wvl,
        F_number = F_number,
        mask_diam = mask_diam,
        coldstop_diam = coldstop_diam_lamOverD,  # in λ/D (DIAMETER)
        coldstop_misalign = [0,0],
        eta = eta,
        phi = phi,
        diameter_in_angular_units = True,
        get_individual_terms = False,
        phaseshift = phaseshift if phaseshift is not None 
                           else util.get_phasemask_phaseshift(wvl=wvl, depth=phasemask_parameters[mask]['depth'], dot_material='N_1405'),
        padding_factor = padding,
        debug = False,
        analytic_solution = False
    )
    return P_, Ic_

# def bin_to_detector(P_amp, I_fine, pupil_pixels_diam=12):
#     """
#     Use your interpolation helper to map the fine-grid pupil intensity to a detector
#     with a given pupil diameter in pixels (e.g. 6 or 12).
#     """
#     Mfine, Nfine = I_fine.shape
#     m = n = 3 * pupil_pixels_diam  # a small field around the pupil; adjust if you want tighter
#     x_c = y_c = m//2
#     new_radius = pupil_pixels_diam // 2  # in pixels (radius = diameter/2)
#     I_det = util.interpolate_pupil_to_measurement(np.abs(P_amp), np.abs(I_fine), 
#                                                   Mfine, Nfine, m, n, x_c, y_c, new_radius)
#     return I_det



def _center_outer_radius(pupil_amp):
    """Center (cx,cy) and OUTER radius r (in fine pixels) from the amplitude mask."""
    P = np.abs(pupil_amp)
    mask = P > (1e-6 * P.max())
    if not np.any(mask):
        raise ValueError("No pupil detected.")
    ys, xs = np.where(mask)
    cx = xs.mean()
    cy = ys.mean()
    # outer radius from bounding box (robust to central obscuration)
    rx = 0.5 * (xs.max() - xs.min() + 1)
    ry = 0.5 * (ys.max() - ys.min() + 1)
    r = max(rx, ry)
    return cx, cy, r


def bin_to_detector(P_amp, I_fine, pupil_pixels_diam=12, mode="mean"):
    """
    Flux-preserving downsample that keeps the SAME FOV ratio as the fine image.
    The output image size is Md x Md where Md ≈ (Nf / Df) * M_pupil.
    The pupil spans ≈ M_pupil pixels across the DIAMETER on the output.
    
    Parameters
    ----------
    I_fine : (Nf,Nf) array
        Fine-grid intensity (I or ΔI).
    P_amp : (Nf,Nf) array
        Fine-grid pupil amplitude (used only to measure the outer radius).
    M_pupil : int
        Desired pixels across the pupil DIAMETER on the output.
    mode : 'mean' or 'sum'
        'mean' preserves intensity scale across different M_pupil (recommended).
        'sum' preserves total counts (will scale with pixel area).
    
    Returns
    -------
    I_det : (Md,Md) array
        Downsampled detector image with same FOV ratio as I_fine.
    Md : int
        Output linear size in pixels.
    """
    M_pupil = pupil_pixels_diam # shorthand 
    Nf = I_fine.shape[0]
    assert I_fine.shape[0] == I_fine.shape[1], "Assumes square frame."

    # measure outer diameter in fine pixels
    _, _, r = _center_outer_radius(P_amp)
    Df = 2.0 * r                         # fine pupil DIAMETER in pixels

    # FOV (in diameters) = Nf / Df ; keep this constant
    Md = int(round((Nf / Df) * M_pupil))
    Md = max(Md, M_pupil)               # guard against rounding pathologies

    # map *entire* fine frame to Md x Md via flux-preserving bin (area average)
    # fast path if divisible
    if Nf % Md == 0:
        k = Nf // Md
        blocks = I_fine.reshape(Md, k, Md, k).swapaxes(1, 2)  # (Md,Md,k,k)
        if mode == "mean":
            I_det = blocks.mean(axis=(2, 3))
        elif mode == "sum":
            I_det = blocks.sum(axis=(2, 3))
        else:
            raise ValueError("mode must be 'mean' or 'sum'")
        return I_det #, Md

    # general path (not divisible): accumulate area averages
    jj, ii = np.indices(I_fine.shape)
    scale = Md / Nf
    ix = np.floor(ii * scale).astype(int)
    iy = np.floor(jj * scale).astype(int)
    ok = (ix >= 0) & (ix < Md) & (iy >= 0) & (iy < Md)

    I_sum = np.zeros((Md, Md), float)
    N_hit = np.zeros((Md, Md), float)
    np.add.at(I_sum, (iy[ok], ix[ok]), I_fine[ok])
    np.add.at(N_hit, (iy[ok], ix[ok]), 1.0)

    if mode == "sum":
        I_det = I_sum
    else:  # mean (recommended)
        I_det = np.zeros_like(I_sum)
        nz = N_hit > 0
        I_det[nz] = I_sum[nz] / N_hit[nz]

    return I_det #, Md

# def bin_to_detector(P_amp, I_fine, pupil_pixels_diam=12 ):
#     """
#     Flux-preserving bin to an MxM detector such that the PUPIL spans exactly M pixels (diameter).
#     Uses fast block-mean if divisible, else a general bin-sum/mean.
#     """
#     M = pupil_pixels_diam # short hand 
#     # 1) center & OUTER radius from the amplitude (or binary) pupil
#     cx, cy, r = _pupil_center_outer_radius(np.abs(P_amp))

#     # 2) crop tight square of side Df around the pupil
#     Df = int(np.round(2*r))
#     x0 = int(np.round(cx - Df/2)); x1 = x0 + Df
#     y0 = int(np.round(cy - Df/2)); y1 = y0 + Df
#     I_crop = I_fine[y0:y1, x0:x1]   # shape ~ (Df, Df)

#     # 3) if divisible, use fast reshape-based block mean
#     Df = I_crop.shape[0]
#     if Df % M == 0:
#         k = Df // M
#         # (M,k,M,k) -> (M,M) block mean
#         I_blocks = I_crop.reshape(M, k, M, k).swapaxes(1,2)  # (M,M,k,k)
#         return I_blocks.mean(axis=(2,3))

#     # 4) general case: map fine pixels to detector bins, accumulate mean
#     ny, nx = I_crop.shape
#     jj, ii = np.indices((ny, nx))
#     # map [0, Df) -> [0, M)
#     x_det = ii * (M / Df)
#     y_det = jj * (M / Df)
#     ix = np.floor(x_det).astype(int)
#     iy = np.floor(y_det).astype(int)
#     ok = (ix>=0)&(ix<M)&(iy>=0)&(iy<M)

#     I_sum = np.zeros((M, M), float)
#     N_hit = np.zeros((M, M), float)
#     np.add.at(I_sum, (iy[ok], ix[ok]), I_crop[ok])
#     np.add.at(N_hit, (iy[ok], ix[ok]), 1.0)

#     I_det = np.zeros_like(I_sum)
#     nz = N_hit > 0
#     I_det[nz] = I_sum[nz] / N_hit[nz]
#     return I_det

def build_sensing_matrix(low_order_modes, coldstop_diam_lamOverD, padding=6):
    """
    For each low-order mode (unit RMS phase), compute ΔI = I(phi) - I(0) on the fine grid.
    Return S as list of ΔI images and also stacks for later fitting.
    """
    # reference (zero-phase)
    P0, I0 = zwfs_intensity(phi=np.zeros_like(low_order_modes[0]), 
                            coldstop_diam_lamOverD=coldstop_diam_lamOverD, 
                            padding=padding)
    S = []
    for u in low_order_modes:
        _, Iu = zwfs_intensity(phi=u, coldstop_diam_lamOverD=coldstop_diam_lamOverD, padding=padding)
        S.append(Iu - I0)
    return P0, I0, S

def fit_low_order_coeffs(deltaI_img, S_list, mask=None):
    """
    Least-squares fit of deltaI_img to low-order sensing images in S_list.
    Returns coefficients and residual norm.
    """
    # vectorize on an in-pupil mask if provided; otherwise use all pixels
    if mask is None:
        mask = np.ones_like(deltaI_img, dtype=bool)
    A = np.stack([S[mask].ravel() for S in S_list], axis=1)  # [Npix, Nmodes]
    y = deltaI_img[mask].ravel()
    coeffs, *_ = np.linalg.lstsq(A, y, rcond=None)
    residual = np.linalg.norm(y - A @ coeffs) / (np.linalg.norm(y) + 1e-12)
    return coeffs, residual


# =========================
# Visual aliasing on fine grid
# =========================

# Choose a high-frequency probe (cycles / pupil diameter)
k_probe = 2.0   # adjust; for D_cs=4 (diameter), passband edge ~ 2 c/pd (radius)
theta_probe = 0.0

phi_probe = make_fourier_probe(X, Y, P, k_cyc_per_diam=k_probe, theta_deg=theta_probe)

# (1) No cold stop (very large diameter ⇒ effectively pass all)
P1, I1 = zwfs_intensity(phi=phi_probe, coldstop_diam_lamOverD=1e6, padding=6)
P0, I0 = zwfs_intensity(phi=np.zeros_like(phi_probe), coldstop_diam_lamOverD=1e6, padding=6)
dI_nostop = I1 - I0

# (2) Tight cold stop: D_cs = 4 λ/D (diameter)
D_cs_tight = 4.0
P2, I2 = zwfs_intensity(phi=phi_probe, coldstop_diam_lamOverD=D_cs_tight, padding=6)
_,   I0cs = zwfs_intensity(phi=np.zeros_like(phi_probe), coldstop_diam_lamOverD=D_cs_tight, padding=6)
dI_stop = I2 - I0cs

# Plot side-by-side to visually see smoothing/low-order-looking pattern
util.nice_heatmap_subplots(
    im_list=[dI_nostop, dI_stop],
    title_list=[fr'$\Delta I$ (no stop, $k={k_probe}$ c/pd)', fr'$\Delta I$ (cold stop $D_{{cs}}={D_cs_tight}\,\lambda/D$)'],
    cbar_label_list=['arb.', 'arb.'],
    fontsize=14, cbar_orientation='bottom', axis_off=True, vlims=None, savefig=None
)
plt.show()


# =========================
# Quantify mixing into low orders (fine grid)
# =========================

low_order = pick_low_order_zernikes(basis, P, noll_indices=(2,3,4,5,6))  # TT, TT, Defocus, Astig±

# Build sensing matrices for "no stop" and "tight stop"
P0_n, I0_n, S_nostop = build_sensing_matrix(low_order, coldstop_diam_lamOverD=1e6, padding=6)
P0_s, I0_s, S_stop   = build_sensing_matrix(low_order, coldstop_diam_lamOverD=D_cs_tight, padding=6)

# Fit coefficients
c_nostop, r_nostop = fit_low_order_coeffs(dI_nostop, S_nostop, mask=np.abs(P0_n)>0)
c_stop,   r_stop   = fit_low_order_coeffs(dI_stop,   S_stop,   mask=np.abs(P0_s)>0)

print("Low-order leakage (no stop):    coeffs =", np.round(c_nostop,3), "  residual =", np.round(r_nostop,3))
print("Low-order leakage (tight stop): coeffs =", np.round(c_stop,3),   "  residual =", np.round(r_stop,3))


# =========================
# Detector-mapped visuals (pixel aliasing view)
# =========================

# Case (a): D_cs = 4 λ/D, 6-pixel pupil diameter
I_det_a = bin_to_detector(P2, dI_stop, pupil_pixels_diam=6)

# Case (b): D_cs = 8 λ/D, 12-pixel pupil diameter
D_cs_wider = 8.0
P3, I3 = zwfs_intensity(phi=phi_probe, coldstop_diam_lamOverD=D_cs_wider, padding=6)
_,  I0w = zwfs_intensity(phi=np.zeros_like(phi_probe), coldstop_diam_lamOverD=D_cs_wider, padding=6)
dI_wide = I3 - I0w
I_det_b = bin_to_detector(P3, dI_wide, pupil_pixels_diam=12)

util.nice_heatmap_subplots(
    im_list=[phi_probe,I_det_a, I_det_b],
    title_list=['probe',r'(a) $\Delta I$ on detector: $D_{cs}=4$, $M=6$ px/diam',
                r'(b) $\Delta I$ on detector: $D_{cs}=8$, $M=12$ px/diam'],
    cbar_label_list=['arb','arb.','arb.'],
    fontsize=14, cbar_orientation='bottom', axis_off=True, vlims=None, savefig=None
)
plt.show()


# =========================
# Convergence test
# =========================

probe_list = [(2.0, 0.0), (2.0, 45.0)]  # (k,theta) pairs
D_list     = [4.0, 8.0]                 # cold-stop diameters to test
pads       = [4, 6, 8]

for D_cs in D_list:
    print(f"\n=== Convergence for D_cs = {D_cs} λ/D ===")
    for pad in pads:
        # recompute dI for this padding
        phi_probe = make_fourier_probe(X, Y, P, k_cyc_per_diam=2.0, theta_deg=0.0)
        P0, I0 = zwfs_intensity(phi=np.zeros_like(phi_probe), coldstop_diam_lamOverD=D_cs, padding=pad)
        _,  I1 = zwfs_intensity(phi=phi_probe, coldstop_diam_lamOverD=D_cs, padding=pad)
        dI = I1 - I0

        # low-order sensing with same padding
        low_order = pick_low_order_zernikes(basis, P, noll_indices=(2,3,4,5,6))
        _, _, S   = build_sensing_matrix(low_order, coldstop_diam_lamOverD=D_cs, padding=pad)

        c, r = fit_low_order_coeffs(dI, S, mask=np.abs(P0)>0)
        print(f" padding={pad}: coeffs={np.round(c,3)}, residual={np.round(r,3)}")




# ============================================================
# 2nd-stage AO aliasing demo:
# High-fidelity phase screen -> remove first 14 Zernikes (incl. TT)
# Measure aliased Tip/Tilt vs cold-stop diameter
# at 6 px/diam and 12 px/diam detector samplings.
# ============================================================

# ---------- utilities from the earlier message ----------
def unit_rms(arr, mask):
    arr = arr.copy()
    arr -= arr[mask].mean()
    rms = np.sqrt((arr[mask]**2).mean()) + 1e-12
    return arr / rms

def zwfs_intensity(phi, coldstop_diam_lamOverD, padding=6, phaseshift=None):
    P_, Ic_ = util.get_theoretical_reference_pupils_with_aber(
        wavelength = wvl,
        F_number = F_number,
        mask_diam = mask_diam,
        coldstop_diam = coldstop_diam_lamOverD,  # DIAMETER in λ/D
        coldstop_misalign = [0,0],
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

# def bin_to_detector(P_amp, I_fine, pupil_pixels_diam=12):
#     Mfine, Nfine = I_fine.shape
#     m = n = 3 * pupil_pixels_diam
#     x_c = y_c = m//2
#     new_radius = pupil_pixels_diam // 2
#     I_det = util.interpolate_pupil_to_measurement(np.abs(P_amp), np.abs(I_fine),
#                                                   Mfine, Nfine, m, n, x_c, y_c, new_radius)
#     return I_det

def bin_block_divisible(I_fine, pupil_center, pupil_radius_pixels, M, mode='mean'):
    """
    Block-average/sum onto an MxM detector, assuming fine pupil diameter is
    an integer multiple of M (i.e., 2*r % M == 0).
    """
    cx, cy = pupil_center
    r  = int(round(pupil_radius_pixels))
    Df = 2*r                      # fine-grid pupil diameter in px
    assert Df % M == 0, "Use this only when 2*r is divisible by M."
    k = Df // M                   # reduction factor per axis

    # Crop a tight square around the pupil
    x0, x1 = cx - r, cx + r
    y0, y1 = cy - r, cy + r
    I_crop = I_fine[y0:y1, x0:x1]           # shape (Df, Df)

    # Block reduce: (M, k, M, k) -> (M, M)
    I_blocks = I_crop.reshape(M, k, M, k).swapaxes(1,2)  # (M, M, k, k)
    if mode == 'mean':
        I_det = I_blocks.mean(axis=(2,3))   # area-average (recommended)
    elif mode == 'sum':
        I_det = I_blocks.sum(axis=(2,3))    # photon-sum (scales with k^2)
    else:
        raise ValueError("mode must be 'mean' or 'sum'")
    return I_det

def fit_coeffs_against_columns(img, cols, mask=None):
    if mask is None:
        mask = np.ones_like(img, dtype=bool)
    A = np.stack([C[mask].ravel() for C in cols], axis=1)
    y = img[mask].ravel()
    c, *_ = np.linalg.lstsq(A, y, rcond=None)
    res = np.linalg.norm(y - A @ c) / (np.linalg.norm(y) + 1e-12)
    return c, res

# ---------- 1) High-fidelity Kolmogorov phase screen ----------
def kolmogorov_phase_screen(N, pixel_scale=1.0, r0_pixels=50.0, seed=42, Pmask=None):
    """
    Simple isotropic phase screen ~ k^{-11/6} using FFT method.
    N: grid size, pixel_scale is arbitrary scaling unit,
    r0_pixels: Fried parameter in pixels (controls strength).
    Returns a real phase screen in radians (unnormalized).
    """
    rng = np.random.default_rng(seed)
    fx = np.fft.fftfreq(N, d=pixel_scale)
    fy = np.fft.fftfreq(N, d=pixel_scale)
    FX, FY = np.meshgrid(fx, fy, indexing='xy')
    k = np.sqrt(FX**2 + FY**2) + 1e-12

    # Kolmogorov PSD ~ k^(-11/3) for OPD; for phase ~ k^(-11/3) as well (up to constants).
    # We use amplitude spectrum ~ k^(-11/6)
    amp = k**(-11/6.0)

    # High/low cutoff to avoid infinities & ringing (tunable):
    kmin = 1.0 / (N * pixel_scale)
    kmax = 0.25 / pixel_scale
    amp *= (k >= kmin) * (k <= kmax)

    # Random complex spectrum with Hermitian symmetry:
    phase_rand = rng.uniform(0, 2*np.pi, size=(N, N))
    Fphi = amp * (np.cos(phase_rand) + 1j*np.sin(phase_rand))
    # Enforce Hermitian symmetry for real IFFT:
    Fphi = np.fft.fftshift(Fphi)
    # IFFT:
    phi = np.fft.ifft2(np.fft.ifftshift(Fphi)).real

    # Optional pupil mask to focus inside
    if Pmask is not None:
        phi *= Pmask

    # Normalize RMS to ~1 rad over the pupil if provided:
    if Pmask is not None:
        phi = unit_rms(phi, Pmask)
    return phi

# ---------- 2) Remove first 14 Zernike modes (incl. TT) ----------
def remove_low_zernikes(phi, zernike_basis, Pmask, noll_indices_remove=list(range(2,16)),scaling_factor = 0.3):
    """
    Project phi onto the given Zernike set and remove modes with indices in noll_indices_remove.
    (Noll indexing, 1-based; we remove Z2..Z15 -> 14 modes incl. TT.)
    """
    out = phi.copy()
    for j in noll_indices_remove:
        Z = np.nan_to_num(zernike_basis[j-1]) * Pmask
        Z = unit_rms(Z, Pmask)
        coeff = (out[Pmask] * Z[Pmask]).mean()  # simple inner product on pupil
        out -= coeff * Z
    # Clean piston
    out -= out[Pmask].mean()
    # Optional: scale to a desired RMS (to ensure small-phase regime)
    out = unit_rms(out, Pmask) * scaling_factor   # ~0.3 rad RMS to stay linear-ish
    return out

# ---------- 3) Build detector-space sensing for Tip/Tilt at a given cold stop ----------
def build_TT_sensing_detector(coldstop_diam, pupil_px_diam, basis_zern, Pmask):
    # Tip/Tilt Zernikes (Noll 2 and 3), unit RMS:
    Zx = unit_rms(np.nan_to_num(basis_zern[1]) * Pmask, Pmask)
    Zy = unit_rms(np.nan_to_num(basis_zern[2]) * Pmask, Pmask)

    # Reference intensity at this stop:
    P0, I0 = zwfs_intensity(phi=np.zeros_like(Pmask, float), coldstop_diam_lamOverD=coldstop_diam, padding=6)
    # ΔI for unit Tip and Tilt:
    _, Ix = zwfs_intensity(phi=Zx, coldstop_diam_lamOverD=coldstop_diam, padding=6)
    _, Iy = zwfs_intensity(phi=Zy, coldstop_diam_lamOverD=coldstop_diam, padding=6)

    dIx = Ix - I0
    dIy = Iy - I0

    # Bin to detector
    dIx_det = bin_to_detector(P0, dIx, pupil_pixels_diam=pupil_px_diam)
    dIy_det = bin_to_detector(P0, dIy, pupil_pixels_diam=pupil_px_diam)

    return dIx_det, dIy_det

# ---------- 4) Main sweep: TT aliasing vs cold-stop size ----------
def sweep_tt_aliasing_vs_stop(N_fine=513, coldstop_list=(3,4,5,6,7,8,9,10), pupil_px_opts=(6,12),noll_indices_remove=list(range(2,16)), scaling_factor=0.3):
    # Fine grid & pupil
    x = np.linspace(-1, 1, N_fine)
    X, Y = np.meshgrid(x, x, indexing='xy')
    Pmask = (X**2 + Y**2) <= 1

    # Zernike basis on the fine grid
    Zbasis = dmbasis.zernike_basis(nterms=120, npix=N_fine, rho=None, theta=None)
    Zbasis = [np.nan_to_num(z) for z in Zbasis]

    # High-fidelity phase screen, remove first 14 Zernikes (incl. TT)
    phi_raw = kolmogorov_phase_screen(N_fine, pixel_scale=1.0, r0_pixels=50.0, seed=7, Pmask=Pmask)
    phi_ho  = remove_low_zernikes(phi_raw, Zbasis, Pmask, noll_indices_remove=noll_indices_remove,scaling_factor=scaling_factor)

    # Precompute zero-phase intensities per stop (for speed we’ll do inside loop anyway)
    results = {M: {'phi_ho':phi_ho,'img':[],'img_det':[],'stop':[], 'TTx':[], 'TTy':[], 'resid':[]} for M in pupil_px_opts}

    for Dcs in coldstop_list:
        # Forward model for the HO screen at this stop
        P0, I0 = zwfs_intensity(phi=np.zeros_like(phi_ho), coldstop_diam_lamOverD=Dcs, padding=6)
        _,  I1 = zwfs_intensity(phi=phi_ho,            coldstop_diam_lamOverD=Dcs, padding=6)
        dI = I1 - I0  # fine-grid ΔI for the HO-only screen

        for Mdet in pupil_px_opts:
            # Build detector-space TT sensing at this stop & sampling
            dIx_det, dIy_det = build_TT_sensing_detector(Dcs, Mdet, Zbasis, Pmask)

            #dI_det = bin_block_divisible(I_fine, pupil_center = dI, pupil_radius_pixels = Mdet//2, M, mode='mean' ) 
            dI_det = bin_to_detector(P0, dI, pupil_pixels_diam=Mdet)

            # Fit only TT columns (detector space):
            coeffs, res = fit_coeffs_against_columns(dI_det, [dIx_det, dIy_det], mask=np.isfinite(dI_det))
            c_tx, c_ty = coeffs
            results[Mdet]['img'].append(dI)
            results[Mdet]['img_det'].append(dI_det)
            results[Mdet]['stop'].append(Dcs)
            results[Mdet]['TTx'].append(c_tx)
            results[Mdet]['TTy'].append(c_ty)
            results[Mdet]['resid'].append(res)
            

    return results, phi_raw, phi_ho

# ---------- 5) Run sweep and plot ----------
coldstop_list = np.arange(1.0, 10.0, 2.0)  # DIAMETERS in λ/D to test
results, phi_raw, phi_ho = sweep_tt_aliasing_vs_stop(N_fine=N, coldstop_list=coldstop_list, pupil_px_opts=(6,12),noll_indices_remove=list(range(2,16)), scaling_factor=0.1)
# only removing TT from first order 
results_LO, phi_raw_LO, phi_ho_LO = sweep_tt_aliasing_vs_stop(N_fine=N, coldstop_list=coldstop_list, pupil_px_opts=(6,12),noll_indices_remove=list(range(2,4)), scaling_factor=0.1)

## intensity vs cold stop size for Naomi like input aberations 
idxs = range(len(coldstop_list)) #coldstop_list# [2,3,5,7]

pup_px = 6
## before detector binning 
img_list = [phi_ho] + [results[pup_px]['img'][ii] for ii in idxs] #results['img_det'][::2]
title_list = [r'$\phi$'] + [f"diam. = {coldstop_list[ii]}"+r"$\lambda/D$" for ii in idxs]
util.nice_heatmap_subplots(im_list=img_list, title_list=title_list)
plt.show()

## after detector binning 
img_list = [phi_ho] + [results[pup_px]['img_det'][ii] for ii in idxs] #results['img_det'][::2]
title_list = [r'$\phi$'] + [f"diam. = {coldstop_list[ii]}"+r"$\lambda/D$" for ii in idxs]
util.nice_heatmap_subplots(im_list=img_list, title_list=title_list)
plt.show()



fig, ax = plt.subplots(1, 2, figsize=(11,4), sharey=True)
for i, Mdet in enumerate((6,12)):
    ax[i].plot(results[Mdet]['stop'], results[Mdet]['TTx'], 'o-', label='TTx coeff')
    ax[i].plot(results[Mdet]['stop'], results[Mdet]['TTy'], 's--', label='TTy coeff')
    ax[i].set_xlabel(r'Cold-stop diameter [$\lambda/D$]')
    ax[i].set_title(f'{Mdet} px / pupil diameter')
    ax[i].grid(True, alpha=0.3)
ax[0].set_ylabel('Apparent Tip/Tilt (Aliased) [rad RMS]')
ax[1].legend()
plt.tight_layout()
plt.show()


#### WITH ALL OF THEM 
fig, ax = plt.subplots(1, 2, figsize=(11,4), sharey=True)
for i, Mdet in enumerate((6,12)):
    ax[i].plot(results[Mdet]['stop'], results[Mdet]['TTx'], 'o-', label='Tip (14 modes removed)')
    #ax[i].plot(results[Mdet]['stop'], results[Mdet]['TTy'], 'o--', label='Tilt (14 modes removed)')
    ax[i].plot(results_LO[Mdet]['stop'], results_LO[Mdet]['TTx'], 'g-', label='Tip (2 modes removed)')
    #ax[i].plot(results_LO[Mdet]['stop'], results_LO[Mdet]['TTy'], 'g--', label='Tilt (2 modes removed)')
    ax[i].set_xlabel(r'Cold-stop diameter [$\lambda/D$]')
    ax[i].set_title(f'{Mdet} px / pupil diameter')
    ax[i].grid(True, alpha=0.3)
ax[0].set_ylabel('Apparent Tip/Tilt (Aliased) [rad RMS]')
ax[1].legend()
plt.tight_layout()
plt.show()




# Optional: residuals vs stop
plt.figure(figsize=(5.2,3.6))
for Mdet in (6,12):
    plt.plot(results[Mdet]['stop'], results[Mdet]['resid'], marker='o', label=f'{Mdet} px/diam')
plt.xlabel(r'$D_{\rm cs}$ [$\lambda/D$]')
plt.ylabel('Detector-space fit residual (TT only)')
plt.grid(True, alpha=0.3)
plt.legend()
plt.tight_layout()
plt.show()

# Optional: show the HO phase screen before/after removing 14 Zernikes
util.nice_heatmap_subplots(
    im_list=[phi_raw*P, phi_ho*P],
    title_list=['Raw Kolmogorov screen (RMS~1 rad)', 'After removing Z2..Z15 & scaled to 0.3 rad RMS'],
    cbar_label_list=['phase [arb]', 'phase [arb]'],
    fontsize=13, cbar_orientation='bottom', axis_off=True, vlims=None, savefig=None
)
plt.show()


### END CURRENT 

##########################################################################
# #########################################################################
# # From previous 
# #########################################################################

# import numpy as np
# import matplotlib.pyplot as plt
# from pyBaldr import utilities as util
# from common import DM_basis_functions as dmbasis
# from xaosim.shmlib import shm

# """
# What happens to a high order aberration that is aliased into the control range in the case of it being fully or half transmitted through the cold stop? Does the simulation framework deal with this properly (e.g. subsampling pixels adequately, and showing that you do by changing the subsampling and seeing the results change).
# """
# # dictionary with depths referenced for beam 2 (1-5 goes big to small)
# phasemask_parameters = {  
#                         "J5": {"depth":0.474 ,  "diameter":32},
#                         "J4": {"depth":0.474 ,  "diameter":36}, 
#                         "J3": {"depth":0.474 ,  "diameter":44}, 
#                         "J2": {"depth":0.474 ,  "diameter":54},
#                         "J1": {"depth":0.474 ,  "diameter":65},
#                         "H1": {"depth":0.654 ,  "diameter":68},  
#                         "H2": {"depth":0.654 ,  "diameter":53}, 
#                         "H3": {"depth":0.654 ,  "diameter":44}, 
#                         "H4": {"depth":0.654 ,  "diameter":37},
#                         "H5": {"depth":0.654 ,  "diameter":31}
#                         }

# """
# email from Mike 5/12/24 ("dichroic curves")
# optically you have 1380-1820nm (50% points) optically, 
# and including the atmosphere it is ~1420-1820nm. 

# """


# T = 1900 #K lab thermal source temperature 
# lambda_cut_on, lambda_cut_off =  1.38, 1.82 # um
# wvl = util.find_central_wavelength(lambda_cut_on, lambda_cut_off, T) # central wavelength of Nice setup
# mask = "J5"
# F_number = 21.2
# coldstop_diam = 4.04 #according to calc in thesis 8.07 lmabda/D bright, 4.04 lambda/D faint
# mask_diam = 1.22 * F_number * wvl / phasemask_parameters[mask]['diameter']
# eta = 138 / 1800 #0.647/4.82 #~= 1.1/8.2 (i.e. UTs) # ratio of secondary obstruction (UTs), secondary obstruction ATs 138 mm / 1800mm, (https://www.eso.org/sci/facilities/paranal/telescopes/vlti/subsystems/at/technic.html)


# ###################
 
# N = 2**9 + 1 # number of pixels

# x = np.linspace(-1, 1, N)
# X, Y = np.meshgrid(x, x, indexing='xy')
# P = (X**2 + Y**2) <= 1  # circular pupil mask
# basis = dmbasis.zernike_basis(nterms=120, npix=N, rho=None, theta=None) #dmbasis.develop_Fourier_basis( n=10, m=10 ,P = 120, Nx =N, Ny = N) #for k in basis:basis[k] *= P
# basis = [np.nan_to_num(b) for b in basis]
# # x = np.linspace(-1, 1, N)
# # X, Y = np.meshgrid(x, x, indexing='xy')
# # P = (X**2 + Y**2) <= 1  # circular pupil mask
# # tip  = X * P
# # tilt = Y * P
# # # Normalize each to 1 rad RMS within the pupil
# # tip  /= np.sqrt(np.mean(tip[P]**2))
# # tilt /= np.sqrt(np.mean(tilt[P]**2))

# # R2 = X**2 + Y**2
# # focus = (2.0*R2 - 1.0) * P          # Zernike-like defocus (∝ Z4)

# # # Remove tiny numerical piston/tilt leakage, then normalize to 1 rad RMS
# # focus -= focus[P].mean()
# # focus -= ((focus[P]*tip[P]).mean()  / (tip[P]**2).mean())  * tip
# # focus -= ((focus[P]*tilt[P]).mean() / (tilt[P]**2).mean()) * tilt
# # focus /= np.sqrt(np.mean(focus[P]**2))

# # basis = [tip,tilt,focus]
# ###################

# """
# before building reconstructors etc, just have some higher order modes that have spatial frequencies around 1/coldstop daim and also < 1/pixel_scale. See 
# binned image (I-I0) and see if we can visually see aliasing to lower order modes. 

# After we have some intuition around this build the reconstructors and show how we get cross coupling/modal leakage (look at covariance?)
# """

# # np.sum( tip**2*P ) / np.sum(P**2) == 1
# P, Ic = util.get_theoretical_reference_pupils_with_aber( wavelength = wvl ,
#                                               F_number = F_number , 
#                                               mask_diam = mask_diam, 
#                                               coldstop_diam=10, #coldstop_diam (lambda/D),
#                                               coldstop_misalign = [0,0], #lambda/D units
#                                               eta = eta, 
#                                               phi = basis[1], #+ -0.3 * basis[2] ,
#                                               diameter_in_angular_units = True, 
#                                               get_individual_terms=False, 
#                                               phaseshift = util.get_phasemask_phaseshift(wvl=wvl, depth = phasemask_parameters[mask]['depth'], dot_material='N_1405') , 
#                                               padding_factor = 6, 
#                                               debug= False, 
#                                               analytic_solution = False )

# ############################################
# ## Plot theoretical intensities on fine grid 
# imgs = [abs(P), Ic]
# titles=['Clear Pupil', 'ZWFS Pupil']
# cbars = ['Intensity', 'Intensity']
# xlabel_list, ylabel_list = ['',''], ['','']
# util.nice_heatmap_subplots(im_list=imgs ,
#                             xlabel_list=xlabel_list, 
#                             ylabel_list=ylabel_list, 
#                             title_list=titles, 
#                             cbar_label_list=cbars, 
#                             fontsize=15, 
#                             cbar_orientation = 'bottom', 
#                             axis_off=True, 
#                             vlims=None, 
#                             savefig='delme.png')
# plt.show()

# ############################################
# ## Plot theoretical intensities on CRED1 Detector (12 pixel diameter)
# # we can use a clear pupil measurement to interpolate this onto 
# # the measured pupil pixels.

# # Original grid dimensions from the theoretical pupil
# M, N = Ic.shape

# m, n = 36, 36  # New grid dimensions (width, height in pixels)
# # To center the pupil, set the center at half of the grid size.
# x_c, y_c = int(m/2), int(n/2)
# # For a 12-pixel diameter pupil, the new pupil radius should be 6 pixels.
# new_radius = 3

# # Interpolate the theoretical intensity onto the new grid.
# detector_intensity = util.interpolate_pupil_to_measurement(abs(P), abs(Ic), M, N, m, n, x_c, y_c, new_radius)

# # Plot the interpolated theoretical pupil intensity.
# imgs = [detector_intensity]
# titles=[ 'Detected\nZWFS Pupil']
# cbars = ['Intensity']
# xlabel_list, ylabel_list = [''], ['']
# util.nice_heatmap_subplots(im_list=imgs ,
#                             title_list=titles,
#                             xlabel_list=xlabel_list, 
#                             ylabel_list=ylabel_list, 
#                             cbar_label_list=cbars, 
#                             fontsize=15, 
#                             cbar_orientation = 'bottom', 
#                             axis_off=True, 
#                             vlims=None, 
#                             savefig='delme2.png')
# plt.show()


# ########## BUILD TT RECONSTRUCTOR WITH ALIGNED SYSTEM 
# # ----- Settings for the aligned system & linearization -----
# aligned_misalign = (0.0, 0.0)   # cold stop aligned (in wvl/D)
# use_coldstop_diam = coldstop_diam  # or set to None to disable the cold stop in this step
# epsilon = 0.01                  # small modal poke [rad RMS] for finite differences

# # Detector geometry (you already defined these above)
# m, n = 36, 36
# x_c, y_c = int(m/2), int(n/2)
# new_radius = 3                  # pupil radius (px) on detector => diameter = 6 px

# # ----- Helper: downsample (same call pattern you used) -----
# def to_detector(P_hr, I_hr):
#     M_hr, N_hr = I_hr.shape
#     return util.interpolate_pupil_to_measurement(np.abs(P_hr), np.abs(I_hr),
#                                                  M_hr, N_hr, m, n, x_c, y_c, new_radius)

# # ----- Build a detector-plane pupil mask (boolean) -----
# # Use a clear pupil intensity as a proxy to locate detector pixels that are "inside the pupil"
# P0_hr, Ic0_hr = util.get_theoretical_reference_pupils_with_aber(
#     wavelength=wvl, F_number=F_number,
#     mask_diam=mask_diam, coldstop_diam=use_coldstop_diam,
#     coldstop_misalign=aligned_misalign, eta=eta,
#     phi=np.zeros_like(basis[0]), diameter_in_angular_units=True,
#     phaseshift=util.get_phasemask_phaseshift(wvl=wvl, depth=phasemask_parameters[mask]['depth'], dot_material='N_1405'),
#     padding_factor=6, analytic_solution=False, debug=False
# )
# det_P = to_detector(P0_hr, P0_hr)       # map clear pupil to detector
# det_ref = to_detector(P0_hr, Ic0_hr)    # reference ZWFS intensity on detector (aligned)

# # Threshold to define valid detector pixels (inside the pupil support)
# det_mask = det_P > (0.5 * det_P.max())
# pix_idx = np.where(det_mask.ravel())[0]   # vectorization indices over pupil pixels
# n_pix = pix_idx.size

# # ----- Build interaction matrix A (n_pix × n_modes) -----
# modes = [tip, tilt]            # 1 rad RMS each (from your earlier basis)
# mode_names = ["tip", "tilt"]
# n_modes = len(modes)
# A = np.zeros((n_pix, n_modes))

# for j, mode in enumerate(modes):
#     # Plus poke
#     _, Ic_plus_hr = util.get_theoretical_reference_pupils_with_aber(
#         wavelength=wvl, F_number=F_number,
#         mask_diam=mask_diam, coldstop_diam=use_coldstop_diam,
#         coldstop_misalign=aligned_misalign, eta=eta,
#         phi=+epsilon * mode, diameter_in_angular_units=True,
#         phaseshift=util.get_phasemask_phaseshift(wvl=wvl, depth=phasemask_parameters[mask]['depth'], dot_material='N_1405'),
#         padding_factor=6, analytic_solution=False, debug=False
#     )
#     det_plus = to_detector(P0_hr, Ic_plus_hr)

#     # Minus poke
#     _, Ic_minus_hr = util.get_theoretical_reference_pupils_with_aber(
#         wavelength=wvl, F_number=F_number,
#         mask_diam=mask_diam, coldstop_diam=use_coldstop_diam,
#         coldstop_misalign=aligned_misalign, eta=eta,
#         phi=-epsilon * mode, diameter_in_angular_units=True,
#         phaseshift=util.get_phasemask_phaseshift(wvl=wvl, depth=phasemask_parameters[mask]['depth'], dot_material='N_1405'),
#         padding_factor=6, analytic_solution=False, debug=False
#     )
#     det_minus = to_detector(P0_hr, Ic_minus_hr)

#     # Finite-difference derivative (remove any DC/piston leakage inside the pupil)
#     resp = (det_plus - det_minus) / (2.0 * epsilon)
#     resp -= resp[det_mask].mean()

#     # Vectorize over pupil pixels
#     A[:, j] = resp.ravel()[pix_idx]

# # Optional: visualize the two columns of A (reshape back to detector)
# # util.nice_heatmap_subplots([...])

# # ----- Regularized reconstructor R -----
# # R maps detector residuals (vectorized over pupil pixels) to modal estimates [tip, tilt]
# # Use SVD Tikhonov regularization with a gentle cutoff
# U, s, Vh = np.linalg.svd(A, full_matrices=False)
# alpha = 1e-3 * s.max()                 # tune as needed; smaller = less regularization
# s_filt = s / (s**2 + alpha**2)
# R = (Vh.T * s_filt) @ U.T              # shape: (n_modes × n_pix)

# # Precompute once after building A, R, det_mask, pix_idx, det_ref:
# C = R @ A                       # cross-talk / gain matrix (n_modes x n_modes)
# C_inv = np.linalg.inv(C)        # or use np.linalg.pinv(C) if you prefer robustness

# def residual_vector(Ic_meas_hr):
#     """High-res intensity -> vectorized detector residual (pupil pixels only)."""
#     det_meas = to_detector(P0_hr, Ic_meas_hr)
#     y = det_meas - det_ref
#     y -= y[det_mask].mean()
#     return y.ravel()[pix_idx]

# def project_to_modes(Ic_meas_hr):
#     """
#     Returns modal amplitudes in calibration units (rad RMS if you poked in rad RMS).
#     For tip/tilt-only calibration, returns [tip, tilt].
#     """
#     yvec = residual_vector(Ic_meas_hr)
#     z = R @ yvec             # reconstructor output (mixed / gain-scaled)
#     m = C_inv @ z            # demix & de-gain -> modal amplitudes
#     return m.tolist()

# # 
# # ----- How to use the reconstructor -----
# # Given a new detector measurement det_meas (aligned configuration),
# # form the residual w.r.t. the reference and estimate [tip, tilt]:
# #   det_meas = to_detector(P0_hr, Ic_meas_hr)
# #   y = (det_meas - det_ref)
# #   y -= y[det_mask].mean()
# #   m_hat = R @ y.ravel()[pix_idx]
# # where m_hat[0] ~ tip (rad RMS), m_hat[1] ~ tilt (rad RMS) in the calibration units.



# #################################################################################
# ############################ ERROR SIGNAL CHECK 
# #################################################################################

# # -------- TIP ramp test: input +/-0.5 waves (+/-pi rad) and reconstruction --------
# # Assumes: project_to_modes(), to_detector(), P0_hr, det_ref, and all calibration params exist.

# # Sweep tip amplitude in WAVES (RMS scaling w.r.t. your tip basis which is 1 rad RMS)
# tip_waves = np.linspace(-0.5, 0.5, 10)    # +/-1/2 lambda
# tip_rad   = 2*np.pi * tip_waves            # convert to radians

# tip_true = []
# tip_rec  = []
# tilt_rec = []

# for a in tip_rad:
#     # Inject pure TIP (RMS-scaled since 'tip' basis is 1 rad RMS)
#     phi_in = a * tip

#     # Aligned system propagation (same params as calibration!)
#     _, Ic_hr = util.get_theoretical_reference_pupils_with_aber(
#         wavelength=wvl, F_number=F_number,
#         mask_diam=mask_diam, coldstop_diam=coldstop_diam,
#         coldstop_misalign=(0.0, 0.0), eta=eta,
#         phi=phi_in, diameter_in_angular_units=True,
#         phaseshift=util.get_phasemask_phaseshift(wvl=wvl, depth=phasemask_parameters[mask]['depth'], dot_material='N_1405'),
#         padding_factor=6, analytic_solution=False, debug=False
#     )

#     # Reconstruct [tip, tilt] in rad (RMS units matching calibration)
#     m_hat = project_to_modes(Ic_hr)   # -> [tip_est, tilt_est]
#     tip_true.append(a)
#     tip_rec.append(m_hat[0])
#     tilt_rec.append(m_hat[1])

# tip_true = np.array(tip_true)
# tip_rec  = np.array(tip_rec)
# tilt_rec = np.array(tilt_rec)

# # Errors (radians)
# tip_err  = tip_rec  - tip_true
# tilt_err = tilt_rec - 0.0

# # Quick numeric summary
# print("TIP sweep (±0.5 waves):")
# print(f"  TIP gain   ~ {np.polyfit(tip_true, tip_rec, 1)[0]:.4f} (slope rec vs true)")
# print(f"  TIP offset ~ {np.polyfit(tip_true, tip_rec, 1)[1]:.4e} rad")
# print(f"  TIP RMSE   ~ {np.sqrt(np.mean(tip_err**2)):.4e} rad")
# print(f"  TILT crosstalk (RMS) ~ {np.sqrt(np.mean(tilt_rec**2)):.4e} rad")

# # Optional plots
# plt.figure(figsize=(6,4))
# plt.plot(2 * np.pi/wvl *tip_true/(2*np.pi), 2 * np.pi/wvl * tip_rec/(2*np.pi), 'o-', label='Reconstructed tip')
# plt.plot(2 * np.pi/wvl *tip_true/(2*np.pi), 2 * np.pi/wvl *tip_true/(2*np.pi), 'k--', label='1:1')
# plt.xlabel('Input tip (rad)',fontsize=15) # wave (tip_* is in wave)
# plt.ylabel('Reconstructed tip (rad)',fontsize=15) # wave 
# plt.title('ZWFS Tip Reconstruction vs Input (Aligned, faint mode)',fontsize=15)
# plt.grid(True, alpha=0.3)
# plt.legend(fontsize=15)
# plt.gca().tick_params(labelsize=15)
# plt.tight_layout()
# plt.ylim([-1,1])
# #plt.xlim([-1,1])
# plt.show()


# plt.figure(figsize=(6,4))
# plt.plot(tip_true/(2*np.pi), tip_err, 'o-')
# plt.xlabel('Input tip (waves)')
# plt.ylabel('Tip error (rad)')
# plt.title('Tip Reconstruction Error')
# plt.grid(True, alpha=0.3)
# plt.tight_layout()
# plt.show()

# plt.figure(figsize=(6,4))
# plt.plot(tip_true/(2*np.pi), tilt_rec, 'o-', label='Tilt cross-talk')
# plt.xlabel('Input tip (waves)')
# plt.ylabel('Reconstructed tilt (rad)')
# plt.title('Tilt Cross-talk vs Tip Input')
# plt.grid(True, alpha=0.3)
# plt.tight_layout()
# plt.show()


# #################################################################################
# # Tip / Tilt error vs cold-stop misalignment (zero input aberrations)
# # Sweep misalignment along +x from 0 to 3 (wvl/D), measure reconstructed tip/tilt (rad)
# #################################################################################
# # zero input aberrations
# # misalign cold stop by 0-3 lambda /D 
# # for each misalignment measure the tip , tilt error signal 
# # plot error signam (y) units of radians rms vs cold stop mis-alignment x (lambda/D units)


# # Sweep settings
# misalign_grid = np.linspace(0.0, 3.0, 16)   # in wvl/D (0 → 3)
# misalign_axis = (1.0, 0.0)                  # shift along +x; set (0,1) for +y

# tip_est = []
# tilt_est = []

# for d in misalign_grid:
#     dx_wvld = d * misalign_axis[0]
#     dy_wvld = d * misalign_axis[1]

#     # ZERO input aberrations
#     phi_in = np.zeros_like(basis[0])

#     # Propagate with MISALIGNED cold stop
#     _, Ic_hr = util.get_theoretical_reference_pupils_with_aber(
#         wavelength=wvl, F_number=F_number,
#         mask_diam=mask_diam, coldstop_diam=coldstop_diam,
#         coldstop_misalign=(dx_wvld, dy_wvld),
#         eta=eta, phi=phi_in, diameter_in_angular_units=True,
#         phaseshift=util.get_phasemask_phaseshift(
#             wvl=wvl, depth=phasemask_parameters[mask]['depth'], dot_material='N_1405'
#         ),
#         padding_factor=6, analytic_solution=False, debug=False
#     )

#     # Reconstruct [tip, tilt] (rad, RMS units of your calibration)
#     m_hat = project_to_modes(Ic_hr)  # -> [tip_est, tilt_est]
#     tip_est.append(m_hat[0])
#     tilt_est.append(m_hat[1])

# tip_est  = np.array(tip_est)
# tilt_est = np.array(tilt_est)

# # Quick text summary
# print("Cold-stop misalignment sweep (0 - 3 wvl/D):")
# print(f"  Tip range:  [{tip_est.min():.3e}, {tip_est.max():.3e}] rad")
# print(f"  Tilt range: [{tilt_est.min():.3e}, {tilt_est.max():.3e}] rad")

# # Plots
# plt.figure(figsize=(6,4))
# plt.plot(misalign_grid, tip_est, 'o-', label='TIP (rad)')
# plt.plot(misalign_grid, tilt_est, 's-', label='TILT (rad)')
# plt.xlabel('Cold-stop misalignment (wvl/D)')
# plt.ylabel('Reconstructed modal error (rad RMS)')
# plt.title('ZWFS: Tip/Tilt vs Cold-stop Misalignment (zero input aberrations)')
# plt.grid(True, alpha=0.3)
# plt.legend()
# plt.tight_layout()
# plt.show()




# #################################################################################
# # Tip vs cold-stop misalignment for three input regimes:
# #   (1) zero aberrations
# #   (2) +0.5 rad RMS defocus
# #   (3) -0.5 rad RMS defocus
# #################################################################################

# misalign_grid = np.linspace(0.0, 3.0, 16)   # wvl/D
# misalign_axis = (1.0, 0.0)                  # shift along +x
# defocus_levels = [0.0, +0.5, -0.5]          # rad RMS on your normalized 'focus' basis

# tip_curves = {}

# for kappa in defocus_levels:
#     tips = []
#     for d in misalign_grid:
#         dx_wvld = d * misalign_axis[0]
#         dy_wvld = d * misalign_axis[1]

#         # Input aberration: kappa * focus (focus is 1 rad RMS-normalized)
#         phi_in = kappa * focus

#         # Propagate with misaligned cold stop
#         _, Ic_hr = util.get_theoretical_reference_pupils_with_aber(
#             wavelength=wvl, F_number=F_number,
#             mask_diam=mask_diam, coldstop_diam=coldstop_diam,
#             coldstop_misalign=(dx_wvld, dy_wvld),
#             eta=eta, phi=phi_in, diameter_in_angular_units=True,
#             phaseshift=util.get_phasemask_phaseshift(
#                 wvl=wvl, depth=phasemask_parameters[mask]['depth'], dot_material='N_1405'
#             ),
#             padding_factor=6, analytic_solution=False, debug=False
#         )

#         # Reconstruct [tip, tilt]; keep only TIP
#         tip_est, tilt_est = project_to_modes(Ic_hr)
#         tips.append(tip_est)

#     tip_curves[kappa] = np.array(tips)

# # ---- Plot: TIP vs misalignment for the three regimes ----
# plt.figure(figsize=(7,4.5))
# labels = {
#     0.0:   "focus = 0.0 rad",
#     +0.5:  "focus = +0.5 rad",
#     -0.5:  "focus = -0.5 rad",
# }
# for kappa, vals in tip_curves.items():
#     plt.plot(misalign_grid, vals, 'o-', label=labels[kappa])

# plt.xlabel('Cold-stop misalignment (wvl/D)',fontsize=15)
# plt.ylabel('Reconstructed TIP (rad RMS)',fontsize=15)
# plt.title('TIP vs Cold-stop Misalignment\n(0, +0.5, -0.5 rad RMS defocus)',fontsize=15)
# plt.grid(True, alpha=0.3)
# plt.gca().tick_params(labelsize=15)
# plt.legend(fontsize=15)
# plt.tight_layout()
# plt.show()







# #################################################################################
# #################################################################################
# # ERROR SIGNAL CHECK with different misalignments
# # Scenarios:
# #  A) aligned: no aberration, no misalignment
# #  B) 0.5 (wvl/D) cold-stop misalignment, no aberration (which in baldr phasemask corresponds to ~15um drift on 1 lambda/D phasmask with f# 21.2 system)
# #  C) 0.5 (wvl/D) cold-stop misalignment + 0.5 rad RMS defocus (130nm RMS focus offset at 1.65um)
# #################################################################################


# tip_waves = np.linspace(-1.5, 1.5, 25)    # ±1/2 lambda
# tip_rad   = 2*np.pi * tip_waves           # radians

# scenarios = {
#     "A: aligned (0 wvl/D, no defocus)" : dict(misalign=(0.0, 0.0), kappa_defocus=0.0),
#     "B: 0.5 wvl/D, no defocus"         : dict(misalign=(0.5, 0.0), kappa_defocus=0.0),
#     "C: 0.5 wvl/D, +0.5 rad defocus"   : dict(misalign=(0.5, 0.0), kappa_defocus=0.5),
# }

# results = {}

# for label, cfg in scenarios.items():
#     print(f"\nLOOKING AT {label}\n")
#     tip_true, tip_rec = [], []

#     for a in tip_rad:
#         # Input phase: tip (±1/2 wave) plus optional defocus (rad RMS)
#         phi_in = a * tip + cfg["kappa_defocus"] * focus

#         # Propagate with specified cold-stop misalignment
#         _, Ic_hr = util.get_theoretical_reference_pupils_with_aber(
#             wavelength=wvl, F_number=F_number,
#             mask_diam=mask_diam, coldstop_diam=coldstop_diam,
#             coldstop_misalign=cfg["misalign"],
#             eta=eta, phi=phi_in, diameter_in_angular_units=True,
#             phaseshift=util.get_phasemask_phaseshift(
#                 wvl=wvl, depth=phasemask_parameters[mask]['depth'], dot_material='N_1405'
#             ),
#             padding_factor=6, analytic_solution=False, debug=False
#         )

#         # Reconstruct [tip, tilt]; keep TIP only
#         m_hat = project_to_modes(Ic_hr)   # -> [tip_est, tilt_est]
#         tip_true.append(a)
#         tip_rec.append(m_hat[0])

#     tip_true = np.array(tip_true)
#     tip_rec  = np.array(tip_rec)
#     tip_err  = tip_rec - tip_true

#     results[label] = dict(
#         tip_true=tip_true,
#         tip_rec=tip_rec,
#         tip_err=tip_err,
#         gain=np.polyfit(tip_true, tip_rec, 1)[0],
#         offset=np.polyfit(tip_true, tip_rec, 1)[1],
#         rmse=np.sqrt(np.mean(tip_err**2))
#     )

# # ---------- Plots ----------
# # Reconstructed tip vs input (in waves)
# fs = 15
# plt.figure(figsize=(7.5,4.8))
# for label, dat in results.items():
#     plt.plot(dat["tip_true"]/(2*np.pi), dat["tip_rec"]/(2*np.pi), 'o-', label=label)
# plt.plot(2*np.pi/wvl * tip_waves, 2*np.pi/wvl * tip_waves, 'k--', lw=1, label='Ideal y=x')
# plt.xlabel('Input TIP (rad)',fontsize=fs)
# plt.ylabel('Reconstructed TIP (rad)',fontsize=fs)
# plt.title('TIP Reconstruction vs Input\n(aligned vs misaligned/defocus cases)',fontsize=fs)
# plt.grid(True, alpha=0.3)
# plt.legend(fontsize=fs)
# plt.gca().tick_params(labelsize=fs)
# plt.tight_layout()
# plt.xlim([-1.5,1.5])
# plt.ylim([-0.3,0.3])
# plt.show()

# # Error (recon − true) in radians vs input (waves)
# plt.figure(figsize=(7.5,4.2))
# for label, dat in results.items():
#     plt.plot(dat["tip_true"]/(2*np.pi), dat["tip_err"], 'o-', label=label)
# plt.axhline(0, color='k', lw=1)
# plt.xlabel('Input TIP (waves)')
# plt.ylabel('TIP error (rad)')
# plt.title('TIP Error vs Input')
# plt.grid(True, alpha=0.3)
# plt.legend()
# plt.tight_layout()
# plt.show()

# # ---------- Text summary ----------
# for label, dat in results.items():
#     print(f"{label}: gain={dat['gain']:.4f}, offset={dat['offset']:.3e} rad, RMSE={dat['rmse']:.3e} rad")



# ################
# # Plot all of them 

# fig, axes = plt.subplots(2, 5, figsize=(15, 6))
# axes = axes.flatten()  # Flatten to iterate easily

# # Loop over each phasemask and generate synthetic intensity data
# for i, (mask, params) in enumerate(phasemask_parameters.items()):
#     mask_diam = 1.22 * F_number * wvl / params['diameter']  # Compute mask diameter
#     phase_shift = util.get_phasemask_phaseshift(wvl=wvl, depth = phasemask_parameters[mask]['depth'], dot_material='N_1405') , 
    
#     P, Ic = util.get_theoretical_reference_pupils( wavelength = wvl ,
#                                                 F_number = F_number , 
#                                                 mask_diam = mask_diam, 
#                                                 coldstop_diam=coldstop_diam,
#                                                 eta = eta, 
#                                                 diameter_in_angular_units = True, 
#                                                 get_individual_terms=False, 
#                                                 phaseshift = util.get_phasemask_phaseshift(wvl=wvl, depth = phasemask_parameters[mask]['depth'], dot_material='N_1405') , 
#                                                 padding_factor = 6, 
#                                                 debug= False, 
#                                                 analytic_solution = False )

#     detector_intensity = util.interpolate_pupil_to_measurement(P, Ic, M, N, m, n, x_c, y_c, new_radius)

#     # Plot the results
#     im = axes[i].imshow(detector_intensity, cmap='inferno')
#     axes[i].set_title(mask, fontsize=20)
#     axes[i].axis('off')

# # Adjust layout and add colorbar
# fig.subplots_adjust(right=0.85)
# cbar_ax = fig.add_axes([0.88, 0.15, 0.02, 0.7])
# fig.colorbar(im, cax=cbar_ax, label='Intensity')

# # Show the final figure
# plt.suptitle("ZWFS Theoretical Intensities on CRED1 Detector", fontsize=14)
# plt.show()





# #################################################################################
# #################################################################################
# # ---------------- SIM: TIP RMS from focus white-noise (0.5 rad RMS) ----------------
# ## we have with a static focus offset of 0.5 rad (130nm RMS at wvl =1.65um) and record ~ 0.2 radian Tip error (with no other aberrations in the system) - plot attached . Therefore if we have a time series of white gaussian noise focus with   0.5 rad rms, would this correspond to 0.2 radian tip rms error signal?

# #################################################################################
# #################################################################################
# Nmc = 100
# sigma_focus_rad = 0.5            # focus noise RMS [rad]
# misalign = (0.5, 0.0)            # 0.5 (wvl/D) cold-stop misalignment along +x
# rng = np.random.default_rng(2025)

# def run_with_focus_amp(kappa_rad):
#     phi_in = kappa_rad * focus    # focus basis is 1 rad RMS-normalized
#     _, Ic_hr = util.get_theoretical_reference_pupils_with_aber(
#         wavelength=wvl, F_number=F_number,
#         mask_diam=mask_diam, coldstop_diam=coldstop_diam,
#         coldstop_misalign=misalign, eta=eta, phi=phi_in,
#         diameter_in_angular_units=True,
#         phaseshift=util.get_phasemask_phaseshift(
#             wvl=wvl, depth=phasemask_parameters[mask]['depth'], dot_material='N_1405'
#         ),
#         padding_factor=6, analytic_solution=False, debug=False
#     )
#     tip_est, tilt_est = project_to_modes(Ic_hr)
#     return tip_est

# # Draw a zero-mean white Gaussian focus series with the desired RMS
# focus_series = sigma_focus_rad * rng.standard_normal(Nmc)

# # Run the model & reconstruct TIP for each sample
# tip_series = np.array([run_with_focus_amp(kappa) for kappa in focus_series])

# # Remove any DC bias (optional; keeps “RMS of fluctuations”)
# tip_series -= tip_series.mean()

# # Report RMS in radians and waves
# tip_rms_rad = np.sqrt(np.mean(tip_series**2))
# tip_rms_waves = tip_rms_rad / (2*np.pi)
# print(f"N={Nmc}, focus noise RMS = {sigma_focus_rad:.3f} rad")
# print(f"TIP RMS = {tip_rms_rad:.4e} rad  ({tip_rms_waves:.4e} waves)")

# # Optional: quick look

# plt.figure(figsize=(6.0,3.2))
# plt.plot(tip_series, 'o-', ms=4)
# plt.axhline(+tip_rms_rad, color='k', ls='--', lw=1)
# plt.axhline(-tip_rms_rad, color='k', ls='--', lw=1)
# plt.title('TIP from 0.5 rad RMS focus noise\n(0.5 wvl/D cold-stop misalignment)')
# plt.xlabel('sample'); plt.ylabel('TIP (rad)')
# plt.tight_layout(); plt.show()



# # =============================================================================
# # Gain margin vs cold-stop misalignment for a basic integrator with latency
# #
# # We linearize the closed-loop measurement around zero modal command and
# # estimate the small-signal plant P(Δ) mapping commanded [tip, tilt] (rad RMS)
# # to reconstructed [tip, tilt] (rad RMS), for a given cold-stop misalignment Δ.
# #
# # Delay-limited stability bound (continuous-time heuristic):
# #   g_max  ≈  (π / (2 τ)) * 1 / |G|
# # where τ = m * T_s is total loop latency, and |G| is:
# #   - SISO: |G_tip| = |∂t̂/∂t|       (tip loop only)
# #   - MIMO:  λ_max(P)               (dominant eigen/singular value of P)
# # =============================================================================

# # ---------- User inputs ----------
# # Sampling and latency (EDIT these for your system)
# T_s = 1/1000.0        # [s] WFS/RTC sample time (e.g. 1 kHz)
# m_delay = 2           # [frames] total integer-frame delay (exposure+RTC+DM)
# tau = m_delay * T_s   # [s] total latency

# # Misalignment sweep (wvl/D); use x-axis only here
# misalign_grid = np.linspace(0.0, 3.0, 13)   # 0 → 3 (wvl/D)

# # Small poke size for finite differences (rad RMS)
# epsilon = 0.01

# # Which cold-stop diameter/configuration to analyze
# use_coldstop_diam = coldstop_diam           # or set None if you want it off

# # ---------- Helper: run the pipeline with a given modal command and misalignment ----------
# def run_modal_command(cmd_tip_rad, cmd_tilt_rad, misalign_wvld):
#     """Apply commanded [tip, tilt] in rad RMS on the high-res pupil and return reconstructed [tip, tilt]."""
#     phi_in = cmd_tip_rad * tip + cmd_tilt_rad * tilt # - 0.5 * basis[2]
#     _, Ic_hr = util.get_theoretical_reference_pupils_with_aber(
#         wavelength=wvl, F_number=F_number,
#         mask_diam=mask_diam, coldstop_diam=use_coldstop_diam,
#         coldstop_misalign=(misalign_wvld, 0.0),   # shift along +x
#         eta=eta, phi=phi_in, diameter_in_angular_units=True,
#         phaseshift=util.get_phasemask_phaseshift(
#             wvl=wvl, depth=phasemask_parameters[mask]['depth'], dot_material='N_1405'
#         ),
#         padding_factor=6, analytic_solution=False, debug=False
#     )
#     t_hat, r_hat = project_to_modes(Ic_hr)   # [tip, tilt] in rad (RMS units)
#     return np.array([t_hat, r_hat])

# # ---------- Main sweep ----------
# G_tip_list = []      # |∂t̂/∂t|
# lam_max_list = []    # λ_max(P)  (dominant eigenvalue/sigma)
# x_talk_list = []     # cross-coupling ratio |∂t̂/∂r| / |∂t̂/∂t|

# for d in misalign_grid:
#     # Finite-difference slopes w.r.t. tip
#     t_plus  = run_modal_command(+epsilon, 0.0, d)
#     t_minus = run_modal_command(-epsilon, 0.0, d)
#     dt = (t_plus - t_minus) / (2*epsilon)   # columns for 'tip' excitation

#     # Finite-difference slopes w.r.t. tilt
#     r_plus  = run_modal_command(0.0, +epsilon, d)
#     r_minus = run_modal_command(0.0, -epsilon, d)
#     dr = (r_plus - r_minus) / (2*epsilon)   # columns for 'tilt' excitation

#     # Small-signal 2x2 plant P(Δ): [t̂; r̂] = P * [t; r]
#     P_delta = np.column_stack([dt, dr])     # shape (2,2)

#     # SISO tip gain magnitude
#     G_tip = np.abs(P_delta[0, 0])           # |∂t̂/∂t|
#     G_tip_list.append(G_tip)

#     # Cross-talk indicator (optional diagnostic)
#     x_talk = np.abs(P_delta[0, 1]) / (G_tip + 1e-16)
#     x_talk_list.append(x_talk)

#     # Dominant eigen/singular value for MIMO bound
#     # (For real 2x2, spectral norm = largest singular value)
#     lam_max = np.linalg.svd(P_delta, compute_uv=False)[0]
#     lam_max_list.append(lam_max)

# G_tip_arr = np.array(G_tip_list)
# lam_max_arr = np.array(lam_max_list)
# x_talk_arr = np.array(x_talk_list)

# # ---------- Delay-limited gain bounds ----------
# # g_max ≈ (π/(2 τ)) * 1/|G|
# const = np.pi / (2.0 * tau)
# gmax_tip  = const / (G_tip_arr + 1e-16)     # SISO (tip-only)
# gmax_mimo = const / (lam_max_arr + 1e-16)   # MIMO (dominant mode)

# # ---------- Report ----------
# print(f"Sample time T_s = {T_s*1e3:.1f} ms, delay m = {m_delay} frames -> τ = {tau*1e3:.1f} ms")
# for d, Gt, xk, lm, gt, gm in zip(misalign_grid, G_tip_arr, x_talk_arr, lam_max_arr, gmax_tip, gmax_mimo):
#     print(f"Δ={d:4.1f} wvl/D | |∂t̂/∂t|={Gt:.3f}  x-talk={xk:.3f}  λ_max={lm:.3f}  "
#           f"g_max^tip={gt:.2f}  g_max^mimo={gm:.2f}")

# # ---------- Plots ----------
# plt.figure(figsize=(7.2,4.6))
# plt.plot(misalign_grid, gmax_tip,  'o-', label='g_max (tip SISO)')
# plt.plot(misalign_grid, gmax_mimo, 's-', label='g_max (MIMO λ_max)')
# plt.xlabel('Cold-stop misalignment (wvl/D)')
# plt.ylabel('Max integrator gain (arb. units)')
# plt.title('Delay-limited gain bound vs cold-stop misalignment')
# plt.grid(True, alpha=0.3)
# plt.yscale('log')
# plt.legend()
# plt.tight_layout()
# plt.show()

# plt.figure(figsize=(7.2,3.6))
# plt.plot(misalign_grid, G_tip_arr, 'o-', label='|∂t̂/∂t|')
# #plt.plot(misalign_grid, x_talk_arr, 's--', label='|∂t̂/∂r| / |∂t̂/∂t|')
# plt.xlabel('Cold-stop misalignment (wvl/D)')
# plt.ylabel('Sensitivity' )# / cross-talk')
# plt.title('ZWFS small-signal sensitivities vs misalignment')
# plt.grid(True, alpha=0.3)
# plt.legend()
# plt.tight_layout()
# plt.show()





# # ========================== Describing-function analysis ==========================
# # Nonlinear measurement y = h(x): reconstructed TIP vs "true TIP" (commanded).
# # We build h(x) numerically under a chosen cold-stop misalignment and focus offset,
# # then compute the describing function N(A), and the delay-limited limit-cycle gain.

# """
# \paragraph{Nonlinear measurement and stability.}
# When the ZWFS measurement becomes nonlinear due to cold-stop misalignment and
# defocus, the error signal can be written as a static nonlinearity $y = h(x)$,
# where $x$ is the true modal phase (e.g.\ tip) and $y$ the reconstructed value.
# For the aligned case $h(x)\!\approx\!k x$, but misalignment causes saturation
# and slope reversals so that $\tfrac{dh}{dx}$ changes sign. The closed-loop system
# then forms a Lur'e feedback structure composed of a linear dynamic element $G(s)$
# (integrator + delay) and a static nonlinearity $h(x)$.
# Stability can be analysed using \emph{describing functions}
# \[
# N(A) = \frac{2}{\pi A} \int_0^\pi h(A\sin\theta)\sin\theta\, d\theta,
# \]
# which quantify the amplitude-dependent loop gain, or by \emph{sector-bounded}
# criteria (Circle or Popov tests) if $h(x)$ lies within a sector
# $[k_1,k_2]$. The multiple zero crossings observed imply $k_1<0$, so the
# effective feedback alternates between stabilising and destabilising, producing
# regions of limit-cycle or bistable behaviour even if the small-signal linearised
# loop is stable.
# """

# # --- Choose operating point (nonlinear curve will depend on these) ---
# misalign = (0.5, 0.0)   # cold-stop misalignment (wvl/D)
# f0_rad   = 0.5          # static defocus offset [rad RMS] (set 0 for aligned)
# # Latency model for the controller (integrator + pure delay)
# T_s    = 1/1000.0       # sample time [s]
# m_delay= 2              # frames of delay
# tau    = m_delay*T_s    # total delay [s]
# # Controller: pure integrator C(s)=g/s with delay e^{-s tau}
# # At the delay-limited phase crossover: ωc = π/(2 τ)
# omega_c = np.pi/(2.0*tau)

# # --- Build the static nonlinearity h(x): x -> hat_tip ---
# def reconstruct_tip_from_true_tip(x_true_rad):
#     """Return reconstructed TIP for a commanded true TIP = x_true_rad (rad RMS),
#        with the operating point (misalign, f0_rad)."""
#     phi_in = f0_rad*focus + x_true_rad*tip  # inject focus offset + true tip
#     _, Ic_hr = util.get_theoretical_reference_pupils_with_aber(
#         wavelength=wvl, F_number=F_number,
#         mask_diam=mask_diam, coldstop_diam=coldstop_diam,
#         coldstop_misalign=misalign, eta=eta, phi=phi_in,
#         diameter_in_angular_units=True,
#         phaseshift=util.get_phasemask_phaseshift(
#             wvl=wvl, depth=phasemask_parameters[mask]['depth'], dot_material='N_1405'
#         ),
#         padding_factor=6, analytic_solution=False, debug=False
#     )
#     tip_est, tilt_est = project_to_modes(Ic_hr)
#     return tip_est

# # --- Describing function N(A) for a static (possibly asymmetric) nonlinearity h(x) ---
# # For odd nonlinearities N(A) is real: N(A) = (2/(π A)) ∫_0^π h(A sinθ) sinθ dθ.
# # We’ll compute both the real part via that formula and the general complex DF
# # using the first-harmonic projection (robust to slight asymmetries).
# def describing_function(h_fun, A, nθ=4096):
#     θ = np.linspace(0, 2*np.pi, nθ, endpoint=False)
#     x = A*np.sin(θ)
#     y = np.array([h_fun(xi) for xi in x])
#     # Complex DF (first-harmonic): N = <y e^{-jθ}> / <A sinθ e^{-jθ}> = (1/πA)∫ y sinθ dθ  (imag cancels for odd)
#     # Use discrete projection on sin(θ):
#     Re = (1/np.pi/A) * np.trapz(y*np.sin(θ), θ)     # equals standard real DF for odd h
#     Im = (1/np.pi/A) * np.trapz(y*(-np.cos(θ)), θ)  # ~0 for odd h; kept for completeness
#     return Re + 1j*Im

# # --- Sweep amplitude and compute DF ---
# A_grid = np.linspace(0.01, 0.8, 40)  # [rad RMS] (choose range covering your plot's linear→nonlinear)
# N_vals = np.array([describing_function(reconstruct_tip_from_true_tip, A) for A in A_grid])

# # --- Predict limit-cycle gain for integrator+delay at ωc = π/(2τ) ---
# # Magnitude condition at that phase: |G(jωc)*N(A)| = 1, with G(jω)=g/(jω) e^{-jωτ}.
# # ⇒ g_lc(A) = ωc / |N(A)|.
# g_lc = omega_c / np.maximum(np.abs(N_vals), 1e-16)

# print(f"Operating point: misalign={misalign[0]:.2f} (wvl/D), focus offset f0={f0_rad:.2f} rad RMS")
# print(f"Delay τ = {tau*1e3:.2f} ms ⇒ ωc = π/(2τ) = {omega_c/(2*np.pi):.1f} Hz crossover at the delay limit.")
# print(f"N(A) near A→0 ≈ {N_vals[0].real:.3f} (real), Im≈{N_vals[0].imag:.3e}")

# # --- Plots: DF magnitude and limit-cycle gain vs amplitude ---
# plt.figure(figsize=(7.2,4.2))
# plt.plot(A_grid, np.abs(N_vals), 'o-', label='|N(A)|')
# plt.xlabel('Sinusoid amplitude A in true TIP (rad RMS)')
# plt.ylabel('|N(A)|  (recon TIP per true TIP)')
# plt.title('Describing function magnitude of ZWFS nonlinearity')
# plt.grid(True, alpha=0.3); plt.legend(); plt.tight_layout(); plt.show()

# plt.figure(figsize=(7.2,4.2))
# plt.plot(A_grid, g_lc, 'o-', label=r'$g_\mathrm{lc}(A)=\omega_c/|N(A)|$')
# plt.yscale('log')
# plt.xlabel('Limit-cycle amplitude A (rad RMS)')
# plt.ylabel('Integrator gain for limit cycle,  g_lc(A)')
# plt.title('Predicted integrator gain for a limit cycle vs amplitude')
# plt.grid(True, which='both', alpha=0.3); plt.legend(); plt.tight_layout(); plt.show()



# # to continue... senstivity vs other masks confirm... 



# ##### END 







# # ### THE FUNCTION NOW IN utilities.py

# # def get_theoretical_reference_pupils_with_aber( wavelength = 1.65e-6 ,F_number = 21.2, mask_diam = 1.2, coldstop_diam=None, coldstop_misalign=None, eta=0, phi= None, diameter_in_angular_units = True, get_individual_terms=False, phaseshift = np.pi/2 , padding_factor = 4, debug= True, analytic_solution = True ) :
# #     """
# #     get theoretical reference pupil intensities of ZWFS with / without phasemask 
    

# #     Parameters
# #     ----------
# #     wavelength : TYPE, optional
# #         DESCRIPTION. input wavelength The default is 1.65e-6.
# #     F_number : TYPE, optional
# #         DESCRIPTION. The default is 21.2.
# #     mask_diam : phase dot diameter. TYPE, optional
# #             if diameter_in_angular_units=True than this has diffraction limit units ( 1.22 * f * lambda/D )
# #             if  diameter_in_angular_units=False than this has physical units (m) determined by F_number and wavelength
# #         DESCRIPTION. The default is 1.2.
# #     coldstop_diam : diameter in lambda / D of focal plane coldstop
# #     coldstop_misalign : alignment offset of the cold stop (in units of image plane pixels)  
# #     phi : input phase aberrations (None by default). should be same size as pupil which by default is 2D grid of 2**9+1
# #     eta : ratio of secondary obstruction radius (r_2/r_1), where r2 is secondary, r1 is primary. 0 meams no secondary obstruction
# #     diameter_in_angular_units : TYPE, optional
# #         DESCRIPTION. The default is True.
# #     get_individual_terms : Type optional
# #         DESCRIPTION : if false (default) with jsut return intensity, otherwise return P^2, abs(M)^2 , phi + mu
# #     phaseshift : TYPE, optional
# #         DESCRIPTION. phase phase shift imparted on input field (radians). The default is np.pi/2.
# #     padding_factor : pad to change the resolution in image plane. TYPE, optional
# #         DESCRIPTION. The default is 4.
# #     debug : TYPE, optional
# #         DESCRIPTION. Do we want to plot some things? The default is True.
# #     analytic_solution: TYPE, optional
# #         DESCRIPTION. use analytic formula or calculate numerically? The default is True.
# #     Returns
# #     -------
# #     Ic, reference pupil intensity with phasemask in 
# #     P, reference pupil intensity with phasemask out 

# #     """
# #     pupil_radius = 1  # Pupil radius in meters

# #     # Define the grid in the pupil plane
# #     N = 2**9+1  # for parity (to not introduce tilt) works better ODD!  # Number of grid points (assumed to be square)
# #     L_pupil = 2 * pupil_radius  # Pupil plane size (physical dimension)
# #     dx_pupil = L_pupil / N  # Sampling interval in the pupil plane
# #     x_pupil = np.linspace(-L_pupil/2, L_pupil/2, N)   # Pupil plane coordinates
# #     y_pupil = np.linspace(-L_pupil/2, L_pupil/2, N) 
# #     X_pupil, Y_pupil = np.meshgrid(x_pupil, y_pupil)
    
    


# #     # Define a circular pupil function
# #     pupil = (np.sqrt(X_pupil**2 + Y_pupil**2) > eta*pupil_radius) & (np.sqrt(X_pupil**2 + Y_pupil**2) <= pupil_radius)
# #     pupil = pupil.astype( complex )
# #     if phi is not None:
# #         pupil *= np.exp(1j * phi)
# #     else:
# #         phi = np.zeros( pupil.shape ) # added aberrations 
        
# #     # Zero padding to increase resolution
# #     # Increase the array size by padding (e.g., 4x original size)
# #     N_padded = N * padding_factor
# #     if (N % 2) != (N_padded % 2):  
# #         N_padded += 1  # Adjust to maintain parity
        
# #     pupil_padded = np.zeros((N_padded, N_padded)).astype(complex)
# #     #start_idx = (N_padded - N) // 2
# #     #pupil_padded[start_idx:start_idx+N, start_idx:start_idx+N] = pupil

# #     start_idx_x = (N_padded - N) // 2
# #     start_idx_y = (N_padded - N) // 2  # Explicitly ensure symmetry

# #     pupil_padded[start_idx_y:start_idx_y+N, start_idx_x:start_idx_x+N] = pupil


# #     phi_padded = np.zeros((N_padded, N_padded), dtype=float)
# #     phi_padded[start_idx_y:start_idx_y+N, start_idx_x:start_idx_x+N] = phi

# #     # Perform the Fourier transform on the padded array (normalizing for the FFT)
# #     #pupil_ft = np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(pupil_padded))) # we do this laters
    
# #     # Compute the Airy disk scaling factor (1.22 * lambda * F)
# #     airy_scale = 1.22 * wavelength * F_number

# #     # Image plane sampling interval (adjusted for padding)
# #     #L_image = wavelength * F_number / dx_pupil  # Total size in the image plane
# #     #dx_image_padded = L_image / N_padded  # Sampling interval in the image plane with padding
    
# #     dx_image_padded = wavelength * F_number * (N / N_padded)
# #     L_image = dx_image_padded * N_padded

# #     if diameter_in_angular_units:
# #         x_image_padded = np.linspace(-L_image/2, L_image/2, N_padded) / airy_scale  # Image plane coordinates in Airy units
# #         y_image_padded = np.linspace(-L_image/2, L_image/2, N_padded) / airy_scale
# #     else:
# #         x_image_padded = np.linspace(-L_image/2, L_image/2, N_padded)  # Image plane coordinates in Airy units
# #         y_image_padded = np.linspace(-L_image/2, L_image/2, N_padded) 
        
# #     X_image_padded, Y_image_padded = np.meshgrid(x_image_padded, y_image_padded)

# #     if diameter_in_angular_units:
# #         mask = np.sqrt(X_image_padded**2 + Y_image_padded**2) <= mask_diam / 2 #4
# #     else: 
# #         mask = np.sqrt(X_image_padded**2 + Y_image_padded**2) <= mask_diam / 2 #4


# #     # --- convert misalignment from wvl/D to your image-plane units ---
# #     # ---- cold stop offset: wvl/D -> grid units ----
# #     if coldstop_misalign is not None:
# #         dx_wvld, dy_wvld = coldstop_misalign
# #     else:
# #         dx_wvld, dy_wvld = [0.0, 0.0]

# #     if diameter_in_angular_units:
# #         wvld_to_units = 1.0/1.22            # Airy radii per (wvl/D)
# #     else:
# #         wvld_to_units = F_number * wavelength  # meters per (wvl/D)
# #     dx_units = dx_wvld * wvld_to_units
# #     dy_units = dy_wvld * wvld_to_units

# #     if coldstop_diam is not None:
# #         if diameter_in_angular_units:
# #             cs_radius_units = (coldstop_diam * (1.0/1.22)) / 2.0
# #         else:
# #             cs_radius_units = (coldstop_diam * (F_number * wavelength)) / 2.0
# #         coldmask = (np.hypot(X_image_padded - dx_units, Y_image_padded - dy_units) <= cs_radius_units).astype(float)
# #     else:
# #         coldmask = np.ones_like(X_image_padded)


# #     # if coldstop_misalign is not None:
# #     #     dx_wvld, dy_wvld = coldstop_misalign
# #     # else:
# #     #     dx_wvld, dy_wvld = [0,0]
    

# #     # if diameter_in_angular_units:
# #     #     # Your X_image_padded, Y_image_padded are in "Airy radii" units set by:
# #     #     # airy_scale = 1.22 * wavelength * F_number
# #     #     # 1 (wvl/D) equals (F_number * wavelength) in meters,
# #     #     # which is (1 / 1.22) Airy radii on this normalized grid.
# #     #     wvld_to_units = 1.0 / 1.22                      # Airy radii per (wvl/D)
# #     #     dx_units = dx_wvld * wvld_to_units
# #     #     dy_units = dy_wvld * wvld_to_units
# #     # else:
# #     #     # Your X_image_padded, Y_image_padded are in meters.
# #     #     # 1 (wvl/D) = F_number * wavelength  [meters]
# #     #     wvld_to_units = F_number * wavelength           # meters per (wvl/D)
# #     #     dx_units = dx_wvld * wvld_to_units
# #     #     dy_units = dy_wvld * wvld_to_units
        
# #     # # if coldstop_diam is not None:
# #     # #     coldmask = np.sqrt(X_image_padded**2 + Y_image_padded**2) <= coldstop_diam / 4
# #     # # else:
# #     # #     coldmask = np.ones(X_image_padded.shape)
# #     # if coldstop_diam is not None: # apply also the cold stop offset 
# #     #     coldmask = np.sqrt((X_image_padded-dx_units)**2 + (Y_image_padded-dy_units)**2) <= coldstop_diam / 2 #4
# #     # else:
# #     #     coldmask = np.ones(X_image_padded.shape)

# #     pupil_ft = np.fft.fft2(np.fft.ifftshift(pupil_padded))  # Remove outer fftshift
# #     pupil_ft = np.fft.fftshift(pupil_ft)  # Shift only once at the end

# #     psi_B = coldmask * pupil_ft
                            
# #     b = np.fft.fftshift( np.fft.ifft2( mask * psi_B ) ) # we do mask here because really the cold stop is after phase mask in physical system

    
# #     if debug: 
        
# #         psf = np.abs(pupil_ft)**2  # Get the PSF by taking the square of the absolute value
# #         psf /= np.max(psf)  # Normalize PSF intensity
        
# #         if diameter_in_angular_units:
# #             zoom_range = 3  # Number of Airy disk radii to zoom in on
# #         else:
# #             zoom_range = 3 * airy_scale 
            
# #         extent = (-zoom_range, zoom_range, -zoom_range, zoom_range)

# #         fig,ax = plt.subplots(1,1)
# #         ax.imshow(psf, extent=(x_image_padded.min(), x_image_padded.max(), y_image_padded.min(), y_image_padded.max()), cmap='gray')
# #         ax.contour(X_image_padded, Y_image_padded, mask, levels=[0.5], colors='red', linewidths=2, label='phasemask')
# #         #ax[1].imshow( mask, extent=(x_image_padded.min(), x_image_padded.max(), y_image_padded.min(), y_image_padded.max()), cmap='gray')
# #         #for axx in ax.reshape(-1):
# #         #    axx.set_xlim(-zoom_range, zoom_range)
# #         #    axx.set_ylim(-zoom_range, zoom_range)
# #         ax.set_xlim(-zoom_range, zoom_range)
# #         ax.set_ylim(-zoom_range, zoom_range)
# #         ax.set_title( 'PSF' )
# #         ax.legend() 
# #         #ax[1].set_title('phasemask')


    
# #     # if considering complex b 
# #     # beta = np.angle(b) # complex argunment of b 
# #     # M = b * (np.exp(1J*theta)-1)**0.5
    
# #     # relabelling
# #     theta = phaseshift # rad , 
# #     P = pupil_padded.copy() 
    
# #     if analytic_solution :
        
# #         M = abs( b ) * np.sqrt((np.cos(theta)-1)**2 + np.sin(theta)**2)
# #         mu = np.angle((np.exp(1J*theta)-1) ) # np.arctan( np.sin(theta)/(np.cos(theta)-1) ) #
        

# #         # out formula ----------
# #         #if measured_pupil!=None:
# #         #    P = measured_pupil / np.mean( P[P > np.mean(P)] ) # normalize by average value in Pupil
# #         P = np.abs(pupil_padded).real  # we already dealt with the complex part in this analytic expression which is in phi
# #         Ic = ( P**2 + abs(M)**2 + 2* P* abs(M) * np.cos(phi_padded + mu) ) #+ beta)
# #         if not get_individual_terms:
# #             return( P, Ic )
# #         else:
# #             return( P, abs(M) , phi+mu )
# #     else:
        
# #         # phasemask filter 
        
# #         T_on = 1
# #         T_off = 1
# #         H = T_off*(1 + (T_on/T_off * np.exp(1j * theta) - 1) * mask  ) 
        
# #         Ic = abs( np.fft.fftshift( np.fft.ifft2( H * psi_B ) ) ) **2 
    
# #         return( P, Ic )
