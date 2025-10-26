# ===============================================================
# ZWFS cold-stop misalignment tolerance: bound and exact Jacobian
# ===============================================================
# This script:
#  1) Builds a high-order phase screen phi_ho (Z2..Z15 removed).
#  2) Computes a conservative centering tolerance bound using
#     || LP{|K|^2} * delta I_C || and K_pass = Dcs/2.
#  3) Computes the exact small-signal 2x2 TT Jacobian vs stop
#     misalignment (finite differences) and the corresponding tolerance.
#
# Requirements in your environment:
#   from pyBaldr import utilities as util
#   from common   import DM_basis_functions as dmbasis
#
# Notes:
# - Units: misalignment delta rho in lambda/D. TT coefficients in rad RMS.
# - Detector binning preserves intensity (mean over blocks).
# - You can sweep Dcs, Mdet, eps_TT to build thesis tables.
# ===============================================================

import numpy as np
import matplotlib.pyplot as plt

# --- your libs ---
from pyBaldr import utilities as util
from common import DM_basis_functions as dmbasis

# ---------------------------
# Experiment / telescope setup
# ---------------------------
phasemask_parameters = {
    "J5": {"depth": 0.474, "diameter": 32},
    "J4": {"depth": 0.474, "diameter": 36},
    "J3": {"depth": 0.474, "diameter": 44},
    "J2": {"depth": 0.474, "diameter": 54},
    "J1": {"depth": 0.474, "diameter": 65},
    "H1": {"depth": 0.654, "diameter": 68},
    "H2": {"depth": 0.654, "diameter": 53},
    "H3": {"depth": 0.654, "diameter": 44},
    "H4": {"depth": 0.654, "diameter": 37},
    "H5": {"depth": 0.654, "diameter": 31},
}

# band / mask / optics (same as your earlier context)
T = 1900  # K lab thermal source temperature
lambda_cut_on, lambda_cut_off = 1.38, 1.82  # um
wvl = util.find_central_wavelength(lambda_cut_on, lambda_cut_off, T)
mask_key = "H3"
F_number = 21.2
mask_diam = 1.22 * F_number * wvl / phasemask_parameters[mask_key]['diameter']
eta = 138 / 1800  # ATs secondary obscuration ratio

# ===============================================================
# Helper utilities (reviewed)
# ===============================================================

def unit_rms(arr, mask):
    arr = arr.copy()
    arr -= arr[mask].mean()
    rms = np.sqrt((arr[mask]**2).mean()) + 1e-12
    return arr / rms

def make_Pmask(N):
    x = np.linspace(-1, 1, N)
    X, Y = np.meshgrid(x, x, indexing='xy')
    return (X**2 + Y**2) <= 1

def fourier_lowpass_mask(shape, cutoff_cpd):
    """
    Ideal circular low-pass mask in cycles per pupil diameter (cpd).
    With an N x N image, FFT freqs fx,fy are in cycles/pixel; to convert to cpd,
    multiply by N/2 (since the pupil diameter spans 2 in normalized units).
    """
    Ny, Nx = shape
    fx = np.fft.fftfreq(Nx, d=1.0)
    fy = np.fft.fftfreq(Ny, d=1.0)
    FX, FY = np.meshgrid(fx, fy, indexing='xy')
    M = Nx
    KX = FX * M / 2.0
    KY = FY * M / 2.0
    KR = np.sqrt(KX**2 + KY**2)
    mask = (KR <= cutoff_cpd).astype(float)
    return np.fft.fftshift(mask)

def _center_outer_radius(pupil_amp):
    """Center (cx,cy) and OUTER radius r (in fine pixels) from amplitude mask."""
    P = np.abs(pupil_amp)
    mask = P > (1e-6 * P.max())
    if not np.any(mask):
        raise ValueError("No pupil detected.")
    ys, xs = np.where(mask)
    cx = xs.mean()
    cy = ys.mean()
    rx = 0.5 * (xs.max() - xs.min() + 1)
    ry = 0.5 * (ys.max() - ys.min() + 1)
    r = max(rx, ry)
    return cx, cy, r

def bin_to_detector(P_amp, I_fine, pupil_pixels_diam=12, mode="mean"):
    """
    Flux-preserving downsample that keeps the same FOV ratio as the fine image.
    Output size Md x Md where Md ≈ (Nf/Df) * M_pupil, and the pupil spans M_pupil px across diameter.
    """
    M_pupil = pupil_pixels_diam
    Nf = I_fine.shape[0]
    assert I_fine.shape[0] == I_fine.shape[1], "Assumes square frame."

    _, _, r = _center_outer_radius(P_amp)
    Df = 2.0 * r  # fine pupil diameter in pixels

    Md = int(round((Nf / Df) * M_pupil))
    Md = max(Md, M_pupil)

    if Nf % Md == 0:
        k = Nf // Md
        blocks = I_fine.reshape(Md, k, Md, k).swapaxes(1, 2)
        if mode == "mean":
            return blocks.mean(axis=(2, 3))
        elif mode == "sum":
            return blocks.sum(axis=(2, 3))
        else:
            raise ValueError("mode must be 'mean' or 'sum'")

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
        return I_sum
    out = np.zeros_like(I_sum)
    nz = N_hit > 0
    out[nz] = I_sum[nz] / N_hit[nz]
    return out

# ===============================================================
# Core functions you provided (reviewed, kept as-is in spirit)
# ===============================================================

def kolmogorov_phase_screen(N, pixel_scale=1.0, r0_pixels=50.0, seed=42, Pmask=None):
    rng = np.random.default_rng(seed)
    fx = np.fft.fftfreq(N, d=pixel_scale)
    fy = np.fft.fftfreq(N, d=pixel_scale)
    FX, FY = np.meshgrid(fx, fy, indexing='xy')
    k = np.sqrt(FX**2 + FY**2) + 1e-12
    amp = k**(-11/6.0)
    kmin = 1.0 / (N * pixel_scale)
    kmax = 0.25 / pixel_scale
    amp *= (k >= kmin) * (k <= kmax)
    phase_rand = rng.uniform(0, 2*np.pi, size=(N, N))
    Fphi = amp * (np.cos(phase_rand) + 1j*np.sin(phase_rand))
    Fphi = np.fft.fftshift(Fphi)
    phi = np.fft.ifft2(np.fft.ifftshift(Fphi)).real
    if Pmask is not None:
        phi *= Pmask
        phi = unit_rms(phi, Pmask)
    return phi

def remove_low_zernikes(phi, zernike_basis, Pmask, noll_indices_remove=list(range(2,16)), scaling_factor=0.3):
    out = phi.copy()
    for j in noll_indices_remove:
        Z = np.nan_to_num(zernike_basis[j-1]) * Pmask
        Z = unit_rms(Z, Pmask)
        coeff = (out[Pmask] * Z[Pmask]).mean()
        out -= coeff * Z
    out -= out[Pmask].mean()
    out = unit_rms(out, Pmask) * scaling_factor
    return out

def zwfs_intensity(phi, coldstop_diam_lamOverD, padding=6, phaseshift=None, coldstop_misalign=[0,0]):
    depth = phasemask_parameters[mask_key]['depth']
    phaseshift = phaseshift if phaseshift is not None else util.get_phasemask_phaseshift(
        wvl=wvl, depth=depth, dot_material='N_1405'
    )
    P_, Ic_ = util.get_theoretical_reference_pupils_with_aber(
        wavelength=wvl,
        F_number=F_number,
        mask_diam=mask_diam,
        coldstop_diam=coldstop_diam_lamOverD,  # diameter in lambda/D
        coldstop_misalign=coldstop_misalign,   # [dx,dy] in lambda/D
        eta=eta,
        phi=phi,
        diameter_in_angular_units=True,
        get_individual_terms=False,
        phaseshift=phaseshift,
        padding_factor=padding,
        debug=False,
        analytic_solution=False
    )
    return P_, Ic_

def build_sensing_matrix(low_order_modes, coldstop_diam_lamOverD, padding=6):
    P0, I0 = zwfs_intensity(phi=np.zeros_like(low_order_modes[0]),
                            coldstop_diam_lamOverD=coldstop_diam_lamOverD,
                            padding=padding)
    S = []
    for u in low_order_modes:
        _, Iu = zwfs_intensity(phi=u, coldstop_diam_lamOverD=coldstop_diam_lamOverD, padding=padding)
        S.append(Iu - I0)
    return P0, I0, S

def fit_low_order_coeffs(deltaI_img, S_list, mask=None):
    if mask is None:
        mask = np.ones_like(deltaI_img, dtype=bool)
    A = np.stack([S[mask].ravel() for S in S_list], axis=1)
    y = deltaI_img[mask].ravel()
    coeffs, *_ = np.linalg.lstsq(A, y, rcond=None)
    residual = np.linalg.norm(y - A @ coeffs) / (np.linalg.norm(y) + 1e-12)
    return coeffs, residual

def fit_coeffs_against_columns(img, cols, mask=None):
    if mask is None:
        mask = np.ones_like(img, dtype=bool)
    A = np.stack([C[mask].ravel() for C in cols], axis=1)
    y = img[mask].ravel()
    c, *_ = np.linalg.lstsq(A, y, rcond=None)
    res = np.linalg.norm(y - A @ c) / (np.linalg.norm(y) + 1e-12)
    return c, res

def build_TT_sensing_detector(coldstop_diam, pupil_px_diam, basis_zern, Pmask):
    Zx = unit_rms(np.nan_to_num(basis_zern[1]) * Pmask, Pmask)  # Noll 2
    Zy = unit_rms(np.nan_to_num(basis_zern[2]) * Pmask, Pmask)  # Noll 3
    P0, I0 = zwfs_intensity(phi=np.zeros_like(Pmask, float), coldstop_diam_lamOverD=coldstop_diam, padding=6)
    _, Ix = zwfs_intensity(phi=Zx, coldstop_diam_lamOverD=coldstop_diam, padding=6)
    _, Iy = zwfs_intensity(phi=Zy, coldstop_diam_lamOverD=coldstop_diam, padding=6)
    dIx = Ix - I0
    dIy = Iy - I0
    dIx_det = bin_to_detector(P0, dIx, pupil_pixels_diam=pupil_px_diam)
    dIy_det = bin_to_detector(P0, dIy, pupil_pixels_diam=pupil_px_diam)
    return dIx_det, dIy_det

# ===============================================================
# Evaluation block
# ===============================================================
if __name__ == "__main__":
    # ---- configuration ----
    N_fine   = 513
    Dcs      = 4.0
    kpass    = Dcs / 2.0           # cycles per pupil diameter
    eps_TT   = 0.01                # rad RMS TT budget
    Mdet_set = (6, 12)             # detector pixels across pupil diameter
    fd_step  = 0.05                # lambda/D finite-difference step

    # pupil and zernikes
    Pmask = make_Pmask(N_fine)
    Zbasis = [np.nan_to_num(z) for z in dmbasis.zernike_basis(nterms=120, npix=N_fine, rho=None, theta=None)]

    # high-order phase screen with Z2..Z15 removed
    phi_raw = kolmogorov_phase_screen(N_fine, r0_pixels=50.0, seed=7, Pmask=Pmask)
    phi_ho  = remove_low_zernikes(phi_raw, Zbasis, Pmask, noll_indices_remove=list(range(2,16)), scaling_factor=0.30)
    rms_ho  = np.sqrt((phi_ho[Pmask]**2).mean())
    print(f"[info] phi_ho RMS ~ {rms_ho:.3f} rad")

    # ---- 1) Conservative bound (uses deltaI_C without cold stop) ----
    Dcs_open = 1e6
    P_ref, I0_open   = zwfs_intensity(np.zeros_like(Pmask, float), coldstop_diam_lamOverD=Dcs_open, padding=6, coldstop_misalign=[0,0])
    _,      Iphi_open = zwfs_intensity(phi_ho,                      coldstop_diam_lamOverD=Dcs_open, padding=6, coldstop_misalign=[0,0])
    dI_C = Iphi_open - I0_open

    LP = fourier_lowpass_mask(dI_C.shape, cutoff_cpd=kpass)
    dI_hat = np.fft.fft2(dI_C)
    filtered = np.fft.ifft2(dI_hat * np.fft.ifftshift(LP)).real

    norm_filtered = np.sqrt((filtered[Pmask]**2).mean())
    S_bound = 2*np.pi*kpass*norm_filtered                   # rad/(lambda/D)
    tol_bound = eps_TT / (S_bound + 1e-16)

    print("\n[Conservative bound]")
    print(f"  kpass = {kpass:.3f} cpd, ||LP*deltaI_C||_RMS = {norm_filtered:.3e}")
    print(f"  S_bound = 2pi kpass ||·|| = {S_bound:.3e} rad/(lambda/D)")
    print(f"  |deltaρ|_max for epsilon_TT={eps_TT:.3f} rad ≈ {tol_bound:.4f} lambda/D")

    # ---- 2) Exact small-signal Jacobian via finite differences ----
    def tt_coeffs_at(misalign_xy, Mdet):
        P0, I0 = zwfs_intensity(np.zeros_like(Pmask, float), coldstop_diam_lamOverD=Dcs, padding=6, coldstop_misalign=list(misalign_xy))
        _,  I1 = zwfs_intensity(phi_ho,                      coldstop_diam_lamOverD=Dcs, padding=6, coldstop_misalign=list(misalign_xy))
        dI = I1 - I0
        dI_det = bin_to_detector(P0, dI, pupil_pixels_diam=Mdet)
        dIx_det, dIy_det = build_TT_sensing_detector(Dcs, Mdet, Zbasis, Pmask)
        c, _ = fit_coeffs_against_columns(dI_det, [dIx_det, dIy_det], mask=np.isfinite(dI_det))
        return np.asarray(c)  # [c_TTx, c_TTy] in rad RMS

    def tt_jacobian(Mdet, delta=fd_step):
        c_px = tt_coeffs_at((+delta, 0.0), Mdet)
        c_mx = tt_coeffs_at((-delta, 0.0), Mdet)
        c_py = tt_coeffs_at((0.0, +delta), Mdet)
        c_my = tt_coeffs_at((0.0, -delta), Mdet)
        J = np.column_stack(((c_px - c_mx)/(2*delta), (c_py - c_my)/(2*delta)))  # 2x2
        svals = np.linalg.svd(J, compute_uv=False)
        sigma_max = svals[0] if svals.size > 0 else 0.0
        return J, sigma_max

    print("\n[Exact Jacobian (finite differences around deltaρ=0)]")
    for Mdet in Mdet_set:
        J, sig = tt_jacobian(Mdet, delta=fd_step)
        tol_exact = eps_TT / (sig + 1e-16)
        print(f"  Mdet = {Mdet:2d} px/diam:")
        print(f"    J = [[{J[0,0]: .3e}, {J[0,1]: .3e}],")
        print(f"         [{J[1,0]: .3e}, {J[1,1]: .3e}]]  rad/(lambda/D)")
        print(f"    ||J||_2 = {sig:.3e}  →  |deltaρ|_max(epsilon_TT={eps_TT:.3f} rad) ≈ {tol_exact:.4f} lambda/D")

    print("\n[Comparison]")
    print("  Bound uses S_bound independent of detector sampling; exact uses detector + TT reconstructor.")
    print("  Typically, ||J||_2 ≤ O(2π kpass ||LP*deltaI_C||), often smaller depending on sampling and basis.")