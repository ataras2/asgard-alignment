import numpy as np
import matplotlib.pyplot as plt
from pyBaldr import utilities as util
from xaosim.shmlib import shm


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

coldstop is has diameter 2.145 mm
baldr beams (30mm collimating lens) fratio 21.2 at focal plane focused by 254mm OAP
is xmm with 200mm imaging lens
2.145e-3 / ( 2 * 200 / (254 / 21.2 * 30 / 254 ) * 1.56e-6 )
wvl = 1.56um
coldstop_diam  ~ 4.8 lambda/D 
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
# TT, focus basis 
N = 2**9 + 1
x = np.linspace(-1, 1, N)
X, Y = np.meshgrid(x, x, indexing='xy')
P = (X**2 + Y**2) <= 1  # circular pupil mask
tip  = X * P
tilt = Y * P
# Normalize each to 1 rad RMS within the pupil
tip  /= np.sqrt(np.mean(tip[P]**2))
tilt /= np.sqrt(np.mean(tilt[P]**2))

R2 = X**2 + Y**2
focus = (2.0*R2 - 1.0) * P          # Zernike-like defocus (∝ Z4)

# Remove tiny numerical piston/tilt leakage, then normalize to 1 rad RMS
focus -= focus[P].mean()
focus -= ((focus[P]*tip[P]).mean()  / (tip[P]**2).mean())  * tip
focus -= ((focus[P]*tilt[P]).mean() / (tilt[P]**2).mean()) * tilt
focus /= np.sqrt(np.mean(focus[P]**2))

basis = [tip,tilt,focus]
###################

# np.sum( tip**2*P ) / np.sum(P**2) == 1
P, Ic = util.get_theoretical_reference_pupils_with_aber( wavelength = wvl ,
                                              F_number = F_number , 
                                              mask_diam = mask_diam, 
                                              coldstop_diam=10, #coldstop_diam,
                                              coldstop_misalign = [1,0], #lambda/D units
                                              eta = eta, 
                                              phi = 0 * basis[1], #+ -0.3 * basis[2] ,
                                              diameter_in_angular_units = True, 
                                              get_individual_terms=False, 
                                              phaseshift = util.get_phasemask_phaseshift(wvl=wvl, depth = phasemask_parameters[mask]['depth'], dot_material='N_1405') , 
                                              padding_factor = 6, 
                                              debug= False, 
                                              analytic_solution = False )

############################################
## Plot theoretical intensities on fine grid 
imgs = [abs(P), Ic]
titles=['Clear Pupil', 'ZWFS Pupil']
cbars = ['Intensity', 'Intensity']
xlabel_list, ylabel_list = ['',''], ['','']
util.nice_heatmap_subplots(im_list=imgs ,
                            xlabel_list=xlabel_list, 
                            ylabel_list=ylabel_list, 
                            title_list=titles, 
                            cbar_label_list=cbars, 
                            fontsize=15, 
                            cbar_orientation = 'bottom', 
                            axis_off=True, 
                            vlims=None, 
                            savefig='delme.png')
plt.show()

############################################
## Plot theoretical intensities on CRED1 Detector (12 pixel diameter)
# we can use a clear pupil measurement to interpolate this onto 
# the measured pupil pixels.

# Original grid dimensions from the theoretical pupil
M, N = Ic.shape

m, n = 36, 36  # New grid dimensions (width, height in pixels)
# To center the pupil, set the center at half of the grid size.
x_c, y_c = int(m/2), int(n/2)
# For a 12-pixel diameter pupil, the new pupil radius should be 6 pixels.
new_radius = 3

# Interpolate the theoretical intensity onto the new grid.
detector_intensity = util.interpolate_pupil_to_measurement(abs(P), abs(Ic), M, N, m, n, x_c, y_c, new_radius)

# Plot the interpolated theoretical pupil intensity.
imgs = [detector_intensity]
titles=[ 'Detected\nZWFS Pupil']
cbars = ['Intensity']
xlabel_list, ylabel_list = [''], ['']
util.nice_heatmap_subplots(im_list=imgs ,
                            title_list=titles,
                            xlabel_list=xlabel_list, 
                            ylabel_list=ylabel_list, 
                            cbar_label_list=cbars, 
                            fontsize=15, 
                            cbar_orientation = 'bottom', 
                            axis_off=True, 
                            vlims=None, 
                            savefig='delme2.png')
plt.show()


########## BUILD TT RECONSTRUCTOR WITH ALIGNED SYSTEM 
# ----- Settings for the aligned system & linearization -----
aligned_misalign = (0.0, 0.0)   # cold stop aligned (in wvl/D)
use_coldstop_diam = coldstop_diam  # or set to None to disable the cold stop in this step
epsilon = 0.01                  # small modal poke [rad RMS] for finite differences

# Detector geometry (you already defined these above)
m, n = 36, 36
x_c, y_c = int(m/2), int(n/2)
new_radius = 3                  # pupil radius (px) on detector => diameter = 6 px

# ----- Helper: downsample (same call pattern you used) -----
def to_detector(P_hr, I_hr):
    M_hr, N_hr = I_hr.shape
    return util.interpolate_pupil_to_measurement(np.abs(P_hr), np.abs(I_hr),
                                                 M_hr, N_hr, m, n, x_c, y_c, new_radius)

# ----- Build a detector-plane pupil mask (boolean) -----
# Use a clear pupil intensity as a proxy to locate detector pixels that are "inside the pupil"
P0_hr, Ic0_hr = util.get_theoretical_reference_pupils_with_aber(
    wavelength=wvl, F_number=F_number,
    mask_diam=mask_diam, coldstop_diam=use_coldstop_diam,
    coldstop_misalign=aligned_misalign, eta=eta,
    phi=np.zeros_like(basis[0]), diameter_in_angular_units=True,
    phaseshift=util.get_phasemask_phaseshift(wvl=wvl, depth=phasemask_parameters[mask]['depth'], dot_material='N_1405'),
    padding_factor=6, analytic_solution=False, debug=False
)
det_P = to_detector(P0_hr, P0_hr)       # map clear pupil to detector
det_ref = to_detector(P0_hr, Ic0_hr)    # reference ZWFS intensity on detector (aligned)

# Threshold to define valid detector pixels (inside the pupil support)
det_mask = det_P > (0.5 * det_P.max())
pix_idx = np.where(det_mask.ravel())[0]   # vectorization indices over pupil pixels
n_pix = pix_idx.size

# ----- Build interaction matrix A (n_pix × n_modes) -----
modes = [tip, tilt]            # 1 rad RMS each (from your earlier basis)
mode_names = ["tip", "tilt"]
n_modes = len(modes)
A = np.zeros((n_pix, n_modes))

for j, mode in enumerate(modes):
    # Plus poke
    _, Ic_plus_hr = util.get_theoretical_reference_pupils_with_aber(
        wavelength=wvl, F_number=F_number,
        mask_diam=mask_diam, coldstop_diam=use_coldstop_diam,
        coldstop_misalign=aligned_misalign, eta=eta,
        phi=+epsilon * mode, diameter_in_angular_units=True,
        phaseshift=util.get_phasemask_phaseshift(wvl=wvl, depth=phasemask_parameters[mask]['depth'], dot_material='N_1405'),
        padding_factor=6, analytic_solution=False, debug=False
    )
    det_plus = to_detector(P0_hr, Ic_plus_hr)

    # Minus poke
    _, Ic_minus_hr = util.get_theoretical_reference_pupils_with_aber(
        wavelength=wvl, F_number=F_number,
        mask_diam=mask_diam, coldstop_diam=use_coldstop_diam,
        coldstop_misalign=aligned_misalign, eta=eta,
        phi=-epsilon * mode, diameter_in_angular_units=True,
        phaseshift=util.get_phasemask_phaseshift(wvl=wvl, depth=phasemask_parameters[mask]['depth'], dot_material='N_1405'),
        padding_factor=6, analytic_solution=False, debug=False
    )
    det_minus = to_detector(P0_hr, Ic_minus_hr)

    # Finite-difference derivative (remove any DC/piston leakage inside the pupil)
    resp = (det_plus - det_minus) / (2.0 * epsilon)
    resp -= resp[det_mask].mean()

    # Vectorize over pupil pixels
    A[:, j] = resp.ravel()[pix_idx]

# Optional: visualize the two columns of A (reshape back to detector)
# util.nice_heatmap_subplots([...])

# ----- Regularized reconstructor R -----
# R maps detector residuals (vectorized over pupil pixels) to modal estimates [tip, tilt]
# Use SVD Tikhonov regularization with a gentle cutoff
U, s, Vh = np.linalg.svd(A, full_matrices=False)
alpha = 1e-3 * s.max()                 # tune as needed; smaller = less regularization
s_filt = s / (s**2 + alpha**2)
R = (Vh.T * s_filt) @ U.T              # shape: (n_modes × n_pix)

# Precompute once after building A, R, det_mask, pix_idx, det_ref:
C = R @ A                       # cross-talk / gain matrix (n_modes x n_modes)
C_inv = np.linalg.inv(C)        # or use np.linalg.pinv(C) if you prefer robustness

def residual_vector(Ic_meas_hr):
    """High-res intensity -> vectorized detector residual (pupil pixels only)."""
    det_meas = to_detector(P0_hr, Ic_meas_hr)
    y = det_meas - det_ref
    y -= y[det_mask].mean()
    return y.ravel()[pix_idx]

def project_to_modes(Ic_meas_hr):
    """
    Returns modal amplitudes in calibration units (rad RMS if you poked in rad RMS).
    For tip/tilt-only calibration, returns [tip, tilt].
    """
    yvec = residual_vector(Ic_meas_hr)
    z = R @ yvec             # reconstructor output (mixed / gain-scaled)
    m = C_inv @ z            # demix & de-gain -> modal amplitudes
    return m.tolist()

# 
# ----- How to use the reconstructor -----
# Given a new detector measurement det_meas (aligned configuration),
# form the residual w.r.t. the reference and estimate [tip, tilt]:
#   det_meas = to_detector(P0_hr, Ic_meas_hr)
#   y = (det_meas - det_ref)
#   y -= y[det_mask].mean()
#   m_hat = R @ y.ravel()[pix_idx]
# where m_hat[0] ~ tip (rad RMS), m_hat[1] ~ tilt (rad RMS) in the calibration units.



#################################################################################
############################ ERROR SIGNAL CHECK 
#################################################################################

# -------- TIP ramp test: input +/-0.5 waves (+/-pi rad) and reconstruction --------
# Assumes: project_to_modes(), to_detector(), P0_hr, det_ref, and all calibration params exist.

# Sweep tip amplitude in WAVES (RMS scaling w.r.t. your tip basis which is 1 rad RMS)
tip_waves = np.linspace(-0.5, 0.5, 10)    # +/-1/2 lambda
tip_rad   = 2*np.pi * tip_waves            # convert to radians

tip_true = []
tip_rec  = []
tilt_rec = []

for a in tip_rad:
    # Inject pure TIP (RMS-scaled since 'tip' basis is 1 rad RMS)
    phi_in = a * tip

    # Aligned system propagation (same params as calibration!)
    _, Ic_hr = util.get_theoretical_reference_pupils_with_aber(
        wavelength=wvl, F_number=F_number,
        mask_diam=mask_diam, coldstop_diam=coldstop_diam,
        coldstop_misalign=(0.0, 0.0), eta=eta,
        phi=phi_in, diameter_in_angular_units=True,
        phaseshift=util.get_phasemask_phaseshift(wvl=wvl, depth=phasemask_parameters[mask]['depth'], dot_material='N_1405'),
        padding_factor=6, analytic_solution=False, debug=False
    )

    # Reconstruct [tip, tilt] in rad (RMS units matching calibration)
    m_hat = project_to_modes(Ic_hr)   # -> [tip_est, tilt_est]
    tip_true.append(a)
    tip_rec.append(m_hat[0])
    tilt_rec.append(m_hat[1])

tip_true = np.array(tip_true)
tip_rec  = np.array(tip_rec)
tilt_rec = np.array(tilt_rec)

# Errors (radians)
tip_err  = tip_rec  - tip_true
tilt_err = tilt_rec - 0.0

# Quick numeric summary
print("TIP sweep (±0.5 waves):")
print(f"  TIP gain   ~ {np.polyfit(tip_true, tip_rec, 1)[0]:.4f} (slope rec vs true)")
print(f"  TIP offset ~ {np.polyfit(tip_true, tip_rec, 1)[1]:.4e} rad")
print(f"  TIP RMSE   ~ {np.sqrt(np.mean(tip_err**2)):.4e} rad")
print(f"  TILT crosstalk (RMS) ~ {np.sqrt(np.mean(tilt_rec**2)):.4e} rad")

# Optional plots
plt.figure(figsize=(6,4))
plt.plot(2 * np.pi/wvl *tip_true/(2*np.pi), 2 * np.pi/wvl * tip_rec/(2*np.pi), 'o-', label='Reconstructed tip')
plt.plot(2 * np.pi/wvl *tip_true/(2*np.pi), 2 * np.pi/wvl *tip_true/(2*np.pi), 'k--', label='1:1')
plt.xlabel('Input tip (rad)',fontsize=15) # wave (tip_* is in wave)
plt.ylabel('Reconstructed tip (rad)',fontsize=15) # wave 
plt.title('ZWFS Tip Reconstruction vs Input (Aligned, faint mode)',fontsize=15)
plt.grid(True, alpha=0.3)
plt.legend(fontsize=15)
plt.gca().tick_params(labelsize=15)
plt.tight_layout()
plt.ylim([-1,1])
#plt.xlim([-1,1])
plt.show()


plt.figure(figsize=(6,4))
plt.plot(tip_true/(2*np.pi), tip_err, 'o-')
plt.xlabel('Input tip (waves)')
plt.ylabel('Tip error (rad)')
plt.title('Tip Reconstruction Error')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

plt.figure(figsize=(6,4))
plt.plot(tip_true/(2*np.pi), tilt_rec, 'o-', label='Tilt cross-talk')
plt.xlabel('Input tip (waves)')
plt.ylabel('Reconstructed tilt (rad)')
plt.title('Tilt Cross-talk vs Tip Input')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()


#################################################################################
# Tip / Tilt error vs cold-stop misalignment (zero input aberrations)
# Sweep misalignment along +x from 0 to 3 (wvl/D), measure reconstructed tip/tilt (rad)
#################################################################################
# zero input aberrations
# misalign cold stop by 0-3 lambda /D 
# for each misalignment measure the tip , tilt error signal 
# plot error signam (y) units of radians rms vs cold stop mis-alignment x (lambda/D units)


# Sweep settings
misalign_grid = np.linspace(0.0, 3.0, 16)   # in wvl/D (0 → 3)
misalign_axis = (1.0, 0.0)                  # shift along +x; set (0,1) for +y

tip_est = []
tilt_est = []

for d in misalign_grid:
    dx_wvld = d * misalign_axis[0]
    dy_wvld = d * misalign_axis[1]

    # ZERO input aberrations
    phi_in = np.zeros_like(basis[0])

    # Propagate with MISALIGNED cold stop
    _, Ic_hr = util.get_theoretical_reference_pupils_with_aber(
        wavelength=wvl, F_number=F_number,
        mask_diam=mask_diam, coldstop_diam=coldstop_diam,
        coldstop_misalign=(dx_wvld, dy_wvld),
        eta=eta, phi=phi_in, diameter_in_angular_units=True,
        phaseshift=util.get_phasemask_phaseshift(
            wvl=wvl, depth=phasemask_parameters[mask]['depth'], dot_material='N_1405'
        ),
        padding_factor=6, analytic_solution=False, debug=False
    )

    # Reconstruct [tip, tilt] (rad, RMS units of your calibration)
    m_hat = project_to_modes(Ic_hr)  # -> [tip_est, tilt_est]
    tip_est.append(m_hat[0])
    tilt_est.append(m_hat[1])

tip_est  = np.array(tip_est)
tilt_est = np.array(tilt_est)

# Quick text summary
print("Cold-stop misalignment sweep (0 - 3 wvl/D):")
print(f"  Tip range:  [{tip_est.min():.3e}, {tip_est.max():.3e}] rad")
print(f"  Tilt range: [{tilt_est.min():.3e}, {tilt_est.max():.3e}] rad")

# Plots
plt.figure(figsize=(6,4))
plt.plot(misalign_grid, tip_est, 'o-', label='TIP (rad)')
plt.plot(misalign_grid, tilt_est, 's-', label='TILT (rad)')
plt.xlabel('Cold-stop misalignment (wvl/D)')
plt.ylabel('Reconstructed modal error (rad RMS)')
plt.title('ZWFS: Tip/Tilt vs Cold-stop Misalignment (zero input aberrations)')
plt.grid(True, alpha=0.3)
plt.legend()
plt.tight_layout()
plt.show()




#################################################################################
# Tip vs cold-stop misalignment for three input regimes:
#   (1) zero aberrations
#   (2) +0.5 rad RMS defocus
#   (3) -0.5 rad RMS defocus
#################################################################################

misalign_grid = np.linspace(0.0, 3.0, 16)   # wvl/D
misalign_axis = (1.0, 0.0)                  # shift along +x
defocus_levels = [0.0, +0.5, -0.5]          # rad RMS on your normalized 'focus' basis

tip_curves = {}

for kappa in defocus_levels:
    tips = []
    for d in misalign_grid:
        dx_wvld = d * misalign_axis[0]
        dy_wvld = d * misalign_axis[1]

        # Input aberration: kappa * focus (focus is 1 rad RMS-normalized)
        phi_in = kappa * focus

        # Propagate with misaligned cold stop
        _, Ic_hr = util.get_theoretical_reference_pupils_with_aber(
            wavelength=wvl, F_number=F_number,
            mask_diam=mask_diam, coldstop_diam=coldstop_diam,
            coldstop_misalign=(dx_wvld, dy_wvld),
            eta=eta, phi=phi_in, diameter_in_angular_units=True,
            phaseshift=util.get_phasemask_phaseshift(
                wvl=wvl, depth=phasemask_parameters[mask]['depth'], dot_material='N_1405'
            ),
            padding_factor=6, analytic_solution=False, debug=False
        )

        # Reconstruct [tip, tilt]; keep only TIP
        tip_est, tilt_est = project_to_modes(Ic_hr)
        tips.append(tip_est)

    tip_curves[kappa] = np.array(tips)

# ---- Plot: TIP vs misalignment for the three regimes ----
plt.figure(figsize=(7,4.5))
labels = {
    0.0:   "focus = 0.0 rad",
    +0.5:  "focus = +0.5 rad",
    -0.5:  "focus = -0.5 rad",
}
for kappa, vals in tip_curves.items():
    plt.plot(misalign_grid, vals, 'o-', label=labels[kappa])

plt.xlabel('Cold-stop misalignment (wvl/D)',fontsize=15)
plt.ylabel('Reconstructed TIP (rad RMS)',fontsize=15)
plt.title('TIP vs Cold-stop Misalignment\n(0, +0.5, -0.5 rad RMS defocus)',fontsize=15)
plt.grid(True, alpha=0.3)
plt.gca().tick_params(labelsize=15)
plt.legend(fontsize=15)
plt.tight_layout()
plt.show()







#################################################################################
#################################################################################
# ERROR SIGNAL CHECK with different misalignments
# Scenarios:
#  A) aligned: no aberration, no misalignment
#  B) 0.5 (wvl/D) cold-stop misalignment, no aberration (which in baldr phasemask corresponds to ~15um drift on 1 lambda/D phasmask with f# 21.2 system)
#  C) 0.5 (wvl/D) cold-stop misalignment + 0.5 rad RMS defocus (130nm RMS focus offset at 1.65um)
#################################################################################


tip_waves = np.linspace(-1.5, 1.5, 25)    # ±1/2 lambda
tip_rad   = 2*np.pi * tip_waves           # radians

scenarios = {
    "A: aligned (0 wvl/D, no defocus)" : dict(misalign=(0.0, 0.0), kappa_defocus=0.0),
    "B: 0.5 wvl/D, no defocus"         : dict(misalign=(0.5, 0.0), kappa_defocus=0.0),
    "C: 0.5 wvl/D, +0.5 rad defocus"   : dict(misalign=(0.5, 0.0), kappa_defocus=0.5),
}

results = {}

for label, cfg in scenarios.items():
    print(f"\nLOOKING AT {label}\n")
    tip_true, tip_rec = [], []

    for a in tip_rad:
        # Input phase: tip (±1/2 wave) plus optional defocus (rad RMS)
        phi_in = a * tip + cfg["kappa_defocus"] * focus

        # Propagate with specified cold-stop misalignment
        _, Ic_hr = util.get_theoretical_reference_pupils_with_aber(
            wavelength=wvl, F_number=F_number,
            mask_diam=mask_diam, coldstop_diam=coldstop_diam,
            coldstop_misalign=cfg["misalign"],
            eta=eta, phi=phi_in, diameter_in_angular_units=True,
            phaseshift=util.get_phasemask_phaseshift(
                wvl=wvl, depth=phasemask_parameters[mask]['depth'], dot_material='N_1405'
            ),
            padding_factor=6, analytic_solution=False, debug=False
        )

        # Reconstruct [tip, tilt]; keep TIP only
        m_hat = project_to_modes(Ic_hr)   # -> [tip_est, tilt_est]
        tip_true.append(a)
        tip_rec.append(m_hat[0])

    tip_true = np.array(tip_true)
    tip_rec  = np.array(tip_rec)
    tip_err  = tip_rec - tip_true

    results[label] = dict(
        tip_true=tip_true,
        tip_rec=tip_rec,
        tip_err=tip_err,
        gain=np.polyfit(tip_true, tip_rec, 1)[0],
        offset=np.polyfit(tip_true, tip_rec, 1)[1],
        rmse=np.sqrt(np.mean(tip_err**2))
    )

# ---------- Plots ----------
# Reconstructed tip vs input (in waves)
fs = 15
plt.figure(figsize=(7.5,4.8))
for label, dat in results.items():
    plt.plot(dat["tip_true"]/(2*np.pi), dat["tip_rec"]/(2*np.pi), 'o-', label=label)
plt.plot(2*np.pi/wvl * tip_waves, 2*np.pi/wvl * tip_waves, 'k--', lw=1, label='Ideal y=x')
plt.xlabel('Input TIP (rad)',fontsize=fs)
plt.ylabel('Reconstructed TIP (rad)',fontsize=fs)
plt.title('TIP Reconstruction vs Input\n(aligned vs misaligned/defocus cases)',fontsize=fs)
plt.grid(True, alpha=0.3)
plt.legend(fontsize=fs)
plt.gca().tick_params(labelsize=fs)
plt.tight_layout()
plt.xlim([-1.5,1.5])
plt.ylim([-0.3,0.3])
plt.show()

# Error (recon − true) in radians vs input (waves)
plt.figure(figsize=(7.5,4.2))
for label, dat in results.items():
    plt.plot(dat["tip_true"]/(2*np.pi), dat["tip_err"], 'o-', label=label)
plt.axhline(0, color='k', lw=1)
plt.xlabel('Input TIP (waves)')
plt.ylabel('TIP error (rad)')
plt.title('TIP Error vs Input')
plt.grid(True, alpha=0.3)
plt.legend()
plt.tight_layout()
plt.show()

# ---------- Text summary ----------
for label, dat in results.items():
    print(f"{label}: gain={dat['gain']:.4f}, offset={dat['offset']:.3e} rad, RMSE={dat['rmse']:.3e} rad")



################
# Plot all of them 

fig, axes = plt.subplots(2, 5, figsize=(15, 6))
axes = axes.flatten()  # Flatten to iterate easily

# Loop over each phasemask and generate synthetic intensity data
for i, (mask, params) in enumerate(phasemask_parameters.items()):
    mask_diam = 1.22 * F_number * wvl / params['diameter']  # Compute mask diameter
    phase_shift = util.get_phasemask_phaseshift(wvl=wvl, depth = phasemask_parameters[mask]['depth'], dot_material='N_1405') , 
    
    P, Ic = util.get_theoretical_reference_pupils( wavelength = wvl ,
                                                F_number = F_number , 
                                                mask_diam = mask_diam, 
                                                coldstop_diam=coldstop_diam,
                                                eta = eta, 
                                                diameter_in_angular_units = True, 
                                                get_individual_terms=False, 
                                                phaseshift = util.get_phasemask_phaseshift(wvl=wvl, depth = phasemask_parameters[mask]['depth'], dot_material='N_1405') , 
                                                padding_factor = 6, 
                                                debug= False, 
                                                analytic_solution = False )

    detector_intensity = util.interpolate_pupil_to_measurement(P, Ic, M, N, m, n, x_c, y_c, new_radius)

    # Plot the results
    im = axes[i].imshow(detector_intensity, cmap='inferno')
    axes[i].set_title(mask, fontsize=20)
    axes[i].axis('off')

# Adjust layout and add colorbar
fig.subplots_adjust(right=0.85)
cbar_ax = fig.add_axes([0.88, 0.15, 0.02, 0.7])
fig.colorbar(im, cax=cbar_ax, label='Intensity')

# Show the final figure
plt.suptitle("ZWFS Theoretical Intensities on CRED1 Detector", fontsize=14)
plt.show()





#################################################################################
#################################################################################
# ---------------- SIM: TIP RMS from focus white-noise (0.5 rad RMS) ----------------
## we have with a static focus offset of 0.5 rad (130nm RMS at wvl =1.65um) and record ~ 0.2 radian Tip error (with no other aberrations in the system) - plot attached . Therefore if we have a time series of white gaussian noise focus with   0.5 rad rms, would this correspond to 0.2 radian tip rms error signal?

#################################################################################
#################################################################################
Nmc = 100
sigma_focus_rad = 0.5            # focus noise RMS [rad]
misalign = (0.5, 0.0)            # 0.5 (wvl/D) cold-stop misalignment along +x
rng = np.random.default_rng(2025)

def run_with_focus_amp(kappa_rad):
    phi_in = kappa_rad * focus    # focus basis is 1 rad RMS-normalized
    _, Ic_hr = util.get_theoretical_reference_pupils_with_aber(
        wavelength=wvl, F_number=F_number,
        mask_diam=mask_diam, coldstop_diam=coldstop_diam,
        coldstop_misalign=misalign, eta=eta, phi=phi_in,
        diameter_in_angular_units=True,
        phaseshift=util.get_phasemask_phaseshift(
            wvl=wvl, depth=phasemask_parameters[mask]['depth'], dot_material='N_1405'
        ),
        padding_factor=6, analytic_solution=False, debug=False
    )
    tip_est, tilt_est = project_to_modes(Ic_hr)
    return tip_est

# Draw a zero-mean white Gaussian focus series with the desired RMS
focus_series = sigma_focus_rad * rng.standard_normal(Nmc)

# Run the model & reconstruct TIP for each sample
tip_series = np.array([run_with_focus_amp(kappa) for kappa in focus_series])

# Remove any DC bias (optional; keeps “RMS of fluctuations”)
tip_series -= tip_series.mean()

# Report RMS in radians and waves
tip_rms_rad = np.sqrt(np.mean(tip_series**2))
tip_rms_waves = tip_rms_rad / (2*np.pi)
print(f"N={Nmc}, focus noise RMS = {sigma_focus_rad:.3f} rad")
print(f"TIP RMS = {tip_rms_rad:.4e} rad  ({tip_rms_waves:.4e} waves)")

# Optional: quick look

plt.figure(figsize=(6.0,3.2))
plt.plot(tip_series, 'o-', ms=4)
plt.axhline(+tip_rms_rad, color='k', ls='--', lw=1)
plt.axhline(-tip_rms_rad, color='k', ls='--', lw=1)
plt.title('TIP from 0.5 rad RMS focus noise\n(0.5 wvl/D cold-stop misalignment)')
plt.xlabel('sample'); plt.ylabel('TIP (rad)')
plt.tight_layout(); plt.show()



# =============================================================================
# Gain margin vs cold-stop misalignment for a basic integrator with latency
#
# We linearize the closed-loop measurement around zero modal command and
# estimate the small-signal plant P(Δ) mapping commanded [tip, tilt] (rad RMS)
# to reconstructed [tip, tilt] (rad RMS), for a given cold-stop misalignment Δ.
#
# Delay-limited stability bound (continuous-time heuristic):
#   g_max  ≈  (π / (2 τ)) * 1 / |G|
# where τ = m * T_s is total loop latency, and |G| is:
#   - SISO: |G_tip| = |∂t̂/∂t|       (tip loop only)
#   - MIMO:  λ_max(P)               (dominant eigen/singular value of P)
# =============================================================================

# ---------- User inputs ----------
# Sampling and latency (EDIT these for your system)
T_s = 1/1000.0        # [s] WFS/RTC sample time (e.g. 1 kHz)
m_delay = 2           # [frames] total integer-frame delay (exposure+RTC+DM)
tau = m_delay * T_s   # [s] total latency

# Misalignment sweep (wvl/D); use x-axis only here
misalign_grid = np.linspace(0.0, 3.0, 13)   # 0 → 3 (wvl/D)

# Small poke size for finite differences (rad RMS)
epsilon = 0.01

# Which cold-stop diameter/configuration to analyze
use_coldstop_diam = coldstop_diam           # or set None if you want it off

# ---------- Helper: run the pipeline with a given modal command and misalignment ----------
def run_modal_command(cmd_tip_rad, cmd_tilt_rad, misalign_wvld):
    """Apply commanded [tip, tilt] in rad RMS on the high-res pupil and return reconstructed [tip, tilt]."""
    phi_in = cmd_tip_rad * tip + cmd_tilt_rad * tilt # - 0.5 * basis[2]
    _, Ic_hr = util.get_theoretical_reference_pupils_with_aber(
        wavelength=wvl, F_number=F_number,
        mask_diam=mask_diam, coldstop_diam=use_coldstop_diam,
        coldstop_misalign=(misalign_wvld, 0.0),   # shift along +x
        eta=eta, phi=phi_in, diameter_in_angular_units=True,
        phaseshift=util.get_phasemask_phaseshift(
            wvl=wvl, depth=phasemask_parameters[mask]['depth'], dot_material='N_1405'
        ),
        padding_factor=6, analytic_solution=False, debug=False
    )
    t_hat, r_hat = project_to_modes(Ic_hr)   # [tip, tilt] in rad (RMS units)
    return np.array([t_hat, r_hat])

# ---------- Main sweep ----------
G_tip_list = []      # |∂t̂/∂t|
lam_max_list = []    # λ_max(P)  (dominant eigenvalue/sigma)
x_talk_list = []     # cross-coupling ratio |∂t̂/∂r| / |∂t̂/∂t|

for d in misalign_grid:
    # Finite-difference slopes w.r.t. tip
    t_plus  = run_modal_command(+epsilon, 0.0, d)
    t_minus = run_modal_command(-epsilon, 0.0, d)
    dt = (t_plus - t_minus) / (2*epsilon)   # columns for 'tip' excitation

    # Finite-difference slopes w.r.t. tilt
    r_plus  = run_modal_command(0.0, +epsilon, d)
    r_minus = run_modal_command(0.0, -epsilon, d)
    dr = (r_plus - r_minus) / (2*epsilon)   # columns for 'tilt' excitation

    # Small-signal 2x2 plant P(Δ): [t̂; r̂] = P * [t; r]
    P_delta = np.column_stack([dt, dr])     # shape (2,2)

    # SISO tip gain magnitude
    G_tip = np.abs(P_delta[0, 0])           # |∂t̂/∂t|
    G_tip_list.append(G_tip)

    # Cross-talk indicator (optional diagnostic)
    x_talk = np.abs(P_delta[0, 1]) / (G_tip + 1e-16)
    x_talk_list.append(x_talk)

    # Dominant eigen/singular value for MIMO bound
    # (For real 2x2, spectral norm = largest singular value)
    lam_max = np.linalg.svd(P_delta, compute_uv=False)[0]
    lam_max_list.append(lam_max)

G_tip_arr = np.array(G_tip_list)
lam_max_arr = np.array(lam_max_list)
x_talk_arr = np.array(x_talk_list)

# ---------- Delay-limited gain bounds ----------
# g_max ≈ (π/(2 τ)) * 1/|G|
const = np.pi / (2.0 * tau)
gmax_tip  = const / (G_tip_arr + 1e-16)     # SISO (tip-only)
gmax_mimo = const / (lam_max_arr + 1e-16)   # MIMO (dominant mode)

# ---------- Report ----------
print(f"Sample time T_s = {T_s*1e3:.1f} ms, delay m = {m_delay} frames -> τ = {tau*1e3:.1f} ms")
for d, Gt, xk, lm, gt, gm in zip(misalign_grid, G_tip_arr, x_talk_arr, lam_max_arr, gmax_tip, gmax_mimo):
    print(f"Δ={d:4.1f} wvl/D | |∂t̂/∂t|={Gt:.3f}  x-talk={xk:.3f}  λ_max={lm:.3f}  "
          f"g_max^tip={gt:.2f}  g_max^mimo={gm:.2f}")

# ---------- Plots ----------
plt.figure(figsize=(7.2,4.6))
plt.plot(misalign_grid, gmax_tip,  'o-', label='g_max (tip SISO)')
plt.plot(misalign_grid, gmax_mimo, 's-', label='g_max (MIMO λ_max)')
plt.xlabel('Cold-stop misalignment (wvl/D)')
plt.ylabel('Max integrator gain (arb. units)')
plt.title('Delay-limited gain bound vs cold-stop misalignment')
plt.grid(True, alpha=0.3)
plt.yscale('log')
plt.legend()
plt.tight_layout()
plt.show()

plt.figure(figsize=(7.2,3.6))
plt.plot(misalign_grid, G_tip_arr, 'o-', label='|∂t̂/∂t|')
#plt.plot(misalign_grid, x_talk_arr, 's--', label='|∂t̂/∂r| / |∂t̂/∂t|')
plt.xlabel('Cold-stop misalignment (wvl/D)')
plt.ylabel('Sensitivity' )# / cross-talk')
plt.title('ZWFS small-signal sensitivities vs misalignment')
plt.grid(True, alpha=0.3)
plt.legend()
plt.tight_layout()
plt.show()





# ========================== Describing-function analysis ==========================
# Nonlinear measurement y = h(x): reconstructed TIP vs "true TIP" (commanded).
# We build h(x) numerically under a chosen cold-stop misalignment and focus offset,
# then compute the describing function N(A), and the delay-limited limit-cycle gain.

"""
\paragraph{Nonlinear measurement and stability.}
When the ZWFS measurement becomes nonlinear due to cold-stop misalignment and
defocus, the error signal can be written as a static nonlinearity $y = h(x)$,
where $x$ is the true modal phase (e.g.\ tip) and $y$ the reconstructed value.
For the aligned case $h(x)\!\approx\!k x$, but misalignment causes saturation
and slope reversals so that $\tfrac{dh}{dx}$ changes sign. The closed-loop system
then forms a Lur'e feedback structure composed of a linear dynamic element $G(s)$
(integrator + delay) and a static nonlinearity $h(x)$.
Stability can be analysed using \emph{describing functions}
\[
N(A) = \frac{2}{\pi A} \int_0^\pi h(A\sin\theta)\sin\theta\, d\theta,
\]
which quantify the amplitude-dependent loop gain, or by \emph{sector-bounded}
criteria (Circle or Popov tests) if $h(x)$ lies within a sector
$[k_1,k_2]$. The multiple zero crossings observed imply $k_1<0$, so the
effective feedback alternates between stabilising and destabilising, producing
regions of limit-cycle or bistable behaviour even if the small-signal linearised
loop is stable.
"""

# --- Choose operating point (nonlinear curve will depend on these) ---
misalign = (0.5, 0.0)   # cold-stop misalignment (wvl/D)
f0_rad   = 0.5          # static defocus offset [rad RMS] (set 0 for aligned)
# Latency model for the controller (integrator + pure delay)
T_s    = 1/1000.0       # sample time [s]
m_delay= 2              # frames of delay
tau    = m_delay*T_s    # total delay [s]
# Controller: pure integrator C(s)=g/s with delay e^{-s tau}
# At the delay-limited phase crossover: ωc = π/(2 τ)
omega_c = np.pi/(2.0*tau)

# --- Build the static nonlinearity h(x): x -> hat_tip ---
def reconstruct_tip_from_true_tip(x_true_rad):
    """Return reconstructed TIP for a commanded true TIP = x_true_rad (rad RMS),
       with the operating point (misalign, f0_rad)."""
    phi_in = f0_rad*focus + x_true_rad*tip  # inject focus offset + true tip
    _, Ic_hr = util.get_theoretical_reference_pupils_with_aber(
        wavelength=wvl, F_number=F_number,
        mask_diam=mask_diam, coldstop_diam=coldstop_diam,
        coldstop_misalign=misalign, eta=eta, phi=phi_in,
        diameter_in_angular_units=True,
        phaseshift=util.get_phasemask_phaseshift(
            wvl=wvl, depth=phasemask_parameters[mask]['depth'], dot_material='N_1405'
        ),
        padding_factor=6, analytic_solution=False, debug=False
    )
    tip_est, tilt_est = project_to_modes(Ic_hr)
    return tip_est

# --- Describing function N(A) for a static (possibly asymmetric) nonlinearity h(x) ---
# For odd nonlinearities N(A) is real: N(A) = (2/(π A)) ∫_0^π h(A sinθ) sinθ dθ.
# We’ll compute both the real part via that formula and the general complex DF
# using the first-harmonic projection (robust to slight asymmetries).
def describing_function(h_fun, A, nθ=4096):
    θ = np.linspace(0, 2*np.pi, nθ, endpoint=False)
    x = A*np.sin(θ)
    y = np.array([h_fun(xi) for xi in x])
    # Complex DF (first-harmonic): N = <y e^{-jθ}> / <A sinθ e^{-jθ}> = (1/πA)∫ y sinθ dθ  (imag cancels for odd)
    # Use discrete projection on sin(θ):
    Re = (1/np.pi/A) * np.trapz(y*np.sin(θ), θ)     # equals standard real DF for odd h
    Im = (1/np.pi/A) * np.trapz(y*(-np.cos(θ)), θ)  # ~0 for odd h; kept for completeness
    return Re + 1j*Im

# --- Sweep amplitude and compute DF ---
A_grid = np.linspace(0.01, 0.8, 40)  # [rad RMS] (choose range covering your plot's linear→nonlinear)
N_vals = np.array([describing_function(reconstruct_tip_from_true_tip, A) for A in A_grid])

# --- Predict limit-cycle gain for integrator+delay at ωc = π/(2τ) ---
# Magnitude condition at that phase: |G(jωc)*N(A)| = 1, with G(jω)=g/(jω) e^{-jωτ}.
# ⇒ g_lc(A) = ωc / |N(A)|.
g_lc = omega_c / np.maximum(np.abs(N_vals), 1e-16)

print(f"Operating point: misalign={misalign[0]:.2f} (wvl/D), focus offset f0={f0_rad:.2f} rad RMS")
print(f"Delay τ = {tau*1e3:.2f} ms ⇒ ωc = π/(2τ) = {omega_c/(2*np.pi):.1f} Hz crossover at the delay limit.")
print(f"N(A) near A→0 ≈ {N_vals[0].real:.3f} (real), Im≈{N_vals[0].imag:.3e}")

# --- Plots: DF magnitude and limit-cycle gain vs amplitude ---
plt.figure(figsize=(7.2,4.2))
plt.plot(A_grid, np.abs(N_vals), 'o-', label='|N(A)|')
plt.xlabel('Sinusoid amplitude A in true TIP (rad RMS)')
plt.ylabel('|N(A)|  (recon TIP per true TIP)')
plt.title('Describing function magnitude of ZWFS nonlinearity')
plt.grid(True, alpha=0.3); plt.legend(); plt.tight_layout(); plt.show()

plt.figure(figsize=(7.2,4.2))
plt.plot(A_grid, g_lc, 'o-', label=r'$g_\mathrm{lc}(A)=\omega_c/|N(A)|$')
plt.yscale('log')
plt.xlabel('Limit-cycle amplitude A (rad RMS)')
plt.ylabel('Integrator gain for limit cycle,  g_lc(A)')
plt.title('Predicted integrator gain for a limit cycle vs amplitude')
plt.grid(True, which='both', alpha=0.3); plt.legend(); plt.tight_layout(); plt.show()



# to continue... senstivity vs other masks confirm... 



##### END 







# ### THE FUNCTION NOW IN utilities.py

# def get_theoretical_reference_pupils_with_aber( wavelength = 1.65e-6 ,F_number = 21.2, mask_diam = 1.2, coldstop_diam=None, coldstop_misalign=None, eta=0, phi= None, diameter_in_angular_units = True, get_individual_terms=False, phaseshift = np.pi/2 , padding_factor = 4, debug= True, analytic_solution = True ) :
#     """
#     get theoretical reference pupil intensities of ZWFS with / without phasemask 
    

#     Parameters
#     ----------
#     wavelength : TYPE, optional
#         DESCRIPTION. input wavelength The default is 1.65e-6.
#     F_number : TYPE, optional
#         DESCRIPTION. The default is 21.2.
#     mask_diam : phase dot diameter. TYPE, optional
#             if diameter_in_angular_units=True than this has diffraction limit units ( 1.22 * f * lambda/D )
#             if  diameter_in_angular_units=False than this has physical units (m) determined by F_number and wavelength
#         DESCRIPTION. The default is 1.2.
#     coldstop_diam : diameter in lambda / D of focal plane coldstop
#     coldstop_misalign : alignment offset of the cold stop (in units of image plane pixels)  
#     phi : input phase aberrations (None by default). should be same size as pupil which by default is 2D grid of 2**9+1
#     eta : ratio of secondary obstruction radius (r_2/r_1), where r2 is secondary, r1 is primary. 0 meams no secondary obstruction
#     diameter_in_angular_units : TYPE, optional
#         DESCRIPTION. The default is True.
#     get_individual_terms : Type optional
#         DESCRIPTION : if false (default) with jsut return intensity, otherwise return P^2, abs(M)^2 , phi + mu
#     phaseshift : TYPE, optional
#         DESCRIPTION. phase phase shift imparted on input field (radians). The default is np.pi/2.
#     padding_factor : pad to change the resolution in image plane. TYPE, optional
#         DESCRIPTION. The default is 4.
#     debug : TYPE, optional
#         DESCRIPTION. Do we want to plot some things? The default is True.
#     analytic_solution: TYPE, optional
#         DESCRIPTION. use analytic formula or calculate numerically? The default is True.
#     Returns
#     -------
#     Ic, reference pupil intensity with phasemask in 
#     P, reference pupil intensity with phasemask out 

#     """
#     pupil_radius = 1  # Pupil radius in meters

#     # Define the grid in the pupil plane
#     N = 2**9+1  # for parity (to not introduce tilt) works better ODD!  # Number of grid points (assumed to be square)
#     L_pupil = 2 * pupil_radius  # Pupil plane size (physical dimension)
#     dx_pupil = L_pupil / N  # Sampling interval in the pupil plane
#     x_pupil = np.linspace(-L_pupil/2, L_pupil/2, N)   # Pupil plane coordinates
#     y_pupil = np.linspace(-L_pupil/2, L_pupil/2, N) 
#     X_pupil, Y_pupil = np.meshgrid(x_pupil, y_pupil)
    
    


#     # Define a circular pupil function
#     pupil = (np.sqrt(X_pupil**2 + Y_pupil**2) > eta*pupil_radius) & (np.sqrt(X_pupil**2 + Y_pupil**2) <= pupil_radius)
#     pupil = pupil.astype( complex )
#     if phi is not None:
#         pupil *= np.exp(1j * phi)
#     else:
#         phi = np.zeros( pupil.shape ) # added aberrations 
        
#     # Zero padding to increase resolution
#     # Increase the array size by padding (e.g., 4x original size)
#     N_padded = N * padding_factor
#     if (N % 2) != (N_padded % 2):  
#         N_padded += 1  # Adjust to maintain parity
        
#     pupil_padded = np.zeros((N_padded, N_padded)).astype(complex)
#     #start_idx = (N_padded - N) // 2
#     #pupil_padded[start_idx:start_idx+N, start_idx:start_idx+N] = pupil

#     start_idx_x = (N_padded - N) // 2
#     start_idx_y = (N_padded - N) // 2  # Explicitly ensure symmetry

#     pupil_padded[start_idx_y:start_idx_y+N, start_idx_x:start_idx_x+N] = pupil


#     phi_padded = np.zeros((N_padded, N_padded), dtype=float)
#     phi_padded[start_idx_y:start_idx_y+N, start_idx_x:start_idx_x+N] = phi

#     # Perform the Fourier transform on the padded array (normalizing for the FFT)
#     #pupil_ft = np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(pupil_padded))) # we do this laters
    
#     # Compute the Airy disk scaling factor (1.22 * lambda * F)
#     airy_scale = 1.22 * wavelength * F_number

#     # Image plane sampling interval (adjusted for padding)
#     #L_image = wavelength * F_number / dx_pupil  # Total size in the image plane
#     #dx_image_padded = L_image / N_padded  # Sampling interval in the image plane with padding
    
#     dx_image_padded = wavelength * F_number * (N / N_padded)
#     L_image = dx_image_padded * N_padded

#     if diameter_in_angular_units:
#         x_image_padded = np.linspace(-L_image/2, L_image/2, N_padded) / airy_scale  # Image plane coordinates in Airy units
#         y_image_padded = np.linspace(-L_image/2, L_image/2, N_padded) / airy_scale
#     else:
#         x_image_padded = np.linspace(-L_image/2, L_image/2, N_padded)  # Image plane coordinates in Airy units
#         y_image_padded = np.linspace(-L_image/2, L_image/2, N_padded) 
        
#     X_image_padded, Y_image_padded = np.meshgrid(x_image_padded, y_image_padded)

#     if diameter_in_angular_units:
#         mask = np.sqrt(X_image_padded**2 + Y_image_padded**2) <= mask_diam / 2 #4
#     else: 
#         mask = np.sqrt(X_image_padded**2 + Y_image_padded**2) <= mask_diam / 2 #4


#     # --- convert misalignment from wvl/D to your image-plane units ---
#     # ---- cold stop offset: wvl/D -> grid units ----
#     if coldstop_misalign is not None:
#         dx_wvld, dy_wvld = coldstop_misalign
#     else:
#         dx_wvld, dy_wvld = [0.0, 0.0]

#     if diameter_in_angular_units:
#         wvld_to_units = 1.0/1.22            # Airy radii per (wvl/D)
#     else:
#         wvld_to_units = F_number * wavelength  # meters per (wvl/D)
#     dx_units = dx_wvld * wvld_to_units
#     dy_units = dy_wvld * wvld_to_units

#     if coldstop_diam is not None:
#         if diameter_in_angular_units:
#             cs_radius_units = (coldstop_diam * (1.0/1.22)) / 2.0
#         else:
#             cs_radius_units = (coldstop_diam * (F_number * wavelength)) / 2.0
#         coldmask = (np.hypot(X_image_padded - dx_units, Y_image_padded - dy_units) <= cs_radius_units).astype(float)
#     else:
#         coldmask = np.ones_like(X_image_padded)


#     # if coldstop_misalign is not None:
#     #     dx_wvld, dy_wvld = coldstop_misalign
#     # else:
#     #     dx_wvld, dy_wvld = [0,0]
    

#     # if diameter_in_angular_units:
#     #     # Your X_image_padded, Y_image_padded are in "Airy radii" units set by:
#     #     # airy_scale = 1.22 * wavelength * F_number
#     #     # 1 (wvl/D) equals (F_number * wavelength) in meters,
#     #     # which is (1 / 1.22) Airy radii on this normalized grid.
#     #     wvld_to_units = 1.0 / 1.22                      # Airy radii per (wvl/D)
#     #     dx_units = dx_wvld * wvld_to_units
#     #     dy_units = dy_wvld * wvld_to_units
#     # else:
#     #     # Your X_image_padded, Y_image_padded are in meters.
#     #     # 1 (wvl/D) = F_number * wavelength  [meters]
#     #     wvld_to_units = F_number * wavelength           # meters per (wvl/D)
#     #     dx_units = dx_wvld * wvld_to_units
#     #     dy_units = dy_wvld * wvld_to_units
        
#     # # if coldstop_diam is not None:
#     # #     coldmask = np.sqrt(X_image_padded**2 + Y_image_padded**2) <= coldstop_diam / 4
#     # # else:
#     # #     coldmask = np.ones(X_image_padded.shape)
#     # if coldstop_diam is not None: # apply also the cold stop offset 
#     #     coldmask = np.sqrt((X_image_padded-dx_units)**2 + (Y_image_padded-dy_units)**2) <= coldstop_diam / 2 #4
#     # else:
#     #     coldmask = np.ones(X_image_padded.shape)

#     pupil_ft = np.fft.fft2(np.fft.ifftshift(pupil_padded))  # Remove outer fftshift
#     pupil_ft = np.fft.fftshift(pupil_ft)  # Shift only once at the end

#     psi_B = coldmask * pupil_ft
                            
#     b = np.fft.fftshift( np.fft.ifft2( mask * psi_B ) ) # we do mask here because really the cold stop is after phase mask in physical system

    
#     if debug: 
        
#         psf = np.abs(pupil_ft)**2  # Get the PSF by taking the square of the absolute value
#         psf /= np.max(psf)  # Normalize PSF intensity
        
#         if diameter_in_angular_units:
#             zoom_range = 3  # Number of Airy disk radii to zoom in on
#         else:
#             zoom_range = 3 * airy_scale 
            
#         extent = (-zoom_range, zoom_range, -zoom_range, zoom_range)

#         fig,ax = plt.subplots(1,1)
#         ax.imshow(psf, extent=(x_image_padded.min(), x_image_padded.max(), y_image_padded.min(), y_image_padded.max()), cmap='gray')
#         ax.contour(X_image_padded, Y_image_padded, mask, levels=[0.5], colors='red', linewidths=2, label='phasemask')
#         #ax[1].imshow( mask, extent=(x_image_padded.min(), x_image_padded.max(), y_image_padded.min(), y_image_padded.max()), cmap='gray')
#         #for axx in ax.reshape(-1):
#         #    axx.set_xlim(-zoom_range, zoom_range)
#         #    axx.set_ylim(-zoom_range, zoom_range)
#         ax.set_xlim(-zoom_range, zoom_range)
#         ax.set_ylim(-zoom_range, zoom_range)
#         ax.set_title( 'PSF' )
#         ax.legend() 
#         #ax[1].set_title('phasemask')


    
#     # if considering complex b 
#     # beta = np.angle(b) # complex argunment of b 
#     # M = b * (np.exp(1J*theta)-1)**0.5
    
#     # relabelling
#     theta = phaseshift # rad , 
#     P = pupil_padded.copy() 
    
#     if analytic_solution :
        
#         M = abs( b ) * np.sqrt((np.cos(theta)-1)**2 + np.sin(theta)**2)
#         mu = np.angle((np.exp(1J*theta)-1) ) # np.arctan( np.sin(theta)/(np.cos(theta)-1) ) #
        

#         # out formula ----------
#         #if measured_pupil!=None:
#         #    P = measured_pupil / np.mean( P[P > np.mean(P)] ) # normalize by average value in Pupil
#         P = np.abs(pupil_padded).real  # we already dealt with the complex part in this analytic expression which is in phi
#         Ic = ( P**2 + abs(M)**2 + 2* P* abs(M) * np.cos(phi_padded + mu) ) #+ beta)
#         if not get_individual_terms:
#             return( P, Ic )
#         else:
#             return( P, abs(M) , phi+mu )
#     else:
        
#         # phasemask filter 
        
#         T_on = 1
#         T_off = 1
#         H = T_off*(1 + (T_on/T_off * np.exp(1j * theta) - 1) * mask  ) 
        
#         Ic = abs( np.fft.fftshift( np.fft.ifft2( H * psi_B ) ) ) **2 
    
#         return( P, Ic )
