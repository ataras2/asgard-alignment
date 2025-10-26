
"""
Original notes from Gordons Narcissus_plot.m: 

   Explore sensor area of C Red 1 to find how much each area (pixel) is affected by
   visibility of Narcissus mirror holes as seen through cold stop. 
   G. Robertson      Heimdallr 3, 39;  25 Oct 2022
   Move Hdr spots to other side of sensor - 11 June 2023  [4,93]

translated to python by Ben Courtney-Barrer 17-10-25 will add additional trade-off study for cold stop size 
"""

#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt

# ---------- Helpers (identical trig behavior to MATLAB) ----------
def sind(x_deg): return np.sin(np.deg2rad(x_deg))
def cosd(x_deg): return np.cos(np.deg2rad(x_deg))

# ---------- Geometry & overlap mapper (from your Narcissus code) ----------
def narcissus_map(
    N_sens_h=256, N_sens_v=320, pix_size=0.024, bin_fac=4, N_grid=250,
    cold_r=1.1*1.95/2, b_cold=37.4, s_N1=78.0, f_no=121.5, pupil_angle=1.04,
    K1_h=92, K1_v=130, K2_h=87, K2_v=-124, pupil_az_zero=0.0,
    K1_extra=1.2, K2_extra=1.05,
    draw_layout=False, draw_selected_circle=False,
    sel_circ_i=40, sel_circ_j=5
):
    N_bins_h = int(np.floor(N_sens_h / bin_fac))
    N_bins_v = int(np.floor(N_sens_v / bin_fac))
    sensor_map = np.zeros((N_bins_v, N_bins_h), float)

    cold_cone = np.degrees(np.arctan(cold_r / b_cold))
    pupil_indiv_r = s_N1 / (2.0 * f_no)
    c = s_N1 - b_cold
    Narc_scale = -c / b_cold
    Narc_r = np.tan(np.deg2rad(cold_cone)) * s_N1

    K1_Narc_h = K1_h * pix_size * Narc_scale
    K1_Narc_v = K1_v * pix_size * Narc_scale
    K2_Narc_h = K2_h * pix_size * Narc_scale
    K2_Narc_v = K2_v * pix_size * Narc_scale

    pupil_sep = np.tan(np.deg2rad(pupil_angle)) * s_N1
    pupil_az = np.array([0.0, pupil_az_zero, pupil_az_zero + 120.0, pupil_az_zero + 240.0])
    pupil_r = np.array([0.0, pupil_sep, pupil_sep, pupil_sep])

    K1_pupils_h = K1_Narc_h + sind(pupil_az) * pupil_r
    K1_pupils_v = K1_Narc_v + cosd(pupil_az) * pupil_r
    K2_pupils_h = K2_Narc_h + sind(pupil_az) * pupil_r
    K2_pupils_v = K2_Narc_v + cosd(pupil_az) * pupil_r

    if draw_layout:
        plt.figure()
        plt.plot(K1_pupils_h, K1_pupils_v, 'kd', markerfacecolor='black', markersize=2)
        plt.plot(K2_pupils_h, K2_pupils_v, 'gd', markerfacecolor='black', markersize=2)
        plt.axis('equal')

        theta = np.arange(0, 361, 10.0)
        dh = pupil_indiv_r * cosd(theta)
        dv = pupil_indiv_r * sind(theta)
        for k in range(4):
            plt.plot(K1_pupils_h[k] + K1_extra*dh, K1_pupils_v[k] + K1_extra*dv, 'k-')
            plt.plot(K2_pupils_h[k] + K2_extra*dh, K2_pupils_v[k] + K2_extra*dv, 'g-')

    for i in range(1, N_bins_h + 1):
        for j in range(1, N_bins_v + 1):
            sens_h = (bin_fac * i - 0.5 * (N_sens_h + bin_fac)) * pix_size
            sens_v = (bin_fac * j - 0.5 * (N_sens_v + bin_fac)) * pix_size

            Narc_h = sens_h * Narc_scale
            Narc_v = sens_v * Narc_scale

            if draw_selected_circle and i == sel_circ_i and j == sel_circ_j:
                theta = np.arange(0, 361, 10.0)
                plt.plot(Narc_h + Narc_r * cosd(theta), Narc_v + Narc_r * sind(theta), 'r-')

            gh = np.linspace(Narc_h - Narc_r, Narc_h + Narc_r, N_grid)
            gv = np.linspace(Narc_v + Narc_r, Narc_v - Narc_r, N_grid)
            grid_h = np.tile(gh, (N_grid, 1))
            grid_v = np.tile(gv[:, None], (1, N_grid))
            in_circle = ((grid_h - Narc_h) ** 2 + (grid_v - Narc_v) ** 2) <= (Narc_r ** 2)

            overlap = np.zeros((N_grid, N_grid), np.int32)
            rK1 = (pupil_indiv_r * K1_extra) ** 2
            rK2 = (pupil_indiv_r * K2_extra) ** 2
            for k in range(4):
                in_K1 = ((grid_h - K1_pupils_h[k]) ** 2 + (grid_v - K1_pupils_v[k]) ** 2) <= rK1
                in_K2 = ((grid_h - K2_pupils_h[k]) ** 2 + (grid_v - K2_pupils_v[k]) ** 2) <= rK2
                overlap += ((in_K1.astype(np.int8) + in_K2.astype(np.int8)) * in_circle.astype(np.int8))

            sensor_map[j - 1, i - 1] = overlap.sum() * 4.0 / (np.pi * (N_grid ** 2))

    return np.flipud(sensor_map)

def thermal_metrics(smap):
    return {
        "mean": float(np.mean(smap)),
        "p95":  float(np.percentile(smap, 95.0)),
        "max":  float(np.max(smap)),
    }

# ---------- Sweep ----------
def sweep_cold_stop(cold_r_nominal_mm, scales, narcissus_kwargs, use_coupling=True, verbose=True):
    """
    If use_coupling=True, returns metrics of f * Omega_tot  (∝ thermal background).
    If use_coupling=False, returns metrics of f (fractional overlap) only.
    Also returns 'f_mean' etc. for reference.
    """
    rows = []
    b_cold = narcissus_kwargs.get("b_cold")  # must be present
    if b_cold is None:
        raise ValueError("narcissus_kwargs must include 'b_cold'")

    for ii, s in enumerate(scales, 1):
        if verbose:
            print(f"[{ii}/{len(scales)}] cold-stop scale = {s:.3f}")

        cold_r_current = cold_r_nominal_mm * s
        f_map = narcissus_map(cold_r=cold_r_current, **narcissus_kwargs)

        # Per-pixel solid angle set by the cold stop (geometric, not F#):
        Omega_tot = np.pi * (cold_r_current / b_cold) ** 2

        # Thermal-coupling proxy (∝ photons/s/pixel for uniform radiance):
        coupling_map = f_map * Omega_tot  # THIS is what to plot for background vs stop size

        # Choose which metrics to report as the main y-values
        main_map = coupling_map if use_coupling else f_map
        mets = {
            "scale": float(s),
            "cold_r_mm": float(cold_r_current),
            "mean": float(np.mean(main_map)),
            "p95":  float(np.percentile(main_map, 95)),
            "max":  float(np.max(main_map)),
            # keep fractional-overlap summaries too (for debugging/plots)
            "f_mean": float(np.mean(f_map)),
            "f_p95":  float(np.percentile(f_map, 95)),
            "f_max":  float(np.max(f_map)),
            "Omega_tot": float(Omega_tot),
        }
        rows.append(mets)
    return rows

# def sweep_cold_stop(cold_r_nominal_mm, scales, narcissus_kwargs):
#     rows = []
#     for ii, s in enumerate(scales):
#         print(f"calculating scale {ii}/{len(scales)} in coldstop scan")
#         smap = narcissus_map(cold_r=cold_r_nominal_mm * s, **narcissus_kwargs)

#         mets = thermal_metrics(smap)
#         mets.update({"scale": float(s), "cold_r_mm": float(cold_r_nominal_mm * s)})
#         rows.append(mets)
#     return rows

def sweep_cold_stop_seen_only(
    cold_r_nominal_mm, scales, narcissus_kwargs,
    use_coupling=True, verbose=True,
    vis_thresh=0.0,           # small >0 to ignore vanishingly small tails
    baseline_scale=1.0        # used for normalization
):
    """
    Metrics restricted to pixels that 'see' the emission holes through the cold stop.
    - warm region is implicitly the union of the four pupil disks in narcissus_map()
    - 'seeing' pixels are those with f_map > vis_thresh
    Returns a list of dict rows with:
      mean_seen, p95_seen, max_seen over the seeing pixels,
      n_seen (# pixels), and normalized factors vs the baseline scale.
    """
    rows = []
    b_cold = narcissus_kwargs.get("b_cold")
    if b_cold is None:
        raise ValueError("narcissus_kwargs must include 'b_cold'")

    # first pass to capture baseline value
    baseline_mean_seen = None

    for ii, s in enumerate(scales, 1):
        if verbose:
            print(f"[{ii}/{len(scales)}] cold-stop scale = {s:.3f}")

        cold_r_current = cold_r_nominal_mm * s
        # Fractional overlap map (0..1): how much of the cold-stop cone is overlapped
        # by the warm 'holes' (the K1/K2 pupil circles) for each sensor bin:
        f_map = narcissus_map(cold_r=cold_r_current, **narcissus_kwargs)

        # Per-pixel solid angle admitted by the stop (uniform radiance assumption):
        Omega_tot = np.pi * (cold_r_current / b_cold) ** 2  # [sr], up to overall constant
        main_map = f_map * Omega_tot if use_coupling else f_map

        # Pixels that actually see the passed warm regions for this stop size
        seeing_mask = f_map > vis_thresh
        n_seen = int(np.count_nonzero(seeing_mask))

        if n_seen == 0:
            mean_seen = p95_seen = max_seen = 0.0
        else:
            vals = main_map[seeing_mask]
            mean_seen = float(vals.mean())
            p95_seen  = float(np.percentile(vals, 95.0))
            max_seen  = float(vals.max())

        row = {
            "scale": float(s),
            "main_map":main_map,
            "cold_r_mm": float(cold_r_current),
            "Omega_tot": float(Omega_tot),
            "mean_seen": mean_seen,
            "p95_seen":  p95_seen,
            "max_seen":  max_seen,
            "n_seen":    n_seen,
        }
        rows.append(row)

        if np.isclose(s, baseline_scale):
            baseline_mean_seen = mean_seen

    # add normalization vs baseline
    for r in rows:
        if baseline_mean_seen and baseline_mean_seen > 0:
            r["mean_seen_factor"] = r["mean_seen"] / baseline_mean_seen
        else:
            r["mean_seen_factor"] = np.nan

    return rows

if __name__ == "__main__":
    cold_r_nominal = 1.1 * 1.95 / 2.0  # mm (same as before)
    scales = np.arange(0.5,2.5,0.1) #np.linspace(0.5, 2.5, 20) # 0.5× to 3×

    narc_kwargs = dict(
        N_sens_h=256, N_sens_v=320, pix_size=0.024,
        bin_fac=4, N_grid=250,
        b_cold=37.4, s_N1=78.0, f_no=121.5, pupil_angle=1.04,
        K1_h=92, K1_v=130, K2_h=87, K2_v=-124,
        pupil_az_zero=0.0, K1_extra=1.2, K2_extra=1.05
    )
    results = sweep_cold_stop_seen_only(
        cold_r_nominal, scales, narc_kwargs,
        use_coupling=True, verbose=True,
        vis_thresh=0.0, baseline_scale=1.0
    )

    s        = np.array([r["scale"] for r in results])
    meanSeen = np.array([r["mean_seen"] for r in results])
    p95Seen  = np.array([r["p95_seen"]  for r in results])
    maxSeen  = np.array([r["max_seen"]  for r in results])
    facSeen  = np.array([r["mean_seen_factor"] for r in results])
    nSeen    = np.array([r["n_seen"] for r in results])





    ####################################################################################
    # -------- physics-forward band integral (photons → e-/s) --------
    ####################################################################################

    # --- pick baseline & comparison from your 'results' (which include "main_map") ---
    baseline_scale = 1.0
    compare_scale  = 2.0

    def _row_for_scale(rows, s):
        return rows[int(np.argmin([abs(r["scale"] - s) for r in rows]))]

    row0 = _row_for_scale(results, baseline_scale)
    row1 = _row_for_scale(results, compare_scale)
    M0   = np.array(row0["main_map"], dtype=float)   # solid-angle proxy per pixel [sr] via f_map*Ω_tot
    M1   = np.array(row1["main_map"], dtype=float)

    # ----------------- PHYSICS: photons -> e-/s/pixel (includes pixel area) -----------------
    import numpy as np, matplotlib.pyplot as plt
    from matplotlib.colors import LogNorm, SymLogNorm

    h  = 6.62607015e-34
    c  = 2.99792458e8
    kB = 1.380649e-23

    def Lgamma_ph(lam, T):  # photon spectral radiance [phot s^-1 m^-2 sr^-1 m^-1]
        x = (h*c)/(lam*kB*T)
        # use expm1 for stability when x is small
        return (2*c)/(lam**4) / np.expm1(x)

    # ----- instrument / scene assumptions (EDIT THESE) -----
    T_warm  = 290.0                 # K, warm structures
    eps     = 1.00                  # emissivity of the warm holes (1 if truly warm/black)
    tau_sys = 0.75                  # total cold throughput (optics * QE) IF you prefer split, set QE separately below
    QE      = 1.00                  # set to 1.0 if tau_sys already includes QE; else move QE here and use tau_sys for optics
    pix_um  = 24.0                  # pixel pitch in microns
    A_pix   = (pix_um*1e-6)**2      # pixel area [m^2]
    lam1, lam2 = 2.20e-6, 2.30e-6   # band edges [m]

    # Radiance integral over band
    lam = np.linspace(lam1, lam2, 800)
    I_band = np.trapz(Lgamma_ph(lam, T_warm) * eps * tau_sys * QE, lam)  # [e-/s/m^2/sr]

    # Scale factor from map units to e-/s/pixel
    # main_map ≡ Ω_pixel [sr] → e-/s/pixel = I_band * A_pix * Ω_pixel
    scale_to_e_per_s = I_band * A_pix

    B0_e = scale_to_e_per_s * M0
    B1_e = scale_to_e_per_s * M1
    dB_e = B1_e - B0_e

    # ----------------- PLOTTING: log for absolute, symlog for difference -----------------
    # Mask out pixels that see essentially nothing at either size (optional)
    tau_mask = 0.0
    seeing = (M0 > tau_mask) | (M1 > tau_mask)
    B0_plot = np.where(seeing, B0_e, np.nan)
    B1_plot = np.where(seeing, B1_e, np.nan)
    dB_plot = np.where(seeing, dB_e, np.nan)

    # Robust limits for log scales (need positive values)
    def _posvals(a):
        a = a[np.isfinite(a)]
        return a[a > 0]

    B0_pos = _posvals(B0_plot)
    B1_pos = _posvals(B1_plot)
    if B0_pos.size == 0 or B1_pos.size == 0:
        raise ValueError("No positive values to show on a log scale.")

    vmin_log = min(1, 1)
    vmax_log = max(np.percentile(B0_pos, 99.9), np.percentile(B1_pos, 99.9))

    # Symmetric log for signed difference
    dB_abs = np.abs(dB_plot[np.isfinite(dB_plot)])
    dmax   = np.percentile(dB_abs, 99)
    linthresh = max(np.percentile(dB_abs, 5), 1e-12)

    fig, axs = plt.subplots(1, 3, figsize=(13.8, 4.8), constrained_layout=True)
    fs = 15
    # Nominal
    im0 = axs[0].imshow(B0_plot, origin='lower', cmap='viridis',
                        norm=LogNorm(vmin=vmin_log, vmax=vmax_log))
    axs[0].set_title(f'Nominal Cold Stop Diam. ', fontsize=15)
    c0 = plt.colorbar(im0, ax=axs[0], orientation='horizontal', pad=0.08)
    c0.set_label('e⁻/s/pixel' ,fontsize=fs)

    # ×2 cold stop
    im1 = axs[1].imshow(B1_plot, origin='lower', cmap='viridis',
                        norm=LogNorm(vmin=vmin_log, vmax=vmax_log))
    axs[1].set_title(f'×2 Cold Stop Diam.', fontsize=15)
    c1 = plt.colorbar(im1, ax=axs[1], orientation='horizontal', pad=0.08)
    c1.set_label('e⁻/s/pixel',fontsize=fs)

    # Difference
    im2 = axs[2].imshow(dB_plot, origin='lower', cmap='RdBu_r',
                        norm=SymLogNorm(linthresh=linthresh, vmin=-dmax, vmax=+dmax, base=10))
    axs[2].set_title('Difference',fontsize=fs)
    c2 = plt.colorbar(im2, ax=axs[2], orientation='horizontal', pad=0.08)
    c2.set_label('Δ e⁻/s/pixel',fontsize=fs)

    for ax in axs:
        ax.set_xticks([]); ax.set_yticks([])

    # Small physics annotation (so future-you knows what was assumed)
    suptxt = (f"T={T_warm:.0f} K,  ε={eps:.2f}, τ={tau_sys:.2f}, QE={QE:.2f}, "
            f"pixel={pix_um:.0f} μm, band=[{lam1*1e6:.2f},{lam2*1e6:.2f}] μm\n"
            f"I_band={I_band:.2e} e⁻ s⁻¹ m⁻² sr⁻¹  → scale={scale_to_e_per_s:.2e} e⁻ s⁻¹ pix⁻¹ sr⁻¹")
    #plt.suptitle(suptxt, fontsize=9, y=1.02)
    plt.show()

    # Quick summary
    mu0, mu1 = np.nanmean(B0_plot), np.nanmean(B1_plot)
    print(f"Mean background (nominal): {mu0:.3e} e-/s/pix")
    print(f"Mean background (×2 stop): {mu1:.3e} e-/s/pix  (ratio {mu1/mu0:.2f}×)")
    print(f"Mean difference: {np.nanmean(dB_plot):.3e} e-/s/pix")
    ####################################################################################
    ####################################################################################


    # Heatmap comparing fractional increase in background for 2 different coldstop diameters 
    # -------- choose the two sizes to compare --------
    baseline_scale = 1.0     # nominal
    compare_scale  = 2.0     # e.g., doubled radius

    def _get_row_for_scale(rows, s_target):
        # pick the closest available scale in case of float rounding
        idx = int(np.argmin([abs(r["scale"] - s_target) for r in rows]))
        return rows[idx]

    row0 = _get_row_for_scale(results, baseline_scale)
    row1 = _get_row_for_scale(results, compare_scale)

    M0 = np.array(row0["main_map"], dtype=float)   # baseline thermal-coupling map (∝ background per pixel)
    M1 = np.array(row1["main_map"], dtype=float)   # comparison map

    # -------- percent increase map (seeing pixels only) --------
    # seeing pixels = those that see any passed warm region at baseline or comparison
    seeing_mask = (M0 > 0) | (M1 > 0)

    # avoid divide-by-zero: compute % only where baseline is nonzero
    pct_increase = np.full_like(M0, np.nan)
    valid = seeing_mask & (M0 > 0)
    pct_increase[valid] = 100.0 * (M1[valid] - M0[valid]) / M0[valid]

    # -------- plots --------
    # common absolute color scale for the two maps
    abs_vmax = np.nanpercentile(np.concatenate([M0[seeing_mask], M1[seeing_mask]]), 99)

    fig, axs = plt.subplots(1, 3, figsize=(14, 4.5), constrained_layout=True)

    im0 = axs[0].imshow(M0, origin='lower', cmap='viridis', vmin=0, vmax=abs_vmax)
    axs[0].set_title(f'Baseline map (scale={row0["scale"]:.2f})')
    cbar0 = plt.colorbar(im0, ax=axs[0])
    cbar0.set_label('∝ thermal background per pixel')

    im1 = axs[1].imshow(M1, origin='lower', cmap='viridis', vmin=0, vmax=abs_vmax)
    axs[1].set_title(f'Comparison map (scale={row1["scale"]:.2f})')
    cbar1 = plt.colorbar(im1, ax=axs[1])
    cbar1.set_label('∝ thermal background per pixel')

    # percent increase heatmap
    # pick symmetric limits around 0 for diverging colormap (helpful if decrease happens)
    p99 = np.nanpercentile(np.abs(pct_increase[np.isfinite(pct_increase)]), 99)
    vlim = max(10.0, p99)  # keep at least ±10% range
    im2 = axs[2].imshow(pct_increase, origin='lower', cmap='RdBu_r', vmin=-vlim, vmax=+vlim)
    axs[2].set_title(f'Percent change: {row0["scale"]:.2f} → {row1["scale"]:.2f}')
    cbar2 = plt.colorbar(im2, ax=axs[2])
    cbar2.set_label('% increase in thermal background')

    # gray out non-seeing pixels
    for ax in axs:
        ax.set_xticks([]); ax.set_yticks([])
    mask_overlay = np.where(seeing_mask, np.nan, 1.0)
    axs[2].imshow(mask_overlay, origin='lower', cmap='gray', alpha=0.3)

    plt.show()


    ##############
    # 2D plot on statistics of seen pixels background 
    # Absolute thermal coupling (seeing pixels only)
    plt.figure(figsize=(8,5))
    plt.plot(s, meanSeen/meanSeen[np.argmin(abs(s-1))], '-', label='mean (seen pixels)')
    plt.plot(s, p95Seen/meanSeen[np.argmin(abs(s-1))],  '-', label='p95')
    #plt.plot(s, maxSeen/meanSeen[np.argmin(abs(s-1))],  '^-', label='max')
    plt.xlabel("Cold-stop radius / nominal",fontsize=15)
    plt.axvline(1.0, color='red', ls='--', lw=1)
    plt.axhline(1.0, color='red', ls='--', lw=1)
    plt.ylabel("Normalized thermal background per pixel",fontsize=15)
    plt.gca().tick_params(labelsize=15)
    plt.grid(True); plt.legend(fontsize=15); plt.tight_layout(); 
    plt.savefig( '/Users/bencb/Downloads/coldstop_size_bkg_tradeoff.jpg',bbox_inches='tight',dpi=200)
    plt.show()


    # #how many pixels are included at each size
    # plt.figure(figsize=(7.2,3.8))
    # plt.plot(s, nSeen, 'o-')
    # plt.xlabel("Cold-stop radius / nominal"); plt.ylabel("# seeing pixels")
    # plt.grid(True); plt.tight_layout(); plt.show()

    # results = sweep_cold_stop(cold_r_nominal, scales, narc_kwargs, use_coupling=True)

    # s = np.array([r["scale"] for r in results])

    # # Thermal coupling (∝ background)
    # mean = np.array([r["mean"] for r in results])
    # p95  = np.array([r["p95"]  for r in results])
    # mx   = np.array([r["max"]  for r in results])

    # # Fractional overlap only (for comparison)
    # f_mean = np.array([r["f_mean"] for r in results])

    # plt.figure(figsize=(8,5))
    # #plt.plot(s, mean/mean[np.argmin(abs(s-1))], label="thermal coupling mean")
    # #plt.plot(s, p95/mean[np.argmin(abs(s-1))],  label="thermal coupling p95")
    # plt.plot(s, mx/mean[np.argmin(abs(s-1))],   label="thermal coupling max")
    # plt.axvline(1,ls=':',color='red')
    # plt.axhline(1,ls=':',color='red')
    # #plt.plot(s, f_mean, "--", label="fractional overlap mean (for reference)")
    # plt.xlabel("Cold-stop radius / nominal",fontsize=15)
    # plt.ylabel("Normalized thermal background per pixel",fontsize=15)
    # #plt.title("Thermal background vs cold-stop size")
    # plt.grid(True); plt.legend(); plt.show()

    # # results = sweep_cold_stop(cold_r_nominal, scales, narc_kwargs)

    # # s = np.array([r["scale"] for r in results])
    # # mean = np.array([r["mean"] for r in results])
    # # p95  = np.array([r["p95"]  for r in results])
    # # mx   = np.array([r["max"]  for r in results])

    # # plt.figure()
    # # plt.plot(s, mean, label="mean")
    # # plt.plot(s, p95,  label="p95")
    # # plt.plot(s, mx,   label="max")
    # # plt.xlabel("Cold-stop radius / nominal")
    # # plt.ylabel("Thermal proxy (fractional overlap)")
    # # plt.title("Thermal background vs cold-stop size (relative)")
    # # plt.grid(True)
    # # plt.legend()
    # # plt.show()
