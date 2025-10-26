#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import annotations
"""
e.g. 

python playground/phasemask_stability.py --plot-all-beams --start 2025-09-13T13:00:00 --end 2025-09-20T13:00:00 --masks H3 --recenter --shade-one-lambda 

python playground/phasemask_stability.py /Volumes/NO\ NAME/QB_computer/baldr_comissioning/phasemask_positions/beam3/ --start 2025-03-13T13:00:00 --end 2025-10-17T13:00:00 --masks H3 --recenter --shade-one-lambda


python playground/phasemask_stability.py /Volumes/NO\ NAME/QB_computer/baldr_comissioning/phasemask_positions/beam1/ --start 2025-09-13T13:00:00 --end 2025-09-17T13:00:00 --masks H3 


python phasemask_stability.py /path/to/folder \
  --beam 1 \
  --start 2025-07-10T13:00:00 \
  --end   2025-07-10T16:00:00 \
  --out beam1_2025-07-10_range \
  --title "Beam 1"


python phasemask_stability.py /path/to/folder \
  --beam 1 \
  --dates 2025-07-10T13:24:41 2025-07-10T14:00:00 \
  --out beam1_selected \
  --title "Beam 1 (selected)"
"""

"""
Phasemask stability plots.

Looks for files named:
  phase_positions_beam<id>_YYYY-MM-DDTHH-MM-SS.json

Each JSON is expected to map mask names (e.g. "H3","J2") to positions,
either as [x, y] or {"x": ..., "y": ...} (units = detector pixels).

Features
- Recursive file search (optional)
- Filter by start/end datetime or by a list of YYYY-MM-DD dates
- Plot single-beam time series (x and y) OR 2x4 “all beams” panel
- Case-insensitive mask selection (default: H3)
- Optional recentering about the mean and shading ±1 λ/D (λ/D = F# * λ)
"""

#from __future__ import annotations
import argparse
import json
import re
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Iterable

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

# ----------------------------- Parsing ---------------------------------

FNAME_RE = re.compile(
    r"""phase_positions_beam(?P<beam>\d+)_
        (?P<date>\d{4}-\d{2}-\d{2})T
        (?P<h>\d{2})-(?P<m>\d{2})(?:-(?P<s>\d{2}))?
        \.json$""",
    re.VERBOSE,
)

@dataclass(frozen=True)
class FileInfo:
    path: Path
    dt: datetime   # naive (local) datetime
    beam: int      # 1..4


import matplotlib.dates as mdates
from matplotlib.dates import AutoDateLocator, ConciseDateFormatter
def format_time_axis(ax, tz=None, fs=15):
    """Compact, readable time axis."""
    locator = AutoDateLocator(minticks=3, maxticks=7, tz=tz)
    formatter = ConciseDateFormatter(locator)
    # Optional: drop seconds for extra compactness
    formatter.formats = ['%Y', '%b', '%d', '%H:%M', '%H:%M', '%H:%M']
    formatter.zero_formats = [''] * 6
    formatter.offset_formats = [''] * 6
    ax.xaxis.set_major_locator(locator)
    ax.xaxis.set_major_formatter(formatter)
    ax.tick_params(axis='x', labelsize=fs)

def parse_dt_from_filename(p: Path) -> FileInfo | None:
    m = FNAME_RE.search(p.name)
    if not m:
        return None
    beam = int(m.group("beam"))
    y, M, d = map(int, m.group("date").split("-"))
    h = int(m.group("h")); mi = int(m.group("m"))
    s = int(m.group("s") or 0)
    return FileInfo(path=p, dt=datetime(y, M, d, h, mi, s), beam=beam)

def parse_iso_dt_or_none(s: str | None) -> datetime | None:
    if not s:
        return None
    # Accept YYYY-MM-DD or full YYYY-MM-DDTHH:MM or with seconds
    try:
        return datetime.fromisoformat(s.replace("Z",""))
    except Exception:
        raise argparse.ArgumentTypeError(f"Could not parse datetime: {s}")

def parse_dates_list(csv: str | None) -> list[datetime] | None:
    if not csv:
        return None
    out = []
    for tok in csv.split(","):
        tok = tok.strip()
        try:
            d = datetime.fromisoformat(tok)  # allow YYYY-MM-DD or full
        except Exception:
            raise argparse.ArgumentTypeError(f"Bad date: {tok}")
        out.append(d)
    return out

# ---------------------------- Discovery --------------------------------

def discover_files(folder: Path, recursive: bool) -> list[FileInfo]:
    pattern = "phase_positions_beam*_*.json"
    it = folder.rglob(pattern) if recursive else folder.glob(pattern)
    out: list[FileInfo] = []
    for p in it:
        fi = parse_dt_from_filename(p)
        if fi:
            out.append(fi)
    return sorted(out, key=lambda f: (f.beam, f.dt, f.path.name))

def filter_files(files: list[FileInfo],
                 start: datetime | None,
                 end: datetime | None,
                 dates: list[datetime] | None,
                 beam: int | None,
                 debug: bool=False) -> list[FileInfo]:
    if start and end and start > end:
        start, end = end, start
    dates_set = {d.date() for d in dates} if dates else None

    kept = []
    for fi in files:
        if beam is not None and fi.beam != beam:
            continue
        if dates_set is not None:
            if fi.dt.date() not in dates_set:
                continue
        else:
            if start and fi.dt < start:
                continue
            if end and fi.dt > end:
                continue
        kept.append(fi)

    if debug:
        print(f"[filter] matched {len(kept)} files")
        for k in kept:
            print(f"  beam={k.beam} dt={k.dt} {k.path}")
    return kept

# --------------------------- JSON loading -------------------------------

def normalize_mask_list(masks: Iterable[str] | None) -> set[str]:
    if not masks:
        return {"h3"}
    return {m.lower() for m in masks}

def extract_xy(obj: dict) -> tuple[float, float] | None:
    # Accept {"x": .., "y": ..} or [x, y]
    if isinstance(obj, dict):
        if "x" in obj and "y" in obj:
            return float(obj["x"]), float(obj["y"])
    if isinstance(obj, (list, tuple)) and len(obj) >= 2:
        return float(obj[0]), float(obj[1])
    return None

def load_positions(fi: FileInfo, wanted_masks: set[str]) -> dict[str, tuple[float,float]]:
    with fi.path.open("r") as f:
        data = json.load(f)
    out = {}
    for k, v in data.items():
        if not isinstance(k, str):
            continue
        if k.lower() not in wanted_masks:
            continue
        xy = extract_xy(v)
        if xy is not None:
            out[k] = xy
    return out

# ----------------------------- Plotting ---------------------------------

def add_stats_lines(ax, values, color="0.3", fs=15):
    if len(values) == 0:
        return
    mu = float(np.nanmean(values))
    p5 = float(np.nanpercentile(values, 5))
    p95 = float(np.nanpercentile(values, 95))
    ax.axhline(mu, color=color, lw=1.8, ls="--", label="mean")
    ax.axhline(p5, color=color, lw=1.0, ls=":")
    ax.axhline(p95, color=color, lw=1.0, ls=":", label="5/95%")
    ax.legend(fontsize=fs-1,loc="lower left") #fontsize=fs-1, loc="upper left")

def shade_one_lambda(ax, center, width, label, fs=15):
    ax.axhspan(center - width, center + width,
               color="tab:blue", alpha=0.10, label=label)
    #ax.plot([], [], color="tab:blue", alpha=0.40, lw=6, label=label)
    # keep legend clean; user sees shading

def make_time_axis(ax, fs=15):
    ax.xaxis.set_major_locator(mdates.AutoDateLocator())
    ax.xaxis.set_major_formatter(mdates.ConciseDateFormatter(ax.xaxis.get_major_locator()))
    for item in (ax.xaxis.get_majorticklabels() + ax.yaxis.get_majorticklabels()):
        item.set_fontsize(fs)

# ---------------------- Plotting backends -------------------------------

def plot_single_beam(times, xs, ys, beam_id: int, mask_label: str,
                     recenter: bool, shade_width: float | None,
                     fs=15):
    fig, (axx, axy) = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
    fig.suptitle(f"Beam {beam_id}  mask {mask_label}", fontsize=fs+1)

    xx = np.array(xs, float)
    yy = np.array(ys, float)

    if recenter:
        xx = xx - np.nanmean(xx)
        yy = yy - np.nanmean(yy)

    axx.plot(times, xx, "o", label="x", color="tab:orange")
    axy.plot(times, yy, "o", label="y", color="tab:green")

    add_stats_lines(axx, xx, fs=fs)
    add_stats_lines(axy, yy, fs=fs)

    if shade_width is not None:
        shade_one_lambda(axx, center=0.0 if recenter else float(np.nanmean(xx)),
                         width=shade_width, label="±1 λ/D", fs=fs)
        shade_one_lambda(axy, center=0.0 if recenter else float(np.nanmean(yy)),
                         width=shade_width, label="±1 λ/D", fs=fs)


    for ax, lab in [(axx, "x [pix]"), (axy, "y [pix]")]:
        ax.set_ylabel(lab, fontsize=fs)
        ax.grid(alpha=0.25)
        make_time_axis(ax, fs=fs)
    #format_time_axis(axx, fs=fs)
    #format_time_axis(axy, fs=fs)

    axy.set_xlabel("Date", fontsize=fs)
    plt.tight_layout()
    plt.show()


############

def _moving_avg(v, w=9):
    """Simple centered moving average for smoothing (odd w)."""
    w = int(max(1, w))
    if w % 2 == 0: w += 1
    if v.size < 3: return v.copy()
    k = np.ones(w, float) / w
    return np.convolve(v, k, mode="same")

def remove_step_offsets(times, x, y, step_thresh=200.0, smooth_win=9, min_sep=3):
    """
    Detect large step changes and subtract a piecewise-constant offset so that
    the series becomes continuous across realignment events.

    Parameters
    ----------
    times : list of datetime or array-like convertible by mdates.date2num
    x, y  : 1D arrays (same length)
    step_thresh : float
        Threshold on sqrt(dx^2+dy^2) (in the same units as x,y) to declare a step.
    smooth_win : int
        Window for moving-average smoothing before diff.
    min_sep : int
        Minimum number of samples between detected steps (debounce).

    Returns
    -------
    x_corr, y_corr : arrays
        Corrected series with step offsets removed.
    step_idx : list of int
        Indices (into x,y) where steps were detected (start of new segment).
    """
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    n = x.size
    if n == 0:
        return x, y, []

    xs = _moving_avg(x, smooth_win)
    ys = _moving_avg(y, smooth_win)

    dx = np.diff(xs)
    dy = np.diff(ys)
    jump_mag = np.hypot(dx, dy)

    # Candidate steps
    cand = np.where(jump_mag > float(step_thresh))[0] + 1
    # Debounce
    step_idx = []
    last = -10**9
    for j in cand:
        if j - last >= int(min_sep):
            step_idx.append(int(j))
            last = j

    # Build piecewise-constant offsets to remove steps
    off_x = np.zeros(n, float)
    off_y = np.zeros(n, float)
    cum_x = 0.0
    cum_y = 0.0
    for j in step_idx:
        # Size of the step from smoothed series
        stepx = xs[j] - xs[j-1]
        stepy = ys[j] - ys[j-1]
        cum_x += stepx
        cum_y += stepy
        off_x[j:] -= stepx
        off_y[j:] -= stepy

    x_corr = x + off_x
    y_corr = y + off_y
    return x_corr, y_corr, step_idx
##############

from datetime import timedelta
def aggregate_xy_by_time(times, x, y, window_hours=1.0, reducer=np.nanmean, min_count=1):
    """
    Group samples that occur within 'window_hours' of each other and replace them
    by a single aggregated sample. Aggregation uses 'reducer' for both time and values:
      - time is averaged in POSIX seconds then converted back to datetime
      - x and y are reduced with the same 'reducer' (default: nanmean)

    Parameters
    ----------
    times : array-like of datetime
    x, y  : array-like of float (same length as times)
    window_hours : float, grouping window in hours (default 1.0)
    reducer : callable, e.g. np.nanmean, np.nanmedian
    min_count : int, minimum number of points to form a group; groups smaller
                than this are kept as-is (i.e. no special handling)

    Returns
    -------
    t_out : np.ndarray of datetime
    x_out : np.ndarray of float
    y_out : np.ndarray of float
    """
    if len(times) == 0:
        return np.asarray(times), np.asarray(x, float), np.asarray(y, float)

    # Convert to numpy and sort by time
    t = np.asarray(times)
    x = np.asarray(x, float)
    y = np.asarray(y, float)

    # Drop rows with non-finite x or y
    ok = np.isfinite(x) & np.isfinite(y)
    t = t[ok]; x = x[ok]; y = y[ok]
    if t.size == 0:
        return t, x, y

    idx = np.argsort(t)
    t = t[idx]; x = x[idx]; y = y[idx]

    # Precompute POSIX seconds for averaging time
    t_sec = np.array([ti.timestamp() for ti in t], dtype=float)
    win = timedelta(hours=window_hours)

    t_out, x_out, y_out = [], [], []
    i = 0
    n = t.size
    while i < n:
        j = i + 1
        # Grow the group while time span ≤ window
        while j < n and (t[j] - t[i]) <= win:
            j += 1

        # Group is [i, j)
        grp_slice = slice(i, j)
        if (j - i) >= min_count:
            # Aggregate
            t_mean = reducer(t_sec[grp_slice])
            x_mean = reducer(x[grp_slice])
            y_mean = reducer(y[grp_slice])
            t_out.append(np.datetime64(int(t_mean), 's').astype('datetime64[s]').astype('datetime64[ms]').astype('datetime64[ms]').astype(object))
            # The above conversion ensures a Python datetime; shorter route:
            # from datetime import datetime
            # t_out.append(datetime.fromtimestamp(t_mean))
            x_out.append(x_mean)
            y_out.append(y_mean)
        else:
            # Single point (or too few): keep as-is
            t_out.append(t[i])
            x_out.append(x[i])
            y_out.append(y[i])

        i = j  # next group

    return np.array(t_out, dtype=object), np.array(x_out, float), np.array(y_out, float)

def plot_all_beams(panel_data: dict[int, tuple[list, list, list, str]],
                   recenter: bool, shade_width: float | None, fs=15):
    """
    panel_data[beam] = (times, xs, ys, mask_label)
    Layout: 2 rows (x,y) × 4 cols (beam 1..4)
    """
    fig, axes = plt.subplots(2, 4, figsize=(16, 7), sharex=True)
    #fig.suptitle("Phasemask positions (all beams)", fontsize=fs+1)

    for b in range(1, 5):
        axx = axes[0, b-1]
        axy = axes[1, b-1]
        if b not in panel_data or len(panel_data[b][0]) == 0:
            axx.text(0.05, 0.8, f"No data (beam {b})", transform=axx.transAxes, fontsize=fs)
            axy.set_visible(False)
            axx.set_visible(True)
            continue

        times, xs, ys, mask_label = panel_data[b]

        # xx = np.array(xs, float)
        # yy = np.array(ys, float)
        # xc, yc, steps = remove_step_offsets(times, xx, yy,
        #                             step_thresh=200,
        #                             smooth_win=9,
        #                             min_sep=3)
    
        xx = np.array(xs, float)
        yy = np.array(ys, float)
        times = np.array(times)


        # ---- simple step-removal by cumulative offset (do BEFORE recenter/filter) ----
        # sort by time
        idx = np.argsort(times)
        times = times[idx]
        xx    = xx[idx]
        yy    = yy[idx]

        thr = 90.0  # μm jump threshold

        # X axis: build cumulative offset so that any jump > thr is removed from all following samples
        dx = np.diff(xx)
        jumps_x = np.where(np.abs(dx) > thr)[0]           # indices of steps (between i and i+1)
        offx = np.zeros_like(xx)
        for j in jumps_x:
            offx[j+1:] += dx[j]                           # subtract this step from all subsequent points
        xx = xx - offx

        # Y axis: same
        dy = np.diff(yy)
        jumps_y = np.where(np.abs(dy) > thr)[0]
        offy = np.zeros_like(yy)
        for j in jumps_y:
            offy[j+1:] += dy[j]
        yy = yy - offy
        # ---- now proceed with your recentering / filtering / plotting ----

        if recenter:
            xx = xx - np.nanmean(xx)
            yy = yy - np.nanmean(yy)

        # --- simple outlier filter & re-centering ---
        clip_thresh = 1e6 # 20000.0  # µm (tune as needed)
        filt = (np.abs(xx) < clip_thresh) & (np.abs(yy) < clip_thresh) #(np.abs(xx - np.median(xx)) < clip_thresh) & (np.abs(yy - np.median(yy)) < clip_thresh) #(xx > clip_thresh) & (yy > clip_thresh)#(np.abs(xx) < clip_thresh) & (np.abs(yy) < clip_thresh)
        #(np.abs(xx) < clip_thresh) & (np.abs(yy) < clip_thresh) #
        # Apply filter to all three consistently
        xx_f = xx[filt]
        yy_f = yy[filt]
        times_f = times[filt]

        # Recenter again after clipping (optional)
        if recenter and xx_f.size:
            xx_f = xx_f - np.nanmean(xx_f)
        if recenter and yy_f.size:
            yy_f = yy_f - np.nanmean(yy_f)

        times_a, xx_a, yy_a = aggregate_xy_by_time(times_f, xx_f, yy_f, window_hours=1.0,
                                           reducer=np.nanmean, min_count=1)
        
        # Now plot the aggregated points (dots only)
        axx.plot(times_a, xx_a, "o", color="tab:orange", markersize=5)
        axy.plot(times_a, yy_a, "o", color="tab:green",  markersize=5)

        # Now plot
        # axx.plot(times_f, xx_f, "o", color="tab:orange")
        # axy.plot(times_f, yy_f, "o", color="tab:green")

        axx.set_ylim([-250,250])
        axy.set_ylim([-250,250])
        # --- end simple filter ---
        # filt = abs(xx) < 1e9 #(abs(xx - np.median(xx)) < 300) & (abs(yy - np.median(yy)) < 300)

        # axx.plot(np.array(times)[filt], np.array(xx)[filt], "o", color="tab:orange")
        # axy.plot(np.array(times)[filt], np.array(yy)[filt], "o", color="tab:green")

        #axx.plot(np.array(times), np.array(xc), "o", color="tab:orange")
        #axy.plot(np.array(times), np.array(yc), "o", color="tab:green")

        if shade_width is not None:
            shade_one_lambda(axx, center=0.0 if recenter else float(np.nanmean(xx)),
                             width=shade_width, label="±1 λ/D", fs=fs)
            shade_one_lambda(axy, center=0.0 if recenter else float(np.nanmean(yy)),
                             width=shade_width, label="±1 λ/D", fs=fs)

        add_stats_lines(axx, xx, fs=fs)
        add_stats_lines(axy, yy, fs=fs)


        axx.set_title(f"Beam {b}  mask {mask_label}", fontsize=fs)
        axx.set_ylabel(r"$\Delta$ x [$\mu$m]", fontsize=fs)
        axy.set_ylabel(r"$\Delta$ y [$\mu$m]", fontsize=fs)

        for ax in (axx, axy):
            ax.grid(alpha=0.25)
            #plt.xticks(rotation=45)
            for tick in ax.get_xticklabels():
                tick.set_rotation(45)
            #labels = [item.get_text() for item in ax.get_xticklabels()]
            #ax.set_xticklabels(labels, rotation=45)
            #make_time_axis(ax, fs=fs)
            #format_time_axis(ax, fs=fs)
    # all_times = [t for d in data.values() for t in d['times']]
    # if len(all_times):
    #     xmin, xmax = min(all_times), max(all_times)
    # else:
    #     xmin = xmax = None

    # for row in range(2):
    #     for col in range(4):
    #         ax = axes[row, col]
    #         format_time_axis(ax, fs=fs)
    #         # only bottom row shows tick labels + xlabel
    #         if row == 0:
    #             ax.tick_params(axis='x', labelbottom=False)
    #             ax.set_xlabel('')
    #         else:
    #             ax.set_xlabel('Date', fontsize=fs)
    #         # common x-limits if we have any data
    #         if xmin is not None:
    #             ax.set_xlim(xmin, xmax)

    # Put a global x label on the bottom row
    for ax in axes[1, :]:
        ax.set_xlabel("Date", fontsize=fs)

    plt.tight_layout()
    plt.show()

# ------------------------------ Main -----------------------------------

def main():
    ap = argparse.ArgumentParser(
        description="Plot phasemask positions vs time.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    ap.add_argument("folder", type=Path, help="Root folder containing JSON files")
    ap.add_argument("--start", type=parse_iso_dt_or_none, default=None,
                    help="Start datetime (e.g. 2025-09-13T13:00:00)")
    ap.add_argument("--end", type=parse_iso_dt_or_none, default=None,
                    help="End datetime (inclusive)")
    ap.add_argument("--dates", type=parse_dates_list, default=None,
                    help="Comma-separated list of YYYY-MM-DD (or ISO datetimes)")

    ap.add_argument("--beam", type=int, default=None,
                    help="Only plot this beam (1..4). Ignored if --plot-all-beams.")
    ap.add_argument("--plot-all-beams", action="store_true",
                    help="Build 2x4 panel of beams 1..4 from the entire tree.")
    ap.add_argument("--recursive", action="store_true",
                    help="Search recursively under folder.")
    ap.add_argument("--debug", action="store_true",
                    help="Print matched files and parsed times.")

    ap.add_argument("--masks", nargs="+", default=["H3"],
                    help="Mask names to plot (case-insensitive); first one found per file is used.")
    ap.add_argument("--recenter", action="store_true",
                    help="Subtract mean so the trace is centered at 0.")
    ap.add_argument("--shade-one-lambda", action="store_true",
                    help="Shade ±1 λ/D about mean or 0.")
    ap.add_argument("--fnumber", type=float, default=21.2,
                    help="F-number for λ/D scale at the mask.")
    ap.add_argument("--lambda-um", type=float, default=1.6,
                    help="Wavelength in microns for λ/D scale.")
    ap.add_argument("--fontsize", type=int, default=15,
                    help="Base font size.")

    args = ap.parse_args()

    wanted_masks = normalize_mask_list(args.masks)
    files_all = discover_files(args.folder, recursive=args.recursive)

    if args.plot_all_beams:
        files = filter_files(files_all, args.start, args.end, args.dates,
                             beam=None, debug=args.debug)
        panel: dict[int, tuple[list, list, list, str]] = {}
        for fi in files:
            sel = load_positions(fi, wanted_masks)
            if not sel:
                continue
            # take the first matching mask in a stable order
            k = sorted(sel.keys())[0]
            x, y = sel[k]
            times, xs, ys, _ = panel.get(fi.beam, ([], [], [], k))
            times.append(fi.dt); xs.append(x); ys.append(y)
            panel[fi.beam] = (times, xs, ys, k)
        width = args.fnumber * args.lambda_um if args.shade_one_lambda else None
        plot_all_beams(panel, recenter=args.recenter, shade_width=width, fs=args.fontsize)
        return

    # single-beam mode
    # choose beam preference: CLI arg if given, else infer if all files same beam
    files = filter_files(files_all, args.start, args.end, args.dates,
                         beam=args.beam, debug=args.debug)
    if not files:
        print("No files matched the filters.")
        return
    # If multiple beams present and no --beam, pick the most frequent beam
    if args.beam is None:
        beams = [fi.beam for fi in files]
        unique, counts = np.unique(beams, return_counts=True)
        pick = int(unique[np.argmax(counts)])
        files = [fi for fi in files if fi.beam == pick]
        if args.debug:
            print(f"[info] multiple beams found, using beam {pick}")
        beam_id = pick
    else:
        beam_id = args.beam

    times, xs, ys = [], [], []
    mask_used = None
    for fi in files:
        sel = load_positions(fi, wanted_masks)
        if not sel:
            continue
        k = sorted(sel.keys())[0]    # first chosen
        mask_used = mask_used or k
        x, y = sel[k]
        times.append(fi.dt); xs.append(x); ys.append(y)

    if len(times) == 0:
        print("No matching mask data in the selected files.")
        return

    width = args.fnumber * args.lambda_um if args.shade_one_lambda else None
    plot_single_beam(times, xs, ys, beam_id=beam_id,
                     mask_label=mask_used or "/",
                     recenter=args.recenter, shade_width=width,
                     fs=args.fontsize)

if __name__ == "__main__":
    main()

# import argparse
# import json
# import re
# from pathlib import Path
# from datetime import datetime
# from typing import Iterable, List, Optional, Tuple, Dict

# import numpy as np
# import matplotlib.pyplot as plt
# import matplotlib.dates as mdates

# # -----------------------------
# # Filename parsing & filtering
# # -----------------------------

# FNAME_RE = re.compile(
#     r"^phase_positions_beam(?P<beam>\d+)_(?P<stamp>\d{4}-\d{2}-\d{2}T\d{2}-\d{2}-\d{2})\.json$"
# )

# def parse_filename_dt(p: Path) -> Tuple[int, datetime]:
#     m = FNAME_RE.match(p.name)
#     if not m:
#         raise ValueError(f"Unexpected filename: {p.name}")
#     beam = int(m.group("beam"))
#     ts = datetime.strptime(m.group("stamp"), "%Y-%m-%dT%H-%M-%S")
#     return beam, ts

# def _coerce_dt(x: str) -> datetime:
#     x = x.strip().replace(" ", "T")
#     x = x.replace(":", "-") if "T" in x and ":" in x else x
#     for fmt in ("%Y-%m-%d", "%Y-%m-%dT%H-%M-%S", "%Y-%m-%dT%H-%M"):
#         try:
#             return datetime.strptime(x, fmt)
#         except ValueError:
#             pass
#     raise ValueError(f"Unrecognized datetime: {x}")

# def filter_files_by_range(
#     folder: Path,
#     start: Optional[datetime] = None,
#     end: Optional[datetime] = None,
#     dates: Optional[Iterable[datetime]] = None,
#     beam_id: Optional[int] = None,
# ) -> List[Tuple[Path, datetime]]:
#     files = []
#     for p in Path(folder).glob("phase_positions_beam*_*.json"):
#         try:
#             beam, ts = parse_filename_dt(p)
#         except ValueError:
#             continue
#         if beam_id is not None and beam != beam_id:
#             continue
#         if dates is not None:
#             if ts in set(dates):
#                 files.append((p, ts))
#         else:
#             if start and ts < start:
#                 continue
#             if end and ts > end:
#                 continue
#             files.append((p, ts))
#     files.sort(key=lambda t: t[1])
#     return files

# # -----------------------------
# # Loading & structuring data
# # -----------------------------

# def load_positions(
#     files_with_time: List[Tuple[Path, datetime]],
#     mask_filter: Optional[List[str]] = None
# ) -> Tuple[List[datetime], Dict[str, np.ndarray], Dict[str, np.ndarray]]:
#     """
#     Load JSON files and build per-mask time series for X and Y.
#     mask_filter: list of mask names to keep (e.g., ["H3","J2"]). If None, use all.
#     """
#     times = [ts for _, ts in files_with_time]
#     keys_seen = None
#     all_X: Dict[str, List[float]] = {}
#     all_Y: Dict[str, List[float]] = {}

#     for p, _ts in files_with_time:
#         with open(p, "r") as f:
#             d = json.load(f)

#         # decide which keys to keep
#         if keys_seen is None:
#             all_keys = sorted(d.keys())
#             if mask_filter:
#                 missing = [k for k in mask_filter if k not in all_keys]
#                 if missing:
#                     print(f"[warn] masks not found in {p.name}: {missing}")
#                 keys_seen = [k for k in mask_filter if k in all_keys]
#                 if not keys_seen:
#                     raise KeyError("None of the requested masks were found in the files.")
#             else:
#                 keys_seen = all_keys
#             for k in keys_seen:
#                 all_X[k], all_Y[k] = [], []

#         # ensure presence
#         missing_in_file = set(keys_seen) - set(d.keys())
#         if missing_in_file:
#             raise KeyError(f"File {p.name} missing keys: {missing_in_file}")

#         for k in keys_seen:
#             x, y = d[k]
#             all_X[k].append(float(x))
#             all_Y[k].append(float(y))

#     X = {k: np.asarray(v) for k, v in all_X.items()}
#     Y = {k: np.asarray(v) for k, v in all_Y.items()}
#     return times, X, Y

# # -----------------------------
# # Plotting
# # -----------------------------

# def _global_stats(values: List[np.ndarray]) -> Tuple[float, float, float]:
#     if not values:
#         return np.nan, np.nan, np.nan
#     cat = np.concatenate(values, axis=0)
#     mean = float(np.nanmean(cat))
#     p5 = float(np.nanpercentile(cat, 5))
#     p95 = float(np.nanpercentile(cat, 95))
#     return mean, p5, p95

# def _maybe_recenter(series_dict: Dict[str, np.ndarray], recenter: bool) -> Dict[str, np.ndarray]:
#     if not recenter:
#         return series_dict
#     out = {}
#     for k, v in series_dict.items():
#         mu = np.nanmean(v)
#         out[k] = v - mu
#     return out

# def plot_xy_vs_time(
#     times: List[datetime],
#     X: Dict[str, np.ndarray],
#     Y: Dict[str, np.ndarray],
#     title_prefix: str = "",
#     out_prefix: Optional[Path] = None,
#     figsize=(12, 6),
#     fontsize: int = 15,
#     recenter: bool = False,
#     shade_one_lambda: bool = False,
#     one_lambda_value: float = 21.2 * 1.6,  # default per your H-band note
#     y_units: str = ""
# ):
#     """
#     Two figures: X vs time and Y vs time.
#     Options:
#       - recenter: subtract each mask's mean so series have zero mean.
#       - shade_one_lambda: draw a horizontal band at ± one_lambda_value.
#       - one_lambda_value: numeric width of 1 λ/D in your detector units.
#     """
#     plt.rcParams.update({
#         "font.size": fontsize,
#         "axes.titlesize": fontsize,
#         "axes.labelsize": fontsize,
#         "xtick.labelsize": fontsize,
#         "ytick.labelsize": fontsize,
#         "legend.fontsize": max(10, fontsize-2)
#     })

#     # apply optional recentering (affects stats and plotting)
#     Xp = _maybe_recenter(X, recenter)
#     Yp = _maybe_recenter(Y, recenter)

#     locator = mdates.AutoDateLocator()
#     formatter = mdates.ConciseDateFormatter(locator)

#     # helper to add bands and lines
#     def _decorate_axis(ax, values_list, title, ylabel):
#         m, p5, p95 = _global_stats(values_list)
#         ax.axhline(m,  color='k', lw=1.5, ls='-',  alpha=0.7, label='Mean')
#         ax.axhline(p5, color='k', lw=1.0, ls='--', alpha=0.7, label='P5/P95')
#         ax.axhline(p95, color='k', lw=1.0, ls='--', alpha=0.7)

#         if shade_one_lambda:
#             ax.axhspan(-one_lambda_value, +one_lambda_value, color='C7', alpha=0.15,
#                        label=r'$\pm 1\,\lambda/D$ band')

#         ax.set_title(title)
#         ax.set_xlabel("Time")
#         ax.set_ylabel(ylabel)
#         ax.grid(True, alpha=0.2)

#     # --- X figure ---
#     figX, axX = plt.subplots(figsize=figsize)
#     axX.xaxis.set_major_locator(locator)
#     axX.xaxis.set_major_formatter(formatter)

#     for k in sorted(Xp.keys()):
#         axX.plot(times, Xp[k], marker='o', ms=3, lw=1.2, label=k)

#     _decorate_axis(axX, list(Xp.values()),
#                    f"{title_prefix} X vs time".strip(),
#                    f"X position {('['+y_units+']') if y_units else ''}")

#     axX.legend(frameon=False)

#     figX.tight_layout()
#     if out_prefix:
#         figX.savefig(f"{out_prefix}_X.png", dpi=200)
#         plt.close(figX)
#     else:
#         plt.show()

#     # --- Y figure ---
#     figY, axY = plt.subplots(figsize=figsize)
#     axY.xaxis.set_major_locator(locator)
#     axY.xaxis.set_major_formatter(formatter)

#     for k in sorted(Yp.keys()):
#         axY.plot(times, Yp[k], marker='o', ms=3, lw=1.2, label=k)

#     _decorate_axis(axY, list(Yp.values()),
#                    f"{title_prefix} Y vs time".strip(),
#                    f"Y position {('['+y_units+']') if y_units else ''}")

#     axY.legend(frameon=False)

#     figY.tight_layout()
#     if out_prefix:
#         figY.savefig(f"{out_prefix}_Y.png", dpi=200)
#         plt.close(figY)
#     else:
#         plt.show()

# # -----------------------------
# # CLI
# # -----------------------------

# def main():
#     ap = argparse.ArgumentParser(
#         description="Filter phase-position JSONs by time and plot selected masks' X/Y vs time with mean, 5/95th percentile lines, optional recentering, and ±1 λ/D shading."
#     )
#     ap.add_argument("folder", type=str, help="Folder with phase_positions_beam*_*.json")
#     ap.add_argument("--beam", type=int, default=None, help="Optional beam_id (e.g., 1)")

#     # Either start/end or dates list
#     ap.add_argument("--start", type=str, default=None, help="Start datetime, e.g. '2025-07-10T13:00:00'")
#     ap.add_argument("--end",   type=str, default=None, help="End datetime,   e.g. '2025-07-10T16:00:00'")
#     ap.add_argument("--dates", type=str, nargs="*", default=None,
#                     help="Explicit list of datetimes: 'YYYY-MM-DDTHH:MM[:SS]'")

#     # Mask selection
#     ap.add_argument("--masks", type=str, nargs="*", default=["H3"],
#                     help="Mask names to plot (default: H3). Example: --masks H3 J2")

#     # Plot options
#     ap.add_argument("--out", type=str, default=None, help="Output file prefix (PNG). If omitted, figures are shown.")
#     ap.add_argument("--title", type=str, default="", help="Optional title prefix for plots")
#     ap.add_argument("--units", type=str, default="", help="Y-axis units label, e.g. 'µm'")

#     # Recenter and 1 lambda/D shading
#     ap.add_argument("--recenter", action="store_true",
#                     help="Subtract each selected mask's mean so the plotted mean is at 0.")
#     ap.add_argument("--shade-one-lambda", action="store_true",
#                     help="Shade ±1 λ/D horizontal band.")
#     ap.add_argument("--fnumber", type=float, default=21.2,
#                     help="F-number to compute 1 λ/D = F * λ (default 21.2).")
#     ap.add_argument("--lambda-um", type=float, default=1.6,
#                     help="Wavelength in microns to compute 1 λ/D (default 1.6 for H band).")

#     args = ap.parse_args()

#     folder = Path(args.folder)
#     if not folder.is_dir():
#         raise SystemExit(f"Not a directory: {folder}")

#     start = _coerce_dt(args.start) if args.start else None
#     end   = _coerce_dt(args.end)   if args.end   else None
#     dates = [_coerce_dt(s) for s in args.dates] if args.dates else None

#     files = filter_files_by_range(folder, start=start, end=end, dates=dates, beam_id=args.beam)
#     if not files:
#         raise SystemExit("No files found that match the filter.")

#     times, X_all, Y_all = load_positions(files, mask_filter=args.masks)

#     kept = sorted(X_all.keys())
#     requested = args.masks
#     missing = [m for m in requested if m not in kept]
#     if missing:
#         print(f"[warn] requested masks not present in data (skipped): {missing}")

#     one_lambda_value = args.fnumber * args.lambda_um  # your note: F * λ
#     out_prefix = Path(args.out) if args.out else None

#     plot_xy_vs_time(times, X_all, Y_all,
#                     title_prefix=args.title,
#                     out_prefix=out_prefix,
#                     fontsize=15,
#                     recenter=args.recenter,
#                     shade_one_lambda=args.shade_one_lambda,
#                     one_lambda_value=one_lambda_value,
#                     y_units=args.units)


# if __name__ == "__main__":
#     main()


# #!/usr/bin/env python3
# # -*- coding: utf-8 -*-

# # import argparse
# # import json
# # import re
# # from pathlib import Path
# # from datetime import datetime
# # from typing import Iterable, List, Optional, Tuple, Dict

# # import numpy as np
# # import matplotlib.pyplot as plt
# # import matplotlib.dates as mdates

# # # -----------------------------
# # # Filename parsing & filtering
# # # -----------------------------

# # FNAME_RE = re.compile(
# #     r"^phase_positions_beam(?P<beam>\d+)_(?P<stamp>\d{4}-\d{2}-\d{2}T\d{2}-\d{2}-\d{2})\.json$"
# # )

# # def parse_filename_dt(p: Path) -> Tuple[int, datetime]:
# #     m = FNAME_RE.match(p.name)
# #     if not m:
# #         raise ValueError(f"Unexpected filename: {p.name}")
# #     beam = int(m.group("beam"))
# #     ts = datetime.strptime(m.group("stamp"), "%Y-%m-%dT%H-%M-%S")
# #     return beam, ts

# # def _coerce_dt(x: str) -> datetime:
# #     x = x.strip().replace(" ", "T")
# #     x = x.replace(":", "-") if "T" in x and ":" in x else x
# #     for fmt in ("%Y-%m-%d", "%Y-%m-%dT%H-%M-%S", "%Y-%m-%dT%H-%M"):
# #         try:
# #             return datetime.strptime(x, fmt)
# #         except ValueError:
# #             pass
# #     raise ValueError(f"Unrecognized datetime: {x}")

# # def filter_files_by_range(
# #     folder: Path,
# #     start: Optional[datetime] = None,
# #     end: Optional[datetime] = None,
# #     dates: Optional[Iterable[datetime]] = None,
# #     beam_id: Optional[int] = None,
# # ) -> List[Tuple[Path, datetime]]:
# #     files = []
# #     for p in Path(folder).glob("phase_positions_beam*_*.json"):
# #         try:
# #             beam, ts = parse_filename_dt(p)
# #         except ValueError:
# #             continue
# #         if beam_id is not None and beam != beam_id:
# #             continue
# #         if dates is not None:
# #             if ts in set(dates):
# #                 files.append((p, ts))
# #         else:
# #             if start and ts < start:
# #                 continue
# #             if end and ts > end:
# #                 continue
# #             files.append((p, ts))
# #     files.sort(key=lambda t: t[1])
# #     return files

# # # -----------------------------
# # # Loading & structuring data
# # # -----------------------------

# # def load_positions(
# #     files_with_time: List[Tuple[Path, datetime]],
# #     mask_filter: Optional[List[str]] = None
# # ) -> Tuple[List[datetime], Dict[str, np.ndarray], Dict[str, np.ndarray]]:
# #     """
# #     Load JSON files and build per-mask time series for X and Y.
# #     mask_filter: list of mask names to keep (e.g., ["H3","J2"]). If None, use all.
# #     """
# #     times = [ts for _, ts in files_with_time]
# #     keys_seen = None
# #     all_X: Dict[str, List[float]] = {}
# #     all_Y: Dict[str, List[float]] = {}

# #     for p, _ts in files_with_time:
# #         with open(p, "r") as f:
# #             d = json.load(f)

# #         # Determine which keys to keep
# #         if keys_seen is None:
# #             all_keys = sorted(d.keys())
# #             if mask_filter:
# #                 # case-sensitive match; warn if a requested key isn't present
# #                 missing = [k for k in mask_filter if k not in all_keys]
# #                 if missing:
# #                     print(f"[warn] masks not found in {p.name}: {missing}")
# #                 keys_seen = [k for k in mask_filter if k in all_keys]
# #                 if not keys_seen:
# #                     raise KeyError("None of the requested masks were found in the files.")
# #             else:
# #                 keys_seen = all_keys

# #             for k in keys_seen:
# #                 all_X[k], all_Y[k] = [], []

# #         # Ensure presence of keys_seen in this file
# #         missing_in_file = set(keys_seen) - set(d.keys())
# #         if missing_in_file:
# #             raise KeyError(f"File {p.name} missing keys: {missing_in_file}")

# #         for k in keys_seen:
# #             x, y = d[k]
# #             all_X[k].append(float(x))
# #             all_Y[k].append(float(y))

# #     X = {k: np.asarray(v) for k, v in all_X.items()}
# #     Y = {k: np.asarray(v) for k, v in all_Y.items()}
# #     return times, X, Y

# # # -----------------------------
# # # Plotting
# # # -----------------------------

# # def _global_stats(values: List[np.ndarray]) -> Tuple[float, float, float]:
# #     if not values:
# #         return np.nan, np.nan, np.nan
# #     cat = np.concatenate(values, axis=0)
# #     mean = float(np.nanmean(cat))
# #     p5 = float(np.nanpercentile(cat, 5))
# #     p95 = float(np.nanpercentile(cat, 95))
# #     return mean, p5, p95

# # def plot_xy_vs_time(
# #     times: List[datetime],
# #     X: Dict[str, np.ndarray],
# #     Y: Dict[str, np.ndarray],
# #     title_prefix: str = "",
# #     out_prefix: Optional[Path] = None,
# #     figsize=(12, 6),
# #     fontsize: int = 15,
# # ):
# #     """
# #     Two figures: X vs time and Y vs time.
# #     Plots only the provided mask keys in X, Y dicts.
# #     Horizontal lines: global mean and 5th/95th percentiles over the plotted masks.
# #     """
# #     plt.rcParams.update({
# #         "font.size": fontsize,
# #         "axes.titlesize": fontsize,
# #         "axes.labelsize": fontsize,
# #         "xtick.labelsize": fontsize,
# #         "ytick.labelsize": fontsize,
# #         "legend.fontsize": max(10, fontsize-2)
# #     })

# #     locator = mdates.AutoDateLocator()
# #     formatter = mdates.ConciseDateFormatter(locator)

# #     # --- X figure ---
# #     figX, axX = plt.subplots(figsize=figsize)
# #     axX.xaxis.set_major_locator(locator)
# #     axX.xaxis.set_major_formatter(formatter)

# #     for k in sorted(X.keys()):
# #         axX.plot(times, X[k], marker='o', ms=3, lw=1.2, label=k)

# #     m, p5, p95 = _global_stats(list(X.values()))
# #     axX.axhline(m,  color='k', lw=1.5, ls='-',  alpha=0.7, label='Mean')
# #     axX.axhline(p5, color='k', lw=1.0, ls='--', alpha=0.7, label='P5/P95')
# #     axX.axhline(p95, color='k', lw=1.0, ls='--', alpha=0.7)

# #     axX.set_title(f"{title_prefix} X vs time".strip())
# #     axX.set_xlabel("Time")
# #     axX.set_ylabel("X position")
# #     axX.grid(True, alpha=0.2)
# #     axX.legend(frameon=False)

# #     figX.tight_layout()
# #     if out_prefix:
# #         figX.savefig(f"{out_prefix}_X.png", dpi=200)
# #         plt.close(figX)
# #     else:
# #         plt.show()

# #     # --- Y figure ---
# #     figY, axY = plt.subplots(figsize=figsize)
# #     axY.xaxis.set_major_locator(locator)
# #     axY.xaxis.set_major_formatter(formatter)

# #     for k in sorted(Y.keys()):
# #         axY.plot(times, Y[k], marker='o', ms=3, lw=1.2, label=k)

# #     m, p5, p95 = _global_stats(list(Y.values()))
# #     axY.axhline(m,  color='k', lw=1.5, ls='-',  alpha=0.7, label='Mean')
# #     axY.axhline(p5, color='k', lw=1.0, ls='--', alpha=0.7, label='P5/P95')
# #     axY.axhline(p95, color='k', lw=1.0, ls='--', alpha=0.7)

# #     axY.set_title(f"{title_prefix} Y vs time".strip())
# #     axY.set_xlabel("Time")
# #     axY.set_ylabel("Y position")
# #     axY.grid(True, alpha=0.2)
# #     axY.legend(frameon=False)

# #     figY.tight_layout()
# #     if out_prefix:
# #         figY.savefig(f"{out_prefix}_Y.png", dpi=200)
# #         plt.close(figY)
# #     else:
# #         plt.show()

# # # -----------------------------
# # # CLI
# # # -----------------------------

# # def main():
# #     ap = argparse.ArgumentParser(
# #         description="Filter phase-position JSONs by time and plot selected masks' X/Y vs time with mean and 5/95th percentile lines."
# #     )
# #     ap.add_argument("folder", type=str, help="Folder with phase_positions_beam*_*.json")
# #     ap.add_argument("--beam", type=int, default=None, help="Optional beam_id (e.g., 1)")

# #     # Either start/end or dates list
# #     ap.add_argument("--start", type=str, default=None, help="Start datetime, e.g. '2025-07-10T13:00:00'")
# #     ap.add_argument("--end",   type=str, default=None, help="End datetime,   e.g. '2025-07-10T16:00:00'")
# #     ap.add_argument("--dates", type=str, nargs="*", default=None,
# #                     help="Explicit list of datetimes: 'YYYY-MM-DDTHH:MM[:SS]'")

# #     # Mask selection
# #     ap.add_argument("--masks", type=str, nargs="*", default=["H3"],
# #                     help="Mask names to plot (default: H3). Example: --masks H3 J2")

# #     ap.add_argument("--out", type=str, default=None, help="Output file prefix (PNG). If omitted, figures are shown.")
# #     ap.add_argument("--title", type=str, default="", help="Optional title prefix for plots")

# #     args = ap.parse_args()

# #     folder = Path(args.folder)
# #     if not folder.is_dir():
# #         raise SystemExit(f"Not a directory: {folder}")

# #     start = _coerce_dt(args.start) if args.start else None
# #     end   = _coerce_dt(args.end)   if args.end   else None
# #     dates = [_coerce_dt(s) for s in args.dates] if args.dates else None

# #     files = filter_files_by_range(folder, start=start, end=end, dates=dates, beam_id=args.beam)
# #     if not files:
# #         raise SystemExit("No files found that match the filter.")

# #     times, X_all, Y_all = load_positions(files, mask_filter=args.masks)

# #     # If user requested masks that are missing entirely, warn now
# #     kept = sorted(X_all.keys())
# #     requested = args.masks
# #     missing = [m for m in requested if m not in kept]
# #     if missing:
# #         print(f"[warn] requested masks not present in data (skipped): {missing}")

# #     out_prefix = Path(args.out) if args.out else None
# #     plot_xy_vs_time(times, X_all, Y_all, title_prefix=args.title, out_prefix=out_prefix)

# # if __name__ == "__main__":
# #     main()

# # # import argparse
# # # import json
# # # import re
# # # from pathlib import Path
# # # from datetime import datetime
# # # from typing import Iterable, List, Optional, Tuple, Dict

# # # import numpy as np
# # # import matplotlib.pyplot as plt
# # # import matplotlib.dates as mdates

# # # # -----------------------------
# # # # Filename parsing & filtering
# # # # -----------------------------

# # # FNAME_RE = re.compile(
# # #     r"^phase_positions_beam(?P<beam>\d+)_(?P<stamp>\d{4}-\d{2}-\d{2}T\d{2}-\d{2}-\d{2})\.json$"
# # # )

# # # def parse_filename_dt(p: Path) -> Tuple[int, datetime]:
# # #     """
# # #     Parse 'phase_positions_beam<beam>_YYYY-MM-DDTHH-MM-SS.json' -> (beam_id, datetime)
# # #     """
# # #     m = FNAME_RE.match(p.name)
# # #     if not m:
# # #         raise ValueError(f"Unexpected filename: {p.name}")
# # #     beam = int(m.group("beam"))
# # #     ts = datetime.strptime(m.group("stamp"), "%Y-%m-%dT%H-%M-%S")
# # #     return beam, ts

# # # def _coerce_dt(x: str) -> datetime:
# # #     """
# # #     Accepts ISO-ish 'YYYY-MM-DDTHH:MM:SS' or 'YYYY-MM-DD HH:MM:SS' or just date.
# # #     Hyphens in time are also accepted (from filename-style).
# # #     """
# # #     x = x.strip().replace(" ", "T")
# # #     x = x.replace(":", "-") if "T" in x and ":" in x else x
# # #     fmts = ["%Y-%m-%d", "%Y-%m-%dT%H-%M-%S", "%Y-%m-%dT%H-%M"]
# # #     for fmt in fmts:
# # #         try:
# # #             return datetime.strptime(x, fmt)
# # #         except ValueError:
# # #             pass
# # #     raise ValueError(f"Unrecognized datetime: {x}")

# # # def filter_files_by_range(
# # #     folder: Path,
# # #     start: Optional[datetime] = None,
# # #     end: Optional[datetime] = None,
# # #     dates: Optional[Iterable[datetime]] = None,
# # #     beam_id: Optional[int] = None,
# # # ) -> List[Tuple[Path, datetime]]:
# # #     """
# # #     Return files in 'folder' that match the naming convention and fall within:
# # #       - [start, end] if start/end are given, OR
# # #       - the explicit 'dates' set (exact match on timestamp from the filename)
# # #     Optionally restrict to a specific beam_id.
# # #     Results are sorted by timestamp.
# # #     """
# # #     files = []
# # #     for p in Path(folder).glob("phase_positions_beam*_*.json"):
# # #         try:
# # #             beam, ts = parse_filename_dt(p)
# # #         except ValueError:
# # #             continue
# # #         if beam_id is not None and beam != beam_id:
# # #             continue

# # #         if dates is not None:
# # #             # exact timestamp match against provided list
# # #             date_set = set(dates)
# # #             if ts in date_set:
# # #                 files.append((p, ts))
# # #         else:
# # #             if start and ts < start:
# # #                 continue
# # #             if end and ts > end:
# # #                 continue
# # #             files.append((p, ts))

# # #     files.sort(key=lambda t: t[1])
# # #     return files

# # # # -----------------------------
# # # # Loading & structuring data
# # # # -----------------------------

# # # def load_positions(files_with_time: List[Tuple[Path, datetime]]) -> Tuple[List[datetime], Dict[str, np.ndarray], Dict[str, np.ndarray]]:
# # #     """
# # #     Load each JSON file and build per-key time series for X and Y.
# # #     Returns:
# # #       times: list of datetimes (sorted)
# # #       X: dict key -> np.array of shape (T,) with X values
# # #       Y: dict key -> np.array of shape (T,) with Y values
# # #     """
# # #     times = [ts for _, ts in files_with_time]
# # #     keys_seen = None
# # #     all_X: Dict[str, List[float]] = {}
# # #     all_Y: Dict[str, List[float]] = {}

# # #     for p, _ts in files_with_time:
# # #         with open(p, "r") as f:
# # #             d = json.load(f)

# # #         if keys_seen is None:
# # #             keys_seen = sorted(d.keys())
# # #             for k in keys_seen:
# # #                 all_X[k] = []
# # #                 all_Y[k] = []

# # #         # Ensure consistent keys across files
# # #         missing = set(keys_seen) - set(d.keys())
# # #         if missing:
# # #             raise KeyError(f"File {p.name} missing keys: {missing}")

# # #         for k in keys_seen:
# # #             x, y = d[k]
# # #             all_X[k].append(float(x))
# # #             all_Y[k].append(float(y))

# # #     X = {k: np.asarray(v) for k, v in all_X.items()}
# # #     Y = {k: np.asarray(v) for k, v in all_Y.items()}
# # #     return times, X, Y

# # # # -----------------------------
# # # # Plotting
# # # # -----------------------------

# # # def _global_stats(values: List[np.ndarray]) -> Tuple[float, float, float]:
# # #     """
# # #     Compute global mean, p5, p95 across a list of 1D arrays.
# # #     """
# # #     if not values:
# # #         return np.nan, np.nan, np.nan
# # #     cat = np.concatenate(values, axis=0)
# # #     mean = float(np.nanmean(cat))
# # #     p5 = float(np.nanpercentile(cat, 5))
# # #     p95 = float(np.nanpercentile(cat, 95))
# # #     return mean, p5, p95

# # # def plot_xy_vs_time(
# # #     times: List[datetime],
# # #     X: Dict[str, np.ndarray],
# # #     Y: Dict[str, np.ndarray],
# # #     title_prefix: str = "",
# # #     out_prefix: Optional[Path] = None,
# # #     figsize=(12, 6),
# # #     fontsize: int = 15,
# # # ):
# # #     """
# # #     Two figures: X vs time and Y vs time.
# # #     - Each mask key (H1..H5, J1..J5) plotted as a line with markers.
# # #     - Horizontal lines for global mean and 5th/95th percentiles (across all keys).
# # #     - Saves PNGs if out_prefix is provided; otherwise shows the plots.
# # #     """
# # #     plt.rcParams.update({
# # #         "font.size": fontsize,
# # #         "axes.titlesize": fontsize,
# # #         "axes.labelsize": fontsize,
# # #         "xtick.labelsize": fontsize,
# # #         "ytick.labelsize": fontsize,
# # #         "legend.fontsize": max(10, fontsize-2)
# # #     })

# # #     # Time axis formatter
# # #     locator = mdates.AutoDateLocator()
# # #     formatter = mdates.ConciseDateFormatter(locator)

# # #     # --- X figure ---
# # #     figX, axX = plt.subplots(figsize=figsize)
# # #     axX.xaxis.set_major_locator(locator)
# # #     axX.xaxis.set_major_formatter(formatter)

# # #     for k in sorted(X.keys()):
# # #         axX.plot(times, X[k], marker='o', ms=3, lw=1.2, label=k)

# # #     m, p5, p95 = _global_stats(list(X.values()))
# # #     axX.axhline(m, color='k', lw=1.5, ls='-', alpha=0.7, label='Mean')
# # #     axX.axhline(p5, color='k', lw=1.0, ls='--', alpha=0.7, label='P5/P95')
# # #     axX.axhline(p95, color='k', lw=1.0, ls='--', alpha=0.7)

# # #     axX.set_title(f"{title_prefix} X vs time".strip())
# # #     axX.set_xlabel("Time")
# # #     axX.set_ylabel("X position")
# # #     axX.grid(True, alpha=0.2)
# # #     axX.legend(ncol=2, frameon=False)

# # #     figX.tight_layout()
# # #     if out_prefix:
# # #         figX.savefig(f"{out_prefix}_X.png", dpi=200)
# # #         plt.close(figX)
# # #     else:
# # #         plt.show()

# # #     # --- Y figure ---
# # #     figY, axY = plt.subplots(figsize=figsize)
# # #     axY.xaxis.set_major_locator(locator)
# # #     axY.xaxis.set_major_formatter(formatter)

# # #     for k in sorted(Y.keys()):
# # #         axY.plot(times, Y[k], marker='o', ms=3, lw=1.2, label=k)

# # #     m, p5, p95 = _global_stats(list(Y.values()))
# # #     axY.axhline(m, color='k', lw=1.5, ls='-', alpha=0.7, label='Mean')
# # #     axY.axhline(p5, color='k', lw=1.0, ls='--', alpha=0.7, label='P5/P95')
# # #     axY.axhline(p95, color='k', lw=1.0, ls='--', alpha=0.7)

# # #     axY.set_title(f"{title_prefix} Y vs time".strip())
# # #     axY.set_xlabel("Time")
# # #     axY.set_ylabel("Y position")
# # #     axY.grid(True, alpha=0.2)
# # #     axY.legend(ncol=2, frameon=False)

# # #     figY.tight_layout()
# # #     if out_prefix:
# # #         figY.savefig(f"{out_prefix}_Y.png", dpi=200)
# # #         plt.close(figY)
# # #     else:
# # #         plt.show()

# # # # -----------------------------
# # # # CLI glue
# # # # -----------------------------

# # # def main():
# # #     ap = argparse.ArgumentParser(
# # #         description="Filter phase-position JSONs by time and plot X/Y vs time with mean and 5/95th percentile lines."
# # #     )
# # #     ap.add_argument("folder", type=str, help="Folder containing phase_positions_beam*_*.json files")
# # #     ap.add_argument("--beam", type=int, default=None, help="Optional beam_id to filter (e.g., 1)")
# # #     # Either start/end or dates list
# # #     ap.add_argument("--start", type=str, default=None, help="Start datetime (e.g. '2025-07-10T13:00:00')")
# # #     ap.add_argument("--end", type=str, default=None, help="End datetime (e.g. '2025-07-10T15:00:00')")
# # #     ap.add_argument("--dates", type=str, nargs="*", default=None,
# # #                     help="Explicit list of datetimes to include; format like 'YYYY-MM-DDTHH:MM:SS'")
# # #     ap.add_argument("--out", type=str, default=None, help="Output file prefix (without extension) to save PNGs")
# # #     ap.add_argument("--title", type=str, default="", help="Optional title prefix for plots")

# # #     args = ap.parse_args()

# # #     folder = Path(args.folder)
# # #     if not folder.is_dir():
# # #         raise SystemExit(f"Not a directory: {folder}")

# # #     # Coerce times
# # #     start = _coerce_dt(args.start) if args.start else None
# # #     end = _coerce_dt(args.end) if args.end else None
# # #     dates = [_coerce_dt(s) for s in args.dates] if args.dates else None

# # #     files = filter_files_by_range(folder, start=start, end=end, dates=dates, beam_id=args.beam)
# # #     if not files:
# # #         raise SystemExit("No files found that match the filter.")

# # #     times, X, Y = load_positions(files)
# # #     out_prefix = Path(args.out) if args.out else None
# # #     plot_xy_vs_time(times, X, Y, title_prefix=args.title, out_prefix=out_prefix)

# # # if __name__ == "__main__":
# # #     main()





