import matplotlib.pyplot as plt
import csv
from datetime import datetime
import sys
import argparse
import os
import glob
import numpy as np
from collections import deque

parser = argparse.ArgumentParser(description="Plot temperature logs from tempWD.")
parser.add_argument(
    "logfile",
    nargs="?",
    default=None,
    help="Path to log file or integer index (e.g. -1 for most recent log)",
)
parser.add_argument(
    "--window",
    type=float,
    default=20,
    help="Rolling average window size in seconds (default: 20)",
)
args = parser.parse_args()


def get_log_by_index(idx):
    log_dir = os.path.join("data", "templogs")
    files = sorted(
        glob.glob(os.path.join(log_dir, "tempWD_*.log")), key=os.path.getmtime
    )
    if not files:
        raise FileNotFoundError("No log files found in data/templogs.")
    return files[idx]


if args.logfile is None:
    log_path = input("Enter path to tempWD log file: ").strip()
else:
    try:
        idx = int(args.logfile)
        log_path = get_log_by_index(idx)
        print(f"Using log file: {log_path}")
    except ValueError:
        log_path = args.logfile

times = []
probe_names = []
probe_data = []

with open(log_path, "r") as f:
    reader = csv.reader(f)
    header = next(reader)
    probe_names = header[1:]
    for row in reader:
        # Parse time and temperatures
        times.append(datetime.strptime(row[0], "%Y-%m-%d %H:%M:%S"))
        probe_data.append([float(x) if x != "None" else None for x in row[1:]])

# Transpose probe_data to get a list per probe
probe_data = list(zip(*probe_data))

plt.figure(figsize=(10, 6))

# Calculate rolling window size in samples
if len(times) > 1:
    # Compute median sampling interval in seconds
    intervals = [(t2 - t1).total_seconds() for t1, t2 in zip(times[:-1], times[1:])]
    median_interval = np.median(intervals)
    window_samples = max(1, int(round(args.window / median_interval)))
else:
    window_samples = 1

ax = plt.gca()
color_cycle = ax._get_lines.prop_cycler

for i, probe in enumerate(probe_names):
    y = np.array(probe_data[i], dtype=np.float64)
    # Get next color from cycler
    color = next(color_cycle)["color"]
    # Plot true data as crosses (with label for legend)
    plt.plot(times, y, "x", alpha=0.5, label=f"{probe} (raw)", color=color)
    # Rolling average (ignoring None)
    y_masked = np.ma.masked_invalid(y)
    if np.sum(~y_masked.mask) >= window_samples and window_samples > 1:
        valid_idx = ~y_masked.mask
        y_valid = y_masked[valid_idx]
        t_valid = np.array(times)[valid_idx]
        if len(y_valid) >= window_samples:
            roll = np.convolve(
                y_valid, np.ones(window_samples) / window_samples, mode="valid"
            )
            t_roll = t_valid[window_samples - 1 :]
            # Plot rolling average as grey line, no label
            plt.plot(
                t_roll,
                roll,
                color="grey",
                linewidth=1.5,
                alpha=0.8,
                zorder=1,
            )
    # If not enough valid points, skip rolling average

plt.xlabel("Time")
plt.ylabel("Temperature (°C)")
date_only = times[0].strftime("%Y-%m-%d")
plt.title(f"Temperature Monitoring Log - {date_only}")
plt.legend()
plt.tight_layout()
plt.show()
