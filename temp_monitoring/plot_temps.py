import matplotlib.pyplot as plt
import csv
from datetime import datetime
import sys
import argparse
import os
import glob
import numpy as np
import time

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
parser.add_argument(
    "--interval",
    type=float,
    default=5,
    help="Update interval in seconds (default: 5)",
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

plt.ion()
fig, ax = plt.subplots(figsize=(10, 6))


def load_data():
    times = []
    probe_names = []
    probe_data = []
    try:
        with open(log_path, "r") as f:
            reader = csv.reader(f)
            header = next(reader)
            probe_names = header[1:]
            for row in reader:
                times.append(datetime.strptime(row[0], "%Y-%m-%d %H:%M:%S"))
                probe_data.append(
                    [
                        42.5 + (float(x) - 512) * 0.11 if x != "None" else None
                        for x in row[1:]
                    ]
                )
        if not times:
            return [], [], []
        probe_data = list(zip(*probe_data))
        return times, probe_names, probe_data
    except Exception as e:
        print(f"Error loading data: {e}")
        return [], [], []


while True:
    ax.clear()
    times, probe_names, probe_data = load_data()
    if not times:
        ax.set_title("No data yet...")
        plt.pause(args.interval)
        continue

    # Calculate rolling window size in samples
    if len(times) > 1:
        intervals = [(t2 - t1).total_seconds() for t1, t2 in zip(times[:-1], times[1:])]
        median_interval = np.median(intervals)
        window_samples = max(1, int(round(args.window / median_interval)))
    else:
        window_samples = 1

    for i, probe in enumerate(probe_names):
        y = np.array(probe_data[i], dtype=np.float64)
        color = f"C{i}"
        ax.plot(times, y, "x", alpha=0.5, label=f"{probe} (raw)", color=color)
        y_masked = np.ma.masked_invalid(y)
        if np.sum(~y_masked.mask) >= window_samples and window_samples > 1:
            valid_idx = ~y_masked.mask
            y_valid = y_masked[valid_idx]
            t_valid = np.array(times)[valid_idx]
            if len(y_valid) >= window_samples:
                kernel = np.ones(window_samples) / window_samples
                roll = np.convolve(y_valid, kernel, mode="same")
                edge = window_samples // 2
                roll[:edge] = np.nan
                roll[-edge if edge != 0 else None :] = np.nan
                valid_ma = ~np.isnan(roll)
                ax.plot(
                    t_valid[valid_ma],
                    roll[valid_ma],
                    color="grey",
                    linewidth=1.5,
                    alpha=0.8,
                    zorder=1,
                )
    ax.set_xlabel("Time")
    ax.set_ylabel("Temperature (°C)")
    date_only = times[0].strftime("%Y-%m-%d")
    ax.set_title(f"Temperature Monitoring Log - {date_only}")
    ax.legend()
    fig.tight_layout()
    plt.draw()
    plt.pause(args.interval)
