import matplotlib.pyplot as plt
import csv
from datetime import datetime
import sys
import argparse
import os
import glob

parser = argparse.ArgumentParser(description="Plot temperature logs from tempWD.")
parser.add_argument(
    "logfile",
    nargs="?",
    default=None,
    help="Path to log file or integer index (e.g. -1 for most recent log)",
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
for i, probe in enumerate(probe_names):
    plt.plot(times, probe_data[i], label=probe)

plt.xlabel("Time")
plt.ylabel("Temperature (°C)")
date_only = times[0].strftime("%Y-%m-%d")
plt.title(f"Temperature Monitoring Log - {date_only}")
plt.legend()
plt.tight_layout()
plt.show()
