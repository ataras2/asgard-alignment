import matplotlib.pyplot as plt
import csv
from datetime import datetime
import sys

# Prompt for log file path or use command line argument
if len(sys.argv) > 1:
    log_path = sys.argv[1]
else:
    log_path = input("Enter path to tempWD log file: ").strip()

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
plt.title("Temperature Monitoring Log")
plt.legend()
plt.tight_layout()
plt.show()
