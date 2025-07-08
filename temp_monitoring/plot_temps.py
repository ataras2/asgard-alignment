import sys
import os
import glob
import csv
from datetime import datetime
import numpy as np
import argparse
import time

from PyQt5 import QtWidgets, QtCore
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

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
parser.add_argument(
    "--lookback",
    type=float,
    default=None,
    help="Number of minutes to look back in the plot (default: all times)",
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


class TempPlotWidget(QtWidgets.QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        # Set height ratios: first subplot double the others
        self.figure = Figure(figsize=(10, 20))
        n_plts = 5
        gs = self.figure.add_gridspec(n_plts, 1, height_ratios=[2, 1, 1, 1, 1], hspace=0.1)
        # Create all axes, sharing x only with the last subplot
        self.axes = [
            self.figure.add_subplot(gs[i, 0])
            for i in range(n_plts)
        ]
        # Now set all except the last to sharex with the last
        for i in range(n_plts - 1):
            self.axes[i].sharex(self.axes[-1])

        self.canvas = FigureCanvas(self.figure)
        layout = QtWidgets.QVBoxLayout()
        layout.addWidget(self.canvas)
        self.setLayout(layout)
        self.figure.subplots_adjust(hspace=0.25)
        self.setWindowTitle("Temperature Monitoring Log")

        # Persistent data structures
        self.times = []
        self.probe_names = []
        self.probe_data = []
        self.file_pos = 0

        self._init_data()

        self.timer = QtCore.QTimer()
        self.timer.timeout.connect(self.update_plot)
        self.timer.start(int(args.interval * 1000))
        self.update_plot()

    def _init_data(self):
        # Read header and any existing data
        self.file_pos = 0
        try:
            with open(log_path, "r") as f:
                reader = csv.reader(f)
                header = next(reader)
                self.probe_names = header[1:]
                for row in reader:
                    if not row:
                        continue
                    self.times.append(datetime.strptime(row[0], "%Y-%m-%d %H:%M:%S"))
                    converted_row = []
                    for name, x in zip(self.probe_names, row[1:]):
                        if x == "None":
                            converted_row.append(None)
                        elif name.endswith(" T") or "setpoint" in name:
                            converted_row.append(42.5 + (float(x) - 512) * 0.11)
                        else:
                            converted_row.append(float(x))
                    self.probe_data.append(converted_row)
                self.file_pos = f.tell()
        except Exception as e:
            print(f"Error initializing data: {e}")
            self.times = []
            self.probe_names = []
            self.probe_data = []
            self.file_pos = 0

    def _append_new_data(self):
        try:
            with open(log_path, "r") as f:
                f.seek(self.file_pos)
                reader = csv.reader(f)
                for row in reader:
                    if not row:
                        continue
                    self.times.append(datetime.strptime(row[0], "%Y-%m-%d %H:%M:%S"))
                    converted_row = []
                    for name, x in zip(self.probe_names, row[1:]):
                        if x == "None":
                            converted_row.append(None)
                        elif name.endswith(" T") or "setpoint" in name:
                            converted_row.append(42.5 + (float(x) - 512) * 0.11)
                        else:
                            converted_row.append(float(x))
                    self.probe_data.append(converted_row)
                self.file_pos = f.tell()
        except Exception as e:
            print(f"Error appending new data: {e}")

    def get_data(self):
        self._append_new_data()
        if not self.times:
            return [], [], []
        # Transpose probe_data for plotting
        probe_data_t = list(zip(*self.probe_data))
        return self.times, self.probe_names, probe_data_t

    def update_plot(self):
        for ax in self.axes:
            ax.clear()
        times, probe_names, probe_data = self.get_data()
        if not times:
            for ax in self.axes:
                ax.set_title("No data yet...")
            self.canvas.draw()
            return

        # --- Filter by lookback window if specified ---
        if args.lookback is not None:
            from datetime import timedelta

            cutoff = times[-1] - timedelta(minutes=args.lookback)
            mask = [t >= cutoff for t in times]
            if any(mask):
                times = [t for t, m in zip(times, mask) if m]
                probe_data = [
                    [v for v, m in zip(probe, mask) if m] for probe in probe_data
                ]
        # --- End lookback filter ---

        # Map probe names to their indices for easy lookup
        probe_idx = {name: i for i, name in enumerate(probe_names)}

        # Define groups for each subplot (probe_names, ylabel, ylims)
        subplot_groups = [
            (
                ["Lower T", "Upper T", "Bench T", "Floor T"],
                "Temperature (°C)",
                (None, None),
            ),  # ylims ignored here
            (["Lower m_pin_val", "Upper m_pin_val"], "m_pin_val", (0, 255)),
            (["Lower integral", "Upper integral"], "Integral", (None, None)),
            (
                ["Lower k_prop", "Upper k_prop", "Lower k_int", "Upper k_int"],
                "PI Coeffs", (0, None)
            ),
            (["outlet5 (MDS)", "outlet6 (C RED)"], "Current (amps)", (0, None)),
        ]

        # Calculate rolling window size in samples
        if len(times) > 1:
            intervals = [
                (t2 - t1).total_seconds() for t1, t2 in zip(times[:-1], times[1:])
            ]
            median_interval = np.median(intervals)
            window_samples = max(1, int(round(args.window / median_interval)))
        else:
            window_samples = 1

        # First subplot: crosses for data, line for moving average
        group, ylabel, _ = subplot_groups[0]
        ax = self.axes[0]

        for i, probe in enumerate(group):
            if probe not in probe_idx:
                continue
            idx = probe_idx[probe]
            y = np.array(probe_data[idx], dtype=np.float64)
            color = f"C{i}"
            # Plot raw data as crosses
            ax.plot(times, y, "x", alpha=0.2, label=f"{probe} (raw)", color=color)
            # Moving average as line
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
                        color=color,
                        linewidth=1.5,
                        alpha=1.0,
                        zorder=1,
                    )

        prev_ylim = ax.get_ylim()

        # --- Add colored patches for y ranges ---
        if times:
            ax.axhspan(15, 18, xmin=0, xmax=1, color="yellow", alpha=0.5, zorder=0)
            ax.axhspan(20, 100, xmin=0, xmax=1, color="red", alpha=0.5, zorder=0)
        # --- End colored patches ---

        # --- Add dashed lines for setpoints if present ---
        for setpoint_name, color in [
            ("Lower setpoint", "C0"),
            ("Upper setpoint", "C1"),
        ]:
            if setpoint_name in probe_idx:
                idx = probe_idx[setpoint_name]
                y = np.array(probe_data[idx], dtype=np.float64)
                ax.plot(
                    times,
                    y,
                    "--",
                    linewidth=1.5,
                    alpha=1.0,
                    color=color,
                    label=f"{setpoint_name} (setpoint)",
                )
        # --- End setpoint lines ---

        ax.set_ylabel(ylabel)
        ax.legend(loc="upper left", fontsize="small")
        date_only = times[0].strftime("%Y-%m-%d")
        ax.set_title(f"Temperature Monitoring Log - {date_only}")
        ax.set_ylim(prev_ylim)

        # Remaining subplots: just lines for data
        for subplot_idx, (group, ylabel, ylims) in enumerate(
            subplot_groups[1:], start=1
        ):
            ax = self.axes[subplot_idx]
            for i, probe in enumerate(group):
                if probe not in probe_idx:
                    continue
                idx = probe_idx[probe]
                y = np.array(probe_data[idx], dtype=np.float64)
                if subplot_idx == 4:
                    color = f"C{i+4}"
                else: 
                    color = f"C{i}"
                ax.plot(
                    times, y, "-", linewidth=1.5, alpha=1.0, label=probe, color=color
                )
            ax.set_ylabel(ylabel)
            ax.legend(loc="upper left", fontsize="small")
            ax.set_ylim(*ylims)
            # Hide xtick labels except for the last subplot
            if subplot_idx != len(self.axes) - 1:
                ax.tick_params(labelbottom=False)

        # Only the last subplot shows xtick labels
        self.axes[-1].set_xlabel("Time")
        self.figure.tight_layout()
        self.canvas.draw()


def main():
    app = QtWidgets.QApplication(sys.argv)
    widget = TempPlotWidget()
    widget.resize(1000, 600)
    widget.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
