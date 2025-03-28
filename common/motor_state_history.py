import streamlit as st
import os
import re
import datetime
from astropy.io import fits
import matplotlib.pyplot as plt
import numpy as np

"""
from asgard-alignment directory
Try streamlit run common/motor_state_history.py. 

This automatically reads and displays the files saved 
from the “quick buttons” -> “quick save state" button in the engineering GUI

"""
st.title("Motor Position Time Series")


# --- Configuration ---
# Base directory where the motor state FITS files are saved.
# Each subdirectory is named in the format "dd-mm-yyyy"
BASE_DIR = "/home/asg/Progs/repos/asgard-alignment/instr_states/stability_analysis/"

# --- User Input for Date Range ---
# Using text input for date range (expected format: dd-mm-yyyy)
default_start = (datetime.date.today() - datetime.timedelta(days=7)).strftime("%d-%m-%Y")
default_end = datetime.date.today().strftime("%d-%m-%Y")

start_date_str = st.text_input("Enter start date (dd-mm-yyyy):", value=default_start)
end_date_str = st.text_input("Enter end date (dd-mm-yyyy):", value=default_end)

try:
    start_date = datetime.datetime.strptime(start_date_str, "%d-%m-%Y").date()
    end_date = datetime.datetime.strptime(end_date_str, "%d-%m-%Y").date()
except ValueError:
    st.error("Invalid date format. Please use dd-mm-yyyy.")
    st.stop()

# --- Locate Subdirectories Within Date Range ---
all_subdirs = [d for d in os.listdir(BASE_DIR) if os.path.isdir(os.path.join(BASE_DIR, d))]
selected_subdirs = []
for subdir in all_subdirs:
    try:
        subdir_date = datetime.datetime.strptime(subdir, "%d-%m-%Y").date()
        if start_date <= subdir_date <= end_date:
            selected_subdirs.append(subdir)
    except ValueError:
        continue

if not selected_subdirs:
    st.write("No directories found in the given date range.")
    st.stop()

# --- Find FITS Files in Selected Subdirectories ---
fits_files = []
for subdir in selected_subdirs:
    subdir_path = os.path.join(BASE_DIR, subdir)
    for file in os.listdir(subdir_path):
        if file.endswith(".fits"):
            fits_files.append(os.path.join(subdir_path, file))

if not fits_files:
    st.write("No FITS files found in the selected directories.")
    st.stop()

# --- Helper Functions ---

def extract_timestamp_from_filename(filename):
    """
    Extracts a timestamp from a filename in the format:
    imgs_n_all_motorstates_dd-mm-YYYYT%H.%M.%S.fits
    """
    pattern = r'(\d{2}-\d{2}-\d{4}T\d{2}\.\d{2}\.\d{2})'
    match = re.search(pattern, filename)
    if match:
        ts_str = match.group(1)
        return datetime.datetime.strptime(ts_str, "%d-%m-%YT%H.%M.%S")
    else:
        # Fall back to file's modification time if pattern not found.
        mod_time = os.path.getmtime(filename)
        return datetime.datetime.fromtimestamp(mod_time)

def read_motor_states_fits(fits_path):
    """
    Reads the "MotorStates" binary table from the FITS file.
    Returns a list of dictionaries for each motor state.
    """
    try:
        with fits.open(fits_path) as hdul:
            data = hdul["MotorStates"].data
            motor_states = []
            for row in data:
                motor_states.append({
                    "name": row["MotorName"].strip(),  # remove any trailing spaces
                    "is_connected": row["IsConnected"],
                    "position": row["Position"],
                })
            return motor_states
    except Exception as e:
        st.write(f"Error reading {fits_path}: {e}")
        return []

# --- Data Aggregation ---
# We split the data into single-instance motors and multi-beam groups.
multi_beam_groups = ["HFO", "HTPP", "HTPI", "HTTP", "HTTI",
                     "BDS", "BTT", "BTP", "BMX", "BMY", "BLF",
                     "BOTT", "BOTP"]
single_motors = ["SDLA", "SDL12", "SDL34", "SSS", "SSF", "BFO"]

# Initialize dictionaries for storing time series data.
data_single = {motor: [] for motor in single_motors}
data_multi = {}
for group in multi_beam_groups:
    data_multi[group] = {}
    # For groups "BOTT" and "BOTP", assume beams 2 to 4; for others, beams 1 to 4.
    beam_range = range(2, 5) if group in ["BOTT", "BOTP"] else range(1, 5)
    for beam in beam_range:
        data_multi[group][beam] = []

# Loop over each FITS file and aggregate the data.
for fits_file in fits_files:
    ts = extract_timestamp_from_filename(os.path.basename(fits_file))
    motor_states = read_motor_states_fits(fits_file)
    for state in motor_states:
        name = state["name"]
        pos = state.get("position", np.nan)
        # Check if the motor is a single-instance motor.
        if name in single_motors:
            data_single[name].append((ts, pos))
        else:
            # For multi-beam motors, assume the name is of the form <Group><BeamNumber>
            m = re.match(r"([A-Z]+)(\d+)$", name)
            if m:
                group, beam_str = m.groups()
                beam = int(beam_str)
                if group in data_multi and beam in data_multi[group]:
                    data_multi[group][beam].append((ts, pos))
                else:
                    # If a group/beam is not preinitialized, create it on the fly.
                    if group not in data_multi:
                        data_multi[group] = {}
                    if beam not in data_multi[group]:
                        data_multi[group][beam] = []
                    data_multi[group][beam].append((ts, pos))
            else:
                # If the name doesn't match the expected pattern, put it into single-instance.
                if name not in data_single:
                    data_single[name] = []
                data_single[name].append((ts, pos))

# Sort the time series for each motor.
for motor in data_single:
    data_single[motor].sort(key=lambda x: x[0])
for group in data_multi:
    for beam in data_multi[group]:
        data_multi[group][beam].sort(key=lambda x: x[0])

# --- Plotting ---
st.header("Motor Position Time Series Plots")

st.subheader("Single-instance Motors")
for motor, records in data_single.items():
    if records:
        times, positions = zip(*records)
        fig, ax = plt.subplots()
        ax.plot(times, positions, marker='o', label=motor)
        ax.set_xlabel("Time")
        ax.set_ylabel("Position")
        ax.tick_params(axis='x', rotation=45)
        ax.set_title(f"Motor: {motor}")
        ax.legend()
        ax.grid(True)
        st.pyplot(fig)
    else:
        st.write(f"No data available for motor: {motor}")

st.subheader("Multi-beam Motors")
for group, beams in data_multi.items():
    # Check if there is at least one beam with data.
    if any(len(records) > 0 for records in beams.values()):
        fig, ax = plt.subplots()
        for beam, records in beams.items():
            if records:
                times, positions = zip(*records)
                ax.plot(times, positions, marker='o', label=f"{group}{beam}")
        ax.set_xlabel("Time")
        ax.set_ylabel("Position")
        ax.tick_params(axis='x', rotation=45)
        ax.set_title(f"Motor Group: {group}")
        ax.legend()
        ax.grid(True)
        st.pyplot(fig)
    else:
        st.write(f"No data available for motor group: {group}")