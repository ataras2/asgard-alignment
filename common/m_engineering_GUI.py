import streamlit as st
import numpy as np
import argparse
import zmq
import time
import json
import os
from PIL import Image
import datetime
import glob
import subprocess
import sys
from PIL import Image, ImageDraw, ImageFont
import matplotlib.pyplot as plt
from io import StringIO
import toml
import time

from asgard_alignment import FLI_Cameras as FLI
import asgard_alignment.Engineering
import common.DM_basis_functions

from asgard_alignment.DM_shm_ctrl import dmclass

try:
    import common.phasemask_centering_tool as pct
    from pyBaldr import utilities as util
except:
    print("CANT IMPORT PHASEMASK CENTERING TOOL!")


# Function to run script
def run_script(command):
    """
    Run an external python script using subprocess.
    """
    try:
        with st.spinner("Running.. drink some water"):
            # Ensure stdout and stderr are properly closed
            with subprocess.Popen(
                command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True
            ) as process:

                stdout, stderr = process.communicate()
                if process.returncode != 0:
                    st.error(f"Script failed: {stderr}")
                    return False
            return True  # Script succeeded
    except Exception as e:
        st.error(f"Error running script: {e}")
        return False


def run_script_with_output(command):
    """
    Run an external script using subprocess and capture its output in Streamlit.
    """
    try:
        with st.spinner("Running..."):
            # Open subprocess with pipes for real-time output capture
            with subprocess.Popen(
                command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True
            ) as process:

                output = []
                for line in process.stdout:
                    st.text(line.strip())  # Stream output in real-time to UI
                    output.append(line.strip())

                stderr_output = process.stderr.read().strip()
                st.write(f"process return code {process.returncode}")
                ### This always fails even when script runs fine.. even when using sys.exit(0) I dont understand
                # if process.returncode != 0:
                #    st.error(f"Script failed: {stderr_output}")
                #    return False, output
        return True, output  # Script succeeded

    except Exception as e:
        st.error(f"Error running script: {e}")
        return False, []


### DEFAULT FOR SAVING SCRIPT OUTPUT FIGURES THAT ARE DISPLAYED IN THE GUI! DO NOT DELETE
tstamp_rough = datetime.datetime.now().strftime("%d-%m-%Y")
quick_data_path = (
    f"/home/asg/Progs/repos/asgard-alignment/calibration/reports/{tstamp_rough}/"
)

os.makedirs(quick_data_path, exist_ok=True)


# make GUI wide
st.set_page_config(layout="wide")

if "message_history" not in st.session_state:
    st.session_state["message_history"] = []

if "socket" not in st.session_state:
    parser = argparse.ArgumentParser(description="ZeroMQ Client")
    parser.add_argument("--host", type=str, default="192.168.100.2", help="Server host")
    parser.add_argument("--port", type=int, default=5555, help="Server port")
    parser.add_argument(
        "--timeout", type=int, default=10000, help="Response timeout in milliseconds"
    )
    st.session_state["args"] = parser.parse_args()

    # Create a ZeroMQ context
    st.session_state["context"] = zmq.Context()

    # Create a socket to communicate with the server
    st.session_state["socket"] = st.session_state["context"].socket(zmq.REQ)

    # Set the receive timeout
    st.session_state["socket"].setsockopt(
        zmq.RCVTIMEO, st.session_state["args"].timeout
    )

    # Connect to the server
    server_address = (
        f"tcp://{st.session_state['args'].host}:{st.session_state['args'].port}"
    )
    st.session_state["socket"].connect(server_address)


if "SSS_fixed_mapping" not in st.session_state:
    st.session_state[f"SSS_fixed_mapping"] = {
        "SRL": 11.5,
        "SGL": 38.5,
        "SLD/SSP": 92.5,
        "SBB": 65.5,
    }
    st.session_state[f"SSS_offset"] = 0.0

st.title("Asgard alignment engineering GUI")


beam_specific_devices = [
    "HFO",
    "HTXP",
    "HTXI",
    "BTX",
    "BDS",
    # "phasemask", # this is in routines
    "SSF",
    "BOTX",
    "HPOL",
    # "DM", # this is based on the old Sardine shared memory, won't work anymore. Could be updated with Frantz's
    "BMX",
    "BMY",
    "BLF",
]

beam_common_devices = [
    "BFO",
    "SSS",
    "SDLA",
    "SDL12",
    "SDL34",
    "lamps",
    "BLF",  # just do all 4 at once in a single page
]

all_devices = beam_common_devices + beam_specific_devices


def send_and_get_response(message):
    # st.write(f"Sending message to server: {message}")
    st.session_state["message_history"].append(
        f":blue[Sending message to server: ] {message}\n"
    )
    print(f"sending: {message}")
    st.session_state["socket"].send_string(message)
    response = st.session_state["socket"].recv_string()
    if "NACK" in response or "not connected" in response:
        colour = "red"
    else:
        colour = "green"
    # st.markdown(f":{colour}[Received response from server: ] {response}")
    st.session_state["message_history"].append(
        f":{colour}[Received response from server: ] {response}\n"
    )

    return response.strip()


def handle_phasemask():

    # need to add session_state for the phasemask
    valid_pos = ["J1", "J2", "J3", "J4", "J5", "H1", "H2", "H3", "H4", "H5"]

    # phasemask interface
    st.subheader("Phasemask Interface")

    st.image(
        "figs/theoretical_ZWFS_intensities.png",
        caption="ZWFS Theoretical Intensities (4.5 lamda/D cold stop)",
        use_column_width=True,
    )

    beam = targets[0].split("phasemask")[1]

    st.session_state["selected_beam"] = beam

    # Initialize session state variables if they don't already exist
    if "selected_mask" not in st.session_state:
        st.session_state["selected_mask"] = "Unknown"

    if st.button("Read Position"):
        st.write(f"target:{targets}")
        # beam = targets[0].split("phasemask")[1]
        for target in [f"BMX{beam}", f"BMY{beam}"]:
            message = f"read {target}"
            res = send_and_get_response(message)
            if "NACK" in res:
                st.write(f"Error reading position for phasemask")
            else:
                st.write(f"Current position {target}: {res} um")

    with st.form(key="move_to_mask"):
        preset_position = st.selectbox(
            "Select Position",
            valid_pos,
            key="preset_position",
        )

        register_mask = st.form_submit_button("Update (register the mask)")
        submit = st.form_submit_button("Move to this mask")

    if register_mask:
        # we update the session state here so can register positions without moving!
        st.session_state["selected_mask"] = [preset_position]

    if submit:

        message = f"fpm_movetomask {targets[0]} {preset_position}"
        st.write(f"{message}")

        res = send_and_get_response(message)
        if "NACK" in res:
            st.write(f"Error moving to phasemask position with {res}")
            st.session_state["selected_mask"] = ["Unknown"]
        else:
            st.write(f"moved {targets[0]} to {preset_position}")
            st.session_state["selected_mask"] = [preset_position]

    st.write("Registered mask", st.session_state["selected_mask"][0])

    st.subheader("Manual Alignment")
    increment = st.number_input(
        "Relative increment (um)",
        min_value=0.0,
        max_value=1000.0,
        step=0.5,
        value=20.0,
        key="Relative_increment",
    )
    # increment = 20.0  # make user input
    # make a 3x3 grid but only use the up, down, left and right

    ul, um, ur = st.columns(3)
    ml, mm, mr = st.columns(3)
    ll, lm, lr = st.columns(3)

    with um:
        if st.button(f"+y: {increment:.2f}"):
            message = f"moverel BMY{beam} {increment}"
            send_and_get_response(message)

    with lm:
        if st.button(f"-y: {increment:.2f}"):
            message = f"moverel BMY{beam} {-increment}"
            send_and_get_response(message)
    with ml:
        if st.button(f"+x: {increment:.2f}"):
            message = f"moverel BMX{beam} {increment}"
            send_and_get_response(message)
    with mr:
        if st.button(f"-x: {increment:.2f}"):
            message = f"moverel BMX{beam} {-increment}"
            send_and_get_response(message)

    AUTOCENTER_SCRIPTS = {
        "auto_center_beam_1": [
            "python",
            "calibration/fine_phasemask_alignment.py",
            "--beam_id",
            "1",
            "--sleeptime",
            f"{0.1}",
            "--fig_path",
            quick_data_path,
        ],
        "auto_center_beam_2": [
            "python",
            "calibration/fine_phasemask_alignment.py",
            "--beam_id",
            "2",
            "--sleeptime",
            f"{0.1}",
            "--fig_path",
            quick_data_path,
        ],
        "auto_center_beam_3": [
            "python",
            "calibration/fine_phasemask_alignment.py",
            "--beam_id",
            "3",
            "--sleeptime",
            f"{0.1}",
            "--fig_path",
            quick_data_path,
        ],
        "auto_center_beam_4": [
            "python",
            "calibration/fine_phasemask_alignment.py",
            "--beam_id",
            "4",
            "--sleeptime",
            f"{0.1}",
            "--fig_path",
            quick_data_path,
        ],
    }

    st.subheader(f"Automatic phase mask centering")
    st.write(
        "Must be aligned to within 50-100um of the phasemask to work. Uses Gradient descent on strehl pixels with a dithering technique to estimate the gradient. "
    )
    cols = st.columns(4)
    for i, col in enumerate(cols):
        with col:
            st.subheader(f"Beam {i+1}")
            btn_key = f"auto_center_beam_{i+1}"

            # Run the script when button is clicked
            if st.button(f"Run {btn_key}"):
                success = run_script(AUTOCENTER_SCRIPTS[btn_key])

                if success:
                    fig_path = os.path.join(
                        AUTOCENTER_SCRIPTS[btn_key][-1],
                        f"phasemask_auto_center_beam{i+1}.png",
                    )
                    if os.path.exists(fig_path):
                        st.image(
                            Image.open(fig_path),
                            caption=f"{btn_key} Output",
                            use_column_width=True,
                        )
                    else:
                        st.warning(f"Cannot find {fig_path}")

    st.subheader("Change the phasemask position file")
    with st.form(key="update_position_file"):
        # f"/home/asg/Progs/repos/asgard-alignment/config_files/phasemask_positions/beam{beam}/*json"
        # Get all valid files
        valid_reference_position_files = glob.glob(
            f"/home/asg/Progs/repos/asgard-alignment/config_files/phasemask_positions/beam{beam}/*json"
        )

        # Sort by modification time (most recent first)
        valid_ref_files_sorted = sorted(
            valid_reference_position_files, key=os.path.getmtime, reverse=True
        )

        # Create display names (just filenames)
        display_names = [os.path.basename(f) for f in valid_ref_files_sorted]

        # Create a mapping from display name to full path
        file_map = dict(zip(display_names, valid_ref_files_sorted))

        # Show selectbox with display names
        selected_display_name = st.selectbox(
            "Select Reference Position File to Calculate Relative Separations Between Masks",
            display_names,
            key="selected_updated_position_file",
        )

        # Retrieve the full file path based on the selected display name
        selected_position_file = file_map[selected_display_name]

        submit_use_new_pos_file = st.form_submit_button(
            f"Use this phasemask position file now"
        )

    if submit_use_new_pos_file:

        # update_position_file(self, phase_positions_json ):

        # Save the updated positions to file
        save_message = f"fpm_update_position_file {targets[0]} {selected_position_file}"
        save_res = send_and_get_response(save_message)

        if "NACK" in save_res:
            st.error(f"Failed to save updated positions")  # to file: {save_res}")
        else:
            st.success(
                "Updated the json positions file successfully"  # at: " + save_path
            )

    update_col, save_col = st.columns(2)

    with update_col:
        if st.button(
            f"Update Registered {st.session_state['selected_mask']} Mask Position (local - not saved)"
        ):
            if "unknown" not in st.session_state["selected_mask"][0].lower():
                # Update the current mask position
                message = f"fpm_updatemaskpos {targets[0]} {st.session_state['selected_mask'][0]}"
                res = send_and_get_response(message)

                if "NACK" in res:
                    st.error(f"Failed to update registered mask: {res}")
                else:
                    st.success(
                        f"Successfully updated registered mask {st.session_state['selected_mask']}"
                    )

            else:
                st.error(f"Cannot update mask position with 'Unknown' mask.")

        # if st.button(
        #    f"Update All Mask Positions Relative to Current registered {st.session_state['selected_mask']} Position (local - not saved)"
        # ):

        with st.form(key="select_reference_file"):
            # select reference position file to update the mask positions from
            # cnt_pth = os.path.dirname(os.path.abspath(__file__))
            # save_path = cnt_pth + os.path.dirname(
            #    "/../config_files/phasemask_positions/"
            # )
            # f"/home/asg/Progs/repos/asgard-alignment/config_files/phasemask_positions/beam{beam}/*json"
            valid_reference_position_files = glob.glob(
                f"/home/asg/Progs/repos/asgard-alignment/config_files/phasemask_positions/beam{beam}/*json"
            )  # save_path + f"/beam{beam}/*json")

            # Sort by modification time (most recent first)
            valid_ref_files_sorted = sorted(
                valid_reference_position_files, key=os.path.getmtime, reverse=True
            )

            # Create display names (just filenames)
            display_names = [os.path.basename(f) for f in valid_ref_files_sorted]

            # Create a mapping from display name to full path
            file_map = dict(zip(display_names, valid_ref_files_sorted))

            # Show selectbox with display names
            selected_display_name = st.selectbox(
                "Select Reference Position File to Calculate Relative Separations Between Masks",
                display_names,
                key="selected_reference_position_file",
            )

            # Retrieve the full file path based on the selected display name
            selected_reference_file = file_map[selected_display_name]

            # # Sort files by modification time (most recent first)
            # valid_ref_files_sorted = sorted(valid_reference_position_files, key=os.path.getmtime, reverse=True)

            # selected_reference_file = st.selectbox(
            #     "Select Reference Position File to Calculate Relative Seperations Between Masks",
            #     display_names,
            #     key="selected_file",
            # )
            submit_reference_file = st.form_submit_button(
                f"Update All Mask Positions Relative to Current registered {st.session_state['selected_mask']} Position (local - not saved)"
            )

            if submit_reference_file:
                if "unknown" not in st.session_state["selected_mask"][0].lower():

                    message = f"fpm_updateallmaskpos {targets[0]} {st.session_state['selected_mask'][0]} {selected_reference_file}"

                    res = send_and_get_response(message)

                    if "NACK" in res:
                        st.error(f"Failed to update registered mask: {res}")
                    else:
                        st.success(
                            f"Successfully updated registered mask {st.session_state['selected_mask']}"
                        )

                else:
                    st.error(f"Cannot update mask position with 'Unknown' mask.")

    with save_col:

        st.write("Default save path is: 'config_files/phasemask_positions/'")

        if st.button(
            f"Save All Registered Mask Positions in json (update first if changed)"
        ):

            if "unknown" not in st.session_state["selected_mask"][0].lower():

                # save_path = send_and_get_response(f"fpm_getsavepath {targets[0]}")

                # Save the updated positions to file
                save_message = f"fpm_writemaskpos {targets[0]}"
                save_res = send_and_get_response(save_message)

                if "NACK" in save_res:
                    st.error(
                        f"Failed to save updated positions"
                    )  # to file: {save_res}")
                else:
                    st.success(
                        "Updated positions successfully saved to file"  # at: " + save_path
                    )
            else:
                st.error(f"Cannot update mask position with 'Unknown' mask.")

    # message = "fpm_updateallmaskpos {} {} {}"

    with st.form("raster_scan_form"):

        st.write(f"Proposed Raster Scan Parameters for beam {beam}")

        x0 = st.number_input("x0", value=10.0, step=10.0)
        y0 = st.number_input("y0", value=10.0, step=10.0)
        dx = st.number_input("dx", value=20.0, step=10.0)
        dy = st.number_input("dy", value=20.0, step=10.0)
        width = st.number_input("Width", value=100.0, step=10.0)
        height = st.number_input("Height", value=100.0, step=10.0)
        orientation = st.number_input("Orientation", value=0.0, step=10.0)

        submit_raster = st.form_submit_button("Update Raster Scan Parameters")
        apply_raster = st.form_submit_button("Apply Raster Scan Parameters")

    if submit_raster:
        st.write("Raster scan updated with new parameters.")

        starting_point = [x0, y0]

        raster_points = pct.raster_scan_with_orientation(
            starting_point, dx, dy, width, height, orientation
        )

        # Extract x and y coordinates
        x_coords, y_coords = zip(*raster_points)

        # Plot the scan points
        fig, ax = plt.subplots(figsize=(6, 6))
        ax.scatter(x_coords, y_coords, color="blue", label="Scan Points")
        ax.plot(x_coords, y_coords, linestyle="--", color="gray", alpha=0.7)
        ax.set_title(f"Raster Scan Pattern with {orientation}° Rotation", fontsize=14)
        ax.set_xlabel("X", fontsize=12)
        ax.set_ylabel("Y", fontsize=12)
        ax.legend()
        ax.grid(True)
        ax.set_aspect("equal")  # Ensure equal axis scaling

        # Show the updated figure
        st.pyplot(fig)
        plt.close("all")

    if apply_raster:
        figure_path = "/home/asg/Progs/repos/asgard-alignment/calibration/reports/phasemask_aquisition/"
        command = [
            "python",
            "calibration/phasemask_raster.py",
            "--beam",
            f"{beam}",
            "--initial_pos",
            f"{int(x0)},{int(y0)}",
            "--dx",
            f"{dx}",
            "--dy",
            f"{dy}",
            "--width",
            f"{width}",
            "--height",
            f"{height}",
            "--orientation",
            f"{orientation}",
            "--data_path",
            f"{figure_path}",
        ]

        # Run the external script
        with st.spinner("Running scan..."):
            process = subprocess.run(command, capture_output=True, text=True)

        # Display output
        st.text_area("Script Output", process.stdout)

        if process.returncode != 0:
            st.error(f"Error: {process.stderr}")
        else:
            st.success("Scan completed successfully!")
            if os.path.exists(figure_path):
                image1 = Image.open(
                    figure_path + f"cluster_search_heatmap_beam{beam}.png"
                )
                st.image(
                    image1,
                    caption="Cluster Analysis On Scan Results",
                    use_column_width=True,
                )

                image2 = Image.open(figure_path + f"clusters_heatmap_beam{beam}.png")
                st.image(
                    image2,
                    caption="Mean Image From Each Cluster",
                    use_column_width=True,
                )
            else:
                st.warning(
                    "Figure not found. Ensure the script generates the file correctly."
                )

    # absolute move option for input with button to move
    st.write("Move absolute")
    s_col1, s_col2 = st.columns(2)

    positions = []

    message = f"read BMX{beam}"
    res = send_and_get_response(message)
    positions.append(float(res))
    if "NACK" in res:
        st.write(f"Error reading position for {target}")
    time.sleep(0.1)
    message = f"read BMY{beam}"
    res = send_and_get_response(message)
    if "NACK" in res:
        st.write(f"Error reading position for {target}")
    positions.append(float(res))

    with s_col1:
        with st.form(key="absolute_move_u"):
            X_position = st.number_input(
                "X Position (um)",
                min_value=0.0,
                max_value=10000.0,
                step=5.0,
                value=positions[0],
                format="%.4f",
                key="X_position",
            )
            submit = st.form_submit_button("Move X")

        if submit:
            # replace the x in target with U
            message = f"moveabs BMX{beam} {X_position}"
            send_and_get_response(message)

    with s_col2:
        with st.form(key="absolute_move_v"):
            Y_position = st.number_input(
                "Y Position (um)",
                min_value=0.0,
                max_value=10000.0,
                value=positions[1],
                format="%.4f",
                step=5.0,
                key="Y_position",
            )
            submit2 = st.form_submit_button("Move Y")

        if submit2:
            message = f"moveabs BMY{beam} {Y_position}"
            send_and_get_response(message)


def handle_deformable_mirror():

    beam = targets[0].split("DM")[1]

    if "dm_targets" not in st.session_state:
        if not targets:
            st.session_state["dm_targets"] = []
        else:
            st.session_state["dm_targets"] = [target for target in targets]
            st.session_state["dm_targets"] = [target for target in targets]

    if "dm_last_command" not in st.session_state:
        st.session_state["dm_last_command"] = None  # To store the last DM command

    if "dm_apply_flat_map" not in st.session_state:
        st.session_state["dm_apply_flat_map"] = False  # Track if Flat Map is applied

    if "dm_apply_cross" not in st.session_state:
        st.session_state["dm_apply_cross"] = False  # Track if Flat Map is applied

    # Add a subheader for Deformable Mirror (DM) control
    st.subheader("Deformable Mirror (DM) Control")

    ff = "/home/asg/Progs/repos/asgard-alignment/config_files/dm_shared_memory_config.json"
    with open(ff, "r") as f:
        config_data = json.load(f)

        url = config_data["commands_urls"][f"{int(beam)-1}"]
        dm_shm = sa.from_url(np.ndarray, url)
        # Note an absolute change here will not change the shared memory variable, only the local copy
        # make sure to use relative changes
        dm_shm *= 0.0  # zero all the DMs
        dm_shm += 1.0  # unity all the DMs

        # for beam in [0,1,2,3]:

        #     url_dict[beam+1] = config_data['commands_urls'][f'{beam}'] # url of the shared memory for the DM

        #     dm_dict[beam+1] = sa.from_url(np.ndarray, url_dict[beam]) # relative changes to this variable changes it in shared memory!
        #     # Note an absolute change here will not change the shared memory variable, only the local copy
        #     # make sure to use relative changes
        #     dm_dict[beam+1] *= 0.0 # zero all the DMs
        #     dm_dict[beam+1] += 1.0 # unity all the DMs

    # Dropdown for selecting the DM
    # dm_name = st.selectbox(
    #     "Select DM",
    #     ["DM1", "DM2", "DM3", "DM4"],  # Assuming DM names from your config
    #     key="dm_name",
    # )

    # Button to apply the flat map to the selected DM
    # s_col1, s_col2, s_col3 = st.columns(3)

    combined_mode = np.zeros(140)

    # if st.button("Apply Flat Map"):
    #     st.session_state["dm_apply_flat_map"] = True

    #     if not targets:
    #         st.error("No targets specified.")

    #     combined_mode += common.DM_basis_functions.dm_flatmap_dict[f"{beam}"]

    #     # for target in targets:
    #     #     message = f"dmapplyflat {target}"
    #     #     response = send_and_get_response(message)
    #     #     if "ACK" in response:
    #     #         st.success(f"Flat map successfully applied to {target}")
    #     #     else:
    #     #         st.error(f"Failed to apply flat map to {target}. Response: {response}")

    #     st.session_state["dm_last_command"] = "Apply Flat Map"

    # if st.button("Apply Cross"):
    #     st.session_state["dm_apply_cross"] = True

    #     if not targets:
    #         st.error("No targets specified.")

    #     combined_mode += 0.2 * common.DM_basis_functions.cross_map

    #     # for target in targets:
    #     #     message = f"dmapplycross {target}"
    #     #     response = send_and_get_response(message)
    #     #     if "ACK" in response:
    #     #         st.success(f"Cross map successfully applied to {target}")
    #     #     else:
    #     #         st.error(f"Failed to apply cross map to {target}. Response: {response}")

    #     st.session_state["dm_last_command"] = "Apply Cross Map"

    apply_flat = st.checkbox(
        "Apply Flat Map",
        value=st.session_state["dm_apply_flat_map"],
        key="flat_checkbox",
    )
    apply_cross = st.checkbox(
        "Apply Cross Map",
        value=st.session_state["dm_apply_cross"],
        key="cross_checkbox",
    )

    if apply_flat:
        combined_mode += common.DM_basis_functions.dm_flatmap_dict[f"{beam}"]
        st.session_state["dm_last_command"] = "Apply Flat Map"

    if apply_cross:
        combined_mode += 0.2 * common.DM_basis_functions.cross_map
        st.session_state["dm_last_command"] = "Apply Cross Map"

    # Add a button to reset all sliders to zero
    if st.button("Reset All Amplitudes", key="reset_amplitudes_button"):
        for i in range(11):
            st.session_state[f"slider_{i}"] = 0.0  # Reset slider values to zero

    basis = common.DM_basis_functions.construct_command_basis(
        basis="Zernike_pinned_edges",
        number_of_modes=11,
        Nx_act_DM=12,
        Nx_act_basis=12,
        act_offset=(0, 0),
        without_piston=False,
    ).T

    basis[0] = (
        1 / np.max(basis[0]) * basis[0]
    )  # make piston = 1 (not normalized as others <M|M>=1)

    # Zernike mode names for the first 11 modes
    zernike_modes = [
        "Z₀₀: Piston",
        "Z₁₋₁: Tip (X Tilt)",
        "Z₁₁: Tilt (Y Tilt)",
        "Z₂₀: Defocus",
        "Z₂₋₂: Astigmatism (45°)",
        "Z₂₂: Astigmatism (0°/90°)",
        "Z₃₋₃: Coma (X)",
        "Z₃₁: Coma (Y)",
        "Z₃₋₁: Trefoil (X)",
        "Z₃₃: Trefoil (Y)",
        "Z₄₀: Spherical Aberration",
    ]

    # Create two columns for sliders and plot
    slider_col, plot_col = st.columns([1, 1])
    with slider_col:
        # Sliders for setting amplitudes for the first 11 modes in the `basis`
        st.subheader("Set Amplitudes for Modes")
        amplitudes = []
        for i, mode_name in enumerate(zernike_modes):
            amp = st.slider(
                f"Amplitude for {mode_name}",
                min_value=-0.2,
                max_value=0.2,
                value=0.0,
                step=0.01,
                key=f"slider_{i}",
            )
            amplitudes.append(amp)

    with plot_col:
        # Compute the weighted sum of basis terms
        # if st.button("Apply Mode Combination", key="apply_mode_combination"):
        if len(basis) != 11:
            st.error(
                "Insufficient basis functions provided. At least 11 modes are required."
            )
        else:
            combined_mode += sum(
                amplitude * mode for amplitude, mode in zip(amplitudes, basis[:11])
            )
            st.session_state["dm_last_command"] = "Apply Mode Combination"
        # else:
        #    combined_mode = np.zeros(140)

        # Add flat and cross maps if ticked
        # if apply_flat:
        #     flat_map = common.DM_basis_functions.dm_flatmap_dict[f"{beam}"]
        #     combined_mode += flat_map

        # if apply_cross:
        #     cross_map = common.DM_basis_functions.cross_map
        #     combined_mode += 0.2 * cross_map

        # Plot the combined mode
        fig, ax = plt.subplots()
        cax = ax.imshow(
            common.DM_basis_functions.get_DM_command_in_2D(combined_mode),
            cmap="viridis",
            # vmax=0.8,
            # vmin=0.2,
        )  # Adjust dimensions as needed
        fig.colorbar(cax, ax=ax)
        ax.set_title("Applied Combined Mode")
        st.pyplot(fig)
        plt.close("all")

    # set the DM to unity and then multiply by the combined mode.. We really need a set method!!!
    dm_shm *= 0.0  # zero all the DMs
    dm_shm += 1.0  # unity all the DMs
    dm_shm *= combined_mode  # add the modes

    ### Below is when we opened DM in the multi device server and communicated via ZMQ
    # Placeholder for sending the combined mode to the DM
    # response = send_and_get_response(f"dmapply {combined_mode}")
    # if "ACK" in response:
    #    st.success("Combined mode successfully applied to the DM.")
    # else:
    #    st.error(f"Failed to apply combined mode. Response: {response}")

    # with s_col2:
    #     pass
    # with s_col3:
    #     pass


def handle_linear_stage():
    # linear stage interface
    st.subheader("Linear Stage Interface")

    if component == "BDS":
        valid_pos = ["BIF_H", "BIF_YJ", "empty"]

        if f"BDS{beam_number}_fixed_mapping" not in st.session_state:
            st.session_state[f"BDS{beam_number}_fixed_mapping"] = {
                "BIF_H": 133.07,  # (white target)
                "BIF_YJ": 63.07,  # (mirror)
                "empty": 0.0,
            }
            st.session_state[f"BDS{beam_number}_offset"] = 0.0

    elif component == "SSS":
        valid_pos = ["SRL", "SGL", "SLD/SSP", "SBB"]

    mapping = {
        k: v + st.session_state[f"{target}_offset"]
        for k, v in st.session_state[f"{target}_fixed_mapping"].items()
    }

    # add two buttons, one for homing and one for reading position
    s_col1, s_col2, s_col3 = st.columns(3)

    with s_col1:
        if st.button("Read Position"):
            message = f"read {target}"
            res = send_and_get_response(message)
            # check if close to any preset position
            for pos, val in mapping.items():
                if np.isclose(float(res), val, atol=0.1):
                    st.write(f"Current position: {float(res):.2f} mm ({pos})")
                    break
            else:
                st.write(f"Current position: {float(res):.2f} mm")
    with s_col2:
        # read state button
        if st.button("Read State"):
            message = f"state {target}"
            res = send_and_get_response(message)
            st.write(res)

    with s_col3:
        if st.button("Home (if needed)"):
            message = f"init {target}"
            send_and_get_response(message)

    ss_col1, ss_col2 = st.columns(2)

    with ss_col1:
        st.write("Preset positions selection")
        with st.form(key="valid_positions"):
            preset_position = st.selectbox(
                "Select Position",
                valid_pos,
                key="preset_position",
            )
            submit = st.form_submit_button("Move")

        if submit:
            message = f"moveabs {target} {mapping[preset_position]}"
            send_and_get_response(message)

    with ss_col2:
        # relative move option for input with button to move
        st.write("Relative Move")
        with st.form(key="relative_move"):
            relative_move = st.number_input(
                "Relative Move (mm)",
                min_value=-100.0,
                max_value=100.0,
                step=0.1,
                value=0.0,
                key="relative_move",
            )
            submit = st.form_submit_button("Move")

        if submit:
            message = f"moverel {target} {relative_move}"
            send_and_get_response(message)

    # add a button to update the preset positions
    st.subheader("Updating positions")
    st.write(f"Current mapping is: {mapping}")
    button_cols = st.columns(3)
    with button_cols[0]:
        if st.button(f"Update only {preset_position}"):
            current_position = send_and_get_response(f"read {target}")
            st.session_state[f"{target}_fixed_mapping"][preset_position] = float(
                current_position
            )
            st.rerun()
    with button_cols[1]:
        if st.button("Update all"):
            current_position = send_and_get_response(f"read {target}")
            st.session_state[f"{target}_offset"] = (
                float(current_position) - mapping[preset_position]
            )
            st.rerun()
    with button_cols[2]:
        if st.button("Reset to original"):
            st.session_state[f"{target}_offset"] = 0.0
            del st.session_state[f"{target}_fixed_mapping"]
            st.rerun()


def handle_tt_motor():
    # TT motor interface
    # no homing, read position should be an option
    # also two fields for absolute value of each axis

    st.subheader("TT Motor Interface")

    # read position button
    if st.button("Read Position"):
        positions = []
        for target in targets:
            message = f"read {target}"
            res = send_and_get_response(message)
            if "NACK" in res:
                st.write(f"Error reading position for {target}")
                break
            positions.append(float(res))
        else:
            st.write(
                f"Current positions: U={positions[0]:.3f} V={positions[1]:.3f} (degrees)"
            )

    positions = []
    for target in targets:
        message = f"read {target}"
        res = send_and_get_response(message)
        if "NACK" in res:
            st.write(f"Error reading position for {target}")
            break
        positions.append(float(res))

    # if the read value is out of bounds, use None
    for index in range(2):
        if positions[index] < -0.75 or positions[index] > 0.75:
            positions[index] = None

    ss_col1, ss_col2 = st.columns(2)
    with ss_col1:
        inc = st.number_input(
            "Step size",
            value=0.01,
            min_value=0.0,
            max_value=0.1,
            key=f"TT_increment",
            step=0.005,
            format="%.3f",
        )

    with ss_col2:
        use_button_to_move = st.checkbox("Use button to move")
        # value to delay after each move
        delay_on_moves = st.number_input(
            "Delay after each move (s)",
            value=1.0,
            min_value=0.0,
            max_value=10.0,
            key="delay_on_moves",
            step=0.1,
            format="%.1f",
        )

    # absolute move option for input with button to move
    st.write("Move absolute")
    s_col1, s_col2 = st.columns(2)
    if use_button_to_move:
        with s_col1:
            with st.form(key="absolute_move_p"):
                p_position = st.number_input(
                    "P Position (degrees)",
                    min_value=-0.750,
                    max_value=0.75,
                    step=inc,
                    value=positions[0],
                    format="%.4f",
                    key="p_position",
                )
                submit = st.form_submit_button("Move P")

            if submit:
                # replace the x in target with P
                target = f"{component}{beam_number}"
                target = target.replace("X", "P")
                message = f"moveabs {target} {p_position}"
                send_and_get_response(message)

        with s_col2:
            with st.form(key="absolute_move_t"):
                t_position = st.number_input(
                    "T Position (degrees)",
                    min_value=-0.750,
                    max_value=0.75,
                    value=positions[1],
                    format="%.4f",
                    step=inc,
                    key="t_position",
                )
                submit2 = st.form_submit_button("Move T")

            if submit2:
                target = f"{component}{beam_number}"
                target = target.replace("X", "T")
                message = f"moveabs {target} {t_position}"
                send_and_get_response(message)
    else:

        def get_onchange_fn(axis, key):
            def onchange_fn():
                target = f"{component}{beam_number}"
                target = target.replace("X", axis)
                message = f"moveabs {target} {st.session_state[key]}"
                send_and_get_response(message)
                if delay_on_moves:
                    time.sleep(1.0)

            return onchange_fn

        sub_col1, sub_col2 = st.columns(2)

        with sub_col1:
            p_position = st.number_input(
                "P Position (degrees)",
                min_value=-0.750,
                max_value=0.75,
                step=inc,
                value=positions[0],
                format="%.4f",
                key="p_position",
                on_change=get_onchange_fn("P", "p_position"),
            )

        with sub_col2:
            t_position = st.number_input(
                "T Position (degrees)",
                min_value=-0.750,
                max_value=0.75,
                step=inc,
                value=positions[1],
                format="%.4f",
                key="t_position",
                on_change=get_onchange_fn("T", "t_position"),
            )


def handle_linear_actuator():
    # Linear actuator interface
    # all units in um

    if component == "HFO":
        bounds = (0.0, 16e3)
    else:
        bounds = (0.0, 10e3)

    st.subheader("Linear Actuator Interface")

    c1, c2, c3 = st.columns(3)
    with c1:
        # read position button
        if st.button("Read Position"):
            message = f"read {target}"
            res = send_and_get_response(message)
            if "NACK" in res:
                st.write(f"Error reading position for {target}")
            else:
                pos = float(res)
                if "HFO" in target:
                    pos *= 1e3
                st.write(f"Current position: {pos:.2f} um")

    with c2:
        # read state button
        if st.button("Read State"):
            message = f"state {target}"
            res = send_and_get_response(message)
            st.write(res)

    with c3:
        # init button
        if st.button("Home (if needed)"):
            message = f"init {target}"
            send_and_get_response(message)

    def get_onchange_fn(key, target):
        def onchange_fn():
            if "HFO" in target:
                desired_position = st.session_state[key] * 1e-3
            else:
                desired_position = st.session_state[key]
            message = f"moveabs {target} {desired_position}"
            send_and_get_response(message)
            time.sleep(1.0)

        return onchange_fn

    message = f"read {target}"
    res = send_and_get_response(message)
    if "NACK" in res:
        st.write(f"Error reading position for {target}")
    cur_pos = float(res)

    if "HFO" in target:
        cur_pos *= 1e3

    # absolute move option for input with button to move
    st.write("Absolute Move")

    inc = st.number_input(
        "Step size",
        value=10.0,
        min_value=0.0,
        max_value=500.0,
        key=f"lin_increment",
        step=2.0,
        format="%.3f",
    )

    position = st.number_input(
        "Position (um)",
        min_value=bounds[0],
        max_value=bounds[1],
        step=inc,
        value=min(max(cur_pos, bounds[0]), bounds[1]),  # cur_pos,
        key="lin_position",
        on_change=get_onchange_fn("lin_position", target),
    )


def handle_source_select():
    st.subheader("Source Selection")
    valid_lamps = ["SGL", "SRL", "SBB"]
    if "selected_source" not in st.session_state:
        st.session_state["selected_source"] = "Unknown"

    # dropdown to select from valid
    lamp_cur = st.selectbox(
        "Select Source",
        valid_lamps,
        key="lamp_cur",
    )

    # read state button
    if st.button("Read State"):
        message = f"is_on {lamp_cur}"
        res = send_and_get_response(message)
        st.write(res)

    # on and off buttons
    on, off = st.columns(2)

    with on:
        if st.button("Turn On"):
            message = f"on {lamp_cur}"
            res = send_and_get_response(message)
            st.session_state["selected_source"] = lamp_cur
            st.write(res)

    with off:
        if st.button("Turn Off"):
            message = f"off {lamp_cur}"
            res = send_and_get_response(message)
            st.session_state["selected_source"] = "Unknown"
            st.write(res)


def handle_bistable_motor():
    st.subheader("Bistable Motor Interface")

    if "selected_state" not in st.session_state:
        st.session_state["selected_state"] = "Unknown"

    # read state button
    if st.button("Read State"):
        message = f"state {target}"
        res = send_and_get_response(message)
        st.write(res)

    if component == "SSF":
        states = ["up", "down"]
        values = [1.0, 0.0]
    else:
        st.error("Component not recognized")

    # state 1 and state 2 buttons
    s1, s2 = st.columns(2)

    with s1:
        if st.button(states[0]):
            message = f"moveabs {target} {values[0]}"
            res = send_and_get_response(message)
            st.session_state["selected_state"] = states[0]
            st.write(res)

    with s2:
        if st.button(states[1]):
            message = f"moveabs {target} {values[1]}"
            res = send_and_get_response(message)
            st.session_state["selected_state"] = states[1]
            st.write(res)


def handle_linbo_motor():
    st.subheader("LiNbO3 Stepper motor interface")

    # 3 buttons: read pos, read state, home
    cols = st.columns(4)
    with cols[0]:
        if st.button("Read Position"):
            message = f"read {target}"
            res = send_and_get_response(message)
            st.write(f"Current position: {res} steps")
    with cols[1]:
        if st.button("Read State"):
            message = f"state {target}"
            res = send_and_get_response(message)
            st.write(res)
    with cols[2]:
        if st.button("Home"):
            message = f"home_steppers {target}"
            res = send_and_get_response(message)
            st.write(res)

    with cols[3]:
        if st.button("Stop", type="primary"):
            message = f"stop {target}"
            res = send_and_get_response(message)
            st.write(res)

    # now setting a position, using relative move
    st.write("Move absolute")
    inc = st.number_input(
        "Value (steps)",
        value=0,
        min_value=-24000 // 4,
        max_value=24000 // 4,
        key="linbo_pos",
        step=100,
        format="%d",
    )
    with st.form(key="abs_move_linbo"):
        submit = st.form_submit_button("Move")
        if submit:
            message = f"moveabs {target} {float(inc)}"  # move only does floats, but motor class will convert back to int
            res = send_and_get_response(message)
            st.write(res)


def handle_lens_flipper():
    st.subheader("Baldr lens flippers")

    beam_nums = list(range(1, 5))

    cols = st.columns(len(beam_nums))

    for beam_num, col in zip(beam_nums, cols):
        target = f"BLF{beam_num}"
        with col:
            st.write(f"beam{beam_num}")
            if st.button("Read State", key=f"read_state_{beam_num}"):
                message = f"read {target}"
                res = send_and_get_response(message)
                st.write(res)

            positions = ["STANDARD", "FAINT"]
            for pos in positions:
                if st.button(pos, key=f"move_{pos}_{beam_num}"):
                    message = f"asg_setup {target} NAME {pos}"
                    res = send_and_get_response(message)


col_main, col_history = st.columns([2, 1])


with col_history:
    with col_history.container(border=True, height=500):
        st.subheader("Message History")
        # join all into a long string with newlines, in reverse order and as a markdown list
        message_history = st.session_state["message_history"]
        message_history_str = "\n".join(reversed(message_history[-200:]))
        st.markdown(message_history_str)


with col_main:
    operating_mode = st.selectbox(
        "Select Operating Mode",
        ["Direct write", "Routines"],
        key="operating_mode",
    )

    if operating_mode == "Direct write":
        # create two dropdowns for selecting component and beam number
        col1, col2 = st.columns(2)

        with col1:
            component = st.selectbox(
                "Select device",
                all_devices,
                key="component",
            )

        if component in beam_specific_devices:
            if component == "BOTX":
                beam_numbers = [2, 3, 4]
            elif component == "HPOL":
                beam_numbers = [1, 2, 4]
            else:
                beam_numbers = [1, 2, 3, 4]

            with col2:
                beam_number = st.selectbox(
                    "Select Beam Number",
                    beam_numbers,
                    key="beam_number",
                )
            targets = [f"{component}{beam_number}"]

            if component in ["HTXP", "HTXI", "BTX", "BOTX"]:
                # replace the X in target with P
                target = f"{component}{beam_number}"
                targets = [target.replace("X", "P"), target.replace("X", "T")]

        elif component == "BLF":
            targets = [f"BLF{ii}" for ii in [1, 2, 3, 4]]
        else:
            beam_number = None
            targets = [component]

        if "lamps" in component:
            handle_source_select()

        else:
            # check if component is connected
            is_connected = all(
                send_and_get_response(f"connected? {target}") == "connected"
                for target in targets
            )
            print(is_connected)
            if not is_connected:
                if "DM" in component:
                    st.write(
                        f"DM uses shared memory, start server \n>ipython -i asgard_alignment/DM_shared_memory_server.py\
                            \nneeds to run in conda base (check >conda env list). \nAlso DMs must not be connected any where else\
                            (check no other server is using them)\n To Do: run this within MDS. Still experimental. Zernike modes disabled"
                    )
                else:
                    st.write(f"Component(s) {targets} is/are not connected!")

                    with st.form(key="connect_request"):
                        submit = st.form_submit_button("Connect")

                    if submit:
                        for target in targets:
                            # debug here
                            if "BLF" in target:
                                st.write(target)
                            message = f"connect {target}"
                            send_and_get_response(message)

            if (
                component not in ["HTXP", "HTXI", "BTX", "BOTX"]
                and component in beam_specific_devices
            ):
                target = f"{component}{beam_number}"
            if (
                component not in ["HTXP", "HTXI", "BTX", "BOTX"]
                and component in beam_common_devices
            ):
                target = component

            if component in ["BDS", "SSS"]:
                handle_linear_stage()

            elif component in ["HTXP", "HTXI", "BTX", "BOTX"]:
                handle_tt_motor()

            elif component in ["BFO", "SDLA", "SDL12", "SDL34", "HFO", "BMX", "BMY"]:
                handle_linear_actuator()

            elif component in ["DM"]:
                handle_deformable_mirror()

            elif component in ["phasemask"]:
                handle_phasemask()

            elif component in ["SSF"]:
                handle_bistable_motor()

            elif component in ["BLF"]:
                handle_lens_flipper()

            elif component in ["HPOL"]:
                handle_linbo_motor()

    elif operating_mode == "Routines":
        # move pupil and move image go here
        # also zero all (for alignment stuff)

        routine_options = st.selectbox(
            "Select Routine",
            [
                "Quick buttons",
                # "Camera & DMs",
                "Illumination",
                "Heimdallr shutters",
                "Move image/pupil",
                "Phasemask Alignment",
                "Scan Mirror",
                "Save state",
                "Load state",
                "See All States",
                "Health",
            ],
            key="routine_options",
        )

        if routine_options == "Quick buttons":

            ## Quick save states and images
            tstamp_rough = datetime.datetime.now().strftime("%d-%m-%Y")

            st.title("Quick save motor states with CRED 1 images")
            st.write(
                "Use frequently! these are very usefull to analyse system stability and recovery after earthquakes etc"
            )
            st.write(
                "saves all motor states along with 20 CRED 1 images in current camera settings. Saves as a fits file in:"
            )
            save_state_data_path = f"/home/asg/Progs/repos/asgard-alignment/instr_states/stability_analysis/{tstamp_rough}/"
            st.write(f"{save_state_data_path}")

            if st.button("qucik save state"):
                command = [
                    "python",
                    "calibration/quick_savestates_n_img.py",
                    "--data_path",
                    save_state_data_path,
                ]
                success = run_script(command)
                if success:
                    st.success("done")
                else:
                    st.warning("could not run script for some reason..")
            ## DMS

            st.title("Quick Check on CRED 1 Server")
            st.write(
                "Checks the camera server comminication and that the camera is updating and not skipping frames. Provides trouble shooting recommendations if errors are found."
            )
            if st.button("health check"):
                command = ["python", "playground/skipped_frames.py"]

                success, log_output = run_script_with_output(command)

                if success:
                    st.success("Camera test completed successfully!")
                else:
                    st.error("Camera test failed. Check logs.")

                # Show full log output in an expandable section
                with st.expander("Detailed Log Output"):
                    for line in log_output:
                        st.text(line)

            # st.title("Deformable Mirrors (DM's)")

            # zbasis = common.DM_basis_functions.zer_bank(1, 10)

            # # use_calibrated_dm_flat = st.checkbox('Use a calibrated "Baldr DM flat')

            # if "dm_shm_dict" not in st.session_state:
            #     st.session_state.dm_shm_dict = {
            #         beam_id: dmclass(beam_id=beam_id) for beam_id in [1, 2, 3, 4]
            #     }

            # col1, col2, col3, col4, col5 = st.columns(5)
            # with col1:
            #     if st.button("Zero all DM's"):
            #         for beam in [1, 2, 3, 4]:
            #             st.session_state.dm_shm_dict[beam].zero_all()

            # with col2:
            #     if st.button("Apply Factory DM flat's"):
            #         for beam in [1, 2, 3, 4]:
            #             st.session_state.dm_shm_dict[beam].activate_flat()
            #             # if use_calibrated_dm_flat:
            #             #    st.session_state.dm_shm_dict[beam].activate_calibrated_flat()
            #             # else:
            # with col3:
            #     if st.button("Apply Baldr DM flat's"):
            #         for beam in [1, 2, 3, 4]:
            #             st.session_state.dm_shm_dict[beam].activate_calibrated_flat()

            # with col4:
            #     if st.button("Apply Heim DM flat's"):
            #         for beam in [1, 2, 3, 4]:
            #             #st.session_state.dm_shm_dict[beam].activate_cross(amp=0.2)
            #             wdirtmp = "/home/asg/Progs/repos/asgard-alignment/DMShapes/" #os.path.dirname(__file__)
            #             flat_cmd = np.loadtxt(st.session_state.dm_shm_dict[beam].select_flat_cmd( wdirtmp ))
            #             flat_cmd_offset = np.loadtxt(st.session_state.dm_shm_dict[beam].select_flat_cmd_offset( wdirtmp))
            #             st.session_state.dm_shm_dict[beam].shms[0].set_data(st.session_state.dm_shm_dict[beam].cmd_2_map2D(flat_cmd + flat_cmd_offset, fill=0.0))
            #             ##
            #             st.session_state.dm_shm_dict[beam].shm0.post_sems(1)
            # with col5:
            #     if st.button("Apply DM cross"):
            #         for beam in [1, 2, 3, 4]:
            #             st.session_state.dm_shm_dict[beam].activate_cross(amp=0.2)

            # st.write("Defocus")

            # strength = st.slider(
            #     "Strength", min_value=-0.5, max_value=0.5, value=0.0, step=0.01
            # )

            # if st.button("Apply defocus"):
            #     for beam in [1, 2, 3, 4]:
            #         st.session_state.dm_shm_dict[beam].set_data(strength * zbasis[3])

            # basis = st.selectbox(
            #     "Select Basis",
            #     options=[
            #         "Hadamard",
            #         "Zonal",
            #         "Zonal_pinned_edges",
            #         "Zernike",
            #         "Zernike_pinned_edges",
            #         "fourier",
            #         "fourier_pinned_edges",
            #     ],
            # )

            # mode = st.number_input("Mode", min_value=0, max_value=50, value=5, step=1)

            # strength = st.slider("Strength", min_value=0.0, max_value=1.0, value=0.1, step=0.01)

            # if st.button("Apply aberration"):
            #     for beam in [1,2,3,4]:
            #         st.session_state.dm_shm_dict[beam].shms[2].set_data(  )

            st.title("Phase masks")

            st.write("Move all beams to mask")
            col1, col2, col3, col4, col5 = st.columns(5)

            for mask, col in zip([1, 2, 3, 4, 5], [col1, col2, col3, col4, col5]):
                with col:
                    if st.button(f"H{mask}"):
                        for beam in [1, 2, 3, 4]:
                            message = f"fpm_movetomask phasemask{beam} H{mask}"
                            res = send_and_get_response(message)
                    if st.button(f"J{mask}"):
                        for beam in [1, 2, 3, 4]:
                            message = f"fpm_movetomask phasemask{beam} J{mask}"
                            res = send_and_get_response(message)

            # col1, col2 = st.columns(2)
            # BMX_offset_tmp = 200.0
            # with col1:
            #     if st.button(f"Offset all masks ({BMX_offset_tmp}um in BMX)"):
            #         for beam in [1,2,3,4]:
            #             message = f"moverel BMX{beam} {BMX_offset_tmp}"
            #             response = send_and_get_response(message)
            #             if "ACK" in response:
            #                 st.success(f"{BMX_offset_tmp}um offset successfully applied to BMX{beam}")
            #             else:
            #                 st.error(f"Failed to apply offset to BMX{beam}. Response: {response}")

            # with col2:
            #     if st.button("Reverse Offset for all masks"):
            #         for beam in [1,2,3,4]:
            #             message = f"moverel BMX{beam} {-BMX_offset_tmp}"
            #             response = send_and_get_response(message)

            #             if "ACK" in response:
            #                 st.success(f"{-BMX_offset_tmp}um offset successfully applied to BMX{beam}")
            #             else:
            #                 st.error(f"Failed to apply offset to BMX{beam}. Response: {response}")

            # To apply offsets to the phasemasks
            st.write("Select Beams for Offsetting Phasemasks")
            col1, col2, col3, col4 = st.columns(4)  # Equal width columns

            with col1:
                beam_1 = st.checkbox("Beam 1", value=True)
            with col2:
                beam_2 = st.checkbox("Beam 2", value=True)
            with col3:
                beam_3 = st.checkbox("Beam 3", value=True)
            with col4:
                beam_4 = st.checkbox("Beam 4", value=True)

            selected_beams = []
            if beam_1:
                selected_beams.append(1)
            if beam_2:
                selected_beams.append(2)
            if beam_3:
                selected_beams.append(3)
            if beam_4:
                selected_beams.append(4)

            st.subheader("Auto Alignment")

            autoAlign_method = st.selectbox(
                "Align method",
                ["gradient_descent", "brute_scan"],
                key="autoalignquickbutton",
            )

            AUTOCENTER_SCRIPTS = {
                f"auto_center_beam_{beam}": [
                    "python",
                    "calibration/fine_phasemask_alignment.py",
                    "--beam_id",
                    f"{beam}",
                    "--method",
                    f"{autoAlign_method}",
                    "--sleeptime",
                    f"{0.1}",
                    "--fig_path",
                    quick_data_path,
                ]
                for beam in selected_beams
            }

            import concurrent.futures

            if st.button("Auto Align Phasemasks for Selected Beams"):
                with concurrent.futures.ThreadPoolExecutor(
                    max_workers=len(AUTOCENTER_SCRIPTS)
                ) as executor:
                    results = list(
                        executor.map(run_script, list(AUTOCENTER_SCRIPTS.values()))
                    )

            st.subheader("Move Phasmasks Manually")

            offset_input = st.text_input("Relative Offset Amp (μm)", value="20.0")
            try:
                increment = float(offset_input)  # Convert input to float
            except ValueError:
                st.error("Please enter a valid number for the offset.")
                increment = 20.0  # Default value if input is invalid

            # Movement Buttons (Grid Layout)
            ul, um, ur = st.columns(3)
            ml, mm, mr = st.columns(3)
            ll, lm, lr = st.columns(3)

            # Move Up (BMY+)
            with um:
                if st.button(f"⬆️ +Y ({increment:.2f} μm)"):
                    for beam in selected_beams:
                        message = f"moverel BMY{beam} {increment}"
                        send_and_get_response(message)

            # Move Down (BMY-)
            with lm:
                if st.button(f"⬇️ -Y ({increment:.2f} μm)"):
                    for beam in selected_beams:
                        message = f"moverel BMY{beam} {-increment}"
                        send_and_get_response(message)

            # Move Left (BMX+)
            with ml:
                if st.button(f"⬅️ +X ({increment:.2f} μm)"):
                    for beam in selected_beams:
                        message = f"moverel BMX{beam} {increment}"
                        send_and_get_response(message)

            # Move Right (BMX-)
            with mr:
                if st.button(f"➡️ -X ({increment:.2f} μm)"):
                    for beam in selected_beams:
                        message = f"moverel BMX{beam} {-increment}"
                        send_and_get_response(message)

            st.title("Quick Scripts")

            # Define scripts and their arguments
            QUICK_SCRIPTS = {
                "detect pupils": [
                    "python",
                    "calibration/detect_cropped_pupils_coords.py",
                    "--fig_path",
                    quick_data_path,
                ],
                "register_pupil_beam_1": [
                    "python",
                    "calibration/pupil_registration.py",
                    "--beam_ids",
                    "1",
                    "--fig_path",
                    quick_data_path,
                ],
                "register_pupil_beam_2": [
                    "python",
                    "calibration/pupil_registration.py",
                    "--beam_ids",
                    "2",
                    "--fig_path",
                    quick_data_path,
                ],
                "register_pupil_beam_3": [
                    "python",
                    "calibration/pupil_registration.py",
                    "--beam_ids",
                    "3",
                    "--fig_path",
                    quick_data_path,
                ],
                "register_pupil_beam_4": [
                    "python",
                    "calibration/pupil_registration.py",
                    "--beam_ids",
                    "4",
                    "--fig_path",
                    quick_data_path,
                ],
                "register_DM_beam_1": [
                    "python",
                    "calibration/dm_registration_calibration.py",
                    "--beam_id",
                    "1",
                    "--fig_path",
                    quick_data_path,
                ],
                "register_DM_beam_2": [
                    "python",
                    "calibration/dm_registration_calibration.py",
                    "--beam_id",
                    "2",
                    "--fig_path",
                    quick_data_path,
                ],
                "register_DM_beam_3": [
                    "python",
                    "calibration/dm_registration_calibration.py",
                    "--beam_id",
                    "3",
                    "--fig_path",
                    quick_data_path,
                ],
                "register_DM_beam_4": [
                    "python",
                    "calibration/dm_registration_calibration.py",
                    "--beam_id",
                    "4",
                    "--fig_path",
                    quick_data_path,
                ],
                "register_strehl_pixels_beam_1": [
                    "python",
                    "calibration/strehl_filter_registration.py",
                    "--beam_id",
                    "1",
                    "--fig_path",
                    quick_data_path,
                ],
                "register_strehl_pixels_beam_2": [
                    "python",
                    "calibration/strehl_filter_registration.py",
                    "--beam_id",
                    "2",
                    "--fig_path",
                    quick_data_path,
                ],
                "register_strehl_pixels_beam_3": [
                    "python",
                    "calibration/strehl_filter_registration.py",
                    "--beam_id",
                    "3",
                    "--fig_path",
                    quick_data_path,
                ],
                "register_strehl_pixels_beam_4": [
                    "python",
                    "calibration/strehl_filter_registration.py",
                    "--beam_id",
                    "4",
                    "--fig_path",
                    quick_data_path,
                ],
            }

            # Initialize session state for each button (if not set)
            for key in QUICK_SCRIPTS.keys():
                if key not in st.session_state:
                    st.session_state[key] = False  # Default: No button has been pressed

            #################-------------------------
            st.subheader("Detect the Sub-Pupils")
            st.write("run this with clear pupils (phase masks out)")

            if st.button("Detect Heimdallr/Baldr Pupils", key="detect_pupils"):
                sucess = run_script(QUICK_SCRIPTS["detect pupils"])

                if sucess:  # st.session_state["detect pupils"]:
                    fig_path = (
                        QUICK_SCRIPTS["detect pupils"][-1] + "detected_pupils.png"
                    )
                    if os.path.exists(fig_path):
                        st.image(
                            Image.open(fig_path),
                            caption="Detected Pupils Output",
                            use_column_width=True,
                        )
                    else:
                        st.write(f"can't find {fig_path}")
                else:
                    st.write("no current output")

            # --- Columns for Beams 1 - 4 ---
            cols = st.columns(4)
            beam_titles = ["Beam 1", "Beam 2", "Beam 3", "Beam 4"]

            for i, col in enumerate(cols):
                with col:
                    st.header(beam_titles[i])

            st.subheader("Register the Pupil Pixels")
            st.write(
                "Run this with clear pupils (phase masks out). This can take a minute."
            )

            btn_key = f"register_pupil_ALL_BEAMS"
            if st.button(f"{btn_key}"):
                success = run_script(
                    [
                        "python",
                        "calibration/pupil_registration.py",
                        "--beam_ids",
                        "1,2,3,4",
                        "--fig_path",
                        quick_data_path,
                    ]
                )

                if success:
                    cols = st.columns(4)
                    for i, col in enumerate(cols):
                        with col:
                            fig_path = os.path.join(
                                quick_data_path, f"pupil_reg_beam{i+1}.png"
                            )
                            if os.path.exists(fig_path):
                                st.image(
                                    Image.open(fig_path),
                                    caption=f"Beam {i+1} Output",
                                    use_column_width=True,
                                )
                            else:
                                st.warning(f"Cannot find {fig_path}")

            cols = st.columns(4)
            for i, col in enumerate(cols):
                with col:
                    # st.header(beam_titles[i])

                    # Unique key for each beam button
                    btn_key = f"register_pupil_beam_{i+1}"

                    # Run the script when button is clicked
                    if st.button(f"Run {btn_key}"):
                        success = run_script(QUICK_SCRIPTS[btn_key])

                        if success:
                            fig_path = os.path.join(
                                QUICK_SCRIPTS[btn_key][-1], f"pupil_reg_beam{i+1}.png"
                            )
                            if os.path.exists(fig_path):
                                st.image(
                                    Image.open(fig_path),
                                    caption=f"{btn_key} Output",
                                    use_column_width=True,
                                )
                            else:
                                st.warning(f"Cannot find {fig_path}")

            st.subheader("Register the DM Actuators in Pixel Space.")
            st.write(
                "This requires alignment on a phasemask (try H3)! This can take a minute."
            )

            btn_key = f"register_DM_ALL_BEAMS"
            if st.button(f"{btn_key}"):
                success = run_script(
                    [
                        "python",
                        "calibration/dm_registration_calibration.py",
                        "--beam_id",
                        "1,2,3,4",
                        "--fig_path",
                        quick_data_path,
                    ]
                )

                if success:
                    cols = st.columns(4)
                    for i, col in enumerate(cols):
                        with col:
                            fig_path = os.path.join(
                                quick_data_path,
                                f"beam{i+1}/DM_registration_in_pixel_space.png",
                            )
                            if os.path.exists(fig_path):
                                st.image(
                                    Image.open(fig_path),
                                    caption=f"Beam {i+1} Output",
                                    use_column_width=True,
                                )
                            else:
                                st.warning(f"Cannot find {fig_path}")

            cols = st.columns(4)
            for i, col in enumerate(cols):
                with col:
                    # st.header(beam_titles[i])

                    btn_key = f"register_DM_beam_{i+1}"

                    # Run the script when button is clicked
                    if st.button(f"Run {btn_key}"):
                        success = run_script(QUICK_SCRIPTS[btn_key])

                        if success:
                            # Note 'DM_registration_in_pixel_space.png' is generated in
                            # calibrate_transform_between_DM_and_image from DM_registration which
                            # doesn't have knowlodge of the beam number - so just overwrites the same
                            # image each time.. so looking at the most recent - fine if done immediately after running script
                            fig_path = os.path.join(
                                QUICK_SCRIPTS[btn_key][-1],
                                f"beam{i+1}/DM_registration_in_pixel_space.png",
                            )
                            if os.path.exists(fig_path):
                                st.image(
                                    Image.open(fig_path),
                                    caption=f"{btn_key} Output",
                                    use_column_width=True,
                                )
                            else:
                                st.warning(f"Cannot find {fig_path}")

            st.subheader("Register Strehl Proxy Pixels")
            st.write(
                "This requires alignment on a phasemask (try H3)! This can take a minute."
            )

            faint_mode = st.checkbox(
                "faint mode (6x6 pixels)", value=False, key="faint_mode"
            )

            btn_key = f"register_strehl_pixels_ALL_BEAMS"
            if st.button(f"{btn_key}"):
                if faint_mode:
                    success = run_script(
                        [
                            "python",
                            "calibration/strehl_filter_registration.py",
                            "--beam_id",
                            "1,2,3,4",
                            "--mode",
                            "faint",
                            "--fig_path",
                            quick_data_path,
                        ]
                    )
                else:
                    success = run_script(
                        [
                            "python",
                            "calibration/strehl_filter_registration.py",
                            "--beam_id",
                            "1,2,3,4",
                            "--mode",
                            "bright",
                            "--fig_path",
                            quick_data_path,
                        ]
                    )

                if success:
                    cols = st.columns(4)
                    for i, col in enumerate(cols):
                        with col:
                            fig_path = os.path.join(
                                quick_data_path, f"strehl_pixel_filter{i+1}.png"
                            )
                            if os.path.exists(fig_path):
                                st.image(
                                    Image.open(fig_path),
                                    caption=f"Beam {i+1} Output",
                                    use_column_width=True,
                                )
                            else:
                                st.warning(f"Cannot find {fig_path}")

            cols = st.columns(4)
            for i, col in enumerate(cols):
                with col:
                    btn_key = f"register_strehl_pixels_beam_{i+1}"
                    if st.button(f"Run {btn_key}"):
                        if faint_mode:
                            success = run_script(
                                QUICK_SCRIPTS[btn_key] + ["--mode", "faint"]
                            )
                        else:
                            success = run_script(QUICK_SCRIPTS[btn_key])

                        if success:
                            # Note 'DM_registration_in_pixel_space.png' is generated in
                            # calibrate_transform_between_DM_and_image from DM_registration which
                            # doesn't have knowlodge of the beam number - so just overwrites the same
                            # image each time.. so looking at the most recent - fine if done immediately after running script
                            fig_path = os.path.join(
                                QUICK_SCRIPTS[btn_key][-1],
                                f"strehl_pixel_filter{i+1}.png",
                            )
                            if os.path.exists(fig_path):
                                st.image(
                                    Image.open(fig_path),
                                    caption=f"{btn_key} Output",
                                    use_column_width=True,
                                )
                            else:
                                st.warning(f"Cannot find {fig_path}")

            ##########################################
            # COMMENT THINGS OUT THAT WE DONT WANT TO SHOW IN DEMO
            ##########################################
            # st.subheader("Build Strehl Models")
            # st.write("have the phasemask's well aligned prior to starting.")
            # st.write("We are going to apply varying degrees of turbulence and analyse the signals")

            # phasemask_input = st.text_input("phasemask", "H3")
            # cam_fps_input = st.number_input("Camera frames per seconds", min_value=100, max_value=1000, value=100, step=1)
            # cam_gain_input = st.number_input("Camera gain", min_value=1, max_value=10, value=1, step=1)

            # # "--max_time",f"{120}","--number_of_iterations",f"{10000}"
            # STREHL_SCRIPTS = {
            # "build_Strehl_model_beam_1": ["python", "calibration/build_strehl_model.py", "--beam_id", "1", "--phasemask", phasemask_input,"--max_time",f"{10}","--number_of_iterations",f"{10000}", "--cam_fps", f"{cam_fps_input}", "--cam_gain", f"{cam_gain_input}", "--fig_path", quick_data_path ],
            # "build_Strehl_model_beam_2": ["python", "calibration/build_strehl_model.py", "--beam_id", "2", "--phasemask", phasemask_input,"--max_time",f"{10}","--number_of_iterations",f"{10000}", "--cam_fps", f"{cam_fps_input}", "--cam_gain", f"{cam_gain_input}", "--fig_path", quick_data_path ],
            # "build_Strehl_model_beam_3": ["python", "calibration/build_strehl_model.py", "--beam_id", "3", "--phasemask", phasemask_input,"--max_time",f"{10}","--number_of_iterations",f"{10000}", "--cam_fps", f"{cam_fps_input}", "--cam_gain", f"{cam_gain_input}", "--fig_path", quick_data_path ],
            # "build_Strehl_model_beam_4": ["python", "calibration/build_strehl_model.py", "--beam_id", "4", "--phasemask", phasemask_input,"--max_time",f"{10}","--number_of_iterations",f"{10000}", "--cam_fps", f"{cam_fps_input}", "--cam_gain", f"{cam_gain_input}", "--fig_path", quick_data_path ],
            # }

            # btn_key = f"build_Strehl_model_ALL_BEAMS"
            # if st.button(f"{btn_key}"):
            #     success = run_script(["python", "calibration/build_strehl_model.py", "--beam_id", "1,2,3,4", "--phasemask", phasemask_input, "--max_time","10.0","--number_of_iterations","10000","--cam_fps", f"{cam_fps_input}", "--cam_gain", f"{cam_gain_input}", "--fig_path", quick_data_path ] )

            #     if success:
            #         cols = st.columns(4)
            #         for i, col in enumerate(cols):
            #             with col:
            #                 fig_path = os.path.join(quick_data_path,  f'strehl_model_beam{i+1}.png')
            #                 if os.path.exists(fig_path):
            #                     st.image(Image.open(fig_path), caption=f"Beam {i+1} Output", use_column_width=True)
            #                 else:
            #                     st.warning(f"Cannot find {fig_path}")

            # cols = st.columns(4)
            # for i, col in enumerate(cols):
            #     with col:
            #         btn_key = f"build_Strehl_model_beam_{i+1}"
            #         if st.button(f"Run {btn_key}"):
            #             success = run_script(STREHL_SCRIPTS[btn_key])

            #             if success:
            #                 # Note 'DM_registration_in_pixel_space.png' is generated in
            #                 # calibrate_transform_between_DM_and_image from DM_registration which
            #                 # doesn't have knowlodge of the beam number - so just overwrites the same
            #                 # image each time.. so looking at the most recent - fine if done immediately after running script
            #                 fig_path = os.path.join(STREHL_SCRIPTS[btn_key][-1], f"strehl_model_beam{i+1}.png")
            #                 if os.path.exists(fig_path):
            #                     st.image(Image.open(fig_path), caption=f"{btn_key} Output", use_column_width=True)
            #                 else:
            #                     st.warning(f"Cannot find {fig_path}")

            # st.subheader("Build Interaction Matrix")
            # st.write("have the phasemask's well aligned prior to starting.")

            # cam_fps = st.number_input("Frames per sec", min_value=100, max_value=1500, value=100, step=50)
            # cam_gain = st.number_input("Gain", min_value=1, max_value=20, value=1, step=1)
            # signal_space =  st.selectbox('signal space for IM',('dm','pixel'))
            # DM_flat = st.selectbox('DM flat',('baldr','factory'))
            # basis_name = st.selectbox("Basis Name",('zonal','zernike')) #st.text_input("Basis Name", "zonal")
            # poke_amp = st.number_input("Poke Amplitude", min_value=0.0, max_value=0.1, value=0.05, step=0.01)
            # Nmodes = st.number_input("Number of Modes to Probe (zonal does 140 automatically)", min_value=1, max_value=140, value=140, step=1)
            # phasemask = st.text_input("Phasemask", "H3")

            # IM_SCRIPTS = {
            # "build_IM_beam_1": ["python", "calibration/build_IM.py", "--beam_id", "1","--cam_fps",f"{cam_fps}","--cam_gain",f"{cam_gain}","--basis_name", f"{basis_name}","--DM_flat",f"{DM_flat}","--signal_space",f"{signal_space}","--Nmodes",f"{Nmodes}", "--phasemask", f"{phasemask}","--fig_path", quick_data_path ],
            # "build_IM_beam_2": ["python", "calibration/build_IM.py", "--beam_id", "2","--cam_fps",f"{cam_fps}","--cam_gain",f"{cam_gain}","--basis_name", f"{basis_name}","--DM_flat",f"{DM_flat}","--signal_space",f"{signal_space}","--Nmodes",f"{Nmodes}", "--phasemask", f"{phasemask}","--fig_path", quick_data_path ],
            # "build_IM_beam_3": ["python", "calibration/build_IM.py", "--beam_id", "3","--cam_fps",f"{cam_fps}","--cam_gain",f"{cam_gain}","--basis_name", f"{basis_name}","--DM_flat",f"{DM_flat}","--signal_space",f"{signal_space}","--Nmodes",f"{Nmodes}", "--phasemask", f"{phasemask}","--fig_path", quick_data_path ],
            # "build_IM_beam_4": ["python", "calibration/build_IM.py", "--beam_id", "4","--cam_fps",f"{cam_fps}","--cam_gain",f"{cam_gain}","--basis_name", f"{basis_name}","--DM_flat",f"{DM_flat}","--signal_space",f"{signal_space}","--Nmodes",f"{Nmodes}", "--phasemask", f"{phasemask}","--fig_path", quick_data_path ],
            # }

            # cols = st.columns(4)
            # for i, col in enumerate(cols):
            #     with col:

            #         btn_key = f"build_IM_beam_{i+1}"

            #         # Run the script when button is clicked
            #         if st.button(f"Run {btn_key}"):
            #             success = run_script(IM_SCRIPTS[btn_key])

            #             if success:
            #                 fig_path = os.path.join(IM_SCRIPTS[btn_key][-1], f'IM_singularvalues_beam{i+1}.png')
            #                 if os.path.exists(fig_path):
            #                     st.image(Image.open(fig_path), caption=f"{btn_key} Output", use_column_width=True)
            #                 else:
            #                     st.warning(f"Cannot find {fig_path}")

            # st.subheader("Process Interaction Matrix (Build Control Matrix)")
            # st.write("have the phasemask's well aligned prior to starting.")

            # LO = st.number_input("Max Zernike Index considered for low order modes (2 = tip/tilt)", min_value=2, max_value=10, value=2, step=1)
            # inverse_method = st.selectbox("IM matrix inverse method",["zonal","map","pinv"]+[f"svd_truncation-{x}" for x in [2,5,10,15,20,30,40,50,60,70,80,90]])
            # phasemask = st.text_input("Phasemask", "H3", key="phasemask_ctrl_matrix")

            # PROCESS_IM_SCRIPTS = {
            # "proc_IM_beam_1": ["python", "calibration/build_baldr_control_matrix.py", "--beam_id", "1","--LO",f"{LO}","--inverse_method",f"{inverse_method}", "--phasemask", f"{phasemask}","--fig_path", quick_data_path ],
            # "proc_IM_beam_2": ["python", "calibration/build_baldr_control_matrix.py", "--beam_id", "2","--LO",f"{LO}","--inverse_method",f"{inverse_method}", "--phasemask", f"{phasemask}","--fig_path", quick_data_path ],
            # "proc_IM_beam_3": ["python", "calibration/build_baldr_control_matrix.py", "--beam_id", "3","--LO",f"{LO}","--inverse_method",f"{inverse_method}", "--phasemask", f"{phasemask}","--fig_path", quick_data_path ],
            # "proc_IM_beam_4": ["python", "calibration/build_baldr_control_matrix.py", "--beam_id", "4","--LO",f"{LO}","--inverse_method",f"{inverse_method}", "--phasemask", f"{phasemask}","--fig_path", quick_data_path ],
            # }

            # cols = st.columns(4)
            # for i, col in enumerate(cols):
            #     with col:

            #         btn_key = f"proc_IM_beam_{i+1}"

            #         # Run the script when button is clicked
            #         if st.button(f"Run {btn_key}"):
            #             success = run_script(PROCESS_IM_SCRIPTS[btn_key])

            #             if success:
            #                 fig_path = os.path.join(PROCESS_IM_SCRIPTS[btn_key][-1], f'IM_singularvalues_beam{i+1}.png')
            #                 if os.path.exists(fig_path):
            #                     st.image(Image.open(fig_path), caption=f"{btn_key} Output", use_column_width=True)
            #                 else:
            #                     st.warning(f"Cannot find {fig_path}")

            # st.subheader("Apply Turbulence")
            # # Number of iterations
            # number_of_iterations_input = st.text_input("Number of Iterations", value="10")

            # # Simulation wavelength (um)
            # wvl_input = st.text_input("Simulation wavelength (um)", value="1.65")

            # # Telescope diameter (m)
            # D_tel_input = st.text_input("Telescope Diameter (m)", value="1.8")

            # # Fried parameter (r0) at 500nm (m)
            # r0_input = st.text_input("Fried Parameter r0 (m)", value="0.15")

            # # Equivalent turbulence velocity (m/s)
            # V_input = st.text_input("Equivalent Turbulence Velocity (m/s)", value="0.50")

            # # Number of Zernike modes removed
            # number_of_modes_removed_input = st.text_input("Number of Zernike Modes Removed", value="0")

            # # DM channel on shared memory
            # DM_chn_input = st.text_input("DM Channel (0,1,2,3)", value="3")

            # # Record telemetry: directory/name.fits or "None"
            # record_telem_input = st.text_input("Record Telemetry (directory/name.fits or None)", value="None")

            # # For record_telem, treat "None" (case insensitive) as an omitted argument.
            # if record_telem_input.strip().lower() == "none":
            #     record_telem_parsed = None
            # else:
            #     record_telem_parsed = record_telem_input.strip()

            # # --- Build the Command Argument List ---
            # args = [
            #     "--number_of_iterations", number_of_iterations_input,
            #     "--wvl", wvl_input,
            #     "--D_tel", D_tel_input,
            #     "--r0", r0_input,
            #     "--V", V_input,
            #     "--number_of_modes_removed", number_of_modes_removed_input,
            #     "--DM_chn", DM_chn_input,
            # ]

            # TURB_SCRIPTS = {
            # "turb_beam_1": ["python", "common/turbulence.py"] + ["--beam_id", "1"] + args,
            # "turb_beam_2": ["python", "common/turbulence.py"] + ["--beam_id", "2"] + args,
            # "turb_beam_3": ["python", "common/turbulence.py"] + ["--beam_id", "3"] + args,
            # "turb_beam_4": ["python", "common/turbulence.py"] + ["--beam_id", "4"] + args,
            # }

            # cols = st.columns(4)
            # for i, col in enumerate(cols):
            #     with col:

            #         btn_key = f"turb_beam_{i+1}"

            #         # Run the script when button is clicked
            #         if st.button(f"Run {btn_key}"):
            #             success = run_script(TURB_SCRIPTS[btn_key])

            #             if success:
            #                 st.write("DONE")

            # st.subheader("Close Loop")

            # CLOSE_LOOP_SCRIPT = {"close_beam_2" :  ["python", "playground/baldr_CL/CL.py", "--number_of_iterations","1000"]}
            # btn_key = f"close_beam_2"
            # if st.button(f"Run {btn_key}"):
            #     success = run_script(CLOSE_LOOP_SCRIPT[btn_key])

        if routine_options == "Heimdallr shutters":
            # all up and all down buttons

            st.title("Heimdallr Shutters Control")

            if st.button("Open All Shutters"):
                msg = "h_shut open all"
                response = send_and_get_response(msg)
                st.write(f"{response}")

            if st.button("Close All Shutters"):
                msg = "h_shut close all"
                response = send_and_get_response(msg)
                st.write(f"{response}")

            cols = st.columns(4)

            for i, col in enumerate(cols):
                with col:
                    if st.button(f"Open {i+1}"):
                        msg = f"h_shut open {i+1}"
                        response = send_and_get_response(msg)
                        st.write(f"{response}")
                    if st.button(f"Close {i+1}"):
                        msg = f"h_shut close {i+1}"
                        response = send_and_get_response(msg)
                        st.write(f"{response}")

        if routine_options == "Camera & DMs":
            st.write("testing")

            c = FLI.fli()
            camera_command = st.text_input(
                "Send Command to Camera:", key="camera_command", placeholder="fps"
            )
            try:
                resp = c.send_fli_cmd(camera_command)
                st.success(
                    f"Command '{camera_command}' sent to camera!\nresponse = {resp}"
                )
            except Exception as e:
                st.error(f"Failed to send command: {e}")

            c.close(erase_file=False)

            # if st.button("get image"):
            #     img = np.mean( c.get_data() , axis=0)

            #     import plotly.express as px

            #     # Generate a random 2D NumPy array
            #     image_array = np.random.rand(100, 100)

            #     # Convert NumPy array to interactive heatmap
            #     fig = px.imshow(img, color_continuous_scale="gray")

            #     # Display in Streamlit
            #     st.plotly_chart(fig, use_container_width=True)
            #     plt.close()
            #     # fig, ax = plt.subplots()
            #     # ax.imshow( np.log10( img ), cmap="gray")
            #     # st.pyplot(fig)

            # c.close(erase_file=False)

            # Initialize Camera and DM Objects in session state
            # if "camera" not in st.session_state:
            #     st.session_state.camera = FLI.fli()  # Open the camera shared memory object

            # beam_ids = [1, 2, 3, 4]  # IDs for the four beams

            # if "dm_shm_dict" not in st.session_state:
            #     st.session_state.dm_shm_dict = {beam_id: dmclass(beam_id=beam_id) for beam_id in beam_ids}

            # if "apply_dark" not in st.session_state:
            #     st.session_state.apply_dark = False  # Default: Dark correction disabled

            # # Streamlit UI Layout
            # st.title("Live Camera & Deformable Mirror Commands")

            # # Placeholder for Camera Frame
            # camera_placeholder = st.empty()

            # # Camera Command Input (Runs on Enter)
            # camera_command = st.text_input("Send Command to Camera:", key="camera_command", placeholder="Enter command and press Enter")

            # if camera_command:
            #     try:
            #         resp = st.session_state.camera.send_fli_cmd(camera_command)
            #         st.success(f"Command '{camera_command}' sent to camera!\nresponse = {resp}")
            #     except Exception as e:
            #         st.error(f"Failed to send command: {e}")

            # # "Build Dark" Button
            # if st.button("Build Dark"):
            #     try:
            #         st.session_state.camera.build_manual_dark()
            #         st.success("Dark frame built successfully.")
            #     except Exception as e:
            #         st.error(f"Failed to build dark frame: {e}")

            # # "Apply Dark" Checkbox
            # apply_dark = st.checkbox("Apply Dark Correction", value=st.session_state.apply_dark)
            # st.session_state.apply_dark = apply_dark
            # #st.write( st.session_state.apply_dark )

            # # Create four columns for DM commands
            # dm_columns = st.columns(4)
            # dm_placeholders = {beam_id: col.empty() for beam_id, col in zip(beam_ids, dm_columns)}

            # # UI Controls: Buttons, Dropdowns, and Input Fields (Created **once**)
            # for beam_id, col in zip(beam_ids, dm_columns):
            #     with col:
            #         st.subheader(f"Beam {beam_id}")

            #         # Buttons for DM control
            #         if st.button(f"Zero All - Beam {beam_id}", key=f"zero_{beam_id}"):
            #             st.session_state.dm_shm_dict[beam_id].zero_all()
            #             st.success(f"Beam {beam_id} set to zero.")

            #         if st.button(f"Flatten DM - Beam {beam_id}", key=f"flatten_{beam_id}"):
            #             st.session_state.dm_shm_dict[beam_id].activate_flat()
            #             st.success(f"Beam {beam_id} flattened.")

            #         # Dropdown menus for "Apply Shape"
            #         st.subheader("Apply Shape")
            #         selected_channel = st.selectbox(f"Select Channel (Beam {beam_id})", ["Channel 1", "Channel 2", "Channel 3"], key=f"channel_{beam_id}")
            #         selected_basis = st.selectbox(f"Select Basis (Beam {beam_id})", ["Basis A", "Basis B", "Basis C"], key=f"basis_{beam_id}")

            #         # Input for amplitude
            #         amplitude = st.text_input(f"Enter Amplitude (Beam {beam_id})", key=f"amplitude_{beam_id}", placeholder="e.g., 0.5")

            # # **Main Loop** for Live Updates (Camera & DM Plots)
            # if st.button("Update Camera & DM Signals"): #while True:
            #     # Get the latest camera frame
            #     camera_frame = np.mean(st.session_state.camera.get_data(apply_manual_reduction=st.session_state.apply_dark, which_index=-1), axis=0)

            #     # Update Camera Frame
            #     fig_cam, ax_cam = plt.subplots(figsize=(8, 6))
            #     ax_cam.imshow(camera_frame, cmap="gray", origin="upper")
            #     ax_cam.set_title("Camera Frame")
            #     ax_cam.axis("off")
            #     camera_placeholder.pyplot(fig_cam)
            #     plt.close(fig_cam)  # Prevent memory leaks

            #     # Update DM Command Visualizations in four columns
            #     for beam_id in beam_ids:
            #         dm_command = st.session_state.dm_shm_dict[beam_id].shm0.get_data()

            #         fig_dm, ax_dm = plt.subplots(figsize=(3, 3))
            #         ax_dm.imshow(dm_command, cmap="viridis", origin="upper")
            #         ax_dm.set_title(f"Beam {beam_id} - DM Command")
            #         ax_dm.axis("off")

            #         dm_placeholders[beam_id].pyplot(fig_dm)
            #         plt.close(fig_dm)  # Prevent memory leaks

            #     # Refresh every second
            #     time.sleep(1)

        if routine_options == "Illumination":
            # a few options to control sources, source position and flipper states

            # refresh button
            if st.button("Refresh"):
                pass

            use_sol = st.checkbox("use solarstein?", True)

            # flippers
            st.subheader("Flippers")

            names = [f"SSF{i}" for i in range(1, 5)]

            if st.button("All up :arrow_double_up:"):
                for i, flipper in enumerate(names):
                    message = f"moveabs {flipper} 1.0"
                    res = send_and_get_response(message)
            if st.button("All down :arrow_double_down:"):
                for i, flipper in enumerate(names):
                    message = f"moveabs {flipper} 0.0"
                    res = send_and_get_response(message)

            flipper_cols = st.columns(4)

            for i, flipper in enumerate(names):
                with flipper_cols[i]:
                    st.markdown(f"<h4><b>{flipper}</b></h4>", unsafe_allow_html=True)
                    if st.button(f"Up", key=f"up__{flipper}"):
                        message = f"moveabs {flipper} 1.0"
                        res = send_and_get_response(message)
                        # st.write(res)
                    if st.button(f"Down", key=f"down__{flipper}"):
                        message = f"moveabs {flipper} 0.0"
                        res = send_and_get_response(message)
                        # st.write(res)
                        # refresh

                    cur_state = send_and_get_response(f"state {flipper}")
                    # st.write(f"Current state: {cur_state}")
                    if "IN" in cur_state:
                        if use_sol:
                            st.success("IN")
                        else:
                            st.error("IN")
                    elif "OUT" in cur_state:
                        if use_sol:
                            st.error("OUT")
                        else:
                            st.success("OUT")
                    else:
                        st.warning("Unknown")

            if use_sol:
                # source position
                st.subheader("Source Position")
                target = "SSS"
                mapping = {
                    k: v + st.session_state[f"{target}_offset"]
                    for k, v in st.session_state[f"{target}_fixed_mapping"].items()
                }

                but, txt = st.columns([1, 1])
                res = None
                with but:
                    if st.button("Read Position"):
                        message = f"read {target}"
                        res = send_and_get_response(message)
                with txt:
                    if res:
                        # check if close to any preset position
                        for pos, val in mapping.items():
                            if np.isclose(float(res), val, atol=0.1):
                                st.write(f"Pos: {float(res):.2f} mm ({pos})")
                                break
                        else:
                            st.write(f"Current position: {float(res):.2f} mm")

                buttons = ["SRL", "SGL", "SLD/SSP", "SBB"]
                button_cols = st.columns(len(buttons))

                for i, button in enumerate(buttons):
                    with button_cols[i]:
                        if st.button(button):
                            print(st.session_state[f"SSS_fixed_mapping"][button])
                            message = f"moveabs SSS {st.session_state[f'SSS_fixed_mapping'][button]}"
                            res = send_and_get_response(message)
                            st.write(res)

                # source on/off vertical buttons
                st.subheader("Source On/Off")
                headers = ["SRL", "SGL", "SBB"]
                header_colours = ["red", "green", "white"]
                button_cols = st.columns(len(headers))

                for i, header in enumerate(headers):
                    with button_cols[i]:
                        st.markdown(
                            f'<h4 style="color:{header_colours[i]};">{header}</h4>',
                            unsafe_allow_html=True,
                        )
                        if st.button(f"{header} On"):
                            message = f"on {header}"
                            res = send_and_get_response(message)
                            # st.write(res)
                        if st.button(f"{header} Off"):
                            message = f"off {header}"
                            res = send_and_get_response(message)
                            # st.write(res)

        if routine_options == "Move image/pupil":
            # we save the intial positions when opening the pannel / changing beams/configs
            # to allow user to easily return to initial state
            # so we init the following session states to save rel info!
            if "moveImPup" not in st.session_state:
                st.session_state.moveImPup = {}  # Initialize as a dictionary

            if "original_positions" not in st.session_state.moveImPup:
                st.session_state.moveImPup["original_positions"] = {}

            if "prev_beam" not in st.session_state.moveImPup:
                st.session_state.moveImPup["prev_beam"] = None

            if "prev_config" not in st.session_state.moveImPup:
                st.session_state.moveImPup["prev_config"] = None

            if "phasemask_offset_BMX" not in st.session_state.moveImPup:
                st.session_state.moveImPup["phasemask_offset_BMX"] = None

            if "phasemask_offset_BMY" not in st.session_state.moveImPup:
                st.session_state.moveImPup["phasemask_offset_BMY"] = None

            # first_instance = True
            # original_pos = {}

            col1, col2, col3 = st.columns(3)
            with col1:
                move_what = st.selectbox(
                    "Pick operating_mode",
                    ["move_image", "move_pupil"],
                    key="move_what",
                )

            with col2:
                config = st.selectbox(
                    "Pick a config",
                    ["c_red_one_focus", "intermediate_focus", "baldr"],
                    key="config",
                )

            if move_what == "move_image":
                units = "pixels"
            else:
                units = "mm"

            with col3:
                beam = st.selectbox(
                    "Pick a beam",
                    list(range(1, 5)),
                    key="beam",
                )

            if (config == "baldr") and (beam == 1):
                st.warning(
                    "warning no BOTX motor on beam 1 - so move pupil / image for baldr on beam 1 is invalid"
                )

            # Detect changes in beam or config, update original_positions if changed!
            if (beam != st.session_state.moveImPup["prev_beam"]) or (
                config != st.session_state.moveImPup["prev_config"]
            ):
                st.write(
                    "Updating original positions due to change in beam or config..."
                )

                # Update stored previous values
                st.session_state.moveImPup["prev_beam"] = beam
                st.session_state.moveImPup["prev_config"] = config

                # Update original_positions
                if config == "baldr":
                    axes = [f"BTP{beam}", f"BTT{beam}", f"BOTP{beam}", f"BOTT{beam}"]
                else:
                    axes = [f"HTPP{beam}", f"HTTP{beam}", f"HTPI{beam}", f"HTTI{beam}"]

                pos_dict_org = {}
                for axis in axes:
                    pos = send_and_get_response(f"read {axis}")
                    pos_dict_org[axis] = pos

                st.session_state.moveImPup["original_positions"] = pos_dict_org.copy()

            # tickbox for button only mode
            button_only = st.checkbox("Use button to move", value=True)

            if not button_only:
                with st.form(key="amount"):
                    delx = st.number_input(f"delta x {units}", key="delx")
                    dely = st.number_input(f"delta y {units}", key="dely")
                    submit = st.form_submit_button("Send command")

                if submit:
                    if move_what == "move_image":
                        # asgard_alignment.Engineering.move_image(
                        #     beam, delx, dely, send_and_get_response, config
                        # )
                        cmd = f"mv_img {config} {beam} {delx} {dely}"
                        send_and_get_response(cmd)
                    elif move_what == "move_pupil":
                        # asgard_alignment.Engineering.move_pupil(
                        #     beam, delx, dely, send_and_get_response, config
                        # )
                        cmd = f"mv_pup {config} {beam} {delx} {dely}"
                        # this had no send cmd - fixed 5/3/25
                        send_and_get_response(cmd)
            else:
                # increment selection for each case

                if config == "baldr":
                    if move_what == "move_image":
                        increment = st.number_input(
                            "Increment (typically mm, remember cold stop is ~2.mm diameter)",
                            min_value=0.0,
                            max_value=1.0,
                            step=0.1,
                            key="increment",
                        )
                    else:
                        increment = st.number_input(
                            "Increment (CRED 1 pixels)",
                            min_value=0.0,
                            max_value=30.0,
                            step=0.5,
                            key="increment",
                        )

                else:
                    if move_what == "move_image":
                        increment = st.number_input(
                            "Increment (pixels)",
                            min_value=0,
                            max_value=5000,
                            step=5,
                            key="increment",
                        )
                    else:
                        increment = st.number_input(
                            "Increment (mm)",
                            min_value=0.0,
                            max_value=5.0,
                            step=0.05,
                            key="increment",
                        )

                if move_what == "move_image":
                    # move_function = asgard_alignment.Engineering.move_image
                    mv_cmd = "mv_img"
                elif move_what == "move_pupil":
                    # move_function = asgard_alignment.Engineering.move_pupil
                    mv_cmd = "mv_pup"
                else:
                    raise ValueError("Invalid move_what")

                increment = float(increment)
                pos_x = f"{mv_cmd} {config} {beam} {increment} 0.0"
                pos_y = f"{mv_cmd} {config} {beam} 0.0 {increment}"
                neg_x = f"{mv_cmd} {config} {beam} {-increment} 0.0"
                neg_y = f"{mv_cmd} {config} {beam} 0.0 {-increment}"

                # make a 3x3 grid but only use the up, down, left and right

                ul, um, ur = st.columns(3)
                ml, mm, mr = st.columns(3)
                ll, lm, lr = st.columns(3)

                # if move_what == "move_image":
                with um:
                    if st.button(f"-y: {increment:.2f}"):
                        send_and_get_response(neg_y)
                with lm:
                    if st.button(f"+y: {increment:.2f}"):
                        send_and_get_response(pos_y)
                with ml:
                    if st.button(f"-x: {increment:.2f}"):
                        send_and_get_response(neg_x)
                with mr:
                    if st.button(f"+x: {increment:.2f}"):
                        send_and_get_response(pos_x)

                # elif move_what == "move_pupil":
                #     with um:
                #         if st.button(f"+y: {increment:.2f}"):
                #             pos_y()
                #     with lm:
                #         if st.button(f"-y: {increment:.2f}"):
                #             neg_y()
                #     with ml:
                #         if st.button(f"-x: {increment:.2f}"):
                #             neg_x()
                #     with mr:
                #         if st.button(f"+x: {increment:.2f}"):
                #             pos_x()

            # also show the state of all of the motors involved
            if config == "baldr":
                axes = [
                    f"BTP{beam}",
                    f"BTT{beam}",
                    f"BOTP{beam}",
                    f"BOTT{beam}",
                ]  # [f"HTPP{beam}", f"HTTP{beam}", f"HTPI{beam}", f"HTTI{beam}"]
            else:
                axes = [f"HTPP{beam}", f"HTTP{beam}", f"HTPI{beam}", f"HTTI{beam}"]
            # print("axes", axes)

            # uncomment this
            st.write(f"Current positions of motors involved in {move_what}.")
            pos_dict = {}
            for axis in axes:
                pos = send_and_get_response(f"read {axis}")
                st.write(f"{axis}: {pos}")
                pos_dict[axis] = pos

            if st.button("update"):
                # force re-read of positions
                st.write("")

            if move_what == "move_image":
                if config == "c_red_one_focus":
                    st.write("No image (yet)")
                elif config == "intermediate_focus":
                    st.image("figs/image_plane_intermediate.png")

            if move_what == "move_pupil":
                if config == "c_red_one_focus":
                    st.write("No pupil (yet)")
                elif config == "intermediate_focus":
                    st.image("figs/pupil_plane_KE.png")

            # Show original positions for the selected beam and config
            st.header("Initial positions")
            st.write(f"for beam {beam}, config = {config}")
            for axis, pos in st.session_state.moveImPup["original_positions"].items():
                st.write(f"{axis} : {pos}")

            # Move back button to save a life ----
            if st.button("SAVE MY LIFE:\nMove Back to Original Position"):
                for axis, pos in st.session_state.moveImPup[
                    "original_positions"
                ].items():
                    msg = f"moveabs {axis} {pos}"

                    send_and_get_response(msg)
                    st.write(
                        f"phew! Moving {axis} back to {pos}. Remember to drink water!"
                    )

            st.title("Update Phasemask Poisitions")

            st.write(
                "Moving the Baldr OAP tip/tilt motor (BOTX) - \
                     which is involved in move image/pupil for the \
                     baldr configuration - obviously moves the beam\
                      on the phasemask. Use this section to correctly \
                     offset the phasemasks based on any BOTX movements. \
                     Use the 'update (write new)' button if you plane to maintain the new positions of the BOTX motors"
            )
            if st.button(
                "Calculate phasemask offset based on BOTX changes (current-original positions)"
            ):

                # matrix to map relative BOTX offsets to phasemask (BMX/BMY) offsets
                phasemask_matrix = asgard_alignment.Engineering.phasemask_botx_matricies

                for axis in [f"BOTP{beam}", f"BOTT{beam}"]:
                    pos = send_and_get_response(f"read {axis}")
                    if axis == f"BOTP{beam}":
                        current_BOTP = float(pos)
                    if axis == f"BOTT{beam}":
                        current_BOTT = float(pos)

                original_BOTP = float(
                    st.session_state.moveImPup["original_positions"].get(
                        f"BOTP{beam}", None
                    )
                )
                if original_BOTP is None:
                    st.Warning("original_BOTP is None. Cannot update phasemasks")
                original_BOTT = float(
                    st.session_state.moveImPup["original_positions"].get(
                        f"BOTT{beam}", None
                    )
                )
                if original_BOTT is None:
                    st.Warning("original_BOTT is None. Cannot update phasemasks")

                delta_BOTP = current_BOTP - original_BOTP
                delta_BOTT = current_BOTT - original_BOTT

                st.write(f"delta_BOTP, delta_BOTT = {delta_BOTP},{delta_BOTT}")
                st.write(
                    f"botx - phasemask offset matrix:{phasemask_matrix[int(beam)]}"
                )

                delta_BMX, delta_BMY = phasemask_matrix[int(beam)] @ [
                    delta_BOTP,
                    delta_BOTT,
                ]

                st.session_state.moveImPup["phasemask_offset_BMX"] = delta_BMX
                st.session_state.moveImPup["phasemask_offset_BMY"] = delta_BMY

                st.write(
                    f'calculated phasemask offset (um): delta_BMX{beam}, delta_BMY{beam} = {round(st.session_state.moveImPup["phasemask_offset_BMX"],1)},{round( st.session_state.moveImPup["phasemask_offset_BMY"],1)}'
                )

            if st.button(
                f'apply phasemask offset: delta_BMX{beam}, delta_BMY{beam} = {st.session_state.moveImPup["phasemask_offset_BMX"]},{st.session_state.moveImPup["phasemask_offset_BMY"]}um'
            ):

                dbmx = st.session_state.moveImPup["phasemask_offset_BMX"]
                dbmy = st.session_state.moveImPup["phasemask_offset_BMY"]

                response = send_and_get_response(f"moverel BMX{beam} {dbmx}")
                if "ACK" in response:
                    st.success(f"{dbmx}um offset successfully applied to BMX{beam}")
                else:
                    st.error(
                        f"Failed to apply offset to BMX{beam}. Response: {response}"
                    )

                resp = send_and_get_response(f"moverel BMY{beam} {dbmy}")
                if "ACK" in response:
                    st.success(f"{dbmy}um offset successfully applied to BMY{beam}")
                else:
                    st.error(
                        f"Failed to apply offset to BMY{beam}. Response: {response}"
                    )

            if st.button(
                f"update (write new) phasemask position file for beam{beam} based on offsets"
            ):

                dbmx = st.session_state.moveImPup["phasemask_offset_BMX"]
                dbmy = st.session_state.moveImPup["phasemask_offset_BMY"]

                response = send_and_get_response(
                    f"fpm_offsetallmaskpositions phasemask{beam} {dbmx} {dbmy}"
                )
                if "ACK" in response:
                    st.success(f"offset all mask positions for beam{beam} locally")
                else:
                    st.error(
                        f"Failed to apply offset to all phasemasks on beam{beam}. Response: {response}"
                    )

                save_message = f"fpm_writemaskpos phasemask{beam}"
                save_res = send_and_get_response(save_message)

                if "NACK" in save_res:
                    st.error(
                        f"Failed to save updated positions"
                    )  # to file: {save_res}")
                else:
                    st.success(
                        "Updated positions successfully saved to file"  # at: " + save_path
                    )

        if routine_options == "Phasemask Alignment":

            beam_numbers = [1, 2, 3, 4]
            beam_number = st.selectbox(
                "Select Beam Number",
                beam_numbers,
                key="beam_number",
            )

            targets = [f"phasemask{beam_number}"]

            handle_phasemask()

        if routine_options == "Save state":
            instruments = ["Heimdallr", "Baldr", "Solarstein", "All"]
            # grid of 3 rows, 2 cols, with first col being the save location
            # and second col being the save button
            for i, instr in enumerate(instruments):
                col1, col2 = st.columns(2)
                with col1:
                    save_location = st.text_input(
                        f"Save {instr}", key=f"save_location_{i}"
                    )
                with col2:
                    if st.button(f"Save {instr}"):
                        motor_names = []
                        if instr == "Solarstein" or instr == "All":
                            motor_names += ["SDLA", "SDL12", "SDL34", "SSS", "SSF"]
                        if instr == "Heimdallr" or instr == "All":
                            send_and_get_response("h_shut open all")
                            time.sleep(2)
                            motor_names_all_beams = [
                                "HFO",
                                "HTPP",
                                "HTPI",
                                "HTTP",
                                "HTTI",
                            ]

                            for motor in motor_names_all_beams:
                                for beam_number in range(1, 5):
                                    motor_names.append(f"{motor}{beam_number}")
                        if instr == "Baldr" or instr == "All":
                            motor_names += ["BFO"]

                            motor_names_all_beams = [
                                "BDS",
                                "BTT",
                                "BTP",
                                "BMX",
                                "BMY",
                                "BLF",
                            ]

                            partially_common_motors = [
                                "BOTT",
                                "BOTP",
                            ]

                            for motor in partially_common_motors:
                                for beam_number in range(2, 5):
                                    motor_names.append(f"{motor}{beam_number}")

                            for motor in motor_names_all_beams:
                                for beam_number in range(1, 5):
                                    motor_names.append(f"{motor}{beam_number}")

                        states = []
                        for name in motor_names:
                            message = f"read {name}"
                            res = send_and_get_response(message)

                            if "NACK" in res:
                                is_connected = False
                            else:
                                is_connected = True

                            state = {
                                "name": name,
                                "is_connected": is_connected,
                            }
                            print(res, type(res), is_connected)
                            if is_connected:
                                if res != "None":
                                    state["position"] = float(res)
                                print()
                            states.append(state)

                        fname = "instr_states/" + save_location + ".json"
                        if os.path.exists(fname):
                            st.error(f"File {fname} already exists")
                        else:
                            # save to json at location
                            with open(fname, "w") as f:
                                json.dump(states, f, indent=4)

        if routine_options == "Scan Mirror":

            st.warning(
                "Bug: sometimes session state gets stuck and provokes and error, just click on a different beam and back again - usually this resolves it"
            )

            # if "scan_running" not in st.session_state:
            #    st.session_state.scan_running = False
            if "moveImPup" not in st.session_state:
                st.session_state.moveImPup = {}  # Initialize as a dictionary

            if "original_positions" not in st.session_state.moveImPup:
                st.session_state.moveImPup["original_positions"] = {}

            if "prev_beam" not in st.session_state.moveImPup:
                st.session_state.moveImPup["prev_beam"] = None

            if "prev_config" not in st.session_state.moveImPup:
                st.session_state.moveImPup["prev_config"] = None

            if "img_json_file" not in st.session_state.moveImPup:
                # to hold the scan dictionary file path if saved
                st.session_state.moveImPup["img_json_file"] = None

            if "baldr_pupils" not in st.session_state:
                st.session_state.baldr_pupils = None
            if "heim_pupils" not in st.session_state:
                st.session_state.heim_pupils = None

            st.title("Scan Mirror Control Panel")
            st.write(
                "Scan a mirror (or combination of mirrors) and analyse the signal in the CRED 1 as a function of scanned coorodinates. Currently does not automatically applying an offset to better center it around the detected edges."
            )

            # User inputs for search parameters

            beam = st.selectbox(
                "Pick a beam",
                list(range(1, 5)),
                key="beam",
            )

            look_where = st.selectbox(
                "What region of the camera to look at?",
                [
                    "Baldr Beam",
                    "Heimdallr K1",
                    "Heimdallr K2",
                    "custom region",
                ],  # ,"whole camera"],
                key="look_where",
            )

            scantype = st.selectbox(
                "type of scan",
                [
                    "square_spiral",
                    "raster",
                    "cross",
                ],  # cross tested in software not hardware (20/6/25)
                key="scantype",
            )

            # configuration info (like where to crop for each beam)
            toml_file = os.path.join("config_files", "baldr_config_#.toml")

            if beam != st.session_state.moveImPup["prev_beam"]:
                # we don't want to hold on to toml file since large!
                with open(toml_file.replace("#", f"{beam}")) as file:
                    configdata = toml.load(file)
                    # Extract the "baldr_pupils" section
                    baldr_pupils = configdata.get("baldr_pupils", {})
                    heim_pupils = configdata.get("heimdallr_pupils", {})

                # Store only the required sections
                st.session_state.baldr_pupils = baldr_pupils
                st.session_state.heim_pupils = heim_pupils

                # Update last selected beam
                st.session_state.last_beam = beam

            if look_where == "Baldr Beam":
                # st.write( st.session_state.baldr_pupils )
                roi = st.session_state.baldr_pupils[f"{beam}"]
            elif look_where == "Heimdallr K1":
                roi = st.session_state.heim_pupils["K1"]
            elif look_where == "Heimdallr K2":
                roi = st.session_state.heim_pupils["K2"]
            elif look_where == "custom region":
                r1_str = st.text_input("Row start (r1)", "0")
                r2_str = st.text_input("Row end (r2)", "256")
                c1_str = st.text_input("Col start (c1)", "0")
                c2_str = st.text_input("Col end (c2)", "320")
                roi = None

                # Parse and validate input
                try:
                    r1 = int(r1_str)
                    r2 = int(r2_str)
                    c1 = int(c1_str)
                    c2 = int(c2_str)

                    # Hardcoded bounds
                    if not (
                        0 <= r1 <= 256
                        and 0 <= r2 <= 256
                        and 0 <= c1 <= 320
                        and 0 <= c2 <= 320
                    ):
                        st.sidebar.error(
                            "All values must be between valid range (CRED 1 is a 256x320 pixel array)."
                        )
                    elif r1 >= r2 or c1 >= c2:
                        st.sidebar.error("Must satisfy: r1 < r2 and c1 < c2.")
                    else:
                        roi = [r1, r2, c1, c2]
                        st.sidebar.success(f"Selected ROI: {roi}")

                except ValueError:
                    st.sidebar.error("Please enter valid integers.")

                roi = [int(r1), int(r2), int(c1), int(c2)]
            else:
                st.write("invalid selection")
                st.warning("invalid selection")

            search_radius = st.text_input("Search Radius:", "0.3")
            st.write(
                "for individual mirrors this is in the motor units. For move pupil it is generally in units of pixels, while for move image it is generally units of mm (try 0.1)"
            )
            dx = st.text_input("Step Size (dx):", "0.05")

            # x0 = st.text_input("Initial X Position (x0):", "0.0")
            # y0 = st.text_input("Initial Y Position (y0):", "0.0")

            st.title("Individual Mirrors")

            motor = st.text_input("Motor Name:", "BTX")
            st.write("enter x,y start point or current to start from current position")
            start_pos = st.text_input("start position", "current")

            data_path_move_ind_mirror = f"/home/asg/Progs/repos/asgard-alignment/calibration/reports/scan_{motor}/"
            if not os.path.exists(data_path_move_ind_mirror):
                print(f"made directory : {data_path_move_ind_mirror}")
                os.makedirs(data_path_move_ind_mirror)

            # "/home/asg/Progs/repos/asgard-alignment/figs/"

            # Button to execute the script
            if st.button("Run Scan", key="run_individual_scan"):
                # copied from Engineering GUI
                if motor in ["HTXP", "HTXI", "BTX", "BOTX"]:
                    # replace the X in target with P
                    target = f"{motor}{beam}"
                    targets = [target.replace("X", "P"), target.replace("X", "T")]

                # try read the positions first as a check
                try:
                    message = f"read {targets[0]}"
                    initial_Xpos = float(send_and_get_response(message))

                    message = f"read {targets[1]}"
                    initial_Ypos = float(send_and_get_response(message))

                    # this could cause bugs between individual mirrors and move pupil
                    # st.session_state.moveImPup["original_positions"] = {targets[0]:initial_Xpos, targets[1]:initial_Ypos}
                except:
                    raise UserWarning(
                        "failed 'read {args.motor}X{args.beam}' or  'read {args.motor}Y{args.beam}'"
                    )

                if (
                    1
                ):  # not st.session_state.scan_running:  # Prevents multiple runs at once
                    st.session_state.scan_running = True
                    command = [
                        "python",
                        "common/m_scan_mirrors.py",
                        "--beam",
                        f"{beam}",
                        "--motor",
                        motor,
                        "--search_radius",
                        search_radius,
                        "--dx",
                        dx,
                        "--initial_pos",
                        start_pos,
                        "--roi",
                        str(roi),
                        "--scantype",
                        scantype,
                        "--data_path",
                        data_path_move_ind_mirror,
                    ]

                    # Run the external script
                    with st.spinner("Running scan..."):
                        process = subprocess.run(
                            command, capture_output=True, text=True
                        )

                    # Display output
                    st.text_area("Script Output", process.stdout)

                    if process.returncode != 0:
                        st.error(f"Error: {process.stderr}")
                    else:
                        st.success("Scan completed successfully!")
                        # make sure this convention matches the m_scan_mirrors.py script
                        st.session_state.moveImPup["img_json_file"] = (
                            data_path_move_ind_mirror
                            + f"img_dict_beam{beam}-{motor}.json"
                        )

                        # if os.path.exists(figure_path):
                        #     image = Image.open(figure_path + 'scanMirror_result.png')
                        #     st.image(image, caption="Scan Results", use_column_width=True)
                        # else:
                        #     st.warning("Figure not found. Ensure the script generates the file correctly.")

            # Stop Scan Button
            if st.button("Stop Scan"):
                st.session_state.scan_running = False

                # this could cause bugs between move pupil
                # # moving back to original position
                # for axis, pos in st.session_state.moveImPup["original_positions"].items():
                #     msg = f"moveabs {axis} {pos}"

                #     send_and_get_response(msg)
                #     st.write(f"Moving {axis} back to {pos}")

            st.title("Combination of Mirrors")

            col1, col2 = st.columns(2)
            with col1:
                move_what = st.selectbox(
                    "Pick operating_mode",
                    ["move_image", "move_pupil"],
                    key="move_what",
                )

            with col2:
                config = st.selectbox(
                    "Pick a config",
                    ["c_red_one_focus", "intermediate_focus", "baldr"],
                    key="config",
                )

            if move_what == "move_image":
                units = "pixels"
            else:
                units = "mm"

            # Update motor axes
            if config == "baldr":
                axes = [f"BTP{beam}", f"BTT{beam}", f"BOTP{beam}", f"BOTT{beam}"]
            else:
                axes = [f"HTPP{beam}", f"HTTP{beam}", f"HTPI{beam}", f"HTTI{beam}"]

            if (config == "baldr") and (beam == 1):
                st.warning(
                    "warning no BOTX motor on beam 1 - so move pupil / image for baldr on beam 1 is invalid"
                )

            # Detect changes in beam or config, update original_positions if changed!
            if (beam != st.session_state.moveImPup["prev_beam"]) or (
                config != st.session_state.moveImPup["prev_config"]
            ):
                st.write(
                    "Updating original positions due to change in beam or config..."
                )

                # Update stored previous values
                # st.session_state.moveImPup["prev_beam"] = beam # < - this gets done with updating pupil coords
                st.session_state.moveImPup["prev_config"] = config

                # Update original_positions
                if config == "baldr":
                    axes = [f"BTP{beam}", f"BTT{beam}", f"BOTP{beam}", f"BOTT{beam}"]
                else:
                    axes = [f"HTPP{beam}", f"HTTP{beam}", f"HTPI{beam}", f"HTTI{beam}"]

                pos_dict = {}
                for axis in axes:
                    pos = send_and_get_response(f"read {axis}")
                    pos_dict[axis] = pos

                st.session_state.moveImPup["original_positions"] = pos_dict.copy()

            # apply a scan (cross, raster, square spiral options)

            # starting point always 0,0 since these are relative offsets for move pupil/image modes !!
            if scantype == "square_spiral":
                scan_pattern = pct.square_spiral_scan(
                    starting_point=[0, 0],
                    step_size=float(dx),
                    search_radius=float(search_radius),
                )
            elif scantype == "raster":
                scan_pattern = pct.raster_scan_with_orientation(
                    starting_point=[0, 0],
                    dx=float(dx),
                    dy=float(dx),
                    width=float(search_radius),
                    height=float(search_radius),
                    orientation=0,
                )
            elif scantype == "cross":  # tested in software, not hardware (20/6/25)
                scan_pattern = pct.cross_scan(
                    starting_point=[0, 0],
                    dx=float(dx),
                    dy=float(dx),
                    width=2 * float(search_radius),
                    height=2 * float(search_radius),
                    angle=0,
                )
            #### FROM HERE WE SHOULD PUT THIS IN A SCRIPT THAT IS RUN HERE!
            # ------------------------------------------
            # where we save output images to
            if st.button("Run Scan", key="run_combined_scan"):
                data_path = f"/home/asg/Progs/repos/asgard-alignment/calibration/reports/scan_{config}_{move_what}/"
                if not os.path.exists(data_path):
                    print(f"made directory : {data_path}")
                    os.makedirs(data_path)

                x_points, y_points = zip(*scan_pattern)

                # convert to relative offsets
                rel_x_points = np.array(list(x_points)[1:]) - np.array(
                    list(x_points)[:-1]
                )
                rel_y_points = np.array(list(y_points)[1:]) - np.array(
                    list(y_points)[:-1]
                )

                img_dict = {}

                motor_pos_dict = {}

                #############
                ## SET UP CAMERA to cropped beam region
                c = FLI.fli(roi=roi)

                # try get a dark
                try:
                    c.build_manual_dark()
                except Exception as e:
                    st.write(f"failed to take dark with exception {e}")

                progress_bar = st.progress(0)
                for it, (delx, dely) in enumerate(zip(rel_x_points, rel_y_points)):
                    progress_bar.progress(it / len(rel_x_points))

                    if move_what == "move_image":
                        # asgard_alignment.Engineering.move_image(
                        #     beam, delx, dely, send_and_get_response, config
                        # )
                        cmd = f"mv_img {config} {beam} {delx} {dely}"
                        send_and_get_response(cmd)
                    elif move_what == "move_pupil":
                        # asgard_alignment.Engineering.move_pupil(
                        #     beam, delx, dely, send_and_get_response, config
                        # )
                        cmd = f"mv_pup {config} {beam} {delx} {dely}"
                        send_and_get_response(cmd)

                    time.sleep(1)

                    # get all the motor positions
                    pos_dict = {}
                    for axis in axes:
                        pos = send_and_get_response(f"read {axis}")
                        # st.write(f"{axis}: {pos}")
                        pos_dict[axis] = pos

                    motor_pos_dict[str((x_points[it], y_points[it]))] = pos_dict

                    # get the images
                    # index dictionary by absolute position of the scan
                    imgtmp = np.mean(c.get_data(apply_manual_reduction=True), axis=0)
                    img_dict[str((x_points[it], y_points[it]))] = imgtmp

                # move back to original position
                st.write("moving back to original position")

                for axis, pos in st.session_state.moveImPup[
                    "original_positions"
                ].items():
                    msg = f"moveabs {axis} {pos}"

                    send_and_get_response(msg)
                    st.write(f"Moving {axis} back to {pos}")

                # save
                img_json_file_path = data_path + f"img_dict_beam{beam}-{move_what}.json"
                with open(img_json_file_path, "w") as json_file:
                    json.dump(util.convert_to_serializable(img_dict), json_file)

                st.write(f"wrote {img_json_file_path}")
                st.session_state.moveImPup["img_json_file"] = img_json_file_path

                motorpos_json_file_path = (
                    data_path + f"motorpos_dict_beam{beam}-{move_what}.json"
                )
                with open(motorpos_json_file_path, "w") as json_file:
                    json.dump(util.convert_to_serializable(motor_pos_dict), json_file)

                st.write(f"wrote {motorpos_json_file_path}")

            st.title("Frame Aggregate Analysis")

            # File uploader for JSON file (if you have)
            st.write(
                "if you don't want to scan now you can also upload a previously scanned json file to analyse."
            )
            uploaded_file = st.file_uploader("Upload JSON File", type="json")

            if uploaded_file is not None:
                if st.button("Load JSON File"):
                    # Read and parse JSON
                    file_contents = json.load(uploaded_file)

                    # Store in session state
                    st.session_state.moveImPup["img_json_file"] = file_contents
                    st.success("JSON file loaded successfully!")

            filter_central_pixels = st.checkbox(
                "only aggregate on pupil registered pixels?"
            )

            func_list = [np.nanmean, np.median, np.nanstd]
            func_label = ["mean", "median", "std"]

            boundary_threshold = st.text_input(
                "inside boundary threshold (to help calculate weighted center of signal)",
                0,
            )

            for fu, fla in zip(func_list, func_label):
                if st.button(f"plot {fla} signal vs coord"):

                    if st.session_state.moveImPup["img_json_file"] is None:
                        st.write(
                            "No scan dictionary has been written to a json file. Complete a scan first"
                        )
                        # we could allow user to read one in here manually (select)
                    else:
                        with open(
                            st.session_state.moveImPup["img_json_file"], "r"
                        ) as file:
                            data_dict = json.load(
                                file
                            )  # Parses the JSON content into a Python dictionary

                        data_dict_ed = {
                            tuple(map(float, key.strip("()").split(","))): value
                            for key, value in data_dict.items()
                        }

                        x_points = np.array([float(x) for x, _ in data_dict_ed.keys()])
                        y_points = np.array([float(y) for _, y in data_dict_ed.keys()])

                        sss = 200  # point size in scatter

                        # use first frame as reference
                        frame0 = np.array(list(data_dict.values()))[0]

                        if not filter_central_pixels:
                            # tmpmask = np.ones_like( frame0 ).astype(bool) # we dont filter for any particular pixels
                            user_sig = [fu(np.array(i)) for i in data_dict.values()]
                        else:
                            user_sig = []
                            for i in data_dict.values():

                                _, _, _, _, _, tmpmask = util.detect_pupil(
                                    i, sigma=2, threshold=0.5, plot=False, savepath=None
                                )  # pct.detect_circle
                                user_sig.append(fu(np.array(i)[tmpmask]))

                        user_sig = np.array(user_sig)

                        # inside_mask = np.ones_like( user_sig ).astype(bool)

                        try:
                            boundary_threshold = float(boundary_threshold)
                        except:
                            st.write(
                                f"boundary_threshold={boundary_threshold} cannot be converted to float. Using boundary_threshold=0"
                            )

                        inside_mask = user_sig > boundary_threshold

                        # Get x, y coordinates where inside_mask is True
                        x_inside = x_points[inside_mask]
                        y_inside = y_points[inside_mask]
                        weights = user_sig[
                            inside_mask
                        ]  # Use mean signal values as weights

                        # Compute weighted mean
                        x_c = np.sum(x_inside * weights) / np.sum(weights)
                        y_c = np.sum(y_inside * weights) / np.sum(weights)

                        # st.write(f"initial position {initial_Xpos},{initial_Ypos}")

                        st.write(f"(signal = {fla}) Weighted Center: ({x_c}, {y_c})")

                        if st.button(
                            f"move motors to this (relative) center at ({x_c}, {y_c})"
                        ):
                            st.write("implement")

                        fig, ax = plt.subplots(figsize=(6, 5))
                        scatter = ax.scatter(
                            x_points,
                            y_points,
                            c=user_sig,
                            s=sss,
                            cmap="viridis",
                            edgecolors="black",
                            label="Data Points",
                        )
                        plt.colorbar(scatter, label=f"frame {fla}")
                        ax.scatter(
                            [x_c], [y_c], color="r", marker="x", label="Weighted Center"
                        )
                        ax.legend()
                        st.pyplot(fig)
                        plt.close("all")

            # cluster analysis
            st.title("Frame Cluster Analysis")
            number_clusters = st.text_input("number of clusters ", 3)

            if st.button("cluster analysis"):
                with open(st.session_state.moveImPup["img_json_file"], "r") as file:
                    data_dict = json.load(
                        file
                    )  # Parses the JSON content into a Python dictionary

                data_dict_ed = {
                    tuple(map(float, key.strip("()").split(","))): value
                    for key, value in data_dict.items()
                }

                x_points = np.array([float(x) for x, _ in data_dict_ed.keys()])
                y_points = np.array([float(y) for _, y in data_dict_ed.keys()])

                image_list = np.array(list(data_dict.values()))
                res = pct.cluster_analysis_on_searched_images(
                    images=image_list,
                    detect_circle_function=pct.detect_circle,
                    n_clusters=int(number_clusters),
                    plot_clusters=False,
                )

                fig, ax = pct.plot_cluster_heatmap(x_points, y_points, res["clusters"])
                # plt.savefig(args.data_path + f'cluster_search_heatmap_beam{args.beam}.png')
                # plt.close()
                st.pyplot(fig)
                plt.close("all")
                fig, ax = pct.plot_aggregate_cluster_images(
                    images=image_list, clusters=res["clusters"], operation="mean"
                )  # std")
                # plt.savefig(args.data_path + f'clusters_heatmap_beam{args.beam}.png')
                st.pyplot(fig)
                plt.close("all")
        if routine_options == "Load state":
            # text box and reading of the json
            text_col, button_col = st.columns(2)

            with text_col:
                st.subheader("Load from mimir instr states")

                pth = os.path.expanduser("~/Progs/repos/asgard-alignment/instr_states/")
                # Get all files (not directories) in the path
                all_files = [
                    f for f in glob.glob(os.path.join(pth, "*")) if os.path.isfile(f)
                ]
                # Sort files by modification time, newest first
                all_files.sort(key=os.path.getmtime, reverse=True)

                # dropdown box
                suffixes = [x.split("/")[-1] for x in all_files]

                selected_file = st.selectbox("Select a file:", suffixes)

            with button_col:
                if st.button("Load"):
                    if selected_file is not None:

                        full_file = os.path.join(pth, selected_file)
                        with open(full_file) as f:
                            states = json.load(f)

                        for state in states:
                            if state["is_connected"]:
                                message = f"moveabs {state['name']} {state['position']}"
                                send_and_get_response(message)

        if routine_options == "See All States":

            # Mapping dictionary for the motor data on the image
            position_map = {
                "SDLA": (50, 20),
                "SDL12": (150, 20),
                "SDL34": (250, 20),
                "SSS": (350, 20),
                "SSF": (450, 20),
                "HFO1": (50, 50),
                "HFO2": (150, 50),
                "HFO3": (250, 50),
                "HFO4": (350, 50),
                "HTPP1": (50, 80),
                "HTPP2": (150, 80),
                "HTPP3": (250, 80),
                "HTPP4": (350, 80),
                "HTPI1": (50, 110),
                "HTPI2": (150, 110),
                "HTPI3": (250, 110),
                "HTPI4": (350, 110),
                "HTTP1": (50, 140),
                "HTTP2": (150, 140),
                "HTTP3": (250, 140),
                "HTTP4": (350, 140),
                "HTTI1": (50, 170),
                "HTTI2": (150, 170),
                "HTTI3": (250, 170),
                "HTTI4": (350, 170),
                "BFO": (1200, 200),
                "BOTT2": (150, 200),
                "BOTT3": (250, 200),
                "BOTT4": (350, 200),
                "BOTP2": (50, 230),
                "BOTP3": (150, 230),
                "BOTP4": (250, 230),
                "BDS1": (350, 230),
                "BDS2": (50, 260),
                "BDS3": (150, 260),
                "BDS4": (250, 260),
                "BTT1": (350, 260),
                "BTT2": (50, 290),
                "BTT3": (150, 290),
                "BTT4": (250, 290),
                "BTP1": (350, 290),
                "BTP2": (50, 320),
                "BTP3": (150, 320),
                "BTP4": (250, 320),
                "BMX1": (1100, 500),
                "BMX2": (1000, 400),
                "BMX3": (900, 300),
                "BMX4": (800, 200),
                "BMY1": (350, 350),
                "BMY2": (50, 380),
                "BMY3": (150, 380),
                "BMY4": (250, 380),
                "BLF1": (350, 380),
                "BLF2": (50, 410),
                "BLF3": (150, 410),
                "BLF4": (250, 410),
            }

            # Load the images from local files.
            try:
                im_top = Image.open("figs/BaldrHeim_optics.png")
                # im_bottom = Image.open("figs/asgard_lower_top.png")
            except Exception as e:
                st.error(f"Error loading images: {e}")

            if st.button("Update Annotations"):
                draw = ImageDraw.Draw(im_top)
                font = ImageFont.truetype(
                    "/home/asg/fonts/montserrat/Montserrat-MediumItalic.otf", size=18
                )

                motor_names = []
                instr = "All"
                if instr == "Solarstein" or instr == "All":
                    motor_names += ["SDLA", "SDL12", "SDL34", "SSS", "SSF"]
                if instr == "Heimdallr" or instr == "All":
                    motor_names_all_beams = [
                        "HFO",
                        "HTPP",
                        "HTPI",
                        "HTTP",
                        "HTTI",
                    ]

                    for motor in motor_names_all_beams:
                        for beam_number in range(1, 5):
                            motor_names.append(f"{motor}{beam_number}")
                if instr == "Baldr" or instr == "All":
                    motor_names += ["BFO"]

                    motor_names_all_beams = [
                        "BDS",
                        "BTT",
                        "BTP",
                        "BMX",
                        "BMY",
                        "BLF",
                    ]

                    partially_common_motors = [
                        "BOTT",
                        "BOTP",
                    ]

                    for motor in partially_common_motors:
                        for beam_number in range(2, 5):
                            motor_names.append(f"{motor}{beam_number}")

                    for motor in motor_names_all_beams:
                        for beam_number in range(1, 5):
                            motor_names.append(f"{motor}{beam_number}")

                states = []
                for name in motor_names:
                    message = f"read {name}"
                    res = send_and_get_response(message)

                    if "NACK" in res:
                        is_connected = False
                    else:
                        is_connected = True

                    state = {
                        "name": name,
                        "is_connected": is_connected,
                    }
                    print(res, type(res), is_connected)
                    if is_connected:
                        if res != "None":
                            state["position"] = float(res)
                        print()
                    states.append(state)

                for motor in states:
                    # st.write( motor.values() )
                    name = motor["name"]
                    pos_value = motor.get("position", -999)
                    is_connected = motor.get("is_connected", False)
                    # Look up the motor's image position from the mapping dictionary.
                    if name in position_map:
                        x, y = position_map[name]
                        status = "Connected" if is_connected else "Disconnected"
                        # Create annotation text (showing the position value and connection status).
                        text = f"{name}:{pos_value:.3f}"  # ({status})"
                        if status == "Connected":
                            color = "green"
                        elif status == "Disconnected":
                            color = "red"

                        # text_width, text_height = draw.textsize(text, font=font)
                        text_width, text_height = 130, 30  # font.getsize(text)
                        draw.rectangle(
                            (x, y, x + text_width, y + text_height), fill="white"
                        )
                        draw.text((x, y), text, fill=color, font=font)

                    else:
                        st.warning(f"No position mapping found for motor: {name}")

                st.image(
                    im_top,
                    caption="HEIMDALLR & BALDR IN ALL ITS GLORY",
                    use_column_width=True,
                )
                # st.image( im_bottom )

        if routine_options == "Health":
            message = "health"
            res = send_and_get_response(message)

            if st.button("Refresh"):
                res = send_and_get_response(message)

            # st.write(res)
            # convert to list of dicts
            data = json.loads(res)

            column_names = [
                "Axis name",
                "Motor type",
                "Controller connected?",
                "State",
                "Connect?",
                "Reset",
            ]
            keys = [
                "axis",
                "motor_type",
                "is_connected",
                "state",
                "connect_button",
                "reset_button",
            ]
            col_widths = [1, 1, 1, 5, 1, 1]

            st.write("Health of all motors")

            n_motors = len(data)

            rows = [st.columns(col_widths) for _ in range(n_motors + 1)]

            # first row is titles
            for i, col in enumerate(rows[0]):
                col.write(column_names[i])

            for i, row in enumerate(rows[1:]):
                for j, col in enumerate(row):
                    if keys[j] == "is_connected":
                        if data[i][keys[j]]:
                            col.success("True")
                        else:
                            col.warning("False")
                    elif keys[j] == "state":
                        if data[i][keys[j]] is not None:
                            if ("READY" in data[i][keys[j]]) or (
                                "No error" in data[i][keys[j]]
                            ):
                                col.success(data[i][keys[j]])
                            else:
                                col.error(data[i][keys[j]])
                    elif keys[j] == "connect_button":
                        if data[i]["is_connected"]:
                            pass
                        else:
                            if col.button("Connect", key=f"connect_{i}"):
                                message = f"connect {data[i]['axis']}"
                                send_and_get_response(message)
                    elif keys[j] == "reset_button":
                        if data[i]["motor_type"] == "LS16P":
                            if col.button("Reset", key=f"reset_{i}"):
                                message = f"reset {data[i]['axis']}"
                                send_and_get_response(message)
                    else:
                        col.write(data[i][keys[j]])
