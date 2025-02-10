import streamlit as st
import numpy as np
import argparse
import zmq
import time
import json
import os
import glob
import sys
import matplotlib.pyplot as plt

import asgard_alignment.Engineering
import common.DM_basis_functions


try:
    from baldr import _baldr as ba
    from baldr import sardine as sa
except ImportError:
    print(f"current conda environment = {sys.prefix}")
    print("need base environment to use sardine shared memory. Try conda activate base")


# make GUI wide
st.set_page_config(layout="wide")

if "message_history" not in st.session_state:
    st.session_state["message_history"] = []

if "socket" not in st.session_state:
    parser = argparse.ArgumentParser(description="ZeroMQ Client")
    parser.add_argument("--host", type=str, default="localhost", help="Server host")
    parser.add_argument("--port", type=int, default=5555, help="Server port")
    parser.add_argument(
        "--timeout", type=int, default=5000, help="Response timeout in milliseconds"
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

st.title("Asgard alignment engineering GUI")


beam_specific_devices = [
    "HFO",
    "HTXP",
    "HTXI",
    "BTX",
    "BDS",
    "BMX",
    "BMY",
    "BOTX",
    "DM",
    "phasemask",
]

beam_common_devices = [
    "BFO",
    "SSS",
    "SDLA",
    "SDL12",
    "SDL34",
]

all_devices = beam_common_devices + beam_specific_devices


def send_and_get_response(message):
    # st.write(f"Sending message to server: {message}")
    st.session_state["message_history"].append(
        f":blue[Sending message to server: ] {message}\n"
    )
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

    beam = targets[0].split("phasemask")[1]

    st.session_state["selected_beam"] = beam

    # Initialize session state variables if they don't already exist
    if "selected_mask" not in st.session_state:
        st.session_state["selected_mask"] = "Unknown"

    if st.button("Read Position"):
        st.write(f"target:{targets}")
        # beam = targets[0].split("phasemask")[1]
        for target in [f"BMX{beam}", f"BMY{beam}"]:
            message = f"!read {target}"
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

        submit = st.form_submit_button("Move")

    if submit:

        message = f"!fpm_movetomask {targets[0]} {preset_position}"
        st.write(f"{message}")

        res = send_and_get_response(message)
        if "NACK" in res:
            st.write(f"Error moving to phasemask position with {res}")
            st.session_state["selected_mask"] = ["Unknown"]
        else:
            st.write(f"moved {targets[0]} to {preset_position}")
            st.session_state["selected_mask"] = [preset_position]

    st.write("Registered mask", st.session_state["selected_mask"][0])

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
            message = f"!moverel BMY{beam} {increment}"
            send_and_get_response(message)

    with lm:
        if st.button(f"-y: {increment:.2f}"):
            message = f"!moverel BMY{beam} {-increment}"
            send_and_get_response(message)
    with ml:
        if st.button(f"+x: {increment:.2f}"):
            message = f"!moverel BMX{beam} {increment}"
            send_and_get_response(message)
    with mr:
        if st.button(f"-x: {increment:.2f}"):
            message = f"!moverel BMX{beam} {-increment}"
            send_and_get_response(message)

    update_col, save_col = st.columns(2)

    with update_col:
        if st.button(
            f"Update Registered {st.session_state['selected_mask']} Mask Position (local - not saved)"
        ):
            if "unknown" not in st.session_state["selected_mask"][0].lower():
                # Update the current mask position
                message = f"!fpm_updatemaskpos {targets[0]} {st.session_state['selected_mask'][0]}"
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
            # f"/home/heimdallr/Documents/asgard-alignment/config_files/phasemask_positions/beam{beam}/*json"
            valid_reference_position_files = glob.glob(
                f"/home/heimdallr/Documents/asgard-alignment/config_files/phasemask_positions/beam{beam}/*json"
            )  # save_path + f"/beam{beam}/*json")

            selected_reference_file = st.selectbox(
                "Select Reference Position File to Calculate Relative Seperations Between Masks",
                valid_reference_position_files,
                key="selected_file",
            )
            submit_reference_file = st.form_submit_button(
                f"Update All Mask Positions Relative to Current registered {st.session_state['selected_mask']} Position (local - not saved)"
            )

            if submit_reference_file:
                if "unknown" not in st.session_state["selected_mask"][0].lower():

                    message = f"!fpm_updateallmaskpos {targets[0]} {st.session_state['selected_mask'][0]} {selected_reference_file}"

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

                # save_path = send_and_get_response(f"!fpm_getsavepath {targets[0]}")

                # Save the updated positions to file
                save_message = f"!fpm_writemaskpos {targets[0]}"
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

    # message = "!fpm_updateallmaskpos {} {} {}"


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

    ff = "/home/heimdallr/Documents/asgard-alignment/config_files/dm_shared_memory_config.json"
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
    #     #     message = f"!dmapplyflat {target}"
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
    #     #     message = f"!dmapplycross {target}"
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

    # set the DM to unity and then multiply by the combined mode.. We really need a set method!!!
    dm_shm *= 0.0  # zero all the DMs
    dm_shm += 1.0  # unity all the DMs
    dm_shm *= combined_mode  # add the modes

    ### Below is when we opened DM in the multi device server and communicated via ZMQ
    # Placeholder for sending the combined mode to the DM
    # response = send_and_get_response(f"!dmapply {combined_mode}")
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
        valid_pos = ["H", "J", "empty"]

        if f"BDS{beam_number}_fixed_mapping" not in st.session_state:
            st.session_state[f"BDS{beam_number}_fixed_mapping"] = {
                "H": 133.07,  # (white target)
                "J": 63.07,  # (mirror)
                "empty": 0.0,
            }
            st.session_state[f"BDS{beam_number}_offset"] = 0.0

    elif component == "SSS":
        valid_pos = ["SRL", "SGL", "SLD/SSP", "SBB"]
        if "SSS_fixed_mapping" not in st.session_state:
            st.session_state[f"SSS_fixed_mapping"] = {
                "SRL": 11.5,
                "SGL": 38.5,
                "SLD/SSP": 92.5,
                "SBB": 65.5,
            }
            st.session_state[f"SSS_offset"] = 0.0

    mapping = {
        k: v + st.session_state[f"{target}_offset"]
        for k, v in st.session_state[f"{target}_fixed_mapping"].items()
    }

    # add two buttons, one for homing and one for reading position
    s_col1, s_col2, s_col3 = st.columns(3)

    with s_col1:
        if st.button("Read Position"):
            message = f"!read {target}"
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
            message = f"!state {target}"
            res = send_and_get_response(message)
            st.write(res)

    with s_col3:
        if st.button("Home (if needed)"):
            message = f"!init {target}"
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
            message = f"!moveabs {target} {mapping[preset_position]}"
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
            message = f"!moverel {target} {relative_move}"
            send_and_get_response(message)

    # add a button to update the preset positions
    st.subheader("Updating positions")
    st.write(f"Current mapping is: {mapping}")
    button_cols = st.columns(3)
    with button_cols[0]:
        if st.button(f"Update only {preset_position}"):
            current_position = send_and_get_response(f"!read {target}")
            st.session_state[f"{target}_fixed_mapping"][preset_position] = float(
                current_position
            )
            st.rerun()
    with button_cols[1]:
        if st.button("Update all"):
            current_position = send_and_get_response(f"!read {target}")
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
            message = f"!read {target}"
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
        message = f"!read {target}"
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
        if use_button_to_move:
            delay_on_moves = st.checkbox("Delay on moves (recommended)", value=True)
        else:
            delay_on_moves = False

    # absolute move option for input with button to move
    st.write("Move absolute")
    s_col1, s_col2 = st.columns(2)
    if use_button_to_move:
        with s_col1:
            with st.form(key="absolute_move_u"):
                u_position = st.number_input(
                    "U Position (degrees)",
                    min_value=-0.750,
                    max_value=0.75,
                    step=inc,
                    value=positions[0],
                    format="%.4f",
                    key="u_position",
                )
                submit = st.form_submit_button("Move U")

            if submit:
                # replace the x in target with U
                target = f"{component}{beam_number}"
                target = target.replace("X", "P")
                message = f"!moveabs {target} {u_position}"
                send_and_get_response(message)

        with s_col2:
            with st.form(key="absolute_move_v"):
                v_position = st.number_input(
                    "V Position (degrees)",
                    min_value=-0.750,
                    max_value=0.75,
                    value=positions[1],
                    format="%.4f",
                    step=inc,
                    key="v_position",
                )
                submit2 = st.form_submit_button("Move V")

            if submit2:
                target = f"{component}{beam_number}"
                target = target.replace("X", "T")
                message = f"!moveabs {target} {v_position}"
                send_and_get_response(message)
    else:

        def get_onchange_fn(axis, key):
            def onchange_fn():
                target = f"{component}{beam_number}"
                target = target.replace("X", axis)
                message = f"!moveabs {target} {st.session_state[key]}"
                send_and_get_response(message)
                if delay_on_moves:
                    time.sleep(1.0)

            return onchange_fn

        sub_col1, sub_col2 = st.columns(2)

        with sub_col1:
            u_position = st.number_input(
                "U Position (degrees)",
                min_value=-0.750,
                max_value=0.75,
                step=inc,
                value=positions[0],
                format="%.4f",
                key="u_position",
                on_change=get_onchange_fn("P", "u_position"),
            )

        with sub_col2:
            v_position = st.number_input(
                "V Position (degrees)",
                min_value=-0.750,
                max_value=0.75,
                step=inc,
                value=positions[1],
                format="%.4f",
                key="v_position",
                on_change=get_onchange_fn("T", "v_position"),
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
            message = f"!read {target}"
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
            message = f"!state {target}"
            res = send_and_get_response(message)
            st.write(res)

    with c3:
        # init button
        if st.button("Home (if needed)"):
            message = f"!init {target}"
            send_and_get_response(message)

    def get_onchange_fn(key, target):
        def onchange_fn():
            if "HFO" in target:
                desired_position = st.session_state[key] * 1e-3
            else:
                desired_position = st.session_state[key]
            message = f"!moveabs {target} {desired_position}"
            send_and_get_response(message)
            time.sleep(1.0)

        return onchange_fn

    message = f"!read {target}"
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
        value=cur_pos,
        key="lin_position",
        on_change=get_onchange_fn("lin_position", target),
    )


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
                "Select Component",
                all_devices,
                key="component",
            )

        if component in beam_specific_devices:
            with col2:
                beam_number = st.selectbox(
                    "Select Beam Number",
                    [1, 2, 3, 4],
                    key="beam_number",
                )
            targets = [f"{component}{beam_number}"]

            if component in ["HTXP", "HTXI", "BTX", "BOTX"]:
                # replace the X in target with P
                target = f"{component}{beam_number}"
                targets = [target.replace("X", "P"), target.replace("X", "T")]
        else:
            beam_number = None
            targets = [component]

        # check if component is connected
        is_connected = all(
            send_and_get_response(f"!connected? {target}") == "connected"
            for target in targets
        )
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
                        message = f"!connect {target}"
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

    elif operating_mode == "Routines":
        # move pupil and move image go here
        # also zero all (for alignment stuff)

        routine_options = st.selectbox(
            "Select Routine",
            ["Quick buttons", "Move image/pupil", "Save state", "Load state", "Health"],
            key="routine_options",
        )

        if routine_options == "Quick buttons":
            # zero_all command button
            st.write("Nothing here (yet)")

        if routine_options == "Move image/pupil":
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
                    ["c_red_one_focus", "intermediate_focus"],
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

            # tickbox for button only mode
            button_only = st.checkbox("Use button to move", value=True)

            if not button_only:
                with st.form(key="amount"):
                    delx = st.number_input(f"delta x {units}", key="delx")
                    dely = st.number_input(f"delta y {units}", key="dely")
                    submit = st.form_submit_button("Send command")

                if submit:
                    if move_what == "move_image":
                        asgard_alignment.Engineering.move_image(
                            beam, delx, dely, send_and_get_response, config
                        )
                    elif move_what == "move_pupil":
                        asgard_alignment.Engineering.move_pupil(
                            beam, delx, dely, send_and_get_response, config
                        )
            else:
                # increment selection for each case
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
                    move_function = asgard_alignment.Engineering.move_image
                elif move_what == "move_pupil":
                    move_function = asgard_alignment.Engineering.move_pupil
                else:
                    raise ValueError("Invalid move_what")

                pos_x = lambda: move_function(
                    beam, increment, 0, send_and_get_response, config
                )
                pos_y = lambda: move_function(
                    beam, 0, increment, send_and_get_response, config
                )
                neg_x = lambda: move_function(
                    beam, -increment, 0, send_and_get_response, config
                )
                neg_y = lambda: move_function(
                    beam, 0, -increment, send_and_get_response, config
                )

                # make a 3x3 grid but only use the up, down, left and right

                ul, um, ur = st.columns(3)
                ml, mm, mr = st.columns(3)
                ll, lm, lr = st.columns(3)

                if move_what == "move_image":
                    with um:
                        if st.button(f"-y: {increment:.2f}"):
                            neg_y()
                    with lm:
                        if st.button(f"+y: {increment:.2f}"):
                            pos_y()
                    with ml:
                        if st.button(f"-x: {increment:.2f}"):
                            neg_x()
                    with mr:
                        if st.button(f"+x: {increment:.2f}"):
                            pos_x()
                elif move_what == "move_pupil":
                    with um:
                        if st.button(f"+y: {increment:.2f}"):
                            pos_y()
                    with lm:
                        if st.button(f"-y: {increment:.2f}"):
                            neg_y()
                    with ml:
                        if st.button(f"-x: {increment:.2f}"):
                            neg_x()
                    with mr:
                        if st.button(f"+x: {increment:.2f}"):
                            pos_x()

            # also show the state of all of the motors involved
            axes = [f"HTTP{beam}", f"HTIP{beam}", f"HTTI{beam}" f"HTTI{beam}"]

            for axis in axes:
                pos = send_and_get_response(f"!read {axis}")
                st.write(f"axis: {pos}")

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

        if routine_options == "Save state":

            instruments = ["Heimdallr", "Baldr", "Solarstein"]
            # grid of 3 rows, 2 cols, with first col being the save location
            # and second col being the save button
            for i in range(3):
                col1, col2 = st.columns(2)
                with col1:
                    save_location = st.text_input(
                        f"Save {instruments[i]}", key=f"save_location_{i}"
                    )
                with col2:
                    if st.button(f"Save {instruments[i]}"):
                        if instruments[i] == "Solarstein":
                            motor_names = ["SDLA", "SDL12", "SDL34", "SSS"]
                        elif instruments[i] == "Heimdallr":
                            motor_names_no_beams = [
                                "HFO",
                                "HTPP",
                                "HTPI",
                                "HTTP",
                                "HTTI",
                            ]

                            motor_names = []
                            for motor in motor_names_no_beams:
                                for beam_number in range(1, 5):
                                    motor_names.append(f"{motor}{beam_number}")
                        elif instruments[i] == "Baldr":
                            motor_names = ["BFO"]

                            motor_names_no_beams = [
                                "BDS",
                                "BTT",
                                "BTP",
                                "BMX",
                                "BMY",
                            ]

                            for motor in motor_names_no_beams:
                                for beam_number in range(1, 5):
                                    motor_names.append(f"{motor}{beam_number}")

                        states = []
                        for name in motor_names:
                            message = f"!read {name}"
                            res = send_and_get_response(message)

                            if "NACK" in res:
                                is_connected = False
                            else:
                                is_connected = True

                            state = {
                                "name": name,
                                "is_connected": is_connected,
                            }
                            if is_connected:
                                state["position"] = float(res)

                            states.append(state)

                        # save to json at location
                        with open("instr_states/" + save_location + ".json", "w") as f:
                            json.dump(states, f, indent=4)

        if routine_options == "Load state":
            # text box and reading of the json
            text_col, button_col = st.columns(2)

            with text_col:
                load_location = st.text_input("Load location", key="load_location")

            with button_col:
                if st.button("Load"):
                    with open("instr_states/" + load_location + ".json", "r") as f:
                        states = json.load(f)

                    for state in states:
                        if state["is_connected"]:
                            message = f"!moveabs {state['name']} {state['position']}"
                            send_and_get_response(message)

        if routine_options == "Health":
            message = "!health"
            res = send_and_get_response(message)

            if st.button("Refresh"):
                res = send_and_get_response(message)

            # convert to list of dicts
            data = json.loads(res)

            column_names = ["Axis name", "Motor type", "Controller connected?", "State"]
            keys = ["axis", "motor_type", "is_connected", "state"]

            st.write("Health of all motors")

            n_motors = len(data)

            rows = [st.columns(len(column_names)) for _ in range(n_motors + 1)]

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
                    else:
                        col.write(data[i][keys[j]])
