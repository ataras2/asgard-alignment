import streamlit as st
import numpy as np
import argparse
import zmq

import asgard_alignment.Engineering

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
]

beam_common_devices = [
    "BFO",
    "SSS",
    "SDLA",
    "SDL12",
    "SDL34",
]

all_devices = beam_common_devices + beam_specific_devices

operating_mode = st.selectbox(
    "Select Operating Mode",
    ["Direct write", "Routines"],
    key="operating_mode",
)


def send_and_get_response(message):
    # st.write(f"Sending message to server: {message}")
    st.markdown(f":blue[Sending message to server: ] {message}")
    st.session_state["socket"].send_string(message)
    response = st.session_state["socket"].recv_string()
    if "NACK" in response:
        colour = "red"
    else:
        colour = "green"
    st.markdown(f":{colour}[Received response from server: ] {response}")


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
        target = f"{component}{beam_number}"
    else:
        beam_number = None
        target = component

    if component in ["BDS", "SSS"]:
        # linear stage interface
        st.subheader("Linear Stage Interface")

        if component == "BDS":
            valid_positions = ["H", "J", "empty"]
        elif component == "SSS":
            valid_positions = ["SRL", "SGL", "SLD/SSP", "SBB"]

        # add two buttons, one for homing and one for reading position
        s_col1, s_col2 = st.columns(2)

        with s_col1:
            if st.button("Home"):
                message = f"home {target}"
                send_and_get_response(message)
        with s_col2:
            if st.button("Read Position"):
                message = f"read_pos {target}"
                send_and_get_response(message)

        st.write("Preset positions selection")
        with st.form(key="valid_positions"):
            position = st.selectbox(
                "Select Position",
                valid_positions,
                key="position",
            )
            submit = st.form_submit_button("Move")

        if submit:
            message = f"move_preset {target} {position}"
            st.write(f"Sending message to server: {message}")
            send_and_get_response(message)

        # relative move option for input with button to move
        st.write("Relative Move")
        with st.form(key="relative_move"):
            relative_move = st.number_input(
                "Relative Move (mm)",
                min_value=-100.0,
                max_value=100.0,
                step=0.1,
                key="relative_move",
            )
            submit = st.form_submit_button("Move")

        if submit:
            message = f"move_rel {target} {relative_move}"
            st.write(f"Sending message to server: {message}")
            send_and_get_response(message)

    elif component in ["HTXP", "HTXI", "BTX"]:
        # TT motor interface
        # no homing, read position should be an option
        # also two fields for absolute value of each axis

        st.subheader("TT Motor Interface")

        # read position button
        if st.button("Read Position"):
            message = f"read_pos {target}"
            send_and_get_response(message)

        move_mode = st.selectbox(
            "Select Move Mode",
            ["Absolute Move", "Relative Move"],
            key="move_mode",
        )

        mode_to_command = {
            "Absolute Move": "move_abs",
            "Relative Move": "move_rel",
        }

        # absolute move option for input with button to move
        st.write(move_mode)
        with st.form(key="absolute_move"):
            s_col1, s_col2 = st.columns(2)
            with s_col1:
                u_position = st.number_input(
                    "U Position (mm)",
                    min_value=-0.750,
                    max_value=0.75,
                    step=0.05,
                    key="u_position",
                )
                submit = st.form_submit_button("Move U")

            if submit:
                message = f"{mode_to_command[move_mode]} {target} U {u_position}"
                send_and_get_response(message)

            with s_col2:
                v_position = st.number_input(
                    "V Position (mm)",
                    min_value=-0.750,
                    max_value=0.75,
                    step=0.05,
                    key="v_position",
                )
                submit2 = st.form_submit_button("Move V")

            if submit2:
                message = f"{mode_to_command[move_mode]} {target} V {v_position}"
                send_and_get_response(message)

    elif component in ["BFO", "SDLA", "SDL12", "SDL34", "HFO"]:
        # Linear actuator interface
        # all units in um

        if component == "HFO":
            bounds = (0.0, 16e3)
        else:
            bounds = (0.0, 10e3)

        st.subheader("Linear Actuator Interface")

        # read position button
        if st.button("Read Position"):
            message = f"read_pos {target}"
            send_and_get_response(message)

        move_mode = st.selectbox(
            "Select Move Mode",
            ["Absolute Move", "Relative Move"],
            key="move_mode",
        )

        mode_to_command = {
            "Absolute Move": "move_abs",
            "Relative Move": "move_rel",
        }

        mode_to_label = {
            "Absolute Move": "Position (um)",
            "Relative Move": "Step (um)",
        }

        mode_bounds = {
            "Absolute Move": bounds,
            "Relative Move": (-1e3, 1e3),
        }

        # absolute move option for input with button to move
        st.write(move_mode)
        with st.form(key="absolute_move"):
            position = st.number_input(
                mode_to_label[move_mode],
                min_value=mode_bounds[move_mode][0],
                max_value=mode_bounds[move_mode][1],
                step=100.0,
                key="position",
            )
            submit = st.form_submit_button("Move")

        if submit:
            message = f"{mode_to_command[move_mode]} {target} {position}"
            send_and_get_response(message)


elif operating_mode == "Routines":
    # move pupil and move image go here
    # also zero all (for alignment stuff)

    # zero_all command button
    if st.button("Zero All"):
        message = "zero_all"
        send_and_get_response(message)

    col1, col2 = st.columns(2)
    with col1:
        operating_mode = st.selectbox(
            "Pick operating_mode",
            ["move_image", "move_pupil"],
            key="operating_mode",
        )

    with col2:
        beam = st.selectbox(
            "Pick a beam",
            list(range(1, 5)),
            key="beam",
        )

    with st.form(key="amount"):
        amount = st.number_input("Amount", key="amount")
        submit = st.form_submit_button("Send command")

    if submit:
        if operating_mode == "move_image":
            asgard_alignment.Engineering.move_image(beam, amount, send_and_get_response)
        elif operating_mode == "move_pupil":
            asgard_alignment.Engineering.move_pupil(beam, amount, send_and_get_response)