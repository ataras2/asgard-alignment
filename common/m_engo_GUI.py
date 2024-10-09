import streamlit as st
import numpy as np
import zmq
import argparse

import asgard_alignment

parser = argparse.ArgumentParser(description="ZeroMQ GUI Client")
parser.add_argument("--host", type=str, default="localhost", help="Server host")
parser.add_argument("--port", type=int, default=5555, help="Server port")
parser.add_argument(
    "--timeout", type=int, default=5000, help="Response timeout in milliseconds"
)
args = parser.parse_args()

if "context" not in st.session_state:
    # Create a ZeroMQ context
    st.session_state["context"] = zmq.Context()

    # Create a socket to communicate with the server
    st.session_state["socket"] = st.session_state["context"].socket(zmq.REQ)

    # Set the receive timeout
    st.session_state["socket"].setsockopt(zmq.RCVTIMEO, args.timeout)

    # Connect to the server
    server_address = f"tcp://{args.host}:{args.port}"
    st.session_state["socket"].connect(server_address)

mode = st.selectbox(
    "Select mode",
    ["Direct_motor", "High_level"],
    key="mode",
)

if mode == "Direct_motor":
    col1, col2 = st.columns(2)
    with col1:
        component = st.selectbox(
            "Pick a component",
            ["HTXP", "HFO", "HTXI", "BTX", "BDS", "SSS"],
            key="component",
        )

    if st.session_state.component not in ["SSS"]:
        with col2:
            beam = st.selectbox(
                "Pick a beam",
                list(range(1, 5)),
                key="beam",
            )
        name = str(st.session_state.component) + str(st.session_state.beam)
    else:
        name = str(st.session_state.component)
elif mode == "High_level":
    col1, col2 = st.columns(2)
    with col1:
        component = st.selectbox(
            "Pick a component",
            ["move_image", "move_pupil"],
            key="component",
        )

    with col2:
        beam = st.selectbox(
            "Pick a beam",
            list(range(1, 5)),
            key="beam",
        )

    name = str(st.session_state.component) + str(st.session_state.beam)

    st.text_input("Amount", key="amount")

if st.button("Send command"):
    if mode == "Direct_motor":
        message = f"move_rel {name} {st.session_state.amount}"
    elif mode == "High_level":
        message = f"{st.session_state.component}({st.session_state.beam}, {st.session_state.amount})"

    st.session_state["socket"].send_string(message)

    try:
        # Wait for a response from the server
        response = st.session_state["socket"].recv_string()
        st.write(f"Received response from server: {response}")
    except zmq.Again as e:
        st.write(f"Timeout waiting for response from server: {e}")
