# import src.Instrument
import streamlit as st

import asgard_alignment.GUI
import asgard_alignment.Instrument


# st.set_page_config(layout="wide")

config = "motor_info_sydney_subset.json"

st.title("Motor control for Heimdallr alignment")

if "instrument" not in st.session_state:
    st.session_state.instrument = asgard_alignment.Instrument.Instrument(
        "motor_info_no_linear_with_zaber.json"
    )

# reset button
if st.button("Reset"):
    st.session_state.instrument.close_connections()
    st.session_state.instrument = asgard_alignment.Instrument.Instrument(
        "motor_info_no_linear_with_zaber.json"
    )


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


if st.session_state.instrument.has_motor(name):
    st.session_state.instrument[name].GUI()
else:
    st.write("No motor found for this component")
