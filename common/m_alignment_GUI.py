# import src.Instrument
import streamlit as st

import asgard_alignment.GUI


st.set_page_config(layout="wide")

st.title("Motor control for Heimdallr alignment")

if "x" not in st.session_state:
    st.session_state.x = 0.0
    st.session_state.y = 0.0
    st.session_state.z = 0.0


def hello():
    print(f"Updating {st.session_state.x, st.session_state.y, st.session_state.z}")


col1, col2 = st.columns(2)

with col1:
    component = st.selectbox(
        "Pick a component",
        ["HFO", "HTXP", "HTXI", "HTXP", "BTX"],
        key="component",
    )

with col2:
    # beam = st.selectbox("Pick a component", list(range(1, 5)), key="beam")
    beam = st.selectbox("Pick a component", list(range(1, 3)), key="beam")


asgard_alignment.GUI.CustomNumeric.variable_increment(
    keys=["x", "y", "z"],
    callback_fns=[hello, hello, hello],
    values=[st.session_state.x, st.session_state.y, st.session_state.z],
    main_bounds=[-1.0, 1.0],
)
