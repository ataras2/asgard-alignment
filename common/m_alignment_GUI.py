import src.GUI
import streamlit as st


st.set_page_config(layout="wide")

st.title("Motor control for Heimdallr alignment")

if "x" not in st.session_state:
    st.session_state.x = 0.0
    st.session_state.y = 0.0
    st.session_state.z = 0.0


def hello():
    print("hello")


src.GUI.CustomNumeric.variable_increment(
    keys=["x", "y", "z"],
    callback_fns=[hello, hello, hello],
    values=[st.session_state.x, st.session_state.y, st.session_state.z],
    main_bounds=[-1.0, 1.0],
)
