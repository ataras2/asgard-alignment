import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sys
import os

# Import bmc (Deformable Mirror controller package)
sys.path.insert(1, '/opt/Boston Micromachines/lib/Python3/site-packages/')
import bmc

# Load predefined shapes and DM serial numbers
DM_serial_number_dict = {'1':'17DW019#122', '2': '17DW019#122', '3': '17DW019#122', '4':'17DW019#122'}
DMshapes_path = 'DMShapes/'

crosshair = pd.read_csv(DMshapes_path + 'Crosshair140.csv', header=None)[0].values
fourTorres = pd.read_csv(DMshapes_path + 'four_torres.csv', header=None)[0].values

def get_DM_command_in_2D(cmd, Nx_act=12):
    corner_indices = [0, Nx_act-1, Nx_act * (Nx_act-1), Nx_act*Nx_act]
    cmd_in_2D = list(cmd.copy())
    for i in corner_indices:
        cmd_in_2D.insert(i, np.nan)
    return np.array(cmd_in_2D).reshape(Nx_act, Nx_act)

def main(beam, shape, strength, plot_shape=False,SIMULATION = True):
    
    if SIMULATION:
        dm = {}
        dm_err_flag = 0
    else:
        dm = bmc.BmcDm()
        dm_err_flag = dm.open_dm(DM_serial_number_dict[beam])

    flatdm = pd.read_csv(DMshapes_path + '{}_FLAT_MAP_COMMANDS.csv'.format(DM_serial_number_dict[beam]), header=None)[0].values

    available_shapes = {'flat': flatdm, 'crosshair': crosshair, 'four_torres': fourTorres}

    if dm_err_flag != 0:
        st.error(f"Error opening DM: {dm_err_flag}")
        return

    if shape not in available_shapes:
        st.error(f"Shape '{shape}' not recognized.")
        return

    selected_shape = available_shapes[shape]
    flat_shape = available_shapes['flat']
    if shape == 'flat':
        dm_command = flat_shape
    else:
        dm_command = flat_shape + strength * selected_shape

    if not SIMULATION:
        dm.send_data(dm_command)

    st.success(f"Applied {shape} shape with strength {strength} on beam {beam}")

    if plot_shape:
        st.write('Plotting...')
        fig, ax = plt.subplots()
        im = ax.imshow(get_DM_command_in_2D(dm_command))
        plt.colorbar(im, ax=ax, label='DM command')
        st.pyplot(fig)

# Streamlit UI
st.title("DM Shape Application")
simulation = st.selectbox("Simulation Mode", options=list([True, False]), index=0)
beam = st.selectbox("Select Beam", options=list(DM_serial_number_dict.keys()), index=0)
shape = st.selectbox("Select Shape", options=['flat', 'crosshair', 'four_torres'])
strength = st.slider("Strength", min_value=0.0, max_value=1.0, value=0.1, step=0.01)
plot_shape = st.checkbox("Plot Shape",value=True)

if st.button("Apply Shape"):
    main(beam, shape, strength, plot_shape, SIMULATION = simulation)
