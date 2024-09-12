import sys 
import glob
import numpy as np 
import pandas as pd
sys.path.insert(1,'/opt/Boston Micromachines/lib/Python3/site-packages/')
import bmc
sys.path.append('pyBaldr/' ) 

from pyBaldr import utilities as util


DM_serial_number = '17DW019#122'# Syd = '17DW019#122', ANU = '17DW019#053'

dm = bmc.BmcDm()
dm_err_flag  = dm.open_dm(DM_serial_number)
DMshapes_path='DMShapes/'
files = glob.glob(DMshapes_path+'*.csv')

"""print(files)
 'DMShapes/17DW019#122_FLAT_MAP_COMMANDS.csv',
 'DMShapes/four_torres.csv',
 'DMShapes/four_torres_2.csv',
 'DMShapes/Crosshair140.csv',
 'DMShapes/waffle.csv'
"""
flatdm = pd.read_csv('DMShapes/17DW019#122_FLAT_MAP_COMMANDS.csv', header=None)[0].values
crosshair = pd.read_csv('DMShapes/Crosshair140.csv', header=None)[0].values
fourTorres = pd.read_csv('DMShapes/four_torres.csv', header=None)[0].values
fourTorres2 = pd.read_csv('DMShapes/four_torres_2.csv', header=None)[0].values


dm.send_data(flatdm)

dm.send_data(flatdm + 0.1*crosshair)

b = util.construct_command_basis( basis='fourier_pinned_edges', number_of_modes = 40, Nx_act_DM = 12, Nx_act_basis = 12, act_offset=(0,0), without_piston=True)

