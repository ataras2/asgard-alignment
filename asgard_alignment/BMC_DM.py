import sys
sys.path.insert(1,'/opt/Boston Micromachines/lib/Python3/site-packages/')
import bmc

DM_serial_number = '17DW019#122' # Syd = '17DW019#122', ANU = '17DW019#053'
dm = bmc.BmcDm() # init DM object
dm_err_flag  = dm.open_dm(DM_serial_number) # open DM