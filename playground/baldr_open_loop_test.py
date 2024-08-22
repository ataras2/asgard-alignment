
from zaber_motion.ascii import Connection

from asgard_alignment.ZaberMotor import BaldrPhaseMask, LAC10AT4A

from pyBaldr import utilities as util

# use real template names
phasemask_pos = 'J3'
focus_pos = 4
dichroic = 
source = 'LLL'



# set up source 


# set up dichroic 

# set up phasemask
con = Connection.open_tcp("192.168.1.111")
print("Found {} devices".format(len(con.detect_devices())))
x_axis = con.get_device(1).get_axis(1)
y_axis = con.get_device(1).get_axis(3)

phasemask = BaldrPhaseMask(
    LAC10AT4A(x_axis), LAC10AT4A(y_axis), "phase_positions_beam_3.json"
)

phasemask.move_absolute(phasemask.phase_positions[phasemask_pos]) 

# set up focus 
focus_axis = con.get_device(1).get_axis(2)
focus_motor = LAC10AT4A(focus_axis)

print(phasemask.get_position())

print(phasemask.phase_positions) # 
"""{'J1': [3.02, 3.13],
 'J2': [3.2, 2.15],
 'J3': [3.38, 1.16],
 'J4': [3.56, 0.18],
 'J5': [2.82, 4.11],
 'H1': [8.09, 4.81],
 'H2': [8.27, 3.83],
 'H3': [8.45, 2.85],
 'H4': [8.63, 1.87],
 'H5': [8.82, 0.88]}
"""
phasemask.update_mask_position('J1', mask_name)

print(focus_motor.get_position() )



