from zaber_motion.ascii import Connection

from asgard_alignment.ZaberMotor import BaldrPhaseMask, LAC10AT4A

from pyBaldr import utilities as util

con = Connection.open_tcp("192.168.1.111")

print("Found {} devices".format(len(con.detect_devices())))

x_axis = con.get_device(1).get_axis(1)
y_axis = con.get_device(1).get_axis(3)

baldr = BaldrPhaseMask(
    LAC10AT4A(x_axis), LAC10AT4A(y_axis), "phase_positions_beam_3.json"
)

print(baldr.get_position())

baldr.move_relative([0.1, 0.1])
print(baldr.get_position())

# example of how to control focus too:

focus_axis = con.get_device(1).get_axis(2)
focus_motor = LAC10AT4A(focus_axis)

print(focus_motor.get_position())

con.close()
