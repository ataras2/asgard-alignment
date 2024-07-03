"""
A module for controlling the Zaber motors: LAC10A-T4A (through a X-MCC), X-LSM and X-LHM

Need to come up with a way to be able to name an axis/optic and move the right controller
Ideas:
- XMCC class with usage like XMCC[<axis number>].move_absolute(1000), + a dictionary that maps 
    the name of the optic to both the axis number and controller
"""

import zaber_motion

from zaber_motion.ascii import Connection

connection = Connection.open_serial_port("COM3")
connection.enable_alerts()

device_list = connection.detect_devices()
print("Found {} devices".format(len(device_list)))


for dev in device_list:
    print(
        f"Device {dev.device_id} with serial number {dev.serial_number} with {dev.axis_count} axes"
    )


connection.close()


class ZaberLinearStage:
    def __init__(self) -> None:
        pass


if __name__ == "__main__":
    pass
