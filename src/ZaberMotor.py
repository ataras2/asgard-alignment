"""
A module for controlling the Zaber motors: LAC10A-T4A (through a X-MCC), X-LSM and X-LHM

Need to come up with a way to be able to name an axis/optic and move the right controller
Ideas:
- XMCC class with usage like XMCC[<axis number>].move_absolute(1000), + a dictionary that maps 
    the name of the optic to both the axis number and controller
"""

import zaber_motion

from zaber_motion.ascii import Connection
import time


class BifrostDichroic:
    def __init__(self, device) -> None:
        self.device = device
        self.axis = device.get_axis(1)
        self.positions = {
            "H": 132.32,
            "J": 62.32,
            "out": 0.0,
        }

        assert self.device.name == "X-LSM150A-SE03"

        if not self.axis.is_homed:
            self.axis.home(wait_until_idle=True)

        self.current_position = "out"
        self.set_position(self.current_position)

    def set_position(self, position):
        """Move the optic to the desired position"""

        if position not in self.positions:
            raise ValueError(f"Position {position} not in {self.positions.keys()}")

        self.axis.move_absolute(
            self.positions[position],
            unit=zaber_motion.Units.LENGTH_MILLIMETRES,
            wait_until_idle=False,
        )
        self.current_position = position

    def get_position(self):
        """Read the position from the device and check that it is consistent"""
        pos = self.axis.get_position(unit=zaber_motion.Units.LENGTH_MILLIMETRES)
        print(f"position: {pos:.3f}mm")
        for key, value in self.positions.items():
            if abs(pos - value) < 0.1:
                return key
        return "unknown"


if __name__ == "__main__":
    connection = Connection.open_serial_port("COM3")
    connection.enable_alerts()

    device_list = connection.detect_devices()
    print("Found {} devices".format(len(device_list)))

    dichroics = []
    for dev in device_list:
        if dev.name == "X-LSM150A-SE03":
            dichroics.append(BifrostDichroic(dev))
    print(f"Found {len(dichroics)} dichroics")

    for dichroic in dichroics:
        dichroic.set_position("J")

    while dichroics[0].get_position() != "J":
        pass

    time.sleep(0.5)
    for dichroic in dichroics:
        print(dichroic.get_position())

    for i in range(10):
        time.sleep(0.5)

        pos = dichroics[0].axis.get_position(unit=zaber_motion.Units.LENGTH_MILLIMETRES)
        print(f"position: {pos:.3f}mm")

    connection.close()
