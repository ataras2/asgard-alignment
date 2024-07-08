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
        self.dichroics = {
            "H": 132.32,
            "J": 62.32,
            "out": 0.0,
        }

        assert self.device.name == "X-LSM150A-SE03"

        if not self.axis.is_homed:
            self.axis.home(wait_until_idle=True)

        self.current_dichroic = "out"
        self.set_dichroic(self.current_dichroic)

    def set_dichroic(self, dichroic):
        """Move the optic to the desired position"""

        if dichroic not in self.dichroics:
            raise ValueError(f"Position {dichroic} not in {self.dichroics.keys()}")

        self.axis.move_absolute(
            self.dichroics[dichroic],
            unit=zaber_motion.Units.LENGTH_MILLIMETRES,
            wait_until_idle=False,
        )
        self.current_dichroic = dichroic

    def get_dichroic(self):
        """Read the position from the device and check that it is consistent"""
        pos = self.axis.get_position(unit=zaber_motion.Units.LENGTH_MILLIMETRES)
        for key, value in self.dichroics.items():
            if abs(pos - value) < 0.1:
                return key
        return "unknown"

    def GUI_section(self):
        pass


class SourceSelection:
    def __init__(self, device) -> None:
        self.device = device
        self.axis = device.get_axis(1)
        self.sources = {
            "SRL": 11.018,
            "SGL": 38.018,
            "SBB": 65.018,
            "SLD": 92.018,
            "none": 0.0,
        }

        assert self.device.name == "X-LHM100A-SE03"

        if not self.axis.is_homed:
            self.axis.home(wait_until_idle=True)

        self.current_source = "none"
        self.set_source(self.current_source)

    def set_source(self, source):
        """Move the optic to the desired position"""
        if source not in self.sources:
            raise ValueError(f"Position {source} not in {self.sources.keys()}")

        self.axis.move_absolute(
            self.sources[source],
            unit=zaber_motion.Units.LENGTH_MILLIMETRES,
            wait_until_idle=True,
        )
        self.current_position = source

    def get_source(self):
        """Read the position from the device and check that it is consistent"""
        pos = self.axis.get_position(unit=zaber_motion.Units.LENGTH_MILLIMETRES)
        for key, value in self.sources.items():
            if abs(pos - value) < 0.1:
                return key
        return "unknown"

    def GUI_section(self):
        pass


if __name__ == "__main__":
    connection = Connection.open_serial_port("COM3")
    connection.enable_alerts()

    device_list = connection.detect_devices()
    print("Found {} devices".format(len(device_list)))

    dichroics = []
    source_selection = None
    for dev in device_list:
        if dev.name == "X-LSM150A-SE03":
            dichroics.append(BifrostDichroic(dev))
        elif dev.name == "X-LHM100A-SE03":
            source_selection = SourceSelection(dev)
    print(f"Found {len(dichroics)} dichroics")
    if source_selection is not None:
        print("Found source selection")

    for dichroic in dichroics:
        dichroic.set_dichroic("J")

    while dichroics[0].get_dichroic() != "J":
        pass

    time.sleep(0.5)
    for dichroic in dichroics:
        print(dichroic.get_dichroic())

    for i in range(10):
        time.sleep(0.5)

        pos = dichroics[0].axis.get_position(unit=zaber_motion.Units.LENGTH_MILLIMETRES)
        print(f"position: {pos:.3f}mm")

    source_selection.set_source("SRL")

    while source_selection.get_source() != "SRL":
        pass

    time.sleep(0.5)
    print(source_selection.get_source())

    connection.close()
