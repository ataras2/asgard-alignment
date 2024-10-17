"""
A module for controlling the Zaber motors: LAC10A-T4A (through a X-MCC), X-LSM and X-LHM

Need to come up with a way to be able to name an axis/optic and move the right controller
Ideas:
- XMCC class with usage like XMCC[<axis number>].move_absolute(1000), + a dictionary that maps 
    the name of the optic to both the axis number and controller
"""

import zaber_motion

import streamlit as st
from zaber_motion.ascii import Connection
import time
import json
import numpy as np

import zaber_motion.binary

from asgard_alignment.AsgardDevice import AsgardDevice


import asgard_alignment.ESOdevice as ESOdevice


class ZaberLinearActutator(ESOdevice.Motor):
    UPPER_LIMIT = 10_000  # um
    LOWER_LIMIT = 0  # um

    IS_BLOCKING = True

    def __init__(self, name, axis) -> None:
        super().__init__(name)
        self.axis = axis

        if not self.axis.is_homed:
            self.axis.home(wait_until_idle=ZaberLinearActutator.IS_BLOCKING)

    def move_absolute(self, new_pos, units=zaber_motion.Units.LENGTH_MICROMETRES):
        """
        Move the motor to the absolute position

        Parameters:
        -----------
        new_pos: float
            The position to move to

        units: zaber_motion.Units
            The units of the position, default is micrometres

        Returns:
        --------
        None
        """
        self.axis.move_absolute(
            new_pos,
            unit=units,
            wait_until_idle=ZaberLinearActutator.IS_BLOCKING,
        )

    def move_relative(self, new_pos, units=zaber_motion.Units.LENGTH_MICROMETRES):
        """
        Move the motor to the relative position

        Parameters:
        -----------
        new_pos: float
            The position to move to, relative to the current position

        units: zaber_motion.Units
            The units of the position, default is micrometres

        Returns:
        --------
        None
        """
        self.axis.move_relative(
            new_pos,
            unit=units,
            wait_until_idle=ZaberLinearActutator.IS_BLOCKING,
        )

    def read_position(self, units=zaber_motion.Units.LENGTH_MICROMETRES):
        return self.axis.read_position(unit=units)

    def is_at_limit(self):
        """
        Check if the motor is at the limit

        Returns:
        --------
        bool
            True if the motor is at the limit, False otherwise
        """
        # us np.isclose to avoid floating point errors
        return np.isclose(self.read_position(), self.UPPER_LIMIT) or np.isclose(
            self.read_position(), self.LOWER_LIMIT
        )

    def stop_now(self):
        """
        Stop the motor immediately

        Returns:
        --------
        None
        """
        self.axis.stop(wait_until_idle=ZaberLinearActutator.IS_BLOCKING)

    def init(self):
        """
        Don't do anything, the motor is already initialised by the constructor
        Might need to home after power cycle
        """
        if not self.axis.is_homed:
            self.axis.home(wait_until_idle=ZaberLinearActutator.IS_BLOCKING)

    def move_abs(self, position):
        """
        Move the motor to the absolute position

        Parameters:
        -----------
        position: float
            The position to move to

        Returns:
        --------
        None
        """
        self.move_absolute(position)

    def is_reset_success(self):
        """
        Check if the reset was successful

        Returns:
        --------
        bool
            True if the reset was successful, False otherwise
        """

        return True  # TODO: Implement?

    def is_stop_success(self):
        """
        Check if the stop was successful

        Returns:
        --------
        bool
            True if the stop was successful, False otherwise
        """

        return not self.axis.is_busy()

    def is_init_success(self):
        """
        Check if the initialisation was successful

        Returns:
        --------
        bool
            True if the initialisation was successful, False otherwise
        """
        return not self.axis.is_busy()

    def is_motion_done(self):
        """
        Check if the motion is done

        Returns:
        --------
        bool
            True if the motion is done, False otherwise
        """
        return not self.axis.is_busy()


class ZaberLinearStage(ESOdevice.Motor):
    """
    A linear stage, e.g. the X-LHM100A-SE03, or the X-LSM150A-SE03
    Default units are milimetres
    """

    def __init__(self, name, device):
        super().__init__(name)

        self.device = device


'''
class BifrostDichroic:
    def __init__(self, device) -> None:
        self.device = device
        self.axis = device.get_axis(1)
        self.dichroics = {
            "H": 133.07,  # 131.82,
            "J": 63.07,
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
            unit=zaber_motion.Units.LENGTH_MICROMETRES,
            wait_until_idle=False,
        )
        self.current_dichroic = dichroic

    def get_dichroic(self):
        """Read the position from the device and check that it is consistent"""
        pos = self.axis.get_position(unit=zaber_motion.Units.LENGTH_MICROMETRES)
        for key, value in self.dichroics.items():
            if abs(pos - value) < 0.1:
                return key
        return "unknown"

    def GUI(self):
        st.header("Baldr dichroic motor")

        st.write(f"Current position: {self.get_dichroic()}")

        # 3 buttons for each position
        for key in self.dichroics.keys():
            if st.button(key):
                self.set_dichroic(key)


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
            unit=zaber_motion.Units.LENGTH_MICROMETRES,
            wait_until_idle=True,
        )
        self.current_position = source

    def get_source(self):
        """Read the position from the device and check that it is consistent"""
        pos = self.axis.get_position(unit=zaber_motion.Units.LENGTH_MICROMETRES)
        for key, value in self.sources.items():
            if abs(pos - value) < 0.1:
                return key
        return "unknown"

    def GUI(self):
        st.header("Source selection")

        st.write(f"Current position: {self.get_source()}")

        # 4 buttons for each position
        for key in self.sources.keys():
            if st.button(key):
                self.set_source(key)


class SolarsteinDelay:
    pass


class BaldrCommonLens:
    pass


class LAC10AT4A:
    def __init__(self, axis) -> None:
        self.axis = axis

        if not self.axis.is_homed:
            self.axis.home(wait_until_idle=True)

    def move_absolute(self, new_pos, units=zaber_motion.Units.LENGTH_MICROMETRES):
        self.axis.move_absolute(new_pos, unit=units, wait_until_idle=True)

    def move_relative(self, new_pos, units=zaber_motion.Units.LENGTH_MICROMETRES):
        self.axis.move_relative(new_pos, unit=units, wait_until_idle=True)

    def get_position(self, units=zaber_motion.Units.LENGTH_MICROMETRES):
        return self.axis.get_position(unit=units)


class BaldrPhaseMask:
    """
    Key here is that this has 2x LAC10A and can control both at once
    """

    def __init__(self, x_axis_motor, y_axis_motor, phase_positions_json) -> None:
        self.motors = {
            "x": x_axis_motor,
            "y": y_axis_motor,
        }

        self.phase_positions = self._load_phase_positions(phase_positions_json)

    @staticmethod
    def _load_phase_positions(phase_positions_json):
        with open(phase_positions_json, "r", encoding="utf-8") as file:
            config = json.load(file)

        assert len(config) == 10, "There must be 10 phase mask positions"

        return config

    def move_relative(self, new_pos, units=zaber_motion.units.Units.LENGTH_MICROMETRES):
        self.motors["x"].move_relative(new_pos[0], units)
        self.motors["y"].move_relative(new_pos[1], units)

    def move_absolute(self, new_pos, units=zaber_motion.units.Units.LENGTH_MICROMETRES):
        self.motors["x"].move_absolute(new_pos[0], units)
        self.motors["y"].move_absolute(new_pos[1], units)

    def get_position(self, units=zaber_motion.units.Units.LENGTH_MICROMETRES):
        return [
            self.motors["x"].get_position(units),
            self.motors["y"].get_position(units),
        ]

    def move_to_mask(self, mask_name):
        self.move_absolute(self.phase_positions[mask_name])

    def update_mask_position(self, mask_name):
        self.phase_positions[mask_name] = self.get_position()


class ZaberLinearStage(AsgardDevice):
    def __init__(self, axis, units=zaber_motion.Units.LENGTH_MILLIMETRES) -> None:
        self.axis = axis
        self.units = units

    def initialise(self):
        if not self.axis.is_homed:
            self.axis.home(wait_until_idle=False)

    def set_position(self, position):
        self.axis.move_absolute(position, wait_until_idle=False, unit=self.units)

    def get_position(self):
        return self.axis.get_position(unit=self.units)


class ZaberLinearActuator:
    def __init__(self, axis, units=zaber_motion.Units.LENGTH_MICROMETRES) -> None:
        self.axis = axis
        self.units = units

    def initalise(self):
        if not self.axis.is_homed:
            self.axis.home(wait_until_idle=True)
        
    def set_position(self, position):
        self.axis.move_absolute(position, wait_until_idle=False, unit=self.units)

    def get_position(self):
        return self.axis.get_position(unit=self.units)


if __name__ == "__main__":

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

    exit()
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

        pos = dichroics[0].axis.get_position(unit=zaber_motion.Units.LENGTH_MICROMETRES)
        print(f"position: {pos:.3f}mm")

    source_selection.set_source("SRL")

    while source_selection.get_source() != "SRL":
        pass

    time.sleep(0.5)
    print(source_selection.get_source())

    connection.close()
'''
