"""
Classes for Instruments
"""

import logging
import json
from typing import Any
import pyvisa
import serial.tools.list_ports
import sys

from NewportMotor import NewportMotor, LS16P, M100D


def compute_serial_to_port_map():
    mapping = {}

    # check if windows:
    if sys.platform.startswith("win"):
        ports = serial.tools.list_ports.comports()

        for port, desc, hwid in sorted(ports):
            if "Newport" in desc and "SER=" in hwid:
                serial_number = hwid.split("SER=")[-1]
                mapping[serial_number] = port
    else:
        raise NotImplementedError("Only windows is supported so far")

    return mapping


class Instrument:
    """
    A class to represent a collection of motors that are connected to the same device
    """

    def __init__(self, config_path) -> None:
        """
        Construct an instrument as a collection of motors

        Created by reading in a configuration file that has the following format:
        [
            {
                "name": "Spherical_1_TipTilt",      // internal name
                "motor_type": "M100D",              // the python type/name of the class
                "motor_config": {                   // optional args for motor constructor
                    "orientation": "reverse"
                },
                "serial_number": "A67BVBOJ"         // the serial number of the motor
            },
            ...
        ]


        Parameters:
        -----------
        config_path: str
            The path to the config file for the instrument
        """
        Instrument._validate_config_file(config_path)
        self._config = Instrument._read_motor_config(config_path)
        self._name_to_port_mapping = self._name_to_port()
        self._motors = self._open_conncetions()

    def has_motor(self, name: str) -> bool:
        """
        Check if the instrument has a motor with the given name

        Parameters:
        -----------
        name: str
            The name of the motor

        Returns:
        --------
        bool
            True if the motor is in the instrument, False otherwise
        """
        return name in self._motors

    def zero_all(self):
        """
        Zero all the motors
        """
        for _, motor in self._motors.items():
            motor.set_to_zero()

    def print_all_positions(self):
        """
        Print the current position of all the motors
        """
        for name, motor in self._motors.items():
            print(f"{name} (COM{self.name_to_port[name]}): {motor.status_string}")

    def _name_to_port(self):
        """
        compute the mapping from the name to the port the motor is connected on
        e.g. spherical_tip_tilt -> /dev/ttyUSB0

        Returns:
        --------
        name_to_port: dict
            A dictionary that maps the name of the motor to the port it is connected to
        """
        serial_to_port = compute_serial_to_port_map()
        name_to_port = {}
        for mapping in self._config:
            serial = mapping["serial_number"]
            logging.info(f"Searching for serial number {serial}")
            try:
                name = mapping["name"]
                port = serial_to_port[serial]
                # check if windows and if so, remove "COM"
                if sys.platform.startswith("win"):
                    port = port[3:]
                name_to_port[name] = port
            except KeyError:
                logging.warning(f" Could not find serial number {serial} in the USBs")
        return name_to_port

    @property
    def name_to_port(self):
        """
        The dictionary that maps the name of the motor to the port it is connected to
        """
        return self._name_to_port_mapping

    def __getitem__(self, key):
        """
        Get a motor by name
        """
        if key not in self._motors:
            raise KeyError(f"Could not find motor {key}")
        return self._motors[key]

    @property
    def motors(self):
        """
        the motors dictionary
        """
        return self._motors

    def _open_conncetions(self):
        """
        For each instrument in the config file, open all the connections and create relevant
        motor objects

        Returns:
        --------
        motors: dict
            A dictionary that maps the name of the motor to the motor object
        """
        resource_manager = pyvisa.ResourceManager()

        motors = {}

        for component in self._config:
            visa_port = f"ASRL{self._name_to_port_mapping[component['name']]}::INSTR"
            motor_class = NewportMotor.string_to_motor_type(component["motor_type"])

            motors[component["name"]] = motor_class(
                visa_port, resource_manager, **component["motor_config"]
            )
        return motors

    @staticmethod
    def _read_motor_config(config_path):
        """
        Read the json config file and return the config dictionary

        Parameters:
        -----------
        config_path: str
            The path to the config file for the instrument

        returns:
        --------
        config: dict
            The config list of dictionaries
        """
        with open(config_path, "r", encoding="utf-8") as file:
            config = json.load(file)
        return config

    @staticmethod
    def _validate_config_file(config_path):
        """
        Reads in the config file and verifies that it is valid

        Parameters:
        -----------
        config_path: str
            The path to the config file for the instrument
        """
        with open(config_path, "r", encoding="utf-8") as file:
            config = json.load(file)

        for component in config:
            if "name" not in component:
                raise ValueError("Each component must have a name")
            if "serial_number" not in component:
                raise ValueError("Each component must have a serial number")
            if "motor_type" not in component:
                raise ValueError("Each component must have a motor type")

            if component["motor_type"] in ["M100D"]:
                M100D.validate_config(component["motor_config"])

        # check that all component names are unique:
        names = [component["name"] for component in config]
        if len(names) != len(set(names)):
            raise ValueError("All component names must be unique")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    mapping = compute_serial_to_port_map()

    print(mapping)

    instrument = Instrument("motor_info_no_linear.json")

    instrument.print_all_positions()
