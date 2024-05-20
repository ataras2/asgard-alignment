"""
Classes for Instruments
"""

import logging
import json
from typing import Any
import pyvisa

from newport_motors.USBs.USBs import USBs
from newport_motors.Motors.motor import M100D, LS16P, Motor


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

    def zero_all(self):
        """
        Zero all the motors
        """
        for _, motor in self._motors.items():
            motor.set_to_zero()

    def _name_to_port(self):
        """
        compute the mapping from the name to the port the motor is connected on
        e.g. spherical_tip_tilt -> /dev/ttyUSB0

        Returns:
        --------
        name_to_port: dict
            A dictionary that maps the name of the motor to the port it is connected to
        """
        filt = {"iManufacturer": "Newport"}
        serial_to_port = USBs.compute_serial_to_port_map(filt)
        name_to_port = {}
        for mapping in self._config:
            serial = mapping["serial_number"]
            logging.info(f"Searching for serial number {serial}")
            try:
                name = mapping["name"]
                port = serial_to_port[serial]
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

    def __getitem__(self, key) -> Motor:
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
        resource_manager = pyvisa.ResourceManager(visa_library="@_py")

        motors = {}

        for component in self._config:
            visa_port = f"ASRL{self._name_to_port_mapping[component['name']]}::INSTR"
            motor_class = Motor.string_to_motor_type(component["motor_type"])

            motors[component["motor_type"]] = motor_class(
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

    @staticmethod
    def create_config_with_plugin(config_path, motor_names, infer_motor_type=True):
        """
        Using a USB monitor, create a config file with all the motors that are
        connected one at a time

        config_path: path for where to save the resulting config file
        motor_names:
        infer_motor_type: will use the end of the name to decide which class to instantiate

        e.g. for an input of ["Spherical_1_TipTilt", "Spherical_2_TipTilt"],
        create a config file such as:
        [
            {
                "name": "Spherical_1_TipTilt",
                "motor_type": "M100D",
                "motor_config": {
                    "orientation": "reverse"
                },
                "serial_number": "A67BVBOJ"
            },
            {
                "name": "Spherical_2_TipTilt",
                "motor_type": "M100D",
                "motor_config": {
                    "orientation": "normal"
                },
                "serial_number": "A675RRE6"
            }
        ]
        """
        logging.info("Creating config file, unplug all motors")
        input("Press enter once all motors have been removed...")

        usb = USBs()
        configs = []

        for motor in motor_names:
            conf_to_add = {}
            input(f"Plug in {motor} and hit enter...")

            new_serial = usb.get_difference()
            if len(new_serial) != 1:
                raise RuntimeError(
                    f"Was expecting one new serial device, got {len(new_serial)}"
                )

            motor_type = None
            if infer_motor_type:
                motor_type = Motor.infer_motor_type(motor)

            conf_to_add["name"] = motor
            conf_to_add["motor_type"] = Motor.motor_type_to_string(motor_type)
            conf_to_add["serial_number"] = new_serial[0]
            conf_to_add["motor_config"] = motor_type.setup_individual_config()

            configs.append(conf_to_add)

        with open(config_path, "w") as f:
            json.dump(configs, f)
