"""
Classes for Instruments
"""

import logging
import json
from typing import Any
import pyvisa
import serial.tools.list_ports
import sys
import parse


from zaber_motion.ascii import Connection

from asgard_alignment.NewportMotor import NewportMotor, LS16P, M100D
from asgard_alignment.ZaberMotor import (
    LAC10AT4A,
    ZaberLinearStage,
    ZaberLinearActuator,
)


# from NewportMotor import NewportMotor, LS16P, M100D


def compute_serial_to_port_map():
    mapping = {}

    ports = serial.tools.list_ports.comports()
    # check if windows:
    if sys.platform.startswith("win"):
        for port, desc, hwid in sorted(ports):
            if "Newport" in desc and "SER=" in hwid:
                serial_number = hwid.split("SER=")[-1]
                mapping[serial_number] = port
    else:
        for port, desc, hwid in sorted(ports):
            if "CONEX" in desc and "SER=" in hwid:
                serial_number = hwid.split("SER=")[-1].split("LOC")[0].strip() + "A"
                mapping[serial_number] = port

    # now for the checking of LS16P devices, which are extra weird since they don't
    # present a serial number

    if sys.platform.startswith("win"):
        pass
    else:
        # list all ttyACM devices
        rm = pyvisa.ResourceManager()

        for device in rm.list_resources():
            if "ttyACM" in device:
                try:
                    pass
                    # connect to the motor and query SA
                    # mapping["SA1"] = port kind of thing
                    sa = LS16P.connect_and_get_SA(device)
                    if sa is not None:
                        # device is of  the form ASRL/dev/ttyACM1::INSTR
                        # want just the /dev/ttyACM1 part
                        port = device.split("::")[0][4:]
                        mapping[sa] = port

                except Exception as e:
                    logging.warning(f"Could not connect to {device}: {e}")

    return mapping


def find_zaber_COM():
    """
    Find the COM port for the Zaber motor

    Returns:
    --------
    str
        The COM port for the Zaber motor
    """
    ports = serial.tools.list_ports.comports()

    for port, _, hwid in sorted(ports):
        if "VID:PID=0403:6001" in hwid:
            return port
    return None


class Instrument:
    """
    A class to represent a collection of motors that are connected to the same machine
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

    def handle_zmq(self, message: Any) -> Any:
        """
        main function to handle the ZMQ messages from the outside world
        Note that zmq messages target the axes, not the motors themselves
        """

        if message == "ls":
            return "\n".join(self._motors)

        if message == "init_all":
            for motor in self._motors.values():
                motor.initialise()
            return "ACK"

        if message.startswith("move_rel"):
            # parse message of form  "move_rel <device_name> <value>"
            parse_results = parse.parse("move_rel {device_name} {value}", message)
            if parse_results is None:
                return "Could not parse message"
            if self.has_motor(parse_results["device_name"]) is False:
                return f"Could not find motor {parse_results['device_name']}"
            motor = self._motors[parse_results["device_name"]]
            motor.move_relative(float(parse_results["value"]))
            return "ACK"
        
        if message.startswith("move_abs"):
            # parse message of form  "move_abs <device_name> <value>"
            parse_results = parse.parse("move_abs {device_name} {value}", message)
            if parse_results is None:
                return "Could not parse message"
            if self.has_motor(parse_results["device_name"]) is False:
                return f"Could not find motor {parse_results['device_name']}"
            motor = self._motors[parse_results["device_name"]]
            motor.move_absolute(float(parse_results["value"]))
            return "ACK"

        def validate_message(parsed_message):
            """
            Check if the message is a valid message
            """

            if parsed_message is None:
                return "Could not parse message"

            if self.has_motor(parsed_message["device_name"]) is False:
                return f"Could not find motor {parsed_message['device_name']}"

            if parsed_message["parameter"] not in ["lrPosition"]:
                return f"Parameter {parsed_message['parameter']} not implemented"

            return None

        def handle_write_message(msg):
            """
            Handle a write message
            """
            # of the form MAIN1.<device name>.<parameter category>.<parameter name>=<value>
            parse_results = parse.parse(
                "MAIN1.{device_name}.{category}.{parameter}={value}", msg
            )

            error = validate_message(parse_results)
            if error is not None:
                return error

            motor = self._motors[parse_results["device_name"]]

            if parse_results["parameter"] == "lrPosition":
                motor.set_position(float(parse_results["value"]))

            return "ACK"

        def handle_read_message(msg):
            # this is a get command
            # of the form MAIN1.<device name>.<parameter category>.<parameter name>
            parse_results = parse.parse(
                "MAIN1.{device_name}.{category}.{parameter}", msg
            )

            error = validate_message(parse_results)
            if error is not None:
                return error

            motor = self._motors[parse_results["device_name"]]

            if parse_results["parameter"] == "lrPosition":
                response = f"r{motor.position}"

            return response

        if "=" in message:
            return handle_write_message(message)
        return handle_read_message(message)

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
        Open all the connections to the motors

        Returns:
        --------
        motors: dict
            A dictionary that maps the name of the motor to the motor object
        """
        # merge both newport and zaber connections
        newport_motors = self._open_newport_conncetions()
        zaber_motors = self._open_zaber_conncetions()

        return {**newport_motors, **zaber_motors}

    def _open_zaber_conncetions(self):
        motors = {}

        # first deal with the USB
        zaber_port = find_zaber_COM()
        if zaber_port is not None:
            self.zaber_com_connection = Connection.open_serial_port(zaber_port)
            self.zaber_com_connection.enable_alerts()

            device_list = self.zaber_com_connection.detect_devices()
            print("Found {} zaber devices on COM port".format(len(device_list)))

            for dev in device_list:
                for motor_config in self._config:
                    if dev.serial_number == motor_config["serial_number"]:
                        if dev.name in ["X-LSM150A-SE03", "X-LHM100A-SE03"]:
                            motors[motor_config["name"]] = ZaberLinearStage(dev)

        # now deal with all the networked motors
        self.zaber_ip_connections = {}  # IP address, connection
        for component in self._config:
            if component["motor_type"] not in ["LAC10A-T4A"]:
                continue
            if component["name"] not in self._name_to_port_mapping:
                continue

            if component["x_mcc_ip_address"] not in self.zaber_ip_connections:
                self.zaber_ip_connections[component["x_mcc_ip_address"]] = (
                    Connection.open_tcp(component["x_mcc_ip_address"])
                )

            axis = (
                self.zaber_ip_connections[component["x_mcc_ip_address"]]
                .get_device(0)
                .get_axis(component["axis_number"])
            )

            motors[component["name"]] = ZaberLinearActuator(axis)

        return motors

    def _open_newport_conncetions(self):
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
            if component["motor_type"] not in ["M100D", "LS16P"]:
                continue
            if component["name"] not in self._name_to_port_mapping:
                continue
            visa_port = f"ASRL{self._name_to_port_mapping[component['name']]}::INSTR"
            motor_class = NewportMotor.string_to_motor_type(component["motor_type"])

            motors[component["name"]] = motor_class(
                visa_port, resource_manager, **component["motor_config"]
            )
        return motors

    def close_connections(self):
        """
        Close all the connections to the motors
        """
        self.zaber_com_connection.close()

        for motor in self._motors.values():
            # check if newport motor
            if isinstance(motor, NewportMotor):
                motor.close_connection()

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