import zmq
import asgard_alignment
import argparse
import sys
import re
from parse import parse

import enum
import asgard_alignment.ESOdevice
import asgard_alignment.Instrument
import asgard_alignment.MultiDeviceServer


class MockMDS:
    def __init__(self):
        pass

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


class MultiDeviceServer:
    """
    A class to run the Instrument MDS.
    """

    def __init__(self, port, host, config_file):
        self.port = port
        self.host = host
        self.config_file = config_file
        self.context = zmq.Context()
        self.server = self.context.socket(zmq.REP)
        self.server.bind(f"tcp://{self.host}:{self.port}")
        self.poller = zmq.Poller()
        self.poller.register(self.server, zmq.POLLIN)

        if config_file == "mock":
            self.instr = MockMDS()
        else:
            self.instr = asgard_alignment.Instrument.Instrument(self.config_file)

    def socket_funct(self, s):
        try:
            message = s.recv_string()
            return message
        except zmq.ZMQError as e:
            print(f"ZMQ Error: {e}")
            return -1

    def log(self, message):
        print(message)

    def run(self):
        running = True
        while running:
            inputready = []
            socks = dict(self.poller.poll(10))
            if self.server in socks and socks[self.server] == zmq.POLLIN:
                inputready.append(self.server)
            for s in inputready:  # loop through our array of sockets/inputs
                data = self.socket_funct(s)
                if data == -1:
                    running = False
                elif data != 0:
                    response = self.handle_message(data)
                    if response == -1:
                        running = False
                        if s == sys.stdin:
                            self.log("Manually shut down. Goodbye.")
                        else:
                            self.log("Shut down by remote connection. Goodbye.")
                    else:
                        s.send_string(response + "\n")

    def handle_message(self, message):
        if "!" in message:
            return self._handle_custom_command(message)

        if ("MAIN" in message) and (message.count(".") > 2):
            valid = True
        else:
            valid = False
        if valid:
            if "=" in message:
                # Received request is "write"
                x = re.split("=", message)
                read_val = x[1]
                y = re.split("\.", x[0])
                device = y[1]
                category = y[2]
                # Deal with the case of parameter arrays (two atoms)
                if len(y) == 5:
                    # Replace dot and braces by underscores
                    parameter = re.sub("\[|\]", "_", y[3]) + "_" + y[4]
                    par_type = y[4][0]
                else:
                    parameter = y[3]
                    par_type = y[3][0]
                if par_type == "b":
                    value = read_val == "TRUE"
                if par_type == "n":
                    value = int(read_val)
                if par_type == "l":
                    value = float(read_val)

                # Update parameter value of device
                print("parameter = ", parameter)
                if hasattr(self.instr.devices[device], parameter):
                    setattr(self.instr.devices[device], parameter, value)

                    self.instr.devices[device].update_param()
                    # if (
                    #     self.instr.devices[device]._dev_type
                    #     == asgard_alignment.ESOdevice.Motor
                    # ):
                    #     self.instr.devices[device].update_param()
                    # elif self.instr.devices[device]._dev_type == asgard_alignment.ESOdevice.Lamp:
                    #     update_lamp_param(d)
                    # """
                    # elif p[d]._devType == SHUTTER:
                    #     update_shutter_params(p[d])
                    # elif p[d]._devType == SENSOR:
                    #     update_sensor_params(p[d])
                    # """
                    # Send back acknowledgement to client
                    print("Updated parameter", parameter, "of", device, "to", value)
                return "ACK"
            else:
                # Received request is "read"
                x = re.split("\.", message)
                device = x[1]
                category = x[2]
                # Deal with the case of parameter arrays (two atoms)
                if len(x) == 5:
                    # Replace dot and braces by underscores
                    parameter = re.sub("\[|\]", "_", x[3]) + "_" + x[4]
                    par_type = x[4][0]
                else:
                    parameter = x[3]
                    par_type = x[3][0]

                if device not in self.instr.devices:
                    return "Device not found"
                if hasattr(self.instr.devices[device], parameter):
                    value = getattr(self.instr.devices[device], parameter)
                    if type(value) == int:
                        reply = "n" + str(value)
                    elif type(value) == float:
                        reply = "r" + str(value)
                    elif type(value) == bool:
                        if value:
                            reply = "bTRUE"
                        else:
                            reply = "bFALSE"
                    elif isinstance(value, enum.Enum):
                        reply = "s" + str(value.name)
                    else:
                        reply = "s--UNKNOWN--"
                    print("Value of parameter", parameter, "of", device, "is:", value)
                else:
                    # Unknown parameter: send back garbage value according to type
                    if par_type == "b":
                        reply = "bFALSE"
                    elif par_type == "n":
                        reply = "n9999"
                    elif par_type == "l":
                        reply = "r99.99"
                    else:
                        reply = "s--UNKNOWN--"

                self.instr.devices[device].update_fsm()
                # Send reply to client
                print("OUT>", reply)
                return reply
        else:
            # Garbage received => send anything to avoid blocking
            reply = "????"
            print("OUT>", reply)
            return reply

    def _handle_custom_command(self, message):
        # this is a custom command, acutally do useful things here lol
        def read_msg(axis):
            return str(self.instr.devices[axis].read_position())

        def stop_msg(axis):
            return str(self.instr.devices[axis].stop())

        def moveabs_msg(axis, position):
            self.instr.devices[axis].move_abs(float(position))
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

        def connect_msg(axis):
            # this is a connection open request
            self.instr._attempt_to_open(axis, recheck_ports=True)

            return "connected" if axis in self.instr.devices else "not connected"

        patterns = {
            "!read {}": read_msg,
            "!stop {}": stop_msg,
            "!moveabs {} {:f}": moveabs_msg,
            "!connected? {}": connected_msg,
            "!connect {}": connect_msg,
        }

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
    parser = argparse.ArgumentParser(description="Run the MDS server.")
    parser.add_argument(
        "-c", "--config", type=str, required=True, help="Path to the configuration file"
    )
    parser.add_argument("--host", type=str, default="localhost", help="Host address")
    parser.add_argument("-p", "--port", type=int, default=5555, help="Port number")

    args = parser.parse_args()

    serv = MultiDeviceServer(args.port, args.host, args.config)
    serv.run()
