import asgard_alignment
import json
import pyvisa
import serial.tools.list_ports
import sys

from zaber_motion.ascii import Connection

import asgard_alignment.ESOdevice
import asgard_alignment.ZaberMotor


class Instrument:
    """
    A class that creates connections to controllers, ESOdevice instances,
    and provides a means for the MDS to communicate with the Instrument.
    """

    def __init__(self, config_pth) -> None:
        self.config_pth = config_pth

        # Validate the config file
        self._validate_config_file(config_pth)
        self._config = self._read_motor_config(config_pth)
        self._config_dict = {component["name"]: component for component in self._config}

        self._controllers = {}
        self._devices = {}  # str of name : ESOdevice

        self._rm = pyvisa.ResourceManager()

        # Create the connections to the controllers
        self._create_controllers_and_motors()
        self._create_lamps()
        self._create_shutters()

    @property
    def devices(self):
        """
        A dictionary of devices with the device name as the key
        """
        return self._devices

    def _create_controllers_and_motors(self):
        """
        Create the connections to the controllers and motors

        Returns:
        --------
        motors: dict
            A dictionary that maps the name of the motor to the motor object
        """
        for name in self._config_dict:
            res = self._attempt_to_open(name)
            if res:
                print(f"Successfully connected to {name}")
            else:
                print(f"Could not connect to {name}")

    def _create_lamps(self):
        """
        Create the connections to the lamps
        """

    def _create_shutters(self):
        """
        Create the connections to the shutters
        """

    def _attempt_to_open(self, name):
        """
        Attempt to open a connection to a device.
        First, check if the controller is already in the connections
        dictionary.

        Parameters:
        -----------
        name: str
            The name of the device to connect to

        Returns:
        --------
        bool
            True if the connection was successful, False otherwise
        """
        if name not in self._config_dict:
            raise ValueError(f"{name} is not in the config file")

        if self._config_dict[name]["motor_type"] in ["M100D", "LS16P"]:
            # this is a newport motor USB connection, create a newport motor
            # object
            pass
        elif self._config_dict[name]["motor_type"] in ["LAC10A-T4A"]:
            # this is a zaber motor, create a ZaberLinearActuator object
            # through the X-MCC
            cfg = self._config_dict[name]
            if cfg["x_mcc_ip_address"] not in self._controllers:
                self._controllers[cfg["x_mcc_ip_address"]] = Connection.open_tcp(
                    cfg["x_mcc_ip_address"]
                )
                self._controllers[cfg["x_mcc_ip_address"]].get_device(1).identify()

            axis = (
                self._controllers[cfg["x_mcc_ip_address"]]
                .get_device(1)
                .get_axis(cfg["axis_number"])
            )

            if "FZ" in axis.warnings.get_flags():
                return False
            
            self._devices[name] = asgard_alignment.ZaberMotor.ZaberLinearActuator(
                name,
                axis,
            )
            return True

        elif self._config_dict[name]["motor_type"] in [
            "X-LSM150A-SE03",
            "X-LHM100A-SE03",
        ]:
            # this is a zaber connection through USB
            # check what the zaber com port is
            zaber_com_port = self.find_zaber_COM()

            if zaber_com_port is None:
                return False
            
            if zaber_com_port not in self._controllers:
                self._controllers[zaber_com_port] = Connection.open_serial_port(
                    zaber_com_port
                )
            
            for dev in self._controllers[zaber_com_port].detect_devices():
                if dev.serial_number == self._config_dict[name]["serial_number"]:
                    self._devices[name] = asgard_alignment.ZaberMotor.ZaberLinearStage(
                        name,
                        dev,
                    )
                    return True

    @staticmethod
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

        # check that all component names are unique:
        names = [component["name"] for component in config]
        if len(names) != len(set(names)):
            raise ValueError("All component names must be unique")

        # check that all combinations of ip address + axis number
        # are unique
        ip_with_axis = []
        for component in config:
            if "x_mcc_ip_address" in component:
                ip_with_axis.append(
                    (component["x_mcc_ip_address"], component["axis_number"])
                )

        if len(ip_with_axis) != len(set(ip_with_axis)):
            raise ValueError(
                "All combinations of ip address and axis number must be unique"
            )

    @staticmethod
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

        def connect_and_get_SA(rm, port):
            """
            Connect to the motor and check the SA
            """

            connection = rm.open_resource(
                port,
                baud_rate=asgard_alignment.NewportMotor.LS16P.SERIAL_BAUD,
                write_termination=asgard_alignment.NewportMotor.LS16P.SERIAL_TERMIN,
                read_termination=asgard_alignment.NewportMotor.LS16P.SERIAL_TERMIN,
            )
            sa = connection.query("SA?").strip()
            connection.before_close()
            connection.close()
            return sa

        rm = pyvisa.ResourceManager()
        # now for the checking of LS16P devices, which are extra weird since they don't
        # present a serial number
        if sys.platform.startswith("win"):
            raise NotImplementedError("Windows not supported")

        # list all ttyACM devices
        rm = pyvisa.ResourceManager()

        for device in rm.list_resources():
            if "ttyACM" in device:
                try:
                    # connect to the motor and query SA
                    # mapping["SA1"] = port kind of thing
                    sa = connect_and_get_SA(rm, device)
                    if sa is not None:
                        # device is of  the form ASRL/dev/ttyACM1::INSTR
                        # want just the /dev/ttyACM1 part
                        port = device.split("::")[0][4:]
                        mapping[sa] = port

                except Exception as e:
                    print(f"Could not connect to {device}: {e}")

        return mapping


if __name__ == "__main__":
    pth = "motor_info_full_system.json"
    # Instrument._validate_config_file(pth)

    # print(Instrument.compute_serial_to_port_map())

    # config = Instrument._read_motor_config(pth)

    instr = Instrument(pth)
