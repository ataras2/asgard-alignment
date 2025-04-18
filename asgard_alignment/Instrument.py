import asgard_alignment
import json
import sys
import pyvisa
from pathlib import Path
import serial.tools.list_ports
import sys
import pandas as pd
from zaber_motion.ascii import Connection
import numpy as np

import asgard_alignment.CustomMotors
import asgard_alignment.ESOdevice
import asgard_alignment.Engineering
import asgard_alignment.Lamps
import asgard_alignment.NewportMotor
import asgard_alignment.ZaberMotor
import asgard_alignment.Baldr_phasemask

import time

# SDK for DM
# sys.path.insert(1, "/opt/Boston Micromachines/lib/Python3/site-packages/")
import asgard_alignment.controllino

# import bmc


phasemask_position_directory = Path.cwd().joinpath(
    "config_files/phasemask_positions"
)


class Instrument:
    """
    A class that creates connections to controllers, ESOdevice instances,
    and provides a means for the MDS to communicate with the Instrument.
    Open an instrument with a configuration file.
    The configuration file must be a JSON file with the following format:

    .. code-block:: json

        [
            {
                "name": "HTPP1",
                "serial_number": "123456",
                "motor_type": "M100D",
                "motor_config": {}
            },
        ]

    The Instrument will attempt to connect to each device in the config file, opening and
    saving connections to the controllers and devices.

    Parameters
    ----------
    config_pth : str
        The path to the configuration file for the instrument

    Returns
    -------
    None
    """

    def __init__(self, config_pth) -> None:
        self.config_pth = config_pth

        # Validate the config file
        self._validate_config_file(config_pth)
        self._config = self._read_motor_config(config_pth)
        self._motor_config = {
            component["name"]: component for component in self._config["motors"]
        }
        self._lamps_config = {
            component["name"]: component for component in self._config["lamps"]
        }
        self._other_config = {
            component["name"]: component for component in self._config["other_devices"]
        }

        self._semaphore_set = set(
            [
                self._motor_config[component]["semaphore_id"]
                for component in self._motor_config
            ]
        )

        self._controllers = {}
        self._devices = {}  # str of name : ESOdevice
        # bcb
        self.compound_devices = {}  # new dictionary for combined devices (e.g. phasemask)

        self._prev_port_mapping = None
        self._prev_zaber_port = None

        self._rm = pyvisa.ResourceManager()

        # Create the connections to the controllers
        self._open_controllino()
        self._create_controllers_and_motors()
        self._create_lamps()
        self._create_shutters()

        # finally do phasemask objects (the respective motors need to be in devices first)
        self._create_phasemask_wrapper()

    @property
    def devices(self):
        """
        A dictionary of devices with the device name as the key
        """
        return self._devices

    #bcb
    @property
    def all_devices(self):
        """
        Return a merged dictionary of all devices,
        giving access to the entries in compound_devices.
        """
        merged = dict(self._devices)  # copy the standard devices
        merged.update(self.compound_devices)  # phasemask devices override if keys overlap
        return merged

    def health(self):
        """
        Summarise the health of the instrument in a json format
        with the following
        - axis name
        - motor type
        - is connected
        - state
        """

        health = []
        for axis in self._motor_config:
            if axis in self.devices:
                self.ping_connection(axis)
            health.append(
                {
                    "axis": axis,
                    "motor_type": self._motor_config[axis]["motor_type"],
                    "is_connected": axis in self.devices,
                    "state": (
                        self.devices[axis].read_state()
                        if axis in self.devices
                        else None
                    ),
                }
            )

        return health

    def _validate_move_img_pup_inputs(self, config, beam_number, x, y):
        # input validation
        if beam_number not in [1, 2, 3, 4]:
            raise ValueError("beam_number must be in the range [1, 4]")
        if config not in ["c_red_one_focus", "intermediate_focus","baldr"]:
            raise ValueError("config must be 'c_red_one_focus' or 'intermediate_focus' or 'baldr'")




    def move_image(self, config, beam_number, x, y):
        """
        Move the heimdallr image to a new location without moving the pupil

        Parameters
        ----------
        config : str
            The configuration to use - either "c_red_one_focus" or "intermediate_focus"

        beam_number : int
            The beam number to move - in the range [1, 4]

        x : float
            The x coordinate to move to, in pixels

        y : float
            The y coordinate to move to, in pixels

        Returns
        -------
        is_successful : bool
            True if the move was successful, False otherwise
        """
        self._validate_move_img_pup_inputs(config, beam_number, x, y)

        desired_deviation = np.array([[x], [y]])

        _, image_move_matricies = asgard_alignment.Engineering.get_matricies(config)

        M_I = image_move_matricies[beam_number]
        M_I_pupil = M_I[0]
        M_I_image = M_I[1]

        changes_to_deviations = np.array(
            [
                [M_I_pupil, 0.0],
                [0.0, M_I_pupil],
                [M_I_image, 0.0],
                [0.0, M_I_image],
            ]
        )

        # used for heimdallr
        ke_matrix = asgard_alignment.Engineering.knife_edge_orientation_matricies
        so_matrix = asgard_alignment.Engineering.spherical_orientation_matricies
        # used for baldr
        LH_motor = asgard_alignment.Engineering.LH_motor
        RH_motor = asgard_alignment.Engineering.RH_motor

        if config == "baldr":
            pupil_motor = RH_motor
            image_motor = LH_motor
        else:    
            pupil_motor = np.linalg.inv(ke_matrix[beam_number])
            image_motor = np.linalg.inv(so_matrix[beam_number])

        deviations_to_uv = np.block(
            [
                [pupil_motor, np.zeros((2, 2))],
                [np.zeros((2, 2)), image_motor],
            ]
        )

        beam_deviations = changes_to_deviations @ desired_deviation

        print(f"beam deviations: {beam_deviations}")

        uv_commands = deviations_to_uv @ beam_deviations
        
        if config == "baldr":
            axis_list = ["BTP", "BTT", "BOTP", "BOTT"]
        else:
            axis_list = ["HTPP", "HTTP", "HTPI", "HTTI"]

        #axis_list = ["HTPP", "HTTP", "HTPI", "HTTI"]

        axes = [axis + str(beam_number) for axis in axis_list]

        # check that the commands are valid
        is_valid = self._check_commands_against_state(axes, uv_commands)

        if not all(is_valid):
            # figure out which axis/axes are invalid
            invalid_axes = [axis for axis, valid in zip(axes, is_valid) if not valid]
            raise ValueError(f"Invalid move commands for axes: {invalid_axes}")

        # shuffle to parallelise
        self.devices[axes[0]].move_relative(uv_commands[0][0])
        self.devices[axes[2]].move_relative(uv_commands[2][0])
        time.sleep(0.5)
        self.devices[axes[1]].move_relative(uv_commands[1][0])
        self.devices[axes[3]].move_relative(uv_commands[3][0])
        time.sleep(0.5)

        return True



    def move_pupil(self, config, beam_number, x, y):
        """
        Move the Heimdallr pupil to a new location, without moving the image
        """
        self._validate_move_img_pup_inputs(config, beam_number, x, y)

        desired_deviation = np.array([[x], [y]])

        pupil_move_matricies, _ = asgard_alignment.Engineering.get_matricies(config)

        M_P = pupil_move_matricies[beam_number]
        M_P_pupil = M_P[0]
        M_P_image = M_P[1]

        changes_to_deviations = np.array(
            [
                [M_P_pupil, 0.0],
                [0.0, M_P_pupil],
                [M_P_image, 0.0],
                [0.0, M_P_image],
            ]
        )

        ke_matrix = asgard_alignment.Engineering.knife_edge_orientation_matricies
        so_matrix = asgard_alignment.Engineering.spherical_orientation_matricies
        # used for baldr
        LH_motor = asgard_alignment.Engineering.LH_motor
        RH_motor = asgard_alignment.Engineering.RH_motor

        if config == "baldr":
            # Baldr has a different orientation. This will be correct
            # up to a sign in front of one of the motors.
            pupil_motor = RH_motor
            image_motor = LH_motor
        else:
            pupil_motor = np.linalg.inv(ke_matrix[beam_number])
            image_motor = np.linalg.inv(so_matrix[beam_number])

        deviations_to_uv = np.block(
            [
                [pupil_motor, np.zeros((2, 2))],
                [np.zeros((2, 2)), image_motor],
            ]
        )

        beam_deviations = changes_to_deviations @ desired_deviation

        print(f"beam deviations: {beam_deviations}")

        uv_commands = deviations_to_uv @ beam_deviations
        if config == "baldr":
            axis_list = ["BTP", "BTT", "BOTP", "BOTT"]
            print( uv_commands ) # bug shooting 
        else:
            axis_list = ["HTPP", "HTTP", "HTPI", "HTTI"]

        axes = [axis + str(beam_number) for axis in axis_list]

        # check that the commands are valid
        is_valid = self._check_commands_against_state(axes, uv_commands)

        if not all(is_valid):
            # figure out which axis/axes are invalid
            invalid_axes = [axis for axis, valid in zip(axes, is_valid) if not valid]
            raise ValueError(f"Invalid move commands for axes: {invalid_axes}")

        # shuffle to parallelise
        self.devices[axes[0]].move_relative(uv_commands[0][0])
        self.devices[axes[2]].move_relative(uv_commands[2][0])
        time.sleep(0.5)
        self.devices[axes[1]].move_relative(uv_commands[1][0])
        self.devices[axes[3]].move_relative(uv_commands[3][0])
        time.sleep(0.5)

        return True

    def _check_commands_against_state(self, axes, commands, type="rel"):
        """
        Check that the commands are valid for the current state of the axes
        """
        res = []

        for axis, command in zip(axes, commands):
            if type == "rel":
                is_val = self.devices[axis].is_relmove_valid(command)
                res.append(is_val)
            elif type == "abs":
                raise NotImplementedError("Absolute moves not implemented yet")

        return res

    def ping_connection(self, axis):
        """
        Ping the connection to the motor

        Parameters
        ----------
        axis : str
            The name of the motor to ping

        Returns
        -------
        bool
            True if the connection is successful, False otherwise
        """
        print(f"Pinging {axis}...", end="")
        if axis not in self.devices:
            print("not in devices")
            return False

        res = self.devices[axis].ping()

        if not res:
            print("failed")
            # need to remove the connection from dict
            # TODO: include check if it is just the axis or the controller that is down,
            # and remove as needed
            del self.devices[axis]

        print("success")
        return res

    def _create_phasemask_wrapper(self):
        """
        wraps the phasemask x,y motors into a specific Baldr_phasemask class that has
        unique read/write update commands to update all phasemask positions
        based on the current one. This class is also required as input to phasemask alignment tools.
        """
        for beam in [1, 2, 3, 4]:
            if (f"BMX{beam}" not in self.devices) or (f"BMY{beam}" not in self.devices):
                print(
                    f"don't have both phasemasks: (BMX in devices = {(f'BMX{beam}' not in self.devices)}, BMY in devices = {(f'BMX{beam}' not in self.devices)}"
                )
                # Prompt the user for input
                user_input = (
                    input("Type 'y' to continue, or 'n' to stop the program: ")
                    .strip()
                    .lower()
                )

                if user_input == "n":
                    print("Stopping the program as requested.")
                    sys.exit(0)  # Exit the program
                elif user_input == "y":
                    print("Continuing the program...")
                else:
                    print("Invalid input. Assuming continuation.")
            else:
                # try to find if configuration file provided in config file
                pth = phasemask_position_directory.joinpath(Path(f"beam{beam}/"))

                # if not try find the most recent in a predefined folder
                files = list(
                    pth.glob("*.json")
                )  # [file for file in pth.iterdir() if file.is_file()]

                # most recent
                if files:
                    phase_positions_json = max(
                        files, key=lambda file: file.stat().st_mtime, default=None
                    )
                    print(
                        f"using most recent file for beam {beam}: {phase_positions_json}"
                    )
                else:
                    raise UserWarning(
                        f"no phasemask configuration files found in {pth}"
                    )
                # otherwise raise error - we do not want to deal with case where we don't have on

                # do I need to update the self._config dictionaries?
                # bcb #self.devices[f"phasemask{beam}"] 
                self.compound_devices[f"phasemask{beam}"] = (
                    asgard_alignment.Baldr_phasemask.BaldrPhaseMask(
                        beam=beam,
                        x_axis_motor=self.devices[f"BMX{beam}"],
                        y_axis_motor=self.devices[f"BMY{beam}"],
                        phase_positions_json=phase_positions_json,
                    )
                )
    # BCB to do , make new variable dictionary (not device) 
    # _combined_device <- new variable dictionary , multiDeviceServer <- custom functions 
    # update Mutil device server 
    #  
    def _open_controllino(self):
        self._controllers["controllino"] = asgard_alignment.controllino.Controllino(
            self._other_config["controllino"]["ip_address"]
        )

    def _create_controllers_and_motors(self):
        """
        Create the connections to the controllers and motors

        Returns:
        --------
        motors: dict
            A dictionary that maps the name of the motor to the motor object
        """

        self._prev_port_mapping = self.compute_serial_to_port_map()
        self._prev_zaber_port = self.find_zaber_usb_port()
        for name in self._motor_config:
            res = self._attempt_to_open(name, recheck_ports=False)
            if res:
                print(f"Successfully connected to {name}")
            else:
                print(f"Could not connect to {name}")

    def _create_lamps(self):
        """
        Create the connections to the lamps
        """
        for name in self._lamps_config:
            self.devices[name] = asgard_alignment.Lamps.LightSource(
                name,
                self._controllers["controllino"],
                **self._lamps_config[name]["config"],
            )

    def _create_shutters(self):
        """
        Create the connections to the shutters
        """

    def _attempt_to_open(self, name, recheck_ports=False):
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
        if name not in self._motor_config:
            raise ValueError(f"{name} is not in the config file")

        if self._motor_config[name]["motor_type"] == "deformable_mirror":
            # using shared memory (set up server such that DM is running and always looking to the shared memory)

            # otherwise we are connecting directly here to the DM
            serial_number = self._motor_config[name].get("serial_number")

            # Load flat map and initialize DM
            dm = bmc.BmcDm()
            if dm.open_dm(serial_number) != 0:
                print(f"Failed to connect to DM with serial number {serial_number}")
                return False
            flat_map_file = self._motor_config[name]["flat_map_file"]
            flat_map = pd.read_csv(flat_map_file, header=None)[0].values
            cross_map = pd.read_csv("DMShapes/Crosshair140.csv", header=None)[0].values
            self._devices[name] = {
                "dm": dm,
                "flat_map": flat_map,
                "cross_map": cross_map,
            }
            # print(f"Connected to {name} with serial {serial_number}")
            return True

        if self._motor_config[name]["motor_type"] in ["M100D", "LS16P"]:
            # this is a newport motor USB connection, create a newport motor
            # object
            cfg = self._motor_config[name]

            if recheck_ports:
                self._prev_port_mapping = self.compute_serial_to_port_map()

            if cfg["serial_number"] not in self._prev_port_mapping:
                return False

            port = self._prev_port_mapping[cfg["serial_number"]]
            if port not in self._controllers:
                self._controllers[port] = (
                    asgard_alignment.NewportMotor.NewportConnection(
                        port,
                        self._rm,
                    )
                )

            if self._motor_config[name]["motor_type"] in ["M100D"]:
                self.devices[name] = asgard_alignment.NewportMotor.M100DAxis(
                    self._controllers[port],
                    cfg["semaphore_id"],
                    cfg["motor_config"]["axis"],
                    name,
                )
                return True

            if self._motor_config[name]["motor_type"] in ["LS16P"]:
                self.devices[name] = asgard_alignment.NewportMotor.LS16PAxis(
                    self._controllers[port],
                    cfg["semaphore_id"],
                    name,
                )
                return True

            raise ValueError(
                f"Unknown motor type {self._motor_config[name]['motor_type']}"
            )

        elif self._motor_config[name]["motor_type"] in ["LAC10A-T4A"]:
            # this is a zaber motor, create a ZaberLinearActuator object
            # through the X-MCC
            cfg = self._motor_config[name]
            if cfg["x_mcc_ip_address"] not in self._controllers:
                self._controllers[cfg["x_mcc_ip_address"]] = Connection.open_tcp(
                    cfg["x_mcc_ip_address"]
                )
                self._controllers[cfg["x_mcc_ip_address"]].get_device(1).identify()
                self._controllers[cfg["x_mcc_ip_address"]].get_device(1).settings.set(
                    "system.led.enable", 0
                )

            axis = (
                self._controllers[cfg["x_mcc_ip_address"]]
                .get_device(1)
                .get_axis(cfg["axis_number"])
            )

            if "FZ" in axis.warnings.get_flags():
                return False
            self._devices[name] = asgard_alignment.ZaberMotor.ZaberLinearActuator(
                name,
                cfg["semaphore_id"],
                axis,
            )
            return True

        elif self._motor_config[name]["motor_type"] in [
            "X-LSM150A-SE03",
            "X-LHM100A-SE03",
        ]:
            # this is a zaber connection through USB
            # check what the zaber com port is
            if recheck_ports:
                self._prev_zaber_port = self.find_zaber_usb_port()

            if self._prev_zaber_port is None:
                return False

            if self._prev_zaber_port not in self._controllers:
                self._controllers[self._prev_zaber_port] = Connection.open_serial_port(
                    self._prev_zaber_port
                )

            for dev in self._controllers[self._prev_zaber_port].detect_devices():
                if dev.serial_number == self._motor_config[name]["serial_number"]:
                    dev.settings.set("system.led.enable", 0)
                    self._devices[name] = asgard_alignment.ZaberMotor.ZaberLinearStage(
                        name,
                        self._motor_config[name]["semaphore_id"],
                        dev,
                    )
                    return True
        elif self._motor_config[name]["motor_type"] in ["8893KM"]:
            self.devices[name] = asgard_alignment.CustomMotors.MirrorFlipper(
                name,
                self._motor_config[name]["semaphore_id"],
                self._controllers["controllino"],
                self._motor_config[name]["motor_config"]["modulation_value"],
                self._motor_config[name]["motor_config"]["delay_time"],
            )
            return True
        elif self._motor_config[name]["motor_type"] in ["MFF101M"]:
            self.devices[name] = asgard_alignment.CustomMotors.MFF101(
                name,
                self._motor_config[name]["semaphore_id"],
                self._controllers["controllino"],
                self._motor_config[name]["named_pos"],
            )
            return True

    @staticmethod
    def find_zaber_usb_port():
        """
        Find the COM port for the Zaber motor

        Returns:
        --------
        str
            The COM port for the Zaber motor
        """
        ports = serial.tools.list_ports.comports()
        possible_ports = []

        #First, we need the list of cusbi ports. If we don't do this, the next time we query, 
        #we get an error. I've put this as a fake list for now.
        #Command to find is cusbi (in ~/bin). Syntax
        #./cusbi /Q:ttyUSB0
        #If we query all ports, then some Newport M100D ports get confused.
        #See:
        #https://sgcdn.startech.com/005329/media/sets/USB_Admin_Software_Manual/USB_Hub_Admin_Software_Manual.pdf
        #If we try to open a Zaber connection to this port, then it seems stuck in an error state.
        cusb_ports = ["/dev/ttyUSBX"]

        for port, _, hwid in sorted(ports):
            if "VID:PID=0403:6001" in hwid:
                if port in cusb_ports:
                    print("Found a Managed USB hub.")
                    continue
                #Try to open it - if we can, it is a Zaber port!
                test_connection = Connection.open_serial_port(port)
                try:
                    devices = test_connection.detect_devices()
                    test_connection.close()
                    print(f"Found a Zaber USB port {port}")
                    return port
                except:
                    #This next line is essential, or the port remains open and no other process
                    #can use it (including MDS later in the code)
                    test_connection.close()
                    print(f"A non-zaber motor using the same USB ID. Port {port}")
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

        sub_dicts = ["motors", "other_devices"]

        for sub_dict in sub_dicts:
            if sub_dict not in config:
                raise ValueError(
                    f"Config file must have a {sub_dict} key with a list of dictionaries"
                )

        for component in config["motors"]:
            if "name" not in component:
                raise ValueError("Each component must have a name")
            if "serial_number" not in component:
                raise ValueError("Each component must have a serial number")
            if "motor_type" not in component:
                raise ValueError("Each component must have a motor type")
            if "semaphore_id" not in component:
                raise ValueError("Each component must have a semaphore id")

        # check that no semaphore id is used more than twice
        semaphore_ids = [component["semaphore_id"] for component in config["motors"]]
        for semaphore_id in set(semaphore_ids):
            if semaphore_ids.count(semaphore_id) > 2:
                raise ValueError(
                    f"Semaphore id must be unique, not true for {semaphore_id}"
                )

        # check that all component names are unique:
        names = []
        for key in config:
            for component in config[key]:
                names.append(component["name"])
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
        """
        By inspecting the list of usb devices, find the serial number of the
        motor and the corresponding port (e.g. /dev/ttyUSB0)

        Returns
        --------
        mapping: dict
            A dictionary that maps the serial number of the motor to the port
        """
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
                baud_rate=asgard_alignment.NewportMotor.NewportConnection.SERIAL_BAUD,
                write_termination=asgard_alignment.NewportMotor.NewportConnection.SERIAL_TERMIN,
                read_termination=asgard_alignment.NewportMotor.NewportConnection.SERIAL_TERMIN,
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

    print(instr.devices["HTPP1"].read_position())
    print(instr.devices["HTTP1"].read_position())
