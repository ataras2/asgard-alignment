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
import os
import glob
import asgard_alignment.CustomMotors
import asgard_alignment.ESOdevice
import asgard_alignment.Engineering
import asgard_alignment.Lamps
import asgard_alignment.NewportMotor
import asgard_alignment.ZaberMotor
import asgard_alignment.Baldr_phasemask

import logging
import time

import asgard_alignment.controllino


phasemask_position_directory = Path.cwd().joinpath("config_files/phasemask_positions")


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
        self.compound_devices = (
            {}
        )  # new dictionary for combined devices (e.g. phasemask)

        self._prev_port_mapping = None
        self._prev_zaber_port = None
        self._zaber_detected_devs = None

        self._rm = pyvisa.ResourceManager()

        # Create the connections to the controllers
        self._open_controllino()
        logging.info("Controllino on")

        time.sleep(
            2
        )  # TODO: look at the value here- longer to let managed port be found?

        self.managed_usb_hub_port = self.find_managed_USB_hub_port()
        logging.info(f"managed port: {self.managed_usb_hub_port}")
        if self.managed_usb_hub_port is None:
            logging.warning(
                "Could not find managed USB hub port."
            )

            # time.sleep(5)
            sys.exit(1)
        else:
            self.managed_usb_port_short = self.managed_usb_hub_port.split("/")[-1]
            logging.info(f"managed port short: {self.managed_usb_port_short}")

        time.sleep(3)
        os.system(f"cusbi /S:{self.managed_usb_port_short} 1:1")
        time.sleep(0.2)
        os.system(f"cusbi /S:{self.managed_usb_port_short} 1:2")
        time.sleep(0.2)
        os.system(f"cusbi /S:{self.managed_usb_port_short} 1:3")

        logging.info("Turned on USB hubs, waiting 10 seconds...")
        time.sleep(10.0)  # wait for all usb connections to be established
        self._create_controllers_and_motors()
        self._create_lamps()
        self._create_shutters()

        self.temp_summary = TemperatureSummary(self._controllers["controllino"])

        # finally do phasemask objects (the respective motors need to be in devices first)
        self._create_phasemask_wrapper()

        self.h_shutter_states = {i: "open" for i in range(1, 5)}
        self.h_shutter_offsets = {
            i: {f"HTTP{i}": 0.0, f"HTPP{i}": 0.0} for i in range(1, 5)
        }
        self.is_h_splay = "off"

    @property
    def devices(self):
        """
        A dictionary of devices with the device name as the key
        """
        return self._devices

    # bcb
    @property
    def all_devices(self):
        """
        Return a merged dictionary of all devices,
        giving access to the entries in compound_devices.
        """
        merged = dict(self._devices)  # copy the standard devices
        merged.update(
            self.compound_devices
        )  # phasemask devices override if keys overlap
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

    def save(self, subset, filename):
        """
        Save the current state of the instrument to a file

        Parameters
        ----------
        subset : str
            The subset of the instrument to save, e.g. "devices", "lamps", "controllers"
        filename : str
            The name of the file to save to
        """
        motor_names = []
        if subset == "solarstein" or subset == "all":
            motor_names += ["SDLA", "SDL12", "SDL34", "SSS", "SSF"]
        if subset == "heimdallr" or subset == "all":
            self.h_shut("open", [1, 2, 3, 4])
            time.sleep(2)
            motor_names_all_beams = [
                "HFO",
                "HTPP",
                "HTPI",
                "HTTP",
                "HTTI",
            ]

            for motor in motor_names_all_beams:
                for beam_number in range(1, 5):
                    motor_names.append(f"{motor}{beam_number}")
        if subset == "baldr" or subset == "all":
            motor_names += ["BFO"]

            motor_names_all_beams = [
                "BDS",
                "BTT",
                "BTP",
                "BMX",
                "BMY",
                "BLF",
            ]

            partially_common_motors = [
                "BOTT",
                "BOTP",
            ]

            for motor in partially_common_motors:
                for beam_number in range(2, 5):
                    motor_names.append(f"{motor}{beam_number}")

            for motor in motor_names_all_beams:
                for beam_number in range(1, 5):
                    motor_names.append(f"{motor}{beam_number}")

        states = []
        for name in motor_names:
            is_connected = self.ping_connection(name)

            state = {
                "name": name,
                "is_connected": is_connected,
            }
            if is_connected:
                res = self.devices[name].read_position()
                if res != "None":
                    state["position"] = float(res)
            states.append(state)
            logging.info(state)

        fname = os.path.expanduser(
            "~/.config/asgard-alignment/instr_states/" + filename + ".json"
        )
        if os.path.exists(fname):
            return "NACK: File already exists, please choose a different name."
        else:
            # save to json at location
            with open(fname, "w") as f:
                json.dump(states, f, indent=4)
            logging.info("state saved")
            return "ACK: saved"

    def h_splay(self, state):
        if state not in ["on", "off"]:
            raise ValueError("state must be 'on' or 'off'")
        if state == self.is_h_splay:
            logging.info(f"H splay already in {state} state, doing nothing")
            return

        offset_mag = 15  # pixels
        beam_deviations = {
            1: {"x": -np.sqrt(3) / 2 * offset_mag, "y": 0.5 * offset_mag},
            2: {"x": np.sqrt(3) / 2 * offset_mag, "y": 0.5 * offset_mag},
            4: {"x": 0, "y": -offset_mag},
        }

        if state == "on":
            for beam_number in [1, 2, 4]:
                self.move_image(
                    "c_red_one_focus",
                    beam_number,
                    beam_deviations[beam_number]["x"],
                    beam_deviations[beam_number]["y"],
                )
            self.is_h_splay = True
        elif state == "off":
            for beam_number in [1, 2, 4]:
                self.move_image(
                    "c_red_one_focus",
                    beam_number,
                    -beam_deviations[beam_number]["x"],
                    -beam_deviations[beam_number]["y"],
                )
            self.is_h_splay = False

    def set_kaya(self, state):
        if state not in ["on", "off"]:
            raise ValueError("state must be 'on' or 'off'")

        if state == "on":
            self._controllers["controllino"].turn_on("Kaya")
        elif state == "off":
            self._controllers["controllino"].turn_off("Kaya")

    def _validate_move_img_pup_inputs(self, config, beam_number, x, y):
        # input validation
        if beam_number not in [1, 2, 3, 4]:
            raise ValueError("beam_number must be in the range [1, 4]")
        if config not in ["c_red_one_focus", "intermediate_focus", "baldr"]:
            raise ValueError(
                "config must be 'c_red_one_focus' or 'intermediate_focus' or 'baldr'"
            )

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

        uv_commands = asgard_alignment.Engineering.move_img_calc(
            config, beam_number, desired_deviation
        )

        if config == "baldr":
            axis_list = ["BTP", "BTT", "BOTP", "BOTT"]
        else:
            axis_list = ["HTPP", "HTTP", "HTPI", "HTTI"]

        # axis_list = ["HTPP", "HTTP", "HTPI", "HTTI"]

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

        uv_commands = asgard_alignment.Engineering.move_pup_calc(
            config, beam_number, desired_deviation
        )

        if config == "baldr":
            axis_list = ["BTP", "BTT", "BOTP", "BOTT"]
            print(uv_commands)  # bug shooting
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

    def _apply_shutter_state_to_beam(self, state, beam_number):
        dev_prefixes = ["HFO", "HTTP", "HTPP", "HTPI", "HTTI"]
        if state not in ["open", "close"]:
            raise ValueError("state must be 'open' or 'close'")

        is_shuttered = state == "close"

        for dev in dev_prefixes:
            name = f"{dev}{beam_number}"
            if name in self.devices:
                logging.info(f"Shuttering {name} to {is_shuttered}")
                self.devices[name].is_shuttered = is_shuttered

    def h_shut(self, state, beam_numbers):
        """
        Apply an internal "shutter" to Heimdallr devices by tilting the knife edge a known amount away
        from the camera. The direction of the deviation is determined by the state when the shutter is first
        set to close, and this is stored to remove the "shutter" later. Also set all internal
        states of the devices to shuttered, preventing motion in the shuttered state.

        Parameters
        ----------
        state : str
            The desired state of the shutter ("open" or "close").
        beam_numbers : list of int
            The beam numbers to apply the shutter state to.
        """

        offest_mag = 0.15  # degrees
        possible_shutter_devs = ["HTTP", "HTPP"]  # knife edge motor only

        if state == "close":
            beams_to_close = []
            for beam_n in beam_numbers:
                if not self.h_shutter_states[beam_n] == "close":
                    beams_to_close.append(beam_n)
            # if we are closing the shutter, we need to pick a direction for the offsets and apply them
            # pick the direction by choosing the axis that has the largest current value (abs value sense)
            for beam_n in beams_to_close:
                axis_pos = {}
                for dev in possible_shutter_devs:
                    numbered_dev = f"{dev}{beam_n}"
                    if numbered_dev in self.devices:
                        axis_pos[numbered_dev] = self.devices[
                            numbered_dev
                        ].read_position()

                # find the axis with the largest absolute value
                max_dev = max(axis_pos, key=lambda x: abs(axis_pos[x]))

                # apply the offset in the opposite direction
                offset = offest_mag if axis_pos[max_dev] < 0 else -offest_mag
                self.h_shutter_offsets[beam_n][max_dev] = offset

                # wait in a blocking way for the device to stop moving
                timeout = 3.0
                start_time = time.time()
                while self.devices[max_dev].is_moving():
                   logging.info(f"waiting for {max_dev} to stop moving") 
                    if time.time() - start_time > timeout:
                        logging.warning(
                            f"Timeout waiting for {max_dev} to stop moving"
                        )
                        break
                    time.sleep(0.1)

                self.devices[max_dev].move_relative(offset)
                self.h_shutter_states[beam_n] = "closed"
                logging.info(f"sending moverel {offset} to {numbered_dev}")
            for beam_n in beam_numbers:
                self._apply_shutter_state_to_beam(state, beam_n)

        if state == "open":
            beams_to_open = []
            for beam_n in beam_numbers:
                if not self.h_shutter_states[beam_n] == "open":
                    beams_to_open.append(beam_n)
            logging.info(f"beams to open{beams_to_open}")
            for beam_n in beam_numbers:
                self._apply_shutter_state_to_beam(state, beam_n)
            # if we are opening the shutter, we need apply the opposite of the offsets and set the
            # offset variable back to 0.0
            for beam_n in beams_to_open:
                for dev in possible_shutter_devs:
                    numbered_dev = f"{dev}{beam_n}"
                    print(f"{self.h_shutter_offsets[beam_n]}")
                    print(
                        f"{numbered_dev} {self.h_shutter_offsets[beam_n][numbered_dev]}"
                    )
                    if not np.isclose(
                        self.h_shutter_offsets[beam_n][numbered_dev], 0.0
                    ):
                        timeout = 3.0
                        start_time = time.time()
                        while self.devices[max_dev].is_moving():
                        logging.info(f"waiting for {max_dev} to stop moving") 
                            if time.time() - start_time > timeout:
                                logging.warning(
                                    f"Timeout waiting for {max_dev} to stop moving"
                                )
                                break
                            time.sleep(0.1)
                        self.devices[numbered_dev].move_relative(
                            -self.h_shutter_offsets[beam_n][numbered_dev]
                        )
                        logging.info(
                            f"sending moverel {-self.h_shutter_offsets[beam_n][numbered_dev]} to {numbered_dev}"
                        )
                        self.h_shutter_offsets[beam_n][numbered_dev] = 0.0
                        self.h_shutter_states[beam_n] = "open"

        logging.info(f"self.h_shutter_states: {self.h_shutter_states}")

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
            logging.info("not in devices")
            return False

        res = self.devices[axis].ping()

        if not res:
            logging.info("failed")
            # need to remove the connection from dict
            # TODO: include check if it is just the axis or the controller that is down,
            # and remove as needed
            del self.devices[axis]
            return res

        logging.info("success")
        return res

    def _create_phasemask_wrapper(self):
        """
        wraps the phasemask x,y motors into a specific Baldr_phasemask class that has
        unique read/write update commands to update all phasemask positions
        based on the current one. This class is also required as input to phasemask alignment tools.
        """
        for beam in [1, 2, 3, 4]:
            if (f"BMX{beam}" not in self.devices) or (f"BMY{beam}" not in self.devices):
                logging.warning(
                    f"don't have both phasemasks: (BMX in devices = {(f'BMX{beam}' not in self.devices)}, BMY in devices = {(f'BMX{beam}' not in self.devices)}"
                )
                # Prompt the user for input
                user_input = (
                    input("Type 'y' to continue, or 'n' to stop the program: ")
                    .strip()
                    .lower()
                )

                if user_input == "n":
                    logging.info("Stopping the program as requested.")
                    sys.exit(0)  # Exit the program
                elif user_input == "y":
                    logging.info("Continuing the program...")
                else:
                    logging.warning("Invalid input. Assuming continuation.")
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
                    logging.info(
                        f"using most recent file for beam {beam}: {phase_positions_json}"
                    )
                else:
                    logging.warning(f"no phasemask configuration files found in {pth}")
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

    def _open_controllino(self):
        self._controllers["controllino"] = asgard_alignment.controllino.Controllino(
            self._other_config["controllino0"]["ip_address"]
        )

        self._controllers["controllino"].set_PI_loop("Lower", 258, 360, 10)
        self._controllers["controllino"].set_PI_loop("Upper", 255, 360, 10)

        self._controllers["stepper_controllino"] = (
            asgard_alignment.controllino.Controllino(
                self._other_config["controllino1"]["ip_address"]
            )
        )

    def _remove_devices(self, dev_list):
        for k, v in self.devices.items():
            if k in dev_list:
                logging.info(f"Removing {k}")

        logging.info(f"Current devices: {list(self.devices.keys())}")
        self._devices = {k: v for k, v in self.devices.items() if k not in dev_list}
        logging.info(f"After deletion: {list(self.devices.keys())}")

    def _remove_controllers(self, controller_list):
        self._controllers = {
            k: v for k, v in self._controllers.items() if k not in controller_list
        }

    def standby(self, device):
        # TODO: work on a list of devices instead? so that we aren't turning things off
        # and getting many warnings! Or could ESO just send a standby for a single dev knowing this?
        """
        Put the device in standby mode

        This has to be done in the instrument class to correctly encode the electrical connections
        and the motors that are connected to the controllino.

        For any of the X-MCC common connections, the controllino will standby all axes connected after parking them.

        """
        logging.info(f"Attempting to place {device} in standby mode...")

        if device not in self.devices:
            logging.warning(f"{device} not in devices dictionary")
            return "NACK"

        zabers = (
            asgard_alignment.ZaberMotor.ZaberLinearActuator,
            asgard_alignment.ZaberMotor.ZaberLinearStage,
        )

        if isinstance(self.devices[device], zabers):  # Zaber
            logging.info("Request to standby a zaber device")
            # id the wire that powers the controller(s)
            if "BM" in device:
                wire_name = "X-MCC (BMX,BMY)"
                all_devs = [f"BMX{i}" for i in range(1, 5)] + [
                    f"BMY{i}" for i in range(1, 5)
                ]
                controller_connctions = [
                    self._motor_config["BMX1"]["x_mcc_ip_address"],
                    self._motor_config["BMY1"]["x_mcc_ip_address"],
                ]

            elif (
                "BFO" in device or "BDS" in device or "SDL" in device or "SSS" in device
            ):
                wire_name = "X-MCC (BFO,SDL,BDS,SSS)"
                all_devs = (
                    ["BFO"]
                    + ["SDLA", "SDL12", "SDL34"]
                    + [f"BDS{i}" for i in range(1, 5)]
                    + ["SSS"]
                )
                controller_connctions = [
                    self._motor_config["BFO"]["x_mcc_ip_address"],
                    self.find_zaber_usb_port(),  # the usb connections for the BDS
                ]
            else:
                logging.warning(f"{device} not in devices dictionary")
                return

            # park all axes - note unpark is done in zaber ctor
            for dev in all_devs:
                if dev in self.devices:
                    # park the axis
                    self.devices[dev].axis.park()
                else:
                    logging.warning(f"WARN: {dev} not in devices dictionary")

            # turn off the relevant power
            self._controllers["controllino"].turn_off(wire_name)

            # and also delete all device instances
            self._remove_devices(all_devs)

            # close all zaber connections
            for controller in controller_connctions:
                logging.info(f"Closing connection to {controller}")
                self._controllers[controller].close()

            # manage instrument internals to no longer show these connections
            self._remove_controllers(controller_connctions)

        elif isinstance(self.devices[device], asgard_alignment.NewportMotor.LS16PAxis):
            logging.info("Request to standby a LS16P device")
            wire_name = "LS16P (HFO)"
            all_devs = [f"HFO{i}" for i in range(1, 5)]
            logging.info(f"preparing removing {all_devs}")

            controller_connctions = []
            for dev in all_devs:
                sn = self._motor_config[dev]["serial_number"]
                port = self._prev_port_mapping[sn]
                controller_connctions.append(port)

            # turn off the power
            self._controllers["controllino"].turn_off(wire_name)

            # remove the devices
            logging.info(f"actually removing {all_devs}")
            self._remove_devices(all_devs)

            # and the controllers:
            self._remove_controllers(controller_connctions)

        elif isinstance(self.devices[device], asgard_alignment.NewportMotor.M100DAxis):
            logging.info("Request to standby a M100D device")
            # in this case, we will need to switch off all grouped motors
            if "BT" in device:
                # this is like the BTX + HFO row
                wire_names = []
                all_devs = (
                    [f"BTP{i}" for i in range(1, 5)]
                    + [f"BTT{i}" for i in range(1, 5)]
                    + [f"HFO{i}" for i in range(1, 5)]
                )
                usb_command = (
                    f"cusbi /S:{self.managed_usb_port_short} 0:3"  # 0 means off
                )
            elif "HT" in device:
                wire_names = []
                prefixes = ["HTTP", "HTPI", "HTPP", "HTTI"]
                all_devs = []
                for prefix in prefixes:
                    for i in range(1, 5):
                        all_devs.append(f"{prefix}{i}")
                usb_command = (
                    f"cusbi /S:{self.managed_usb_port_short} 0:1"  # 0 means off
                )
            elif "BOT" in device:
                wire_names = []
                all_devs = [f"BOTP{i}" for i in range(2, 5)] + [
                    f"BOTT{i}"
                    for i in range(2, 5)  # noting that BOTX is only beams 2-4
                ]
                usb_command = f"cusbi /S:{self.managed_usb_port_short} 0:2"

            controller_connctions = []
            for dev in all_devs:
                sn = self._motor_config[dev]["serial_number"]
                port = self._prev_port_mapping[sn]
                controller_connctions.append(port)

            wire_names = set(wire_names)
            # usb_commands = set(usb_command)

            logging.info(f"Turning off wires: {wire_names}")
            logging.info(f"Sending USB command: {usb_command}")

            # turn off the USB
            # for cmd in usb_commands:
            os.system(usb_command)

            logging.info("USB command sent")

            # turn off the power
            for wire_name in wire_names:
                self._controllers["controllino"].turn_off(wire_name)
            logging.info("Power turned off")

            # remove the devices
            self._remove_devices(all_devs)

            # and the controllers:
            self._remove_controllers(controller_connctions)
            logging.info("Devices and controllers removed")
            return "ACK"

        else:
            # just remove the device from the list
            self._remove_devices([device])

            logging.info(f"{device} is now in standby mode.")
            return "ACK"

    def home_steppers(self, dev_list, blocking=True):
        """
        Home the steppers in the given list of devices.
        """
        for dev in dev_list:
            if dev in self.devices:
                self.devices[dev].home()
            else:
                logging.warning(f"WARN: {dev} not found in device list")

        # block until homed if needed
        if blocking:
            # use is_homed to check if the device is homed for all devices
            all_homed = False
            while not all_homed:
                all_homed = True
                for dev in dev_list:
                    if dev in self.devices:
                        if not self.devices[dev].is_homed():
                            all_homed = False
                            logging.info(f"{dev} is not homed yet, waiting...")
                    else:
                        logging.warning(f"WARN: {dev} not found in device list")
                if not all_homed:
                    time.sleep(0.5)

    def online(self, dev_list):

        devs = []
        for dev in dev_list:
            if dev not in self._motor_config:
                logging.warning(f"WARN: {dev} not in motor config")
                continue
            if dev in self.devices:
                logging.info(f"{dev} is already online")
                continue
            devs.append(dev)

        dev_list = devs
        if len(dev_list) == 0:
            logging.info("No devices to turn on")
            return

        # turn on any nessecary power supplies
        wire_list = []
        usb_commands = []
        for dev in dev_list:
            if "BM" in dev:
                wire_name = "X-MCC (BMX,BMY)"
                wire_list.append(wire_name)
            elif "BFO" in dev or "BDS" in dev or "SDL" in dev:
                wire_name = "X-MCC (BFO,SDL,BDS,SSS)"
                wire_list.append(wire_name)
            elif "HFO" in dev or "BT" in dev:
                wire_name = "LS16P (HFO)"
                wire_list.append(wire_name)
                wire_name = "USB hubs"
                wire_list.append(wire_name)
                usb_commands.append(
                    f"cusbi /S:{self.managed_usb_port_short} 1:3"
                )  # 1 means on, 3 means HFO
            elif "HT" in dev:
                wire_name = "USB hubs"
                wire_list.append(wire_name)
                usb_commands.append(
                    f"cusbi /S:{self.managed_usb_port_short} 1:1"
                )  # 1 means on, 1 means HT
            elif "BOT" in dev:
                usb_commands.append(
                    f"cusbi /S:{self.managed_usb_port_short} 1:2"
                )  # 1 means on, 2 means BOT

        wire_list = set(wire_list)
        usb_commands = set(usb_commands)

        logging.info(f"Turning on wires: {wire_list}")

        for wire in set(wire_list):
            self._controllers["controllino"].turn_on(wire)

        logging.info(f"Sending USB commands: {usb_commands}")

        for usb_command in usb_commands:
            os.system(usb_command)
            time.sleep(0.1)

        time.sleep(5.0)

        # reconnect all
        self._prev_port_mapping = self.compute_serial_to_port_map()
        self._prev_zaber_port = self.find_zaber_usb_port()
        for name in dev_list:
            res = self._attempt_to_open(name, recheck_ports=False)
            if res:
                logging.info(f"Successfully connected to {name}")
            else:
                logging.warning(f"WARN: Could not connect to {name}")

        # add phasemask compound_devices (registeres phasemask positions withj added functionality to update them)
        self._create_phasemask_wrapper()

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
                logging.info(f"Successfully connected to {name}")
            else:
                logging.warning(f"WARN: Could not connect to {name}")

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
                logging.warning(
                    f"Failed to connect to DM with serial number {serial_number}"
                )
                return False
            flat_map_file = self._motor_config[name]["flat_map_file"]
            flat_map = pd.read_csv(flat_map_file, header=None)[0].values
            cross_map = pd.read_csv("DMShapes/Crosshair140.csv", header=None)[0].values
            self._devices[name] = {
                "dm": dm,
                "flat_map": flat_map,
                "cross_map": cross_map,
            }
            # print(f"Connected to {name} with serial number {serial_number}")
            return True

        if self._motor_config[name]["motor_type"] in ["M100D", "LS16P"]:
            # this is a newport motor USB connection, create a newport motor
            # object
            cfg = self._motor_config[name]

            if recheck_ports:
                self._prev_port_mapping = self.compute_serial_to_port_map()
                logging.info(f"New port mapping: {self._prev_port_mapping}")

            if cfg["serial_number"] not in self._prev_port_mapping:
                logging.warning("WARN: Could not find serial number in port mapping")
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
                try:
                    self._controllers[cfg["x_mcc_ip_address"]] = Connection.open_tcp(
                        cfg["x_mcc_ip_address"]
                    )
                except Exception as e:
                    logging.warning(e)
                    return False
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

            if ("BMX" in name) or ("BMY" in name):
                if "BMX" in name:
                    beam_id_tmp = name.split("BMX")[-1]
                if "BMY" in name:
                    beam_id_tmp = name.split("BMY")[-1]
                phasemask_folder_path = f"/home/asg/Progs/repos/asgard-alignment/config_files/phasemask_positions/beam{beam_id_tmp}/"
                phasemask_files = glob.glob(
                    os.path.join(phasemask_folder_path, "*.json")
                )
                recent_phasemask_file = max(
                    phasemask_files, key=os.path.getmtime
                )  # most recently created
                with open(recent_phasemask_file, "r", encoding="utf-8") as pfile:
                    positions_tmp = json.load(pfile)
                if "BMX" in name:
                    oneAxis_dict = {
                        key: value[0] for key, value in positions_tmp.items()
                    }
                elif "BMY" in name:
                    oneAxis_dict = {
                        key: value[1] for key, value in positions_tmp.items()
                    }

                self._devices[name] = asgard_alignment.ZaberMotor.ZaberLinearActuator(
                    name,
                    cfg["semaphore_id"],
                    axis,
                    named_positions=oneAxis_dict,
                )
            else:
                if "named_pos" in self._motor_config[name]:
                    named_positions = self._motor_config[name]["named_pos"]
                else:
                    named_positions = None

                self._devices[name] = asgard_alignment.ZaberMotor.ZaberLinearActuator(
                    name,
                    cfg["semaphore_id"],
                    axis,
                    named_positions=named_positions,
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
                logging.info(f"Zaber port found at: {self._prev_zaber_port}")

            if self._prev_zaber_port is None:
                return False

            if self._prev_zaber_port not in self._controllers:
                self._controllers[self._prev_zaber_port] = Connection.open_serial_port(
                    self._prev_zaber_port
                )
                self._zaber_detected_devs = None

            if self._zaber_detected_devs is None or recheck_ports:
                self._zaber_detected_devs = self._controllers[
                    self._prev_zaber_port
                ].detect_devices()

            for dev in self._zaber_detected_devs:
                if dev.serial_number == self._motor_config[name]["serial_number"]:
                    dev.settings.set("system.led.enable", 0)
                    self._devices[name] = asgard_alignment.ZaberMotor.ZaberLinearStage(
                        name,
                        self._motor_config[name]["semaphore_id"],
                        dev,
                        named_positions=self._motor_config[name]["named_pos"],
                    )
                    return True
        elif self._motor_config[name]["motor_type"] in ["8893KM"]:
            self.devices[name] = asgard_alignment.CustomMotors.Flip8893KM(
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
        elif self._motor_config[name]["motor_type"] in ["GD40Z"]:
            self.devices[name] = asgard_alignment.CustomMotors.GD40Z(
                name,
                self._motor_config[name]["semaphore_id"],
                self._controllers["stepper_controllino"],
            )
            return True

    @staticmethod
    def find_managed_USB_hub_port():
        ports = serial.tools.list_ports.comports()

        for port, _, hwid in sorted(ports):
            if "SER=B001DGUX" in hwid:  # the serial for managed hub
                return port
        return None

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

        # First, we need the list of cusbi ports. If we don't do this, the next time we query,
        # we get an error. I've put this as a fake list for now.
        # Command to find is cusbi (in ~/bin). Syntax
        # ./cusbi /Q:ttyUSB0
        # If we query all ports, then some Newport M100D ports get confused.
        # See:
        # https://sgcdn.startech.com/005329/media/sets/USB_Admin_Software_Manual/USB_Hub_Admin_Software_Manual.pdf
        # If we try to open a Zaber connection to this port, then it seems stuck in an error state.
        # See:
        # https://github.com/pyserial/pyserial/issues/678
        cusb_ports = ["/dev/ttyUSBX"]

        for port, _, hwid in sorted(ports):
            if "SER=B001DGUX" in hwid:  # the serial for managed hub
                if port in cusb_ports:
                    logging.info("Found a Managed USB hub.")
                    continue
            if "SER=AB0NSCTM" in hwid:
                return port

            # # # Try to open it - if we can, it is a Zaber port!
            # test_connection = Connection.open_serial_port(port)
            # try:
            #     devices = test_connection.detect_devices()
            #     test_connection.close()
            #     print(f"Found a Zaber USB port {port}")
            #     return port
            # except:
            #     # This next line is essential, or the port remains open and no other process
            #     # can use it (including MDS later in the code)
            #     test_connection.close()
            #     print(f"A non-zaber motor using the same USB ID. Port {port}")
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
                    logging.warning(f"Could not connect to {device}: {e}")

        return mapping


if __name__ == "__main__":
    pth = "motor_info_full_system.json"
    # Instrument._validate_config_file(pth)

    # print(Instrument.compute_serial_to_port_map())

    # config = Instrument._read_motor_config(pth)

    instr = Instrument(pth)

    print(instr.devices["HTPP1"].read_position())
    print(instr.devices["HTTP1"].read_position())


class TemperatureSummary:
    temp_probes = [
        "Lower T",
        "Upper T",
        "Bench T",
        "Floor T",
    ]

    # control loop info
    PI_infos_of_interest = [
        "m_pin_val",
        "setpoint",
        "integral",
        "k_prop",
        "k_int",
    ]

    servo_names = ["Lower", "Upper"]

    def __init__(self, controllino):
        self.controllino = controllino

    def get_temp_keys(self):
        keys = []
        keys += self.temp_probes

        for servo in self.servo_names:
            for info in self.PI_infos_of_interest:
                keys.append(f"{servo} {info}")

        return keys

    @staticmethod
    def raw_to_celsius(raw_value):
        return 42.5 + (float(raw_value) - 512) * 0.11

    def get_temp_status(self, probes_only=False, raw_temps=True):
        temps = []
        for probe in self.temp_probes:
            try:
                temp = self.controllino.analog_input(probe)
                temps.append(float(temp))
            except Exception as e:
                print(f"Error getting temperature for {probe}: {e}")
                temps.append(None)

        if not raw_temps:
            temps = [self.raw_to_celsius(t) if t is not None else None for t in temps]

        if probes_only:
            return temps

        PI_infos = []
        for i, servo in enumerate(self.servo_names):
            try:
                info = self.controllino.read_PI_loop_info(servo)
                for key in TemperatureSummary.PI_infos_of_interest:
                    PI_infos.append(info[key])

            except Exception as e:
                print(f"Error getting PI info for {servo}: {e}")
                PI_infos.append(None)

        return temps + PI_infos
