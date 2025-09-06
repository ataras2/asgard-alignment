import zmq
import asgard_alignment
import argparse
import sys
from parse import parse
import time

import os

import json
import datetime

# deepcopy
from copy import deepcopy

import enum
import asgard_alignment.ESOdevice
import asgard_alignment.Instrument
import asgard_alignment.MultiDeviceServer
import asgard_alignment.Engineering
import asgard_alignment.NewportMotor
import asgard_alignment.controllino
import asgard_alignment.ESOdevice

import logging

# guarantees that errors are logged
def handle_exception(exc_type, exc_value, exc_traceback):
    if issubclass(exc_type, KeyboardInterrupt):
        sys.__excepthook__(exc_type, exc_value, exc_traceback)
        return
    logging.error("Uncaught exception", exc_info=(exc_type, exc_value, exc_traceback))


sys.excepthook = handle_exception


class MockMDS:
    def __init__(self):
        pass

    def handle_zmq(self, message):
        logging.info(f"Received message: {message}")
        return "Dummy response"


class MultiDeviceServer:
    """
    A class to run the Instrument MDS.
    """

    DATABASE_MSG_TEMPLATE = {
        "command": {
            "name": "write",
            "time": "YYYY-MM-DDThh:mm:ss",
            "parameters": [],
        }
    }

    def __init__(self, port, host, config_file):
        self.port = port
        self.host = host
        self.config_file = config_file

        self.context = zmq.Context()
        self.server = self.context.socket(zmq.REP)
        self.server.bind(f"tcp://{self.host}:{self.port}")

        self.poller = zmq.Poller()
        self.poller.register(self.server, zmq.POLLIN)
        self.batch = 0

        self.db_update_socket = self.context.socket(zmq.PUSH)
        self.db_update_socket.connect("tcp://wag:5561")

        self._reset_setup_ls()
        self.batch = 0
        self.is_stopped = True

        self.database_message = self.DATABASE_MSG_TEMPLATE.copy()

        if config_file == "mock":
            self.instr = MockMDS()
        else:
            self.instr = asgard_alignment.Instrument.Instrument(self.config_file)

        logging.info("Instrument all set up, ready to accept messages")

    def socket_funct(self, s):
        try:
            message = s.recv_string()
            return message
        except zmq.ZMQError as e:
            logging.error(f"ZMQ Error: {e}")
            return -1

    def log(self, message):
        logging.info(message)

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
                    logging.info(f"Received message: {data}")
                    is_custom_msg, response = self.handle_message(data)
                    if response == -1:
                        running = False
                        if s == sys.stdin:
                            self.log("Manually shut down. Goodbye.")
                        else:
                            self.log("Shut down by remote connection. Goodbye.")
                    else:
                        if response is None:
                            response = ""
                        # if is_custom_msg:
                        s.send_string(response + "\n")

    @staticmethod
    def get_time_stamp():
        # time_now = datetime.datetime.now()
        # time_now = time.gmtime()
        # return time.strftime("%Y-%m-%dT%H:%M:%S", time_now)

        # Get the current UTC time
        current_utc_time = datetime.datetime.utcnow()

        # Format the UTC time
        return current_utc_time.strftime("%Y-%m-%dT%H:%M:%S")

    def _reset_setup_ls(self):
        self.setup_ls = [[], []]

    def check_if_batch_done(self):
        is_done = True
        for dev in self.setup_ls[self.batch]:
            logging.info(f"Checking if {dev.device_name} is moving... ")
            is_moving = self.instr.devices[dev.device_name].is_moving()
            logging.info(f"Value is {is_moving}")
            if is_moving == True:
                is_done = False
                break
        return is_done

    def handle_message(self, message):
        """
        Handles a recieved message. Custom messages are indicated by lowercase commands
        """

        # if "!" in message:
        if message[0].islower():
            logging.info(f"Custom command: {message}")
            return True, self._handle_custom_command(message)

        if message[0] == "!":
            logging.info("Old custom command")
            return True, "NACK: Are you using old custom commands?"

        try:
            # message = message.rstrip(message[-1])
            json_data = json.loads(json.loads(message.strip()))
            logging.info(f"ESO msg recv: {json_data} (type {type(json_data)})")
        except:
            logging.error("Error: Invalid JSON message")
            return False, "NACK: Invalid JSON message"
        command_name = json_data["command"]["name"]
        time_stampIn = json_data["command"]["time"]

        # Acceptable window: ±5 minutes from current UTC time
        try:
            received_time = datetime.datetime.strptime(
                time_stampIn, "%Y-%m-%dT%H:%M:%S"
            )
            now_utc = datetime.datetime.utcnow()
            delta = abs((now_utc - received_time).total_seconds())
            if delta > 300:  # 5 minutes
                logging.warning(
                    f"Received time-stamp {time_stampIn} is out of range (delta={delta}s)"
                )
                command_name = "none"
        except Exception as e:
            logging.error(f"Invalid time-stamp format: {time_stampIn} ({e})")
            command_name = "none"

        reply = {
            "reply": {
                "content": "????",
                "time": "YYYY-MM-DDThh:mm:ss",
                "parameters": [],
            }
        }

        # Verification of received time-stamp (TODO)
        # If the time_stamp is invalid, set command_name to "none",
        # so no command will be processed but a reply will be sent
        # back to the client (set reply to "ERROR")

        ################################
        # Process the received command:
        ################################

        # Case of "online" (sent by wag when bringing ICS online, to check
        # that MCUs are alive and ready)

        self.database_message["command"]["parameters"].clear()

        # Case of "setup" (sent by wag to move devices)
        if "setup" in command_name:
            self.is_stopped = False
            n_devs_commanded = len(json_data["command"]["parameters"])

            semaphore_array = [0] * 100  # TODO: implement this maximum correctly
            # Create a double-list of devices to move
            self._reset_setup_ls()
            for i in range(n_devs_commanded):
                kwd = json_data["command"]["parameters"][i]["name"]
                val = json_data["command"]["parameters"][i]["value"]
                logging.info(f"Setup: {kwd} to {val}")

                # Keywords are in the format: INS.<device>.<motion type>
                prefixes = kwd.split(".")
                dev_name = prefixes[1]
                motion_type = prefixes[2]
                logging.info(f"Device: {dev_name} - motion type: {motion_type}")

                # motion_type can be one of these words:
                # NAME   = Named position (e.g., IN, OUT, J1, H3, ...)
                # ENC    = Absolute encoder position
                # ENCREL = Relative encoder postion (can be negative)
                # ST     = State. Given value is equal to either T or F.
                #          if device is shutter: T = open, F = closed.
                #          if device is lamp: T = on, F = off.

                # Look if device exists in list
                # (something should be done if device does not exist)
                device = self.instr.devices[dev_name]
                semaphore_id = device.semaphore_id
                if semaphore_array[semaphore_id] == 0:
                    # Semaphore is free =>
                    # Device can be moved now
                    self.setup_ls[0].append(
                        asgard_alignment.ESOdevice.SetupCommand(
                            dev_name, motion_type, val
                        )
                    )
                    semaphore_array[semaphore_id] = 1
                else:
                    # Semaphore is already taken =>
                    # Device will be moved in a second batch
                    self.setup_ls[1].append(
                        asgard_alignment.ESOdevice.SetupCommand(
                            dev_name, motion_type, val
                        )
                    )

            # Move devices (if two batches, move the first one)
            self.batch = 0
            if len(self.setup_ls[self.batch]) > 0:
                logging.info(f"batch {self.batch} of devices to move:")
                reply["reply"]["parameters"].clear()
                for s in self.setup_ls[self.batch]:
                    logging.info(
                        f"Moving: {s.device_name} to: {s.value} ( setting {s.motion_type} )"
                    )

                    self.instr.devices[s.device_name].setup(s.motion_type, s.value)

                    # Inform wag ICS that the device is moving
                    attribute = "<alias>" + s.device_name + ":DATA.status0"
                    reply["reply"]["parameters"].append(
                        {"attribute": attribute, "value": "MOVING"}
                    )
            # Once setup is forwarded to the devices, reply OK if everything is
            # normal. This means that the setup has started, no that it is done!
            reply["reply"]["content"] = "OK"

        # Case of "poll" (sent by wag to get the status of the
        # last setup sent. Normally, wag sends a "poll" every
        # second during a setup)

        elif "poll" in command_name:
            # --------------------------------------------------
            # TODO: Add here call to query the status of the batch of
            # devices that is concerned by the last setup command
            # If they all reach the target position or if
            # a STOP command occured, set is_batch_done to 1
            #
            # In this example of back-end server, we simulate
            # that by checking the cntdwnSetup variable
            # --------------------------------------------------
            is_batch_done = self.check_if_batch_done()

            reply["reply"]["parameters"].clear()
            if len(self.setup_ls[self.batch]) > 0:
                for s in self.setup_ls[self.batch]:
                    attribute = "<alias>" + s.device_name + ":DATA.status0"
                    # Case of motor with named position requested
                    if s.motion_type == "NAME":
                        # If motor reached the position, we set the
                        # attribute to the target named position
                        # (given in the setup) otherwise we set it
                        # to MOVING
                        if is_batch_done:
                            reply["reply"]["parameters"].append(
                                {"attribute": attribute, "value": s.value}
                            )
                        else:
                            reply["reply"]["parameters"].append(
                                {"attribute": attribute, "value": "MOVING"}
                            )

                        # Note: normally the encoder position shall be
                        # reported along with the named position
                        # ...............................................
                        # => Call function to read the encoder position
                        #    store it in a variable "posEnc" and execute:
                        #
                        # attribute = "<alias>" + s.device_name +":DATA.posEnc"
                        # dbMsg['command']['parameters'].\
                        # append({"attribute":attribute, "value":posEnc})

                    # Case of shutter or lamp
                    if s.motion_type == "ST":
                        # Here the device can be either a lamp or a shutter
                        # Add here code to find out the type of s.device_name

                        if isinstance(s.device_name, asgard_alignment.ESOdevice.Lamp):
                            value_map = {"T": "ON", "F": "OFF"}
                        elif isinstance(
                            s.device_name, asgard_alignment.ESOdevice.Motor
                        ):
                            value_map = {"T": "OPEN", "F": "CLOSED"}
                        else:
                            logging.error(
                                f"Device {s.device_name} is not lamp or shutter"
                            )
                            continue

                        # If it is a shutter do:
                        if is_batch_done:
                            reply["reply"]["parameters"].append(
                                {"attribute": attribute, "value": value_map[s.value]}
                            )
                        else:
                            reply["reply"]["parameters"].append(
                                {"attribute": attribute, "value": "MOVING"}
                            )

                    # Case of motor with absolute encoder position requested
                    if s.motion_type == "ENC":
                        if is_batch_done:
                            reply["reply"]["parameters"].append(
                                {"attribute": attribute, "value": ""}
                            )
                        else:
                            reply["reply"]["parameters"].append(
                                {"attribute": attribute, "value": "MOVING"}
                            )

                        # Note: if motor is at limit, do:
                        # dbMsg['command']['parameters'].append({"attribute":attribute, "value":"LIMIT"})
                        # Report the absolute encoder position
                        # Here (simulation), we simply use the target
                        # position (even if the motor is supposed to move)
                        attribute = "<alias>" + s.device_name + ":DATA.posEnc"
                        reply["reply"]["parameters"].append(
                            {"attribute": attribute, "value": s.value}
                        )
                        # Case of motor with relative encoder position
                        # not considered yet
                        # The simplest would be to read the encoder position
                        # and to update the database as for the previous case

            # Check if second batch remains to setup
            # (if no STOP command has been sent)
            if is_batch_done:
                if (
                    (self.batch == 0)
                    and (len(self.setup_ls[1]) > 0)
                    and (not self.is_stopped)
                ):
                    self.batch = 1
                    logging.info(f"batch {self.batch} of devices to move:")
                    for s in self.setup_ls[self.batch]:
                        logging.info(
                            f"Moving: {s.device_name} to: {s.value} ( setting {s.motion_type} )"
                        )
                        self.instr.devices[s.device_name].setup(s.motion_type, s.value)

                        # Inform wag ICS that the device is moving
                        attribute = "<alias>" + s.device_name + ":DATA.status0"
                        reply["reply"]["parameters"].append(
                            {"attribute": attribute, "value": "MOVING"}
                        )

                    reply["reply"]["content"] = "PENDING"
                else:
                    # All batches of setup are done
                    reply["reply"]["content"] = "DONE"
            else:
                reply["reply"]["content"] = "PENDING"

        # Case of sensor reading request
        elif "read" in command_name:
            reply["reply"]["parameters"].clear()
            temps = self.instr.temp_summary.get_temp_status(
                probes_only=True, raw_temps=False
            )

            for t in temps:
                reply["reply"]["parameters"].append({"value": t})

            reply["reply"]["content"] = "OK"

        # Case of other commands. The parameters are either a list
        # of devices, or "all" to apply the command to all the devices
        else:
            reply["reply"]["parameters"].clear()
            n_devs_commanded = len(json_data["command"]["parameters"])
            is_all_devs = False
            # Check if command applies to all the existing devices
            if (n_devs_commanded == 1) and (
                json_data["command"]["parameters"][0]["device"] == "all"
            ):
                n_devs_commanded = len(self.instr.devices)  # total number of devices
                is_all_devs = True
                dev_names = list(self.instr.devices.keys())
            else:
                dev_names = [
                    json_data["command"]["parameters"][i]["device"]
                    for i in range(n_devs_commanded)
                ]

            # if online, it is more efficient for instrument to do it in a batch call
            if command_name == "online":
                self.instr.online(dev_names)

                for dev_name in dev_names:
                    attribute = "<alias>" + dev_name + ".state"
                    reply["reply"]["parameters"].append(
                        {"attribute": attribute, "value": 3}
                    )

            # standby is also a weird case, as standing by some devices shuts off others - need to iterate
            if command_name == "standby":
                for dev_name in dev_names:
                    attribute = "<alias>" + dev_name + ".state"
                    reply["reply"]["parameters"].append(
                        {"attribute": attribute, "value": 2}
                    )

                devs_to_standby = dev_names.copy()
                while len(devs_to_standby) > 0:
                    logging.info(f"Standing by device: {devs_to_standby[0]}")
                    self.instr.standby(devs_to_standby[0])

                    devs_to_standby = list(
                        set(self.instr.devices.keys()).intersection(devs_to_standby)
                    )

            # for all other commands, do them one device at a time...
            for i in range(n_devs_commanded):
                if is_all_devs:
                    dev_name = dev_names[i]
                else:
                    dev_name = json_data["command"]["parameters"][i]["device"].upper()

                if command_name == "disable":
                    logging.info(f"Power off device: {dev_name}")

                    self.instr.devices[dev_name].disable()

                elif command_name == "enable":
                    logging.info(f"Power on device: {dev_name}")

                    self.instr.devices[dev_name].enable()

                elif command_name == "off":
                    logging.info(f"Turning off device: {dev_name}")
                    # .........................................................
                    # If needed, call controller-specific functions to power
                    # down the device. It may require initialization
                    # after a power up
                    # .........................................................

                    # Update the wagics database to show that the device is
                    # in LOADED state (value of "state" attribute has to be
                    # set to 3)

                    attribute = "<alias>" + dev_name + ".state"
                    reply["reply"]["parameters"].append(
                        {"attribute": attribute, "value": 1}
                    )

                elif command_name == "simulat":
                    logging.info(f"Simulation of device {dev_name}")
                    # Set the simulation flag of dev_name to 1
                    # TODO: add code here that changes the device to simulation mode
                    # for devIdx in range(nbCtrlDevs):
                    #     if d[devIdx].name == dev_name:
                    #         break
                    # d[devIdx].simulated = 1

                    # Update the wagics database  to show that the device
                    # is in simulation and is in LOADED state

                    attribute = "<alias>" + dev_name + ".simulation"
                    reply["reply"]["parameters"].append(
                        {"attribute": attribute, "value": 1}
                    )
                    attribute = "<alias>" + dev_name + ".state"
                    reply["reply"]["parameters"].append(
                        {"attribute": attribute, "value": 1}
                    )

                elif command_name == "stop":
                    logging.info(f"Stop device: {dev_name}")
                    self.instr.devices[dev_name].stop()

                    # If setup is in progress, consider it done

                    # Update of the device status (positions, etc...) will be
                    # done by the next "poll" command sent by wag

                elif command_name == "stopsim":
                    logging.info(f"Normal mode for device {dev_name}")
                    # Set the simulation flag of dev_name to 0
                    # TODO: add code here that changes the device to normal mode

                    # Update the wagics database  to show that the device
                    # is not in simulation and is in LOADED state
                    # (it may require an initialization when going to
                    # ONLINE state)

                    attribute = "<alias>" + dev_name + ".simulation"
                    reply["reply"]["parameters"].append(
                        {"attribute": attribute, "value": 0}
                    )
                    attribute = "<alias>" + dev_name + ".state"
                    reply["reply"]["parameters"].append(
                        {"attribute": attribute, "value": 1}
                    )

            if command_name == "stop":
                self.is_stopped = True

            reply["reply"]["content"] = "OK"

        # Send back reply to ic0fb process (wag)

        timeNow = datetime.datetime.now()
        timeStamp = timeNow.strftime("%Y-%m-%dT%H:%M:%S")
        reply["reply"]["time"] = timeStamp

        # Convert reply JSON structure into a character string
        # terminated with null character (because ic0fb process on wag
        # in coded in C++ and needs null character to mark end of the string)

        repMsg = json.dumps(reply) + "\0"
        print(repMsg)
        # self.server.send_string(repMsg)

        return False, repMsg

    def _handle_custom_command(self, message):
        # this is a custom command, acutally do useful things here lol
        def read_msg(axis):
            return str(self.instr.devices[axis].read_position())

        def stop_msg(axis):
            return str(self.instr.devices[axis].stop())

        def moveabs_msg(axis, position):
            self.instr.devices[axis].move_abs(float(position))
            return "ACK"

        def connected_msg(axis):
            return "connected" if axis in self.instr.devices else "not connected"

        def connect_msg(axis):
            # this is a connection open request
            logging.info(f"attempting open connection to {axis}")
            res = self.instr._attempt_to_open(axis, recheck_ports=True)
            logging.info(f"attempted to open {axis} with result {res}")

            return "connected" if axis in self.instr.devices else "not connected"

        def home_steppers_msg(motor):
            if motor == "all":
                motor = list(asgard_alignment.controllino.STEPPER_NAME_TO_NUM.keys())
            else:
                motor = [motor]

            self.instr.home_steppers(motor)

        def init_msg(axis):
            self.instr.devices[axis].init()
            return "ACK"

        def tt_step_msg(axis, n_steps):
            """
            Move the tip-tilt stage by n_steps.
            """
            if "HT" not in axis:
                raise ValueError(f"{axis} is not a valid tip-tilt stage")

            if axis not in self.instr.devices:
                raise ValueError(f"{axis} not found in instrument")

            n_steps = int(n_steps)

            self.instr.devices[axis].move_stepping(n_steps)

        def tt_config_step_msg(axis, step_size):
            if "HT" not in axis:
                raise ValueError(f"{axis} is not a valid tip-tilt stage")
            self.instr.devices[axis].config_step_size(int(step_size))

        def moverel_msg(axis, position):
            logging.info(f"moverel {axis} {position}")
            self.instr.devices[axis].move_relative(float(position))
            return "ACK"

        def state_msg(axis):
            return self.instr.devices[axis].read_state()

        def save_msg(subset, fname):
            if subset.lower() not in ["heimdallr", "baldr", "solarstein", "all"]:
                return "NACK: Invalid subset, must be 'heimdallr', 'baldr', 'solarstein' or 'all'"

            return self.instr.save(subset.lower(), fname)

        def ping_msg(axis):
            res = self.instr.ping_connection(axis)

            if res:
                return "ACK: connected"
            else:
                return "NACK: not connected"

        def health_msg():
            """
            check the health of the whole instrument, and return a json list of dicts
            to make a table, with columns
            - axis name,
            - motor type,
            - connected,
            - state,
            """

            health = self.instr.health()

            # convert to string
            health_str = json.dumps(health)

            return health_str

        def mv_img_msg(config, beam_number, x, y):
            try:
                res = self.instr.move_image(config, int(beam_number), x, y)
            except ValueError as e:
                return f"NACK: {e}"

            if res:
                return "ACK: moved"
            else:
                return "NACK: not moved"

        def mv_pup_msg(config, beam_number, x, y):
            logging.info(f"{beam_number} {type(beam_number)}")
            try:
                res = self.instr.move_pupil(config, int(beam_number), x, y)
            except ValueError as e:
                return f"NACK: {e}"

            if res:
                return "ACK: moved"
            else:
                return "NACK: not moved"

        def on_msg(lamp_name):
            self.instr.devices[lamp_name].turn_on()
            return "ACK"

        def off_msg(lamp_name):
            self.instr.devices[lamp_name].turn_off()
            return "ACK"

        def is_on_msg(lamp_name):
            return str(self.instr.devices[lamp_name].is_on())

        def reset_msg(axis):
            try:
                self.instr.devices[axis].reset()
                return "ACK"
            except Exception as e:
                return f"NACK: {e}"

        def asg_setup_msg(axis, mtype, value):
            try:
                # if it is a float, convert it to a python float
                try:
                    value = float(value)
                except ValueError:
                    pass
                self.instr.devices[axis].setup(mtype, value)
                return "ACK"
            except Exception as e:
                return f"NACK: {e}"

        def apply_flat_msg(dm_name):
            if dm_name not in self.instr.devices:
                return f"NACK: DM {dm_name} not found"

            # Retrieve the DM instance and its flat map
            dm_device = self.instr.devices[dm_name]
            # dm = dm_device["dm"]
            # flat_map = dm_device["flat_map"]

            # Apply the flat map to the DM
            dm_device["dm"].send_data(dm_device["flat_map"])

            logging.info(f"Flat map applied to {dm_name}")
            return f"ACK: Flat map applied to {dm_name}"

        def apply_cross_msg(dm_name):
            if dm_name not in self.instr.devices:
                return f"NACK: DM {dm_name} not found"

            # Retrieve the DM instance and its flat map
            dm_device = self.instr.devices[dm_name]
            # dm = dm_device["dm"]
            # flat_map = dm_device["flat_map"]

            # Apply the flat map to the DM
            dm_device["dm"].send_data(
                dm_device["flat_map"] + 0.3 * dm_device["cross_map"]
            )

            logging.info(f"Cross map applied to {dm_name}")
            return f"ACK: Cross map applied to  {dm_name}"

        def fpm_get_savepath_msg(axis):
            device = self.instr.all_devices[axis]
            if device is None:
                return f"NACK: Axis {axis} not found"
            else:
                return device.savepath
            # if axis not in self.instr.devices:
            #     return f"NACK: Axis {axis} not found"
            # else:
            #     return self.instr.devices[axis].savepath

        def fpm_mask_positions_msg(axis):
            device = self.instr.all_devices[axis]
            if device is None:
                return f"NACK: Axis {axis} not found"
            else:
                return device.mask_positions
            # if axis not in self.instr.devices:
            #     return f"NACK: Axis {axis} not found"
            # else:
            #     return self.instr.devices[axis].mask_positions

        def fpm_update_position_file_msg(axis, filename):
            device = self.instr.all_devices[axis]
            if device is None:
                return f"NACK: Axis {axis} not found"
            else:
                device.update_position_file(filename)
                return "ACK"
            # if axis not in self.instr.devices:
            #     return f"NACK: Axis {axis} not found"
            # else:
            #     self.instr.devices[axis].update_position_file(filename)
            #     return "ACK"

        def fpm_move_to_phasemask_msg(axis, maskname):
            device = self.instr.all_devices[axis]
            if device is None:
                return f"NACK: Axis {axis} not found"
            else:
                device.move_to_mask(maskname)
                return "ACK"
            # if axis not in self.instr.devices:
            #     return f"NACK: Axis {axis} not found"
            # else:
            #     self.instr.devices[axis].move_to_mask(maskname)
            #     return "ACK"

        def fpm_move_relative_msg(axis, new_pos):
            device = self.instr.all_devices[axis]
            if device is None:
                return f"NACK: Axis {axis} not found"
            else:
                device.move_relative(new_pos)
                return "ACK"
            # if axis not in self.instr.devices:
            #     return f"NACK: Axis {axis} not found"
            # else:
            #     self.instr.devices[axis].move_relative(new_pos)
            #     return "ACK"

        def fpm_move_absolute_msg(axis, new_pos):
            device = self.instr.all_devices[axis]
            if device is None:
                return f"NACK: Axis {axis} not found"
            else:
                device.move_absolute(new_pos)
                return "ACK"
            # if axis not in self.instr.devices:
            #     return f"NACK: Axis {axis} not found"
            # else:
            #     self.instr.devices[axis].move_absolute(new_pos)
            #     return "ACK"

        def fpm_read_position_msg(axis):
            device = self.instr.all_devices[axis]
            if device is None:
                return f"NACK: Axis {axis} not found"
            else:
                return device.read_position()
            # if axis not in self.instr.devices:
            #     return f"NACK: Axis {axis} not found"
            # else:
            #     return self.instr.devices[axis].read_position()

        def fpm_update_mask_position_msg(axis, mask_name):
            device = self.instr.all_devices[axis]
            if device is None:
                return f"NACK: Axis {axis} not found"
            else:
                device.update_mask_position(mask_name)
                return "ACK"
            # if axis not in self.instr.devices:
            #     return f"NACK: Mask {mask_name} not found"
            # else:
            #     self.instr.devices[axis].update_mask_position(mask_name)
            #     return "ACK"

        def fpm_offset_all_mask_positions_msg(axis, rel_offset_x, rel_offset_y):
            device = self.instr.all_devices[axis]
            if device is None:
                return f"NACK: Axis {axis} not found"
            else:
                device.offset_all_mask_positions(rel_offset_x, rel_offset_y)
                return "ACK"
            # if axis not in self.instr.devices:
            #     return f"NACK: Axis {axis} not found"
            # else:
            #     self.instr.devices[axis].offset_all_mask_positions(
            #         rel_offset_x, rel_offset_y
            #     )
            #     return "ACK"

        def fpm_write_mask_positions_msg(axis):
            device = self.instr.all_devices[axis]
            if device is None:
                return f"NACK: Axis {axis} not found"
            else:
                device.write_current_mask_positions()
                return "ACK"

        def fpm_update_all_mask_positions_relative_to_current_msg(
            axis, current_mask_name, reference_mask_position_file
        ):
            device = self.instr.all_devices[axis]
            if device is None:
                return f"NACK: Axis {axis} not found"
            else:
                device.update_all_mask_positions_relative_to_current(
                    current_mask_name, reference_mask_position_file, write_file=False
                )
                return "ACK"

        def standby_msg(axis):
            return self.instr.standby(axis)

        def online_msg(axes):
            # parse axes into list
            axis_list = axes.split(",")
            return self.instr.online(axis_list)

        def h_shut_msg(state, beam_numbers):
            if beam_numbers == "all":
                beam_numbers = list(range(1, 5))
            else:
                beam_numbers = [int(b) for b in beam_numbers.split(",")]

            if state not in ["open", "close"]:
                return "NACK: Invalid state for h_shut, must be 'open' or 'close'"

            return self.instr.h_shut(state, beam_numbers)

        def h_splay_msg(state):
            return self.instr.h_splay(state)

        def temp_status_msg(mode):
            """
            Get the temperature status of the instrument.
            Returns a list of values in order, see instrument documentation.
            """
            if mode == "now":
                return str(self.instr.temp_summary.get_temp_status())
            if mode == "keys":
                keys = self.instr.temp_summary.get_temp_keys()
                return f"[{','.join(keys)}]"
            return "NACK: Invalid mode for temp_status, must be 'now' or 'keys'"

        first_word_to_function = {
            "read": read_msg,
            "stop": stop_msg,
            "moveabs": moveabs_msg,
            "connected?": connected_msg,
            "connect": connect_msg,
            "init": init_msg,
            "tt_step": tt_step_msg,
            "tt_config_step": tt_config_step_msg,
            "moverel": moverel_msg,
            "state": state_msg,
            "save": save_msg,
            "dmapplyflat": apply_flat_msg,
            "dmapplycross": apply_cross_msg,
            "fpm_getsavepath": fpm_get_savepath_msg,
            "fpm_maskpositions": fpm_mask_positions_msg,
            "fpm_movetomask": fpm_move_to_phasemask_msg,
            "fpm_moverel": fpm_move_relative_msg,
            "fpm_moveabs": fpm_move_absolute_msg,
            "fpm_readpos": fpm_read_position_msg,
            "fpm_update_position_file": fpm_update_position_file_msg,
            "fpm_updatemaskpos": fpm_update_mask_position_msg,
            "fpm_offsetallmaskpositions": fpm_offset_all_mask_positions_msg,
            "fpm_writemaskpos": fpm_write_mask_positions_msg,
            "fpm_updateallmaskpos": fpm_update_all_mask_positions_relative_to_current_msg,
            "ping": ping_msg,
            "health": health_msg,
            "on": on_msg,
            "off": off_msg,
            "is_on": is_on_msg,
            "reset": reset_msg,
            "mv_img": mv_img_msg,
            "mv_pup": mv_pup_msg,
            "asg_setup": asg_setup_msg,
            "home_steppers": home_steppers_msg,
            "standby": standby_msg,
            "online": online_msg,
            "h_shut": h_shut_msg,
            "h_splay": h_splay_msg,
            "temp_status": temp_status_msg,
        }

        first_word_to_format = {
            "read": "read {}",
            "stop": "stop {}",
            "moveabs": "moveabs {} {:f}",
            "connected?": "connected? {}",
            "connect": "connect {}",
            "init": "init {}",
            "tt_step": "tt_step {} {}",
            "tt_config_step": "tt_config_step {} {}",
            "moverel": "moverel {} {:f}",
            "state": "state {}",
            "save": "save {} {}",
            "dmapplyflat": "dmapplyflat {}",
            "dmapplycross": "dmapplycross {}",
            "fpm_getsavepath": "fpm_getsavepath {}",
            "fpm_maskpositions": "fpm_maskpositions {}",
            "fpm_movetomask": "fpm_movetomask {} {}",
            "fpm_moverel": "fpm_moverel {} {}",
            "fpm_moveabs": "fpm_moveabs {} {}",
            "fpm_readpos": "fpm_readpos {}",
            "fpm_update_position_file": "fpm_update_position_file {} {}",
            "fpm_updatemaskpos": "fpm_updatemaskpos {} {}",
            "fpm_offsetallmaskpositions": "fpm_offsetallmaskpositions {} {} {}",
            "fpm_writemaskpos": "fpm_writemaskpos {}",
            "fpm_updateallmaskpos": "fpm_updateallmaskpos {} {} {}",
            "ping": "ping {}",
            "health": "health",
            "on": "on {}",
            "off": "off {}",
            "is_on": "is_on {}",
            "reset": "reset {}",
            "mv_img": "mv_img {} {} {:f} {:f}",  # mv_img {config} {beam_number} {x} {y}
            "mv_pup": "mv_pup {} {} {:f} {:f}",  # mv_pup {config} {beam_number} {x} {y}
            "asg_setup": "asg_setup {} {} {}",  # 2nd input is either a named position or a float
            "home_steppers": "home_steppers {}",
            "standby": "standby {}",
            "online": "online {}",
            "h_shut": "h_shut {} {}",
            "h_splay": "h_splay {}",
            "temp_status": "temp_status {}",
        }

        try:
            first_word = message.split(" ")[0]
            if first_word in first_word_to_function:
                format_str = first_word_to_format[first_word]
                result = parse(format_str, message)
                return first_word_to_function[first_word](*result)
            else:
                return "NACK: Unkown custom command"

            # old
            # for pattern, func in patterns.items():
            #     result = parse(pattern, message)
            #     if result:
            #         return func(*result)
        except Exception as e:
            logging.error(f"Custom command error: {e}")
            return f"NACK: {e}"


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the MDS server.")
    parser.add_argument(
        "-c", "--config", type=str, required=True, help="Path to the configuration file"
    )
    parser.add_argument(
        "--host", type=str, default="192.168.100.2", help="Host address"
    )
    parser.add_argument(
        "--log-location",
        type=str,
        default="~/logs/mds/",
        help="Path to the log directory",
    )
    parser.add_argument("-p", "--port", type=int, default=5555, help="Port number")

    args = parser.parse_args()

    # logname from the current time
    log_fname = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S") + ".log"
    logging.basicConfig(
        filename=os.path.join(os.path.expanduser(args.log_location), log_fname),
        level=logging.INFO,
    )

    # Add stream handler to also log to stdout
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    formatter = logging.Formatter("%(asctime)s %(levelname)s %(message)s")
    console.setFormatter(formatter)
    logging.getLogger().addHandler(console)

    serv = MultiDeviceServer(args.port, args.host, args.config)
    serv.run()
    log_fname = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S") + ".log"
    logging.basicConfig(
        filename=os.path.join(os.path.expanduser(args.log_location), log_fname),
        level=logging.INFO,
    )

    # Add stream handler to also log to stdout
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    formatter = logging.Formatter("%(asctime)s %(levelname)s %(message)s")
    console.setFormatter(formatter)
    logging.getLogger().addHandler(console)

    serv = MultiDeviceServer(args.port, args.host, args.config)
    serv.run()
