import zmq
import asgard_alignment
import argparse
import sys
from parse import parse
import time

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


class MockMDS:
    def __init__(self):
        pass

    def handle_zmq(self, message):
        print(f"Received message: {message}")
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

        self.db_update_socket = self.context.socket(zmq.PUSH)
        self.db_update_socket.connect("tcp://wag:5561")

        self.database_message = self.DATABASE_MSG_TEMPLATE.copy()

        if config_file == "mock":
            self.instr = MockMDS()
        else:
            self.instr = asgard_alignment.Instrument.Instrument(self.config_file)

        print("Instrument all set up, ready to accept messages")

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
                    print(f"Received message: {data}")
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
                        if is_custom_msg:
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

    def handle_message(self, message):
        """
        Handles a recieved message. Custom messages are indicated by lowercase commands
        """

        # if "!" in message:
        if message[0].islower():
            print(f"Custom command: {message}")
            return True, self._handle_custom_command(message)

        if message[0] == "!":
            print("Old custom command")
            return True, "NACK: Are you using old custom commands?"

        try:
            message = message.rstrip(message[-1])
            print(f"ESO msg recv: {message}")
            json_data = json.loads(message)
        except:
            print("Error: Invalid JSON message")
            return False, "NACK: Invalid JSON message"
        command_name = json_data["command"]["name"]
        time_stampIn = json_data["command"]["time"]

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

        if "online" in command_name:
            # TODO: make sure all devices are powered on
            # .............................................................
            # If needed, call controller-specific functions to power up
            # the devices and have them ready for operations
            # .............................................................
            for key in self.instr.devices:
                self.instr.devices[key].online()

            # Update the wagics database to show all the devices in ONLINE
            # state (value of "state" attribute has to be set to 3)

            for key in self.instr.devices:
                attribute = f"<alias>{key}.state"
                self.database_message["command"]["parameters"].append(
                    {"attribute": attribute, "value": 3}
                )

            # Send message to wag to update the database

            self.database_message["command"]["time"] = self.get_time_stamp()
            output_msg = json.dumps(self.database_message) + "\0"

            self.db_update_socket.send_string(output_msg)
            print(output_msg)

            reply = "OK"

        # Case of "standby" (sent by wag when bringing ICS standby,
        # usually when the instrument night operations are finished)

        if "standby" in command_name:
            # .............................................................
            # If needed, call controller-specific functions to bring some
            # devices to a "parking" position and to power them off
            # .............................................................
            # can have no parameters (standby all) or a subset indicated by parameters

            if "parameters" not in json_data["command"]:
                print("No parameters in standby command, going to standby all")

                for key in self.instr.devices:
                    self.instr.devices[key].standby()

                # Update the wagics database to show all the devices in STANDBY
                # state (value of "state" attrivute has to be set to 2)
                for key in self.instr.devices:
                    attribute = f"<alias>{key}.state"
                    self.database_message["command"]["parameters"].append(
                        {"attribute": attribute, "value": 2}
                    )

            else:
                n_devs_commanded = len(json_data["command"]["parameters"])
                for i in range(n_devs_commanded):
                    dev = json_data["command"]["parameters"][i]["device"]
                    print(f"Standby device: {dev}")

                    self.instr.devices[dev].standby()

                # Update the wagics database to show all the devices in STANDBY
                # state (value of "state" attrivute has to be set to 2)

                self.database_message["command"]["parameters"].clear()
                for i in range(n_devs_commanded):
                    dev = json_data["command"]["parameters"][i]["device"]
                    attribute = f"<alias>{dev}.state"
                    self.database_message["command"]["parameters"].append(
                        {"attribute": attribute, "value": 2}
                    )

            # Send message to wag to update the database
            time_now = datetime.datetime.now()
            time_stamp = time_now.strftime("%Y-%m-%dT%H:%M:%S")
            self.database_message["command"]["time"] = time_stamp
            output_msg = json.dumps(self.database_message) + "\0"

            self.db_update_socket.send_string(output_msg)
            print(output_msg)

            reply = "OK"

        # Case of "setup" (sent by wag to move devices)
        if "setup" in command_name:
            n_devs_to_setup = len(json_data["command"]["parameters"])

            semaphore_array = [0] * 100  # TODO: implement this maximum correctly

            # Create a double-list of devices to move
            setup_cmds = [[], []]
            for i in range(n_devs_to_setup):
                kwd = json_data["command"]["parameters"][i]["name"]
                val = float(json_data["command"]["parameters"][i]["value"])
                print(f"Setup: {kwd} to {val}")

                # Keywords are in the format: INS.<device>.<motion type>

                prefixes = kwd.split(".")
                dev_name = prefixes[1]
                motion_type = prefixes[2]
                print(f"Device: {dev_name} - motion type: {motion_type}")

                # motion_type can be one of these words:
                # NAME   = Named position (e.g., IN, OUT, J1, H3, ...)
                # ENC    = Absolute encoder position
                # ENCREL = Relative encoder postion (can be negative)
                # ST     = State. Given value is equal to either T or F.
                #          if device is shutter: T = open, F = closed.
                #          if device is lamp: T = on, F = off.

                # Look if device exists in list
                # (something should be done if device does not exist) TODO
                device = self.instr.devices[dev_name]

                semaphore_id = device.semaphore_id
                if semaphore_array[semaphore_id] == 0:
                    # Semaphore is free =>
                    # Device can be moved now
                    setup_cmds[0].append(
                        asgard_alignment.ESOdevice.SetupCommand(
                            dev_name, motion_type, val
                        )
                    )
                    semaphore_array[semaphore_id] = 1
                else:
                    # Semaphore is already taken =>
                    # Device will be moved in a second batch
                    setup_cmds[1].append(
                        asgard_alignment.ESOdevice.SetupCommand(
                            dev_name, motion_type, val
                        )
                    )

            # Move devices (two batches if needed)
            for batch in range(2):
                if len(setup_cmds[batch]) > 0:
                    print(f"batch {batch} of devices to move:")
                    self.database_message["command"]["parameters"].clear()
                    for s in setup_cmds[batch]:
                        print(
                            f"Moving: {s.device_name} to: {s.value} ( setting {s.motion_type} )"
                        )

                        # do the actual move...
                        self.instr.devices[s.device_name].setup(s.motion_type, s.value)

                        # Inform wag ICS that the device is moving
                        attribute = f"<alias>{s.device_name}:DATA.status0"
                        self.database_message["command"]["parameters"].append(
                            {"attribute": attribute, "value": "MOVING"}
                        )

                    # Send message to wag to update the database
                    self.database_message["command"]["time"] = self.get_time_stamp()
                    output_msg = json.dumps(self.database_message) + "\0"

                    self.db_update_socket.send_string(output_msg)
                    print(output_msg)

                    # TODO
                    # ........................................................
                    # Add here calls to read (every 1 to 3 seconds) the position
                    # of (all of the relevant) devices and update the database of wag (using the
                    # code below to generate the JSON message)
                    # ........................................................

                    still_moving_prev = setup_cmds[batch]
                    still_moving = setup_cmds[batch]
                    while len(still_moving) > 0:
                        time.sleep(1.0)

                        still_moving = []
                        still_moving_prev = setup_cmds[batch]

                        for s in still_moving_prev:
                            dev = s.device_name
                            pos = self.instr.devices[dev].read_position()

                            self.database_message["command"]["parameters"].append(
                                {
                                    "attribute": f"<alias>{dev}:DATA.posEnc",
                                    "value": pos,
                                }
                            )
                            if self.instr.devices[dev].is_moving():
                                still_moving.append(s)
                            else:
                                # not moving, so also send the done moving status
                                if s.motion_type == "NAME":
                                    self.database_message["command"][
                                        "parameters"
                                    ].append(
                                        {
                                            "attribute": f"<alias>{dev}:DATA.status0",
                                            "value": s.value,
                                        }
                                    )
                                elif s.motion_type == "ST":
                                    # TODO: change this to a mapping T -> OPEN, F -> CLOSED, and lamp case...
                                    if s.val == "T":
                                        self.database_message["command"][
                                            "parameters"
                                        ].append(
                                            {
                                                "attribute": f"<alias>{dev}:DATA.status0",
                                                "value": "OPEN",
                                            }
                                        )
                                    else:
                                        self.database_message["command"][
                                            "parameters"
                                        ].append(
                                            {
                                                "attribute": f"<alias>{dev}:DATA.status0",
                                                "value": "CLOSED",
                                            }
                                        )
                                elif s.motion_type == "ENC":
                                    self.database_message["command"][
                                        "parameters"
                                    ].append(
                                        {
                                            "attribute": f"<alias>{dev}:DATA.status0",
                                            "value": "",
                                        }
                                    )
                            # Case of motor with relative encoder position
                            # not considered yet
                            # The simplest would be to read the encoder position
                            # and to update the database as for the previous case

                        still_moving_prev = deepcopy(still_moving)

                        # Send message to wag to update its database
                        self.database_message["command"]["time"] = self.get_time_stamp()
                        output_msg = json.dumps(self.database_message) + "\0"

                        self.db_update_socket.send_string(output_msg)
                        print(output_msg)

                        self.database_message["command"]["parameters"].clear()
            
            reply = "OK"

        # Case of "stop" (sent by wag to immediately stop the devices)

        if "stop" in command_name:
            n_devs_commanded = len(json_data["command"]["parameters"])
            for i in range(n_devs_commanded):
                dev = json_data["command"]["parameters"][i]["device"]
                print(f"Stop device: {dev}")

                self.instr.devices[dev].stop()

            reply = "OK"

        # Case of "disable" (sent by wag to power-off devices)

        if "disable" in command_name:
            n_devs_commanded = len(json_data["command"]["parameters"])
            for i in range(n_devs_commanded):
                dev = json_data["command"]["parameters"][i]["device"]
                print(f"Power off device: {dev}")

                self.instr.devices[dev].disable()

            reply = "OK"

        # Case of "enable" (sent by wag to power-on devices)

        if "enable" in command_name:
            n_devs_commanded = len(json_data["command"]["parameters"])
            for i in range(n_devs_commanded):
                dev = json_data["command"]["parameters"][i]["device"]
                print(f"Power on device: {dev}")

                self.instr.devices[dev].enable()

            reply = "OK"

        # Send back reply to ic0fb process

        time_stamp = MultiDeviceServer.get_time_stamp()
        reply = f'{{\n\t"reply" :\n\t{{\n\t\t"content" : "{reply}",\n\t\t"time" : "{time_stamp}"\n\t}}\n}}\n\0'
        print(reply)
        self.server.send_string(reply)

        return False, None

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
            self.instr._attempt_to_open(axis, recheck_ports=True)

            return "connected" if axis in self.instr.devices else "not connected"

        def init_msg(axis):
            self.instr.devices[axis].init()
            return "ACK"

        def moverel_msg(axis, position):
            self.instr.devices[axis].move_relative(float(position))
            return "ACK"

        def state_msg(axis):
            return self.instr.devices[axis].read_state()

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
            print(beam_number, type(beam_number))
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

        def asg_setup_msg(axis, value):
            try:
                # if it is a float, convert it to a python float
                try:
                    value = float(value)
                except ValueError:
                    pass
                self.instr.devices[axis].setup(value)
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

            print(f"Flat map applied to {dm_name}")
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

            print(f"Cross map applied to {dm_name}")
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
            # if axis not in self.instr.devices:
            #     return f"NACK: Axis {axis} not found"
            # else:
            #     self.instr.devices[axis].write_current_mask_positions()
            #     return "ACK"

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
            # if axis not in self.instr.devices:
            #     return f"NACK: Axis {axis} not found"
            # else:
            #     self.instr.devices[axis].update_all_mask_positions_relative_to_current(
            #         current_mask_name, reference_mask_position_file, write_file=False
            #     )
            #     return "ACK"

        # patterns = {
        #     "read {}": read_msg,
        #     "stop {}": stop_msg,
        #     "moveabs {} {:f}": moveabs_msg,
        #     "connected? {}": connected_msg,
        #     "connect {}": connect_msg,
        #     "init {}": init_msg,
        #     "moverel {} {:f}": moverel_msg,
        #     "state {}": state_msg,
        #     "dmapplyflat {}": apply_flat_msg,
        #     "dmapplycross {}": apply_cross_msg,
        #     "fpm_getsavepath {}": fpm_get_savepath_msg,
        #     "fpm_maskpositions {}": fpm_mask_positions_msg,
        #     "fpm_movetomask {} {}": fpm_move_to_phasemask_msg,
        #     "fpm_moverel {} {}": fpm_move_relative_msg,
        #     "fpm_moveabs {} {}": fpm_move_absolute_msg,
        #     "fpm_readpos {}": fpm_read_position_msg,
        #     "fpm_updatemaskpos {} {}": fpm_update_mask_position_msg,
        #     "fpm_writemaskpos {}": fpm_write_mask_positions_msg,
        #     "fpm_updateallmaskpos {} {} {}": fpm_update_all_mask_positions_relative_to_current_msg,
        #     #            "on {}": on_msg,
        #     #            "off {}": off_msg,
        # }

        first_word_to_function = {
            "read": read_msg,
            "stop": stop_msg,
            "moveabs": moveabs_msg,
            "connected?": connected_msg,
            "connect": connect_msg,
            "init": init_msg,
            "moverel": moverel_msg,
            "state": state_msg,
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
        }

        first_word_to_format = {
            "read": "read {}",
            "stop": "stop {}",
            "moveabs": "moveabs {} {:f}",
            "connected?": "connected? {}",
            "connect": "connect {}",
            "init": "init {}",
            "moverel": "moverel {} {:f}",
            "state": "state {}",
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
            "asg_setup": "asg_setup {} {}",  # 2nd input is either a named position or a float
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
            return f"NACK: {e}"


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the MDS server.")
    parser.add_argument(
        "-c", "--config", type=str, required=True, help="Path to the configuration file"
    )
    parser.add_argument("--host", type=str, default="172.16.8.6", help="Host address")
    parser.add_argument("-p", "--port", type=int, default=5555, help="Port number")

    args = parser.parse_args()

    serv = MultiDeviceServer(args.port, args.host, args.config)
    serv.run()
