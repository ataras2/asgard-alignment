import zmq
import asgard_alignment
import argparse
import sys
from parse import parse
import time

import json
import datetime

import enum
import asgard_alignment.ESOdevice
import asgard_alignment.Instrument
import asgard_alignment.MultiDeviceServer
import asgard_alignment.Engineering
import asgard_alignment.NewportMotor
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

        self.database_message = self.DATABASE_MSG_TEMPLATE.copy()

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

    @staticmethod
    def get_timestamp():
        time_now = datetime.datetime.now()
        return time_now.strftime("%Y-%m-%dT%H:%M:%S")

    def handle_message(self, message):
        """
        Handles a recieved message. Custom messages are indicated by a leading "!".
        """

        if "!" in message:
            return self._handle_custom_command(message)

        message = message.rstrip(message[-1])
        print(message)
        json_data = json.loads(message)
        command_name = json_data["command"]["name"]
        timeStampIn = json_data["command"]["time"]

        # Verification of received time-stamp (TODO)
        # If the timestamp is invalid, set command_name to "none",
        # so no command will be processed but a reply will be sent
        # back to the client (set replyContent to "ERROR")

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

            self.database_message["command"]["time"] = self.get_timestamp()
            outputMsg = json.dumps(self.database_message) + "\0"

            cliSocket.send_string(outputMsg)
            print(outputMsg)

            replyContent = "OK"

        # Case of "standby" (sent by wag when bringing ICS standby,
        # usually when the instrument night operations are finished)

        if "standby" in command_name:
            # .............................................................
            # If needed, call controller-specific functions to bring some
            # devices to a "parking" position and to power them off
            # .............................................................
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
            timeNow = datetime.datetime.now()
            timeStamp = timeNow.strftime("%Y-%m-%dT%H:%M:%S")
            self.database_message["command"]["time"] = timeStamp
            outputMsg = json.dumps(self.database_message) + "\0"

            cliSocket.send_string(outputMsg)
            print(outputMsg)

            replyContent = "OK"

        # Case of "setup" (sent by wag to move devices)
        if "setup" in command_name:
            n_devs_to_setup = len(json_data["command"]["parameters"])

            semaphore_array = [0] * 100  # TODO: implement this maximum correctly

            # Create a double-list of devices to move
            setupList = [[], []]
            for i in range(n_devs_to_setup):
                kwd = json_data["command"]["parameters"][i]["name"]
                val = json_data["command"]["parameters"][i]["value"]
                print(f"Setup: {kwd} to {val}")

                # Keywords are in the format: INS.<device>.<motion type>

                prefixes = kwd.split(".")
                dev_name = prefixes[1]
                mType = prefixes[2]
                print(f"Device: {dev_name} - motion type: {mType}")

                # mType can be one of these words:
                # NAME   = Named position (e.g., IN, OUT, J1, H3, ...)
                # ENC    = Absolute encoder position
                # ENCREL = Relative encoder postion (can be negative)
                # ST     = State. Given value is equal to either T or F.
                #          if device is shutter: T = open, F = closed.
                #          if device is lamp: T = on, F = off.

                # Look if device exists in list
                # (something should be done if device does not exist) TODO
                device = self.instr.devices[dev_name]

                semId = device.semId
                if semaphore_array[semId] == 0:
                    # Semaphore is free =>
                    # Device can be moved now
                    setupList[0].append(
                        asgard_alignment.ESOdevice.SetupCommand(dev, mType, val)
                    )
                    semaphore_array[semId] = 1
                else:
                    # Semaphore is already taken =>
                    # Device will be moved in a second batch
                    setupList[1].append(
                        asgard_alignment.ESOdevice.SetupCommand(dev, mType, val)
                    )

            # Move devices (two batches if needed)
            for batch in range(2):
                if len(setupList[batch]) > 0:
                    print(f"batch {batch} of devices to move:")
                    self.database_message["command"]["parameters"].clear()
                    for s in setupList[batch]:
                        print(
                            f"Moving: {s.dev} to: {s.val} ( setting {s.mType} )"
                        )

                        # do the actual move...
                        self.instr.devices[s.dev].setup(s.mType, s.val)

                        # Inform wag ICS that the device is moving
                        attribute = f"<alias>{s.dev}:DATA.status0"
                        self.database_message["command"]["parameters"].append(
                            {"attribute": attribute, "value": "MOVING"}
                        )

                    # Send message to wag to update the database
                    self.database_message["command"]["time"] = self.get_timestamp()
                    outputMsg = json.dumps(self.database_message) + "\0"

                    cliSocket.send_string(outputMsg)
                    print(outputMsg)

                    # ........................................................
                    # Add here calls to read (every 1 to 3 seconds) the position
                    # of (all of the relevant) devices and update the database of wag (using the
                    # code below to generate the JSON message)
                    # ........................................................

                    # ........................................................
                    # Add here call to check that devices have reached their
                    # requested positions. Once done, inform wag as follows:
                    # ........................................................

                    for s in setupList[batch]:
                        attribute = "<alias>" + s.dev + ":DATA.status0"
                        # Case of motor with named position requested
                        if s.mType == "NAME":
                            self.database_message["command"]["parameters"].append(
                                {"attribute": attribute, "value": s.val}
                            )
                            # Note: normally the encoder position shall be
                            # reported along with the named position
                            # ...............................................
                            # => Call function to read the encoder position
                            #    store it in a variable "posEnc" and execute:
                            #
                            # attribute = "<alias>" + s.dev +":DATA.posEnc"
                            # self.database_message['command']['parameters'].append({"attribute":attribute, "value":posEnc})

                        # Case of shutter or lamp
                        if s.mType == "ST":
                            # Here the device can be either a lamp or a shutter
                            # Add here code to find out the type of s.dev
                            # If it is a shutter do:
                            if s.val == "T":
                                self.database_message["command"]["parameters"].append(
                                    {"attribute": attribute, "value": "OPEN"}
                                )
                            else:
                                self.database_message["command"]["parameters"].append(
                                    {"attribute": attribute, "value": "CLOSED"}
                                )
                        # If it is a lamp, reuse the code above replacing
                        # OPEN  by ON and CLOSED by OFF

                        # Case of motor with absolute encoder position requested
                        if s.mType == "ENC":
                            self.database_message["command"]["parameters"].append(
                                {"attribute": attribute, "value": ""}
                            )
                            # Note: if motor is at limit, do:
                            # self.database_message['command']['parameters'].append({"attribute":attribute, "value":"LIMIT"})
                            attribute = "<alias>" + s.dev + ":DATA.posEnc"
                            self.database_message["command"]["parameters"].append(
                                {"attribute": attribute, "value": s.val}
                            )

                    # Send message to wag to update its database
                    timeNow = datetime.datetime.now()
                    timeStamp = timeNow.strftime("%Y-%m-%dT%H:%M:%S")
                    self.database_message["command"]["time"] = timeStamp
                    outputMsg = json.dumps(self.database_message) + "\0"

                    cliSocket.send_string(outputMsg)
                    print(outputMsg)

        # Case of "stop" (sent by wag to immediately stop the devices)

        if "stop" in command_name:
            n_devs_commanded = len(json_data["command"]["parameters"])
            for i in range(n_devs_commanded):
                dev = json_data["command"]["parameters"][i]["device"]
                print(f"Stop device: {dev}")

                self.instr.devices[dev].stop()

            replyContent = "OK"

        # Case of "disable" (sent by wag to power-off devices)

        if "disable" in command_name:
            n_devs_commanded = len(json_data["command"]["parameters"])
            for i in range(n_devs_commanded):
                dev = json_data["command"]["parameters"][i]["device"]
                print(f"Power off device: {dev}")

                self.instr.devices[dev].disable()

            replyContent = "OK"

        # Case of "enable" (sent by wag to power-on devices)

        if "enable" in command_name:
            n_devs_commanded = len(json_data["command"]["parameters"])
            for i in range(n_devs_commanded):
                dev = json_data["command"]["parameters"][i]["device"]
                print(f"Power on device: {dev}")

                self.instr.devices[dev].enable()

            replyContent = "OK"

        # Send back reply to ic0fb process

        timeNow = datetime.datetime.now()
        timeStamp = timeNow.strftime("%Y-%m-%dT%H:%M:%S")
        reply = (
            f'{{\n\t"reply" :\n\t{{\n\t\t"content" : "{replyContent}",\n\t\t"time" : "{timeStamp}"\n\t}}\n}}\n\0'
        )
        print(reply)
        srvSocket.send_string(reply)

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

        patterns = {
            "!read {}": read_msg,
            "!stop {}": stop_msg,
            "!moveabs {} {:f}": moveabs_msg,
            "!connected? {}": connected_msg,
            "!connect {}": connect_msg,
            "!init {}": init_msg,
            "!moverel {} {:f}": moverel_msg,
            "!state {}": state_msg,
        }

        try:
            for pattern, func in patterns.items():
                result = parse(pattern, message)
                if result:
                    return func(*result)
        except Exception as e:
            return f"NACK: {e}"
        return "NACK: Unkown custom command"


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
