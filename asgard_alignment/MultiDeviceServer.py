import zmq
import asgard_alignment
import argparse
import sys
import re
from parse import parse
import time

import enum
import asgard_alignment.ESOdevice
import asgard_alignment.Instrument
import asgard_alignment.MultiDeviceServer
import asgard_alignment.Engineering
import asgard_alignment.NewportMotor


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
