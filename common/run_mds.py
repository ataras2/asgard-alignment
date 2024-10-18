import argparse
import zmq
import sys

import asgard_alignment
import asgard_alignment.old_MultiDeviceServer


class MockMDS:
    def __init__(self):
        pass

    def handle_zmq(self, message):
        print(f"Received message: {message}")
        return "Dummy response"


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
            self.mds = MockMDS()
        else:
            self.mds = asgard_alignment.old_MultiDeviceServer.Instrument(
                self.config_file
            )

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
                    response = self.mds.handle_zmq(data)
                    if response == -1:
                        running = False
                        if s == sys.stdin:
                            self.log("Manually shut down. Goodbye.")
                        else:
                            self.log("Shut down by remote connection. Goodbye.")
                    else:
                        s.send_string(response + "\n")


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
