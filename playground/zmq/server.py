# This python object is designed to take as inputs a socket number, a server name
# and a hardware object, and to do all communication in the same way for all servers.

# *************************************************************************#
# Code the runs on module import starts here.
# 1) Import all modules that we need.
from __future__ import division, print_function

import sys, time
import string
import zmq
import select
from datetime import datetime


# *************************************************************************#
# 2) Now define our main class, the ServerSocket
class ServerSocket:
    # Some properties needed by multiple methods.
    clients = []
    jobs = []

    def __init__(self, port, hardware_name, hardware_object):
        try:
            self.context = zmq.Context()
            self.server = self.context.socket(zmq.REP)
            tcpstring = "tcp://*:" + str(port)
            print(tcpstring)
            self.server.bind(tcpstring)
            self.poller = zmq.Poller()
            self.poller.register(self.server, zmq.POLLIN)
            self.connected = True
        except Exception as e:
            print("Error: " + str(e))
            self.connected = False

        # Set up the object "hardware_name" and "hardware_object"
        self.hardware_object = hardware_object
        self.hardware_name = hardware_name
        # Still use an input array, even though this is text only now.
        self.input = [sys.stdin]

    # This method deals with the various inputs from stdin and connected clients
    def socket_funct(self, s):
        if s == self.server:
            # handle server socket
            return s.recv()
        raise UserWarning

    # We will use this to log what is happening in a file with a timestamp, but for now, print to screen
    # I should also add something to document which client sent which command
    def log(self, message):
        print(str(datetime.now()) + " " + str(message))

    # This closes the connections to the cliente neatly.
    def close(self):
        self.server.close

    # This medhod adds a new job to the queue.
    def add_job(self, new_job):
        self.jobs.append(new_job)

    # This method runs the jobs and waits for new input
    def run(self):
        self.log(
            "Waiting for connection, number of clients connected: "
            + str(len(self.clients))
        )
        running = True
        while running:
            time.sleep(0.1)
            # inputready, outputready, exceptready = select.select(self.input, [], [], 0)
            inputready = []
            # pdb.set_trace()
            if self.connected:
                socks = dict(self.poller.poll(10))
                if self.server in socks and socks[self.server] == zmq.POLLIN:
                    inputready.append(self.server)
            for s in inputready:  # loop through our array of sockets/inputs
                data = self.socket_funct(s)
                if data == -1:
                    running = False
                elif data != 0:
                    # response = cl.execute_command(data,self.hardware_object)
                    response = "hello"
                    if response == -1:
                        running = False
                        if s == sys.stdin:
                            self.log("Manually shut down. Goodbye.")
                        else:
                            self.log("Shut down by remote connection. Goodbye.")
                    else:
                        s.send_string(response + "\n")


if __name__ == "__main__":
    serv = ServerSocket(5555, "test", None)

    serv.run()
