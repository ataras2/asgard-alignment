import argparse
import zmq
import time
import json


standby_cmd = """
{
     "command" :
     {
         "name" : "standby",
         "time" : "2025-04-21T09:42:46"
     }
}
"""

online_cmd = """
{
     "command" :
     {
         "name" : "online",
         "time" : "2025-04-21T09:43:39"
     }
}
"""

disable_cmd = """
{
     "command" :
     {
         "name" : "disable",
         "time" : "2025-04-21T09:46:39",
         "parameters" :
         [
             {
                 "device" : "http1"
             }
         ]
     }
}
"""

enable_cmd = """  
{
     "command" :
     {
         "name" : "enable",
         "time" : "2025-04-21T09:47:39",
         "parameters" :
         [
             {
                 "device" : "http1"
             }
         ]
     }
}
"""


# setup socket
def setup_socket(host, port):
    # Create a ZeroMQ context
    context = zmq.Context()

    # Create a socket to communicate with the server
    socket = context.socket(zmq.REQ)

    # Set the receive timeout
    socket.setsockopt(zmq.RCVTIMEO, 20000)

    # Connect to the server
    server_address = f"tcp://{host}:{port}"
    socket.connect(server_address)
    return socket


def send_and_recv_cmd(socket, cmd):
    # Send the command to the server
    print(f"Sending command to server: {cmd}")
    socket.send_string(cmd)

    try:
        # Wait for a response from the server
        response = socket.recv_string()
        print(f"Received response from server: {response}")
    except zmq.Again as e:
        print(f"Timeout waiting for response from server: {e}")


def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="ZeroMQ Client")
    parser.add_argument("--host", type=str, default="172.16.8.6", help="Server host")
    parser.add_argument("--port", type=int, default=5555, help="Server port")
    args = parser.parse_args()

    # Setup socket
    socket = setup_socket(args.host, args.port)

    # Send commands
    cmds = [
        online_cmd,
        # standby_cmd,
        # disable_cmd,
        # enable_cmd,
    ]
    for cmd in cmds:
        send_and_recv_cmd(socket, cmd)
        time.sleep(1)

    # Close the socket
    socket.close()


if __name__ == "__main__":
    main()
