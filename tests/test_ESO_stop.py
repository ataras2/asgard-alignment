import argparse
import zmq
import time
import json


def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="ZeroMQ Client")
    parser.add_argument("--host", type=str, default="localhost", help="Server host")
    parser.add_argument("--port", type=int, default=5555, help="Server port")
    parser.add_argument(
        "--timeout", type=int, default=5000, help="Response timeout in milliseconds"
    )
    args = parser.parse_args()

    # Create a ZeroMQ context
    context = zmq.Context()

    # Create a socket to communicate with the server
    socket = context.socket(zmq.REQ)

    # Set the receive timeout
    socket.setsockopt(zmq.RCVTIMEO, args.timeout)

    # Connect to the server
    server_address = f"tcp://{args.host}:{args.port}"
    socket.connect(server_address)

    # time now, in format "YYYY-MM-DDThh:mm:ss"
    time_now = time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime())

    print(time_now)
    # test 1: stop a motor
    msg = {
        "command": {
            "name": "stop",
            "time": time_now,
            "Paramters": [
                {
                    "device": "HFO1",
                }
            ],
        }
    }

    message = json.dumps(msg) + "\n"

    print(f"Sending message to server: {message}")
    socket.send_string(message)

    try:
        # Wait for a response from the server
        response = socket.recv_string()
        print(f"Received response from server: {response}")
    except zmq.Again as e:
        print(f"Timeout waiting for response from server: {e}")


if __name__ == "__main__":
    main()
