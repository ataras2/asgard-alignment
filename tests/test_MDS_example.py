"""
Communication example shared in email with subject line "preparation of wag software for September run"
"""

import argparse
import zmq
import time
import json
import test_utils as utils


def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="MDS example")
    # parser.add_argument("--host", type=str, default="localhost", help="Server host")
    # parser.add_argument("--port", type=int, default=5555, help="Server port")
    # parser.add_argument(
    #     "--timeout", type=int, default=5000, help="Response timeout in milliseconds"
    # )
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

    setup_cmd = {
        "command": {
            "name": "setup",
            "time": time_now,
            "parameters": [
                {"name": "INS.HTTP1.ENC", "value": "100"},
                {"name": "INS.HTPP1.ENC", "value": "100"},
            ],
        }
    }
    expected_reply = {
        "reply": {
            "content": "OK",
            "time": "2025-07-31T12:13:02",
            "parameters": [
                {"attribute": "<alias>HTTP1:DATA.status0", "value": "MOVING"}
            ],
        }
    }

    res = utils.send_and_receive(socket, json.dumps(setup_cmd))
    utils.compare_reply(dict(res), expected_reply)

    poll_cmd = {"command": {"name": "poll", "time": "2025-07-31T12:13:03"}}

    expected_reply = {
        "reply": {
            "content": "PENDING",
            "time": time_now,
            "parameters": [
                {"attribute": "<alias>HTTP1:DATA.status0", "value": "MOVING"},
                {"attribute": "<alias>HTTP1:DATA.posEnc", "value": "100"},
            ],
        }
    }

    res = utils.send_and_receive(socket, json.dumps(poll_cmd))
    utils.compare_reply(dict(res), expected_reply)


if __name__ == "__main__":
    main()
