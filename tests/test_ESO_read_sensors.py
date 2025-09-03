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
    parser = argparse.ArgumentParser(description="MDS sensors read")
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

    setup_cmd = {"command": {"name": "read", "time": time_now}}
    expected_reply = {
        "reply": {
            "content": "OK",
            "time": time_now,
            "parameters": [
                {"value": 11.11},
                {"value": 22.22},
                {"value": 33.33},
                {"value": 44.44},
            ],
        }
    }

    res = utils.send_and_receive(socket, json.dumps(setup_cmd))
    # compare the reply, but the values of "value" are arbitrary (just need to check floats)
    reply = dict(res)
    reply_copy = reply.copy()
    if "time" in reply_copy["reply"]:
        del reply_copy["reply"]["time"]
    for param in reply_copy["reply"]["parameters"]:
        param["value"] = float(param["value"])
    expected_reply_copy = expected_reply.copy()
    if "time" in expected_reply_copy["reply"]:
        del expected_reply_copy["reply"]["time"]
    for param in expected_reply_copy["reply"]["parameters"]:
        param["value"] = float(param["value"])
    if reply_copy == expected_reply_copy:
        print("Test passed: Reply matches expected reply.")
    else:
        print("Test failed: Reply does not match expected reply.")
        print(f"Expected: {expected_reply_copy}")
        print(f"Received: {reply_copy}")

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
