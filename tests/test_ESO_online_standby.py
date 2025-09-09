import argparse
import zmq
import time
import json


standby_cmd = {
     "command" :
     {
         "name" : "standby",
         "time" : "2025-04-21T09:42:46",
         "parameters" :
         [
             {"device": "all"}
         ]
     }
}

online_cmd = {
     "command" :
     {
         "name" : "online",
         "time" : "2025-04-21T09:43:39"
     }
}


standby_subset_cmd = {
     "command" :
     {
         "name" : "standby",
         "time" : "2025-04-21T09:42:46",
         "parameters" :
         [
             {
                 "device" : "BOTP2"
             },
            #  {
            #      "device" : "HFO1"
            #  }
         ]
     }
}

online_subset_cmd = {
     "command" :
     {
         "name" : "online",
         "time" : "2025-04-21T09:43:39",
         "parameters" :
         [
             {
                 "device" : "BOTP2"
             },
             {
                 "device" : "BOTT2"
             },
             {
                 "device" : "BOTP3"
             }
            #  {
            #      "device" : "HFO1"
            #  }
         ]
     }
}


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
    socket.setsockopt(zmq.RCVTIMEO, 10000)

    # Connect to the server
    server_address = f"tcp://{'mimir'}:{5555}"
    socket.connect(server_address)

    # time now, in format "YYYY-MM-DDThh:mm:ss"
    time_now = time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime())

    use_subset = True
    # use_subset = False

    if use_subset:
        standby = standby_subset_cmd
    else:
        standby = standby_cmd

    res = utils.send_and_receive(socket, json.dumps(standby))
    standby["command"]["time"] = time_now
    print(res)

    time.sleep(5)

    time_now = time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime())
    online_subset_cmd["command"]["time"] = time_now
    res = utils.send_and_receive(socket, json.dumps(online_subset_cmd))
    print(res)
    

if __name__ == "__main__":
    main()
