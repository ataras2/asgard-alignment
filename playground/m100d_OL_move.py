import time
import zmq


def open_mds_connection():
    context = zmq.Context()
    socket = context.socket(zmq.REQ)
    socket.setsockopt(zmq.RCVTIMEO, 10000)
    server_address = "tcp://192.168.100.2:5555"
    socket.connect(server_address)
    return socket


def send_and_get_response(socket, message):
    print(f"sending: {message}")
    socket.send_string(message)
    response = socket.recv_string()
    return response.strip()


mds = open_mds_connection()

step_axis = "HTTI1"
step_size = 35  # default on startup
n_steps = 100

cmd = f"tt_config_step {step_axis} {step_size}"
res = send_and_get_response(mds, cmd)

cmd = f"tt_step {step_axis} {n_steps}"
res = send_and_get_response(mds, cmd)

cmd = f"tt_step {step_axis} {-n_steps}"
res = send_and_get_response(mds, cmd)
