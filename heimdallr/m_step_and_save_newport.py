import PySpin

import numpy as np

import matplotlib.pyplot as plt
import os
import time
import argparse
import zmq
from tqdm import tqdm

parser = argparse.ArgumentParser(description="ZeroMQ Client")
parser.add_argument("--path", type=str, help="Path to save images and data")
parser.add_argument(
    "--beam", type=int, help="Beam number to move", choices=[1, 2, 3, 4]
)
parser.add_argument("--host", type=str, default="localhost", help="Server host")
parser.add_argument("--port", type=int, default=5555, help="Server port")
parser.add_argument(
    "--timeout", type=int, default=5000, help="Response timeout in milliseconds"
)
parser.add_argument("--start", type=float, default=6, help="Start position in mm")
parser.add_argument("--stop", type=float, default=10, help="End position in mm")
parser.add_argument("--step_size", type=float, default=0.010, help="Step size in mm")
parser.add_argument(
    "--n_imgs", type=int, default=3, help="Number of images to average per position"
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


def set_motor_position(socket, beam_number, position):
    message = f"!moveabs HFO{beam_number} {position}"
    # print(f"Sending message to server: {message}")
    socket.send_string(message)
    response = socket.recv_string()
    # print(f"Received response from server: {response}")
    return response


# pth = "data/Oct22/heimdallr_13_run0_sld"
pth = args.path

# beam = 2

# middle = 8.0


# start_pos = middle - 2  # mm
# end_pos = middle + 2  # mm
# step_size = 10e-3  # mm

start_pos = args.start  # mm
end_pos = args.stop  # mm
step_size = args.step_size  # mm
beam = args.beam


# step_size = 0.9  # mm
# start_pos = 5000  # um
# end_pos = 8500  # um
# step_size = 5  # um

set_motor_position(socket, beam, start_pos)
input("Press enter to start")
set_motor_position(socket, beam, end_pos)
input("Press enter to start")


if not os.path.exists(pth):
    os.makedirs(pth)
else:
    inp = input("path exists (possible overwrite!), press y to continue")
    if inp.lower() != "y":
        exit()


# motor.set_absolute_position(start_pos)
# time.sleep(2)
# motor.set_absolute_position(end_pos)

# exit()
# positions = list(range(start_pos, end_pos, step_size))
positions = np.round(np.arange(start_pos, end_pos, step_size), 5)

n_imgs = 3


# setup camera
system = PySpin.System.GetInstance()
cam_list = system.GetCameras()
cam = cam_list[0]

nodemap_tldevice = cam.GetTLDeviceNodeMap()

# Initialize camera
cam.Init()

# Retrieve GenICam nodemap
nodemap = cam.GetNodeMap()
cam.BeginAcquisition()
image_result = cam.GetNextImage(5000)
image_result.Release()

img = image_result.GetNDArray()
n_positions = len(positions)

img_stack = np.zeros((n_positions, n_imgs, img.shape[0], img.shape[1]), dtype=np.uint8)

if img.shape[0] > 550:
    # raise warning about large image size
    print("Warning: Image size is large, consider reducing the ROI size")


for i, pos in enumerate(tqdm(positions)):
    # print(f"\rMoving to {pos} um ({i+1}/{len(positions)})", end="")
    # axis.move_absolute(pos, Units.LENGTH_MICROMETRES)
    set_motor_position(socket, beam, pos)

    time.sleep(0.3)

    # image_result = cam.GetNextImage(2000)
    # image_result.Release()

    for j in range(n_imgs):
        image_result = cam.GetNextImage(2000)

        if image_result.IsIncomplete():
            print(
                "Image incomplete with image status %d ..."
                % image_result.GetImageStatus()
            )

        img = image_result.GetNDArray()
        img_stack[i, j] = img
        image_result.Release()

    plt.imsave(
        os.path.join(pth, f"img_{pos:.4f}.png"),
        img,
        vmin=0,
        vmax=255,
        cmap="gray",
    )

cam.EndAcquisition()

np.savez(
    os.path.join(pth, "img_stack.npz"),
    img_stack=img_stack,
    positions=positions,
    n_imgs=n_imgs,
)

# np.save(os.path.join(pth, "img_stack.npy"), img_stack)
del cam
cam_list.Clear()
system.ReleaseInstance()
