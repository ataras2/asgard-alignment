import argparse
from zaber_motion.ascii import Connection
from zaber_motion import Units
import PySpin
import numpy as np
import matplotlib.pyplot as plt
import os
import time
import zmq
from tqdm import tqdm


def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Motion and Save Script")
    parser.add_argument(
        "--savepath", type=str, required=True, help="Path to save images and data"
    )
    # parser.add_argument(
    #     "--bs_num",
    #     type=int,
    #     choices=[1, 7, 9],
    #     required=True,
    #     help="BS number (1, 7, or 9)",
    # )
    parser.add_argument(
        "--axis", type=str, required=True, help="Axis number (SDL12, SDLA, SDL34)"
    )
    parser.add_argument(
        "--start", type=int, required=True, help="Start position in micrometers"
    )
    parser.add_argument(
        "--end", type=int, required=True, help="End position in micrometers"
    )
    parser.add_argument(
        "--step", type=int, required=True, help="Step size in micrometers"
    )
    parser.add_argument(
        "--n_imgs",
        type=int,
        default=3,
        help="Number of images to capture at each position (default: 3)",
    )

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

    args = parser.parse_args()

    pth = args.savepath
    start_pos = args.start
    end_pos = args.end
    step_size = args.step
    n_imgs = args.n_imgs
    axis = args.axis

    if not os.path.exists(pth):
        os.makedirs(pth)

    # positions = list(range(start_pos, end_pos, step_size))
    positions = np.arange(start_pos, end_pos, step_size, dtype=float)

    # move to start and check
    print(f"sending !moveabs {axis} {positions[0]}")
    socket.send_string(f"!moveabs {axis} {positions[0]}")
    res = socket.recv_string()
    print(f"Response: {res}")
    input("check we are at start")
    print(f"sending !moveabs {axis} {positions[-1]}")
    socket.send_string(f"!moveabs {axis} {positions[-1]}")
    res = socket.recv_string()
    print(f"Response: {res}")
    input("check we are at end")

    # if BS_num == 1:
    #     axis_num = 4
    # elif BS_num == 7:
    #     axis_num = 3
    # elif BS_num == 9:
    #     axis_num = 2
    # else:
    #     raise ValueError("BS_num must be 1, 7, or 9")

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
    image_result = cam.GetNextImage(1000)
    image_result.Release()

    img = image_result.GetNDArray()
    n_positions = len(positions)

    img_stack = np.zeros(
        (n_positions, n_imgs, img.shape[0], img.shape[1]), dtype=np.uint8
    )

    # move and take photos
    for i, pos in enumerate(tqdm(positions)):
        # print(f"\rMoving to {pos} um ({i+1}/{len(positions)})", end="")
        # axis.move_absolute(pos, Units.LENGTH_MICROMETRES)

        socket.send_string(f"!moveabs {axis} {pos}")
        res = socket.recv_string()

        time.sleep(0.1)

        image_result = cam.GetNextImage(1000)
        image_result.Release()

        for j in range(n_imgs):
            image_result = cam.GetNextImage(1000)

            if image_result.IsIncomplete():
                print(
                    "Image incomplete with image status %d ..."
                    % image_result.GetImageStatus()
                )

            img = image_result.GetNDArray()
            img_stack[i, j] = img
            image_result.Release()

        plt.imsave(
            os.path.join(pth, f"img_{pos}.png"),
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
        # BS_num=BS_num,
        axis=axis,
    )

    # np.save(os.path.join(pth, "img_stack.npy"), img_stack)
    del cam
    cam_list.Clear()
    system.ReleaseInstance()


if __name__ == "__main__":
    main()
