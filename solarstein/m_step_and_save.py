import argparse
import PySpin
import numpy as np
import matplotlib.pyplot as plt
import os
import asgard_alignment.Cameras
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

    parser.add_argument("--host", type=str, default="192.168.100.2", help="Server host")
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

    if axis not in ["SDL12", "SDLA", "SDL34"]:
        raise ValueError("Invalid axis")

    if not os.path.exists(pth):
        os.makedirs(pth)

    # positions = list(range(start_pos, end_pos, step_size))
    positions = np.arange(start_pos, end_pos, step_size, dtype=float)

    # move to start and check
    print(f"sending !moveabs {axis} {positions[0]}")
    socket.send_string(f"moveabs {axis} {positions[0]}")
    res = socket.recv_string()
    print(f"Response: {res}")
    input("check we are at start")
    print(f"sending !moveabs {axis} {positions[-1]}")
    socket.send_string(f"moveabs {axis} {positions[-1]}")
    res = socket.recv_string()
    print(f"Response: {res}")
    input("check we are at end")

    socket.send_string(f"moveabs {axis} {positions[0]}")
    res = socket.recv_string()

    # setup camera
    cam = asgard_alignment.Cameras.PointGrey()

    # take a photo and ask the user to crop the image using ginput
    cam.start_stream()
    img = cam.get_frame()
    cam.stop_stream()

    plt.imshow(img, cmap="gray")
    plt.title("Click on the top left and bottom right of the region of interest")

    pts = plt.ginput(2)
    plt.close()

    x1, y1 = pts[0]
    x2, y2 = pts[1]

    x1, x2 = sorted([x1, x2])
    y1, y2 = sorted([y1, y2])

    cam.set_region_from_corners(int(x1), int(y1), int(x2), int(y2))

    cam.start_stream()
    img = cam.get_frame()
    cam.stop_stream()

    n_positions = len(positions)

    img_stack = np.zeros(
        (n_positions, n_imgs, img.shape[0], img.shape[1]), dtype=np.uint8
    )

    if img.shape[0] > 550:
        # raise warning about large image size
        print("Warning: Image size is large, consider reducing the ROI size")

    if img.shape[0] < 100:
        # raise warning about small image size
        print("Warning: Image size is small, consider increasing the ROI size")

    n_positions = len(positions)

    img_stack = np.zeros(
        (n_positions, n_imgs, img.shape[0], img.shape[1]), dtype=np.uint8
    )

    cam.start_stream()
    # move and take photos
    for i, pos in enumerate(tqdm(positions)):
        # print(f"\rMoving to {pos} um ({i+1}/{len(positions)})", end="")
        # axis.move_absolute(pos, Units.LENGTH_MICROMETRES)

        socket.send_string(f"moveabs {axis} {pos}")
        res = socket.recv_string()

        time.sleep(0.1)

        for j in range(n_imgs):
            img = cam.get_frame()

            img_stack[i, j] = img

        plt.imsave(
            os.path.join(pth, f"img_{pos}.png"),
            img,
            vmin=0,
            vmax=255,
            cmap="gray",
        )

    cam.stop_stream()

    np.savez(
        os.path.join(pth, "img_stack.npz"),
        img_stack=img_stack,
        positions=positions,
        n_imgs=n_imgs,
        # BS_num=BS_num,
        axis=axis,
    )

    # np.save(os.path.join(pth, "img_stack.npy"), img_stack)
    cam.release()


if __name__ == "__main__":
    main()
