from zaber_motion.ascii import Connection
from zaber_motion import Units

import PySpin

import numpy as np

import matplotlib.pyplot as plt
import os
import time

pth = "data/may/heimdallr_34_run1"

BS_num = 9

if not os.path.exists(pth):
    os.makedirs(pth)

start_pos = 0  # um
end_pos = 2000  # um
step_size = 5  # um
# start_pos = 5000  # um
# end_pos = 8500  # um
# step_size = 5  # um


positions = list(range(start_pos, end_pos, step_size))

n_imgs = 5


if BS_num == 1:
    axis = 4
elif BS_num == 7:
    axis = 3
elif BS_num == 9:
    axis = 2
else:
    raise ValueError("BS_num must be 1, 7, or 9")

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

img_stack = np.zeros((n_positions, n_imgs, img.shape[0], img.shape[1]), dtype=np.uint8)

with Connection.open_tcp("zaber-120408.local", Connection.TCP_PORT_CHAIN) as connection:
    # setup motion device
    device_list = connection.detect_devices()
    print("Found {} devices".format(len(device_list)))

    device = device_list[0]

    axis = device.get_axis(axis)
    if not axis.is_homed():
        print("homing...")
        axis.home()
        print("done")

    # move and take photos

    for i, pos in enumerate(positions):
        print(f"\rMoving to {pos} um ({i+1}/{len(positions)})", end="")
        axis.move_absolute(pos, Units.LENGTH_MICROMETRES)

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
    BS_num=BS_num,
)

# np.save(os.path.join(pth, "img_stack.npy"), img_stack)
del cam
cam_list.Clear()
system.ReleaseInstance()
