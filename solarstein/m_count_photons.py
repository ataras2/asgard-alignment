import PySpin
import matplotlib.pyplot as plt
import numpy as np
import os


pth = "data/solarstein/photon_count"

beam = 4
n_imgs = 100
plot = True

if not os.path.exists(pth):
    os.makedirs(pth)

system = PySpin.System.GetInstance()
cam_list = system.GetCameras()
cam = cam_list[0]

nodemap_tldevice = cam.GetTLDeviceNodeMap()


input("turn laser off and press enter")


# Initialize camera
cam.Init()

# Retrieve GenICam nodemap
nodemap = cam.GetNodeMap()
cam.BeginAcquisition()

dark_stack = []
for i in range(n_imgs):
    print(f"Image {i}/{n_imgs}")
    image_result = cam.GetNextImage(1000)
    image_result.Release()

    dark_stack.append(image_result.GetNDArray())

cam.EndAcquisition()
dark_stack = np.array(dark_stack, dtype=np.float32)

input("turn laser on and press enter")

cam.BeginAcquisition()

img_stack = []

for i in range(n_imgs):
    print(f"Image {i}/{n_imgs}")
    image_result = cam.GetNextImage(1000)
    image_result.Release()

    img_stack.append(image_result.GetNDArray())

cam.EndAcquisition()

img_stack = np.array(img_stack, dtype=np.float32)

print(f"image flux = {np.sum(img_stack-dark_stack)}")

# append result to file
with open(f"{pth}/photon_count.txt", "a") as f:
    f.write(f"beam {beam} ({n_imgs} imgs): {np.sum(img_stack-dark_stack)}\n")

if plot:
    plt.subplot(1, 3, 1)
    plt.imshow(np.mean(img_stack, axis=0))
    plt.title("Mean image")
    plt.colorbar()

    plt.subplot(1, 3, 2)
    plt.imshow(np.mean(dark_stack, axis=0))
    plt.title("Mean dark")
    plt.colorbar()

    plt.subplot(1, 3, 3)
    plt.imshow(np.mean(img_stack - dark_stack, axis=0))
    plt.title("Mean image - dark")
    plt.colorbar()

    plt.show()

del cam
cam_list.Clear()
system.ReleaseInstance()
