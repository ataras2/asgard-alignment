import PySpin

system = PySpin.System.GetInstance()

cam_list = system.GetCameras()


print()
for cam in cam_list:
    cam.Init()

    print(f"Camera {cam.DeviceID()} is initialized")
