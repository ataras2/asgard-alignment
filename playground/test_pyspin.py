import PySpin


system = PySpin.System.GetInstance()
cam_list = system.GetCameras()

print(len(cam_list), "cameras found")

exit()
# cam.start_stream()
# img = cam.get_frame()
# cam.stop_stream()

# print(f"Image shape: {img.shape}")
# print(f"image max: {img.max()}")
