import PySpin
import numpy as np


class CameraStream:
    def __init__(self):
        self.system = PySpin.System.GetInstance()
        self.cam_list = self.system.GetCameras()
        self.cam = self.cam_list.GetByIndex(0)

        self.cam.Init()
        self.cam.UserSetSelector.SetValue(PySpin.UserSetSelector_Default)
        self.cam.UserSetLoad()
        self.cam.AcquisitionMode.SetValue(PySpin.AcquisitionMode_Continuous)
        self.cam.ExposureAuto.SetValue(PySpin.ExposureAuto_Off)
        self.cam.ExposureMode.SetValue(
            PySpin.ExposureMode_Timed
        )  # Timed or TriggerWidth (must comment out trigger parameters other that Line)

        self.cam.GainAuto.SetValue(PySpin.GainAuto_Off)
        self.cam.Gain.SetValue(0.00)
        self.cam.BeginAcquisition()

    def get_frame(self):
        image_result = self.cam.GetNextImage()
        frame = np.array(image_result.GetData(), dtype="uint8").reshape(
            (image_result.GetHeight(), image_result.GetWidth())
        )
        image_result.Release()

        return frame

    def set_exposure_time(self, new_time):
        self.cam.ExposureTime.SetValue(new_time)

    def get_exposure_time(self):
        try:
            return self.cam.ExposureTime.GetValue()
        except:
            return -1

    def release(self):
        self.cam.EndAcquisition()
        self.cam.DeInit()
        self.cam_list.Clear()
        self.system.ReleaseInstance()


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    camera = CameraStream()

    frame = camera.get_frame()
    plt.subplot(121)
    plt.imshow(frame)
    plt.colorbar()

    camera.set_exposure_time(1000)
    print(camera.get_exposure_time())
    camera.set_exposure_time(5000)
    print(camera.get_exposure_time())

    frame = camera.get_frame()
    plt.subplot(122)
    plt.imshow(frame)
    plt.colorbar()

    plt.show()
    camera.release()
