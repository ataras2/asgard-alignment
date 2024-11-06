from typing import Any
import PySpin


class PointGrey:
    def __init__(self, cam_index=0):
        self.system = PySpin.System.GetInstance()
        self.cam_list = self.system.GetCameras()
        self.cam = self.cam_list.GetByIndex(cam_index)
        self.cam.Init()

        self._set_default_configs()

    def _set_default_configs(self):
        self.cam.AcquisitionMode.SetValue(PySpin.AcquisitionMode_Continuous)
        self.cam.ExposureMode.SetValue(PySpin.ExposureMode_Timed)

        self.cam.ExposureAuto.SetValue(PySpin.ExposureAuto_Off)
        self.cam.ExposureTime.SetValue(1000)

        self.cam.GainAuto.SetValue(PySpin.GainAuto_Off)
        self.cam.Gain.SetValue(0)

        # remove offsets, restore full frame size
        self.cam.OffsetX.SetValue(0)
        self.cam.OffsetY.SetValue(0)
        self.cam.Width.SetValue(self.cam.Width.GetMax())
        self.cam.Height.SetValue(self.cam.Height.GetMax())

        # frame rate
        self.cam.AcquisitionFrameRateEnable.SetValue(True)
        self.cam.AcquisitionFrameRate.SetValue(self.cam.AcquisitionFrameRate.GetMax())

    def start_stream(self):
        self.cam.BeginAcquisition()

    def stop_stream(self):
        self.cam.EndAcquisition()

    def get_frame(self):
        frame = self.cam.GetNextImage()
        image = frame.GetNDArray()
        frame.Release()
        return image

    def release(self):
        self.cam.DeInit()
        del self.cam
        self.cam_list.Clear()
        self.system.ReleaseInstance()

    def __setattr__(self, name: str, value: Any) -> None:
        if value in ["max", "min"]:
            if value == "max":
                value = self[name].GetMax()
            else:
                value = self[name].GetMin()

        self.cam.__getattribute__(name).SetValue(value)

    def __getattr__(self, name: str) -> Any:
        return self.cam.__getattribute__(name)

    @property
    def camera(self):
        return self.cam

    @property
    def img_size(self):
        return (self.cam.Width.GetValue(), self.cam.Height.GetValue())


if __name__ == "__main__":
    # run some tests quickly

    cam = PointGrey()

    cam.start_stream()
    img = cam.get_frame()
    cam.stop_stream()

    assert img.shape == cam.img_size

    # try setting exposure, gain, offset and frame rate
    cur_exposure_time = cam["ExpousreTime"]

    cam["ExposureTime"] = cur_exposure_time + 1000

    assert cam["ExposureTime"] >= cur_exposure_time

    cam["Gain"] = "max"
    cur_gain = cam["Gain"]
    cam["Gain"] = "min"

    assert cam["Gain"] <= cur_gain

    # height and width
    cur_width = cam["Width"]
    cur_height = cam["Height"]

    cam["Width"] = cur_width // 2
    cam["Height"] = cur_height // 2

    # should be about half the size (up to discreitation, no larger than 5 pixels diff)
    assert abs(cam["Width"] - cur_width // 2) <= 5
    assert abs(cam["Height"] - cur_height // 2) <= 5

    # frame rate
    cur_frame_rate = cam["AcquisitionFrameRate"]
    cam["AcquisitionFrameRate"] = "min"

    assert cam["AcquisitionFrameRate"] <= cur_frame_rate

    cam["AcquisitionFrameRate"] = "max"

    cam.release()
