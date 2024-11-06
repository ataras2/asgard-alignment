from typing import Any
import PySpin


class PointGrey:

    PARAMS = [
        "AcquisitionFrameRate",
        "ExposureTime",
        "Gain",
        "Height",
        "Width",
        "PixelFormat",
        "ExposureAuto",
        "GainAuto",
        "AcquisitionMode",
        "ExposureMode",
        "OffsetX",
        "OffsetY",
    ]

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

        # pixel format
        self.cam.PixelFormat.SetValue(PySpin.PixelFormat_Mono8)

        # remove offsets, restore full frame size
        self.cam.OffsetX.SetValue(0)
        self.cam.OffsetY.SetValue(0)
        self.cam.Width.SetValue(self.cam.Width.GetMax())
        self.cam.Height.SetValue(self.cam.Height.GetMax())

        # frame rate stuff
        # self.cam.AcquisitionFrameRateEnable.SetValue(True)
        self.cam.AcquisitionFrameRate.SetValue(self.cam.AcquisitionFrameRate.GetMax())

    def start_stream(self):
        self.cam.BeginAcquisition()

    def stop_stream(self):
        self.cam.EndAcquisition()

    def get_frame(self):
        img = self.cam.GetNextImage()
        img = img.GetNDArray()
        return img

    def __setattr__(self, name, value):
        if name not in self.PARAMS:
            super().__setattr__(name, value)
            return

        if value == "max":
            value = getattr(self.cam, name).GetMax()
        elif value == "min":
            value = getattr(self.cam, name).GetMin()
        getattr(self.cam, name).SetValue(value)

    def __getitem__(self, key):
        if key not in self.PARAMS:
            raise KeyError(f"Invalid key {key}")
        return getattr(self.cam, key).GetValue()

    def __setitem__(self, key, value):
        if key not in self.PARAMS:
            raise KeyError(f"Invalid key {key}")

        if value == "max":
            value = getattr(self.cam, key).GetMax()
        elif value == "min":
            value = getattr(self.cam, key).GetMin()
        getattr(self.cam, key).SetValue(value)

    def release(self):
        self.cam.DeInit()
        del self.cam
        self.cam_list.Clear()
        self.system.ReleaseInstance()

    @property
    def camera(self):
        return self.cam

    @property
    def img_size(self):
        return (self.cam.Height.GetValue(), self.cam.Width.GetValue())


if __name__ == "__main__":
    # run some tests quickly

    c = PointGrey()

    c.start_stream()
    img = c.get_frame()
    c.stop_stream()

    assert img.shape == c.img_size

    # try setting exposure, gain, offset and frame rate
    cur_exposure_time = c["ExposureTime"]

    c["ExposureTime"] = cur_exposure_time + 1000

    assert c["ExposureTime"] >= cur_exposure_time

    c["Gain"] = "max"
    cur_gain = c["Gain"]
    c["Gain"] = "min"

    assert c["Gain"] <= cur_gain

    # height and width
    cur_width = c["Width"]
    cur_height = c["Height"]

    c["Width"] = cur_width // 2
    c["Height"] = cur_height // 2

    # should be about half the size (up to discreitation, no larger than 5 pixels diff)
    assert abs(c["Width"] - cur_width // 2) <= 5
    assert abs(c["Height"] - cur_height // 2) <= 5

    # check the image size has actually changed
    c.start_stream()
    img = c.get_frame()
    c.stop_stream()

    assert img.shape == c.img_size

    # frame rate
    cur_frame_rate = c["AcquisitionFrameRate"]
    c["AcquisitionFrameRate"] = "min"

    assert c["AcquisitionFrameRate"] <= cur_frame_rate

    c["AcquisitionFrameRate"] = "max"

    c.release()
