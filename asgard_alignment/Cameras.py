import PySpin


class PointGrey:

    EDITABLE_PARAMS = [
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
        """
        Initialize the PointGrey camera.

        Parameters
        ----------
        cam_index : int, optional
            Index of the camera to initialize, by default 0
        """
        self.system = PySpin.System.GetInstance()
        self.cam_list = self.system.GetCameras()
        self.cam = self.cam_list.GetByIndex(cam_index)
        self.cam.Init()

        self._set_default_configs()

    def _set_default_configs(self):
        """
        Set the default configurations for the camera.
        """
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
        """
        Start the camera acquisition stream.
        """
        self.cam.AcquisitionFrameRate.SetValue(self.cam.AcquisitionFrameRate.GetMax())
        self.cam.BeginAcquisition()

    def stop_stream(self):
        """
        Stop the camera acquisition stream.
        """
        self.cam.EndAcquisition()

    def get_frame(self):
        """
        Get the next image frame from the camera.

        Returns
        -------
        numpy.ndarray
            The next image frame as a numpy array.
        """
        img = self.cam.GetNextImage()
        img_numpy = img.GetNDArray().copy()
        img.Release()
        return img_numpy

    def __setattr__(self, name, value):
        """
        Set the attribute of the camera.

        Parameters
        ----------
        name : str
            The name of the attribute to set.
        value : Any
            The value to set the attribute to.
        """
        if name not in self.EDITABLE_PARAMS:
            super().__setattr__(name, value)
            return

        if value == "max":
            value = getattr(self.cam, name).GetMax()
        elif value == "min":
            value = getattr(self.cam, name).GetMin()
        getattr(self.cam, name).SetValue(value)

    def __getitem__(self, key):
        """
        Get the value of a camera parameter.

        Parameters
        ----------
        key : str
            The name of the parameter to get.

        Returns
        -------
        Any
            The value of the parameter.

        Raises
        ------
        KeyError
            If the key is not a valid parameter.
        """
        if key not in self.EDITABLE_PARAMS:
            raise KeyError(f"Invalid key {key}")
        return getattr(self.cam, key).GetValue()

    def __setitem__(self, key, value):
        """
        Set the value of a camera parameter.

        Parameters
        ----------
        key : str
            The name of the parameter to set.
        value : Any
            The value to set the parameter to.

        Raises
        ------
        KeyError
            If the key is not a valid parameter.
        """
        if key not in self.EDITABLE_PARAMS:
            raise KeyError(f"Invalid key {key}")

        if value == "max":
            value = getattr(self.cam, key).GetMax()
        elif value == "min":
            value = getattr(self.cam, key).GetMin()
        getattr(self.cam, key).SetValue(value)

    def set_region_from_corners(self, x1, y1, x2, y2):
        """
        Set the region of interest from the top left and bottom right corners.

        Parameters
        ----------
        x1 : int
            The x coordinate of the top left corner.
        y1 : int
            The y coordinate of the top left corner.
        x2 : int
            The x coordinate of the bottom right corner.
        y2 : int
            The y coordinate of the bottom right corner.
        """
        x1, x2 = sorted([x1, x2])
        y1, y2 = sorted([y1, y2])

        self["Width"] = x2 - x1
        self["Height"] = y2 - y1
        self["OffsetX"] = x1
        self["OffsetY"] = y1

    def release(self):
        """
        Release the camera resources.
        """
        self.cam.DeInit()
        del self.cam
        self.cam_list.Clear()
        self.system.ReleaseInstance()

    @property
    def camera(self):
        """
        Get the camera instance.

        Returns
        -------
        PySpin.Camera
            The camera instance.
        """
        return self.cam

    @property
    def img_size(self):
        """
        Get the image size.

        Returns
        -------
        tuple
            The height and width of the image.
        """
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

    # now test cropping using ginput clicks
    import matplotlib.pyplot as plt

    c["OffsetX"] = 0
    c["OffsetY"] = 0
    c["Width"] = "max"
    c["Height"] = "max"

    c.start_stream()
    img = c.get_frame()
    c.stop_stream()

    plt.imshow(img, cmap="gray")
    plt.title("Click on the top left and bottom right of the region of interest")

    pts = plt.ginput(2)
    plt.close()

    x1, y1 = pts[0]
    x2, y2 = pts[1]

    c.set_region_from_corners(int(x1), int(y1), int(x2), int(y2))

    c.start_stream()
    img = c.get_frame()
    c.stop_stream()

    # should be close to targets to within 5 pixels
    assert abs(c["Width"] - int(x2 - x1)) <= 5
    assert abs(c["Height"] - int(y2 - y1)) <= 5

    assert abs(c["OffsetX"] - int(x1)) <= 5
    assert abs(c["OffsetY"] - int(y1)) <= 5

    c.release()
