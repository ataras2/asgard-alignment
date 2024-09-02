import pyvisa
import parse


class Motor:
    """
    Base class for all the newport motors
    """

    # The serial config for the newport motors:
    SERIAL_BAUD = 921600
    SERIAL_TERMIN = "\r\n"

    def __init__(self, serial_port: str, resource_manager: pyvisa.ResourceManager):
        self._serial_port = serial_port
        self.open_connection(resource_manager)
        self._verify_valid_connection()

    def open_connection(self, resource_manager: pyvisa.ResourceManager):
        """
        resource_manager : pyvisa.ResourceManager object (to avoid constructing it many times)
        """
        self._connection = resource_manager.open_resource(
            self._serial_port,
            baud_rate=self.SERIAL_BAUD,
            write_termination=self.SERIAL_TERMIN,
            read_termination=self.SERIAL_TERMIN,
        )

    def _verify_valid_connection(self):
        raise NotImplementedError()

    def write_str(self, str_to_write):
        """
        Write a string through serial and do not expect anything to be returned

        Parameters:
        -----------
        str_to_write: str
            The string to write to the serial port
        """
        self._connection.write(str_to_write)

    def query_str(self, str_to_write):
        """
        Send a query through serial and return the response

        Parameters:
        -----------
        str_to_write: str
            The string to write to the serial port

        Returns:
        --------
        return_str: str
            The string returned from the serial port
        """
        return_str = self._connection.query(str_to_write).strip()
        return return_str

    def set_to_zero(self):
        """
        Set the motor to the zero position
        """
        raise NotImplementedError()

    @classmethod
    def validate_config(cls, config):
        """
        Validate the config dictionary for the motor
        """
        pass

    @staticmethod
    def setup_individual_config():
        raise NotImplementedError()

    @staticmethod
    def infer_motor_type(motor_name):
        """
        Given the internal name of the motor, attempt to infer the type of the class to instantiate

        Parameters:
        -----------
        motor_name: str
            The internal name of the motor

        Returns:
        --------
        motor_type: type
            The python type of the motor to instantiate
        """

        motor_type = None
        ending = motor_name.split("_")[-1]
        if ending.lower() == "tiptilt" or ending.lower() == "m100d":
            motor_type = M100D
        elif ending.lower() == "linear" or ending.lower() == "ls16p":
            motor_type = LS16P

        if motor_type is None:
            raise KeyError(f"could not infer motor type from {motor_name}")
        return motor_type

    @staticmethod
    def motor_type_to_string(motor_type):
        """
        Convert the motor type to a string

        Parameters:
        -----------
        motor_type: type
            The python type of the motor

        Returns:
        --------
        motor_str: str
            The string representation of the motor (to use for e.g. saving to a config file)
        """
        m = None
        if motor_type == M100D:
            m = "M100D"
        elif motor_type == LS16P:
            m = "LS16P"

        if m is None:
            raise ValueError(f"Could not find motor from {motor_type}")

        return m

    @staticmethod
    def string_to_motor_type(motor_str):
        """
        Convert the motor string to a type

        Parameters:
        -----------
        motor_str: str
            The string representation of the motor (to use for e.g. saving to a config file)

        Returns:
        --------
        motor_type: Motor
            The python type of the motor
        """
        m = None
        if motor_str.lower() == "m100d":
            m = M100D
        elif motor_str.lower() == "ls16p":
            m = LS16P

        if m is None:
            raise ValueError(f"Could not find motor from {motor_str}")

        return m


class LS16P(Motor):
    """
    A linear motor driver class
    https://www.newport.com/p/CONEX-SAG-LS16P
    """

    HW_BOUNDS = [-8.0, 8.0]

    def __init__(self, serial_port: str, resource_manager: pyvisa.ResourceManager):
        super().__init__(serial_port, resource_manager)
        self._current_pos = 0.0

    def _verify_valid_connection(self):
        """
        Verify that the serial connection opened by the class is indeed to to a NEWPORT LS16P
        """
        id_number = self._connection.query("1ID?").strip()
        assert "LS16P" in id_number

    def set_absolute_position(self, value: float):
        """
        Set the absolute position of the motor

        Parameters:
            value (float) : The new position in mm
        """
        str_to_write = f"1PA{value}"
        self._connection.write(str_to_write)
        self._current_pos = value

    def read_pos(self) -> float:
        """
        Set the absolute position of the motor

        Returns:
            value (float) : The new position in mm
        """
        return_str = self._connection.query("1TP").strip()
        subset = parse.parse("{}TP{}", return_str)
        if subset is not None:
            return float(subset[1])
        raise ValueError(f"Could not parse {return_str}")

    def set_to_zero(self):
        """
        Set the motor to the zero position
        """
        self.set_absolute_position(0.0)

    @property
    def get_current_pos(self):
        """
        Return the software internal position of the motor
        """
        return self._current_pos

    @staticmethod
    def setup_individual_config():
        return {}


import PySpin

import numpy as np

import matplotlib.pyplot as plt
import os
import time


pth = "data/early_sept/heimdallr_13_run0_sld"


# make motor
# beam 4 : asrl5
# beam1 : asrl31
motor = LS16P("ASRL31", pyvisa.ResourceManager())
print(motor.read_pos())
print(motor.query_str("SA?"))

assert motor.query_str("SA?") == "SA1"

motor.write_str("OR")
motor.write_str("RFP")

time.sleep(2)
motor.set_absolute_position(7.888)
exit()

middle = 8 - 1


start_pos = middle - 3  # mm
end_pos = middle + 3  # mm
step_size = 5e-3  # mm
# step_size = 0.9  # mm
# start_pos = 5000  # um
# end_pos = 8500  # um
# step_size = 5  # um

motor.set_absolute_position(start_pos)
input("Press enter to start")
motor.set_absolute_position(end_pos)
input("Press enter to start")

if not os.path.exists(pth):
    os.makedirs(pth)
else:
    inp = input("path exists (possible overwrite!), press y to continue")
    if inp.lower() != "y":
        exit()


# motor.set_absolute_position(start_pos)
# time.sleep(2)
# motor.set_absolute_position(end_pos)

# exit()
# positions = list(range(start_pos, end_pos, step_size))
positions = np.round(np.arange(start_pos, end_pos, step_size), 5)

n_imgs = 5


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
image_result = cam.GetNextImage(2000)
image_result.Release()

img = image_result.GetNDArray()
n_positions = len(positions)

img_stack = np.zeros((n_positions, n_imgs, img.shape[0], img.shape[1]), dtype=np.uint8)


for i, pos in enumerate(positions):
    print(f"\rMoving to {pos} um ({i+1}/{len(positions)})", end="")
    # axis.move_absolute(pos, Units.LENGTH_MICROMETRES)
    motor.set_absolute_position(pos)

    time.sleep(0.2)

    image_result = cam.GetNextImage(2000)
    image_result.Release()

    for j in range(n_imgs):
        image_result = cam.GetNextImage(2000)

        if image_result.IsIncomplete():
            print(
                "Image incomplete with image status %d ..."
                % image_result.GetImageStatus()
            )

        img = image_result.GetNDArray()
        img_stack[i, j] = img
        image_result.Release()

    plt.imsave(
        os.path.join(pth, f"img_{pos:.4f}.png"),
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
)

# np.save(os.path.join(pth, "img_stack.npy"), img_stack)
del cam
cam_list.Clear()
system.ReleaseInstance()
