from abc import ABC, abstractmethod


class AsgardDevice(ABC):
    """
    A class to define the interface for the Asgard devices.
    """

    def __init__(self):
        """
        Create the AsgardDevice object.
        """

    @abstractmethod
    def initialise(self):
        """
        Do all the initialisation for the device. This can include homing,
        setting the current position, etc.
        """

    @abstractmethod
    def set_position(self, position):
        """
        Set the position of the device, in the units of the device.
        """

    @abstractmethod
    def get_position(self):
        """
        Get the current position of the device, in the units of the device.
        """
