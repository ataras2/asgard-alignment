"""
Scaffolds for devices to run with ESO command sets
"""

# abstract base class
import abc
from enum import Enum

import math
from datetime import datetime

from typing import Union


class SemaphoreState(Enum):
    RELEASED = 0
    TAKEN = 1


class ESOdevice(abc.ABC):
    def __init__(self, name) -> None:
        super().__init__()

        self.name = name


class SetupCommand:
    def __init__(self, device_name, motion_type, value) -> None:
        self.device_name = device_name
        self.motion_type = motion_type
        if motion_type in ["ENC","ENCREL"]:
            value = float(value)
        self.value = value


class Motor(ESOdevice):
    """
    that in IC0FB, all motors are considered as “discrete”. A discrete motor can be set to any
    given encoder position, so continuous motors are actually discrete motors with no named
    positions defined.

    This class covers both continuous and discrete motors.
    """

    def __init__(self, name, semaphore_id, named_positions=None) -> None:
        super().__init__(
            name,
        )
        if named_positions is None:
            named_positions = {}
        self.named_positions = named_positions
        self.semaphore_id = semaphore_id

    #######################################################
    # functions for compatibility with custom command set
    #######################################################

    @abc.abstractmethod
    def move_abs(self, position: float):
        pass

    @abc.abstractmethod
    def move_relative(self, position: float):
        pass

    @abc.abstractmethod
    def read_state(self):
        pass

    @abc.abstractmethod
    def read_position(self):
        """
        This command is used to read the current position of the device.
        It should return a float value representing the position in encoder counts.
        """
        pass

    @abc.abstractmethod
    def ping(self):
        """
        The PING command is used to check the status of the controller,
        sending a dummy command such as *IDN? and making sure there is a reply
        """

    # note that the stop command is common to both command sets!
    @abc.abstractmethod
    def stop(self):
        """
        The STOP command is issued by the ICS on wag to immediately stop the motion (initiated by a
        SETUP command) of devices.
        """

    #######################################################
    # functions for compatibility with the ESO command set
    #######################################################
    @abc.abstractmethod
    def ESO_read_position(self):
        """
        This command is used to update the ESO database back end.
        Hence, it always must return an int.
        """
        pass

    @abc.abstractmethod
    def is_moving(self):
        """
        This command is used to check if the device is moving.
        Hence, it always must return a boolean.
        """
        pass

    @abc.abstractmethod
    def setup(self, motion_type: str, value: Union[str, float]):
        """
        Command to move a device to a given position. The position is given in the value field.

        The motion_type field indicates the type of motion to be performed. It can take the following
        values:
        - "ENC": The value field is an absolute position in encoder counts.
        - "ENCREL": The value field is a relative position in encoder counts.
        - "NAME": The value field is a named position, in which case the value field is a string
        - "ST": state, equal to either "T" or "F" as a string
        """
        pass

    @abc.abstractmethod
    def disable(self):
        """
        The DISABLE command can be used to request the MCU to power off devices.
        """
        pass

    @abc.abstractmethod
    def enable(self):
        """
        The ENABLE command can be used to request the MCU to power on devices.
        """
        pass

    @abc.abstractmethod
    def online(self):
        """
        Upon reception, the ICS back-end server of the MCU shall power on all the
        controlled devices and have them ready to accept SETUP commands.
        """
        pass

    @abc.abstractmethod
    def standby(self):
        """
        Upon reception, the ICS
        back-end server of the MCU shall move some of the controlled devices to a safe “parking”
        position (if required) and power off all the controlled devices.
        """
        pass


class Lamp(ESOdevice):
    def __init__(self, name) -> None:
        super().__init__(name)

    @abc.abstractmethod
    def is_on(self):
        pass

    @abc.abstractmethod
    def is_off(self):
        pass

    @abc.abstractmethod
    def turn_on(self):
        pass

    @abc.abstractmethod
    def turn_off(self):
        pass
