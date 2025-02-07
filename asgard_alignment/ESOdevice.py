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
    def __init__(self, name, semaphore_id) -> None:
        super().__init__()

        self.name = name
        self.semaphore_id = semaphore_id


class SetupCommand:
    def __init__(self, device_name, m_type, value) -> None:
        self.device_name = device_name
        self.m_type = m_type
        self.value = value


class Motor(ESOdevice):
    """
    that in IC0FB, all motors are considered as “discrete”. A discrete motor can be set to any
    given encoder position, so continuous motors are actually discrete motors with no named
    positions defined.

    This class covers both continuous and discrete motors.
    """

    def __init__(self, name, semaphore_id, named_positions={}) -> None:
        super().__init__(name, semaphore_id)
        self._named_positions = named_positions

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

    # note that the stop command is common to both command sets!
    @abc.abstractmethod
    def stop(self):
        """
        The STOP command is issued by the ICS on wag to immediately stop the motion (initiated by a
        SETUP command) of devices.
        """

    @abc.abstractmethod
    def ping(self):
        """
        The PING command is used to check the status of the controller,
        sending a dummy command such as *IDN? and making sure there is a reply
        """

    #######################################################
    # functions for compatibility with the ESO command set
    #######################################################
    @abc.abstractmethod
    def setup(self, value: Union[str, float]):
        """
        Command to move a device to a given position. The position is given in the value field.
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
    def __init__(self, name, semaphore_id) -> None:
        super().__init__(name, semaphore_id)

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
