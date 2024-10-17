"""
Module for the newport motors.
"""

from enum import Enum
import logging
from typing import Literal
import parse
import numpy as np
import streamlit as st
import asgard_alignment.GUI
import time

import pyvisa
import asgard_alignment.ESOdevice as ESOdevice


class NewportConnection:
    """
    A class to handle the connection to the newport motors
    One per controller
    """

    # The serial config for the newport motors:
    SERIAL_BAUD = 921600
    SERIAL_TERMIN = "\r\n"

    def __init__(self, serial_port: str, resource_manager: pyvisa.ResourceManager):
        self._serial_port = serial_port
        self.open_connection(resource_manager)

    def open_connection(self, resource_manager: pyvisa.ResourceManager):
        """
        resource_manager : pyvisa.ResourceManager object (to avoid constructing it many times)
        """
        self._connection = resource_manager.open_resource(
            f"ASRL{self._serial_port}::INSTR",
            # baud_rate=self.SERIAL_BAUD,
            # write_termination=self.SERIAL_TERMIN,
            # read_termination=self.SERIAL_TERMIN,
        )

        self._connection.baud_rate = self.SERIAL_BAUD
        self._connection.write_termination = self.SERIAL_TERMIN
        self._connection.read_termination = self.SERIAL_TERMIN

    def close_connection(self):
        """
        Close the connection to the motor
        """
        self._connection.before_close()
        self._connection.close()

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


class M100DAxis(ESOdevice.Motor):
    """
    A class for the tip or tilt M100D motors
    https://www.newport.com.cn/p/CONEX-AG-M100D
    """

    def __init__(
        self,
        connection: NewportConnection,
        axis: Literal["U", "V"],
    ) -> None:
        """
        A class for the tip or tilt M100D motors
        https://www.newport.com.cn/p/CONEX-AG-M100D
        """
        assert axis in ["U", "V"]
        self._connection = connection
        self._axis = axis

    def _verify_valid_connection(self):
        """
        Verify that the connection is valid
        """
        # Check that the motor is connected
        id_number = self._connection.query("1ID?").strip()
        assert "M100D" in id_number

    def move_abs(self, position: float):
        """
        Move the motor to an absolute position

        Parameters:
        -----------
        position: float
            The position to move to
        """
        self._connection.write_str(f"PA{self.axis}{position:.5f}")

    def move_rel(self, position: float):
        """
        Move the motor to a relative position

        Parameters:
        -----------
        position: float
            The position to move to
        """
        self._connection.write_str(f"PR{self.axis}{position:.5f}")

    def read_position(self):
        """
        Read the position of the motor

        Returns:
        --------
        position: float
            The position of the motor
        """
        position_str = self._connection.query_str(f"TP{self.axis}?")
        position = float(position_str)
        return position

    def is_moving(self):
        """
        Check if the motor is moving

        Returns:
        --------
        is_moving: bool
            True if the motor is moving, False otherwise
        """
        raise NotImplementedError

    def is_reset_success(self):
        """
        Check if the reset was successful

        Returns:
        --------
        bool
            True if the reset was successful, False otherwise
        """
        return self._verify_valid_connection()

    def is_stop_success(self):
        """
        Check if the stop was successful

        Returns:
        --------
        bool
            True if the stop was successful, False otherwise
        """
        return not self.is_moving()

    def is_init_success(self):
        """
        Check if the initialisation was successful

        Returns:
        --------
        bool
            True if the initialisation was successful, False otherwise
        """
        return not self.is_moving()

    def is_motion_done(self):
        """
        Check if the motion is done

        Returns:
        --------
        bool
            True if the motion is done, False otherwise
        """
        return not self.is_moving()

    @property
    def axis(self):
        return self._axis

    def is_at_limit(self):
        """
        Check if the motor is at the limit

        Returns:
        --------
        bool
            True if the motor is at the limit, False otherwise
        """
        raise NotImplementedError

    def init(self):
        """
        Initialise the motor
        """
