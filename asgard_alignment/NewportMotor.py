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

import parse
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
            self._serial_port,
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
        self._connection.write_str(f"1PA{self.axis}{position:.5f}")

    def move_rel(self, position: float):
        """
        Move the motor to a relative position

        Parameters:
        -----------
        position: float
            The position to move to
        """
        self._connection.write_str(f"1PR{self.axis}{position:.5f}")

    def read_position(self):
        """
        Read the position of the motor

        Returns:
        --------
        position: float
            The position of the motor
        """
        reply = self._connection.query_str(f"1TP{self.axis}?")
        # parse reply of form 1TP{self.axis}{position}
        parse_results = parse.parse(f"1TP{self.axis}" + "{}", reply)

        position = float(parse_results[0])  
        return position

    def is_moving(self):
        """
        Set the absolute position of the motor in a given axis

        Parameters:
            value (float) : The new position in degrees
            axis (M100D.AXES) : the axis to set
        """
        str_to_write = f"1PA{axis.name}{value}"
        logging.info(f"sending {str_to_write}")
        self._connection.write(str_to_write)
        self._current_pos[axis] = value

    @classmethod
    def validate_config(cls, config):
        """
        Validate the config dictionary for the motor
        """
        if "orientation" not in config:
            raise KeyError("orientation not in config")

    @staticmethod
    def setup_individual_config():
        inp = input("is the motor mounted normally with the text right way up? (Y/N)")
        orientation = None
        if inp.lower() == "y":
            orientation = "normal"
        elif inp.lower() == "n":
            orientation = "reverse"

        if orientation is None:
            raise ValueError(f"invalid input {inp}")
        return {"orientation": orientation}

    def _get_callback(self, axis: AXES):
        """
        Get the callback function for the GUI
        """
        if axis == self.AXES.U:

            def callback():
                self.set_absolute_position(st.session_state.U, axis)

        elif axis == self.AXES.V:

            def callback():
                self.set_absolute_position(st.session_state.V, axis)

        else:
            raise ValueError(f"invalid axis {axis}")

        return callback

    def GUI(self):
        """
        A GUI to control the motor
        """
        st.header("M100D motor")
        asgard_alignment.GUI.CustomNumeric.variable_increment(
            keys=["U", "V"],
            callback_fns=[
                self._get_callback(self.AXES.U),
                self._get_callback(self.AXES.V),
            ],
            values=[self.get_current_pos[0], self.get_current_pos[1]],
            main_bounds=[-0.75, 0.75],
        )


class LS16P(NewportMotor):
    """
    A linear motor driver class
    https://www.newport.com/p/CONEX-SAG-LS16P
    """

    HW_BOUNDS = [-8.0, 8.0]

    ERROR_BITS = {
        "0010": "Bit motor stall timeout",
        "0020": "Bit time out motion",
        "0040": "Bit time out homing",
        "0080": "Bit bad memory parameters",
        "0100": "Bit supply voltage too low",
        "0200": "Bit internal error",
        "0400": "Bit memory problem",
        "0800": "Bit over temperature",
    }

    CONTROLLER_STATES = {
        "0A": "READY OPEN LOOP: after reset",
        "0B": "READY OPEN LOOP: after HOMING state",
        "0C": "READY OPEN LOOP: after STEPPING state",
        "0D": "READY OPEN LOOP: after CONFIGURATION state",
        "0E": "READY OPEN LOOP: after with no parameters",
        "0F": "READY OPEN LOOP: after JOGGING state",
        "10": "READY OPEN LOOP: after SCANNING state",
        "11": "READY OPEN LOOP: after READY CLOSED LOOP state",
        "14": "CONFIGURATION",
        "1E": "HOMING",
        "1F": "REFERENCING",
        "28": "MOVING OPEN LOOP (OL)",
        "29": "MOVING CLOSED LOOP (CL)",
        "32": "READY CLOSED LOOP: after HOMING state",
        "33": "READY CLOSED LOOP: after MOVING CL state",
        "34": "READY CLOSED LOOP: after DISABLE state",
        "35": "READY CLOSED LOOP: after REFERENCING state",
        "36": "READY CLOSED LOOP: after HOLDING state",
        "3C": "DISABLE: after READY CLOSED LOOP state",
        "3D": "DISABLE: after MOVING CL state",
        "46": "JOGGING",
        "50": "SCANNING",
        "5A": "HOLDING",
    }

    def __init__(self, serial_port: str, resource_manager: pyvisa.ResourceManager):
        super().__init__(serial_port, resource_manager)

        # self.set_absolute_position(8.0)

    def initialise(self):
        self._current_pos = 0.0

        # we always set the motor to the closed loop mode
        self._connection.write("OR")
        time.sleep(2.0)
        self._connection.write("RFP")

    @classmethod
    def connect_and_get_SA(cls, port):
        """
        Connect to the motor and check the SA
        """
        rm = pyvisa.ResourceManager()

        connection = rm.open_resource(
            port,
            baud_rate=cls.SERIAL_BAUD,
            write_termination=cls.SERIAL_TERMIN,
            read_termination=cls.SERIAL_TERMIN,
        )

        sa = connection.query("SA?").strip()

        connection.before_close()
        connection.close()

        return sa

    def read_state(self, echo=False):
        """
        Read the state of the motor
        """
        msg = self._connection.query("1TS?").strip()

        error_bits = msg[3:8]

        error_str = self.ERROR_BITS[error_bits] if error_bits in self.ERROR_BITS else ""
        state = msg[8:10]
        state_str = (
            self.CONTROLLER_STATES[state] if state in self.CONTROLLER_STATES else ""
        )

        if echo:
            print(f"Error: {error_str}, State: {state_str}")

        return error_str, state_str

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
        return self._current_pos

    def GUI(self):
        """
        A GUI to control the motor
        """
        st.header("LS16P motor")

        def callback():
            self.set_absolute_position(st.session_state.pos)

        asgard_alignment.GUI.CustomNumeric.variable_increment(
            keys=["pos"],
            callback_fns=[
                callback,
            ],
            values=[self.get_current_pos],
            main_bounds=[0.0, 16.0],
        )
