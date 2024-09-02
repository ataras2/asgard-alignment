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

import pyvisa


class NewportMotor:
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

    def close_connection(self):
        """
        Close the connection to the motor
        """
        self._connection.before_close()
        self._connection.close()

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
        motor_type: NewportMotor
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


class M100D(NewportMotor):
    """
    A tip tilt motor driver class
    https://www.newport.com.cn/p/CONEX-AG-M100D
    """

    AXES = Enum("AXES", ["U", "V"])
    HW_BOUNDS = {AXES.U: [-0.75, 0.75], AXES.V: [-0.75, 0.75]}

    def __init__(
        self,
        serial_port,
        resource_manager: pyvisa.ResourceManager,
        orientation: Literal["normal", "reversed"] = "normal",
    ) -> None:
        """
        A class for the tip tile M100D motors

        Parameters:
        -----------
        serial_port: str
            The serial port that the motor is connected to
        resource_manager: pyvisa.ResourceManager
            The resource manager that is used to open the serial connection
        orientation: str
            The orientation of the motor. Either "normal" or "reversed". In the case of reversed, the U and V axes are
            swapped, such that asking a function to move the U axis will actually move the V axis and vice versa.
            This correctly accounts for the flipped sign of the U axis in the reversed orientation.
        """
        super().__init__(serial_port, resource_manager)

        if isinstance(orientation, float):
            if np.isclose(orientation, 0.0):
                orientation = "normal"
            elif np.isclose(orientation, 90.0):
                orientation = "reversed"
            else:
                raise ValueError(
                    f"orientation must be either 'normal' or 'reverse', not {orientation}"
                )

        if orientation not in ["normal", "reverse"]:
            raise ValueError(
                f"orientation must be either 'normal' or 'reverse', not {orientation}"
            )

        # TODO: this needs some thinking about how to implement so that the external interface doesn't notice
        self._is_reversed = orientation == "reverse"
        self._current_pos = {
            self.AXES.U: self.read_pos(self.AXES.U),
            self.AXES.V: self.read_pos(self.AXES.V),
        }

    def _verify_valid_connection(self):
        """
        Verify that the serial connection opened by the class is indeed to to a NEWPORT M100D
        """
        id_number = self._connection.query("1ID?").strip()
        assert "M100D" in id_number

    def _get_axis(self, axis: AXES):
        """
        Get the axis and apply the reverse flag if needed
        """
        if self._is_reversed:
            if axis == self.AXES.U:
                axis = self.AXES.V
            elif axis == self.AXES.V:
                axis = self.AXES.U
        return axis

    def _alter_value(self, axis: AXES, value: float):
        """
        if the motor is reversed, then the value for the U axis needs to be flipped
        the function should always be called after _get_axis i.e. the axis should be already changed
        """
        if self._is_reversed and axis == self.AXES.U:
            return -value
        return value

    @property
    def get_current_pos(self):
        """
        Return the current position of the motor in degrees
        """
        return [
            self._alter_value(self._get_axis(ax), self._current_pos[self._get_axis(ax)])
            for ax in M100D.AXES
        ]

    def set_to_zero(self):
        """
        Set all the motor axes positions to zero
        """
        for axis in self.AXES:
            self.set_absolute_position(0.0, axis)

    @property
    def status_string(self):
        """
        Return the status of the motor, and the position of both axes
        """

        status = self._connection.query("1TS?").strip()
        pos_u = self.read_pos(self.AXES.U)
        pos_v = self.read_pos(self.AXES.V)

        return f"Status: {status}, U: {pos_u}, V: {pos_v}"

    def read_pos(self, axis: AXES) -> float:
        """
        Read the position of a given axis.

        Parameters:
            axis (M100D.AXES) : the axis to read from

        Returns:
            position (float) : the position of the axis in degrees
        """
        axis = self._get_axis(axis)

        return_str = self._connection.query(f"1TP{axis.name}").strip()
        subset = parse.parse("{}" + f"TP{axis.name}" + "{}", return_str)

        if subset is not None:
            return self._alter_value(axis, float(subset[1]))
        raise ValueError(f"Could not parse {return_str}")

    def set_absolute_position(self, value: float, axis: AXES):
        """
        Set the absolute position of the motor in a given axis

        Parameters:
            value (float) : The new position in degrees
            axis (M100D.AXES) : the axis to set
        """
        axis = self._get_axis(axis)
        value = self._alter_value(axis, value)
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
        self._current_pos = 0.0

        # we always set the motor to the closed loop mode
        self._connection.write("1OR")
        self._connection.write("1RF")

        self.set_absolute_position(8.0)

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

    def read_state(self):
        """
        Read the state of the motor
        """
        msg = self._connection.query("1TS?").strip()

        error_bits = msg[3:8]

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
