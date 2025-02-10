"""
Module for the newport motors.
"""

from typing import Literal
import parse
import numpy as np

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

    CONTROLLER_STATES = {
        "14": "CONFIGURATION",
        "28": "MOVING CL",
        "29": "STEPPING OL",
        "32": "READY from Reset",
        "33": "READY from MOVING CL",
        "34": "READY from DISABLE",
        "35": "READY from JOGGING OL",
        "36": "READY from STEPPING OL",
        "3C": "DISABLE from READY OL",
        "3D": "DISABLE from MOVING CL",
        "46": "JOGGING OL",
    }

    UPPER_LIMIT = 0.75
    LOWER_LIMIT = -0.75

    def __init__(
        self,
        connection: NewportConnection,
        semaphore_id: int,
        axis: Literal["U", "V"],
        name: str,
    ) -> None:
        """
        A class for the tip or tilt M100D motors
        https://www.newport.com.cn/p/CONEX-AG-M100D
        """

        super().__init__(name=name, semaphore_id=semaphore_id)

        assert axis in ["U", "V"]
        self._connection = connection
        self._axis = axis
        self._name = name

        self.internal_position = self.read_position()

    def _verify_valid_connection(self):
        """
        Verify that the connection is valid
        """
        # Check that the motor is connected
        id_number = self._connection.query("1ID?").strip()
        assert "M100D" in id_number

    def ping(self):
        try:
            self._verify_valid_connection()
            return True
        except Exception as e:
            print(f"Error: {e}")
            return False

    def read_state(self, echo=False):
        """
        Read the state of the motor
        """
        msg = self._connection.query_str("1TS?")
        state = msg[7:]
        # print(f"State at M100D: {state}, from msg {msg}")
        state_str = (
            self.CONTROLLER_STATES[state] if state in self.CONTROLLER_STATES else ""
        )

        if echo:
            print(f"State: {state_str}")

        return state_str

    def move_abs(self, position: float):
        """
        Move the motor to an absolute position

        Parameters:
        -----------
        position: float
            The position to move to
        """
        self._connection.write_str(f"1PA{self.axis}{position:.5f}")
        self.internal_position = position

    def move_relative(self, position: float):
        """
        Move the motor to a relative position

        Parameters:
        -----------
        position: float
            The position to move to
        """
        self._connection.write_str(f"1PR{self.axis}{position:.5f}")
        self.internal_position += position

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
        Check if the motor is moving

        Returns:
        --------
        is_moving: bool
            True if the motor is moving, False otherwise
        """
        return self.read_state() in ["MOVING CL", "STEPPING OL", "JOGGING OL"]

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
        state = self.read_state()
        return state in ["READY from Reset"]

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
        position = self.read_position()
        return np.isclose(position, self.UPPER_LIMIT) or np.isclose(
            position, self.LOWER_LIMIT
        )

    def init(self):
        pass

    def stop(self):
        self._connection.write_str(f"1ST{self.axis}")

    def setup(self, value):
        # option 1: blind absolute move
        self.move_abs(value)

        # option 2: relative move using internal state (assuming encoder drifts and not motor)
        # self.move_relative(value - self.internal_position)

    def disable(self):
        pass

    def enable(self):
        pass

    def online(self):
        self._connection.write_str("1RS")

    def standby(self):
        pass


class LS16PAxis(ESOdevice.Motor):

    UPPER_LIMIT = 16.0
    LOWER_LIMIT = 0.0

    MIDDLE = 8.0

    ERROR_BITS = {
        "0000": "No error",
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

    def __init__(self, connection, semaphore_id, name):
        super().__init__(name, semaphore_id)

        self._connection = connection
        self._name = name

        self.init()

    def init(self):
        self._connection.write_str("OR")
        time.sleep(0.5)
        self._connection.write_str("RFP")
        time.sleep(0.5)

    def move_abs(self, position: float):
        self._connection.write_str(f"1PA{position:.5f}")

    def move_relative(self, position):
        self._connection.write_str(f"1PR{position:.5f}")

    def read_position(self):
        reply = self._connection.query_str("1TP?")
        parse_results = parse.parse("1TP{}", reply)
        position = float(parse_results[0])
        return position

    def read_state(self, echo=False):
        """
        Read the state of the motor
        """
        msg = self._connection.query_str("1TS?").strip()

        error_bits = msg[3:7]
        state = msg[7:9]

        error_str = self.ERROR_BITS[error_bits] if error_bits in self.ERROR_BITS else ""
        state_str = (
            self.CONTROLLER_STATES[state] if state in self.CONTROLLER_STATES else ""
        )

        if echo:
            print(f"Error: {error_str}, State: {state_str}")

        return f"{error_str}\n {state_str}"

    def ping(self):
        try:
            self.read_state()
            return True
        except Exception as e:
            print(f"Error: {e}")
            return False

    def is_moving(self):
        _, state_str = self.read_state()
        return state_str in ["MOVING OPEN LOOP (OL)", "MOVING CLOSED LOOP (CL)"]

    def is_reset_success(self):
        _, state_str = self.read_state()
        return state_str in ["READY OPEN LOOP: after reset"]

    def is_stop_success(self):
        _, state_str = self.read_state()
        return state_str in ["READY OPEN LOOP: after reset"]

    def is_init_success(self):
        _, state_str = self.read_state()
        return state_str in ["READY CLOSED LOOP: after HOMING state"]

    def is_motion_done(self):
        _, state_str = self.read_state()
        return state_str in ["READY CLOSED LOOP: after MOVING CL state"]

    def is_at_limit(self):
        position = self.read_position()
        return np.isclose(position, self.UPPER_LIMIT) or np.isclose(
            position, self.LOWER_LIMIT
        )

    def stop(self):
        self._connection.write_str("1ST")

    def setup(self, value):
        self.move_abs(value)

    def disable(self):
        pass
        # TODO: controllino power off

    def enable(self):
        pass
        # TODO: controllino power on

    def online(self):
        self.enable()

        # execute referencing
        self.init()

    def standby(self):
        self.move_abs(self.MIDDLE)

        self.disable()
