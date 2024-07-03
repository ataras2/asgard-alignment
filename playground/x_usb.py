import serial.tools.list_ports

ports = serial.tools.list_ports.comports()

for port, desc, hwid in sorted(ports):
    print("{}: {} [{}]".format(port, desc, hwid))


exit()
import sys
import glob
import serial


def serial_ports():
    """Lists serial port names

    :raises EnvironmentError:
        On unsupported or unknown platforms
    :returns:
        A list of the serial ports available on the system
    """
    if sys.platform.startswith("win"):
        ports = ["COM%s" % (i + 1) for i in range(256)]
    elif sys.platform.startswith("linux") or sys.platform.startswith("cygwin"):
        # this excludes your current terminal "/dev/tty"
        ports = glob.glob("/dev/tty[A-Za-z]*")
    elif sys.platform.startswith("darwin"):
        ports = glob.glob("/dev/tty.*")
    else:
        raise EnvironmentError("Unsupported platform")

    result = []
    for port in ports:
        try:
            s = serial.Serial(port)
            s.close()
            result.append(port)
        except (OSError, serial.SerialException):
            pass
    return result


if __name__ == "__main__":
    print(serial_ports())


import pyvisa


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


rm = pyvisa.ResourceManager()

for resource in rm.list_resources():
    resource = resource.split("::")[0]
    print(resource)
    try:
        motor = LS16P(resource, rm)
        print(f"Motor {resource} opened successfully")
        print(motor._connection.query("1ID?"))
        print(motor._connection.query("1TS?"))
    except Exception as e:
        print(f"Failed to open {resource}: {e}")
