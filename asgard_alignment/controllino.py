import socket
import time

# List of devices and the associated arduino pin
CONNEXIONS = {
    "SSF1+": 3,
    "SSF2+": 4,
    "SSF3+": 5,
    "SSF4+": 6,
    "SSF1-": 7,
    "SSF2-": 8,
    "SSF3-": 9,
    "SSF4-": 10,
    "Lower Fan": 12,
    "Upper Fan": 13,
    "DM1": 42,
    "DM2": 43,
    "DM3": 44,
    "DM4": 45,
    "X-MCC (BMX,BMY)": 46,
    "X-MCC (BFO,SDL,BDS)": 47,
    "MFF101 (BLF)": 49,
    "USB hubs": 48,
    "LS16P (HFO)": 77,
    "Lower Kickstart", 78,
    "Upper Kickstart", 79,
    "Piezo/Laser": 80,
    "BLF1": 26,
    "BLF2": 23,
    "BLF3": 24,
    "BLF4": 25,
    "SBB": 22,
    "SRL": 30,
    "SGL": 31,
    "Lower T": 54, 
    "Upper T": 56, 
    "Bench T": 55, 
    "Floor T": 58, 
}


def get_devices():
    """
    List of devices.

    Returns
    -------
    list
        List of device names.
    """
    return list(CONNEXIONS.keys())


class Controllino:
    def __init__(self, ip, port=23):
        """
        Initialize the Controllino class.

        Parameters
        ----------
        ip : str
            IP address of the device.
        port : int, optional
            Port number, by default 23.
        """
        self.ip = ip
        self.port = port
        self._maintain_connection = True
        self.client = None

	    #The turn-on command needs a string, not a number! 
        self.turn_on("Piezo/Laser")
        self.turn_on("MFF101 (BLF)")
        self.turn_on("LS16P (HFO)")
        self.turn_on("X-MCC (BMX,BMY)")
        self.turn_on("X-MCC (BFO,SDL,BDS)")
        self.turn_on("USB hubs")
        self.turn_on("Upper Kickstart")
        time.sleep(0.1)
        self.turn_on("Lower Kickstart")
        
        #Wait for the piezo to settle and fans to start up, then we will 
        #set piezos and fans to mid range.
        time.sleep(1)
        self.turn_off("Upper Kickstart")
        self.modulate("Upper Fan", 128)
        time.sleep(0.1)
        self.turn_off("Lower Kickstart")
        self.modulate("Lower Fan", 128)
        self.set_piezo_dac(0,2048)
        self.set_piezo_dac(1,2048)
        self.set_piezo_dac(2,2048)
        self.set_piezo_dac(3,2048)
        

    def _ensure_device(self, key: str):
        """
        Ensure the device is known.

        Parameters
        ----------
        key : str
            Device key.

        Raises
        ------
        ValueError
            If the device is unknown.
        """
        if key not in CONNEXIONS:
            raise ValueError(f"Unknown device '{key}'")

    def connect(self):
        """
        Create a socket to communicate with the device.
        """
        self.client = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.client.settimeout(10)
        self.client.connect((self.ip, self.port))

    def disconnect(self):
        """
        Close the socket.
        """
        self.client.close()
        self.client = None

    @property
    def maintain_connection(self) -> bool:
        """
        Maintain connection if there is only one user.

        Returns
        -------
        bool
            Connection maintenance status.
        """
        return self._maintain_connection

    @maintain_connection.setter
    def maintain_connection(self, value: bool):
        """
        Set the connection maintenance status.

        Parameters
        ----------
        value : bool
            Connection maintenance status.
        """
        if value:
            self.connect()
        else:
            self.disconnect()
        self._maintain_connection = value

    def _clear_buffer(self):
        """
        Clear the buffer before sending a command to avoid bug when reading the answer.
        """
        self.client.settimeout(1e-20)
        try:
            while True:
                data = self.client.recv(1024)
                if not data:
                    break
        except BlockingIOError:
            pass
        except TimeoutError:
            pass
        self.client.settimeout(10)

    def send_command_anyreply(self, command: str) -> str:
        """
        Send a command to the device.

        Parameters
        ----------
        command : str
            Command to send.

        Returns
        -------
        str
            Reply from the device.
        """
        if self.client is None:
            self.connect()
        self._clear_buffer()
        self.client.sendall(bytes(f"{command}\n", "utf-8"))
        r = self.client.recv(1024).decode().replace("\n", "").replace("\r", "")
        if not self.maintain_connection:
            self.disconnect()
        return r

    def send_command(self, command: str) -> bool:
        """
        Send a command, expecting a boolean reply.

        Parameters
        ----------
        command : str
            Command to send.

        Returns
        -------
        bool
            Reply from the device.
        """
        return bool(int(self.send_command_anyreply(command)))

    def turn_on(self, key: str) -> bool:
        """
        Command to turn on a device.

        Parameters
        ----------
        key : str
            Device key.

        Returns
        -------
        bool
            Status of the command.
        """
        self._ensure_device(key)
        return self.send_command(f"o{CONNEXIONS[key]}")

    def turn_off(self, key: str) -> bool:
        """
        Command to turn off a device.

        Parameters
        ----------
        key : str
            Device key.

        Returns
        -------
        bool
            Status of the command.
        """
        self._ensure_device(key)
        return self.send_command(f"c{CONNEXIONS[key]}")

    def get_status(self, key: str) -> bool:
        """
        Command to get the power status of a device.

        Parameters
        ----------
        key : str
            Device key.

        Returns
        -------
        bool
            Power status of the device.
        """
        self._ensure_device(key)
        return self.send_command(f"g{CONNEXIONS[key]}")

    def modulate(self, key: str, value: int) -> bool:
        """
        Command to modulate a device.

        Parameters
        ----------
        key : str
            Device key.
        value : int
            Modulation value (0-255).

        Returns
        -------
        bool
            Status of the command.

        Raises
        ------
        ValueError
            If the value is not between 0 and 255.
        """
        self._ensure_device(key)
        if value < 0 or value > 255:
            raise ValueError("The value must be between 0 and 255")
        return self.send_command(f"m{CONNEXIONS[key]} {value}")

    def flip_down(self, key: str, value: int, dt: float) -> bool:
        """
        Command to move a flipper to the down (out) position.

        Parameters
        ----------
        key : str
            Device key.
        value : int
            Modulation value (0-255).
        dt : float
            Delay time in seconds.

        Returns
        -------
        bool
            Status of the command.
        """
        self._ensure_device(f"{key}+")
        self.send_command(f"m{CONNEXIONS[key + '+']} 0")
        self.send_command(f"m{CONNEXIONS[key + '-']} {value}")
        time.sleep(dt)
        return self.send_command(f"m{CONNEXIONS[key + '-']} 0")

    def flip_up(self, key: str, value: int, dt: float) -> bool:
        """
        Command to move a flipper to the up (in) position.

        Parameters
        ----------
        key : str
            Device key.
        value : int
            Modulation value (0-255).
        dt : float
            Delay time in seconds.

        Returns
        -------
        bool
            Status of the command.
        """
        self._ensure_device(f"{key}+")
        self.send_command(f"m{CONNEXIONS[key + '-']} 0")
        self.send_command(f"m{CONNEXIONS[key + '+']} {value}")
        time.sleep(dt)
        return self.send_command(f"m{CONNEXIONS[key + '+']} 0")

    def analog_input(self, key: str) -> int:
        """
        Command to ask for an analog input.

        Parameters
        ----------
        key : str
            Device key.

        Returns
        -------
        int
            Analog input value.

        Raises
        ------
        ValueError
            If the returned value is not an integer between 0 and 1023.
        """
        self._ensure_device(key)
        return_str = self.send_command_anyreply(f"i{CONNEXIONS[key]}")
        try:
            return_int = int(return_str)
            assert 0 <= return_int < 1024
            return return_int
        except:
            raise ValueError("Returned value was not an integer between 0 and 1023")

    def set_piezo_dac(self, channel: int, value: int) -> bool:
        """
        Command to set the piezo DAC value.

        Parameters
        ----------
        channel : int
            DAC channel (0-4095).
        value : int
            DAC value (0-4095).

        Returns
        -------
        bool
            Status of the command.

        Raises
        ------
        ValueError
            If the channel or value is not between 0 and 4095.
        """
        if channel < 0 or channel > 4095:
            raise ValueError("The channel must be between 0 and 4095")
        if value < 0 or value > 4095:
            raise ValueError("The value must be between 0 and 4095")
        value = int(value)
        return self.send_command(f"a{channel} {value}")
