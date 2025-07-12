import telnetlib
import time


class AtenEcoPDU:
    """
    A Python class to interact with ATEN PE Series eco PDUs via Telnet.

    This class provides methods to connect, authenticate, and send various
    commands to the eco PDU as described in the ATEN eco PDU Telnet User Guide.
    """

    def __init__(self, host, port=23, timeout=60, debug=False):
        """
        Initializes the AtenEcoPDU client.

        Args:
            host (str): The IP address or hostname of the eco PDU.
            port (int): The Telnet port (default is 23).
            timeout (int): The timeout in seconds for Telnet operations.
        """
        self.host = host
        self.port = port
        self.timeout = timeout
        self.tn = None
        self.connected = False
        self.debug = debug

    def connect(self, username="teladmin", password="asgard", verbose=True):
        """
        Establishes a Telnet connection to the eco PDU and logs in.

        Args:
            username (str): The Telnet login username (default: 'teladmin').
            password (str): The Telnet login password (default: 'telpwd').

        Returns:
            bool: True if connection and login are successful, False otherwise.
        """
        try:
            # Establish Telnet connection
            self.tn = telnetlib.Telnet(self.host, self.port, self.timeout)
            self.connected = True
            if verbose:
                print(f"Connected to {self.host}:{self.port}")

            # Read initial login prompts

            login_prompt = self.tn.read_until(b"Login: ", self.timeout).decode("ascii")
            if verbose:
                print(f"PDU Login Prompt: {login_prompt.strip()}")

            # Send username
            self.tn.write(username.encode("ascii") + b"\r")
            if verbose:
                print(f"Sent username: {username}")

            # Read password prompt
            password_prompt = self.tn.read_until(b"Password: ", self.timeout).decode(
                "ascii"
            )
            if verbose:
                print(f"PDU Password Prompt: {password_prompt.strip()}")

            # Send password
            self.tn.write(password.encode("ascii") + b"\r")
            if verbose:
                print(f"Sent password (hidden)")

            # Read response after login
            response = self.tn.read_until(b"Logged in successfully", self.timeout)
            if b"Logged in successfully" in response:
                print("Login successful!")
                # Read the command line prompt (e.g., 'PE Telnet server 1.1\n>')
                self.tn.read_until(
                    b">", self.timeout
                )  # Consume the rest of the login message
                return True
            else:
                print(response)
                print("Login failed. Check username/password or PDU status.")
                self.close()
                return False
        except Exception as e:
            print(f"Error connecting or logging in: {e}")
            self.connected = False
            return False

    def _send_command(self, command, expected_response=b">"):
        """
        Sends a command to the PDU and reads the response.

        Args:
            command (str): The command string to send.
            expected_response (bytes): The byte string to wait for, indicating
                                       the end of the PDU's response (default is b">").

        Returns:
            str: The decoded response from the PDU, or an empty string if an error occurs.
        """
        if not self.connected:
            print("Not connected to PDU. Please call connect() first.")
            return ""
        try:
            full_command = command.strip() + "\r"
            self.tn.write(full_command.encode("ascii"))
            if self.debug:
                print(f"Sent command: {full_command.strip()}")
            # Read until the expected prompt or timeout
            response = self.tn.read_until(expected_response, self.timeout).decode(
                "ascii"
            )
            # Remove the command echoed back and the prompt from the response
            # The PDU often echoes the command back before its actual response.
            # We also remove the final prompt (e.g., '>')
            clean_response = (
                response.replace(full_command, "")
                .replace(expected_response.decode("ascii"), "")
                .strip()
            )
            return clean_response
        except Exception as e:
            print(f"Error sending command '{command}': {e}")
            return ""

    def read_outlet_status(self, outlet_number, return_string="simple"):
        """
        Reads the power status of a specific outlet.

        Args:
            outlet_number (int): The outlet number (e.g., 1 for 001, 12 for 012).
            return_string (str): 'simple' or 'format' (default: 'simple').

        Returns:
            str: The outlet status.
        """
        if not (1 <= outlet_number <= 999):  # Assuming 3-digit outlet numbers
            print("Invalid outlet number. Must be between 1 and 999.")
            return ""
        if return_string not in ["simple", "format"]:
            print("Invalid return_string. Must be 'simple' or 'format'.")
            return ""

        padded_outlet = f"{outlet_number:02d}"
        command = f"read status o{padded_outlet} {return_string}"
        return self._send_command(command)

    def switch_outlet_status(self, outlet_number, control_action, option="imme"):
        """
        Changes the power status of a specific outlet.

        Args:
            outlet_number (int): The outlet number.
            control_action (str): 'on', 'off', or 'reboot'.
            option (str): 'imme' (immediately) or 'delay' (with time delay).
                          'delay' is default if not specified in PDU.
                          'reboot' does not use an option.

        Returns:
            str: The PDU's response to the command.
        """
        if not (1 <= outlet_number <= 999):
            print("Invalid outlet number. Must be between 1 and 999.")
            return ""
        if control_action not in ["on", "off", "reboot"]:
            print("Invalid control_action. Must be 'on', 'off', or 'reboot'.")
            return ""
        if option not in ["imme", "delay"] and control_action != "reboot":
            print(
                "Invalid option. Must be 'imme' or 'delay' for 'on'/'off'. 'reboot' does not use an option."
            )
            return ""

        padded_outlet = f"{outlet_number:02d}"
        if control_action == "reboot":
            command = f"sw o{padded_outlet} reboot"
        else:
            command = f"sw o{padded_outlet} {control_action} {option}"
        return self._send_command(command)

    def read_power_value(
        self, target, number=None, measurement=None, return_string="simple"
    ):
        """
        Reads power measurement values from PDU, bank, or outlet.

        Args:
            target (str): 'dev' (PDU), 'bnk' (bank), or 'olt' (outlet).
            number (int, optional): Bank or outlet number. Required for 'bnk' and 'olt'.
            measurement (str): 'curr', 'volt', 'pow', 'pd', 'freq'.
            return_string (str): 'simple' or 'format' (default: 'simple').

        Returns:
            str: The power measurement value.
        """
        if target not in ["dev", "bnk", "olt"]:
            print("Invalid target. Must be 'dev', 'bnk', or 'olt'.")
            return ""
        if target in ["bnk", "olt"] and (number is None or not (1 <= number <= 999)):
            print(f"For target '{target}', a valid number (1-999) is required.")
            return ""
        if measurement not in ["curr", "volt", "pow", "pd", "freq"]:
            print(
                "Invalid measurement. Must be 'curr', 'volt', 'pow', 'pd', or 'freq'."
            )
            return ""
        if return_string not in ["simple", "format"]:
            print("Invalid return_string. Must be 'simple' or 'format'.")
            return ""

        command_parts = ["read", "meter", target]
        if number is not None:
            command_parts.append(f"o{number:02d}")
        command_parts.append(measurement)
        command_parts.append(return_string)

        command = " ".join(command_parts)
        return self._send_command(command)

    def read_environmental_value(self, sensor_number, return_string="simple"):
        """
        Reads environmental sensor values.

        Args:
            sensor_number (int): The environmental sensor number (01-04).
            return_string (str): 'simple' or 'format' (default: 'simple').

        Returns:
            str: The environmental sensor value.
        """
        if not (1 <= sensor_number <= 4):
            print("Invalid sensor number. Must be between 1 and 4.")
            return ""
        if return_string not in ["simple", "format"]:
            print("Invalid return_string. Must be 'simple' or 'format'.")
            return ""

        padded_sensor = f"{sensor_number:02d}"
        command = f"read sensor o{padded_sensor} {return_string}"
        return self._send_command(command)

    def close(self):
        """
        Closes the Telnet session.
        """
        if self.connected and self.tn:
            try:
                print("Closing Telnet session...")
                self._send_command(
                    "quit", expected_response=b""
                )  # 'quit' command doesn't always return a prompt
                self.tn.close()
                self.connected = False
                print("Telnet session closed.")
            except Exception as e:
                print(f"Error closing Telnet session: {e}")
        else:
            print("No active Telnet session to close.")
