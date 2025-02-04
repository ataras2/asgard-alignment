# Code from: https://github.com/VForiel/ASGARD-Controllino
# Documentation: https://asgard-controllino.readthedocs.io

import socket

# List of devices and the associated arduino pin
CONNEXIONS = {
    'SSF1+' : 3,
    'SSF2+' : 4,
    'SSF3+' : 5,
    'SSF4+' : 6,
    'SSF1-' : 7,
    'SSF2-' : 8,
    'SSF3-' : 9,
    'SSF4-' : 10,
    'Lower Fan' : 11,
    'Upper Fan' : 12,
    'DM1' : 42,
    'DM2' : 43,
    'DM3' : 44,
    'DM4' : 45,
    'X-MCC (BMX,BMY)' : 46,
    'X-MCC (BFO,SDL,BDS)' : 47,
    'MFF101 (BLF)': 48,
    'USB hubs' : 49,
    'Thermal' : 77,
    'LS16P (LFO)' : 78,
    'Piezo/Laser' : 80,
    'BLF1' : 22,
    'BLF2' : 23,
    'BLF3' : 24,
    'BLF4' : 25,
    'SRL' : 30,
    'SGL' : 31,
    'Lower T' : 54,
    'Upper T' : 55,
    'Bench T' : 56,
    'Floor T' : 57
}

# List of devices
def get_devices():
    return list(CONNEXIONS.keys())

class Controllino():
    def __init__(self, ip, port=23):
        self.ip = ip
        self.port = port

        self._maintain_connection = True #Set to false if this doesn't work.
        self.client = None

    # Ensure the device is known
    def _ensure_device(self, key:str):
        if key not in CONNEXIONS:
            raise ValueError(f"Unkown device '{key}'")
    
    # Create a socket to communicate with the device
    def connect(self):
        self.client = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.client.settimeout(10)
        self.client.connect((self.ip, self.port))
        
    # Close the socket
    def disconnect(self):
        self.client.close()
        self.client = None

    # Maintain connexion if there is only one user
    @property
    def maintain_connection(self) -> bool:
        return self._maintain_connection
    @maintain_connection.setter
    def maintain_connection(self, value:bool):
        if value:
            self.connect()
        else:
            self.disconnect()
        self._maintain_connection = value

    # Clear the buffer before sending a command to avoid bug when reading the answer
    def _clear_buffer(self):
        # Set very very short timeout to avoid waiting for new data
        self.client.settimeout(1e-20)
        try:
            while True:
                data = self.client.recv(1024)
                if not data:
                    break
        except BlockingIOError:
            pass  # Nothing to read, normal
        except TimeoutError:
            pass # No answer, normal too

        # Reset timeout
        self.client.settimeout(10)

    # Send a command to the device
    def send_command_anyreply(self, command:str) -> str:
        # If the connection is not maintained, we need to connect before sending the command
        if self.client is None:
            self.connect()

        # Clear the buffer before sending the command
        self._clear_buffer()

        # Send the command
        self.client.sendall(bytes(command + "\n", "utf-8"))
        # Wait for the answer
        r = self.client.recv(1024).decode().replace("\n", "").replace("\r", "")

        # Disconnect to allow other users to send commands
        if not self.maintain_connection:
            self.disconnect()

        return bool(int(r)) # Convert the answer to a boolean
    
    #Send a command, expecting a boolean reply
    def send_command(self, command:str) -> bool:
        return bool(int(self.send_command_anyreply(command)))

    # Command to turn on a device
    def turn_on(self, key:str) -> bool:
        self._ensure_device(key)
        return self.send_command("o" + str(CONNEXIONS[key]))
    
    # Command to turn off a device
    def turn_off(self, key:str) -> bool:
        self._ensure_device(key)
        return self.send_command("c" + str(CONNEXIONS[key]))
        
    # Command to get the power status of a device
    def get_status(self, key:str) -> bool:
        self._ensure_device(key)
        return self.send_command("g" + str(CONNEXIONS[key]))

    # Command to get the power status of a device
    def modulate(self, key:str, value:int) -> bool:
        self._ensure_device(key)
        if value < 0 or value > 255:
            raise ValueError ("The value must be between 0 and 255")
        return self.send_command("m" + str(CONNEXIONS[key]) + f" {value}")
    
    # Command to move a flipper to the down (out) position
    def flip_down(self, key:str, value:int, dt:float) -> bool:
        self._ensure_device(key + "+")
        self.send_command("m" + str(CONNEXIONS[key+"+"]) + f" 0")
        self.send_command("m" + str(CONNEXIONS[key+"-"]) + f" {value}")
        time.sleep(dt)
        return self.send_command("m" + str(CONNEXIONS[key+"-"]) + f" 0")
        
    # Command to move a flipper to the up (in) position
    def flip_up(self, key:str, value:int, dt:float) -> bool:
        self._ensure_device(key + "+")
        self.send_command("m" + str(CONNEXIONS[key+"-"]) + f" 0")
        self.send_command("m" + str(CONNEXIONS[key+"+"]) + f" {value}")
        time.sleep(dt)
        return self.send_command("m" + str(CONNEXIONS[key+"+"]) + f" 0")        
            
    # Command to ask for an analog input.
    def analog_input(self, key:str) -> int:
        self._ensure_device(key)
        return_str = self.send_command("i" + str(CONNEXIONS[key]))
        try:
            return_int = int(return_str)
            assert 0<=return_int<1024
            return return_int
        except:
            raise ValueError("Returned value was not an integer between 0 and 1023")
    
    # Command to set the piezo DAC value
    def set_piezo_dac(self, channel:int, value:int) -> bool:
        if channel < 0 or channel > 4095:
            raise ValueError("The chanel must be between 0 and 4095")
        if value < 0 or value > 4095:
            raise ValueError("The value must be between 0 and 4095")
        value = int(value) # Convert the value to the 12 bits DAC range
        return self.send_command(f"a{channel} {value}")