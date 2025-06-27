import asgard_alignment.ESOdevice as ESOdevice
from typing import Union


class LightSource(ESOdevice.Lamp):
    """
    All light sources in H/B/S use this framework
    """

    def __init__(self, name, controllino_connection, nCooldown, nWarmup, nMaxOn):
        super().__init__(name)

        self.controllino_connection = controllino_connection

        self.nCooldown = nCooldown
        self.nWarmup = nWarmup
        self.nMaxOn = nMaxOn

        self.turn_off()
        self._on = False

    def turn_on(self):
        """
        Turn the thermal source on
        """
        self.controllino_connection.turn_on(self.name)

    def turn_off(self):
        """
        Turn the thermal source off
        """
        self.controllino_connection.turn_off(self.name)

    def init(self):
        """
        Initialise the laser
        """

    def is_moving(self):
        return False

    def is_on(self):
        """
        Check if the light source is on
        """
        res = self.controllino_connection.get_status(self.name)
        print(f"Light source {self.name} is on: {res}")
        return res

    def is_off(self):
        """
        Check if the light source is off
        """
        return not self.is_on()

    def ESO_read_position(self):
        """
        Read the "position" of the light source
        """
        return int(self.controllino_connection.get_status(self.name))

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
        if motion_type == "ST":
            if value == "T":
                self.turn_on()
            elif value == "F":
                self.turn_off()
            else:
                print(f"ERROR: Invalid state {value} for {self.name}")
        else:
            print(f"ERROR: Motion type {motion_type} not implemented for {self.name}")

    def disable(self):
        """
        The DISABLE command can be used to request the MCU to power off devices.
        """
        pass

    def enable(self):
        """
        The ENABLE command can be used to request the MCU to power on devices.
        """
        pass

    def online(self):
        """
        Upon reception, the ICS back-end server of the MCU shall power on all the
        controlled devices and have them ready to accept SETUP commands.
        """
        pass

    def standby(self):
        """
        Upon reception, the ICS
        back-end server of the MCU shall move some of the controlled devices to a safe “parking”
        position (if required) and power off all the controlled devices.
        """
        pass
