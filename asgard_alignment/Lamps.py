import asgard_alignment.ESOdevice as ESOdevice


class Laser(ESOdevice.Lamp):
    """
    A laser, e.g. Thorlabs CPS532 or CPS635R.
    """

    def __init__(self, name, io_pin, controllino_connection):
        super().__init__(name)

        self.io_pin = io_pin
        self.controllino_connection = controllino_connection

        self.nCooldown = (
            2  # time in seconds to wait before turning lamp on again, can be 0
        )
        self.nWarmup = 1  # time in seconds to wait before lamp is fully on
        self.nMaxOn = 1 * 60 * 60  # 1 hour

    def turn_on(self):
        """
        Turn the laser on
        """
        self.controllino_connection.send_command(f"h{self.io_pin}")

    def turn_off(self):
        """
        Turn the laser off
        """
        self.controllino_connection.send_command(f"l{self.io_pin}")

    def init(self):
        """
        Initialise the laser
        """


class ThermalSource(ESOdevice.Lamp):
    """
    The thermal source, e.g. a SLS201/M
    """

    def __init__(self, name, io_pin, controllino_connection):
        super().__init__(name)

        self.io_pin = io_pin
        self.controllino_connection = controllino_connection

        self.nCooldown = (
            5  # time in seconds to wait before turning lamp on again, can be 0
        )
        self.nWarmup = 5  # time in seconds to wait before lamp is fully on
        self.nMaxOn = 8 * 60 * 60  # 8 hours

    def turn_on(self):
        """
        Turn the thermal source on
        """
        self.controllino_connection.send_command(f"h{self.io_pin}")

    def turn_off(self):
        """
        Turn the thermal source off
        """
        self.controllino_connection.send_command(f"l{self.io_pin}")

    def init(self):
        """
        Initialise the thermal source
        """
