import asgard_alignment.ESOdevice as ESOdevice


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
