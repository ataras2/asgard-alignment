import zmq


class ControllinoConnection:
    """
    A connection to a Controllino, which is an arduino-based controller.
    Able to send ethernet based commands to the Controllino.
    """

    def __init__(self, ip_address: str, port: int):
        self._ip_address = ip_address
        self._port = port

        self._context = None
        self._socket = None
        self.open_connection()

    def open_connection(self):
        """
        Open a connection to the Controllino
        """
        self._context = zmq.Context()
        self._socket = self._context.socket(zmq.REQ)
        self._socket.connect(f"tcp://{self._ip_address}:{self._port}")

    def close_connection(self):
        """
        Close the connection to the Controllino
        """
        self._socket.close()
        self._context.term()

    def send_command(self, command: str):
        """
        Send a command to the Controllino

        Parameters:
        -----------
        command: str
            The command to send to the Controllino
        """
        self._socket.send_string(command)
        message = self._socket.recv_string()
        return message
