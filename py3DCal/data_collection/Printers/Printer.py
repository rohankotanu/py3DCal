from abc import ABC, abstractmethod

class Printer(ABC):
    """
    Printer: An abstract base class for 3D printers.
    """
    def __init__(self):
        self.name = ""
        
    @abstractmethod
    def connect(self):
        """ Connects to the 3D printer.
        """
        pass

    @abstractmethod
    def disconnect(self):
        """ Disconnects from the 3D printer.
        """
        pass

    @abstractmethod
    def send_gcode(self, command: str):
        """ Sends a G-code command to the 3D printer.

        Args:
            command (str): The G-code command to be sent to the 3D printer.
        """
        pass

    @abstractmethod
    def get_response(self):
        """ Gets messages sent by the 3D printer.

        Returns:
            str: The message sent by the printer.
        """
        pass

    @abstractmethod
    def initialize(self):
        """ Initializes the 3D printer (homes the printer, sets units, adjusts fans, etc).
        """
        pass

    def go_to(self, x: float = None, y: float = None, z: float = None):
        """
        Moves the printer to a specific position.
        Args:
            x (float, optional): The X coordinate to move to.
            y (float, optional): The Y coordinate to move to.
            z (float, optional): The Z coordinate to move to.
        """
        # Code to move the printer to a specific position
        command = "G0"
        if x is not None:
            command += f" X{x}"
        if y is not None:
            command += f" Y{y}"
        if z is not None:
            command += f" Z{z}"

        self.send_gcode(command)