from abc import ABC, abstractmethod

class Printer(ABC):
    def __init__(self):
        self.name = ""
        
    @abstractmethod
    def connect(self):
        """ Connects to the 3D printer
        """
        pass

    @abstractmethod
    def disconnect(self):
        """ Disconnects from the 3D printer
        """
        pass

    @abstractmethod
    def send_gcode(self, command):
        """ Sends gcode command to the 3D printer

        Args:
            command (str): The gcode command to be sent to the 3D printer.
        """
        pass

    @abstractmethod
    def get_response(self):
        """ Gets messages sent by the 3D printer

        Returns:
            str: The message sent by the printer.
        """
        pass

    @abstractmethod
    def initialize(self):
        """ Initializes the 3D printer (homes the printer, sets units, adjusts fans, etc)
        """
        pass

    def go_to(self, x=None, y=None, z=None):
        # Code to move the printer to a specific position
        command = "G0"
        if x is not None:
            command += f" X{x}"
        if y is not None:
            command += f" Y{y}"
        if z is not None:
            command += f" Z{z}"

        self.send_gcode(command)