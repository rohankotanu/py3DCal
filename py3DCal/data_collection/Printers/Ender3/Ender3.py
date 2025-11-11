from ..Printer import Printer
import serial
from typing import Union
from pathlib import Path

class Ender3(Printer):
    """
    Ender3: A Printer Class for the Ender 3
    Args:
        port (str or pathlib.Path): The COM port the printer is connected to.
    """
    def __init__(self, port: Union[str, Path]):
        self.port = port
        self.name = "Ender 3"

    def connect(self):
        """
        Connects to the Ender 3 printer.
        """
        # Code to connect to the printer
        self.ser = serial.Serial(self.port, 115200)

    def disconnect(self):
        """
        Disconnects from the Ender 3 printer.
        """
        # Code to disconnect from the printer
        self.ser.close()

    def send_gcode(self, command: str):
        """
        Sends a G-code command to the Ender 3 printer.
        Args:
            command (str): The G-code command to be sent to the 3D printer.
        """
        # Code to execute gcode command on the printer
        self.ser.write(str.encode(command + "\r\n"))

    def get_response(self):
        """
        Gets messages sent by the Ender 3 printer.

        Returns:
            response (str): The message sent by the printer.
        """
        # Code to return message from the printer
        response = self.ser.readline().decode('utf-8')

        return response

    def initialize(self, xy_only: bool = False):
        """
        Initializes the Ender 3 printer (homes the printer, sets units, adjusts fans, etc).
        Args:
            xy_only (bool): If True, only homes the X and Y axes.
        """
        # Code to initialize printer (home, set units, set absolute/relative movements, adjust fan speeds, etc.)

        # Use Metric Values
        self.send_gcode("G21")

        # Absolute Positioning
        self.send_gcode("G90")

        # Fan Off
        self.send_gcode("M107")

        if xy_only:
            # Home Printer X Y
            self.send_gcode("G28 X Y")
        else:
            # Home Printer X Y Z
            self.send_gcode("G28")

        # Check if homing is complete
        ok_count = 0

        while ok_count < 4:
            if "ok" in self.get_response():
                ok_count += 1

        return True