from ..Printer import Printer
import serial

class Ender3(Printer):
    def __init__(self, port):
        self.port = port
        self.name = "Ender 3"

    def connect(self):
        # Code to connect to the printer
        self.ser = serial.Serial(self.port, 115200)

    def disconnect(self):
        # Code to disconnect from the printer
        self.ser.close()

    def send_gcode(self, command):
        # Code to execute gcode command on the printer
        self.ser.write(str.encode(command + "\r\n"))

    def get_response(self):
        # Code to return message from the printer
        reading = self.ser.readline().decode('utf-8')

        return reading

    def initialize(self, xy_only=False):
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