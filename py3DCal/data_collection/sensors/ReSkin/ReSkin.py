from ..Sensor import Sensor
import serial
import os


class ReSkin(Sensor):
    """
    ReSkin: A Sensor Class for the ReSkin sensor
    """
    def __init__(self, port: str):
        self.port = port
        self.name = "ReSkin"
        self.x_offset = 110.7
        self.y_offset = 107.6
        self.z_offset = 38.3
        self.z_clearance = 2
        self.max_penetration = 2
        self.default_calibration_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), "default.csv")

    def connect(self):
        """
        Connects to the ReSkin sensor.
        """
        # Code to connect to the sensor
        self.ser = serial.Serial(self.port, 115200, timeout=0.1)
        
    def disconnect(self):
        """
        Disconnects from the ReSkin sensor.
        """
        # Code to disconnect from the sensor
        self.ser.close()

    def capture_image(self):
        """
        Takes a reading from the ReSkin sensor.
        """
        # Code to return an image from the sensor
        last_line = None

        # Read everything currently available
        while self.ser.in_waiting:
            try:
                line = self.ser.readline().decode(errors="ignore").strip()
                if line:
                    last_line = line
            except Exception:
                pass

        data = last_line.split('\t')
        data = [float(val) for val in data]

        return data