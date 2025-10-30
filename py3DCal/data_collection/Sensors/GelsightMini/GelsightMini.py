from ..Sensor import Sensor
import cv2
import os

try:
    import gsdevice
except:
    pass


class GelsightMini(Sensor):
    """
    GelsightMini: A Sensor Class for the Gelsight Mini sensor
    """
    def __init__(self):
        self.name = "Gelsight Mini"
        self.x_offset = 108
        self.y_offset = 110
        self.z_offset = 67
        self.z_clearance = 2
        self.max_penetration = 3.5
        self.default_calibration_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), "default.csv")

    def connect(self):
        """
        Connects to the Gelsight Mini sensor.
        """
        # Code to connect to the sensor
        self.sensor = gsdevice.Camera("GelSight Mini")
        self.sensor.connect()
        
    def disconnect(self):
        """
        Disconnects from the Gelsight Mini sensor.
        """
        # Code to disconnect from the sensor
        self.sensor.stop_video()

    def capture_image(self):
        """
        Captures an image from the Gelsight Mini sensor.
        """
        # Code to return an image from the sensor
        image = cv2.cvtColor(self.sensor.get_image(), cv2.COLOR_BGR2RGB)
        return cv2.flip(image, 1)