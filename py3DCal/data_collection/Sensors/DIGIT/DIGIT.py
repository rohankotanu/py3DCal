from ..Sensor import Sensor
import cv2
import os

try:
    from digit_interface import Digit
except:
    pass

class DIGIT(Sensor):
    def __init__(self, serial_number):
        self.serial_number = serial_number
        self.name = "DIGIT"
        self.x_offset = 110
        self.y_offset = 111.5
        self.z_offset = 137
        self.z_clearance = 2
        self.max_penetration = 4
        self.default_calibration_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), "default.csv")

    def connect(self):
        # Code to connect to the sensor
        self.sensor = Digit(self.serial_number)
        self.sensor.connect()
        self.sensor.set_fps(30)
        
    def disconnect(self):
        # Code to disconnect from the sensor
        self.sensor.disconnect()

    def capture_image(self):
        # Code to return an image from the sensor
        return cv2.flip(self.sensor.get_frame(), 1)