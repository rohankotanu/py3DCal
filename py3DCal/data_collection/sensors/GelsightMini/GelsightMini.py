from ..Sensor import Sensor
import cv2
import os
import platform

try:
    import gsdevice
except:
    pass

def resize_crop_mini(image, width, height):
    """
    Resize and center-crop image to given width and height.
    """
    h, w = image.shape[:2]

    # Resize while keeping aspect ratio
    scale = max(width / w, height / h)
    resized = cv2.resize(image, (int(w * scale), int(h * scale)))

    # Center crop
    h_resized, w_resized = resized.shape[:2]
    x_start = (w_resized - width) // 2
    y_start = (h_resized - height) // 2

    return resized[y_start:y_start + height, x_start:x_start + width]

class GelsightMini(Sensor):
    """
    GelsightMini: A Sensor Class for the GelSight Mini sensor
    """
    def __init__(self, camera_id: int = 0):
        self.camera_id = camera_id
        self.name = "GelSight Mini"
        self.x_offset = 108
        self.y_offset = 110
        self.z_offset = 67
        self.z_clearance = 2
        self.max_penetration = 3.5
        self.default_calibration_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), "default.csv")

    if platform.system() == "Linux":
        def connect(self):
            """
            Connects to the GelSight Mini sensor.
            """
            # Code to connect to the sensor
            self.sensor = gsdevice.Camera("GelSight Mini")
            self.sensor.connect()
            
        def disconnect(self):
            """
            Disconnects from the GelSight Mini sensor.
            """
            # Code to disconnect from the sensor
            self.sensor.stop_video()

        def capture_image(self):
            """
            Captures an image from the GelSight Mini sensor.
            """
            # Code to return an image from the sensor
            image = cv2.cvtColor(self.sensor.get_image(), cv2.COLOR_BGR2RGB)
            return cv2.flip(image, 1)
        
    elif platform.system() == "Windows" or platform.system() == "Darwin":
        def connect(self):
            """
            Connects to the GelSight Mini sensor.
            """
            # Code to connect to the sensor
            self.sensor = cv2.VideoCapture(0, cv2.CAP_AVFOUNDATION)

            if self.sensor is None or not self.sensor.isOpened():
                print('Warning: unable to open video source...')
            
        def disconnect(self):
            """
            Disconnects from the GelSight Mini sensor.
            """
            # Code to disconnect from the sensor
            self.sensor.release()

        def capture_image(self):
            """
            Captures an image from the GelSight Mini sensor.
            """
            # Code to return an image from the sensor
            _, image = self.sensor.read()
            
            image = resize_crop_mini(image, 320, 240)

            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            return cv2.flip(image, 1)