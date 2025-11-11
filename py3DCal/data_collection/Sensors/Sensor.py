from abc import ABC, abstractmethod

class Sensor(ABC):
    """
    Sensor: An abstract base class for tactile sensors.
    """
    def __init__(self):
        self.name = ""
        self.x_offset = 5
        self.y_offset = 5
        self.z_offset = 5
        self.z_clearance = 2
        self.max_penetration = 0
        self.default_calibration_file = "calibration_procs/digit/default.csv"

    @abstractmethod
    def connect(self):
        """ Connects to the sensor
        """
        pass

    @abstractmethod
    def disconnect(self):
        """ Disconnects from the sensor
        """
        pass

    @abstractmethod
    def capture_image(self):
        """ Captures an image from the sensor

        Returns:
            numpy.ndarray: The image from the sensor.
        """
        pass

    def flush_frames(self, n: int = 5):
        """Discards the next n frames to clear camera buffer.
        
        Args:
            n (int): Number of frames to discard. Default is 5.
        """
        for _ in range(n):
            self.capture_image()