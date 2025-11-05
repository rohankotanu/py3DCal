import numpy as np
import time
import csv
import os
from typing import Union
from pathlib import Path
from PIL import Image
from tqdm import tqdm

from .Printers.Printer import Printer
from .Sensors.Sensor import Sensor

class Calibrator:
    """ Calibrator class to automatically probe a tactile sensor.
    Args:
        printer (Printer): An instance of a Printer class.
        sensor (Sensor): An instance of a Sensor class.
    """
    def __init__(self, printer: Printer, sensor: Sensor):
        self.printer = printer
        self.sensor = sensor

        self.printer_connected = False
        self.sensor_connected = False

        try:
            self.printer_name = self.printer.name
        except:
            self.printer_name = "printer"
        try:
            self.sensor_name = self.sensor.name
        except:
            self.sensor_name = "tactile"
    
    def connect_printer(self):
        """ Connects to the 3D Printer

        Returns:
            bool: Returns True if connection was successful.
        """
        print("Connecting to " + str(self.printer_name) + "...")

        try:
            self.printer.connect()
            self.printer_connected = True

            print("Connected to " + str(self.printer_name) + "!")
            print("")
            return True
        except:
            self.printer_connected = False
            print("Error connecting to " + str(self.printer_name) + ".")
            print("")
            return False
        
    def disconnect_printer(self):
        """ Disconnects from the 3D Printer

        Returns:
            bool: Returns True if disconnection was successful.
        """
        print("Disconnecting from " + str(self.printer_name) + "...")

        try:
            self.printer.disconnect()
            self.printer_connected = False
            print("Disconnected from " + str(self.printer_name) + "!")
            print("")
            return True
        except:
            print("Error disconnecting from " + str(self.printer_name) + ".")
            print("")
            return False

    def initialize_printer(self):
        """ Sends gcode to configure and home 3D Printer

        Returns:
            bool: Returns True if initialization was successful.
        """
        # Probe must be detached to home printer
        print("Make sure probe is not attached to print head. ", end="")
        input("Press Enter to continue...")
        print("")
        
        if not self.printer_connected:
            self.connect_printer()
        
        try:
            print("Initializing printer...")

            self.printer.initialize()

            print("Printer initialization complete!")
            print("")
            return True
        except:
            print("Error sending initialization gcode to printer.")
            print("")
            return False
        
    def connect_sensor(self):
        """ Connects to the sensor

        Returns:
            bool: Returns True if connection was successful.
        """
        print("Connecting to " + self.sensor_name + " sensor...")

        try:
            # Connect to sensor
            self.sensor.connect()
            print("Connected to sensor!")
            print("")
        except:
            print("Error connecting to sensor.")
            print("")

    def disconnect_sensor(self):
        """ Disconnects from the sensor

        Returns:
            bool: Returns True if disconnection was successful.
        """
        print("Disconnecting from " + self.sensor.name + " sensor...")

        self.sensor.connect()
        try:
            self.printer.disconnect()
            self.printer_connected = False
            print("Disconnected from sensor!")
            print("")
            return True
        except:
            print("Error disconnecting from sensor.")
            print("")
            return False

    def probe(self, home_printer: bool = True, save_images: bool = True, calibration_file_path: Union[str, Path] = None, data_save_path: str = "."):
        """ Executes the probing procedure on 3D printer

        Args:
            home_printer (bool, optional): Determines whether to home the printer prior to probing. Defaults to True.
            save_images (bool, optional): Determines whether sensor images are saved. Defaults to True.
            calibration_file_path (str, optional): The path of the calibration file. For the DIGIT and GelSight Mini,
                if no file is specified, a default calibration file will be used.
            data_save_path (str, optional): The folder in which the data should be saved. If no folder is specified,
                data will be stored in a directory named "sensor_calibration_data" within the current working directory.

        Returns:
            bool: Returns True when the probing procedure is complete.
        """
        # Connect to 3D printer if not already connected
        if not self.printer_connected:
            self.connect_printer()
            self.printer.send_gcode("M117 Sensor Calibration In Progress")

        # Connect to sensor
        if save_images == True:
            self.connect_sensor()

            for i in range(30):
                self.sensor.capture_image()

        # Send initialization gcode to printer
        if home_printer == True:
            self.initialize_printer()

        # If no data path was provided, set default path to a folder called "sensor_calibration_data" in the Downloads folder
        if data_save_path is not None:
            data_save_path = os.path.join(data_save_path, "sensor_calibration_data")
        
        # Create necessary directories if they don't exist
        Path(data_save_path).mkdir(parents=True, exist_ok=True)
        Path(os.path.join(data_save_path, "annotations")).mkdir(parents=True, exist_ok=True)
        Path(os.path.join(data_save_path, "blank_images")).mkdir(parents=True, exist_ok=True)
        Path(os.path.join(data_save_path, "probe_images")).mkdir(parents=True, exist_ok=True)

        # Open a csv file to write calibration data
        with open(os.path.join(data_save_path, "annotations", "probe_data.csv"), 'w', newline='') as csv_file:
            csv_writer = csv.writer(csv_file)
            csv_writer.writerow(['img_name', 'x_mm', 'y_mm', 'penetration_depth_mm'])

        # Save blank image
        if save_images == True:
            blank_img = self.sensor.capture_image()
            img = Image.fromarray(blank_img)
            img.save(os.path.join(data_save_path, "blank_images", "blank.png"))

        # If no calibration file path was provided, use the default calibration file for the specified sensor
        if calibration_file_path == None:
            calibration_file_path = self.sensor.default_calibration_file

        # Load CSV file into numpy array
        self.calibration_points = np.genfromtxt(calibration_file_path, delimiter=',', skip_header=1)

        # Get number of rows (i.e. calibration points)
        N = self.calibration_points.shape[0]

        # Variable to keep track of image number
        img_idx = 0

        # Move to offset Z position
        self.printer.send_gcode("G0  Z" + str(self.sensor.z_offset + self.sensor.z_clearance))

        time.sleep(1 + (self.sensor.z_offset + self.sensor.z_clearance) / 4)

        print("Attach probe to printer head. ", end="")
        input("Press Enter to continue...")
        print("")

        # Move to offset XY position
        self.printer.send_gcode("G0 X" + str(self.sensor.x_offset) + " Y" + str(self.sensor.y_offset))

        time.sleep(1 + max(self.sensor.x_offset, self.sensor.y_offset) / 10)

        print("Beginning sensor calibration procedure...")
        print("")

        x_prev = self.sensor.x_offset
        y_prev = self.sensor.y_offset
        z_prev = self.sensor.z_offset + self.sensor.z_clearance

        # Loop through every calibration point
        for i in tqdm(range(N), desc="Sensor Calibration Progress"):
            # If specified penetration depth exceeds maximum value, print message
            if abs(self.calibration_points[i][2]) > self.sensor.max_penetration:
                print("Line " + str(i+1) + ": Maximum penetration depth for sensor exceeded. Skipping calibration point.")
            # If penetration depth does not exceed maximum value, move to calibration point
            else:
                # Get absolute XYZ coordinates
                x = self.sensor.x_offset + self.calibration_points[i][0]
                y = self.sensor.y_offset + self.calibration_points[i][1]
                z = self.sensor.z_offset - abs(self.calibration_points[i][2])

                # Move to Z clearance height
                self.printer.send_gcode("G0 Z" + str(self.sensor.z_offset + self.sensor.z_clearance))

                # Move to desired XY location
                self.printer.send_gcode("G0 X" + str(x) + " Y" + str(y))

                # Move to desired Z penetration
                self.printer.send_gcode("G0 Z" + str(z))
                
                # Calculate time required to reach position
                # Assumes x and y speed of 10 mm/s, z speed of 4 mm/s
                travel_time = abs(self.sensor.z_offset + self.sensor.z_clearance - z_prev) / 4 + max(abs(x - x_prev), abs(y - y_prev)) / 10 + abs(z - (self.sensor.z_offset + self.sensor.z_clearance)) / 4
                time.sleep(travel_time + 1)

                # Update variables
                x_prev = x
                y_prev = y
                z_prev = z

                # Take desired number of images
                if save_images == True:
                    with open(os.path.join(data_save_path, "annotations", "probe_data.csv"), 'a', newline='') as csv_file:
                        csv_writer = csv.writer(csv_file)

                        for j in range(int(self.calibration_points[i][3])):
                            frame = self.sensor.capture_image()

                            img_name =  str(img_idx) + "_" + "X" + str(self.calibration_points[i][0]) + "Y" + str(self.calibration_points[i][1]) + "Z" + str(self.calibration_points[i][2]) + ".png"
                            img_path = os.path.join(data_save_path, "probe_images",img_name)

                            img = Image.fromarray(frame)
                            img.save(img_path)

                            csv_writer.writerow([img_name, self.calibration_points[i][0], self.calibration_points[i][1], self.calibration_points[i][2]])

                            img_idx += 1

                            time.sleep(0.5)

                else:
                    with open(os.path.join(data_save_path, "annotations", "probe_data.csv"), 'a', newline='') as csv_file:
                        csv_writer = csv.writer(csv_file)
                        csv_writer.writerow(["---", self.calibration_points[i][0], self.calibration_points[i][1], self.calibration_points[i][2]])

        # Move to Z clearance height
        self.printer.send_gcode("G0 Z" + str(self.sensor.z_offset + self.sensor.z_clearance))

        print("")

        # Update printer display
        self.printer.send_gcode("M117 Sensor Calibration Complete!")

        # Disconnect from 3D printer
        self.disconnect_printer()

        # Disconnect from sensor
        if save_images == True:
            self.disconnect_sensor()

        print("Sensor calibration procedure complete!")
        print("")

        return True