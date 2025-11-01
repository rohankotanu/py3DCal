# touchnet.py
from enum import Enum
import os
from PIL import Image
import numpy as np
import torch
import torch.nn as nn
from torch.hub import load_state_dict_from_url
import torch.optim as optim
from torchvision import transforms
import cv2
import pandas as pd
from sklearn.model_selection import train_test_split
from torch.utils.data import Subset, DataLoader
import math
import copy
import matplotlib.pyplot as plt
import requests
import tarfile
from tqdm import tqdm
from pathlib import Path
from typing import Union
from .dataset import TactileSensorDataset
from .touchnet_architecture import TouchNetArchitecture
from ..lib.fast_poisson import fast_poisson

class SensorType(Enum):
    """
    SensorType: Available sensor types with pretrained weights and compiled datasets
    """
    DIGIT = "DIGIT"
    GELSIGHTMINI = "GelSightMini"
    CUSTOM = "Custom"

class TouchNet:
    """
    TouchNet: A Deep Learning Model for Enhanced Calibration and Sensing of Vision-Based Tactile Sensors
    Args:
        root (str or pathlib.Path, optional): Root directory for datasets and models. Defaults to current directory.
        sensor_type (py3DCal.SensorType, optional): Type of tactile sensor. Defaults to py3DCal.SensorType.CUSTOM.
        load_pretrained_model (bool, optional): If True, loads the pretrained model for the specified sensor type. Defaults to False.
        download_dataset (bool, optional): If True, downloads the dataset for the specified sensor type. Defaults to False.
        device (str, optional): Device to run the model on. Defaults to "cpu".
    """
    def __init__(self, root: Union[str, Path] = Path("."), sensor_type: SensorType = SensorType.CUSTOM, load_pretrained_model: bool = False, download_dataset: bool = False, device: str = "cpu"):

        self._validate_parameters(sensor_type, load_pretrained_model, download_dataset, device)

        self.model = TouchNetArchitecture()
        self.sensor_type = sensor_type
        self.load_pretrained_model = load_pretrained_model
        self.download_dataset = download_dataset
        self.transform = transforms.Compose([transforms.ToTensor()])
        self.root = root
        if sensor_type == SensorType.DIGIT:
            self.dataset_path = os.path.join(root, "digit_calibration_data")
        elif sensor_type == SensorType.GELSIGHTMINI:
            self.dataset_path = os.path.join(root, "gsmini_calibration_data")
        else:
            self.dataset_path = "."
        self.blank_image_path = os.path.join(self.dataset_path, "blank_images", "blank.png")
        self.annotation_path = os.path.join(self.dataset_path, "annotations", "annotations.csv")
        self.device = device

        if self.load_pretrained_model:
            self._load_pretrained_model()

        self.model.to(self.device)
    
        if self.download_dataset:
            self._download_dataset()


    def _validate_parameters(self, sensor_type: SensorType, load_pretrained_model: bool, download_dataset: bool, device: str):
        """
        Validates the parameters.
        Args:
            sensor_type (py3DCal.SensorType): Type of tactile sensor.
            load_pretrained_model (bool): If True, loads the pretrained model for the specified sensor type.
            download_dataset (bool): If True, downloads the dataset for the specified sensor type.
            device (str): Device to run the model on.
        Returns:
            None.
        """
        self._validate_sensor_type(sensor_type)
        self._validate_device(device)
        self._validate_load_pretrained_download_dataset(sensor_type, load_pretrained_model, download_dataset)


    def _validate_device(self, device: str):
        """
        Validates the device by converting it to a torch.device object.
        Args:
            device (str): Device to run the model on.
        Returns:
            None.
        Raises:
            ValueError: If the device is not specified or invalid.
        """
        try:
            device = torch.device(device)
        except Exception as e:
            raise ValueError(
                f"Invalid device '{device}'. Valid options include:\n"
                "  - 'cpu': CPU processing\n"
                "  - 'cuda' or 'cuda:0': NVIDIA GPU\n"
                "  - 'mps': Apple Silicon GPU\n"
                "See: https://pytorch.org/docs/stable/tensor_attributes.html#torch.device"
            ) from e

    def _validate_sensor_type(self, sensor_type):
        """
        Validates the sensor type.
        Args:
            sensor_type (py3DCal.SensorType): Type of tactile sensor.
        Returns:
            None.
        Raises:
            ValueError: If the sensor type is not specified or invalid.
        """
        if sensor_type is None or sensor_type not in [SensorType.DIGIT, SensorType.GELSIGHTMINI, SensorType.CUSTOM]:
            raise ValueError(f"Invalid sensor type: {sensor_type}. Sensor type must be either {SensorType.DIGIT}, {SensorType.GELSIGHTMINI}, or {SensorType.CUSTOM}.")

    def _validate_load_pretrained_download_dataset(self, sensor_type: SensorType, load_pretrained_model: bool, download_dataset: bool):
        """
        Validates that load pretrained and donwload dataset are being called with either SensorType.DIGIT or SensorType.GELSIGHTMINI.
        Args:
            sensor_type (py3DCal.SensorType): Type of tactile sensor.
            load_pretrained_model (bool): If True, loads the pretrained model for the specified sensor type.
            download_dataset (bool): If True, downloads the dataset for the specified sensor type.
        Returns:
            None.
        Raises:
            ValueError: If load pretrained and download dataset are being called with SensorType.CUSTOM.
        """
        if (load_pretrained_model is True or download_dataset is True) and (sensor_type is not SensorType.DIGIT and sensor_type is not SensorType.GELSIGHTMINI):
            raise ValueError("Cannot load pretrained model or download dataset for custom sensor type. Sensor type must be either SensorType.DIGIT or SensorType.GELSIGHTMINI.")

    def set_dataset_path(self, dataset_path: Union[str, Path]):
        """
        Set the dataset path for custom datasets.
        Args:
            dataset_path (str or pathlib.Path): Path to the dataset.
        Returns:
            None.
        """
        self.dataset_path = dataset_path
        self.annotation_path = os.path.join(dataset_path, "annotated_data.csv")
        # self.blank_image_path = os.path.join(dataset_path, "blank.png")
        self.blank_image_path = "../data/sensors/DIGIT/Images/blank.png"

    def set_blank_image_path(self, blank_image_path: Union[str, Path]):
        """
        Set the blank image path for custom datasets.
        Args:
            blank_image_path (str or pathlib.Path): Path to the blank image.
        Returns:
            None.
        """
        self.blank_image_path = blank_image_path

    def train(self, num_epochs: int = 60, batch_size: int = 64, learning_rate: float = 1e-4, train_split: float = 0.8, loss_fn: nn.Module = nn.MSELoss()):
        """
        Train TouchNet model on a dataset for 60 epochs with a
        64 batch size, and AdamW optimizer with learning rate 1e-4.

        Args:
            num_epochs (int): Number of epochs to train for. Defaults to 60.
            batch_size (int): Batch size. Defaults to 64.
            learning_rate (float): Learning rate. Defaults to 1e-4.
            train_split (float): Proportion of data to use for training. Defaults to 0.8.
            loss_fn (nn.Module): Loss function. Defaults to nn.MSELoss().

        Outputs:
            weights.pth: Trained model weights.
            loss.csv: Training and testing losses.
        """
        optimizer = optim.AdamW(self.model.parameters(), lr=learning_rate, weight_decay=1e-4)
        df = pd.read_csv(self.annotation_path, comment='#')
        unique_coords = df[['x_mm', 'y_mm']].drop_duplicates()
        coord_tuples = [(row['x_mm'], row['y_mm']) for _, row in unique_coords.iterrows()]
        train_coords, test_coords = train_test_split(coord_tuples, train_size=train_split, random_state=42)
        train_coords_set = set(train_coords)
        test_coords_set = set(test_coords)
        train_idx = []
        test_idx = []
        for i in range(len(df)):
            coord = (df.loc[i, 'x_mm'], df.loc[i, 'y_mm'])
            if coord in train_coords_set:
                train_idx.append(i)
            elif coord in test_coords_set:
                test_idx.append(i)
        transform = transforms.Compose([transforms.ToTensor()])
        dataset = TactileSensorDataset(dataset_path=os.path.join(self.dataset_path, "probe_images"), annotation_path=self.annotation_path, blank_image_path=self.blank_image_path, transform=transform)
        train_dataset = Subset(dataset, train_idx)
        test_dataset = Subset(dataset, test_idx)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=16, pin_memory=True, persistent_workers=True)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=16, pin_memory=True, persistent_workers=True)
        epoch_train_losses = []
        epoch_test_losses = []
        for epoch in range(num_epochs):
            self.model.train()
            train_loss = 0.0
            for batch_idx, (inputs, targets) in enumerate(train_loader):
                inputs = inputs.to(torch.float32).to(self.device)
                targets = targets.to(torch.float32).to(self.device)
                optimizer.zero_grad()
                outputs = self.model(inputs)

                loss = loss_fn(outputs, targets)
                loss.backward()
                optimizer.step()

                train_loss += loss.item()

                print(f"  [Batch {batch_idx}/{len(train_loader)}] - Loss: {loss.item():.4f}")
            avg_train_loss = train_loss / len(train_loader)
            epoch_train_losses.append(avg_train_loss)
            print(f"Epoch {epoch+1}/{num_epochs} | Train Loss: {avg_train_loss:.4f}")
            self.model.eval()
            test_loss = 0.0
            with torch.no_grad():
                for inputs, targets in test_loader:
                    inputs = inputs.to(torch.float32).to(self.device)
                    targets = targets.to(torch.float32).to(self.device)
                    outputs = self.model(inputs)
                    loss = loss_fn(outputs, targets)
                    test_loss += loss.item()

            avg_test_loss = test_loss / len(test_loader)
            epoch_test_losses.append(avg_test_loss)
            print(f"TEST LOSS: {avg_test_loss:.4f}")

        with open("losses.csv", "w") as f:
            f.write("epoch,train_loss,test_loss\n")
            for i in range(len(epoch_train_losses)):
                f.write(f"{i+1},{epoch_train_losses[i]},{epoch_test_losses[i]}\n")
        torch.save(self.model.state_dict(), "weights.pth")

    def _add_coordinate_channels(self, image: torch.Tensor) -> torch.Tensor:
        """
        Adds positional embedding to the input image by appending x and y coordinate channels.
        
        Args:
            image (torch.Tensor): Input image tensor of shape (C, H, W).
        
        Returns:
            torch.Tensor: Image tensor with added coordinate channels of shape (C+2, H, W).
            - X channel: column indices (1s in first column, 2s in second column, etc.)
            - Y channel: row indices (1s in first row, 2s in second row, etc.)
        """
        # Get image dimensions
        _, height, width = image.shape
        
        # Create x coordinate channel (column indices)
        x_coords = torch.arange(1, width + 1, dtype=torch.float32).unsqueeze(0).repeat(height, 1)
        x_channel = x_coords.unsqueeze(0)  # Add channel dimension
        
        # Create y coordinate channel (row indices)
        y_coords = torch.arange(1, height + 1, dtype=torch.float32).unsqueeze(1).repeat(1, width)
        y_channel = y_coords.unsqueeze(0)  # Add channel dimension
        
        # Concatenate original image with coordinate channels
        image_with_coords = torch.cat([image, x_channel, y_channel], dim=0)
        
        return image_with_coords
    
    def get_depthmap(self, image_path: str) -> np.ndarray:
        """
        Returns the depthmap for a given input image.
        Args:
            image_path (str): Path to the input image.
        Returns:
            depthmap (numpy.ndarray): The computed depthmap.
        """
        model = self.model.to(self.device)
        model.eval()
        image = Image.open(image_path)
        image_pil = image.convert("RGB")
        transformed_image = self.transform(image_pil)
        blank_image = Image.open(self.blank_image_path)
        blank_image_pil = blank_image.convert("RGB")
        transformed_blank_image = self.transform(blank_image_pil)
        augmented_image = transformed_image - transformed_blank_image
        augmented_image = self._add_coordinate_channels(augmented_image)
        augmented_image = augmented_image.unsqueeze(0).to(self.device)

        with torch.no_grad():
            output = model(augmented_image)
        
        output = output.squeeze(0).permute(1, 2, 0).cpu().numpy()
        
        depthmap = fast_poisson(output[:,:,0], output[:,:,1])

        depthmap = np.clip(-depthmap, a_min=0, a_max=None)

        return depthmap

    def save_depthmap_image(self, image_path: str, save_path: Union[str, Path] = Path("depthmap.png")):
        """
        Save an image of the depthmap for a given input image.
        Args:
            image_path (str): Path to the input image.
            save_path (str or pathlib.Path): Path to save the depthmap image.
        Returns:
            None.
        """
        depthmap = self.get_depthmap(image_path)

        plt.imsave(save_path, depthmap, cmap='viridis')

    def show_depthmap(self, image_path: str):
        """
        Show the depthmap for a given input image.
        Args:
            image_path (str): Path to the input image.
        Returns:
            None.
        """
        depthmap = self.get_depthmap(image_path)

        plt.imshow(depthmap)
        plt.show()
    
    def _load_pretrained_model(self):
        """
        Loads a pretrained model for either the DIGIT or GelSightMini sensor.
        Args:
            None.
        Returns:
            None.
        """
        if self.sensor_type == SensorType.DIGIT:
            file_path = os.path.join(self.root, "digit_pretrained_weights.pth")
            
            # Check if DIGIT pretrained weights exist locally, if not download them
            if not os.path.exists(file_path):

                print(f"Downloading DIGIT pretrained weights ...")
                response = requests.get('https://zenodo.org/records/17487330/files/digit_pretrained_weights.pth?download=1', stream=True)
                response.raise_for_status()

                total_size = int(response.headers.get('content-length', 0))
                block_size = 1024

                # Save file in chunks to handle large datasets
                with open(file_path, 'wb') as f, tqdm(
                    total=total_size,
                    unit='B',
                    unit_scale=True,
                    desc="Downloading",
                    ncols=80
                ) as progress_bar:
                    for chunk in response.iter_content(chunk_size=block_size):
                        if chunk:
                            f.write(chunk)
                            progress_bar.update(len(chunk))

                print(f"Download complete!")
            else:
                print(f"DIGIT pretrained weights already exists at: {file_path}/")

        elif self.sensor_type == SensorType.GELSIGHTMINI:
            file_path = os.path.join(self.root, "gsmini_pretrained_weights.pth")

            # Check if GelSight Mini pretrained weights exist locally, if not download them
            if not os.path.exists(file_path):

                print(f"Downloading GelSight Mini pretrained weights ...")
                response = requests.get('https://zenodo.org/records/17487330/files/gsmini_pretrained_weights.pth?download=1', stream=True)
                response.raise_for_status()

                total_size = int(response.headers.get('content-length', 0))
                block_size = 1024

                # Save file in chunks to handle large datasets
                with open(file_path, 'wb') as f, tqdm(
                    total=total_size,
                    unit='B',
                    unit_scale=True,
                    desc="Downloading",
                    ncols=80
                ) as progress_bar:
                    for chunk in response.iter_content(chunk_size=block_size):
                        if chunk:
                            f.write(chunk)
                            progress_bar.update(len(chunk))

                print(f"Download complete!")
            else:
                print(f"GelSight Mini pretrained weights already exists at: {file_path}/")

        state_dict = torch.load(file_path, map_location="cpu")
        self.model.load_state_dict(state_dict)
    
    def load_model_weights(self, model_path: Union[str, Path]):
        """
        Loads custom model weights from a .pth file.
        Args:
            model_path (str or pathlib.Path): Path to the model weights file.
        Returns:
            None.
        """
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))

    def _download_dataset(self):
        """
        Downloads the dataset for either the SensorType.DIGIT or SensorType.GELSIGHTMINI sensor. Used for model training.
        Args:
            None.
        Returns:
            None.
        """
        if self.sensor_type == SensorType.DIGIT:
            self.dataset_path = os.path.join(self.root, "digit_calibration_data")

            # Check if self.dataset_path exists
            if not os.path.exists(self.dataset_path):
                os.makedirs(self.root, exist_ok=True)

                tar_path = os.path.join(self.root, "digit_calibration_data.tar.gz")

                print(f"Downloading DIGIT dataset ...")
                response = requests.get('https://zenodo.org/records/17487330/files/digit_calibration_data.tar.gz?download=1', stream=True)
                response.raise_for_status()

                total_size = int(response.headers.get('content-length', 0))
                block_size = 1024

                # Save file in chunks to handle large datasets
                with open(tar_path, 'wb') as f, tqdm(
                    total=total_size,
                    unit='B',
                    unit_scale=True,
                    desc="Downloading",
                    ncols=80
                ) as progress_bar:
                    for chunk in response.iter_content(chunk_size=block_size):
                        if chunk:
                            f.write(chunk)
                            progress_bar.update(len(chunk))

                print(f"Download complete!")

                # Extract .tar.gz file
                print("Extracting files ...")
                with tarfile.open(tar_path, "r:gz") as tar:
                    tar.extractall(path=self.root)

                os.remove(tar_path)

                print(f"Extraction complete! Files are in: {self.root}/")

            else:
                print(f"DIGIT dataset already exists at: {self.dataset_path}/")

            
        elif self.sensor_type == SensorType.GELSIGHTMINI:
            self.dataset_path = os.path.join(self.root, "gsmini_calibration_data")

            # Check if self.dataset_path exists
            if not os.path.exists(self.dataset_path):
                os.makedirs(self.root, exist_ok=True)
                
                tar_path = os.path.join(self.root, "gsmini_calibration_data.tar.gz")

                print(f"Downloading GelSight Mini dataset ...")
                response = requests.get('https://zenodo.org/records/17487330/files/gsmini_calibration_data.tar.gz?download=1', stream=True)
                response.raise_for_status()

                total_size = int(response.headers.get('content-length', 0))
                block_size = 1024

                # Save file in chunks to handle large datasets
                with open(tar_path, 'wb') as f, tqdm(
                    total=total_size,
                    unit='B',
                    unit_scale=True,
                    desc="Downloading",
                    ncols=80
                ) as progress_bar:
                    for chunk in response.iter_content(chunk_size=block_size):
                        if chunk:
                            f.write(chunk)
                            progress_bar.update(len(chunk))

                print(f"Download complete!")

                # Extract .tar.gz file
                print("Extracting files ...")
                with tarfile.open(tar_path, "r:gz") as tar:
                    tar.extractall(path=self.root)

                os.remove(tar_path)

                print(f"Extraction complete! Files are in: {self.root}/")

            else:
                print(f"GelSight Mini dataset already exists at: {self.dataset_path}/")

        self.annotation_path = os.path.join(self.dataset_path, "annotations", "annotations.csv")
        self.blank_image_path = os.path.join(self.dataset_path, "blank_images", "blank.png")

    def _fit_circle(self, img_path, instructions, initial_pos=(100,100,30)):
        """
        Fits a circle to an image.
        Args:
            img_path: Path to the image.
            instructions: Instructions to display on the image.
            initial_pos: Initial position of the circle.
        Returns:
            x: x-coordinate of the circle.
            y: y-coordinate of the circle.
            r: radius of the circle.
        """
        x, y, r = initial_pos

        # Read the image
        image = cv2.imread(img_path)

        # Define rectangle parameters
        rectangle_height = 250  # Adjust as needed
        rectangle_color = (255, 255, 255)  # White color (BGR format)

        # Create a new image with the rectangle
        new_image = np.zeros((image.shape[0] + rectangle_height, image.shape[1], 3), dtype=np.uint8)
        new_image[:image.shape[0], :, :] = image
        cv2.rectangle(new_image, (0, image.shape[0]), (image.shape[1], image.shape[0] + rectangle_height), rectangle_color, -1)

        # Add text inside the rectangle
        for i, line in enumerate(instructions.split('\n')):
            str_y = image.shape[0] + 40 + i*30
            cv2.putText(new_image, line, (10, str_y ), cv2.FONT_HERSHEY_SIMPLEX, 1, 2)

        while True:
            # Create a copy of the original image
            annotated_image = new_image.copy()

            # Draw the circle
            cv2.circle(annotated_image, (x, y), r, (0, 0, 255), 1)

            # Display the image
            cv2.imshow('Circle Fitting', annotated_image)

            # Update circle position based on key presses
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('w'):
                y -= 1
            elif key == ord('s'):
                y += 1
            elif key == ord('a'):
                x -= 1
            elif key == ord('d'):
                x += 1
            elif key == ord('r'):
                r += 1
            elif key == ord('f'):
                r -= 1

        return x, y, r

    def _scale_px_to_mm(self, root_dir, csv_path, instructions, initial_val, anchor_idx, circle_vals):
        """
        Scales the pixel-to-millimeter calibration.
        Args:
            root_dir: Path to the root directory.
            csv_path: Path to the CSV file.
            instructions: Instructions to display on the image.
            initial_val: Initial value of the pixel/mm ratio.
            anchor_idx: Index of the anchor image.
            circle_vals: Values of the circles.
        Returns:
            px_per_mm: Pixel/millimeter ratio.
            calibration_data: Calibration data.
        """
        # Convert data labels into dataframe
        calibration_data = pd.read_csv(csv_path)
        
        achor_image_path = os.path.join(root_dir, calibration_data["img_name"][anchor_idx])
        anchor_image = cv2.imread(achor_image_path)
        image_list = [anchor_idx]

        height, width, _ = anchor_image.shape 

        # Define rectangle parameters
        rectangle_height = 250  # Adjust as needed
        rectangle_color = (255, 255, 255)  # White color (BGR format)

        # Create a new image with the rectangle
        new_image = np.zeros((height * 3 + rectangle_height, width * 3, 3), dtype=np.uint8)
        new_image[:height, :width, :] = anchor_image
        cv2.rectangle(new_image, (0, height * 3), (width * 3, height * 3 + rectangle_height), rectangle_color, -1)

        px_per_mm = initial_val
        circle1_x, circle1_y, circle1_r = circle_vals
        img1_x_mm = calibration_data["x_mm"][anchor_idx]
        img1_y_mm = calibration_data["y_mm"][anchor_idx]

        calibration_data['x_px'] = circle1_x + (calibration_data['x_mm'] - img1_x_mm) * px_per_mm
        calibration_data['y_px'] = circle1_y + (img1_y_mm - calibration_data['y_mm']) * px_per_mm

        idx = 1

        while len(image_list) < 9:
            random_row = calibration_data.sample(n=1)

            if random_row.iloc[0,4] < width and random_row.iloc[0,5] < height:
                image_path = os.path.join(root_dir, random_row.iloc[0,0])
                image = cv2.imread(image_path)

                image_list += [random_row.index[0]]

                row = math.floor(idx / 3)
                column = idx % 3

                new_image[(height * row):(height * (row+1)), (width * column):(width * (column+1)), :] = image

                idx += 1

        # Add text inside the rectangle
        for i, line in enumerate(instructions.split('\n')):
            str_y = height * 3 + 40 + i*30
            cv2.putText(new_image, line, (10, str_y ), cv2.FONT_HERSHEY_SIMPLEX, 1, 2)

        # Display the image
        cv2.imshow('Circle Fitting', new_image)
        
        while True:
            # Create a copy of the original image
            annotated_image = new_image.copy()

            calibration_data['x_px'] = circle1_x + (calibration_data['x_mm'] - img1_x_mm) * px_per_mm
            calibration_data['y_px'] = circle1_y + (img1_y_mm - calibration_data['y_mm']) * px_per_mm

            # Draw the circles
            for i in range(9):
                row = math.floor(i / 3)
                column = i % 3

                idx = image_list[i]
                x = int(calibration_data['x_px'][idx]) + column * width
                y = int(calibration_data['y_px'][idx]) + row * height
                
                cv2.circle(annotated_image, (x, y), circle1_r, (0, 0, 255), 1)

            # Display the image
            cv2.imshow('Circle Fitting', annotated_image)

            # Update circle position based on key presses
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('w'):
                circle1_y -= 1
            elif key == ord('s'):
                circle1_y += 1
            elif key == ord('a'):
                circle1_x -= 1
            elif key == ord('d'):
                circle1_x += 1
            elif key == ord('r'):
                px_per_mm += 1
            elif key == ord('f'):
                px_per_mm -= 1

        return px_per_mm, calibration_data

    def annotate_data(self, dataset_path, csv_file="sensor_data.csv", img_idxs=None):
        """
        Tool to annotate custom dataset with pixel-to-millimeter calibration.
        Creates an annotated_data.csv file required for training.
        
        Controls:
            - w/s: Move circle up/down
            - a/d: Move circle left/right
            - r/f: Increase/decrease circle size or pixel/mm ratio
            - q: Proceed to next step
        
        Args:
            dataset_path: Path to the dataset directory containing images and CSV file.
            csv_file: Name of the CSV file containing sensor data. Default: "sensor_data.csv".
            img_idxs: Tuple of two image indices to use for calibration. Default: None (auto-selected at 25th and 75th percentiles).
        
        Returns:
            Saves annotated_data.csv in the dataset_path directory with pixel coordinates.
        """
        # Convert data labels into dataframe
        csv_path = os.path.join(dataset_path, csv_file)
        calibration_data = pd.read_csv(csv_path)

        # Extract data from middle of the sensor (media Y value)
        middle_row = calibration_data.loc[calibration_data['y_mm'] == calibration_data["y_mm"].median()]

        if img_idxs is None:
            # Get the indices of the 25th percentile and 75th percentile X values
            img1 = middle_row.loc[middle_row['x_mm'] == middle_row['x_mm'].quantile(0.25)]
            img2 = middle_row.loc[middle_row['x_mm'] == middle_row['x_mm'].quantile(0.75)]
            idx_1 = img1.index[0]
            idx_2 = img2.index[0]
        else:
            idx_1 = img_idxs[0]
            idx_2 = img_idxs[1]

        # Get the image names and XY probe coordinates
        image1_name = os.path.join(dataset_path, calibration_data.iloc[idx_1, 0])
        img1_x_mm = calibration_data.iloc[idx_1, 1]
        img1_y_mm = calibration_data.iloc[idx_1, 2]

        image2_name = os.path.join(dataset_path, calibration_data.iloc[idx_2, 0])
        img2_x_mm = calibration_data.iloc[idx_2, 1]
        img2_y_mm = calibration_data.iloc[idx_2, 2]

        # Present the images to the user and have them fit the circles
        circle1_x, circle1_y, circle1_r = self._fit_circle(image1_name, instructions="w: Up\ns: Down\na: Left\nd: Right\nr: Bigger\nf: Smaller\nq: Next")
        circle2_x, circle2_y, circle2_r = self._fit_circle(image2_name, instructions="w: Up\ns: Down\na: Left\nd: Right\nr: Bigger\nf: Smaller\nq: Next", initial_pos=(circle1_x, circle1_y, circle1_r))

        print(circle1_x, circle1_y, circle1_r)
        print(circle2_x, circle2_y, circle2_r)

        # Close all opencv windows
        cv2.destroyAllWindows()

        # Calculate pixels/mm
        known_x_distance = abs(img2_x_mm - img1_x_mm) # mm
        px_per_mm = abs(circle2_x - circle1_x) / known_x_distance

        px_per_mm, calibration_data = self._scale_px_to_mm(root_dir=dataset_path, csv_path=csv_path, instructions="w: Up\ns: Down\na: Left\nd: Right\nr: Increase pixel/mm value\nf: Decrease pixel/mm value\nq: Quit", initial_val=px_per_mm, anchor_idx=idx_1, circle_vals=(circle1_x, circle1_y, circle1_r))
        
        # Print out the pixels/mm value
        print("pixels per mm:", px_per_mm)

        # Create CSV file with annotated data
        annotated_file_path = os.path.join(dataset_path, "annotated_data.csv")
        calibration_data.to_csv(annotated_file_path, index=False)