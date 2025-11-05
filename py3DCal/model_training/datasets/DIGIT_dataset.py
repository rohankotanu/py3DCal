import os
import requests
import tarfile
from tqdm import tqdm
from typing import Union
from pathlib import Path
from .tactile_sensor_dataset import TactileSensorDataset
from ..lib.validate_parameters import validate_root


class DIGIT(TactileSensorDataset):
    """
    DIGIT: A Dataset Class for the DIGIT sensor
    Args:
        root (str or pathlib.Path): The root directory containing digit_calibration_data.
        download (bool, optional): If True, downloads the dataset for the specified sensor type. Defaults to False.
        add_coordinate_embeddings (bool, optional): If True, adds xy coordinate embeddings to each image. Defaults to True.
        subtract_blank (bool, optional): If True, subtracts a blank image from each input image. Defaults to False.
        transform (callable, optional): A function/transform that takes in an PIL image and returns a transformed version. Default: ``transforms.ToTensor()``
    """
    def __init__(self, root: Union[str, Path] = Path("."), download=False, add_coordinate_embeddings=True, subtract_blank=True, transform=None):
        validate_root(root)

        self.root = root

        self.dataset_path = os.path.join(self.root, "digit_calibration_data")

        if download:
            self._download_dataset()

        super().__init__(root=self.dataset_path, add_coordinate_embeddings=add_coordinate_embeddings, subtract_blank=subtract_blank, transform=transform)

    def _download_dataset(self):
        """
        Downloads the dataset for either the DIGIT sensor.

        """
        # Check if self.dataset_path exists
        if not os.path.exists(self.dataset_path):
            os.makedirs(self.root, exist_ok=True)

            tar_path = os.path.join(self.root, "digit_calibration_data.tar.gz")

            print(f"Downloading DIGIT dataset ...")
            response = requests.get('https://zenodo.org/records/17517028/files/digit_calibration_data.tar.gz?download=1', stream=True)
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

