import copy
import pandas as pd
from sklearn.model_selection import train_test_split
from .tactile_sensor_dataset import TactileSensorDataset

def split_dataset(dataset, train_ratio=0.8):
    """
    Splits a dataset into training and validation sets.

    Args:
        dataset (py3DCal.datasets.TactileSensorDataset): The dataset to split.
        train_ratio (float): The proportion of the dataset to include in the training set. Default is 0.8.

    Returns:
        tuple: A tuple containing the training and validation datasets.
    """
    if not isinstance(dataset, TactileSensorDataset):
        raise TypeError("Expected dataset to be an instance of py3DCal.datasets.TactileSensorDataset")

    df = dataset.data.copy()

    unique_coords = df[['x_mm', 'y_mm']].drop_duplicates().reset_index(drop=True)

    train_df, val_df = train_test_split(unique_coords, train_size=train_ratio, random_state=42)

    # Merge back to get full rows
    train_data = pd.merge(df, train_df, on=['x_mm', 'y_mm'])
    val_data = pd.merge(df, val_df, on=['x_mm', 'y_mm'])

    # Create two copies of the original dataset
    train_dataset = copy.deepcopy(dataset)
    val_dataset = copy.deepcopy(dataset)

    # Update the data attribute of each dataset
    train_dataset.data = train_data
    val_dataset.data = val_data

    return train_dataset, val_dataset