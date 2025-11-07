import os
import cv2
import math
import json
import numpy as np
import pandas as pd
from typing import Union
from pathlib import Path
from matplotlib import pyplot as plt
from matplotlib.patches import Circle
from .validate_parameters import validate_root

def annotate(dataset_path: Union[str, Path], probe_radius_mm: Union[int, float], img_idxs=None):
    """
    Tool to annotate custom dataset with pixel-to-millimeter calibration.
    Creates an annotated_data.csv file required for training.
    
    Controls:
        - w/s: Move circle up/down
        - a/d: Move circle left/right
        - r/f: Increase/decrease circle size or pixel/mm ratio
        - q: Proceed to next step
    
    Args:
        dataset_path (str or pathlib.Path): Path to the dataset directory.
        probe_radius_mm (int or float): Radius of the probe used to collect data (in mm).
        img_idxs (tuple or list, optional): The two image indices to use for circle fitting. Default: None (auto-selects images at 25th and 75th percentile columns of middle row).
    
    Returns:
        Saves annotated_data.csv in the dataset_path/annotations directory.
    """
    validate_root(dataset_path, must_exist=True)
    validate_probe_radius(probe_radius_mm)
    validate_indices(img_idxs)

    # Open probe data
    probe_data_path = os.path.join(dataset_path, "annotations", "probe_data.csv")
    probe_data = pd.read_csv(probe_data_path)

    # Get middle row
    middle_row = probe_data.loc[probe_data["y_mm"] == probe_data["y_mm"].median()]

    # Get 25th and 75th percentile indices if img_idxs not provided
    if img_idxs is None:
        # Get the indices of the 25th percentile and 75th percentile X values
        idx1 = middle_row.loc[middle_row["x_mm"] == middle_row["x_mm"].quantile(0.25)].index[0]
        idx2 = middle_row.loc[middle_row["x_mm"] == middle_row["x_mm"].quantile(0.75)].index[0]
    else:
        idx1 = img_idxs[0]
        idx2 = img_idxs[1]

    # Get the image names and probe coordinates
    image1_name = os.path.join(dataset_path, "probe_images", probe_data["img_name"][idx1])
    img1_x_mm = probe_data["x_mm"][idx1]
    img1_y_mm = probe_data["y_mm"][idx1]

    image2_name = os.path.join(dataset_path, "probe_images", probe_data["img_name"][idx2])
    img2_x_mm = probe_data["x_mm"][idx2]
    img2_y_mm = probe_data["y_mm"][idx2]

    # Blank image path
    blank_image_path = os.path.join(dataset_path, "blank_images", "blank.png")

    # Fit 2 circles
    circle1_x, circle1_y, circle1_r = _fit_circle(image1_name, blank_image_path)
    circle2_x, _, _ = _fit_circle(image2_name, blank_image_path)

    # Compute pixels/mm
    dx_mm = abs(img2_x_mm - img1_x_mm)
    px_per_mm = abs(circle2_x - circle1_x) / dx_mm

    # Fine tune the fitting
    px_per_mm, annotations = _adjust_fitting(dataset_path, anchor_idx=idx1, px_per_mm=px_per_mm, anchor_data=(circle1_x, circle1_y, circle1_r))

    print("pixels per mm:", px_per_mm)

    # Save metadata file
    metadata_path = os.path.join(dataset_path, "annotations", 'metadata.json')
    data = {"px_per_mm": px_per_mm, "probe_radius_mm": probe_radius_mm}
    with open(metadata_path, "w") as json_file:
        json.dump(data, json_file, indent=4)

    # Create CSV file with annotated data
    annotations_path = os.path.join(dataset_path, "annotations", "annotations.csv")
    annotations.to_csv(annotations_path, index=False)

def _fit_circle(image_path: Union[str, Path], blank_image_path: Union[str, Path]):
        """
        Fits a circle to an image.

        Args:
            image_path: Path to the image.
            blank_image_path: Path to the blank image.

        Returns:
            x: x-coordinate of the circle.
            y: y-coordinate of the circle.
            r: radius of the circle.
        """
        # Load original image (default view)
        image = cv2.cvtColor(cv2.imread(image_path), cv2.COLOR_BGR2RGB)
        blank_image = cv2.cvtColor(cv2.imread(blank_image_path), cv2.COLOR_BGR2RGB)
        bitwise_not_blank = cv2.bitwise_not(blank_image)

        # Initial circle position and radius
        x = image.shape[1] // 2
        y = image.shape[0] // 2
        r = 30

        # Flags for image display modes
        subtract_blank = False
        bitwise_not = False

        # Disable Matplotlib’s conflicting keymaps
        plt.rcParams['keymap.save'] = []
        plt.rcParams['keymap.fullscreen'] = []

        # Prepare figure with two subplots: text (left), image (right)
        fig, (ax_text, ax_img) = plt.subplots(1, 2, figsize=(14, 8), gridspec_kw={'width_ratios': [1, 3]})
        plt.subplots_adjust(wspace=0.4, bottom=0, top=1, left=0, right=1)

        fig.canvas.manager.set_window_title('Fit Circle to Generated Annotations')

        # Right: Image panel
        img_artist = ax_img.imshow(image)
        ax_img.set_axis_off()
        circle_artist = plt.Circle((x, y), r, color='red', fill=False, linewidth=1)
        ax_img.add_patch(circle_artist)
        center_artist, = ax_img.plot(x, y, marker='*', color='lime', markersize=6)

        # Left: Instruction panel
        ax_text.set_axis_off()
        ax_text.text(
            0.30, 0.75,
            "Commands:\n\nw: Up\ns: Down\na: Left\nd: Right\nr: Bigger\nf: Smaller\nq: Next\n\n\n1: View 1 (RGB image)\n2: View 2 (Difference image)\n3: View 3 (Bitwise not image)",
            fontsize=20, color='black', va='top', ha='left', wrap=True
        )

        plt.ion()
        plt.show(block=False)

        done = False

        def on_key(event):
            nonlocal x, y, r, done, subtract_blank, bitwise_not, image, blank_image, bitwise_not_blank

            if event.key == 'q':
                done = True
            elif event.key in ('w', 'up'):
                y -= 1
            elif event.key in ('s', 'down'):
                y += 1
            elif event.key in ('a', 'left'):
                x -= 1
            elif event.key in ('d', 'right'):
                x += 1
            elif event.key == 'r':
                r += 1
            elif event.key == 'f':
                r -= 1
            elif event.key == '1': # Normal image
                subtract_blank = False
                bitwise_not = False
                img_artist.set_data(image)

            elif event.key == '2':  # Difference image
                subtract_blank = not subtract_blank
                bitwise_not = False

                if subtract_blank:
                    diff_image = cv2.absdiff(image, blank_image)
                    img_artist.set_data(diff_image)
                else:
                    img_artist.set_data(image)

            elif event.key == '3': # Bitwise not image
                bitwise_not = not bitwise_not
                subtract_blank = False

                if bitwise_not:
                    bitwise_not_image = cv2.addWeighted(image, 0.5, bitwise_not_blank, 0.5, 0.0)
                    img_artist.set_data(bitwise_not_image)
                else:
                    img_artist.set_data(image)

        fig.canvas.mpl_connect('key_press_event', on_key)

        # Interactive update loop
        while not done:
            circle_artist.center = (x, y)
            circle_artist.set_radius(r)
            center_artist.set_data([x], [y])
            fig.canvas.draw_idle()
            plt.pause(0.01)

        plt.close(fig)
        plt.ioff()  # Turn off interactive mode
        fig.canvas.flush_events()  # Flush any pending events

        return x, y, r

def _adjust_fitting(dataset_path: Union[str, Path], anchor_idx, px_per_mm, anchor_data):
        """
        Scales the pixel-to-millimeter calibration using an interactive Matplotlib GUI.
        Args:
            dataset_path: Path to the dataset.
            csv_path: Path to the CSV file.
            initial_val: Initial pixel/mm ratio.
            anchor_idx: Index of the anchor image.
            circle_vals: Values of the anchor circle (x, y, r).
        Returns:
            px_per_mm: Pixel/millimeter ratio.
            calibration_data: Updated calibration dataframe.
        """

        # Load calibration data
        calibration_data_path = os.path.join(dataset_path, "annotations", "probe_data.csv")
        calibration_data = pd.read_csv(calibration_data_path)

        # Load anchor image
        anchor_image_path = os.path.join(dataset_path, "probe_images", calibration_data["img_name"][anchor_idx])
        anchor_image = cv2.cvtColor(cv2.imread(anchor_image_path), cv2.COLOR_BGR2RGB)
        anchor_x_mm = calibration_data["x_mm"][anchor_idx]
        anchor_y_mm = calibration_data["y_mm"][anchor_idx]
        anchor_x_px, anchor_y_px, anchor_r_px = anchor_data
        height, width, _ = anchor_image.shape

        # Add initial annotations (pixel coordinates)
        calibration_data['x_px'] = anchor_x_px + (calibration_data['x_mm'] - anchor_x_mm) * px_per_mm
        calibration_data['y_px'] = anchor_y_px + (anchor_y_mm - calibration_data['y_mm']) * px_per_mm

        # Load blank image
        blank_image_path = os.path.join(dataset_path, "blank_images", "blank.png")
        blank_image = cv2.cvtColor(cv2.imread(blank_image_path), cv2.COLOR_BGR2RGB)

        # Generate blank mosaic
        blank_mosaic = np.zeros((height * 3, width * 3, 3), dtype=np.uint8)

        for row in range(3):
            for col in range(3):
                blank_mosaic[(row * height):((row + 1) * height),
                             (col * width):((col + 1) * width), :] = blank_image

        # Create bitwise not mosaic
        bitwise_not_blank = cv2.bitwise_not(blank_mosaic)

        # Generate 3×3 mosaic
        image_list = [anchor_idx]
        mosaic = np.zeros((height * 3, width * 3, 3), dtype=np.uint8)
        mosaic[:height, :width, :] = anchor_image

        idx = 1
        while len(image_list) < 9:
            random_row = calibration_data.sample(n=1)

            # Make sure circles are within the camera's FOV
            if random_row["x_px"].values[0] > width * 0.15 and random_row["x_px"].values[0] < width * 0.85 and random_row["y_px"].values[0] > height * 0.15 and random_row["y_px"].values[0] < height * 0.85:
                image_path = os.path.join(dataset_path, "probe_images", random_row["img_name"].values[0])
                image = cv2.cvtColor(cv2.imread(image_path), cv2.COLOR_BGR2RGB)

                image_list.append(random_row.index[0])

                row = math.floor(idx / 3)
                col = idx % 3

                mosaic[(height * row):(height * (row + 1)),
                        (width * col):(width * (col + 1)), :] = image
                idx += 1

        # Flags for image display modes
        subtract_blank = False
        bitwise_not = False

        # Initialize Matplotlib figure
        plt.rcParams['keymap.save'] = []
        plt.rcParams['keymap.fullscreen'] = []

        fig, (ax_text, ax_img) = plt.subplots(1, 2, figsize=(14, 8), gridspec_kw={'width_ratios': [1, 3]})
        plt.subplots_adjust(wspace=0.4, bottom=0, top=1, left=0, right=1)
        fig.canvas.manager.set_window_title('Validate and Refine Calibration Annotations')

        # Right panel: image grid
        img_artist = ax_img.imshow(mosaic)
        ax_img.set_axis_off()

        ax_img.text(
            width * 0.19,
            height * 0.1,
            "Anchor Image",
            color='yellow',
            fontsize=13,
            bbox=dict(facecolor='black', alpha=0.1, boxstyle='round,pad=0.3')
        )

       
        # Overlay circles
        circle_artists = []
        for i in range(9):
            row = math.floor(i / 3)
            col = i % 3
            idx = image_list[i]
            x = int(calibration_data.loc[idx, 'x_px']) + col * width
            y = int(calibration_data.loc[idx, 'y_px']) + row * height
            circ = Circle((x, y), anchor_r_px, color='red', fill=False, lw=1)
            ax_img.add_patch(circ)
            circle_artists.append(circ)

        # Left panel: instructions
        ax_text.set_axis_off()
        ax_text.text(
            0.30, 0.75,
            f"Commands:\n\nw: Up\ns: Down\na: Left\nd: Right\nr: Increase pixel/mm value\nf: Decrease pixel/mm value\nq: Quit\n\n\n1: View 1 (RGB image)\n2: View 2 (Difference image)\n3: View 3 (Bitwise not image)",
            fontsize=20, color='black', va='top', ha='left', wrap=True
        )

        plt.ion()
        plt.show(block=False)

        done = False

        # Keyboard event handler
        def on_key(event):
            nonlocal anchor_x_px, anchor_y_px, anchor_r_px, px_per_mm, done, subtract_blank, bitwise_not, mosaic, blank_mosaic, bitwise_not_blank

            if event.key == 'q':
                done = True
            elif event.key in ('w', 'up'):
                anchor_y_px -= 1
            elif event.key in ('s', 'down'):
                anchor_y_px += 1
            elif event.key in ('a', 'left'):
                anchor_x_px -= 1
            elif event.key in ('d', 'right'):
                anchor_x_px += 1
            elif event.key == 'r':
                px_per_mm += 1
            elif event.key == 'f':
                px_per_mm -= 1
            elif event.key == '1':
                subtract_blank = False
                bitwise_not = False
                img_artist.set_data(mosaic)
            elif event.key == '2':
                subtract_blank = not subtract_blank
                bitwise_not = False
                
                if subtract_blank:
                    diff_mosaic = cv2.absdiff(mosaic, blank_mosaic)
                    img_artist.set_data(diff_mosaic)
                else:
                    img_artist.set_data(mosaic)

            elif event.key == '3':
                bitwise_not = not bitwise_not
                subtract_blank = False
                
                if bitwise_not:
                    bitwise_not_mosaic = cv2.addWeighted(mosaic, 0.5, bitwise_not_blank, 0.5, 0.0)
                    img_artist.set_data(bitwise_not_mosaic)
                else:
                    img_artist.set_data(mosaic)

            # Recompute coordinates
            calibration_data['x_px'] = anchor_x_px + (calibration_data['x_mm'] - anchor_x_mm) * px_per_mm
            calibration_data['y_px'] = anchor_y_px + (anchor_y_mm - calibration_data['y_mm']) * px_per_mm

            for i in range(9):
                row = math.floor(i / 3)
                col = i % 3
                idx = image_list[i]
                x = int(calibration_data.loc[idx, 'x_px']) + col * width
                y = int(calibration_data.loc[idx, 'y_px']) + row * height
                circle_artists[i].center = (x, y)

        fig.canvas.mpl_connect('key_press_event', on_key)

        # Main interactive loop
        while not done:
            fig.canvas.draw_idle()
            plt.pause(0.01)

        plt.close(fig)

        return px_per_mm, calibration_data

def validate_probe_radius(probe_radius_mm):
    """
    Validates the probe radius specified by the user.

    Args:
        probe_radius_mm: Probe radius specified by the user.
    Returns:
        None.
    Raises:
        ValueError: If the probe radius is not specified or invalid.
    """
    if probe_radius_mm is None:
         raise ValueError(
              "Probe radius cannot be None.\n"
         )    
    if not isinstance(probe_radius_mm, (int, float)) or probe_radius_mm <= 0:
       raise ValueError(
           "Probe radius must be a positive number (int or float).\n"
       )
    

def validate_indices(idxs):
    """
    Validates the image indices specified by the user.

    Args:
        idxs: Tuple of indices specified by the user.
    Returns:
        None.
    Raises:
        ValueError: If the indices are not specified or invalid.
    """
    if idxs is not None:
       if not (isinstance(idxs, (tuple, list)) and len(idxs) == 2 and all(isinstance(i, int) for i in idxs)):
           raise ValueError(
               "Image indices must be a tuple or list of two integers.\n"
           )