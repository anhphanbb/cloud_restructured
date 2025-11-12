# -*- coding: utf-8 -*-
"""
Modified on Wed Dec  4 2024

@author: Anh

Script to create prediction images using consecutive frames from all .nc files in a folder,
with multiprocessing.

Now reads 'space', 'min_radiance', and 'max_radiance' from config.txt.
"""

import os
from netCDF4 import Dataset
import cv2
import numpy as np
import re
from multiprocessing import Pool, cpu_count

# ---------------------- Config handling ---------------------- #

def load_config(config_path):
    """
    Load key = value pairs from a simple config.txt file.

    Expected lines like:
        space = 8,
        min_radiance = 0,
        max_radiance = 20,
        cloud_chance = 0.5,
        no_cloud_chance = 0.1,
        threshold = 3

    Trailing commas and comments are allowed.
    """
    cfg = {}

    if not os.path.exists(config_path):
        print(f"[WARN] config.txt not found at {config_path}. Using defaults.")
        return cfg

    with open(config_path, "r") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            if "=" not in line:
                continue

            key, value = line.split("=", 1)
            key = key.strip()

            # Remove comments after "#"
            value = value.split("#", 1)[0].strip()

            # Remove trailing comma, if any
            if value.endswith(","):
                value = value[:-1].strip()

            # Try to interpret as int or float; fallback to string
            try:
                if "." in value:
                    num = float(value)
                    # cast to int if it's like "8.0"
                    if num.is_integer():
                        num = int(num)
                    cfg[key] = num
                else:
                    cfg[key] = int(value)
            except ValueError:
                # just keep as string
                cfg[key] = value

    return cfg

# Path to config.txt: same folder as this script
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
CONFIG_PATH = os.path.join(SCRIPT_DIR, "config.txt")
CONFIG = load_config(CONFIG_PATH)

# Defaults if not provided in config.txt
DEFAULT_SPACE = 5
DEFAULT_MIN_RAD = 0
DEFAULT_MAX_RAD = 24

space = CONFIG.get("space", DEFAULT_SPACE)
MIN_RADIANCE = CONFIG.get("min_radiance", DEFAULT_MIN_RAD)
MAX_RADIANCE = CONFIG.get("max_radiance", DEFAULT_MAX_RAD)

print(f"[INFO] Using space = {space}, min_radiance = {MIN_RADIANCE}, max_radiance = {MAX_RADIANCE}")

# ---------------------- User paths ---------------------- #

nc_folder = r'E:\soc\l1r\2025\09'
output_folder = r'E:\soc\l1r\2025\09\images_to_predict'
min_orbit_number = 1

os.makedirs(output_folder, exist_ok=True)

# ---------------------- Grid boxes ---------------------- #

def define_boxes():
    x_ranges = [
        (0, 19), (20, 39), (40, 59), (60, 79), (80, 99), (100, 119),
        (120, 139), (140, 159), (160, 179), (180, 199), (200, 219),
        (220, 239), (240, 259), (260, 279), (280, 299)
    ]
    y_ranges = [
        (0, 99), (100, 199), (200, 299)
    ]
    boxes = {}
    for j, y_range in enumerate(y_ranges):
        for i, x_range in enumerate(x_ranges):
            box_id = f"({i},{j})"
            boxes[box_id] = {'x': x_range, 'y': y_range}
    return boxes

ignored_boxes = [
    "(0,0)", "(1,0)", "(2,0)", "(3,0)", "(4,0)", "(10,0)", "(11,0)", "(12,0)", "(13,0)", "(14,0)",
    "(0,1)", "(1,1)", "(2,1)", "(3,1)", "(4,1)", "(10,1)", "(11,1)", "(12,1)", "(13,1)", "(14,1)",
    "(0,2)", "(1,2)", "(2,2)", "(3,2)", "(4,2)", "(10,2)", "(11,2)", "(12,2)", "(13,2)", "(14,2)"
]

grid_boxes = define_boxes()

# ---------------------- Helper functions ---------------------- #

def normalize_radiance(frame, min_radiance=MIN_RADIANCE, max_radiance=MAX_RADIANCE):
    return np.clip(
        (frame - min_radiance) / (max_radiance - min_radiance) * 255,
        0, 255
    ).astype(np.uint8)

def create_images_from_nc_file(nc_file_path):
    orbit_match = re.search(r'_(\d{5})_', nc_file_path)
    if orbit_match:
        orbit_number = orbit_match.group(1)
    else:
        print(f"Could not determine orbit number from file: {nc_file_path}")
        return

    if int(orbit_number) < min_orbit_number:
        print(f"Skipping orbit {orbit_number} (below threshold of {min_orbit_number})")
        return

    orbit_output_folder = os.path.join(output_folder, f"orbit_{orbit_number}")
    os.makedirs(orbit_output_folder, exist_ok=True)

    try:
        with Dataset(nc_file_path, 'r') as nc:
            radiance = nc.variables['Radiance'][:]
            num_frames = radiance.shape[0]

            for i in range(space, num_frames - space):
                norm_radiance = normalize_radiance(radiance[i])
                prev_frame_norm = normalize_radiance(radiance[i - space])
                next_frame_norm = normalize_radiance(radiance[i + space])

                three_layer_image = np.zeros(
                    (radiance.shape[1], radiance.shape[2], 3),
                    dtype=np.uint8
                )
                three_layer_image[..., 0] = prev_frame_norm
                three_layer_image[..., 1] = norm_radiance
                three_layer_image[..., 2] = next_frame_norm

                for box, coords in grid_boxes.items():
                    if box not in ignored_boxes:
                        x_start, x_end = coords['x']
                        y_start, y_end = coords['y']
                        cropped_image = three_layer_image[y_start:y_end+1, x_start:x_end+1]
                        file_path = os.path.join(
                            orbit_output_folder,
                            f"frame_{i}_box_{box}.png"
                        )
                        cv2.imwrite(file_path, cropped_image)
        print(f"Finished processing {nc_file_path}")
    except Exception as e:
        print(f"Error processing {nc_file_path}: {e}")

def process_all_nc_files():
    nc_files = []
    for root, _, files in os.walk(nc_folder):
        for file in files:
            if file.endswith('.nc') and 'q20' in file:
                nc_files.append(os.path.join(root, file))

    print(f"Found {len(nc_files)} .nc files to process.")
    print(f"Using space={space}, min_radiance={MIN_RADIANCE}, max_radiance={MAX_RADIANCE}")

    with Pool(processes=max(cpu_count() - 1, 1)) as pool:
        pool.map(create_images_from_nc_file, nc_files)

if __name__ == "__main__":
    process_all_nc_files()
