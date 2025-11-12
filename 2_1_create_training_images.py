# -*- coding: utf-8 -*-
"""
Created on Mon Dec  2 18:51:37 2024

@author: Anh

Update: Reads runtime parameters from a config text file.

Example config.txt (commas or one-per-line both work):
    space = 5, min_radiance = 0, max_radiance = 24
    cloud_chance = 0.5, no_cloud_chance = 0.025
    threshold = 3
"""

import os
import re
import cv2
import json
import shutil
import random
import numpy as np
import pandas as pd
from netCDF4 import Dataset

# ----------------------- CONFIG -----------------------
CONFIG_PATH = 'config.txt'  # change if you want

# Defaults if a key is missing in config.txt
DEFAULTS = {
    'space': 5,
    'min_radiance': 0.0,
    'max_radiance': 24.0,
    'cloud_chance': 0.5,
    'no_cloud_chance': 0.025,
    'threshold': 3,
}

def parse_config(config_path):
    """
    Parse key=value pairs from a text file.
    - Accepts multiple pairs per line separated by commas.
    - Ignores comments starting with '#'.
    - Keys supported: space, min_radiance, max_radiance, cloud_chance, no_cloud_chance, threshold
    """
    cfg = DEFAULTS.copy()
    if not os.path.isfile(config_path):
        print(f"[warn] Config file not found: {config_path}. Using defaults: {cfg}")
        return cfg

    with open(config_path, 'r', encoding='utf-8') as f:
        text = f.read()

    # Remove comments
    text = re.sub(r'#.*', '', text)

    # Find all key=value pairs, allowing commas in between pairs
    # e.g., "space = 5, min_radiance = 0, max_radiance = 24"
    pairs = re.findall(r'([A-Za-z_][A-Za-z0-9_]*)\s*=\s*([^,\n\r]+)', text)

    for key, val_raw in pairs:
        key = key.strip()
        val = val_raw.strip()
        try:
            if key in ('space', 'threshold'):
                cfg[key] = int(float(val))
            elif key in ('min_radiance', 'max_radiance', 'cloud_chance', 'no_cloud_chance'):
                cfg[key] = float(val)
        except ValueError:
            print(f"[warn] Could not parse '{key}={val}'. Keeping default {key}={cfg[key]}")

    # Sanity checks
    if cfg['max_radiance'] <= cfg['min_radiance']:
        print("[warn] max_radiance <= min_radiance. Resetting to defaults 24.0 and 0.0")
        cfg['min_radiance'] = DEFAULTS['min_radiance']
        cfg['max_radiance'] = DEFAULTS['max_radiance']
    if not (0.0 <= cfg['cloud_chance'] <= 1.0):
        print("[warn] cloud_chance out of [0,1]. Resetting to default.")
        cfg['cloud_chance'] = DEFAULTS['cloud_chance']
    if not (0.0 <= cfg['no_cloud_chance'] <= 1.0):
        print("[warn] no_cloud_chance out of [0,1]. Resetting to default.")
        cfg['no_cloud_chance'] = DEFAULTS['no_cloud_chance']

    print("[info] Loaded config:", json.dumps(cfg, indent=2))
    return cfg

cfg = parse_config(CONFIG_PATH)
space = cfg['space']
min_radiance = cfg['min_radiance']
max_radiance = cfg['max_radiance']
cloud_chance = cfg['cloud_chance']
no_cloud_chance = cfg['no_cloud_chance']
threshold = cfg['threshold']

# ----------------------- PATHS ------------------------
# Path to the CSV file with filenames and intervals
csv_file_path = 'csv/cloud_intervals_cleaned_filtered_10272025.csv'
# parent_directory = r'Z:\soc\l1r'
parent_directory = 'training_nc_files'

# Define output folders
cloud_folder = 'training_images/class_1_cloud'
no_cloud_folder = 'training_images/class_0_no_cloud'

# Ensure output folders exist
os.makedirs(no_cloud_folder, exist_ok=True)
os.makedirs(cloud_folder, exist_ok=True)

# ----------------------- HELPERS ----------------------
def clear_images(folder_path):
    """Delete all files and subfolders under folder_path."""
    for filename in os.listdir(folder_path):
        file_path = os.path.join(folder_path, filename)
        try:
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.unlink(file_path)
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)
        except Exception as e:
            print(f'Failed to delete {file_path}. Reason: {e}')

# Clear both folders before saving new images
clear_images(cloud_folder)
clear_images(no_cloud_folder)

def define_boxes():
    x_ranges = [
        (0, 19), (20, 39), (40, 59), (60, 79), (80, 99), (100, 119),
        (120, 139), (140, 159), (160, 179), (180, 199), (200, 219),
        (220, 239), (240, 259), (260, 279), (280, 299)
    ]
    y_ranges = [(0, 99), (100, 199), (200, 299)]

    boxes = {}
    for j, y_range in enumerate(y_ranges):
        for i, x_range in enumerate(x_ranges):
            box_id = f"({i},{j})"
            boxes[box_id] = {'x': x_range, 'y': y_range}
    return boxes

grid_boxes = define_boxes()

def extract_intervals_per_orbit(dataframe):
    orbit_intervals = {}
    for _, row in dataframe.iterrows():
        orbit = row.get('Orbit #', None)
        if pd.notna(orbit):
            orbit = int(orbit)
            if orbit not in orbit_intervals:
                orbit_intervals[orbit] = {}
            for col in dataframe.columns:
                if "start" in col:
                    box = col.split("start")[0].strip()
                    end_col = f"{box}end"
                    if end_col in dataframe.columns:
                        start = row[col]
                        end = row[end_col]
                        if pd.notna(start) and pd.notna(end):
                            if box not in orbit_intervals[orbit]:
                                orbit_intervals[orbit][box] = []
                            orbit_intervals[orbit][box].append((int(start), int(end)))
    return orbit_intervals

def find_nc_file(parent_dir, orbit_number):
    orbit_str = str(int(orbit_number)).zfill(5)
    pattern = re.compile(r'awe_l1r_q20_(.*)_' + orbit_str + r'_(.*)\.nc')
    for root, _, files in os.walk(parent_dir):
        for file in files:
            if pattern.match(file):
                return os.path.join(root, file)
    raise FileNotFoundError(f"No file found for orbit number {orbit_str}")

def save_image(radiance_3d, folder, orbit_number, frame_index, box_idx, boxes,
               space_val, min_rad, max_rad):
    # Normalize frames to 0..255
    norm = np.clip((radiance_3d[frame_index] - min_rad) / (max_rad - min_rad) * 255, 0, 255).astype(np.uint8)
    prev_norm = None
    next_norm = None

    if frame_index >= space_val:
        prev = radiance_3d[frame_index - space_val]
        prev_norm = np.clip((prev - min_rad) / (max_rad - min_rad) * 255, 0, 255).astype(np.uint8)
    if frame_index < radiance_3d.shape[0] - space_val:
        nxt = radiance_3d[frame_index + space_val]
        next_norm = np.clip((nxt - min_rad) / (max_rad - min_rad) * 255, 0, 255).astype(np.uint8)

    x_start, x_end = boxes[box_idx]['x']
    y_start, y_end = boxes[box_idx]['y']

    three = np.zeros((radiance_3d.shape[1], radiance_3d.shape[2], 3), dtype=np.uint8)
    if prev_norm is not None:
        three[..., 0] = prev_norm
    three[..., 1] = norm
    if next_norm is not None:
        three[..., 2] = next_norm

    cropped = three[y_start:y_end+1, x_start:x_end+1]
    outpath = os.path.join(folder, f"orbit{orbit_number}_box{box_idx}_{frame_index}.png")
    cv2.imwrite(outpath, cropped)

def process_intervals_and_save_images(df, boxes, space_val, thresh,
                                      cloud_p, no_cloud_p,
                                      min_rad, max_rad):
    orbit_intervals = extract_intervals_per_orbit(df)
    for orbit_number, box_map in orbit_intervals.items():
        print(f"Processing orbit: {orbit_number}")
        try:
            nc_file_path = find_nc_file(parent_directory, orbit_number)
        except FileNotFoundError as e:
            print(e)
            continue

        with Dataset(nc_file_path, 'r') as nc:
            radiance = nc.variables['Radiance'][:]
            num_frames = radiance.shape[0]

            for box, intervals in box_map.items():
                print(f"  Box {box} intervals: {intervals}")
                for i in range(space_val, num_frames - space_val):
                    # Inside any interval with safety threshold
                    inside_any = any((start + thresh) <= i <= (end - thresh) for (start, end) in intervals)
                    # Outside all intervals with safety threshold
                    outside_all = all(i < (start - thresh) or i > (end + thresh) for (start, end) in intervals)

                    if inside_any:
                        if random.random() < cloud_p:
                            save_image(radiance, cloud_folder, orbit_number, i, box, boxes,
                                       space_val, min_rad, max_rad)
                    elif outside_all:
                        if random.random() < no_cloud_p:
                            save_image(radiance, no_cloud_folder, orbit_number, i, box, boxes,
                                       space_val, min_rad, max_rad)

# ----------------------- RUN -------------------------
data = pd.read_csv(csv_file_path)
process_intervals_and_save_images(
    data,
    grid_boxes,
    space_val=space,
    thresh=threshold,
    cloud_p=cloud_chance,
    no_cloud_p=no_cloud_chance,
    min_rad=min_radiance,
    max_rad=max_radiance
)
