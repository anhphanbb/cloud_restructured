# -*- coding: utf-8 -*-
"""
Created on Tue Feb 25 12:17:35 2025

@author: domin
"""

import os
from netCDF4 import Dataset
import cv2
import numpy as np
import re

# Folder containing .nc files
nc_folder = r'Z:\soc\l1r'
output_folder = 'images_to_predict'

# List of specific orbits to process
orbit_list = [135, 180, 225, 270, 315, 360, 405, 450, 495, 545, 590, 635, 815, 1130, 1175, 1265, 1310, 1355, 1400, 1445, 1490, 2525, 2570, 2615, 2660, 2705, 2750, 2795, 2840, 2885 ]
space = 5  # Number of frames before and after

# Ensure output folder exists
os.makedirs(output_folder, exist_ok=True)

# Function to search for .nc file
def find_nc_file(parent_directory, orbit_number):
    orbit_str = str(int(orbit_number)).zfill(5)
    pattern = re.compile(r'awe_l1r_q20_(.*)_' + orbit_str + r'_(.*)\.nc')
    
    for root, dirs, files in os.walk(parent_directory):
        for file in files:
            if pattern.match(file):
                return os.path.join(root, file)
    raise FileNotFoundError(f"No file found for orbit number {orbit_str}")

# Define new grid boxes
def define_boxes():
    x_ranges = [(0, 19), (20, 39), (40, 59), (60, 79), (80, 99), (100, 119),
                (120, 139), (140, 159), (160, 179), (180, 199), (200, 219),
                (220, 239), (240, 259), (260, 279), (280, 299)]
    y_ranges = [(0, 99), (100, 199), (200, 299)]
    
    boxes = {}
    for j, y_range in enumerate(y_ranges):
        for i, x_range in enumerate(x_ranges):
            box_id = f"({i},{j})"
            boxes[box_id] = {'x': x_range, 'y': y_range}
    return boxes

grid_boxes = define_boxes()
ignored_boxes = ["(0,0)", "(1,0)", "(2,0)", "(3,0)", "(4,0)", "(10,0)", "(11,0)", "(12,0)", "(13,0)", "(14,0)",
                 "(0,1)", "(1,1)", "(2,1)", "(3,1)", "(4,1)", "(10,1)", "(11,1)", "(12,1)", "(13,1)", "(14,1)", 
                 "(0,2)", "(1,2)", "(2,2)", "(3,2)", "(4,2)", "(10,2)", "(11,2)", "(12,2)", "(13,2)", "(14,2)"]

# Function to normalize radiance values
def normalize_radiance(frame, min_radiance=0, max_radiance=24):
    return np.clip((frame - min_radiance) / (max_radiance - min_radiance) * 255, 0, 255).astype(np.uint8)

# Function to create images from nc file
def create_images_from_nc_file(nc_file_path, grid_boxes, output_folder, space):
    orbit_number = re.search(r'_(\d{5})_', nc_file_path).group(1)
    orbit_output_folder = os.path.join(output_folder, f"orbit_{orbit_number}")
    os.makedirs(orbit_output_folder, exist_ok=True)

    with Dataset(nc_file_path, 'r') as nc:
        radiance = nc.variables['Radiance'][:]
        num_frames = radiance.shape[0]

        for i in range(space, num_frames - space):
            norm_radiance = normalize_radiance(radiance[i])
            prev_frame_norm = normalize_radiance(radiance[i - space])
            next_frame_norm = normalize_radiance(radiance[i + space])
            
            three_layer_image = np.zeros((radiance.shape[1], radiance.shape[2], 3), dtype=np.uint8)
            three_layer_image[..., 0] = prev_frame_norm
            three_layer_image[..., 1] = norm_radiance
            three_layer_image[..., 2] = next_frame_norm

            for box, coords in grid_boxes.items():
                if box not in ignored_boxes:
                    x_start, x_end = coords['x']
                    y_start, y_end = coords['y']
                    cropped_image = three_layer_image[y_start:y_end+1, x_start:x_end+1]
                    file_path = os.path.join(orbit_output_folder, f"frame_{i}_box_{box}.png")
                    cv2.imwrite(file_path, cropped_image)

# Process selected orbits
def process_selected_orbits(parent_directory, output_folder, grid_boxes, space, orbit_list):
    for orbit_number in orbit_list:
        try:
            nc_file_path = find_nc_file(parent_directory, orbit_number)
            print(f"Processing orbit {orbit_number} from file: {nc_file_path}")
            create_images_from_nc_file(nc_file_path, grid_boxes, output_folder, space)
        except FileNotFoundError as e:
            print(e)

# Run the script
process_selected_orbits(nc_folder, output_folder, grid_boxes, space, orbit_list)
