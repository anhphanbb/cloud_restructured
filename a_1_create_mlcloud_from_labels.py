# -*- coding: utf-8 -*-
"""
Created on Tue Dec  3 18:44:44 2024

@author: anhph
"""

import os
import numpy as np
import pandas as pd
from netCDF4 import Dataset
import re

# Define input and output folders
nc_input_folder = 'selected_nc_files'
csv_intervals_path = 'csv/cloud_intervals_selected_orbits_april_14_2025.csv'
nc_output_folder = 'selected_nc_files_with_mlcloud_from_labels'

# Ensure the output folder exists
os.makedirs(nc_output_folder, exist_ok=True)

# Define the box-to-index mapping
box_mapping = {
    f"({x},{y})": (x, y) for y in range(3) for x in range(15)
}

# Function to extract orbit number from filename
def extract_orbit_number(filename):
    return filename.split('_')[4]

# Function to extract intervals from the CSV file
def extract_intervals_per_orbit(data):
    orbit_intervals = {}
    for _, row in data.iterrows():
        orbit = row['Orbit #']
        if pd.notna(orbit):
            orbit = int(orbit)
            if orbit not in orbit_intervals:
                orbit_intervals[orbit] = {}
            for col in data.columns:
                if "start" in col:
                    box = col.split("start")[0].strip()
                    end_col = f"{box}end"
                    if end_col in data.columns:
                        start = row[col]
                        end = row[end_col]
                        if pd.notna(start) and pd.notna(end):
                            if box not in orbit_intervals[orbit]:
                                orbit_intervals[orbit][box] = []
                            orbit_intervals[orbit][box].append((int(start), int(end)))
    return orbit_intervals

# Function to create a new NetCDF file with the MLCloud variable
def add_mlcloud_to_nc_file(input_file_path, output_file_path, mlcloud_data):
    with Dataset(input_file_path, 'r') as src_nc, Dataset(output_file_path, 'w', format=src_nc.file_format) as dst_nc:
        # Copy global attributes
        dst_nc.setncatts({attr: src_nc.getncattr(attr) for attr in src_nc.ncattrs()})
        
        # Copy dimensions
        for name, dimension in src_nc.dimensions.items():
            dst_nc.createDimension(name, (len(dimension) if not dimension.isunlimited() else None))
        
        # Add new dimensions for MLCloud
        dst_nc.createDimension('y_box_across_track', 3)
        dst_nc.createDimension('x_box_along_track', 15)
        
        # Copy variables
        for name, variable in src_nc.variables.items():
            new_var = dst_nc.createVariable(
                name, 
                variable.datatype, 
                variable.dimensions, 
                zlib=True, 
                complevel=4
            )
            new_var[:] = variable[:]
            new_var.setncatts({attr: variable.getncattr(attr) for attr in variable.ncattrs()})
        
        # Add the MLCloud variable
        mlcloud_var = dst_nc.createVariable(
            'MLCloud', 'f4', 
            ('time', 'y_box_across_track', 'x_box_along_track'), 
            zlib=True, 
            complevel=4
        )
        mlcloud_var[:] = mlcloud_data
        print(f"Created file with MLCloud variable: {output_file_path}")

# Main script to process all files
data = pd.read_csv(csv_intervals_path)
orbit_intervals = extract_intervals_per_orbit(data)

for file_name in os.listdir(nc_input_folder):
    if file_name.endswith('.nc'):
        orbit_number = extract_orbit_number(file_name)
        print(f"Processing orbit: {orbit_number}")
        if orbit_number and int(orbit_number) in orbit_intervals:
            nc_file_path = os.path.join(nc_input_folder, file_name)
            output_file_path = os.path.join(nc_output_folder, file_name.replace('l1r', 'l1c'))
            
            # Open NetCDF file to get the time dimension
            with Dataset(nc_file_path, 'r') as nc_file:
                time_dim = len(nc_file.dimensions['time'])  # Extract the number of frames
            
            # Initialize MLCloud array
            mlcloud_data = np.zeros((time_dim, 3, 15))  # Dimensions: (time, y_box_across_track, x_box_along_track)
            
            # Populate MLCloud
            for box, intervals in orbit_intervals[int(orbit_number)].items():
                if box in box_mapping:
                    x_idx, y_idx = box_mapping[box]
                    for start, end in intervals:
                        mlcloud_data[start:end + 1, y_idx, x_idx] = 1  # Set MLCloud value to 1 for frames inside intervals
            
            # Handle edge frames
            if time_dim > 5:
                mlcloud_data[:5, :, :] = mlcloud_data[5, :, :]  # Fill first 4 frames with the 5th frame
                mlcloud_data[-5:, :, :] = mlcloud_data[-6, :, :]  # Fill last 4 frames with the 5th-to-last frame
            
            # Write to NetCDF
            add_mlcloud_to_nc_file(nc_file_path, output_file_path, mlcloud_data)
        else:
            print(f"No intervals found for orbit {orbit_number}")

print("Processing completed.")
