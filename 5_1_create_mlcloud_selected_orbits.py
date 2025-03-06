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
# nc_input_folder = 'nc_files_to_predict' 
nc_input_folder = r'Z:\soc\l1r'
  
csv_predictions_folder = 'orbit_predictions' 
# csv_predictions_folder = r'E:\soc\l1r\2024\08\orbit_predictions' 

nc_output_folder = 'nc_files_with_mlcloud' 
# nc_output_folder = r'E:\soc\l1r\2024\08\nc_files_with_mlcloud' 

# List of specific orbits to process
orbit_list = [135, 180, 225, 270, 315, 360, 405, 450, 495, 545, 590, 635, 815, 1130, 1175, 1265, 1310, 1355, 1400, 1445, 1490, 2525, 2570, 2615, 2660, 2705, 2750, 2795, 2840, 2885]

# Ensure the output folder exists
os.makedirs(nc_output_folder, exist_ok=True)

# Define the box-to-index mapping
box_mapping = {
    f"({x},{y})": (x, y) for y in range(3) for x in range(15)
}

# Function to extract orbit number from filename
def extract_orbit_number(filename):
    return filename.split('_')[4]

# Function to search for .nc file
def find_nc_file(parent_directory, orbit_number):
    orbit_str = str(int(orbit_number)).zfill(5)
    pattern = re.compile(r'awe_l1r_q20_(.*)_' + orbit_str + r'_(.*)\.nc')
    
    for root, dirs, files in os.walk(parent_directory):
        for file in files:
            if pattern.match(file):
                return os.path.join(root, file)
    raise FileNotFoundError(f"No file found for orbit number {orbit_str}")

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
        
        # Copy variables with compression and chunking
        for name, variable in src_nc.variables.items():
            # Check if the variable needs custom chunk sizes
            if name in ['Radiance', 'Latitude', 'Longitude']:
                chunksizes = (1, 300, 300)
            else:
                chunksizes = variable.chunking()  # Use default chunk sizes from the source file if available

            # Create variable with compression and chunking
            new_var = dst_nc.createVariable(
                name, 
                variable.datatype, 
                variable.dimensions, 
                zlib=True, 
                complevel=4, 
                chunksizes=chunksizes
            )
            new_var[:] = variable[:]
            new_var.setncatts({attr: variable.getncattr(attr) for attr in variable.ncattrs()})
        
        # Add the MLCloud variable with default chunking and compression
        mlcloud_var = dst_nc.createVariable(
            'MLCloud', 'f4', 
            ('time', 'y_box_across_track', 'x_box_along_track'), 
            zlib=True, 
            complevel=4
        )
        
        # Write MLCloud data
        mlcloud_var[:] = mlcloud_data
        
        print(f"Created file with MLCloud variable: {output_file_path}")


# Main script to process all files
for orbit_number in orbit_list:
    print(orbit_number)
    try:
        nc_file_path = find_nc_file(nc_input_folder, orbit_number)
        orbit_str = str(int(orbit_number)).zfill(5)
        print(f"Processing orbit {orbit_number} from file: {nc_file_path}")
        
        csv_file_path = os.path.join(csv_predictions_folder, f"orbit_{orbit_str}_predictions.csv")
        if os.path.exists(csv_file_path):
            output_file_path = nc_file_path.replace('l1r', 'l1c')
            
            # Read CSV
            predictions_df = pd.read_csv(csv_file_path)
            predictions_df = predictions_df.sort_values(by=['Frame', 'Box'])
            
            # Open the NetCDF file to get the time dimension
            with Dataset(nc_file_path, 'r') as nc_file:
                time_dim = len(nc_file.dimensions['time'])  # Extract the number of frames from NetCDF
            
            # Initialize MLCloud array
            mlcloud_data = np.zeros((time_dim, 3, 15))  # Dimensions: (time, y_box_across_track, x_box_along_track)
            
            # Populate MLCloud
            for _, row in predictions_df.iterrows():
                frame = int(row['Frame'])
                box = row['Box']
                probability = row['Probability']
                
                if box in box_mapping:
                    x_idx, y_idx = box_mapping[box]
                    mlcloud_data[frame, y_idx, x_idx] = probability
                else:
                    print(f"Box {box} not found in mapping.")
                    
            # Handle first and last four frames (Because of combining frames)
            if time_dim > 5:
                mlcloud_data[:5, :, :] = mlcloud_data[5, :, :]  # Fill first 4 frames with the 5th frame
                mlcloud_data[-5:, :, :] = mlcloud_data[-6, :, :]  # Fill last 4 frames with the 5th-to-last frame
            
            # Write to NetCDF
            add_mlcloud_to_nc_file(nc_file_path, output_file_path, mlcloud_data)
        else:
            print(f"CSV file for orbit {orbit_number} not found in {csv_predictions_folder}")   
        
        # create_images_from_nc_file(nc_file_path, grid_boxes, output_folder, space)
    except FileNotFoundError as e:
        print(e)


print("Processing completed.")
