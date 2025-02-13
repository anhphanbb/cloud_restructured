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
nc_input_folder = '839'

csv_predictions_folder = 'ls_orbit_predictions'

nc_output_folder = 'nc_files_with_mlls'

# Ensure the output folder exists
os.makedirs(nc_output_folder, exist_ok=True)

# Define the box-to-index mapping
box_mapping = {
    f"({x},{y})": (x, y) for y in range(3) for x in range(15)
}

# Function to extract orbit number from filename
def extract_orbit_number(filename):
    return filename.split('_')[4]

# Function to create a new NetCDF file with the MLLS variable
def add_mlls_to_nc_file(input_file_path, output_file_path, mlls_data):
    with Dataset(input_file_path, 'r') as src_nc, Dataset(output_file_path, 'w', format=src_nc.file_format) as dst_nc:
        # Copy global attributes
        dst_nc.setncatts({attr: src_nc.getncattr(attr) for attr in src_nc.ncattrs()})
        
        # Copy dimensions
        for name, dimension in src_nc.dimensions.items():
            dst_nc.createDimension(name, (len(dimension) if not dimension.isunlimited() else None))
        
        # Add new dimensions for MLLS
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
        
        # Add the MLLS variable with default chunking and compression
        mlls_var = dst_nc.createVariable(
            'MLLS', 'f4', 
            ('time', 'y_box_across_track', 'x_box_along_track'), 
            zlib=True, 
            complevel=4
        )
        
        # Write MLLS data
        mlls_var[:] = mlls_data
        
        print(f"Created file with MLLS variable: {output_file_path}")


# Main script to process all files
for file_name in os.listdir(nc_input_folder):
    if file_name.endswith('.nc'):
        print(file_name)
        orbit_number = extract_orbit_number(file_name)
        print(orbit_number)
        if orbit_number:
            nc_file_path = os.path.join(nc_input_folder, file_name)
            csv_file_path = os.path.join(csv_predictions_folder, f"orbit_{orbit_number}_predictions.csv")
            if os.path.exists(csv_file_path):
                output_file_path = os.path.join(nc_output_folder, file_name.replace('l1r', 'l1l'))
                
                # Read CSV
                predictions_df = pd.read_csv(csv_file_path)
                predictions_df = predictions_df.sort_values(by=['Frame', 'Box'])
                
                # Open the NetCDF file to get the time dimension
                with Dataset(nc_file_path, 'r') as nc_file:
                    time_dim = len(nc_file.dimensions['time'])  # Extract the number of frames from NetCDF
                
                # Initialize MLLS array
                mlls_data = np.zeros((time_dim, 3, 15))  # Dimensions: (time, y_box_across_track, x_box_along_track)
                
                # Populate MLLS
                for _, row in predictions_df.iterrows():
                    frame = int(row['Frame'])
                    box = row['Box']
                    probability = row['Probability']
                    
                    if box in box_mapping:
                        x_idx, y_idx = box_mapping[box]
                        mlls_data[frame, y_idx, x_idx] = probability
                    else:
                        print(f"Box {box} not found in mapping.")
                        
                # Handle first and last four frames (Because of combining frames)
                if time_dim > 5:
                    mlls_data[:5, :, :] = mlls_data[5, :, :]  # Fill first 4 frames with the 5th frame
                    mlls_data[-5:, :, :] = mlls_data[-6, :, :]  # Fill last 4 frames with the 5th-to-last frame
                
                # Write to NetCDF
                add_mlls_to_nc_file(nc_file_path, output_file_path, mlls_data)
            else:
                print(f"CSV file for orbit {orbit_number} not found in {csv_predictions_folder}")

print("Processing completed.")
