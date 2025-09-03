# -*- coding: utf-8 -*-
"""
Script to process MLCloud variable from NetCDF files.

Created on: Jan 8, 2025
@author: anhph
"""

import os
import numpy as np
from netCDF4 import Dataset

# Define input and output folders
nc_input_folder = r'D:\Github\labeled_nc_files_with_mlcloud_from_labels'
nc_output_folder = r'D:\Github\labeled_nc_files_with_processed_mlcloud_from_labels'

# Ensure the output folder exists
os.makedirs(nc_output_folder, exist_ok=True)

# Define the running average window size
window_size = 5

# Function to perform a running average
# Function to perform a running average
def running_average(data, window_size):
    cumsum = np.cumsum(np.insert(data, 0, 0), axis=0)
    smoothed = (cumsum[window_size:] - cumsum[:-window_size]) / float(window_size)
    
    # Create output array with same size as original
    result = np.zeros_like(data)
    
    # Fill the valid range with smoothed values
    result[window_size // 2: -(window_size // 2)] = smoothed
    
    # Fill the boundary frames
    result[:window_size // 2] = data[window_size // 2]  # Fill the first few with the nearest valid value
    result[-(window_size // 2):] = data[-(window_size // 2) - 1]  # Fill the last few with the nearest valid value
    
    return result

# Function to process MLCloud variable and save results
def process_mlcloud(input_file_path, output_file_path):
    with Dataset(input_file_path, 'r') as src_nc:
        # Read dimensions
        time_dim = len(src_nc.dimensions['time'])
        y_dim = len(src_nc.dimensions['y_box_across_track'])
        x_dim = len(src_nc.dimensions['x_box_along_track'])

        # Read MLCloud variable
        mlcloud_data = src_nc.variables['MLCloud'][:]

        # Apply running average to each (x, y) time series
        smoothed_data = np.copy(mlcloud_data)
        for y in range(y_dim):
            for x in range(x_dim):
                smoothed_data[:, y, x] = running_average(mlcloud_data[:, y, x], window_size)

        # Create processed data array
        processed_data = np.zeros_like(mlcloud_data)

        # Calculate averages as specified
        for t in range(time_dim):
            for y in range(y_dim):
                for x in range(x_dim):
                    values = []

                    # Add the current box value
                    if smoothed_data[t, y, x] != 0:
                        values.append(smoothed_data[t, y, x])

                    # Add neighboring box values only if non-zero
                    if x > 0 and t + 6 < time_dim and smoothed_data[t + 6, y, x - 1] != 0:
                        values.append(smoothed_data[t + 6, y, x - 1])
                    if x > 1 and t + 13 < time_dim and smoothed_data[t + 13, y, x - 2] != 0:
                        values.append(smoothed_data[t + 13, y, x - 2])
                    if x > 2 and t + 19 < time_dim and smoothed_data[t + 19, y, x - 3] != 0:
                        values.append(smoothed_data[t + 19, y, x - 3])
                    if x + 1 < x_dim and t - 6 >= 0 and smoothed_data[t - 6, y, x + 1] != 0:
                        values.append(smoothed_data[t - 6, y, x + 1])
                    if x + 2 < x_dim and t - 13 >= 0 and smoothed_data[t - 13, y, x + 2] != 0:
                        values.append(smoothed_data[t - 13, y, x + 2])
                    if x + 3 < x_dim and t - 19 >= 0 and smoothed_data[t - 19, y, x + 3] != 0:
                        values.append(smoothed_data[t - 19, y, x + 3])

                    # Calculate the average of the non-zero values
                    if values:
                        processed_data[t, y, x] = np.mean(values)
                    else:
                        processed_data[t, y, x] = 0  # Assign zero if no valid values exist

        # Write processed data to a new NetCDF file
        with Dataset(output_file_path, 'w', format=src_nc.file_format) as dst_nc:
            # Copy global attributes
            dst_nc.setncatts({attr: src_nc.getncattr(attr) for attr in src_nc.ncattrs()})

            # Copy dimensions
            for name, dimension in src_nc.dimensions.items():
                dst_nc.createDimension(name, (len(dimension) if not dimension.isunlimited() else None))

            # Copy variables except MLCloud
            for name, variable in src_nc.variables.items():
                if name != 'MLCloud':
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

            # Add the processed MLCloud variable
            mlcloud_var = dst_nc.createVariable(
                'Processed_MLCloud', 'f4', 
                ('time', 'y_box_across_track', 'x_box_along_track'), 
                zlib=True, 
                complevel=4
            )
            mlcloud_var[:] = processed_data

            print(f"Processed file saved: {output_file_path}")

# Main script to process all files
for file_name in os.listdir(nc_input_folder):
    if file_name.endswith('.nc'):
        input_file_path = os.path.join(nc_input_folder, file_name)
        output_file_path = os.path.join(nc_output_folder, file_name)
        process_mlcloud(input_file_path, output_file_path)

print("Processing completed.")
