# -*- coding: utf-8 -*-
"""
Script to process MLCloud variable from NetCDF files using multiprocessing.

Modified on: May 5, 2025
@author: anhph
"""

import os
import numpy as np
from netCDF4 import Dataset
from concurrent.futures import ProcessPoolExecutor, as_completed

# Define input and output folders
original_nc_folder = r'E:\soc\l1r\2025\03'  # Original nc files with full variables
nc_input_folder = r'E:\soc\l1r\2025\03\nc_files_with_mlcloud'  # Only contains MLCloud
nc_output_folder = r'E:\soc\l1c\2025\03'  # Final output

# Ensure the output folder exists
os.makedirs(nc_output_folder, exist_ok=True)

# Define the running average window size
window_size = 5

def running_average(data, window_size):
    cumsum = np.cumsum(np.insert(data, 0, 0), axis=0)
    smoothed = (cumsum[window_size:] - cumsum[:-window_size]) / float(window_size)
    result = np.zeros_like(data)
    result[window_size // 2: -(window_size // 2)] = smoothed
    result[:window_size // 2] = data[window_size // 2]
    result[-(window_size // 2):] = data[-(window_size // 2) - 1]
    return result

def process_mlcloud(file_name):
    original_nc_path = os.path.join(original_nc_folder, file_name.replace('l1c', 'l1r'))
    mlcloud_nc_path = os.path.join(nc_input_folder, file_name)
    output_nc_path = os.path.join(nc_output_folder, file_name)
    
    with Dataset(original_nc_path, 'r') as orig_nc, Dataset(mlcloud_nc_path, 'r') as ml_nc:
        mlcloud_data = ml_nc.variables['MLCloud'][:]
        time_dim, y_dim, x_dim = mlcloud_data.shape

        smoothed_data = np.copy(mlcloud_data)
        for y in range(y_dim):
            for x in range(x_dim):
                smoothed_data[:, y, x] = running_average(mlcloud_data[:, y, x], window_size)

        processed_data = np.zeros_like(mlcloud_data)
        for t in range(time_dim):
            for y in range(y_dim):
                for x in range(x_dim):
                    values = []
                    if smoothed_data[t, y, x] != 0:
                        values.append(smoothed_data[t, y, x])
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
                    processed_data[t, y, x] = np.mean(values) if values else 0

        with Dataset(output_nc_path, 'w', format=orig_nc.file_format) as dst_nc:
            dst_nc.setncatts({attr: orig_nc.getncattr(attr) for attr in orig_nc.ncattrs()})
            for name, dimension in orig_nc.dimensions.items():
                dst_nc.createDimension(name, (len(dimension) if not dimension.isunlimited() else None))
            if 'y_box_across_track' not in dst_nc.dimensions:
                dst_nc.createDimension('y_box_across_track', y_dim)
            if 'x_box_along_track' not in dst_nc.dimensions:
                dst_nc.createDimension('x_box_along_track', x_dim)
            for name, variable in orig_nc.variables.items():
                if name != 'MLCloud':
                    chunksizes = (1, 300, 300) if name in ['Radiance', 'Latitude', 'Longitude'] else variable.chunking()
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
            mlcloud_var = dst_nc.createVariable(
                'Processed_MLCloud', 'f4',
                ('time', 'y_box_across_track', 'x_box_along_track'),
                zlib=True,
                complevel=4
            )
            mlcloud_var[:] = processed_data
            print(f"Processed file saved: {output_nc_path}")

if __name__ == "__main__":
    nc_files = [f for f in os.listdir(nc_input_folder) if f.endswith('.nc')]

    with ProcessPoolExecutor() as executor:
        futures = {executor.submit(process_mlcloud, f): f for f in nc_files}
        for future in as_completed(futures):
            file = futures[future]
            try:
                future.result()
            except Exception as e:
                print(f"Error processing {file}: {e}")

    print("Processing completed.")
