# -*- coding: utf-8 -*-
"""
Created on Thu Feb 20 14:53:57 2025

@author: domin
"""

import os
import numpy as np
from netCDF4 import Dataset

# Define input and output folders
nc_input_folder = 'processed_nc_files'
nc_output_folder = r'E:\soc\l1c\2024\01'

# Ensure the output folder exists
os.makedirs(nc_output_folder, exist_ok=True)

# Function to rename Processed_MLLS to Processed_MLCloud and update file names
def rename_variable_and_save(input_file_path, output_file_path):
    with Dataset(input_file_path, 'r') as src_nc:
        # Read Processed_MLLS variable
        if 'Processed_MLLS' in src_nc.variables:
            processed_mlls_data = src_nc.variables['Processed_MLLS'][:]
        else:
            print(f"Processed_MLLS not found in {input_file_path}")
            return

        # Write updated NetCDF file
        with Dataset(output_file_path, 'w', format=src_nc.file_format) as dst_nc:
            # Copy global attributes
            dst_nc.setncatts({attr: src_nc.getncattr(attr) for attr in src_nc.ncattrs()})

            # Copy dimensions
            for name, dimension in src_nc.dimensions.items():
                dst_nc.createDimension(name, (len(dimension) if not dimension.isunlimited() else None))

            # Copy variables except Processed_MLLS
            for name, variable in src_nc.variables.items():
                if name != 'Processed_MLLS':
                    # Check if the variable needs custom chunk sizes
                    if name in ['Radiance', 'Latitude', 'Longitude']:
                        chunksizes = (1, 300, 300)
                    else:
                        chunksizes = variable.chunking()  # Use default chunk sizes from source file

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

            # Add the renamed Processed_MLCloud variable
            mlcloud_var = dst_nc.createVariable(
                'Processed_MLCloud', 'f4', 
                ('time', 'y_box_across_track', 'x_box_along_track'), 
                zlib=True, 
                complevel=4
            )
            mlcloud_var[:] = processed_mlls_data

            print(f"Updated file saved: {output_file_path}")

# Process all files
for file_name in os.listdir(nc_input_folder):
    if file_name.endswith('.nc'):
        new_file_name = file_name.replace('l1l', 'l1c')
        input_file_path = os.path.join(nc_input_folder, file_name)
        output_file_path = os.path.join(nc_output_folder, new_file_name)
        rename_variable_and_save(input_file_path, output_file_path)

print("Processing completed.")
