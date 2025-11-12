# -*- coding: utf-8 -*-
"""
Created on Tue Dec  3 18:44:44 2024

@author: anh

Now modified to read 'space' from config.txt.
"""

import os
import numpy as np
import pandas as pd
from netCDF4 import Dataset

# --------------------------
# Load config.txt
# --------------------------

def load_config(config_path):
    cfg = {}
    if not os.path.exists(config_path):
        print(f"[WARN] config.txt not found at {config_path}. Using defaults.")
        return cfg

    with open(config_path, "r") as f:
        for line in f:
            line = line.strip()
            if not line or "=" not in line:
                continue

            key, value = line.split("=", 1)
            key = key.strip()

            # remove trailing comma
            value = value.split("#", 1)[0].strip()
            if value.endswith(","):
                value = value[:-1].strip()

            # convert to float or int if applicable
            try:
                if "." in value:
                    num = float(value)
                    if num.is_integer():
                        num = int(num)
                    cfg[key] = num
                else:
                    cfg[key] = int(value)
            except:
                cfg[key] = value

    return cfg


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
CONFIG = load_config(os.path.join(SCRIPT_DIR, "config.txt"))

# Default space = 5 if not found in config
space = CONFIG.get("space", 5)

print(f"[INFO] Loaded space = {space} from config.txt")


# --------------------------
# Input/output folders
# --------------------------

nc_input_folder = r'E:\soc\l1r\2025\09'
csv_predictions_folder = r'E:\soc\l1r\2025\09\orbit_predictions'
nc_output_folder = r'E:\soc\l1r\2025\09\nc_files_with_mlcloud'

os.makedirs(nc_output_folder, exist_ok=True)


# --------------------------
# Box mapping
# --------------------------

box_mapping = {
    f"({x},{y})": (x, y) for y in range(3) for x in range(15)
}


def extract_orbit_number(filename):
    return filename.split('_')[4]


# --------------------------
# Write MLCloud to NetCDF
# --------------------------

def add_mlcloud_to_nc_file(input_file_path, output_file_path, mlcloud_data):
    with Dataset(input_file_path, 'r') as src_nc, Dataset(output_file_path, 'w', format=src_nc.file_format) as dst_nc:

        # Copy global attributes
        dst_nc.setncatts({attr: src_nc.getncattr(attr) for attr in src_nc.ncattrs()})

        # Copy dimensions
        dst_nc.createDimension('time', len(src_nc.dimensions['time']))
        dst_nc.createDimension('y_box_across_track', 3)
        dst_nc.createDimension('x_box_along_track', 15)

        # Create MLCloud variable
        mlcloud_var = dst_nc.createVariable(
            'MLCloud', 'f4',
            ('time', 'y_box_across_track', 'x_box_along_track'),
            zlib=True,
            complevel=4
        )
        mlcloud_var[:] = mlcloud_data

        print(f"Created file with ONLY MLCloud variable: {output_file_path}")


# --------------------------
# Main processing loop
# --------------------------

for file_name in os.listdir(nc_input_folder):
    if file_name.endswith('.nc') and 'q20' in file_name:
        print(file_name)
        orbit_number = extract_orbit_number(file_name)
        print(orbit_number)

        if orbit_number:
            nc_file_path = os.path.join(nc_input_folder, file_name)
            csv_file_path = os.path.join(csv_predictions_folder, f"orbit_{orbit_number}_predictions.csv")

            if os.path.exists(csv_file_path):
                output_file_path = os.path.join(nc_output_folder, file_name.replace('l1r', 'l1c'))

                # Read CSV
                predictions_df = pd.read_csv(csv_file_path)
                predictions_df = predictions_df.sort_values(by=['Frame', 'Box'])

                # Read time dimension from NetCDF
                with Dataset(nc_file_path, 'r') as nc_file:
                    time_dim = len(nc_file.dimensions['time'])

                # Init MLCloud array
                mlcloud_data = np.zeros((time_dim, 3, 15))

                # Fill MLCloud values
                for _, row in predictions_df.iterrows():
                    frame = int(row['Frame'])
                    box = row['Box']
                    probability = row['Probability']

                    if box in box_mapping:
                        x_idx, y_idx = box_mapping[box]
                        mlcloud_data[frame, y_idx, x_idx] = probability
                    else:
                        print(f"Box {box} not found in mapping.")

                # --------------------------
                # Handle first/last frames using "space"
                # --------------------------
                if time_dim > 2 * space:
                    mlcloud_data[:space, :, :] = mlcloud_data[space, :, :]
                    mlcloud_data[-space:, :, :] = mlcloud_data[-space-1, :, :]

                # Write to output NC file
                add_mlcloud_to_nc_file(nc_file_path, output_file_path, mlcloud_data)

            else:
                print(f"CSV file for orbit {orbit_number} not found in {csv_predictions_folder}")

print("Processing completed.")
