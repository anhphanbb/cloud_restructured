import os
import shutil
import re

# Define paths
nc_folder = r'Z:\soc\l1r'  # Source folder containing .nc files
selected_folder = 'selected_nc_files'  # Destination folder

# List of specific orbits to copy
orbit_list = [135, 180, 225, 270, 315, 360, 405, 450, 495, 545, 590, 635, 815, 1130, 1175, 1265, 1310, 1355, 1400, 1445, 1490, 2525, 2570, 2615, 2660, 2705, 2750, 2795, 2840, 2885]

# Ensure output folder exists
os.makedirs(selected_folder, exist_ok=True)

# Function to find .nc file
def find_nc_file(parent_directory, orbit_number):
    orbit_str = str(int(orbit_number)).zfill(5)
    pattern = re.compile(r'awe_l1r_q20_(.*)_' + orbit_str + r'_(.*)\.nc')
    
    for root, dirs, files in os.walk(parent_directory):
        for file in files:
            if pattern.match(file):
                return os.path.join(root, file)
    return None

# Copy selected .nc files to the new folder
def copy_selected_nc_files(parent_directory, destination_folder, orbit_list):
    for orbit_number in orbit_list:
        nc_file_path = find_nc_file(parent_directory, orbit_number)
        if nc_file_path:
            dest_path = os.path.join(destination_folder, os.path.basename(nc_file_path))
            shutil.copy2(nc_file_path, dest_path)
            print(f"Copied {nc_file_path} to {dest_path}")
        else:
            print(f"No file found for orbit {orbit_number}")

# Run the script
copy_selected_nc_files(nc_folder, selected_folder, orbit_list)
