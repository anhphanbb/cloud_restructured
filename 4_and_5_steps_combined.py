import os
import shutil
import re
import subprocess

# Define paths
nc_folder = r'E:\soc\l1r\2024\06'
images_folder = os.path.join(nc_folder, 'images_to_predict')
csv_predictions_folder = os.path.join(nc_folder, 'orbit_predictions')
nc_output_folder = os.path.join(nc_folder, 'nc_files_with_mlcloud')
processed_nc_folder = r'E:\soc\l1c\2024\06'

# Ensure necessary folders exist
os.makedirs(images_folder, exist_ok=True)
os.makedirs(csv_predictions_folder, exist_ok=True)
os.makedirs(nc_output_folder, exist_ok=True)
os.makedirs(processed_nc_folder, exist_ok=True)

# List of netCDF files (each corresponds to an orbit)
nc_files = [f for f in os.listdir(nc_folder) if f.endswith('.nc') and 'q20' in f]

for nc_file in nc_files:
    # Extract orbit number
    orbit_match = re.search(r'_(\d{5})_', nc_file)
    if not orbit_match:
        print(f"Skipping file: {nc_file}, orbit number not found.")
        continue

    orbit_number = orbit_match.group(1)
    orbit_folder = os.path.join(images_folder, f"orbit_{orbit_number}")

    print(f"Processing Orbit {orbit_number}...")

    # Step 1: Create images for prediction
    subprocess.run(["python", "4_create_images_to_predict.py"])

    # Step 2: Predict cloud probabilities
    subprocess.run(["python", "4_predict_all_orbits.py"])

    # Step 3: Create MLCloud NetCDF files
    subprocess.run(["python", "5_1_create_mlcloud.py"])

    # Step 4: Process MLCloud variable
    subprocess.run(["python", "5_2_process_mlcloud.py"])

    # Step 5: Delete the images of the processed orbit
    if os.path.exists(orbit_folder):
        shutil.rmtree(orbit_folder)
        print(f"Deleted images for Orbit {orbit_number}")

print("All orbits processed successfully.")
