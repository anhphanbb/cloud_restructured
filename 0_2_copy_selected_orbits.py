# import os
# import shutil
# import re

# # Define paths
# nc_folder = r'Z:\soc\l1r'  # Source folder containing .nc files
# selected_folder = r'D:\Github\labeled_nc_files'  # Destination folder

# # List of specific orbits to copy
# # orbit_list = list(range(2, 139, 2))#, 180, 225, 270, 315, 360, 405, 450, 495, 545, 590, 635, 815, 1130, 1175, 1265, 1310, 1355, 1400, 1445, 1490, 2525, 2570, 2615, 2660, 2705, 2750, 2795, 2840, 2885]
# orbit_list = [1100, 1110, 1115, 1117, 1125, 1130, 1134, 1140, 1145, 1151, 1160, 1168, 1175, 1183, 1185, 1202, 1219, 1236, 2495, 2865, 2880, 2895, 2910, 2925, 2940, 2955, 2970, 4200, 4215, 4230, 4245, 4260, 4275, 4290, 4305, 4320]

# # Ensure output folder exists
# os.makedirs(selected_folder, exist_ok=True)

# # Function to find .nc file
# def find_nc_file(parent_directory, orbit_number):
#     orbit_str = str(int(orbit_number)).zfill(5)
#     pattern = re.compile(r'awe_l1r_q20_(.*)_' + orbit_str + r'_(.*)\.nc')
    
#     for root, dirs, files in os.walk(parent_directory):
#         for file in files:
#             if pattern.match(file):
#                 return os.path.join(root, file)
#     return None

# # Copy selected .nc files to the new folder
# def copy_selected_nc_files(parent_directory, destination_folder, orbit_list):
#     for orbit_number in orbit_list:
#         nc_file_path = find_nc_file(parent_directory, orbit_number)
#         if nc_file_path:
#             dest_path = os.path.join(destination_folder, os.path.basename(nc_file_path))
#             shutil.copy2(nc_file_path, dest_path)
#             print(f"Copied {nc_file_path} to {dest_path}")
#         else:
#             print(f"No file found for orbit {orbit_number}")

# # Run the script
# copy_selected_nc_files(nc_folder, selected_folder, orbit_list)

#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import re
import csv
import shutil
import argparse

# ---------- helpers ----------

def find_nc_file(parent_directory, orbit_number):
    """Recursively find a .nc file whose name contains the 5-digit orbit number."""
    orbit_str = str(int(orbit_number)).zfill(5)
    pattern = re.compile(rf'awe_l1r_q20_(.*)_{orbit_str}_(.*)\.nc$', re.IGNORECASE)

    for root, _, files in os.walk(parent_directory):
        for fname in files:
            if pattern.match(fname):
                return os.path.join(root, fname)
    return None


def copy_selected_nc_files(parent_directory, destination_folder, orbit_list, dry_run=False):
    os.makedirs(destination_folder, exist_ok=True)
    copied = 0
    missing = 0

    for orbit_number in orbit_list:
        nc_file_path = find_nc_file(parent_directory, orbit_number)
        if nc_file_path:
            dest_path = os.path.join(destination_folder, os.path.basename(nc_file_path))
            if dry_run:
                print(f"[DRY RUN] Would copy {nc_file_path} -> {dest_path}")
            else:
                shutil.copy2(nc_file_path, dest_path)
                print(f"Copied {nc_file_path} -> {dest_path}")
            copied += 1
        else:
            print(f"No file found for orbit {orbit_number}")
            missing += 1

    print(f"\nDone. Copied: {copied}, Missing: {missing}, Total requested: {len(orbit_list)}")


def load_orbits_from_csv(csv_path, column_name):
    """
    Read orbit numbers from a CSV column.
    - Tries to coerce values to int
    - Drops blanks/NaN
    - Returns sorted unique list
    """
    orbits = []
    with open(csv_path, "r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        if column_name not in reader.fieldnames:
            raise ValueError(
                f"Column '{column_name}' not found. Available columns: {reader.fieldnames}"
            )
        for row in reader:
            val = row.get(column_name, "").strip()
            if not val:
                continue
            try:
                orbits.append(int(float(val)))
            except ValueError:
                # ignore non-numeric cells in the column
                continue
    # unique + sorted
    return sorted(set(orbits))


# ---------- main ----------

def main():
    parser = argparse.ArgumentParser(
        description="Copy selected AWE .nc files by orbit. "
                    "Use a CSV column or a hardcoded list."
    )
    parser.add_argument("--nc_folder", default=r"Z:\soc\l1r", help="Root folder containing .nc files")
    parser.add_argument("--dest_folder", default=r"D:\Github\cloud_restructured\training_nc_files", help="Destination folder")

    # Option A: read from CSV
    parser.add_argument("--csv", dest="csv_path", default=r"D:\Github\cloud_restructured\csv\cloud_intervals_cleaned_filtered.csv",
                        help="Path to CSV that has orbit numbers")
    parser.add_argument("--csv_column", default="Orbit #",
                        help="Column name in the CSV that contains orbit numbers")

    # Option B: fallback hardcoded list (used only when --csv is not provided)
    parser.add_argument("--use_default_list", action="store_true",
                        help="Use the hardcoded orbit_list if --csv is not provided")

    parser.add_argument("--dry_run", action="store_true",
                        help="Print what would be copied without writing files")

    args = parser.parse_args()

    # Hardcoded list (only used if --csv is omitted and --use_default_list is given)
    orbit_list_default = [
        1100, 1110, 1115, 1117, 1125, 1130, 1134, 1140, 1145, 1151,
        1160, 1168, 1175, 1183, 1185, 1202, 1219, 1236, 2495, 2865,
        2880, 2895, 2910, 2925, 2940, 2955, 2970, 4200, 4215, 4230,
        4245, 4260, 4275, 4290, 4305, 4320
    ]

    # Determine the source of orbit numbers
    if args.csv_path:
        print(f"Loading orbits from CSV: {args.csv_path} (column: '{args.csv_column}')")
        orbit_list = load_orbits_from_csv(args.csv_path, args.csv_column)
        if not orbit_list:
            raise SystemExit("No valid orbit numbers found in the CSV.")
        print(f"Found {len(orbit_list)} unique orbit(s) in CSV.")
    elif args.use_default_list:
        print("Using hardcoded orbit_list.")
        orbit_list = orbit_list_default
    else:
        raise SystemExit(
            "Please provide --csv <path> (and optionally --csv_column <name>) "
            "or pass --use_default_list."
        )

    copy_selected_nc_files(args.nc_folder, args.dest_folder, orbit_list, dry_run=args.dry_run)


if __name__ == "__main__":
    main()
