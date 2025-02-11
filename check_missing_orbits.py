import os
import re

# Define the base folder path
base_folder_path = r'Z:\soc'

# Define the l1c variations and month-year range
l1c_variations = ['l1r', 'l1c', 'l1a', 'l2a', 'l3a', 'l3c']
months_years = [
    (11, 2023), (12, 2023), (1, 2024), (2, 2024), (3, 2024), (4, 2024)
]

# Define the regex pattern to extract relevant parts
pattern = re.compile(r'awe_(\w{3})_?(\w{3})?_\d{7}T\d{4}_(\d{5})_v\d{2}\.nc')

# Dictionary to store grouped files
month_year_data = {}

# Iterate through the l1c variations and month-year combinations
for l1c in l1c_variations:
    for month, year in months_years:
        folder_path = os.path.join(base_folder_path, l1c, f"{year:04}", f"{month:02}")
        print(f"Checking directory: {folder_path}")  # Print out the directories
        if not os.path.exists(folder_path):
            continue

        # Initialize data for this month and year if not already
        month_year_key = (month, year)
        if month_year_key not in month_year_data:
            month_year_data[month_year_key] = {}

        # Iterate through the folder
        for file_name in os.listdir(folder_path):
            match = pattern.match(file_name)
            if match:
                l1c_key, q20, orbit = match.groups()
                key = (l1c_key, q20)
                orbit_num = int(orbit)

                if key not in month_year_data[month_year_key]:
                    month_year_data[month_year_key][key] = []

                month_year_data[month_year_key][key].append(orbit_num)

# Process the grouped data
for month_year, groups in month_year_data.items():
    print(f"\nMonth-Year: {month_year[1]:04}-{month_year[0]:02}")
    for key, orbits in groups.items():
        orbits.sort()
        first_orbit = orbits[0]
        last_orbit = orbits[-1]
        missing_orbits = [o for o in range(first_orbit, last_orbit + 1) if o not in orbits]

        print(f"  Group {key}:")
        print(f"    First Orbit: {first_orbit}")
        print(f"    Last Orbit: {last_orbit}")
        print(f"    Missing Orbits: {missing_orbits}")
