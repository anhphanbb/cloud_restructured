#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Process 'cloud_intervals_cleaned_10272025.csv'

Steps:
1. Read the CSV file.
2. Rename columns:
   Orbit #	Start (Top)	End (Top)	Start (Middle)	End (Middle)	Start (Bottom)	End (Bottom)	Ground Lights
   → Orbit #	(7,2)start	(7,2)end	(7,1)start	(7,1)end	(7,0)start	(7,0)end	Ground Lights
3. Keep only rows where "Ground Lights" != "Ground Lights".
4. Exclude rows where Orbit # is divisible by 45.
5. Save result as 'cloud_intervals_cleaned_filtered.csv' in the same folder.
"""

import pandas as pd
from pathlib import Path

# ====== INPUT ======
input_path = Path(r"D:\Github\cloud_restructured\csv\cloud_intervals_cleaned_10272025.csv")
output_path = input_path.with_name("cloud_intervals_cleaned_filtered.csv")

# ====== READ FILE ======
df = pd.read_csv(input_path, sep=",", engine="python")

# ====== RENAME COLUMNS ======
rename_map = {
    "Start (Top)": "(7,2)start",
    "End (Top)": "(7,2)end",
    "Start (Middle)": "(7,1)start",
    "End (Middle)": "(7,1)end",
    "Start (Bottom)": "(7,0)start",
    "End (Bottom)": "(7,0)end",
}
df = df.rename(columns=rename_map)

# ====== FILTER ======
# Remove header-like rows where Ground Lights == "Ground Lights"
df = df[df["Ground Lights"] != "Ground Lights"]

# Convert Orbit # to numeric (in case it's read as string)
df["Orbit #"] = pd.to_numeric(df["Orbit #"], errors="coerce")

# Drop rows where Orbit # divisible by 45
df = df[df["Orbit #"] % 45 != 0]

# ====== SAVE ======
df.to_csv(output_path, index=False)
print(f"Filtered CSV saved to: {output_path}")
print(f"Rows kept: {len(df)}")
