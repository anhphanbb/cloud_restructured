# -*- coding: utf-8 -*-
"""
Created on Wed Feb 12 13:04:57 2025

@author: domin
"""

import subprocess

scripts = [
    "4_1_create_images_to_predict.py",
    "4_2_predict_all_orbits.py",
    "5_1_create_mlcloud.py",
    "5_2_process_mlcloud.py"
]

for script in scripts:
    print(f"Running {script}...")
    result = subprocess.run(["python", script], capture_output=True, text=True)
    print(result.stdout)  # Print script output
    if result.stderr:
        print(f"Error in {script}:\n{result.stderr}")
