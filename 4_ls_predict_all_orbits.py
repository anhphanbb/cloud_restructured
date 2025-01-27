# -*- coding: utf-8 -*-
"""
Updated on Tue Dec  3 20:10:39 2024

@author: anhph
"""

import os
import cv2
import numpy as np
import pandas as pd
import re
import time
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.resnet50 import preprocess_input

# Path to the folder containing orbit subfolders with images
input_folder = 'images_to_predict'
# input_folder = r'D:\soc\l1r\2024\03\images_to_predict'

# Output folder for CSV results
csv_output_folder = 'ls_orbit_predictions'
# csv_output_folder = r'D:\soc\l1r\2024\03\orbit_predictions'

# Path to the pre-trained model
model_path = 'models/DeepLearning_resnet_model_ls.h5'

# Ensure the output folder for CSV files exists
os.makedirs(csv_output_folder, exist_ok=True)

# Load the pre-trained model
model = load_model(model_path)

# Function to preprocess a batch of images for the model
def preprocess_images_batch(image_paths):
    """
    Preprocess a batch of images to match the model's expected input.
    Resize to (43, 43) and apply ResNet-50 preprocessing.
    """
    batch = []
    for image_path in image_paths:
        image = cv2.imread(image_path)  # Load image in RGB directly
        image_resized = cv2.resize(image, (100, 20))  # Resize to match training size # Set height=100 and width=20
        image_resized = image  # Resize to match training size
        image_preprocessed = preprocess_input(image_resized)  # Apply ResNet-50 preprocessing
        batch.append(image_preprocessed)
    return np.array(batch)  # Convert to a batch for prediction

# Function to compute running averages
def compute_running_average(predictions, window_size):
    """
    Compute the running average over a given window size.
    """
    return np.convolve(predictions, np.ones(window_size) / window_size, mode='valid')

# Function to process a single orbit folder with batching
def process_orbit_folder(orbit_folder, orbit_number, batch_size=32, avg_window_size=9):
    start_time = time.time()
    results = []  # To store results for the current orbit
    images = os.listdir(orbit_folder)

    # Extract frame numbers and boxes from filenames
    image_data = []
    for image_name in images:
        match = re.match(r"frame_(\d+)_box_\((\d+),(\d+)\)\.png", image_name)
        if match:
            frame_number = int(match.group(1))
            box = f"({match.group(2)},{match.group(3)})"
            image_path = os.path.join(orbit_folder, image_name)
            image_data.append((frame_number, box, image_path))

    # Sort images by frame number for proper averaging
    image_data.sort(key=lambda x: x[0])

    # Process images in batches
    total_images = len(image_data)
    for start_idx in range(0, total_images, batch_size):
        batch_data = image_data[start_idx:start_idx + batch_size]
        batch_frame_numbers = [item[0] for item in batch_data]
        batch_boxes = [item[1] for item in batch_data]
        batch_image_paths = [item[2] for item in batch_data]

        # Preprocess the batch
        batch_images = preprocess_images_batch(batch_image_paths)

        # Predict probabilities for the batch
        probabilities = model.predict(batch_images, verbose=0).flatten()

        # Store results
        for frame_number, box, probability in zip(batch_frame_numbers, batch_boxes, probabilities):
            results.append({"Frame": frame_number, "Box": box, "Probability": probability})

        # Show progress
        print(f"Processed {start_idx + len(batch_data)}/{total_images} images in Orbit {orbit_number}...")

    # Compute running averages for each box
    df_results = pd.DataFrame(results)
    running_averages = []
    for box, group in df_results.groupby("Box"):
        probabilities = group["Probability"].to_list()
        frames = group["Frame"].to_list()
        averaged_probs = compute_running_average(probabilities, avg_window_size)
        
        # Add raw probabilities and running averages to the results
        for i, avg_prob in enumerate(averaged_probs):
            running_averages.append({
                "Frame": frames[i + avg_window_size // 2],  # Center frame of the window
                "Box": box,
                "Probability": probabilities[i + avg_window_size // 2],  # Raw probability
                "RunningAverageProbability": avg_prob  # Running average
            })

    # Save the results to a CSV file for the orbit
    output_csv_path = os.path.join(csv_output_folder, f"orbit_{orbit_number}_predictions.csv")
    pd.DataFrame(running_averages).to_csv(output_csv_path, index=False)
    
    # Print total time taken
    end_time = time.time()
    print(f"Completed processing for Orbit {orbit_number}. Time taken: {end_time - start_time:.2f} seconds")
    print(f"Results saved to {output_csv_path}")

# Main script to process all orbit subfolders
def process_all_orbits(input_folder, batch_size=32, avg_window_size=9):
    for orbit_folder_name in os.listdir(input_folder):
        orbit_folder_path = os.path.join(input_folder, orbit_folder_name)
        if os.path.isdir(orbit_folder_path):
            # Extract orbit number from folder name
            orbit_match = re.search(r'orbit_(\d+)', orbit_folder_name)
            if orbit_match:
                orbit_number = orbit_match.group(1)
                print(f"Processing orbit folder: {orbit_folder_name}")
                process_orbit_folder(orbit_folder_path, orbit_number, batch_size=batch_size, avg_window_size=avg_window_size)

# Run the script
process_all_orbits(input_folder, batch_size=64, avg_window_size=1)
