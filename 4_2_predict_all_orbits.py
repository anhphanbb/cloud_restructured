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
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.resnet50 import preprocess_input
import gc
from tensorflow.keras import backend as K
from concurrent.futures import ThreadPoolExecutor

# Path to the folder containing orbit subfolders with images 
# input_folder = 'images_to_predict' 
input_folder = r'E:\soc\l1r\2025\09\images_to_predict' 

# Output folder for CSV results 
# csv_output_folder = 'orbit_predictions' 
csv_output_folder = r'E:\soc\l1r\2025\09\orbit_predictions' 

# Path to the pre-trained model
model_path = 'models/tf_model_cloud_py310_space_8_max_rad_20_oct_27_2025.h5'

# Ensure the output folder for CSV files exists
os.makedirs(csv_output_folder, exist_ok=True)

print("CUDA version:", tf.sysconfig.get_build_info().get("cuda_version", "Not Found")) 
print("cuDNN version:", tf.sysconfig.get_build_info().get("cudnn_version", "Not Found")) 
print("GPU detected:", tf.config.list_physical_devices('GPU')) 

# Load the pre-trained model
model = load_model(model_path)

gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(e)

# @tf.function
def predict_batch(model, batch_images):
    return model(batch_images, training=False)

def load_image(path):
    return cv2.imread(path)

def preprocess_images_batch(image_paths):
    with ThreadPoolExecutor(max_workers=8) as executor:
        images = list(executor.map(load_image, image_paths))
    return preprocess_input(np.array(images))

# Function to compute running averages
def compute_running_average(predictions, window_size):
    """
    Compute the running average over a given window size.
    """
    return np.convolve(predictions, np.ones(window_size) / window_size, mode='valid')

# Function to remove all images from an orbit folder after processing
def remove_orbit_images(orbit_folder):
    for filename in os.listdir(orbit_folder):
        file_path = os.path.join(orbit_folder, filename)
        try:
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.unlink(file_path)
        except Exception as e:
            print(f'Failed to delete {file_path}. Reason: {e}')


# Function to process a single orbit folder with batching
def process_orbit_folder(orbit_folder, orbit_number, batch_size=32, avg_window_size=9):
    global model
    
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

    # Remove images after processing
    # remove_orbit_images(orbit_folder)
    
    # Invoking garbage collection at the end of each orbit. 
    del df_results, running_averages, results, image_data, batch_images
    gc.collect()
    
    # Clear session. Helps with slowing down after a few orbits. 
    K.clear_session()
    model = load_model(model_path)
    
# Main script to process all orbit subfolders
def process_all_orbits(input_folder, batch_size=32, avg_window_size=1):
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
process_all_orbits(input_folder, batch_size=2048, avg_window_size=1)
