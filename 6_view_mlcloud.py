# -*- coding: utf-8 -*-
"""
Created on Wed Dec  4 13:15:23 2024

@author: anhph
"""

import netCDF4 as nc
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.widgets import Slider, RangeSlider, Button
import os
import re
from matplotlib.patches import Rectangle

# Define the path to the parent directory where the dataset is located
parent_directory = 'nc_files_with_mlcloud'

# Define the orbit number
orbit_number = 45  # orbit number

# Pad the orbit number with zeros until it has 5 digits
orbit_str = str(orbit_number).zfill(5)

# Search for the correct file name in all subdirectories
pattern = re.compile(r'awe_l(.*)_' + orbit_str + r'_(.*)\.nc')
dataset_filename = None
dataset_path = None

for root, dirs, files in os.walk(parent_directory):
    for file in files:
        if pattern.match(file):
            dataset_filename = file
            dataset_path = os.path.join(root, file)
            break
    if dataset_filename:
        break

if dataset_filename is None:
    raise FileNotFoundError(f"No file found for orbit number {orbit_str}")

# Load the dataset
dataset = nc.Dataset(dataset_path, 'r')
radiance = dataset.variables['Radiance'][:]
mlcloud = dataset.variables['MLCloud'][:]  # Load MLCloud variable
iss_latitude = dataset.variables['ISS_Latitude'][:]  # Load ISS latitude data
iss_longitude = dataset.variables['ISS_Longitude'][:]  # Load ISS longitude data

# Function to define grid boxes
def define_boxes():
    x_ranges = [
        (0, 19), (20, 39), (40, 59), (60, 79), (80, 99), (100, 119),
        (120, 139), (140, 159), (160, 179), (180, 199), (200, 219),
        (220, 239), (240, 259), (260, 279), (280, 299)
    ]
    y_ranges = [
        (0, 99), (100, 199), (200, 299)
    ]

    boxes = {}
    for j, y_range in enumerate(y_ranges):
        for i, x_range in enumerate(x_ranges):
            box_id = f"({i},{j})"
            boxes[box_id] = {'x': x_range, 'y': y_range}
    return boxes

grid_boxes = define_boxes()

# Optional box mapping (if needed)
box_mapping = {
    f"({x},{y})": (x, y) for y in range(3) for x in range(15)
}

# Initial setup
current_time_step = 0
show_highlight = True  # Toggle to show/hide highlights
show_numbers = False
show_lines = False
running_average_window = 5
threshold = 0.5

# Calculate initial vmin and vmax using the 0.4th and 99.7th percentiles
radiance_at_time_0 = radiance[0, :, :]
radiance_flat = radiance_at_time_0.flatten()
radiance_flat = radiance_flat[~np.isnan(radiance_flat)]
vmin_default = np.percentile(radiance_flat, 0.4) * 0.85
vmax_default = np.percentile(radiance_flat, 99.7) * 1.2

# Create figure and axes for the plot
fig, ax = plt.subplots(figsize=(12, 12))

# Adjust the figure to add space for the sliders
plt.subplots_adjust(bottom=0.3)

# Create the range slider for vmin and vmax on the bottom right
ax_range_slider = plt.axes([0.1, 0.1, 0.7, 0.03], facecolor='lightgoldenrodyellow')
range_slider = RangeSlider(ax_range_slider, 'vmin - vmax', 0, 40, valinit=(vmin_default, vmax_default))

# Create the slider for time step on the bottom left
ax_slider = plt.axes([0.1, 0.05, 0.7, 0.03], facecolor='lightgoldenrodyellow')
slider = Slider(ax_slider, 'Time Step', 0, radiance.shape[0] - 1, valinit=current_time_step, valfmt='%0.0f')

# Add a text label to show the running average value
running_avg_text = fig.text(0.5, 0.2, f"Running Average: {running_average_window}", fontsize=14, ha='center', va='center', color='blue')
threshold_text = fig.text(0.3, 0.2, f"ML Threshold: {threshold}", fontsize=14, ha='center', va='center', color='blue')

colorbar = None  # To keep track of the colorbar

# Set initial vmin and vmax
vmin, vmax = vmin_default, vmax_default

# Ensure the global variables are declared properly in the update function
def update_plot(time_step):
    global colorbar, current_time_step, show_highlight, show_numbers, show_lines
    current_time_step = int(time_step)
    
    # Get vmin and vmax from the range slider
    vmin, vmax = range_slider.val
    
    # Ensure vmin <= vmax
    vmin, vmax = min(vmin, vmax), max(vmin, vmax)
    
    # Clear previous content
    ax.clear()
    radiance_at_time = radiance[current_time_step, :, :]
    iss_lat = iss_latitude[current_time_step]
    iss_lon = iss_longitude[current_time_step]
    
    # Calculate the running average for mlcloud over a x-frame window
    kernel = np.ones(running_average_window) / running_average_window
    averaged_mlcloud = np.apply_along_axis(lambda m: np.convolve(m, kernel, mode='same'), axis=0, arr=mlcloud)
    averaged_mlcloud_at_time = averaged_mlcloud[current_time_step, :, :]
    
    # Plot the radiance data
    img = ax.imshow(radiance_at_time, origin='lower', cmap='gray', aspect='auto', vmin=vmin, vmax=vmax)
    ax.set_title(f'Radiance at Time Step {current_time_step}\nISS Position: Lat {iss_lat:.2f}, Lon {iss_lon:.2f}\nOrbit Number: {orbit_str}')
    ax.set_xlabel('Spatial Dimension X')
    ax.set_ylabel('Spatial Dimension Y')
    
    # Highlight boxes with averaged MLCloud > threshold and count them
    cloud_count = 0
    
    if show_highlight:
        for box, coords in grid_boxes.items():
            x_start, x_end = coords['x']
            y_start, y_end = coords['y']
            x_idx, y_idx = map(int, box.strip('()').split(','))
            
            # Use the averaged MLCloud to determine highlights
            if averaged_mlcloud_at_time[y_idx, x_idx] > threshold:
                cloud_count += 1
                ax.add_patch(Rectangle(
                    (x_start, y_start), x_end - x_start, y_end - y_start,
                    linewidth=0, edgecolor='none', facecolor='blue', alpha=0.3
                ))
    
    # Display solar panel count
    ax.text(
        0.0, 1.05, f"Cloud Boxes: {cloud_count}",
        transform=ax.transAxes, fontsize=14, color='Blue', weight='bold'
    )
    
    # Draw vertical and horizontal lines if enabled
    if show_lines:
        for x in range(20, radiance_at_time.shape[1], 20):
            ax.axvline(x=x, color='green', linestyle='-')
        for y in range(100, radiance_at_time.shape[0], 100):
            ax.axhline(y=y, color='green', linestyle='-')
            
    # Display the box numbers with a semi-transparent background if enabled
    if show_numbers:
        for x in range(0, radiance_at_time.shape[1], 20):
            for y in range(0, radiance_at_time.shape[0], 100):
                box_label_x = x + 10
                box_label_y = y + 40 + ((x//20)%2)*20
                ax.text(
                    box_label_x, box_label_y, f"({x//20},{y//100})",
                    color='blue', fontsize=16, ha='center', va='center',
                    bbox=dict(facecolor='white', alpha=0.6, edgecolor='none')  # Semi-transparent background
                )
    
    # Set the aspect of the plot axis to equal, enforcing a 1:1 aspect ratio
    ax.set_aspect('equal')

    # Add colorbar if it doesn't exist
    if colorbar is None:
        colorbar = plt.colorbar(img, ax=ax, orientation='vertical')
    else:
        colorbar.update_normal(img)

    # Explicitly redraw the figure canvas
    fig.canvas.draw_idle()

# Correct the slider and range slider callbacks
slider.on_changed(lambda val: update_plot(val))
range_slider.on_changed(lambda val: update_plot(slider.val))

# Ensure button callbacks correctly toggle global variables
def toggle_highlight(event):
    global show_highlight
    show_highlight = not show_highlight
    update_plot(current_time_step)

def toggle_numbers(event):
    global show_numbers
    show_numbers = not show_numbers
    update_plot(current_time_step)

def toggle_lines(event):
    global show_lines
    show_lines = not show_lines
    update_plot(current_time_step)
    
def update_vmin_vmax(event):
    global vmin, vmax, current_time_step
    radiance_at_time = radiance[current_time_step, :, :]
    radiance_flat = radiance_at_time.flatten()
    radiance_flat = radiance_flat[~np.isnan(radiance_flat)]
    
    if len(radiance_flat) == 0:
        raise ValueError("No valid data to compute percentiles.")
    
    vmin = np.percentile(radiance_flat, 0.4) * 0.85
    vmax = np.percentile(radiance_flat, 99.7) * 1.2
    range_slider.set_val((vmin, vmax))

# Create buttons and connect their functions
ax_button_vmin_vmax = plt.axes([0.1, 0.15, 0.14, 0.03], facecolor='lightgoldenrodyellow')
button_vmin_vmax = Button(ax_button_vmin_vmax, 'Set vmin-vmax (V)')
button_vmin_vmax.on_clicked(update_vmin_vmax)

# Create buttons and connect their functions
ax_button_highlight = plt.axes([0.26, 0.15, 0.14, 0.03], facecolor='lightgoldenrodyellow')
button_highlight = Button(ax_button_highlight, 'Toggle Highlight')
button_highlight.on_clicked(toggle_highlight)

# Create additional buttons and connect their functions
ax_button_numbers = plt.axes([0.42, 0.15, 0.14, 0.03], facecolor='lightgoldenrodyellow')
button_numbers = Button(ax_button_numbers, 'Toggle Numbers')
button_numbers.on_clicked(toggle_numbers)

ax_button_lines = plt.axes([0.58, 0.15, 0.14, 0.03], facecolor='lightgoldenrodyellow')
button_lines = Button(ax_button_lines, 'Toggle Lines')
button_lines.on_clicked(toggle_lines)


# Connect the slider and range slider to the update_plot function
slider.on_changed(lambda val: update_plot(val))
range_slider.on_changed(lambda val: update_plot(slider.val))

# Key press event
def on_key(event):
    global current_time_step
    if event.key == 'right':
        current_time_step = min(current_time_step + 1, radiance.shape[0] - 1)
    elif event.key == 'left':
        current_time_step = max(current_time_step - 1, 0)
    elif event.key == 'up':
        current_time_step = max(current_time_step - 20, 0)
    elif event.key == 'down':
        current_time_step = min(current_time_step + 20, radiance.shape[0] - 1)
    elif event.key == 'v':
        update_vmin_vmax(None)

    slider.set_val(current_time_step)  # This will automatically update the plot via the slider's on_changed event

fig.canvas.mpl_connect('key_press_event', on_key)

update_plot(current_time_step)

plt.show()
