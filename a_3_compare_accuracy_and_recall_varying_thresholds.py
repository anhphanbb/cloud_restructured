# -*- coding: utf-8 -*-
"""
Created on Sat Jun 14 22:16:08 2025

@author: domin
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from netCDF4 import Dataset
from sklearn.metrics import confusion_matrix, accuracy_score, recall_score

folder_a = 'processed_selected_nc_files_with_mlcloud'
folder_b = 'processed_selected_nc_files_with_mlcloud_from_labels'

coords = [(0, 7), (1, 7), (2, 7)]

def extract_orbit_number(filename):
    return filename.split('_')[4]

# Training orbits for highlighting
training_orbits = {'', '00090', ''}

# Latitude-based threshold function
def latitude_based_threshold(lat):
    base_threshold = .5
    slope = .02
    if np.isnan(lat):
        return base_threshold
    if -20 <= lat <= 30:
        return base_threshold
    elif lat < -20:
        return min(base_threshold + slope * (-20 - lat), 1.00)
    elif lat > 30:
        return min(base_threshold + slope * (lat - 30), 1.00)
    return base_threshold

results = []

for file_name in os.listdir(folder_a):
    if not file_name.endswith('.nc'):
        continue

    path_a = os.path.join(folder_a, file_name)
    path_b = os.path.join(folder_b, file_name)

    if not os.path.exists(path_b):
        continue

    with Dataset(path_a, 'r') as nc_a, Dataset(path_b, 'r') as nc_b:
        data_a = nc_a.variables['Processed_MLCloud'][:]  # predictions
        data_b = nc_b.variables['Processed_MLCloud'][:]  # labels

        if data_a.shape != data_b.shape:
            continue

        try:
            center_lat = nc_a.variables['Center_Latitude'][:]
        except KeyError:
            print(f"Skipping {file_name}: 'Center_Latitude' not found.")
            continue

        preds_a, preds_b = [], []

        for t in range(data_a.shape[0]):
            lat = center_lat[t]
            threshold = latitude_based_threshold(lat)

            for y, x in coords:
                val_a = 1 if data_a[t, y, x] >= threshold else 0
                val_b = 1 if data_b[t, y, x] >= 0.5 else 0
                preds_a.append(val_a)
                preds_b.append(val_b)

        cm = confusion_matrix(preds_b, preds_a, labels=[0, 1])
        acc = accuracy_score(preds_b, preds_a)
        recall = recall_score(preds_b, preds_a)

        tn, fp, fn, tp = cm.ravel() if cm.shape == (2, 2) else (0, 0, 0, 0)

        results.append({
            'orbit': extract_orbit_number(file_name),
            'TN': tn,
            'FP': fp,
            'FN': fn,
            'TP': tp,
            'accuracy': acc,
            'recall': recall
        })

# Save CSV
df = pd.DataFrame(results)
df.to_csv('mlcloud_latitude_threshold_comparison.csv', index=False)
print("Saved comparison CSV: 'mlcloud_latitude_threshold_comparison.csv'")

# Compute average accuracy and recall
avg_metrics = {
    'avg_accuracy': round(df['accuracy'].mean(), 2),
    'std_accuracy': round(df['accuracy'].std(), 2),
    'avg_recall': round(df['recall'].mean(), 2),
    'std_recall': round(df['recall'].std(), 2)
}
pd.DataFrame([avg_metrics]).to_csv('mlcloud_latitude_threshold_avg_metrics.csv', index=False)
print("Saved average metrics to 'mlcloud_latitude_threshold_avg_metrics.csv'")

# Plotting helper
def highlight_training_orbits(ax, x_labels, training_set, color='lightgray'):
    for i, orbit in enumerate(x_labels):
        if orbit in training_set:
            ax.axvspan(i - 0.5, i + 0.5, color=color, alpha=0.5)

# Plot accuracy
plt.figure(figsize=(10, 5))
ax1 = plt.gca()
orbits = df['orbit'].astype(str)
ax1.plot(orbits, df['accuracy'], marker='o', label='Latitude-based Threshold')
highlight_training_orbits(ax1, orbits, training_orbits)
plt.title('Accuracy per Orbit (Latitude-Based Threshold)')
plt.xlabel('Orbit')
plt.ylabel('Accuracy')
plt.legend()
plt.xticks(rotation=45)
plt.grid(True)
plt.tight_layout()
plt.savefig('accuracy_vs_orbit_latitude_based.png')

# Plot recall
plt.figure(figsize=(10, 5))
ax2 = plt.gca()
ax2.plot(orbits, df['recall'], marker='o', label='Latitude-based Threshold')
highlight_training_orbits(ax2, orbits, training_orbits)
plt.title('Recall per Orbit (Latitude-Based Threshold)')
plt.xlabel('Orbit')
plt.ylabel('Recall')
plt.legend()
plt.xticks(rotation=45)
plt.grid(True)
plt.tight_layout()
plt.savefig('recall_vs_orbit_latitude_based.png')

print("Plots with latitude-based threshold saved.")
