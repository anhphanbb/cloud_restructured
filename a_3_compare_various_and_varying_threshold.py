# -*- coding: utf-8 -*-
"""
Created on Sat Jun 14 22:50:25 2025

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

training_orbits = {'', '00090', ''}

# --- Fixed Threshold Evaluation ---
threshold_results = {}

for threshold in np.array([0.4, 0.5, 0.6, 1.1]):
    results = []
    for file_name in os.listdir(folder_a):
        if not file_name.endswith('.nc'):
            continue

        path_a = os.path.join(folder_a, file_name)
        path_b = os.path.join(folder_b, file_name)
        if not os.path.exists(path_b):
            continue

        with Dataset(path_a, 'r') as nc_a, Dataset(path_b, 'r') as nc_b:
            data_a = nc_a.variables['Processed_MLCloud'][:]
            data_b = nc_b.variables['Processed_MLCloud'][:]
            if data_a.shape != data_b.shape:
                continue

            preds_a, preds_b = [], []
            for t in range(data_a.shape[0]):
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

    df = pd.DataFrame(results)
    threshold_results[threshold] = df

# Combine and save fixed threshold results
combined_df = pd.concat([df.assign(threshold=thresh) for thresh, df in threshold_results.items()], ignore_index=True)
combined_df.to_csv('mlcloud_comparison_all_thresholds.csv', index=False)

# Save average metrics for fixed thresholds
avg_metrics = []
for thresh, df in threshold_results.items():
    avg_metrics.append({
        'threshold': thresh,
        'avg_accuracy': round(df['accuracy'].mean(), 2),
        'std_accuracy': round(df['accuracy'].std(), 2),
        'avg_recall': round(df['recall'].mean(), 2),
        'std_recall': round(df['recall'].std(), 2)
    })

avg_df = pd.DataFrame(avg_metrics)
avg_df.to_csv('mlcloud_avg_metrics_per_threshold.csv', index=False)
print("Saved fixed-threshold results.")

# --- Latitude-Based Threshold Evaluation ---
def latitude_based_threshold(lat):
    base = 0.5
    slope = 0.02
    top = 1
    if np.isnan(lat):
        return base
    if -20 <= lat <= 30:
        return base
    elif lat < -20:
        return min(base + slope * (-20 - lat), top)
    elif lat > 30:
        return min(base + slope * (lat - 30), top)
    return base

results_lat = []

for file_name in os.listdir(folder_a):
    if not file_name.endswith('.nc'):
        continue

    path_a = os.path.join(folder_a, file_name)
    path_b = os.path.join(folder_b, file_name)
    if not os.path.exists(path_b):
        continue

    with Dataset(path_a, 'r') as nc_a, Dataset(path_b, 'r') as nc_b:
        data_a = nc_a.variables['Processed_MLCloud'][:]
        data_b = nc_b.variables['Processed_MLCloud'][:]
        if data_a.shape != data_b.shape:
            continue

        try:
            center_lat = nc_a.variables['Center_Latitude'][:]
        except KeyError:
            print(f"Skipping {file_name}: 'Center_Latitude' not found.")
            continue

        preds_a, preds_b = [], []
        for t in range(data_a.shape[0]):
            threshold = latitude_based_threshold(center_lat[t])
            for y, x in coords:
                val_a = 1 if data_a[t, y, x] >= threshold else 0
                val_b = 1 if data_b[t, y, x] >= 0.5 else 0
                preds_a.append(val_a)
                preds_b.append(val_b)

        cm = confusion_matrix(preds_b, preds_a, labels=[0, 1])
        acc = accuracy_score(preds_b, preds_a)
        recall = recall_score(preds_b, preds_a)
        tn, fp, fn, tp = cm.ravel() if cm.shape == (2, 2) else (0, 0, 0, 0)

        results_lat.append({
            'orbit': extract_orbit_number(file_name),
            'TN': tn,
            'FP': fp,
            'FN': fn,
            'TP': tp,
            'accuracy': acc,
            'recall': recall
        })

df_lat = pd.DataFrame(results_lat)
df_lat.to_csv('mlcloud_latitude_threshold_comparison.csv', index=False)

avg_lat = {
    'threshold': 'latitude_based',
    'avg_accuracy': round(df_lat['accuracy'].mean(), 2),
    'std_accuracy': round(df_lat['accuracy'].std(), 2),
    'avg_recall': round(df_lat['recall'].mean(), 2),
    'std_recall': round(df_lat['recall'].std(), 2)
}
pd.DataFrame([avg_lat]).to_csv('mlcloud_latitude_threshold_avg_metrics.csv', index=False)
print("Saved latitude-based threshold results.")

# --- Plotting ---
def highlight_training_orbits(ax, x_labels, training_set, color='lightgray'):
    for i, orbit in enumerate(x_labels):
        if orbit in training_set:
            ax.axvspan(i - 0.5, i + 0.5, color=color, alpha=0.5)

# Accuracy plot
plt.figure(figsize=(12, 5))
ax1 = plt.gca()
for thresh in sorted(threshold_results.keys()):
    df = threshold_results[thresh]
    orbits = df['orbit'].astype(str)
    ax1.plot(orbits, df['accuracy'], label=f'Fixed {thresh:.2f}', marker='o')
ax1.plot(df_lat['orbit'].astype(str), df_lat['accuracy'], marker='s', linestyle='--', color='black', label='Latitude-Based')
highlight_training_orbits(ax1, df_lat['orbit'].astype(str), training_orbits)
plt.title('Accuracy per Orbit (Fixed vs Latitude-Based Threshold)')
plt.xlabel('Orbit')
plt.ylabel('Accuracy')
plt.legend()
plt.xticks(rotation=45)
plt.grid(True)
plt.tight_layout()
plt.savefig('accuracy_vs_orbit_combined.png')
print("Saved accuracy plot.")

# Recall plot
plt.figure(figsize=(12, 5))
ax2 = plt.gca()
for thresh in sorted(threshold_results.keys()):
    df = threshold_results[thresh]
    orbits = df['orbit'].astype(str)
    ax2.plot(orbits, df['recall'], label=f'Fixed {thresh:.2f}', marker='o')
ax2.plot(df_lat['orbit'].astype(str), df_lat['recall'], marker='s', linestyle='--', color='black', label='Latitude-Based')
highlight_training_orbits(ax2, df_lat['orbit'].astype(str), training_orbits)
plt.title('Recall per Orbit (Fixed vs Latitude-Based Threshold)')
plt.xlabel('Orbit')
plt.ylabel('Recall')
plt.legend()
plt.xticks(rotation=45)
plt.grid(True)
plt.tight_layout()
plt.savefig('recall_vs_orbit_combined.png')
print("Saved recall plot.")
