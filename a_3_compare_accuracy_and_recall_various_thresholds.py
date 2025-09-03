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

# Replace with your actual training orbit numbers (as strings)
training_orbits = {'', '00090', ''}

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

# Save combined results to CSV
combined_df = pd.concat(
    [df.assign(threshold=thresh) for thresh, df in threshold_results.items()],
    ignore_index=True
)
combined_df.to_csv('mlcloud_comparison_all_thresholds.csv', index=False)

# Calculate average accuracy and recall for each threshold with standard deviation
avg_metrics = []

for thresh, df in threshold_results.items():
    avg_acc = df['accuracy'].mean()
    std_acc = df['accuracy'].std()
    avg_rec = df['recall'].mean()
    std_rec = df['recall'].std()
    avg_metrics.append({
        'threshold': thresh,
        'avg_accuracy': round(avg_acc, 2),
        'std_accuracy': round(std_acc, 2),
        'avg_recall': round(avg_rec, 2),
        'std_recall': round(std_rec, 2)
    })

avg_df = pd.DataFrame(avg_metrics)
avg_df.to_csv('mlcloud_avg_metrics_per_threshold.csv', index=False)
print("Saved average metrics (with std) to 'mlcloud_avg_metrics_per_threshold.csv'")

# Function to add background overlays
def highlight_training_orbits(ax, x_labels, training_set, color='lightgray'):
    for i, orbit in enumerate(x_labels):
        if orbit in training_set:
            ax.axvspan(i - 0.5, i + 0.5, color=color, alpha=0.5)

# Plot accuracy
plt.figure(figsize=(10, 5))
ax1 = plt.gca()
for thresh in sorted(threshold_results.keys()):
    df = threshold_results[thresh]
    orbits = df['orbit'].astype(str)
    ax1.plot(orbits, df['accuracy'], label=f'Thresh {thresh:.2f}', marker='o')

highlight_training_orbits(ax1, orbits, training_orbits)

plt.title('Accuracy per Orbit at Different Thresholds')
plt.xlabel('Orbit')
plt.ylabel('Accuracy')
plt.legend()
plt.xticks(rotation=45)
plt.grid(True)
plt.tight_layout()
plt.savefig('accuracy_vs_orbit_may_30.png')

# Plot recall
plt.figure(figsize=(10, 5))
ax2 = plt.gca()
for thresh in sorted(threshold_results.keys()):
    df = threshold_results[thresh]
    orbits = df['orbit'].astype(str)
    ax2.plot(orbits, df['recall'], label=f'Thresh {thresh:.2f}', marker='o')

highlight_training_orbits(ax2, orbits, training_orbits)

plt.title('Recall per Orbit at Different Thresholds')
plt.xlabel('Orbit')
plt.ylabel('Recall')
plt.legend()
plt.xticks(rotation=45)
plt.grid(True)
plt.tight_layout()
plt.savefig('recall_vs_orbit_may_30.png')

print("Comparison CSV and plots with training orbit overlay saved.")
