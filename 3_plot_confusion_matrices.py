import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Load CSV
df = pd.read_csv('resnet_confusion_matrix_with_accuracy_recall.csv')

# Thresholds to plot
thresholds_to_plot = [0.3, 0.4, 0.5, 0.6]

# Set up 2x2 subplot
fig, axes = plt.subplots(2, 2, figsize=(10, 10))
axes = axes.flatten()

for i, threshold in enumerate(thresholds_to_plot):
    row = df[df['Threshold'] == threshold].iloc[0]

    tn = row['True_Negatives']
    fp = row['False_Positives']
    fn = row['False_Negatives']
    tp = row['True_Positives']

    total = tn + fp + fn + tp
    cm_percent = np.array([
        [tn / total * 100, fp / total * 100],
        [fn / total * 100, tp / total * 100]
    ])

    ax = axes[i]
    im = ax.imshow(cm_percent, cmap='Blues', vmin=0, vmax=100)
    ax.set_title(f"Threshold = {threshold}", fontsize=14)
    ax.set_xlabel("Predicted Label", fontsize=12)
    ax.set_ylabel("True Label", fontsize=12)
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(['No Cloud', 'Cloud'], fontsize=12)
    ax.set_yticklabels(['No Cloud', 'Cloud'], fontsize=12, rotation=90)

    ax.tick_params(axis='both', labelsize=12)

    for y in range(2):
        for x in range(2):
            ax.text(x, y, f"{cm_percent[y, x]:.1f}%", ha='center', va='center', color='black', fontsize=16)

# Add colorbar
fig.colorbar(im, ax=axes.ravel().tolist(), shrink=0.7, label='Percentage')
# plt.tight_layout()
plt.show()
