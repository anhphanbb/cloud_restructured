# -*- coding: utf-8 -*-
"""
Created on Thu Jul  4 13:10:28 2024

@author: Anh
"""

import tensorflow as tf
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.applications.resnet50 import preprocess_input
from tensorflow.keras.layers import Dense, Dropout, GlobalAveragePooling2D, Input
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import EarlyStopping, TensorBoard
import matplotlib.pyplot as plt
import pandas as pd
import time
import os
import numpy as np
from sklearn.metrics import confusion_matrix
from tensorflow.keras import backend as K
from tensorflow.keras.mixed_precision import set_global_policy
from tensorflow.keras.metrics import Precision, Recall

# # Use mixed precision training (If using GPUs)
# set_global_policy('mixed_float16')


# Ensure TensorFlow is using GPU if available
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(e)

# Define your data augmentation pipeline
def augment(image):
    image = tf.image.random_flip_left_right(image)
    image = tf.image.random_flip_up_down(image)
    # image = tf.image.random_brightness(image, max_delta=0.2)
    # image = tf.image.random_contrast(image, lower=0.8, upper=1.2)
    return image

# Set new image size based on bounding box dimensions
new_image_size = (100, 20)  # Set height=100 and width=20


# Load Data and Resize
data = tf.keras.utils.image_dataset_from_directory(
    'training_images',
    image_size=new_image_size,
    color_mode='rgb',  # Load images as RGB
    batch_size=32,
    label_mode='int'  # Ensure labels are integers for binary classification
)

# Apply augmentations
data = data.map(lambda x, y: (augment(x), y))

# Further preprocessing for ResNet-50
data = data.map(lambda x, y: (preprocess_input(x), y))

# Shuffle the dataset first
data = data.shuffle(buffer_size=200000, seed=42, reshuffle_each_iteration=False)

# Split Data
data_size = data.cardinality().numpy()
train_size = int(data_size * 0.7)
val_size = int(data_size * 0.2)
test_size = int(data_size * 0.1)

train = data.take(train_size)
val = data.skip(train_size).take(val_size)
test = data.skip(train_size + val_size).take(test_size)

# Define Transfer Learning Model using ResNet-50
def create_resnet_model():
    base_model = ResNet50(weights='imagenet', include_top=False, input_tensor=Input(shape=(new_image_size[0], new_image_size[1], 3)))
    base_model.trainable = False  # Freeze the base model layers

    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(512, activation='relu')(x)
    x = Dropout(0.5)(x)
    x = Dense(256, activation='relu')(x)
    x = Dropout(0.5)(x)
    predictions = Dense(1, activation='sigmoid')(x)  # Assuming binary classification
    
    model = Model(inputs=base_model.input, outputs=predictions)
    return model

model = create_resnet_model()

# Training and evaluating model
results = []
histories = []

model.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy', Recall(name='recall')]
    # metrics=['accuracy']
)

model.summary()

logdir = f'logs/resnet_model'
tensorboard_callback = TensorBoard(log_dir=logdir)
early_stopping_callback = EarlyStopping(
    monitor='val_loss',
    patience=100,
    restore_best_weights=True
)

start_time = time.time()

# Add class weights during training
hist = model.fit(
    train,
    epochs=500,
    validation_data=val,
    callbacks=[tensorboard_callback, early_stopping_callback]
)
end_time = time.time()
training_time = end_time - start_time

histories.append(hist)

# Save the Model
os.makedirs('models', exist_ok=True)
model.save('models/tf_model_cloud_py310_april_15_labels.h5')


# Plot validation metrics for ResNet-50 model
plt.figure(figsize=(12, 8))

# List of metrics to track
# metrics_to_plot = ['val_accuracy', 'val_loss', 'val_precision', 'val_recall']
metrics_to_plot = ['val_accuracy']

for metric in metrics_to_plot:
    plt.plot(histories[0].history[metric], label=metric.replace('val_', '').capitalize())

plt.axhline(y=0.75, color='r', linestyle='--', label='75% Accuracy (Baseline)')
plt.axhline(y=0.85, color='g', linestyle='--', label='85% Accuracy (Baseline)')
plt.title('Validation Metrics Trends for ResNet-50')
plt.xlabel('Epochs')
plt.ylabel('Metrics')
plt.legend()
plt.show()

# Evaluate model once on the test set
y_true = []
y_pred = []

for batch in test.as_numpy_iterator():
    X, y = batch
    yhat = model.predict(X)
    y_true.extend(y)
    y_pred.extend(yhat)

y_true = np.array(y_true)
y_pred = np.array(y_pred)

# Calculate confusion matrix, accuracy, and recall for thresholds
confusion_matrix_results = []

for threshold in range(0, 101):  # Thresholds from 0 to 100
    threshold_value = threshold / 100.0
    y_pred_thresholded = (y_pred >= threshold_value).astype(int)

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred_thresholded).ravel()
    accuracy = (tp + tn) / (tp + tn + fp + fn)
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0

    confusion_matrix_results.append({
        'Threshold': threshold_value,
        'True_Negatives': tn,
        'False_Positives': fp,
        'False_Negatives': fn,
        'True_Positives': tp,
        'Accuracy': accuracy,
        'Recall': recall
    })

# Save results to CSV
confusion_matrix_df = pd.DataFrame(confusion_matrix_results)
confusion_matrix_df.to_csv('resnet_confusion_matrix_with_accuracy_recall.csv', index=False)

print("Confusion matrix with accuracy and recall saved to resnet_confusion_matrix_with_accuracy_recall.csv")

# Plot accuracy and recall over thresholds
plt.figure(figsize=(12, 8))
plt.plot(confusion_matrix_df['Threshold'], confusion_matrix_df['Accuracy'], label='Accuracy', color='blue')
plt.plot(confusion_matrix_df['Threshold'], confusion_matrix_df['Recall'], label='Recall', color='orange')
plt.title('Accuracy and Recall Over Thresholds')
plt.xlabel('Threshold')
plt.ylabel('Metric Value')
plt.legend()
plt.grid()
plt.show()
