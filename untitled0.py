# gridsearch_create_and_train.py
# -*- coding: utf-8 -*-
"""
End-to-end grid search over:
  - space: 0..6 (inclusive)
  - max_radiance: {20, 22, 24, 26, 28, 30}

For each grid cell:
  1) Generate training images using config + (space, max_radiance)
  2) Train a ResNet50-based classifier for 100 epochs with patience=20
  3) Save model, TensorBoard logs, metrics plot, and threshold sweep CSV

Base configuration is read from config.txt, which can contain either
comma-separated or line-separated key=value pairs. Example:

  space = 5,
  min_radiance = 0,
  max_radiance = 24,
  cloud_chance = 0.5,
  no_cloud_chance = 0.1,
  threshold = 3

Notes
  - min_radiance, cloud_chance, no_cloud_chance, threshold come from config.
  - max_radiance & space are overridden by each grid combo.
  - Each combo stores outputs under: grid_runs/space{S}_max{M}/...
"""

import os
import re
import json
import time
import shutil
import random
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.applications.resnet50 import preprocess_input
from tensorflow.keras.layers import Dense, Dropout, GlobalAveragePooling2D, Input
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import EarlyStopping, TensorBoard
from tensorflow.keras import backend as K
from tensorflow.keras.metrics import Recall
from sklearn.metrics import confusion_matrix
from netCDF4 import Dataset
import cv2

# ----------------------- CONFIG -----------------------
CONFIG_PATH = 'config.txt'  # change if needed

DEFAULTS = {
    'space': 5,
    'min_radiance': 0.0,
    'max_radiance': 24.0,
    'cloud_chance': 0.5,
    'no_cloud_chance': 0.1,
    'threshold': 3,
}

def parse_config(config_path):
    cfg = DEFAULTS.copy()
    if not os.path.isfile(config_path):
        print(f"[warn] Config not found: {config_path}. Using defaults.")
        return cfg
    with open(config_path, 'r', encoding='utf-8') as f:
        text = f.read()
    text = re.sub(r'#.*', '', text)
    pairs = re.findall(r'([A-Za-z_][A-Za-z0-9_]*)\s*=\s*([^,\n\r]+)', text)
    for key, val_raw in pairs:
        key = key.strip()
        val = val_raw.strip()
        try:
            if key in ('space', 'threshold'):
                cfg[key] = int(float(val))
            elif key in ('min_radiance', 'max_radiance', 'cloud_chance', 'no_cloud_chance'):
                cfg[key] = float(val)
        except ValueError:
            print(f"[warn] Could not parse '{key}={val}'. Keeping default {key}={cfg[key]}")
    if cfg['max_radiance'] <= cfg['min_radiance']:
        print("[warn] max_radiance <= min_radiance. Resetting to defaults (24, 0).")
        cfg['min_radiance'] = DEFAULTS['min_radiance']
        cfg['max_radiance'] = DEFAULTS['max_radiance']
    for key in ('cloud_chance', 'no_cloud_chance'):
        if not (0.0 <= cfg[key] <= 1.0):
            print(f"[warn] {key} out of [0,1]. Resetting to default.")
            cfg[key] = DEFAULTS[key]
    print("[info] Loaded config:", json.dumps(cfg, indent=2))
    return cfg

cfg = parse_config(CONFIG_PATH)

# ----------------------- PATHS ------------------------
# CSV of labeled intervals and parent directory of .nc files.
CSV_FILE_PATH = 'csv/cloud_intervals_cleaned_filtered_10272025.csv'
# parent_directory = r'Z:\soc\l1r'
PARENT_DIR = 'training_nc_files'

# -------------------- GPU SETTINGS -------------------
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(e)

# ------------------- IMAGE BOXES ---------------------
def define_boxes():
    # 15 boxes across (x), 3 vertically (y): total 45
    x_ranges = [(i*20, i*20+19) for i in range(15)]
    y_ranges = [(0, 99), (100, 199), (200, 299)]
    boxes = {}
    for j, y in enumerate(y_ranges):
        for i, x in enumerate(x_ranges):
            boxes[f"({i},{j})"] = {'x': x, 'y': y}
    return boxes

GRID_BOXES = define_boxes()

# ------------------- DATA HELPERS --------------------
def clear_dir(folder_path):
    """Delete all files/subfolders under folder_path."""
    if not os.path.isdir(folder_path):
        return
    for name in os.listdir(folder_path):
        p = os.path.join(folder_path, name)
        try:
            if os.path.isfile(p) or os.path.islink(p):
                os.unlink(p)
            else:
                shutil.rmtree(p)
        except Exception as e:
            print(f"[warn] Failed to delete {p}: {e}")

def extract_intervals_per_orbit(df):
    orbit_intervals = {}
    for _, row in df.iterrows():
        orbit = row.get('Orbit #', None)
        if pd.notna(orbit):
            orbit = int(orbit)
            if orbit not in orbit_intervals:
                orbit_intervals[orbit] = {}
            for col in df.columns:
                if "start" in col:
                    box = col.split("start")[0].strip()
                    end_col = f"{box}end"
                    if end_col in df.columns:
                        start = row[col]
                        end = row[end_col]
                        if pd.notna(start) and pd.notna(end):
                            orbit_intervals[orbit].setdefault(box, []).append((int(start), int(end)))
    return orbit_intervals

def find_nc_file(parent_dir, orbit_number):
    orbit_str = str(int(orbit_number)).zfill(5)
    pat = re.compile(r'awe_l1r_q20_(.*)_' + orbit_str + r'_(.*)\.nc')
    for root, _, files in os.walk(parent_dir):
        for f in files:
            if pat.match(f):
                return os.path.join(root, f)
    raise FileNotFoundError(f"No file found for orbit {orbit_str}")

def normalize_to_uint8(img, min_rad, max_rad):
    return np.clip((img - min_rad) / (max_rad - min_rad) * 255, 0, 255).astype(np.uint8)

def save_image_triplet(radiance_3d, out_png_path, frame_index, space_val, box_idx, boxes, min_rad, max_rad):
    # Make RGB from [t-space, t, t+space] frames (when available)
    curr = normalize_to_uint8(radiance_3d[frame_index], min_rad, max_rad)
    prev = normalize_to_uint8(radiance_3d[frame_index - space_val], min_rad, max_rad) if frame_index >= space_val else None
    nxt  = normalize_to_uint8(radiance_3d[frame_index + space_val], min_rad, max_rad) if frame_index + space_val < radiance_3d.shape[0] else None

    H, W = radiance_3d.shape[1], radiance_3d.shape[2]
    rgb = np.zeros((H, W, 3), dtype=np.uint8)
    if prev is not None: rgb[..., 0] = prev
    rgb[..., 1] = curr
    if nxt  is not None: rgb[..., 2] = nxt

    x0, x1 = boxes[box_idx]['x']
    y0, y1 = boxes[box_idx]['y']
    cropped = rgb[y0:y1+1, x0:x1+1]
    cv2.imwrite(out_png_path, cropped)

def generate_images_for_combo(run_img_dir, df, boxes, space_val, min_rad, max_rad, thresh, cloud_p, no_cloud_p):
    cloud_dir = os.path.join(run_img_dir, 'class_1_cloud')
    nocld_dir = os.path.join(run_img_dir, 'class_0_no_cloud')
    os.makedirs(cloud_dir, exist_ok=True)
    os.makedirs(nocld_dir, exist_ok=True)
    # clear old just in case
    clear_dir(cloud_dir); clear_dir(nocld_dir)

    orbit_intervals = extract_intervals_per_orbit(df)
    saved_cloud, saved_clear = 0, 0

    for orbit, box_map in orbit_intervals.items():
        print(f"[gen] Orbit {orbit}")
        try:
            nc_path = find_nc_file(PARENT_DIR, orbit)
        except FileNotFoundError as e:
            print(f"[gen] {e}")
            continue
        with Dataset(nc_path, 'r') as nc:
            rad = nc.variables['Radiance'][:]  # (T, Y, X)
            T = rad.shape[0]
            for box, intervals in box_map.items():
                for i in range(max(space_val, 0), T - max(space_val, 0)):
                    inside_any = any((s + thresh) <= i <= (e - thresh) for (s, e) in intervals)
                    outside_all = all(i < (s - thresh) or i > (e + thresh) for (s, e) in intervals)
                    if inside_any:
                        if random.random() < cloud_p:
                            out = os.path.join(cloud_dir, f"orbit{orbit}_box{box}_{i}.png")
                            save_image_triplet(rad, out, i, space_val, box, boxes, min_rad, max_rad)
                            saved_cloud += 1
                    elif outside_all:
                        if random.random() < no_cloud_p:
                            out = os.path.join(nocld_dir, f"orbit{orbit}_box{box}_{i}.png")
                            save_image_triplet(rad, out, i, space_val, box, boxes, min_rad, max_rad)
                            saved_clear += 1
    print(f"[gen] Saved: cloud={saved_cloud}, no_cloud={saved_clear}")
    return saved_cloud + saved_clear

# ------------------- TRAINING HELPERS ----------------
NEW_IMAGE_SIZE = (100, 20)  # (height, width)

def build_model():
    base = ResNet50(weights='imagenet', include_top=False,
                    input_tensor=Input(shape=(NEW_IMAGE_SIZE[0], NEW_IMAGE_SIZE[1], 3)))
    base.trainable = False
    x = base.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(512, activation='relu')(x)
    x = Dropout(0.5)(x)
    x = Dense(256, activation='relu')(x)
    x = Dropout(0.5)(x)
    out = Dense(1, activation='sigmoid')(x)
    m = Model(inputs=base.input, outputs=out)
    m.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy', Recall(name='recall')])
    return m

def load_dataset(run_img_dir, batch_size=32, seed=42):
    ds = tf.keras.utils.image_dataset_from_directory(
        run_img_dir,
        image_size=NEW_IMAGE_SIZE,
        color_mode='rgb',
        batch_size=batch_size,
        label_mode='int'
    )
    # augment → preprocess → shuffle (fixed reshuffle to keep split stable per run)
    ds = ds.map(lambda x, y: (tf.image.random_flip_left_right(x), y))
    ds = ds.map(lambda x, y: (tf.image.random_flip_up_down(x), y))
    ds = ds.map(lambda x, y: (preprocess_input(x), y))
    ds = ds.shuffle(buffer_size=200000, seed=seed, reshuffle_each_iteration=False)

    n = ds.cardinality().numpy()
    if n is None:
        # Fallback when unknown: convert to list to count (small sets); otherwise, skip training
        batches = list(ds)
        n = len(batches)
        ds = tf.data.Dataset.from_generator(lambda: (b for b in batches),
                                            output_signature=(
                                                tf.TensorSpec(shape=(None, *NEW_IMAGE_SIZE, 3), dtype=tf.float32),
                                                tf.TensorSpec(shape=(None,), dtype=tf.int32)
                                            ))
    if n == 0:
        return None, None, None

    n_train = int(n * 0.7)
    n_val   = int(n * 0.2)
    n_test  = max(n - n_train - n_val, 0)

    train = ds.take(n_train)
    val   = ds.skip(n_train).take(n_val)
    test  = ds.skip(n_train + n_val).take(n_test)
    return train, val, test

def evaluate_threshold_sweep(model, test_ds, out_csv):
    if test_ds is None:
        return
    y_true, y_pred = [], []
    for X, y in test_ds.as_numpy_iterator():
        yhat = model.predict(X, verbose=0)
        y_true.extend(y)
        y_pred.extend(yhat)
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    rows = []
    for t in range(0, 101):
        thr = t / 100.0
        y_bin = (y_pred >= thr).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_true, y_bin).ravel()
        acc = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) > 0 else 0.0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        rows.append({"Threshold": thr, "True_Negatives": tn, "False_Positives": fp,
                     "False_Negatives": fn, "True_Positives": tp,
                     "Accuracy": acc, "Recall": rec})
    df = pd.DataFrame(rows)
    df.to_csv(out_csv, index=False)

def plot_val_metrics(history, out_png):
    plt.figure(figsize=(10, 6))
    # available keys: history.history
    if 'val_accuracy' in history.history:
        plt.plot(history.history['val_accuracy'], label='Accuracy')
    if 'val_recall' in history.history:
        plt.plot(history.history['val_recall'], label='Recall')
    if 'val_loss' in history.history:
        plt.plot(history.history['val_loss'], label='Loss')
    plt.title('Validation Metrics')
    plt.xlabel('Epoch')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(out_png, dpi=150)
    plt.close()

# ----------------------- MAIN ------------------------
def main():
    # Fixed from config (min_rad, cloud/no_cloud draw, threshold).
    min_rad = float(cfg['min_radiance'])
    cloud_p = float(cfg['cloud_chance'])
    no_cld_p = float(cfg['no_cloud_chance'])
    thresh = int(cfg['threshold'])

    # Grid search sets:
    space_values = list(range(2, 11))                       # 
    max_rads = [20, 22, 26]                    # 

    # Load intervals CSV once
    intervals_df = pd.read_csv(CSV_FILE_PATH)

    for sp in space_values:
        for mr in max_rads:
            run_name = f"space{sp}_max{mr}"
            run_root = Path("grid_runs") / run_name
            run_imgs = run_root / "training_images"
            run_logs = run_root / "logs"
            run_models = run_root / "models"
            run_plots = run_root / "plots"
            for p in (run_imgs, run_logs, run_models, run_plots):
                p.mkdir(parents=True, exist_ok=True)

            print(f"\n================ GRID RUN: {run_name} ================\n")
            # 1) Generate images for this combo
            t0 = time.time()
            n_imgs = generate_images_for_combo(
                str(run_imgs),
                intervals_df,
                GRID_BOXES,
                space_val=int(sp),
                min_rad=min_rad,
                max_rad=float(mr),
                thresh=thresh,
                cloud_p=cloud_p,
                no_cloud_p=no_cld_p
            )
            gen_time = time.time() - t0
            if n_imgs == 0:
                print(f"[skip] No images generated for {run_name}. Skipping training.")
                continue

            # 2) Load dataset and train
            train_ds, val_ds, test_ds = load_dataset(str(run_imgs))
            if train_ds is None:
                print(f"[skip] Dataset empty for {run_name}.")
                continue

            model = build_model()
            model.summary()

            tb = TensorBoard(log_dir=str(run_logs / f"tb_{run_name}"))
            es = EarlyStopping(monitor='val_loss', patience=50, restore_best_weights=True)

            print(f"[train] Training {run_name} (epochs=100, patience=20)")
            t1 = time.time()
            history = model.fit(
                train_ds,
                validation_data=val_ds,
                epochs=250,
                callbacks=[tb, es],
                verbose=1
            )
            train_time = time.time() - t1

            # 3) Save model, plots, and threshold sweep
            model_path = run_models / f"model_{run_name}.h5"
            model.save(str(model_path))
            plot_val_metrics(history, str(run_plots / f"val_metrics_{run_name}.png"))
            evaluate_threshold_sweep(model, test_ds, str(run_root / f"threshold_sweep_{run_name}.csv"))

            # Simple run summary
            with open(run_root / "run_summary.json", "w", encoding="utf-8") as f:
                json.dump({
                    "space": sp,
                    "max_radiance": mr,
                    "min_radiance": min_rad,
                    "cloud_chance": cloud_p,
                    "no_cloud_chance": no_cld_p,
                    "threshold": thresh,
                    "images_generated": int(n_imgs),
                    "generation_seconds": round(gen_time, 2),
                    "training_seconds": round(train_time, 2),
                    "epochs_trained": len(history.history.get('loss', []))
                }, f, indent=2)

            # Free resources between grid cells
            del model
            K.clear_session()

    print("\n[done] Grid search complete.")

if __name__ == "__main__":
    main()
