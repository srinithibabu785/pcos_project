"""
================================================================================
Image Preprocessing and tf.data Pipeline for PCOS Ultrasound Classification
Project: An Explainable AI-Based Ensemble Transfer Learning Approach
         for Early PCOS Prediction and Clinical Decision Support
================================================================================
Features:
- Architecture-specific input scaling (EfficientNetB0 vs DenseNet121)
- Medically sensible data augmentation strictly on training data
- Validation and Test datasets remain 100% deterministic and un-augmented
- High-performance tf.data pipeline with parallel loading and prefetching
- Dynamic inverse class weight calculation for binary cross-entropy loss
================================================================================
"""

import os
import pandas as pd
import numpy as np
import tensorflow as tf
from pathlib import Path
from typing import Tuple, Dict, Any

IMG_SIZE = (224, 224)
AUTOTUNE = tf.data.AUTOTUNE

def get_class_weights(csv_path: str) -> Dict[int, float]:
    """
    Computes balanced class weights dynamically:
      w0 = N / (2 * N0)
      w1 = N / (2 * N1)
    """
    df = pd.read_csv(csv_path)
    total = len(df)
    n0 = (df["label"] == 0).sum()
    n1 = (df["label"] == 1).sum()

    w0 = total / (2.0 * n0) if n0 > 0 else 1.0
    w1 = total / (2.0 * n1) if n1 > 0 else 1.0

    print(f"Class Distribution: Normal (0) = {n0:,} ({n0/total*100:.1f}%), PCOS (1) = {n1:,} ({n1/total*100:.1f}%)")
    print(f"Computed Balanced Class Weights: Normal (0): {w0:.4f}, PCOS (1): {w1:.4f}")
    return {0: float(w0), 1: float(w1)}

class AugmentationPipeline(tf.keras.layers.Layer):
    """
    Medically conservative ultrasound image augmentation:
    - Small rotation (+/- 10 deg)
    - Small zoom (+/- 5%)
    - Small translation (+/- 3%)
    - Small brightness adjustment (+/- 5%)
    - Small contrast adjustment (+/- 5%)
    - Subtle horizontal flip
    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.rot = tf.keras.layers.RandomRotation(0.03, fill_mode="nearest")
        self.zoom = tf.keras.layers.RandomZoom(0.05, fill_mode="nearest")
        self.trans = tf.keras.layers.RandomTranslation(0.03, 0.03, fill_mode="nearest")
        self.flip = tf.keras.layers.RandomFlip("horizontal")
        self.bright = tf.keras.layers.RandomBrightness(0.05)
        self.contrast = tf.keras.layers.RandomContrast(0.05)

    def call(self, images, training=True):
        if not training:
            return images
        x = self.rot(images)
        x = self.zoom(x)
        x = self.trans(x)
        x = self.flip(x)
        x = self.bright(x)
        x = self.contrast(x)
        return x

def preprocess_image_path(image_path: tf.Tensor, label: tf.Tensor, model_type: str = "efficientnet") -> Tuple[tf.Tensor, tf.Tensor]:
    """
    Reads an image from disk, decodes JPEG to RGB, resizes to (224, 224),
    and applies architecture-specific preprocessing.
    """
    img_bytes = tf.io.read_file(image_path)
    img = tf.image.decode_jpeg(img_bytes, channels=3)
    img = tf.image.resize(img, IMG_SIZE)
    img = tf.cast(img, tf.float32)

    # Architecture-specific scaling
    if model_type.lower() == "efficientnet":
        # EfficientNet expects inputs in [0, 255] or uses tf.keras.applications.efficientnet.preprocess_input
        img = tf.keras.applications.efficientnet.preprocess_input(img)
    elif model_type.lower() == "densenet":
        # DenseNet expects inputs scaled using tf.keras.applications.densenet.preprocess_input
        img = tf.keras.applications.densenet.preprocess_input(img)
    else:
        # Standard normalization [0, 1]
        img = img / 255.0

    return img, tf.cast(label, tf.float32)

def create_dataset(
    csv_path: str,
    model_type: str = "efficientnet",
    batch_size: int = 32,
    is_training: bool = False,
    shuffle_buffer: int = 1024
) -> tf.data.Dataset:
    """
    Constructs a highly efficient tf.data pipeline.
    """
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"CSV manifest not found: {csv_path}")

    df = pd.read_csv(csv_path)
    paths = df["image_path"].values
    labels = df["label"].values.astype(np.float32)

    ds = tf.data.Dataset.from_tensor_slices((paths, labels))

    if is_training:
        ds = ds.shuffle(buffer_size=min(len(df), shuffle_buffer), seed=42)

    # Parallel image loading & architecture-specific preprocessing
    ds = ds.map(
        lambda path, lbl: preprocess_image_path(path, lbl, model_type=model_type),
        num_parallel_calls=AUTOTUNE
    )

    # Batching
    ds = ds.batch(batch_size)

    # Training-only medical augmentation
    if is_training:
        aug_layer = AugmentationPipeline()
        ds = ds.map(lambda x, y: (aug_layer(x, training=True), y), num_parallel_calls=AUTOTUNE)

    ds = ds.prefetch(buffer_size=AUTOTUNE)
    return ds
