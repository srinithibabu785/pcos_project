"""
================================================================================
Two-Stage Transfer Learning for DenseNet121
Project: An Explainable AI-Based Ensemble Transfer Learning Approach
         for Early PCOS Prediction and Clinical Decision Support
================================================================================
Architecture:
  Input (224x224x3)
  ↓
  DenseNet121 (ImageNet Pretrained)
  ↓
  GlobalAveragePooling2D
  ↓
  BatchNormalization
  ↓
  Dropout(0.3)
  ↓
  Dense(64, activation='relu')
  ↓
  Dropout(0.15)
  ↓
  Dense(1, activation='sigmoid')

Two-Stage Strategy:
  Stage 1: Freeze backbone, train classification head with Adam(lr=1e-3).
  Stage 2: Unfreeze top DenseBlock layers (conv5_block...), fine-tune with Adam(lr=5e-5).

Callbacks:
  - EarlyStopping(monitor='val_loss', patience=4, restore_best_weights=True)
  - ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=2)
  - ModelCheckpoint('models/DenseNet121_best.keras', save_best_only=True)
================================================================================
"""

import os
import sys
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from datetime import datetime
from typing import Tuple, Dict, Any

# Ensure project root is in sys.path
BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

import tensorflow as tf
from tensorflow.keras.applications import DenseNet121
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Dropout, BatchNormalization, Input
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

from src.preprocessing import IMG_SIZE, create_dataset, get_class_weights

# Directories
BASE_DIR = Path(__file__).resolve().parent.parent
MODELS_DIR = BASE_DIR / "models"
RESULTS_DIR = BASE_DIR / "results"
REPORTS_DIR = BASE_DIR / "reports"
MODELS_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
REPORTS_DIR.mkdir(parents=True, exist_ok=True)

def build_densenet_model(dropout_rate: float = 0.3) -> Tuple[Model, Model]:
    """
    Builds the binary classification transfer model with DenseNet121 backbone.
    """
    inputs = Input(shape=(IMG_SIZE[0], IMG_SIZE[1], 3), name="input_image")

    # Pretrained ImageNet backbone
    backbone = DenseNet121(
        include_top=False,
        weights="imagenet",
        input_tensor=inputs
    )
    # Freeze backbone for Stage 1
    backbone.trainable = False

    x = backbone.output
    x = GlobalAveragePooling2D(name="dense_gap")(x)
    x = BatchNormalization(name="dense_head_bn")(x)
    x = Dropout(dropout_rate, name="dense_head_drop1")(x)
    x = Dense(64, activation="relu", name="dense_head_dense")(x)
    x = Dropout(dropout_rate * 0.5, name="dense_head_drop2")(x)
    outputs = Dense(1, activation="sigmoid", name="dense_head_output")(x)

    model = Model(inputs=inputs, outputs=outputs, name="DenseNet121_PCOS")
    return model, backbone

def plot_presentation_curves(history_combined: dict, model_name: str = "densenet"):
    """
    Generates presentation-quality training curves for accuracy and loss.
    """
    sns.set_theme(style="whitegrid", palette="muted")
    epochs = range(1, len(history_combined["accuracy"]) + 1)

    # 1. Accuracy Curve
    plt.figure(figsize=(8, 6), dpi=300)
    plt.plot(epochs, history_combined["accuracy"], "o-", color="#1f77b4", linewidth=2.2, label="Training Accuracy")
    plt.plot(epochs, history_combined["val_accuracy"], "s--", color="#ff7f0e", linewidth=2.2, label="Validation Accuracy")
    best_acc_epoch = np.argmax(history_combined["val_accuracy"]) + 1
    best_acc = np.max(history_combined["val_accuracy"])
    plt.axvline(best_acc_epoch, color="#2ca02c", linestyle=":", alpha=0.7, label=f"Best Val Acc ({best_acc:.4f} @ Ep {best_acc_epoch})")
    plt.title(f"{model_name.capitalize()} - Training & Validation Accuracy", fontsize=14, fontweight="bold", pad=12)
    plt.xlabel("Epoch", fontsize=12, labelpad=8)
    plt.ylabel("Accuracy", fontsize=12, labelpad=8)
    plt.ylim([0.5, 1.02])
    plt.legend(loc="lower right", frameon=True, fontsize=11)
    plt.tight_layout()
    acc_path = RESULTS_DIR / f"{model_name}_accuracy.png"
    plt.savefig(acc_path, dpi=300)
    plt.close()
    print(f"Saved presentation accuracy curve to: {acc_path}")

    # 2. Loss Curve
    plt.figure(figsize=(8, 6), dpi=300)
    plt.plot(epochs, history_combined["loss"], "o-", color="#d62728", linewidth=2.2, label="Training Loss")
    plt.plot(epochs, history_combined["val_loss"], "s--", color="#9467bd", linewidth=2.2, label="Validation Loss")
    min_loss_epoch = np.argmin(history_combined["val_loss"]) + 1
    min_loss = np.min(history_combined["val_loss"])
    plt.axvline(min_loss_epoch, color="#2ca02c", linestyle=":", alpha=0.7, label=f"Min Val Loss ({min_loss:.4f} @ Ep {min_loss_epoch})")
    plt.title(f"{model_name.capitalize()} - Training & Validation Binary Cross-Entropy Loss", fontsize=14, fontweight="bold", pad=12)
    plt.xlabel("Epoch", fontsize=12, labelpad=8)
    plt.ylabel("Loss", fontsize=12, labelpad=8)
    plt.legend(loc="upper right", frameon=True, fontsize=11)
    plt.tight_layout()
    loss_path = RESULTS_DIR / f"{model_name}_loss.png"
    plt.savefig(loss_path, dpi=300)
    plt.close()
    print(f"Saved presentation loss curve to: {loss_path}")

def log_experiment(record: dict):
    """Logs experiment configuration and results to reports/experiment_results.csv."""
    csv_path = REPORTS_DIR / "experiment_results.csv"
    df_new = pd.DataFrame([record])
    if csv_path.exists():
        df_old = pd.read_csv(csv_path)
        df_combined = pd.concat([df_old, df_new], ignore_index=True)
    else:
        df_combined = df_new
    df_combined.to_csv(csv_path, index=False)
    print(f"Updated experiment log at: {csv_path}")

def train_densenet(
    batch_size: int = 32,
    epochs_stage1: int = 4,
    epochs_stage2: int = 5,
    random_seed: int = 42
):
    tf.random.set_seed(random_seed)
    np.random.seed(random_seed)

    train_csv = str(BASE_DIR / "data" / "train.csv")
    val_csv = str(BASE_DIR / "data" / "validation.csv")
    checkpoint_path = str(MODELS_DIR / "DenseNet121_best.keras")

    print("=" * 80)
    print("STARTING DENSENET-121 TWO-STAGE TRAINING PIPELINE")
    print("=" * 80)

    # 1. Datasets and Class Weights (DenseNet Preprocessing)
    class_weights = get_class_weights(train_csv)
    train_ds = create_dataset(train_csv, model_type="densenet", batch_size=batch_size, is_training=True)
    val_ds = create_dataset(val_csv, model_type="densenet", batch_size=batch_size, is_training=False)

    # 2. Build Model
    model, backbone = build_densenet_model(dropout_rate=0.3)

    # ==========================================
    # STAGE 1: Train Classification Head
    # ==========================================
    print("\n--- STAGE 1: Training Classification Head (Backbone Frozen) ---")
    model.compile(
        optimizer=Adam(learning_rate=1e-3),
        loss="binary_crossentropy",
        metrics=["accuracy", tf.keras.metrics.AUC(name="auc")]
    )

    cb_stage1 = [
        EarlyStopping(monitor="val_loss", patience=3, restore_best_weights=True, verbose=1),
        ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=2, min_lr=1e-6, verbose=1),
        ModelCheckpoint(checkpoint_path, monitor="val_loss", save_best_only=True, verbose=1)
    ]

    hist1 = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=epochs_stage1,
        class_weight=class_weights,
        callbacks=cb_stage1,
        verbose=1
    )

    # ==========================================
    # STAGE 2: Fine-Tuning Top Backbone Layers
    # ==========================================
    print("\n--- STAGE 2: Fine-Tuning Upper DenseBlock of DenseNet121 ---")
    backbone.trainable = True
    # Freeze all but top 30 layers
    for layer in backbone.layers[:-30]:
        layer.trainable = False

    trainable_count = len([l for l in backbone.layers if l.trainable])
    print(f"Backbone layers unfrozen for fine-tuning: {trainable_count} / {len(backbone.layers)}")

    model.compile(
        optimizer=Adam(learning_rate=5e-5),
        loss="binary_crossentropy",
        metrics=["accuracy", tf.keras.metrics.AUC(name="auc")]
    )

    cb_stage2 = [
        EarlyStopping(monitor="val_loss", patience=3, restore_best_weights=True, verbose=1),
        ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=2, min_lr=1e-6, verbose=1),
        ModelCheckpoint(checkpoint_path, monitor="val_loss", save_best_only=True, verbose=1)
    ]

    hist2 = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=epochs_stage2,
        class_weight=class_weights,
        callbacks=cb_stage2,
        verbose=1
    )

    # Combine History
    history_combined = {}
    for k in hist1.history.keys():
        history_combined[k] = hist1.history[k] + hist2.history[k]

    # Save presentation curves
    plot_presentation_curves(history_combined, model_name="densenet")

    # Load best checkpoint for validation evaluation
    print(f"\nLoading best checkpoint from: {checkpoint_path}")
    best_model = tf.keras.models.load_model(checkpoint_path)

    # Validation Evaluation
    val_labels = []
    val_preds_prob = []
    for x_b, y_b in val_ds:
        preds = best_model.predict(x_b, verbose=0).ravel()
        val_preds_prob.extend(preds)
        val_labels.extend(y_b.numpy().ravel())

    val_labels = np.array(val_labels)
    val_preds_prob = np.array(val_preds_prob)
    val_preds = (val_preds_prob >= 0.5).astype(int)

    val_acc = accuracy_score(val_labels, val_preds)
    val_prec = precision_score(val_labels, val_preds, zero_division=0)
    val_rec = recall_score(val_labels, val_preds, zero_division=0)
    val_f1 = f1_score(val_labels, val_preds, zero_division=0)
    val_auc = roc_auc_score(val_labels, val_preds_prob)

    print("\n" + "=" * 80)
    print("DENSENET-121 VALIDATION PERFORMANCE (Threshold = 0.50)")
    print("=" * 80)
    print(f"Validation Accuracy : {val_acc * 100:.2f}%")
    print(f"Validation Precision: {val_prec * 100:.2f}%")
    print(f"Validation Recall   : {val_rec * 100:.2f}%")
    print(f"Validation F1-Score : {val_f1 * 100:.2f}%")
    print(f"Validation ROC-AUC  : {val_auc:.4f}")
    print("=" * 80)

    # Log to experiment_results.csv
    exp_record = {
        "Experiment_ID": f"EXP_DENSE_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
        "Model": "DenseNet121_PCOS",
        "Dataset_version": "v1.0_clean_dedup",
        "Image_size": "224x224",
        "Batch_size": batch_size,
        "Learning_rate": "1e-3 (S1) -> 5e-5 (S2)",
        "Frozen_layers": len(backbone.layers) - trainable_count,
        "Fine_tuning_layers": trainable_count,
        "Augmentation": "Rotation(0.03)+Zoom(0.05)+Trans(0.03)+Brightness(0.05)+Contrast(0.05)",
        "Class_weights": str(class_weights),
        "Epochs": len(history_combined["accuracy"]),
        "Best_epoch": int(np.argmin(history_combined["val_loss"]) + 1),
        "Validation_accuracy": round(val_acc, 4),
        "Validation_F1": round(val_f1, 4),
        "Validation_AUC": round(val_auc, 4),
        "Test_accuracy": "Pending Final Evaluation",
        "Test_precision": "Pending Final Evaluation",
        "Test_recall": "Pending Final Evaluation",
        "Test_specificity": "Pending Final Evaluation",
        "Test_F1": "Pending Final Evaluation",
        "Test_AUC": "Pending Final Evaluation",
        "Notes": "Two-stage transfer learning with dynamic class weighting. Test set strictly isolated."
    }
    log_experiment(exp_record)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train DenseNet121 for PCOS Prediction")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size for training")
    parser.add_argument("--epochs_stage1", type=int, default=4, help="Stage 1 frozen epochs")
    parser.add_argument("--epochs_stage2", type=int, default=5, help="Stage 2 fine-tuning epochs")
    args = parser.parse_args()
    train_densenet(
        batch_size=args.batch_size,
        epochs_stage1=args.epochs_stage1,
        epochs_stage2=args.epochs_stage2
    )
