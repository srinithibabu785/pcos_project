"""
Common Evaluation, Visualization, and Diagnostic Utilities for PCOS AI Project.
"""

import json
from pathlib import Path
from typing import Dict, List, Tuple, Any
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score,
    confusion_matrix, roc_curve, brier_score_loss
)
from sklearn.calibration import calibration_curve

def compute_detailed_metrics(y_true: np.ndarray, y_prob: np.ndarray, threshold: float = 0.5) -> Dict[str, Any]:
    """Computes clinical and statistical classification metrics."""
    y_pred = (y_prob >= threshold).astype(int)
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = int(cm[0, 0]), int(cm[0, 1]), int(cm[1, 0]), int(cm[1, 1])
    
    acc = float(accuracy_score(y_true, y_pred))
    prec = float(precision_score(y_true, y_pred, zero_division=0))
    rec = float(recall_score(y_true, y_pred, zero_division=0))
    f1 = float(f1_score(y_true, y_pred, zero_division=0))
    auc = float(roc_auc_score(y_true, y_prob)) if len(np.unique(y_true)) > 1 else 0.5
    spec = float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0
    brier = float(brier_score_loss(y_true, y_prob))

    return {
        "accuracy": acc,
        "precision": prec,
        "recall": rec,
        "specificity": spec,
        "f1": f1,
        "auc": auc,
        "brier_score": brier,
        "decision_threshold": float(threshold),
        "confusion_matrix": {
            "tn": tn,
            "fp": fp,
            "fn": fn,
            "tp": tp
        }
    }

def plot_training_curves(history: Dict[str, List[float]], model_name: str, save_path: Path):
    """
    Plots training vs validation accuracy and loss curves.
    Detects and displays underfitting, overfitting, or good fit diagnosis.
    """
    save_path.parent.mkdir(parents=True, exist_ok=True)
    epochs = range(1, len(history["loss"]) + 1)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Loss plot
    axes[0].plot(epochs, history["loss"], 'b-o', label="Training Loss")
    if "val_loss" in history and len(history["val_loss"]) == len(epochs):
        axes[0].plot(epochs, history["val_loss"], 'r-s', label="Validation Loss")
    axes[0].set_title(f"{model_name} — Loss Curve", fontsize=12, fontweight='bold')
    axes[0].set_xlabel("Epochs")
    axes[0].set_ylabel("Binary Crossentropy Loss")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Accuracy / AUC plot
    acc_key = "accuracy" if "accuracy" in history else list(history.keys())[0]
    axes[1].plot(epochs, history[acc_key], 'b-o', label=f"Training {acc_key.capitalize()}")
    val_acc_key = f"val_{acc_key}"
    if val_acc_key in history and len(history[val_acc_key]) == len(epochs):
        axes[1].plot(epochs, history[val_acc_key], 'r-s', label=f"Validation {acc_key.capitalize()}")
    axes[1].set_title(f"{model_name} — Accuracy/Metric Curve", fontsize=12, fontweight='bold')
    axes[1].set_xlabel("Epochs")
    axes[1].set_ylabel("Metric Score")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()

def diagnose_fit(history: Dict[str, List[float]]) -> str:
    """
    Diagnoses model fit from training history.
    - UNDERFITTING: Both train and val performance remain low (<0.65).
    - OVERFITTING: Train improves while val deteriorates (val_loss increases or val_acc drops).
    - GOOD FIT: Train and val improve concurrently with reasonable generalization gap.
    """
    train_loss = history.get("loss", [])
    val_loss = history.get("val_loss", [])
    train_acc = history.get("accuracy", [])
    val_acc = history.get("val_accuracy", [])

    if not train_loss or not val_loss:
        return "INCONCLUSIVE (insufficient history)"

    final_train_acc = train_acc[-1] if train_acc else 0.0
    final_val_acc = val_acc[-1] if val_acc else 0.0
    
    if final_train_acc < 0.65 and final_val_acc < 0.65:
        return "UNDERFITTING"
    elif (final_train_acc - final_val_acc) > 0.15 or (val_loss[-1] > val_loss[0] and final_train_acc > 0.85):
        return "OVERFITTING"
    else:
        return "GOOD FIT"

def plot_confusion_matrices(
    cm_dict: Dict[str, Dict[str, int]],
    save_path: Path,
    class_names: List[str] = ["Healthy", "PCOS"]
):
    """Plots side-by-side confusion matrices for EfficientNet, DenseNet, and Ensemble."""
    save_path.parent.mkdir(parents=True, exist_ok=True)
    num_models = len(cm_dict)
    fig, axes = plt.subplots(1, num_models, figsize=(6 * num_models, 5))
    if num_models == 1:
        axes = [axes]

    for ax, (model_name, cm_vals) in zip(axes, cm_dict.items()):
        matrix = np.array([
            [cm_vals["tn"], cm_vals["fp"]],
            [cm_vals["fn"], cm_vals["tp"]]
        ])
        sns.heatmap(
            matrix, annot=True, fmt="d", cmap="Blues", cbar=False, ax=ax,
            xticklabels=class_names, yticklabels=class_names,
            annot_kws={"size": 14, "weight": "bold"}
        )
        ax.set_title(f"{model_name}\nConfusion Matrix", fontsize=13, fontweight="bold")
        ax.set_xlabel("Predicted Class", fontsize=11)
        ax.set_ylabel("True Class", fontsize=11)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()

def plot_roc_curves(
    roc_data: Dict[str, Tuple[np.ndarray, np.ndarray, float]],
    save_path: Path
):
    """Plots comparative ROC curves."""
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(8, 6))
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c"]

    for (name, (fpr, tpr, auc)), color in zip(roc_data.items(), colors):
        plt.plot(fpr, tpr, color=color, lw=2.5, label=f"{name} (AUC = {auc:.4f})")

    plt.plot([0, 1], [0, 1], color="gray", lw=1.5, linestyle="--", label="Random Chance (AUC = 0.5000)")
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate (1 - Specificity)", fontsize=12)
    plt.ylabel("True Positive Rate (Sensitivity / Recall)", fontsize=12)
    plt.title("ROC Curves Comparison on Independent Test Set", fontsize=14, fontweight="bold")
    plt.legend(loc="lower right", fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
