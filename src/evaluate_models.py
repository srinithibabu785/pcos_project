"""
================================================================================
Standalone Model Evaluation Module for PCOS Prediction
Project: An Explainable AI-Based Ensemble Transfer Learning Approach
         for Early PCOS Prediction and Clinical Decision Support
================================================================================
Evaluates saved .keras models on validation or test sets.
Calculates:
  - Accuracy
  - Precision
  - Recall / Sensitivity
  - Specificity
  - F1-score
  - ROC-AUC
  - Confusion Matrix (TN, FP, FN, TP)
  - Brier Score
Generates annotated confusion matrix plots and ROC curves.
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
from typing import Dict

# Ensure project root is in sys.path
BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

import tensorflow as tf
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, roc_curve, confusion_matrix, brier_score_loss
)

from src.preprocessing import create_dataset

MODELS_DIR = BASE_DIR / "models"
RESULTS_DIR = BASE_DIR / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

def evaluate_model_on_manifest(
    model_path: Path,
    manifest_csv: Path,
    model_type: str = "efficientnet",
    threshold: float = 0.50,
    batch_size: int = 32
) -> Dict[str, float]:
    print(f"\nEvaluating Model: {model_path.name}")
    print(f"Manifest: {manifest_csv.name} | Architecture: {model_type} | Threshold: {threshold:.2f}")

    model = tf.keras.models.load_model(str(model_path))
    ds = create_dataset(str(manifest_csv), model_type=model_type, batch_size=batch_size, is_training=False)

    df = pd.read_csv(manifest_csv)
    y_true = df["label"].values.astype(int)

    y_prob = model.predict(ds, verbose=0).ravel()
    y_pred = (y_prob >= threshold).astype(int)

    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()

    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    spec = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    f1 = f1_score(y_true, y_pred, zero_division=0)
    auc = roc_auc_score(y_true, y_prob)
    brier = brier_score_loss(y_true, y_prob)

    metrics = {
        "Accuracy": round(acc, 4),
        "Precision": round(prec, 4),
        "Recall": round(rec, 4),
        "Specificity": round(spec, 4),
        "F1": round(f1, 4),
        "ROC-AUC": round(auc, 4),
        "Brier_Score": round(brier, 4),
        "TN": int(tn),
        "FP": int(fp),
        "FN": int(fn),
        "TP": int(tp)
    }

    print(f"Accuracy   : {acc * 100:.2f}%")
    print(f"Precision  : {prec * 100:.2f}%")
    print(f"Recall/Sens: {rec * 100:.2f}%")
    print(f"Specificity: {spec * 100:.2f}%")
    print(f"F1-Score   : {f1:.4f}")
    print(f"ROC-AUC    : {auc:.4f}")
    print(f"Confusion Matrix: TN={tn}, FP={fp}, FN={fn}, TP={tp}")

    # Plot confusion matrix
    cm_path = RESULTS_DIR / f"{model_type}_confusion_matrix.png"
    plt.figure(figsize=(7, 6), dpi=300)
    annot_text = [[f"TN\n{tn:,}", f"FP\n{fp:,}"], [f"FN\n{fn:,}", f"TP\n{tp:,}"]]
    sns.heatmap(
        cm,
        annot=annot_text,
        fmt="",
        cmap="Blues",
        cbar=True,
        annot_kws={"fontsize": 14, "fontweight": "bold"},
        xticklabels=["Normal (0)", "PCOS (1)"],
        yticklabels=["Normal (0)", "PCOS (1)"]
    )
    plt.title(f"{model_path.stem} - Confusion Matrix", fontsize=14, fontweight="bold", pad=12)
    plt.xlabel("Predicted", fontsize=12, labelpad=8)
    plt.ylabel("Ground Truth", fontsize=12, labelpad=8)
    plt.tight_layout()
    plt.savefig(cm_path, dpi=300)
    plt.close()
    print(f"Saved confusion matrix plot to: {cm_path}")

    return metrics

def run_evaluation(manifest_csv: str = "data/test.csv", threshold: float = 0.50):
    manifest_path = BASE_DIR / manifest_csv
    eff_path = MODELS_DIR / "EfficientNetB0_best.keras"
    dense_path = MODELS_DIR / "DenseNet121_best.keras"

    if eff_path.exists():
        evaluate_model_on_manifest(eff_path, manifest_path, model_type="efficientnet", threshold=threshold)
    else:
        print(f"Notice: {eff_path} not found.")

    if dense_path.exists():
        evaluate_model_on_manifest(dense_path, manifest_path, model_type="densenet", threshold=threshold)
    else:
        print(f"Notice: {dense_path} not found.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate Trained PCOS Models")
    parser.add_argument("--manifest_csv", type=str, default="data/test.csv", help="Path to manifest CSV")
    parser.add_argument("--threshold", type=float, default=0.50, help="Classification threshold")
    args = parser.parse_args()
    run_evaluation(manifest_csv=args.manifest_csv, threshold=args.threshold)
