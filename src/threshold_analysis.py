"""
================================================================================
Validation-Only Threshold Analysis Script
Project: An Explainable AI-Based Ensemble Transfer Learning Approach
         for Early PCOS Prediction and Clinical Decision Support
================================================================================
Examines decision thresholds from 0.20 to 0.80 strictly on the VALIDATION SET.
Evaluates Accuracy, Precision, Recall/Sensitivity, Specificity, and F1-score.
Selects the optimal threshold to balance sensitivity & specificity for clinical
decision support without touching the test set.

Outputs:
  - reports/validation_threshold_analysis.csv
  - reports/optimal_threshold.json
================================================================================
"""

import os
import sys
import json
import argparse
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, Any

# Ensure project root is in sys.path
BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

import tensorflow as tf
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score, accuracy_score

from src.preprocessing import create_dataset

MODELS_DIR = BASE_DIR / "models"
REPORTS_DIR = BASE_DIR / "reports"
REPORTS_DIR.mkdir(parents=True, exist_ok=True)

def evaluate_threshold_metrics(y_true: np.ndarray, y_prob: np.ndarray, threshold: float) -> Dict[str, float]:
    y_pred = (y_prob >= threshold).astype(int)
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()

    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    spec = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    f1 = f1_score(y_true, y_pred, zero_division=0)
    bal_acc = (rec + spec) / 2.0

    return {
        "threshold": round(threshold, 2),
        "accuracy": round(acc, 4),
        "precision": round(prec, 4),
        "recall_sensitivity": round(rec, 4),
        "specificity": round(spec, 4),
        "f1_score": round(f1, 4),
        "balanced_accuracy": round(bal_acc, 4),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp)
    }

def run_threshold_analysis(
    val_csv: str = "data/validation.csv",
    batch_size: int = 32
):
    val_csv_path = BASE_DIR / val_csv
    eff_model_path = MODELS_DIR / "EfficientNetB0_best.keras"
    dense_model_path = MODELS_DIR / "DenseNet121_best.keras"

    print("=" * 80)
    print("STARTING VALIDATION-ONLY THRESHOLD OPTIMIZATION")
    print("=" * 80)
    print(f"Validation Manifest: {val_csv_path}")

    # Load validation datasets
    val_ds_eff = create_dataset(str(val_csv_path), model_type="efficientnet", batch_size=batch_size, is_training=False)
    val_ds_dense = create_dataset(str(val_csv_path), model_type="densenet", batch_size=batch_size, is_training=False)

    # Get ground truth labels
    val_df = pd.read_csv(val_csv_path)
    y_true = val_df["label"].values.astype(int)

    # Load models
    print(f"Loading EfficientNet checkpoint: {eff_model_path}")
    eff_model = tf.keras.models.load_model(str(eff_model_path))

    print(f"Loading DenseNet checkpoint: {dense_model_path}")
    dense_model = tf.keras.models.load_model(str(dense_model_path))

    # Generate probabilities on validation set
    print("Generating validation probabilities...")
    eff_probs = eff_model.predict(val_ds_eff, verbose=0).ravel()
    dense_probs = dense_model.predict(val_ds_dense, verbose=0).ravel()
    ensemble_probs = (eff_probs + dense_probs) / 2.0

    thresholds = [0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80]

    all_results = []
    print("\n--- Sweeping Thresholds for Soft-Voting Ensemble on Validation Set ---")
    print(f"{'Thresh':<8}{'Accuracy':<10}{'Precision':<11}{'Sensitivity':<13}{'Specificity':<13}{'F1-Score':<10}{'Bal Acc':<10}")
    print("-" * 75)

    best_thresh = 0.50
    best_score = -1.0

    for t in thresholds:
        metrics = evaluate_threshold_metrics(y_true, ensemble_probs, t)
        all_results.append(metrics)
        print(f"{metrics['threshold']:<8.2f}{metrics['accuracy']*100:<10.2f}%{metrics['precision']*100:<11.2f}%{metrics['recall_sensitivity']*100:<13.2f}%{metrics['specificity']*100:<13.2f}%{metrics['f1_score']:<10.4f}{metrics['balanced_accuracy']:<10.4f}")

        # Optimization criterion: maximize Balanced Accuracy (Sensitivity + Specificity) / 2
        # to ensure good clinical balance in screening
        score = metrics["balanced_accuracy"]
        if score > best_score:
            best_score = score
            best_thresh = t

    # Save CSV
    df_out = pd.DataFrame(all_results)
    out_csv = REPORTS_DIR / "validation_threshold_analysis.csv"
    df_out.to_csv(out_csv, index=False)
    print(f"\nSaved threshold analysis table to: {out_csv}")

    best_metrics = [m for m in all_results if m["threshold"] == best_thresh][0]
    print("\n" + "=" * 80)
    print("OPTIMAL VALIDATION THRESHOLD SELECTION")
    print("=" * 80)
    print(f"Optimal Operating Threshold : {best_thresh:.2f}")
    print(f"Selection Criterion         : Maximum Balanced Accuracy (avg of Sensitivity & Specificity)")
    print(f"Validation Accuracy         : {best_metrics['accuracy'] * 100:.2f}%")
    print(f"Validation Sensitivity      : {best_metrics['recall_sensitivity'] * 100:.2f}%")
    print(f"Validation Specificity      : {best_metrics['specificity'] * 100:.2f}%")
    print(f"Validation F1-Score         : {best_metrics['f1_score']:.4f}")
    print("=" * 80)

    # Save to optimal_threshold.json
    out_json = REPORTS_DIR / "optimal_threshold.json"
    with open(out_json, "w") as f:
        json.dump({
            "optimal_threshold": best_thresh,
            "criterion": "Balanced Accuracy on Validation Set",
            "validation_metrics": best_metrics
        }, f, indent=2)
    print(f"Saved threshold metadata to: {out_json}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Optimize Classification Threshold on Validation Set")
    parser.add_argument("--val_csv", type=str, default="data/validation.csv", help="Path to validation CSV")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
    args = parser.parse_args()
    run_threshold_analysis(val_csv=args.val_csv, batch_size=args.batch_size)
