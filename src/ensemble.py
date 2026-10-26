"""
================================================================================
Soft-Voting Ensemble & Final Untouched Test Evaluation
Project: An Explainable AI-Based Ensemble Transfer Learning Approach
         for Early PCOS Prediction and Clinical Decision Support
================================================================================
Combines predictions from EfficientNetB0 and DenseNet121:
  P_ensemble = 0.5 * P_eff + 0.5 * P_dense

Features:
- Evaluates the untouched Test Set (data/test.csv) ONCE
- Applies the optimal validation threshold derived strictly from validation data
- Computes Accuracy, Precision, Sensitivity, Specificity, F1-Score, ROC-AUC, Brier Score
- Generates publication/presentation quality figures:
  - results/ensemble_confusion_matrix.png
  - results/ensemble_roc_curve.png
  - results/ensemble_calibration_curve.png
- Generates final comparative benchmark: reports/final_results.csv
================================================================================
"""

import os
import sys
import json
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, Tuple

# Ensure project root is in sys.path
BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

import tensorflow as tf
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, roc_curve, confusion_matrix, brier_score_loss
)
from sklearn.calibration import calibration_curve

from src.preprocessing import create_dataset

MODELS_DIR = BASE_DIR / "models"
RESULTS_DIR = BASE_DIR / "results"
REPORTS_DIR = BASE_DIR / "reports"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
REPORTS_DIR.mkdir(parents=True, exist_ok=True)

def compute_model_metrics(y_true: np.ndarray, y_prob: np.ndarray, threshold: float = 0.5) -> Dict[str, float]:
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

    return {
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

def plot_confusion_matrix_annotated(cm: np.ndarray, model_name: str, save_path: Path):
    """Plots presentation-grade confusion matrix with TP, TN, FP, FN cell annotations."""
    plt.figure(figsize=(7, 6), dpi=300)
    labels = [["TN", "FP"], ["FN", "TP"]]
    annot_text = [[f"{labels[i][j]}\n{cm[i, j]:,}" for j in range(2)] for i in range(2)]

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
    plt.title(f"{model_name} - Confusion Matrix", fontsize=14, fontweight="bold", pad=12)
    plt.xlabel("AI Predicted Class", fontsize=12, labelpad=8)
    plt.ylabel("Ground Truth Class", fontsize=12, labelpad=8)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"Saved confusion matrix to: {save_path}")

def plot_multi_roc_curve(roc_data: list, save_path: Path):
    """Plots comparative ROC curves for individual models and ensemble."""
    plt.figure(figsize=(8, 6), dpi=300)
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c"]

    for idx, (name, fpr, tpr, auc) in enumerate(roc_data):
        plt.plot(fpr, tpr, color=colors[idx % len(colors)], linewidth=2.2, label=f"{name} (AUC = {auc:.4f})")

    plt.plot([0, 1], [0, 1], "k--", alpha=0.6, label="Chance Level (AUC = 0.50)")
    plt.title("ROC Curves - Model Comparison on Untouched Test Set", fontsize=14, fontweight="bold", pad=12)
    plt.xlabel("False Positive Rate (1 - Specificity)", fontsize=12, labelpad=8)
    plt.ylabel("True Positive Rate (Sensitivity / Recall)", fontsize=12, labelpad=8)
    plt.xlim([-0.02, 1.02])
    plt.ylim([-0.02, 1.02])
    plt.legend(loc="lower right", frameon=True, fontsize=11)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"Saved multi-ROC curve to: {save_path}")

def plot_calibration(y_true: np.ndarray, y_prob: np.ndarray, save_path: Path):
    """Plots probability calibration curve and reliability diagram."""
    prob_true, prob_pred = calibration_curve(y_true, y_prob, n_bins=10, strategy="uniform")
    plt.figure(figsize=(7, 6), dpi=300)
    plt.plot([0, 1], [0, 1], "k--", alpha=0.6, label="Perfect Calibration")
    plt.plot(prob_pred, prob_true, "s-", color="#9467bd", linewidth=2.2, label="Ensemble Output")
    plt.title("AI Risk Score Calibration Curve (Reliability Diagram)", fontsize=14, fontweight="bold", pad=12)
    plt.xlabel("Mean Predicted PCOS Likelihood Score", fontsize=12, labelpad=8)
    plt.ylabel("Fraction of True Positives", fontsize=12, labelpad=8)
    plt.legend(loc="upper left", frameon=True, fontsize=11)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"Saved calibration curve to: {save_path}")

def evaluate_ensemble(
    test_csv: str = "data/test.csv",
    batch_size: int = 32
):
    test_csv_path = BASE_DIR / test_csv
    eff_path = MODELS_DIR / "EfficientNetB0_best.keras"
    dense_path = MODELS_DIR / "DenseNet121_best.keras"

    print("=" * 80)
    print("FINAL BENCHMARK EVALUATION ON UNTOUCHED TEST SET")
    print("=" * 80)
    print(f"Test Manifest: {test_csv_path}")

    # Read optimal threshold determined on validation set
    thresh_json = REPORTS_DIR / "optimal_threshold.json"
    if thresh_json.exists():
        with open(thresh_json, "r") as f:
            data = json.load(f)
            chosen_threshold = data.get("optimal_threshold", 0.50)
            print(f"Applying validation-optimized threshold: {chosen_threshold:.2f}")
    else:
        chosen_threshold = 0.50
        print(f"Warning: optimal_threshold.json not found. Using default threshold 0.50")

    # Load test datasets
    test_ds_eff = create_dataset(str(test_csv_path), model_type="efficientnet", batch_size=batch_size, is_training=False)
    test_ds_dense = create_dataset(str(test_csv_path), model_type="densenet", batch_size=batch_size, is_training=False)

    df_test = pd.read_csv(test_csv_path)
    y_true = df_test["label"].values.astype(int)

    # Load models
    print(f"\nLoading EfficientNetB0: {eff_path}")
    eff_model = tf.keras.models.load_model(str(eff_path))

    print(f"Loading DenseNet121: {dense_path}")
    dense_model = tf.keras.models.load_model(str(dense_path))

    # Generate probabilities
    print("\nGenerating model probabilities on test set...")
    eff_probs = eff_model.predict(test_ds_eff, verbose=0).ravel()
    dense_probs = dense_model.predict(test_ds_dense, verbose=0).ravel()
    ensemble_probs = 0.5 * eff_probs + 0.5 * dense_probs

    # Evaluate all three configurations
    m_eff = compute_model_metrics(y_true, eff_probs, threshold=chosen_threshold)
    m_dense = compute_model_metrics(y_true, dense_probs, threshold=chosen_threshold)
    m_ens = compute_model_metrics(y_true, ensemble_probs, threshold=chosen_threshold)

    # Save individual confusion matrices
    cm_eff = confusion_matrix(y_true, (eff_probs >= chosen_threshold).astype(int), labels=[0, 1])
    cm_dense = confusion_matrix(y_true, (dense_probs >= chosen_threshold).astype(int), labels=[0, 1])
    cm_ens = confusion_matrix(y_true, (ensemble_probs >= chosen_threshold).astype(int), labels=[0, 1])

    plot_confusion_matrix_annotated(cm_eff, "EfficientNetB0", RESULTS_DIR / "efficientnet_confusion_matrix.png")
    plot_confusion_matrix_annotated(cm_dense, "DenseNet121", RESULTS_DIR / "densenet_confusion_matrix.png")
    plot_confusion_matrix_annotated(cm_ens, "Ensemble (EfficientNet + DenseNet)", RESULTS_DIR / "ensemble_confusion_matrix.png")

    # Multi-ROC Curve
    fpr_eff, tpr_eff, _ = roc_curve(y_true, eff_probs)
    fpr_dense, tpr_dense, _ = roc_curve(y_true, dense_probs)
    fpr_ens, tpr_ens, _ = roc_curve(y_true, ensemble_probs)

    roc_data = [
        ("EfficientNetB0", fpr_eff, tpr_eff, m_eff["ROC-AUC"]),
        ("DenseNet121", fpr_dense, tpr_dense, m_dense["ROC-AUC"]),
        ("Soft-Voting Ensemble", fpr_ens, tpr_ens, m_ens["ROC-AUC"])
    ]
    plot_multi_roc_curve(roc_data, RESULTS_DIR / "ensemble_roc_curve.png")

    # Calibration Curve
    plot_calibration(y_true, ensemble_probs, RESULTS_DIR / "ensemble_calibration_curve.png")

    # Comparative Table
    results_rows = [
        {"Model": "EfficientNetB0", **m_eff},
        {"Model": "DenseNet121", **m_dense},
        {"Model": "Ensemble (Soft-Voting)", **m_ens}
    ]
    df_results = pd.DataFrame(results_rows)
    final_csv = REPORTS_DIR / "final_results.csv"
    df_results.to_csv(final_csv, index=False)
    print(f"\nSaved final benchmark table to: {final_csv}")

    # Print table to console
    print("\n" + "=" * 95)
    print("FINAL UNTOUCHED TEST SET BENCHMARK RESULTS")
    print("=" * 95)
    print(f"{'Model':<25}{'Accuracy':<11}{'Precision':<11}{'Recall':<10}{'Specificity':<13}{'F1':<10}{'ROC-AUC':<10}{'Brier':<10}")
    print("-" * 95)
    for row in results_rows:
        print(f"{row['Model']:<25}{row['Accuracy']*100:<11.2f}%{row['Precision']*100:<11.2f}%{row['Recall']*100:<10.2f}%{row['Specificity']*100:<13.2f}%{row['F1']:<10.4f}{row['ROC-AUC']:<10.4f}{row['Brier_Score']:<10.4f}")
    print("=" * 95)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate Ensemble on Untouched Test Set")
    parser.add_argument("--test_csv", type=str, default="data/test.csv", help="Path to test CSV")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
    args = parser.parse_args()
    evaluate_ensemble(test_csv=args.test_csv, batch_size=args.batch_size)
