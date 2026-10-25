"""
================================================================================
Interactive Results Viewer for Project Guide Presentation
Project: An Explainable AI-Based Ensemble Transfer Learning Approach
         for Early PCOS Prediction and Clinical Decision Support
================================================================================
Run this script inside VS Code terminal or PowerShell:
    python show_results.py

It displays a presentation dashboard showing:
1. Ensemble Confusion Matrix (TN, FP, FN, TP)
2. EfficientNetB0 Training & Validation Accuracy Curve
3. DenseNet121 Training & Validation Accuracy Curve
4. Comparative Model Performance Table
================================================================================
"""

import os
import sys
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

BASE_DIR = Path(__file__).resolve().parent
RESULTS_DIR = BASE_DIR / "results"

cm_path = RESULTS_DIR / "ensemble_confusion_matrix.png"
eff_acc_path = RESULTS_DIR / "efficientnet_accuracy.png"
dense_acc_path = RESULTS_DIR / "densenet_accuracy.png"
roc_path = RESULTS_DIR / "ensemble_roc_curve.png"

def display_dashboard():
    print("=" * 80)
    print("OPENING PRESENTATION DASHBOARD...")
    print("=" * 80)

    fig = plt.figure(figsize=(16, 9), dpi=150)
    fig.canvas.manager.set_window_title("PCOS Project - Guide Presentation Dashboard")

    # 1. Confusion Matrix
    ax1 = plt.subplot(2, 2, 1)
    if cm_path.exists():
        img_cm = mpimg.imread(str(cm_path))
        ax1.imshow(img_cm)
    ax1.axis("off")
    ax1.set_title("1. Soft-Voting Ensemble Confusion Matrix (Test N=594)", fontsize=13, fontweight="bold", pad=8)

    # 2. ROC Curve
    ax2 = plt.subplot(2, 2, 2)
    if roc_path.exists():
        img_roc = mpimg.imread(str(roc_path))
        ax2.imshow(img_roc)
    ax2.axis("off")
    ax2.set_title("2. Comparative ROC Curves (AUC = 0.9958)", fontsize=13, fontweight="bold", pad=8)

    # 3. EfficientNet Accuracy
    ax3 = plt.subplot(2, 2, 3)
    if eff_acc_path.exists():
        img_eff = mpimg.imread(str(eff_acc_path))
        ax3.imshow(img_eff)
    ax3.axis("off")
    ax3.set_title("3. EfficientNetB0 Two-Stage Training Accuracy", fontsize=13, fontweight="bold", pad=8)

    # 4. DenseNet Accuracy
    ax4 = plt.subplot(2, 2, 4)
    if dense_acc_path.exists():
        img_dense = mpimg.imread(str(dense_acc_path))
        ax4.imshow(img_dense)
    ax4.axis("off")
    ax4.set_title("4. DenseNet121 Two-Stage Training Accuracy", fontsize=13, fontweight="bold", pad=8)

    plt.suptitle(
        "Explainable AI Ensemble Transfer Learning for Early PCOS Prediction\n"
        "Benchmark: Test Accuracy: 96.63% | Precision: 98.92% | Recall: 96.82% | Specificity: 95.93% | ROC-AUC: 0.9958",
        fontsize=14,
        fontweight="bold",
        y=0.98
    )

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    print("\nDisplaying dashboard window on screen. Close the window when done.")
    plt.show()

if __name__ == "__main__":
    display_dashboard()
