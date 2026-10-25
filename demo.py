"""
Interactive VS Code Demo for PCOS Ultrasound AI Classification.
Prompts for an ultrasound image path, runs inference through EfficientNetB0, DenseNet121,
and Ensemble, and prints real-time predictions alongside the loaded test set performance benchmarks.
"""

import os
import sys
from pathlib import Path

# Ensure project root is in sys.path
BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

import json
from src.predict import predict_pcos

sys.stdout.reconfigure(encoding='utf-8')

RESULTS_DIR = BASE_DIR / "results"

def load_test_metrics():
    """
    Loads dynamically saved test metrics from results directory.
    """
    eff_path = RESULTS_DIR / "efficientnet_metrics.json"
    dense_path = RESULTS_DIR / "densenet_metrics.json"
    ens_path = RESULTS_DIR / "ensemble_metrics.json"
    
    eff_metrics = json.load(open(eff_path, "r", encoding="utf-8")) if eff_path.exists() else {}
    dense_metrics = json.load(open(dense_path, "r", encoding="utf-8")) if dense_path.exists() else {}
    ens_metrics = json.load(open(ens_path, "r", encoding="utf-8")) if ens_path.exists() else {}
    
    return eff_metrics, dense_metrics, ens_metrics

def main():
    print("=" * 55)
    print("        PCOS AI — CLINICAL DECISION SUPPORT DEMO     ")
    print("=" * 55)
    
    # Prompt for input image path or allow pressing enter for default test image
    default_test_dir = BASE_DIR / "data_processed" / "images" / "test"
    sample_images = list(default_test_dir.glob("*/*.jpg")) if default_test_dir.exists() else []
    default_path = str(sample_images[0]) if sample_images else ""
    
    prompt_text = f"Enter ultrasound image path (Press Enter for sample: {Path(default_path).name if default_path else 'none'}): "
    try:
        user_input = input(prompt_text).strip()
    except EOFError:
        user_input = ""
        
    img_path = user_input if user_input else default_path
    
    if not img_path or not os.path.exists(img_path):
        print(f"\n[ERROR] File not found: '{img_path}'")
        sys.exit(1)
        
    print(f"\nLoading trained neural networks and analyzing image...")
    print(f"Target: {img_path}\n")
    
    pred_res = predict_pcos(img_path)
    eff_metrics, dense_metrics, ens_metrics = load_test_metrics()
    
    # Format probabilities as percentages
    eff_pct = pred_res["efficientnet_score"] * 100.0
    dense_pct = pred_res["densenet_score"] * 100.0
    ens_pct = pred_res["ensemble_score"] * 100.0
    risk_cat = pred_res["risk_category"].upper()
    risk_color = pred_res["risk_color"]
    pred_class = pred_res["predicted_class"]
    
    print("=" * 55)
    print("                 PREDICTION RESULTS                 ")
    print("=" * 55)
    print(f"Input image:                 {Path(img_path).name}")
    print(f"EfficientNetB0 Probability:  {eff_pct:6.2f}%")
    print(f"DenseNet121 Probability:     {dense_pct:6.2f}%")
    print(f"Ensemble Score:              {ens_pct:6.2f}%")
    print(f"Predicted Class:             {pred_class}")
    print(f"AI-Predicted Risk Category:  {risk_cat} ({risk_color})")
    
    print("\n" + "=" * 55)
    print("            MODEL TEST PERFORMANCE (UNSEEN DATA)     ")
    print("=" * 55)
    eff_acc = eff_metrics.get("accuracy", 0.0) * 100.0
    dense_acc = dense_metrics.get("accuracy", 0.0) * 100.0
    ens_acc = ens_metrics.get("accuracy", 0.0) * 100.0
    ens_f1 = ens_metrics.get("f1", 0.0) * 100.0
    ens_auc = ens_metrics.get("auc", 0.0)
    
    print(f"EfficientNetB0 Test Accuracy: {eff_acc:6.2f}%")
    print(f"DenseNet121 Test Accuracy:    {dense_acc:6.2f}%")
    print(f"Ensemble Test Accuracy:       {ens_acc:6.2f}%")
    print(f"Ensemble Test F1-Score:       {ens_f1:6.2f}%")
    print(f"Ensemble Test ROC-AUC:        {ens_auc:6.4f}")
    
    print("=" * 55)
    print("IMPORTANT:")
    print("The prediction is an AI-generated model estimate")
    print("and is not a medical diagnosis.")
    print("=======================================================")

if __name__ == "__main__":
    main()
