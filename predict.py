"""
Standalone Prediction CLI Module for PCOS AI Project.
Directly loads serialized models from disk and scores ultrasound images.
"""

import sys
import argparse
import json
from pathlib import Path

# Ensure project root is in sys.path
BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from src.model_loader import get_model_manager

def predict_pcos(image_path: str) -> dict:
    """Loads models and predicts PCOS likelihood for a given ultrasound image."""
    manager = get_model_manager()
    return manager.predict_image(image_path)

def main():
    parser = argparse.ArgumentParser(description="PCOS AI Decision-Support Inference CLI")
    parser.add_argument("--image", type=str, required=True, help="Path to ultrasound image file")
    parser.add_argument("--json", action="store_true", help="Output results as JSON")
    args = parser.parse_args()

    result = predict_pcos(args.image)

    if args.json:
        print(json.dumps(result, indent=4))
    else:
        print("\n-----------------------------------")
        print("PCOS AI PREDICTION (DECISION SUPPORT)")
        print("-----------------------------------")
        print(f"Image:              {result['filename']}")
        print(f"EfficientNetB0 Score: {result['efficientnet_score']*100:6.2f}%")
        print(f"DenseNet121 Score:    {result['densenet_score']*100:6.2f}%")
        print(f"Ensemble Score:       {result['ensemble_score']*100:6.2f}%")
        print(f"Predicted Class:      {result['predicted_class']}")
        print(f"Decision Threshold:   {result['decision_threshold']:6.2f}")
        print(f"Clinical Risk Tier:   {result['risk_category']} ({result['risk_color']}, {result['risk_range']})")
        print(f"Model Status:         {result['model_status']}")
        print("-----------------------------------")
        print("NOTE: AI-predicted PCOS likelihood/model score for academic research and")
        print("clinical decision support only. Not a medical diagnosis.\n")

if __name__ == "__main__":
    main()
