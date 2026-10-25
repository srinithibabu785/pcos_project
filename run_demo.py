"""
PCOS AI Decision-Support Demo CLI for Visual Studio Code.
Directly loads serialized models without retraining and provides interactive or CLI prediction.
"""

import os
import sys
import json
from pathlib import Path

# Ensure project root is in sys.path
BASE_DIR = Path(__file__).resolve().parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from src.model_loader import get_model_manager
from src.report import format_clinical_decision_report

def run_prediction_demo(image_path: str):
    """Executes single-image decision-support prediction using pre-loaded models."""
    if not os.path.exists(image_path):
        print(f"Error: Specified image file does not exist: {image_path}")
        return

    print("\nLoading saved models from models/ directory (NO retraining)...")
    manager = get_model_manager()
    result = manager.predict_image(image_path)

    print("\n" + "-" * 35)
    print("PCOS AI PREDICTION")
    print("-" * 35)
    print("Image:")
    print(f"  {result['filename']}")
    print()
    print("EfficientNetB0 Score:")
    print(f"  {result['efficientnet_score']*100:5.2f}%")
    print()
    print("DenseNet121 Score:")
    print(f"  {result['densenet_score']*100:5.2f}%")
    print()
    print("Ensemble Score:")
    print(f"  {result['ensemble_score']*100:5.2f}%")
    print()
    print("Predicted Class:")
    print(f"  {result['predicted_class']}")
    print()
    print("Decision Threshold:")
    print(f"  {result['decision_threshold']:5.2f}")
    print()
    print("Model Status:")
    print(f"  {result['model_status']}")
    print("-" * 35)
    print()
    print("Clinical Presentation Category:")
    print(f"  Tier {result['risk_tier']} - {result['risk_category']} ({result['risk_color']}, Band: {result['risk_range'].replace('–', '-')})")
    print()
    print("IMPORTANT DISCLAIMER:")
    print("  Output represents AI-predicted PCOS likelihood/model score for academic")
    print("  decision support. It is NOT a confirmed PCOS diagnosis or guaranteed probability.")
    print("-" * 35 + "\n")

    # Ask if user wants formatted clinical report
    return result

def main():
    print("=" * 65)
    print("  PCOS AI: EXPLAINABLE ENSEMBLE TRANSFER LEARNING (VS CODE DEMO)")
    print("=" * 65)

    if len(sys.argv) > 1:
        img_path = sys.argv[1]
        run_prediction_demo(img_path)
    else:
        # Default test images for interactive demo selection
        test_dir = BASE_DIR / "data_processed" / "clean_images" / "test"
        if not test_dir.exists():
            test_dir = BASE_DIR / "data_processed" / "images" / "test"

        healthy_samples = list((test_dir / "healthy").glob("*.jpg"))[:3]
        pcos_samples = list((test_dir / "infected").glob("*.jpg"))[:3]
        all_samples = healthy_samples + pcos_samples

        if not all_samples:
            print("No test images found automatically. Please provide an image path:")
            print("Usage: python run_demo.py <path_to_image.jpg>")
            return

        print("\nSelect an ultrasound image to analyze:")
        for idx, sample in enumerate(all_samples, 1):
            lbl = "Healthy" if "healthy" in str(sample) else "PCOS"
            print(f"  [{idx}] {sample.name} (True Label: {lbl})")
        print("  [0] Enter custom image path")

        choice = input("\nEnter choice [1-6, or 0]: ").strip()
        if choice == "0":
            custom_path = input("Enter path to image: ").strip().strip('"')
            run_prediction_demo(custom_path)
        elif choice.isdigit() and 1 <= int(choice) <= len(all_samples):
            chosen_img = str(all_samples[int(choice) - 1])
            res = run_prediction_demo(chosen_img)
            
            # Offer clinical decision-support report view
            rep_choice = input("Display formatted clinical decision report? (y/n): ").strip().lower()
            if rep_choice == 'y':
                print(format_clinical_decision_report(res))
        else:
            print("Invalid selection. Exiting.")

if __name__ == "__main__":
    main()
