"""
================================================================================
Grad-CAM Explainable AI (XAI) Module for PCOS Ultrasound Classification
Project: An Explainable AI-Based Ensemble Transfer Learning Approach
         for Early PCOS Prediction and Clinical Decision Support
================================================================================
Implements Gradient-weighted Class Activation Mapping (Grad-CAM) to visualize
salient ovarian morphological structures (follicles, stroma) guiding model predictions.

Supports:
- EfficientNetB0: Target Conv layer 'top_conv'
- DenseNet121: Target Conv layer 'conv5_block16_2_conv'

Generates:
- Original Ultrasound Image
- Grad-CAM Salience Heatmap
- Colorized Overlay
- 3-Panel Clinical Decision Support Composite
================================================================================
"""

import os
import sys
import argparse
import numpy as np
import pandas as pd
import cv2
import matplotlib.pyplot as plt
from pathlib import Path
from PIL import Image

# Ensure project root is in sys.path
BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

import tensorflow as tf
from tensorflow.keras.models import Model

from src.preprocessing import IMG_SIZE, preprocess_image_path

MODELS_DIR = BASE_DIR / "models"
GRADCAM_DIR = BASE_DIR / "gradcam"
GRADCAM_DIR.mkdir(parents=True, exist_ok=True)

def find_target_conv_layer(model: Model, preferred_names: list) -> str:
    """Finds the deepest available 2D convolutional layer matching target names."""
    layer_names = [l.name for l in model.layers]
    for name in preferred_names:
        if name in layer_names:
            return name

    # Fallback to inspecting layers in reverse
    for l in reversed(model.layers):
        if isinstance(l, tf.keras.layers.Conv2D):
            return l.name
        # If backbone is nested
        if hasattr(l, "layers"):
            for sub_l in reversed(l.layers):
                if isinstance(sub_l, tf.keras.layers.Conv2D):
                    return f"{l.name}/{sub_l.name}"

    raise ValueError(f"Could not automatically locate a Conv2D layer in {model.name}")

def compute_gradcam_heatmap(
    model: Model,
    img_tensor: tf.Tensor,
    target_layer_name: str
) -> np.ndarray:
    """
    Computes Grad-CAM heatmap using tf.GradientTape.
    """
    target_layer = model.get_layer(target_layer_name)
    grad_model = Model(
        inputs=model.inputs,
        outputs=[target_layer.output, model.output]
    )

    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_tensor)
        # Class 1 (PCOS) likelihood score
        class_score = predictions[:, 0]

    grads = tape.gradient(class_score, conv_outputs)
    # Global average pooling over spatial gradient dimensions
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    conv_outputs = conv_outputs[0]
    heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)

    # ReLU to filter positive activations only
    heatmap = tf.maximum(heatmap, 0.0)
    max_val = tf.math.reduce_max(heatmap)
    if max_val > 0:
        heatmap = heatmap / max_val

    return heatmap.numpy()

def generate_and_save_gradcam(
    model_path: Path,
    image_path: Path,
    output_subdir: Path,
    model_type: str = "efficientnet",
    ground_truth_label: str = "PCOS"
):
    output_subdir.mkdir(parents=True, exist_ok=True)

    print(f"\nProcessing Grad-CAM for [{model_type.upper()}] on: {image_path.name}")
    model = tf.keras.models.load_model(str(model_path))

    # Identify deepest conv layer
    if model_type.lower() == "efficientnet":
        target_conv = find_target_conv_layer(model, ["top_conv", "block7a_project_conv"])
    elif model_type.lower() == "densenet":
        target_conv = find_target_conv_layer(model, ["conv5_block16_2_conv", "relu"])
    else:
        target_conv = find_target_conv_layer(model, ["conv_final"])

    print(f"Target Convolutional Layer selected: '{target_conv}'")

    # Load & preprocess input image
    orig_pil = Image.open(str(image_path)).convert("RGB")
    orig_np = np.array(orig_pil.resize(IMG_SIZE))

    img_tensor, _ = preprocess_image_path(str(image_path), 0, model_type=model_type)
    img_tensor = tf.expand_dims(img_tensor, axis=0)

    # Prediction
    pred_prob = float(model.predict(img_tensor, verbose=0)[0, 0])
    pred_label = "PCOS" if pred_prob >= 0.5 else "Normal"
    risk_pct = pred_prob * 100

    # Compute Heatmap
    heatmap = compute_gradcam_heatmap(model, img_tensor, target_conv)
    heatmap_resized = cv2.resize(heatmap, (IMG_SIZE[0], IMG_SIZE[1]))
    heatmap_uint8 = np.uint8(255 * heatmap_resized)

    # Apply Colormap
    jet = plt.colormaps["jet"]
    jet_colors = jet(np.arange(256))[:, :3]
    jet_heatmap = jet_colors[heatmap_uint8]
    jet_heatmap = np.uint8(jet_heatmap * 255)

    # Overlay heatmap onto original image
    overlay = cv2.addWeighted(orig_np, 0.6, jet_heatmap, 0.4, 0)

    # Save individual components
    orig_save = output_subdir / "sample_original.png"
    heat_save = output_subdir / "sample_heatmap.png"
    over_save = output_subdir / "sample_overlay.png"

    Image.fromarray(orig_np).save(orig_save)
    Image.fromarray(jet_heatmap).save(heat_save)
    Image.fromarray(overlay).save(over_save)

    print(f"Saved: {orig_save.name}, {heat_save.name}, {over_save.name}")

    # Generate 3-Panel Clinical Decision Support Composite
    composite_save = output_subdir / f"{model_type}_gradcam_composite.png"
    fig, axes = plt.subplots(1, 3, figsize=(15, 5), dpi=300)

    axes[0].imshow(orig_np)
    axes[0].set_title(f"Original Ultrasound\nGround Truth: {ground_truth_label}", fontsize=12, fontweight="bold")
    axes[0].axis("off")

    im1 = axes[1].imshow(heatmap_resized, cmap="jet")
    axes[1].set_title(f"Grad-CAM Salience Heatmap\nLayer: '{target_conv}'", fontsize=12, fontweight="bold")
    axes[1].axis("off")
    fig.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)

    axes[2].imshow(overlay)
    axes[2].set_title(f"Explainability Overlay\nAI Likelihood: {risk_pct:.1f}% ({pred_label})", fontsize=12, fontweight="bold")
    axes[2].axis("off")

    plt.suptitle(f"{model_type.upper()} - Explainable AI Clinical Decision Support Analysis", fontsize=14, fontweight="bold", y=0.98)
    plt.tight_layout()
    plt.savefig(composite_save, dpi=300)
    plt.close()
    print(f"Saved Clinical Composite to: {composite_save}")

def run_gradcam(test_csv: str = "data/test.csv"):
    test_csv_path = BASE_DIR / test_csv
    if not test_csv_path.exists():
        raise FileNotFoundError(f"Test CSV not found: {test_csv_path}")

    df_test = pd.read_csv(test_csv_path)
    
    # Pick a representative PCOS test sample
    pcos_samples = df_test[df_test["label"] == 1]
    if len(pcos_samples) == 0:
        sample_row = df_test.iloc[0]
    else:
        sample_row = pcos_samples.iloc[0]

    sample_img_path = Path(sample_row["image_path"])
    gt_label = sample_row["class_name"]
    print(f"Selected test sample for Grad-CAM: {sample_img_path.name} ({gt_label})")

    eff_model_path = MODELS_DIR / "EfficientNetB0_best.keras"
    dense_model_path = MODELS_DIR / "DenseNet121_best.keras"

    if eff_model_path.exists():
        generate_and_save_gradcam(
            eff_model_path,
            sample_img_path,
            GRADCAM_DIR / "efficientnet",
            model_type="efficientnet",
            ground_truth_label=gt_label
        )
    else:
        print(f"Notice: {eff_model_path} not found. Skipping EfficientNet Grad-CAM until trained.")

    if dense_model_path.exists():
        generate_and_save_gradcam(
            dense_model_path,
            sample_img_path,
            GRADCAM_DIR / "densenet",
            model_type="densenet",
            ground_truth_label=gt_label
        )
    else:
        print(f"Notice: {dense_model_path} not found. Skipping DenseNet Grad-CAM until trained.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate Grad-CAM Visualizations for PCOS Models")
    parser.add_argument("--test_csv", type=str, default="data/test.csv", help="Path to test CSV")
    args = parser.parse_args()
    run_gradcam(test_csv=args.test_csv)
