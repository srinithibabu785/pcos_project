"""
Model Loader and Inference Module for PCOS AI Project.
Directly loads serialized .keras models and json configurations without retraining.
"""

import json
from pathlib import Path
from typing import Dict, Any, Optional
import numpy as np
import tensorflow as tf

from src.preprocessing import load_and_preprocess_image, preprocess_image_for_model
from src.ensemble import PCOSEnsemble, get_risk_category

BASE_DIR = Path(__file__).resolve().parent.parent
MODELS_DIR = BASE_DIR / "models"

class PCOSModelManager:
    _instance = None

    def __init__(self, models_dir: Path = MODELS_DIR):
        self.models_dir = models_dir
        self.eff_model: Optional[tf.keras.Model] = None
        self.dense_model: Optional[tf.keras.Model] = None
        self.ensemble: Optional[PCOSEnsemble] = None
        self.threshold_config: Dict[str, Any] = {}
        self.class_names: Dict[int, str] = {0: "Healthy", 1: "PCOS"}
        self.loaded = False

    def load_models(self):
        """Loads serialized .keras models and configuration files."""
        eff_path = self.models_dir / "efficientnetb0_pcos.keras"
        dense_path = self.models_dir / "densenet121_pcos.keras"
        ensemble_config_path = self.models_dir / "ensemble_config.json"
        threshold_config_path = self.models_dir / "threshold_config.json"
        class_names_path = self.models_dir / "class_names.json"

        if not eff_path.exists():
            raise FileNotFoundError(f"Missing EfficientNetB0 model at {eff_path}")
        if not dense_path.exists():
            raise FileNotFoundError(f"Missing DenseNet121 model at {dense_path}")

        print(f"Loading EfficientNetB0 from {eff_path.name}...")
        self.eff_model = tf.keras.models.load_model(str(eff_path))
        
        print(f"Loading DenseNet121 from {dense_path.name}...")
        self.dense_model = tf.keras.models.load_model(str(dense_path))

        if ensemble_config_path.exists():
            self.ensemble = PCOSEnsemble.load_config(ensemble_config_path)
            print(f"Loaded ensemble config: w_eff={self.ensemble.w_eff:.2f}, w_dense={self.ensemble.w_dense:.2f}, threshold={self.ensemble.threshold:.2f}")
        else:
            self.ensemble = PCOSEnsemble(0.5, 0.5, 0.5)

        if threshold_config_path.exists():
            with open(threshold_config_path, "r", encoding="utf-8") as f:
                self.threshold_config = json.load(f)
                
        if class_names_path.exists():
            with open(class_names_path, "r", encoding="utf-8") as f:
                raw_classes = json.load(f)
                self.class_names = {int(k): v for k, v in raw_classes.items()}

        self.loaded = True
        return self

    def predict_image(self, image_path: str) -> Dict[str, Any]:
        """
        Executes end-to-end inference on a single ultrasound image.
        Returns model scores, ensemble fusion, and clinical risk tier.
        """
        if not self.loaded:
            self.load_models()

        raw_img = load_and_preprocess_image(str(image_path))
        
        # Prepare inputs for each model
        eff_input = np.expand_dims(preprocess_image_for_model(raw_img, "efficientnet"), axis=0)
        dense_input = np.expand_dims(preprocess_image_for_model(raw_img, "densenet"), axis=0)

        # Predict
        p_eff = float(self.eff_model.predict(eff_input, verbose=0)[0][0])
        p_dense = float(self.dense_model.predict(dense_input, verbose=0)[0][0])
        p_ens = float(self.ensemble.predict_proba(np.array([p_eff]), np.array([p_dense]))[0])

        threshold = self.ensemble.threshold
        predicted_class_id = 1 if p_ens >= threshold else 0
        predicted_class = self.class_names.get(predicted_class_id, "PCOS" if predicted_class_id == 1 else "Healthy")
        
        risk_info = get_risk_category(p_ens)

        return {
            "image_path": str(image_path),
            "filename": Path(image_path).name,
            "efficientnet_score": p_eff,
            "densenet_score": p_dense,
            "ensemble_score": p_ens,
            "decision_threshold": threshold,
            "predicted_class_id": predicted_class_id,
            "predicted_class": predicted_class,
            "risk_tier": risk_info["tier"],
            "risk_category": risk_info["category"],
            "risk_color": risk_info["color"],
            "risk_range": risk_info["range"],
            "clinical_note": risk_info["clinical_note"],
            "model_status": "Loaded successfully from disk"
        }

def get_model_manager() -> PCOSModelManager:
    """Singleton getter for the model manager."""
    if PCOSModelManager._instance is None:
        PCOSModelManager._instance = PCOSModelManager()
    return PCOSModelManager._instance
