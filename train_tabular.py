"""
Tabular Clinical Model Training Module for PCOS Prediction.
Trains, validates, and evaluates baseline tabular models (Random Forest, XGBoost, GBDT, Logistic Regression).
Saves the best model, fitted preprocessor pipeline, and evaluation metrics.
"""

import sys
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

import json
import numpy as np
import pandas as pd
import joblib

from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
)

from src.preprocessing import identify_tabular_feature_types, build_tabular_preprocessor

sys.stdout.reconfigure(encoding='utf-8')

BASE_DIR = Path(r"C:\Users\srini\Documents\pcos_project")
PROCESSED_DIR = BASE_DIR / "data_processed"
MODELS_DIR = BASE_DIR / "models"
RESULTS_DIR = BASE_DIR / "results"

def train_and_evaluate_tabular():
    print("\n==========================================")
    print("TRAINING TABULAR CLINICAL MODELS")
    print("==========================================")
    
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    
    # 1. Load pre-split datasets
    train_df = pd.read_csv(PROCESSED_DIR / "tabular_train.csv")
    val_df = pd.read_csv(PROCESSED_DIR / "tabular_val.csv")
    test_df = pd.read_csv(PROCESSED_DIR / "tabular_test.csv")
    
    target_col = "PCOS (Y/N)"
    num_cols, cat_cols = identify_tabular_feature_types(train_df, target_col=target_col)
    
    print(f"Features: {len(num_cols)} numerical, {len(cat_cols)} categorical")
    
    # 2. Build and fit preprocessor on training data only (Zero Preprocessing Leakage)
    preprocessor = build_tabular_preprocessor(num_cols, cat_cols)
    
    X_train = preprocessor.fit_transform(train_df[num_cols + cat_cols])
    y_train = train_df[target_col].values
    
    X_val = preprocessor.transform(val_df[num_cols + cat_cols])
    y_val = val_df[target_col].values
    
    X_test = preprocessor.transform(test_df[num_cols + cat_cols])
    y_test = test_df[target_col].values
    
    # 3. Define candidate algorithms
    candidates = {
        "RandomForest": RandomForestClassifier(n_estimators=150, max_depth=10, random_state=42, class_weight="balanced"),
        "GradientBoosting": GradientBoostingClassifier(n_estimators=120, max_depth=4, learning_rate=0.08, random_state=42),
        "XGBoost": XGBClassifier(n_estimators=120, max_depth=4, learning_rate=0.08, eval_metric="logloss", random_state=42),
        "LogisticRegression": LogisticRegression(max_iter=1000, class_weight="balanced", random_state=42)
    }
    
    val_results = {}
    print("\n--- Validation Performance Comparison ---")
    for name, model in candidates.items():
        model.fit(X_train, y_train)
        val_probs = model.predict_proba(X_val)[:, 1]
        val_preds = (val_probs >= 0.5).astype(int)
        
        auc = roc_auc_score(y_val, val_probs)
        f1 = f1_score(y_val, val_preds)
        acc = accuracy_score(y_val, val_preds)
        rec = recall_score(y_val, val_preds)
        prec = precision_score(y_val, val_preds)
        
        val_results[name] = {
            "model": model,
            "accuracy": float(acc),
            "precision": float(prec),
            "recall": float(rec),
            "f1": float(f1),
            "auc": float(auc)
        }
        print(f"{name:18s} | Acc: {acc:.4f} | Prec: {prec:.4f} | Rec: {rec:.4f} | F1: {f1:.4f} | AUC: {auc:.4f}")
        
    # Select best model based on validation ROC-AUC
    best_name = max(val_results.keys(), key=lambda k: val_results[k]["auc"])
    best_model = val_results[best_name]["model"]
    print(f"\nBest Selected Tabular Model: {best_name} (Val AUC: {val_results[best_name]['auc']:.4f})")
    
    # 4. Final Evaluation on strictly unseen Test Set
    test_probs = best_model.predict_proba(X_test)[:, 1]
    test_preds = (test_probs >= 0.5).astype(int)
    
    test_acc = accuracy_score(y_test, test_preds)
    test_prec = precision_score(y_test, test_preds)
    test_rec = recall_score(y_test, test_preds)
    test_f1 = f1_score(y_test, test_preds)
    test_auc = roc_auc_score(y_test, test_probs)
    test_cm = confusion_matrix(y_test, test_preds).tolist()
    
    test_metrics = {
        "model_type": best_name,
        "test_accuracy": float(test_acc),
        "test_precision": float(test_prec),
        "test_recall": float(test_rec),
        "test_f1": float(test_f1),
        "test_auc": float(test_auc),
        "confusion_matrix": test_cm,
        "feature_count": len(num_cols) + len(cat_cols)
    }
    
    print("\n--- Final Test Set Evaluation (Unseen Test Set) ---")
    print(f"Accuracy:  {test_acc * 100:.2f}%")
    print(f"Precision: {test_prec * 100:.2f}%")
    print(f"Recall:    {test_rec * 100:.2f}%")
    print(f"F1-Score:  {test_f1 * 100:.2f}%")
    print(f"ROC-AUC:   {test_auc:.4f}")
    print(f"Confusion Matrix:\n  TN: {test_cm[0][0]}, FP: {test_cm[0][1]}\n  FN: {test_cm[1][0]}, TP: {test_cm[1][1]}")
    
    # 5. Save best model, preprocessor, and metrics
    joblib.dump(best_model, MODELS_DIR / "tabular_model.pkl")
    joblib.dump(preprocessor, MODELS_DIR / "tabular_preprocessor.pkl")
    
    with open(RESULTS_DIR / "tabular_metrics.json", "w", encoding="utf-8") as f:
        json.dump(test_metrics, f, indent=4)
        
    print(f"\nSaved models/tabular_model.pkl")
    print(f"Saved models/tabular_preprocessor.pkl")
    print(f"Saved results/tabular_metrics.json")
    
    return test_metrics

if __name__ == "__main__":
    train_and_evaluate_tabular()
