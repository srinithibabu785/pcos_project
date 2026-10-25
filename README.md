# Explainable AI-Based Ensemble Transfer Learning for Early PCOS Prediction

**Project Title:**  
*An Explainable AI-Based Ensemble Transfer Learning Approach for Early PCOS Prediction and Clinical Decision Support*

---

## 1. Project Objective & Clinical Decision Support Context
Polycystic Ovary Syndrome (PCOS) is a common endocrine-metabolic disorder affecting women of reproductive age. Early identification is crucial to prevent long-term metabolic, cardiovascular, and reproductive complications. 

This project establishes a **rigorous, reproducible, and explainable Deep Learning training pipeline** for ovarian ultrasound images. Using a dual-backbone transfer learning approach with **EfficientNetB0** and **DenseNet121**, soft-voting ensemble aggregation, validation-only decision threshold optimization, and **Grad-CAM explainability**, the system outputs an **AI-predicted PCOS likelihood/risk score** to assist clinicians.

> **Disclaimer**: This system is designed as an AI-assisted clinical decision-support research prototype and does not substitute for definitive diagnostic confirmation by a certified radiologist/gynecologist.

---

## 2. Dataset Architecture & Location
The ultrasound dataset is sourced from:
```
C:\Users\srini\Documents\pcos_project\dataset\datasett.zip
```

Archive internal structure:
```
PCOS/
├── infected/       # Class 1: Polycystic ovarian appearance (PCOS-positive proxy)
└── noninfected/    # Class 0: Normal ovarian appearance (PCOS-negative proxy)
```

---

## 3. Dataset Audit & Crucial Quality Findings
Before model training, a complete programmatic in-memory audit was conducted across all **11,784 files** in the archive:

* **100% JPEG, 100% 3-channel RGB**, 0 corrupted files, 0 zero-byte files.
* **Massive Duplication (66.09% redundant copies)**:
  * Only **3,996 unique SHA-256 binary hashes** exist among the 11,784 files.
  * Files named `Image_XXX.jpg`, `Image_SETA_XXX.jpg`, and `Image_SETB_XXX.jpg` are exact byte-for-byte triplicates.
  * 100 rotated images (`Image_Rotate_`), 200 doubly-rotated images (`Image_Rotate_Rotate_Right_`), and 685 resized images (`Image_resize_`) are synthetic transformations of base images.
* **Cross-Class Ground-Truth Conflicts**: 18 perceptual groups share identical visual scans labeled as "infected" in one folder and "noninfected" in the other (mean pixel diff $< 1.0$ / 255).
* **Label Verification**: No documentation or clinical codebook was provided inside `datasett.zip`. As mandated by scientific integrity:
  > *"Label semantics could not be independently verified from the supplied dataset."*
  Labels are treated as research proxy annotations.

Full audit reports:
* Text summary: `reports/dataset_audit_report.txt`
* Per-file records: `reports/dataset_audit_report.csv`

---

## 4. Strict Anti-Leakage Protocol & Data Splitting
A naive random shuffle split on the raw archive would place identical or rotated/resized twins of the same scan in both train and test splits, causing **severe data leakage and a fraudulent $\sim 99\%$ test accuracy**.

### Resolution Protocol:
1. **Deduplication**: Redundant byte duplicates were eliminated.
2. **Conflict Pruning**: Contradictory cross-class conflict pairs were quarantined.
3. **Family Grouping**: Rotated, resized, and base versions of each physical scan were clustered into unique `family_id` groups.
4. **Group-Aware Stratified Splitting**:
   * **70% Training** (`data/train.csv`): 2,753 images (2,200 PCOS, 553 Normal)
   * **15% Validation** (`data/validation.csv`): 613 images (495 PCOS, 118 Normal)
   * **15% Test (UNTOUCHED)** (`data/test.csv`): 594 images (471 PCOS, 123 Normal)
5. **Leakage Verification**:
   * Hash overlap across splits: **0 images**
   * Family overlap across splits: **0 families**

---

## 5. Model Architecture & Two-Stage Training

### Model 1: EfficientNetB0_PCOS
* **Input**: $224 \times 224 \times 3$, scaled via Keras EfficientNet preprocessing.
* **Backbone**: EfficientNetB0 pretrained on ImageNet.
* **Head**: GlobalAveragePooling2D $\rightarrow$ BatchNorm $\rightarrow$ Dropout(0.3) $\rightarrow$ Dense(64, ReLU) $\rightarrow$ Dropout(0.15) $\rightarrow$ Dense(1, Sigmoid).
* **Two-Stage Training**:
  * *Stage 1*: Frozen backbone, train classification head with Adam ($\text{lr} = 10^{-3}$).
  * *Stage 2*: Unfreeze top 25 layers (`block7a`, `top_conv`), fine-tune with Adam ($\text{lr} = 5 \times 10^{-5}$).

### Model 2: DenseNet121_PCOS
* **Input**: $224 \times 224 \times 3$, scaled via Keras DenseNet preprocessing (zero-centered $[-1, 1]$).
* **Backbone**: DenseNet121 pretrained on ImageNet.
* **Head**: GlobalAveragePooling2D $\rightarrow$ BatchNorm $\rightarrow$ Dropout(0.3) $\rightarrow$ Dense(64, ReLU) $\rightarrow$ Dropout(0.15) $\rightarrow$ Dense(1, Sigmoid).
* **Two-Stage Training**:
  * *Stage 1*: Frozen backbone, train classification head with Adam ($\text{lr} = 10^{-3}$).
  * *Stage 2*: Unfreeze top 30 layers (`conv5_block...`), fine-tune with Adam ($\text{lr} = 5 \times 10^{-5}$).

### Training Safeguards:
* Dynamic class weighting ($w_0 \approx 2.49, w_1 \approx 0.63$) to mitigate $\sim 4:1$ class imbalance without duplicate oversampling.
* Conservative medical augmentations (subtle rotation, translation, zoom, brightness, contrast) strictly on the training set. Validation and test sets are **never augmented**.
* `EarlyStopping`, `ReduceLROnPlateau`, and `ModelCheckpoint` saving best validation models.

---

## 6. Validation Threshold Tuning & Soft-Voting Ensemble
* **Threshold Sweep**: Decision thresholds from $0.20$ to $0.80$ were evaluated **strictly on the Validation Set** to maximize Balanced Accuracy and harmonize Sensitivity and Specificity. The test set was never accessed during threshold tuning.
* **Soft-Voting Ensemble**:
  $$P_{\text{ensemble}} = 0.5 \cdot P_{\text{eff}} + 0.5 \cdot P_{\text{dense}}$$
* **Untouched Test Benchmark**: The frozen ensemble is evaluated **once** on the untouched test set at the validation-selected threshold.
* **Calibration**: Brier score and reliability diagrams verify probability score reliability.

---

## 7. Explainable AI with Grad-CAM
Gradient-weighted Class Activation Mapping (Grad-CAM) generates visual explanations of network decisions:
* **EfficientNetB0**: Computes gradients from layer `top_conv`.
* **DenseNet121**: Computes gradients from layer `conv5_block16_2_conv`.
* **Output**: Original image, activation heatmap, blended overlay, and 3-panel clinical decision support composites under `gradcam/`.

---

## 8. Step-by-Step Guide: Running in Visual Studio Code

Open the project folder in Visual Studio Code:
```powershell
code C:\Users\srini\Documents\pcos_project
```
Open the integrated PowerShell or Command Prompt terminal (`Ctrl + ` `).

### Step 1: Run Dataset Audit
```powershell
python src/dataset_audit.py
```
*Generates: `reports/dataset_audit_report.txt`, `reports/dataset_audit_report.csv`*

### Step 2: Clean Dataset & Generate Anti-Leakage Splits
```powershell
python src/data_split.py
```
*Generates: `data/train.csv`, `data/validation.csv`, `data/test.csv`*

### Step 3: Train EfficientNetB0
```powershell
python src/train_efficientnet.py --batch_size 32 --epochs_stage1 4 --epochs_stage2 4
```
*Saves: `models/EfficientNetB0_best.keras`, `results/efficientnet_accuracy.png`, `results/efficientnet_loss.png`*

### Step 4: Train DenseNet121
```powershell
python src/train_densenet.py --batch_size 32 --epochs_stage1 4 --epochs_stage2 4
```
*Saves: `models/DenseNet121_best.keras`, `results/densenet_accuracy.png`, `results/densenet_loss.png`*

### Step 5: Optimize Decision Threshold on Validation Set
```powershell
python src/threshold_analysis.py
```
*Generates: `reports/validation_threshold_analysis.csv`, `reports/optimal_threshold.json`*

### Step 6: Evaluate Ensemble on Untouched Test Set
```powershell
python src/ensemble.py
```
*Generates: `reports/final_results.csv`, `results/ensemble_confusion_matrix.png`, `results/ensemble_roc_curve.png`, `results/ensemble_calibration_curve.png`*

### Step 7: Generate Grad-CAM Visualizations
```powershell
python src/gradcam.py
```
*Generates: Explanatory heatmaps and composites in `gradcam/efficientnet/` and `gradcam/densenet/`*

---

## 9. Project Directory Tree
```
pcos_project/
├── dataset/
│   └── datasett.zip                      # Original raw archive
├── data/
│   ├── clean_images/                     # Deduplicated image files
│   ├── train.csv                         # 70% Training set
│   ├── validation.csv                    # 15% Validation set
│   └── test.csv                          # 15% Untouched test set
├── models/
│   ├── EfficientNetB0_best.keras         # Checkpoint for EfficientNetB0
│   └── DenseNet121_best.keras            # Checkpoint for DenseNet121
├── reports/
│   ├── dataset_audit_report.txt          # Complete audit text report
│   ├── dataset_audit_report.csv          # Per-file image metadata
│   ├── validation_threshold_analysis.csv # Validation threshold sweep
│   ├── optimal_threshold.json            # Selected operating threshold
│   ├── experiment_results.csv            # Training log and hyperparameter records
│   └── final_results.csv                 # Final test benchmark comparison
├── results/
│   ├── efficientnet_accuracy.png         # Accuracy curves
│   ├── efficientnet_loss.png             # Loss curves
│   ├── densenet_accuracy.png             # Accuracy curves
│   ├── densenet_loss.png                 # Loss curves
│   ├── efficientnet_confusion_matrix.png # Confusion matrix
│   ├── densenet_confusion_matrix.png     # Confusion matrix
│   ├── ensemble_confusion_matrix.png     # Ensemble confusion matrix
│   ├── ensemble_roc_curve.png            # Comparative ROC curves
│   └── ensemble_calibration_curve.png    # Reliability diagram
├── gradcam/
│   ├── efficientnet/                     # Heatmaps, overlays, 3-panel composites
│   └── densenet/                         # Heatmaps, overlays, 3-panel composites
├── src/
│   ├── dataset_audit.py                  # Standalone audit CLI
│   ├── data_split.py                     # Deduplication & group splitting
│   ├── preprocessing.py                  # tf.data pipelines & scaling
│   ├── train_efficientnet.py             # 2-stage EfficientNetB0 training
│   ├── train_densenet.py                 # 2-stage DenseNet121 training
│   ├── threshold_analysis.py             # Validation threshold optimizer
│   ├── ensemble.py                       # Soft-voting & test evaluation
│   ├── evaluate_models.py                # Standalone model evaluation
│   └── gradcam.py                        # Grad-CAM XAI generator
├── requirements.txt                      # Python dependencies
└── README.md                             # Documentation and user guide
```
