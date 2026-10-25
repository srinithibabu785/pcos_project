# Dataset Inspection and Preparation Report

## 1. Executive Summary
This document summarizes the comprehensive inspection, cleaning, and preparation of the two separate datasets used in the project:
**"An Explainable AI-Based Ensemble Transfer Learning Approach for Early PCOS Prediction and Clinical Decision Support"**

- **Tabular Modality:** Clinical, metabolic, and demographic parameters from `dataset_1.zip`.
- **Imaging Modality:** Ultrasound transvaginal and pelvic images from `dataset_5.zip` (`PCOSGen-train`).

Both datasets are kept strictly separated to avoid invalid multimodal conflation, and rigorous anti-leakage splitting strategies have been enforced.

---

## 2. Tabular Dataset Inspection & Preparation

- **Source File:** `PCOS_extended_dataset.csv` (inside `dataset_1.zip`)
- **Total Records:** 2000 rows × 44 columns
- **Target Variable:** `PCOS (Y/N)` (Binary: `0` = Negative / Healthy, `1` = Positive / PCOS)
- **Target Distribution:**
  - Class `0` (Negative): 1392 (69.6%)
  - Class `1` (Positive): 608 (30.4%)
- **Patient Identifier & Leakage Prevention:**
  - `Patient File No.` contains 530 unique patient identifiers across 2,000 rows.
  - To prevent patient-level leakage across splits, a **Stratified Group Split** was implemented on `Patient File No.`.
  - **Zero Patient Overlap Confirmed:**
    - Train patients & Val patients: 0
    - Train patients & Test patients: 0
    - Val patients & Test patients: 0
- **Split Distribution:**
  - **Train:** 1428 rows (Target: {0: 994, 1: 434})
  - **Validation:** 286 rows (Target: {0: 199, 1: 87})
  - **Test:** 286 rows (Target: {0: 199, 1: 87})
- **Data Sanitization Performed:**
  - Trimmed trailing/leading whitespaces from column headers.
  - Fixed typo `'1.99.'` in `II    beta-HCG(mIU/mL)` to valid float `1.99`.
  - Fixed non-numeric string `'a'` in `AMH(ng/mL)` to `NaN`.
  - Handled 3 missing values in `Marraige Status (Yrs)` via median imputation in pipeline.

---

## 3. Ultrasound Image Dataset Inspection & Preparation

- **Source:** `dataset_5.zip` (`PCOSGen-train`)
- **Image Format:** All images are 3-channel RGB in `.jpg` format.
- **Image Resolutions:** Predominantly 300×300 pixels (standardized to 224×224 for transfer learning backbones).
- **Corrupted Images:** 0 corrupted or unreadable images.
- **Class Labels:**
  - `healthy`: Class 0 (Normal ovarian anatomy)
  - `infected`: Class 1 (PCOS / Polycystic appearance)
- **Deduplication & Anti-Leakage Protocol:**
  - Original Train images: 3,200 (625 healthy, 2,575 infected)
  - Intra-train duplicates detected & removed: 80 duplicate files (28 healthy, 52 infected)
  - Deduplicated unique train images: 3,120
  - Original Test images preserved as unseen benchmark: 1468 (416 healthy, 1052 infected)
  - Overlap between train and test: **0 images** (Strictly zero train/test leakage)
- **Final Split Summary:**
  - **Train Set:** 2635 images (511 healthy, 2124 infected)
  - **Validation Set:** 464 images (90 healthy, 374 infected)
  - **Test Set (Unseen):** 1468 images (416 healthy, 1052 infected)
  - **Total Images:** 4567 images

---

## 4. Class Imbalance Mitigation Strategy
- Imaging class ratio: ~19% Healthy vs ~81% PCOS in train.
- Rather than naive synthetic oversampling (which bloats training time and risks memorization), class weights will be dynamically computed during loss calculation:
  - $w_0 = \frac{N}{2 \cdot N_0}$
  - $w_1 = \frac{N}{2 \cdot N_1}$
  This penalizes errors on the minority class proportionally without modifying the underlying distribution or inflating test/val sets.
