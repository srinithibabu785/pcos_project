"""
Data Cleaning & Anti-Leakage Splitting Pipeline for PCOS Ultrasound Dataset.
- Preserves raw dataset completely untouched.
- Creates:
    data_processed/
        clean_images/
            train/{healthy, pcos}/
            val/{healthy, pcos}/
            test/{healthy, pcos}/
        labels/
        manifests/
            train.csv
            val.csv
            test.csv
        duplicates/
        rejected/
- Purges redundant copies and conflicting-label images from training.
- Retains independent benchmark test set untouched (1,468 images).
- Performs stratified 80% train / 20% validation split on clean training set.
- Exports complete manifests with hashes and metadata.
"""

import os
import io
import shutil
import zipfile
import hashlib
from pathlib import Path
import numpy as np
import pandas as pd
from PIL import Image
from sklearn.model_selection import train_test_split

BASE_DIR = Path(r"C:\Users\srini\Documents\pcos_project")
DATASET_DIR = BASE_DIR / "dataset"
ZIP_PATH = DATASET_DIR / "dataset_5.zip"

PROCESSED_DIR = BASE_DIR / "data_processed"
CLEAN_IMAGES_DIR = PROCESSED_DIR / "clean_images"
LEGACY_IMAGES_DIR = PROCESSED_DIR / "images"
LABELS_DIR = PROCESSED_DIR / "labels"
MANIFESTS_DIR = PROCESSED_DIR / "manifests"
DUPLICATES_DIR = PROCESSED_DIR / "duplicates"
REJECTED_DIR = PROCESSED_DIR / "rejected"

RANDOM_SEED = 42

def hash_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()

def clean_and_split():
    print(f"Starting Data Cleaning Pipeline from {ZIP_PATH}...")
    
    # 1. Prepare target directories
    for d in [CLEAN_IMAGES_DIR, LABELS_DIR, MANIFESTS_DIR, DUPLICATES_DIR, REJECTED_DIR]:
        d.mkdir(parents=True, exist_ok=True)
        
    for split in ["train", "val", "test"]:
        for cls in ["healthy", "infected"]:
            (CLEAN_IMAGES_DIR / split / cls).mkdir(parents=True, exist_ok=True)
            (LEGACY_IMAGES_DIR / split / cls).mkdir(parents=True, exist_ok=True)

    with zipfile.ZipFile(ZIP_PATH, 'r') as z:
        namelist = z.namelist()
        
        # ---------------------------------------------------------
        # Step A: Identify All Hashes & Detect Conflicts
        # ---------------------------------------------------------
        hash_to_files = {}
        conflicting_hashes = set()
        
        for n in namelist:
            if n.lower().endswith(('.jpg', '.jpeg', '.png')):
                cls = "healthy" if "/healthy/" in n else ("infected" if "/infected/" in n else None)
                if cls:
                    data = z.read(n)
                    h = hash_bytes(data)
                    if h in hash_to_files:
                        prev_cls = hash_to_files[h]["class"]
                        if prev_cls != cls:
                            conflicting_hashes.add(h)
                    else:
                        hash_to_files[h] = {"class": cls, "name": n}
                        
        print(f"Conflicting hash groups detected: {len(conflicting_hashes)}")

        # ---------------------------------------------------------
        # Step B: Process Independent Test Set (Untouched 1,468 images)
        # ---------------------------------------------------------
        test_entries = [n for n in namelist if "PCOSGen-train/test/" in n and n.lower().endswith(('.jpg', '.jpeg', '.png'))]
        print(f"Preserving independent test set: {len(test_entries)} images...")
        
        test_records = []
        for n in test_entries:
            data = z.read(n)
            fname = Path(n).name
            raw_cls = "healthy" if "/healthy/" in n else "infected"
            cls_name = "Healthy" if raw_cls == "healthy" else "PCOS"
            label = 0 if raw_cls == "healthy" else 1
            h = hash_bytes(data)
            
            with Image.open(io.BytesIO(data)) as img:
                w, h_dim = img.size
                
            dest_clean = CLEAN_IMAGES_DIR / "test" / raw_cls / fname
            dest_legacy = LEGACY_IMAGES_DIR / "test" / raw_cls / fname
            with open(dest_clean, "wb") as f:
                f.write(data)
            with open(dest_legacy, "wb") as f:
                f.write(data)
                
            test_records.append({
                "image_path": str(dest_clean.relative_to(BASE_DIR)).replace("\\", "/"),
                "label": label,
                "class_name": cls_name,
                "source": "PCOSGen-train/test",
                "image_width": w,
                "image_height": h_dim,
                "hash": h
            })

        # ---------------------------------------------------------
        # Step C: Clean Train Set (Remove Duplicates and Conflicting)
        # ---------------------------------------------------------
        train_entries = [n for n in namelist if "PCOSGen-train/train/" in n and n.lower().endswith(('.jpg', '.jpeg', '.png'))]
        print(f"Raw train images to clean: {len(train_entries)}...")
        
        seen_train_hashes = set()
        clean_train_entries = []
        removed_duplicates = 0
        removed_conflicts = 0
        
        for n in train_entries:
            data = z.read(n)
            fname = Path(n).name
            raw_cls = "healthy" if "/healthy/" in n else "infected"
            h = hash_bytes(data)
            
            if h in conflicting_hashes:
                # Save to rejected/ folder so nothing is deleted
                rej_path = REJECTED_DIR / f"{raw_cls}_{fname}"
                with open(rej_path, "wb") as f:
                    f.write(data)
                removed_conflicts += 1
                continue
                
            if h in seen_train_hashes:
                # Duplicate copy, archive to duplicates/ folder
                dup_path = DUPLICATES_DIR / f"{raw_cls}_{fname}"
                with open(dup_path, "wb") as f:
                    f.write(data)
                removed_duplicates += 1
                continue
                
            seen_train_hashes.add(h)
            with Image.open(io.BytesIO(data)) as img:
                w, h_dim = img.size
                
            clean_train_entries.append({
                "zip_path": n,
                "filename": fname,
                "raw_cls": raw_cls,
                "class_name": "Healthy" if raw_cls == "healthy" else "PCOS",
                "label": 0 if raw_cls == "healthy" else 1,
                "width": w,
                "height": h_dim,
                "hash": h,
                "data": data
            })
            
        print(f"Clean Unique Train Images: {len(clean_train_entries)}")
        print(f"  - Removed intra-train duplicate copies: {removed_duplicates}")
        print(f"  - Removed conflicting-label copies: {removed_conflicts}")

        # ---------------------------------------------------------
        # Step D: Stratified 80% Train / 20% Validation Split
        # ---------------------------------------------------------
        labels = [e["label"] for e in clean_train_entries]
        train_sub, val_sub = train_test_split(
            clean_train_entries,
            test_size=0.20,
            stratify=labels,
            random_state=RANDOM_SEED
        )
        
        print(f"Stratified Split:")
        print(f"  - Training Set (80%): {len(train_sub)} images")
        print(f"  - Validation Set (20%): {len(val_sub)} images")

        train_records = []
        for e in train_sub:
            dest_clean = CLEAN_IMAGES_DIR / "train" / e["raw_cls"] / e["filename"]
            dest_legacy = LEGACY_IMAGES_DIR / "train" / e["raw_cls"] / e["filename"]
            with open(dest_clean, "wb") as f:
                f.write(e["data"])
            with open(dest_legacy, "wb") as f:
                f.write(e["data"])
            train_records.append({
                "image_path": str(dest_clean.relative_to(BASE_DIR)).replace("\\", "/"),
                "label": e["label"],
                "class_name": e["class_name"],
                "source": "PCOSGen-train/train",
                "image_width": e["width"],
                "image_height": e["height"],
                "hash": e["hash"]
            })

        val_records = []
        for e in val_sub:
            dest_clean = CLEAN_IMAGES_DIR / "val" / e["raw_cls"] / e["filename"]
            dest_legacy = LEGACY_IMAGES_DIR / "val" / e["raw_cls"] / e["filename"]
            with open(dest_clean, "wb") as f:
                f.write(e["data"])
            with open(dest_legacy, "wb") as f:
                f.write(e["data"])
            val_records.append({
                "image_path": str(dest_clean.relative_to(BASE_DIR)).replace("\\", "/"),
                "label": e["label"],
                "class_name": e["class_name"],
                "source": "PCOSGen-train/train",
                "image_width": e["width"],
                "image_height": e["height"],
                "hash": e["hash"]
            })

        # ---------------------------------------------------------
        # Step E: Save Manifests and Label Mappings
        # ---------------------------------------------------------
        df_train = pd.DataFrame(train_records)
        df_val = pd.DataFrame(val_records)
        df_test = pd.DataFrame(test_records)
        
        df_train.to_csv(MANIFESTS_DIR / "train.csv", index=False)
        df_val.to_csv(MANIFESTS_DIR / "val.csv", index=False)
        df_test.to_csv(MANIFESTS_DIR / "test.csv", index=False)
        
        print(f"Manifests successfully exported to data_processed/manifests/")
        print(f"  - train.csv: {len(df_train)} rows (Healthy: {sum(df_train['label'] == 0)}, PCOS: {sum(df_train['label'] == 1)})")
        print(f"  - val.csv:   {len(df_val)} rows (Healthy: {sum(df_val['label'] == 0)}, PCOS: {sum(df_val['label'] == 1)})")
        print(f"  - test.csv:  {len(df_test)} rows (Healthy: {sum(df_test['label'] == 0)}, PCOS: {sum(df_test['label'] == 1)})")

        # Save class mapping reference in labels/
        class_mapping = {
            0: "Healthy",
            1: "PCOS"
        }
        with open(LABELS_DIR / "class_mapping.json", "w", encoding="utf-8") as f:
            import json
            json.dump(class_mapping, f, indent=4)
        print(f"Saved data_processed/labels/class_mapping.json")

    # Anti-leakage assertion
    train_val_hashes = set(df_train["hash"]).union(set(df_val["hash"]))
    test_hashes = set(df_test["hash"])
    leakage = train_val_hashes.intersection(test_hashes)
    assert len(leakage) == 0, f"Critical Data Leakage: {len(leakage)} hashes overlap between train/val and test!"
    print(f"Strict Zero-Leakage Assertion: PASSED (0 overlapping hashes between train/val and test sets)")

if __name__ == "__main__":
    clean_and_split()
